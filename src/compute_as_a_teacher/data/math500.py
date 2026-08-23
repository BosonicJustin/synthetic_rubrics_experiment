"""Pinned, deterministic MATH-500 acquisition and preparation.

Only :func:`load_locked_questions` is intended for future generation/training code.
The source and label readers live here so acquisition and verification can enforce
the firewall, but they must not be passed downstream to model-facing code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOCK_PATH = REPOSITORY_ROOT / "configs/datasets/math500.lock.json"
SOURCE_KEYS = frozenset(
    {"problem", "solution", "answer", "subject", "level", "unique_id"}
)
QUESTION_KEYS = frozenset({"id", "problem"})
LABEL_KEYS = frozenset({"id", "answer", "solution", "subject", "level"})
STRING_SOURCE_KEYS = ("problem", "solution", "answer", "subject", "unique_id")
CHUNK_SIZE = 1024 * 1024


class DatasetPreparationError(RuntimeError):
    """Raised when data violates the checked-in MATH-500 contract."""


@dataclass(frozen=True, slots=True)
class QuestionRecord:
    """The only record type that future model-facing code should accept."""

    id: str
    problem: str


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_digest(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise DatasetPreparationError(f"{name} must be a JSON object")
    return value


def load_dataset_lock(lock_path: Path = DEFAULT_LOCK_PATH) -> dict[str, Any]:
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetPreparationError(f"Cannot read dataset lock {lock_path}: {exc}") from exc

    _require_mapping(lock, "dataset lock")
    dataset = _require_mapping(lock.get("dataset"), "dataset")
    source = _require_mapping(lock.get("source"), "source")
    outputs = _require_mapping(lock.get("outputs"), "outputs")

    if lock.get("schema_version") != 1:
        raise DatasetPreparationError("Unsupported dataset-lock schema_version")

    requested = dataset.get("requested_revision")
    resolved = dataset.get("resolved_revision")
    if not isinstance(requested, str) or not re.fullmatch(r"[0-9a-f]{40}", requested):
        raise DatasetPreparationError("requested_revision must be a full 40-character commit SHA")
    if requested != resolved:
        raise DatasetPreparationError("requested_revision and resolved_revision must match")
    if f"/resolve/{requested}/" not in str(dataset.get("source_url", "")):
        raise DatasetPreparationError("source_url must contain the immutable revision")

    if source.get("rows") != 500:
        raise DatasetPreparationError("The MATH-500 lock must require exactly 500 rows")
    if set(_require_mapping(source.get("columns"), "source.columns")) != SOURCE_KEYS:
        raise DatasetPreparationError("The locked source column set is invalid")
    for artifact_name, artifact in (("source", source), *outputs.items()):
        artifact = _require_mapping(artifact, str(artifact_name))
        if not isinstance(artifact.get("path"), str):
            raise DatasetPreparationError(f"{artifact_name}.path must be a string")
        if not re.fullmatch(r"[0-9a-f]{64}", str(artifact.get("sha256", ""))):
            raise DatasetPreparationError(f"{artifact_name}.sha256 must be a SHA-256 hex digest")
        byte_count = artifact.get("bytes")
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count <= 0:
            raise DatasetPreparationError(f"{artifact_name}.bytes must be a positive integer")
    for output_name, expected_keys in (
        ("questions", QUESTION_KEYS),
        ("labels", LABEL_KEYS),
    ):
        output = _require_mapping(outputs.get(output_name), f"outputs.{output_name}")
        if set(output.get("keys", [])) != expected_keys:
            raise DatasetPreparationError(f"Locked keys for {output_name} are invalid")
        if output.get("rows") != source.get("rows"):
            raise DatasetPreparationError(f"Locked row count for {output_name} is invalid")

    return lock


def _locked_path(repository_root: Path, relative_path: Any) -> Path:
    if not isinstance(relative_path, str):
        raise DatasetPreparationError("Locked artifact path must be a string")
    root = repository_root.resolve()
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise DatasetPreparationError(
            f"Locked path escapes repository root: {relative_path}"
        ) from exc
    return path


def _check_file_contract(path: Path, spec: Mapping[str, Any], label: str) -> None:
    if not path.is_file():
        raise DatasetPreparationError(f"Missing {label}: {path}")
    actual_sha256, actual_bytes = file_digest(path)
    if actual_sha256 != spec.get("sha256") or actual_bytes != spec.get("bytes"):
        raise DatasetPreparationError(
            f"{label} checksum/size mismatch at {path}: "
            f"got sha256={actual_sha256}, bytes={actual_bytes}"
        )


def _preflight_destination(path: Path, payload: bytes, force: bool) -> None:
    if not path.exists():
        return
    if not path.is_file():
        raise DatasetPreparationError(f"Destination exists but is not a file: {path}")
    existing_sha, existing_size = file_digest(path)
    if existing_sha == _sha256_bytes(payload) and existing_size == len(payload):
        return
    if not force:
        raise DatasetPreparationError(
            f"Refusing to overwrite mismatched file {path}; inspect it or rerun with --force"
        )


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _install_source_bytes(
    payload: bytes,
    destination: Path,
    source_spec: Mapping[str, Any],
    force: bool,
) -> Path:
    actual_sha = _sha256_bytes(payload)
    if actual_sha != source_spec.get("sha256") or len(payload) != source_spec.get("bytes"):
        raise DatasetPreparationError(
            "Downloaded/supplied source does not match the pinned snapshot: "
            f"got sha256={actual_sha}, bytes={len(payload)}"
        )
    _preflight_destination(destination, payload, force)
    if not destination.exists() or destination.read_bytes() != payload:
        _atomic_write(destination, payload)
    return destination


def acquire_source(
    lock: Mapping[str, Any],
    repository_root: Path,
    *,
    source_file: Path | None = None,
    offline: bool = False,
    force: bool = False,
) -> Path:
    """Acquire the exact locked raw snapshot, or reuse a valid local copy."""

    dataset = _require_mapping(lock.get("dataset"), "dataset")
    source_spec = _require_mapping(lock.get("source"), "source")
    destination = _locked_path(repository_root, source_spec.get("path"))

    # An explicit mirror is an assertion by the caller, so validate it even when a
    # good cached destination already exists. Silently ignoring a bad mirror would
    # make the command appear to have tested something that it did not test.
    if source_file is not None:
        try:
            payload = source_file.read_bytes()
        except OSError as exc:
            raise DatasetPreparationError(
                f"Cannot read --source-file {source_file}: {exc}"
            ) from exc
        return _install_source_bytes(payload, destination, source_spec, force)

    if destination.is_file():
        existing_sha, existing_bytes = file_digest(destination)
        if (
            existing_sha == source_spec.get("sha256")
            and existing_bytes == source_spec.get("bytes")
        ):
            return destination
        if not force:
            raise DatasetPreparationError(
                f"Existing raw snapshot is corrupt or unexpected: {destination}; "
                "inspect it or rerun with --force"
            )
    elif destination.exists():
        raise DatasetPreparationError(f"Raw destination is not a file: {destination}")

    if offline:
        raise DatasetPreparationError(
            f"Offline mode requires a valid existing raw snapshot at {destination}"
        )

    request = urllib.request.Request(
        str(dataset.get("source_url")),
        headers={"User-Agent": "compute-as-a-teacher-dataset-preparer/0.1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            chunks: list[bytes] = []
            downloaded_bytes = 0
            expected_bytes = int(source_spec["bytes"])
            while chunk := response.read(
                min(CHUNK_SIZE, expected_bytes - downloaded_bytes + 1)
            ):
                downloaded_bytes += len(chunk)
                if downloaded_bytes > expected_bytes:
                    raise DatasetPreparationError(
                        f"Download exceeded locked size of {expected_bytes} bytes"
                    )
                chunks.append(chunk)
            payload = b"".join(chunks)
    except (OSError, urllib.error.URLError) as exc:
        raise DatasetPreparationError(f"MATH-500 download failed: {exc}") from exc
    return _install_source_bytes(payload, destination, source_spec, force)


def read_source_rows(
    path: Path,
    *,
    expected_rows: int,
    expected_subjects: Mapping[str, int] | None = None,
    expected_levels: Mapping[str, int] | None = None,
    expected_asymptote_count: int | None = None,
) -> list[dict[str, Any]]:
    """Parse JSONL while preserving source strings and enforcing the source schema."""

    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_problems: set[str] = set()
    subject_counts: Counter[str] = Counter()
    level_counts: Counter[int] = Counter()
    asymptote_count = 0

    try:
        handle = path.open("r", encoding="utf-8", newline="")
    except OSError as exc:
        raise DatasetPreparationError(f"Cannot open source JSONL {path}: {exc}") from exc

    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise DatasetPreparationError(f"Blank source row at line {line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetPreparationError(
                    f"Invalid JSON at {path}:{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise DatasetPreparationError(f"Source row {line_number} is not an object")
            if set(row) != SOURCE_KEYS:
                raise DatasetPreparationError(
                    f"Source row {line_number} has keys {sorted(row)}, "
                    f"expected {sorted(SOURCE_KEYS)}"
                )
            for key in STRING_SOURCE_KEYS:
                value = row[key]
                if not isinstance(value, str) or not value.strip():
                    raise DatasetPreparationError(
                        f"Source row {line_number} field {key!r} must be a nonempty string"
                    )
            level = row["level"]
            if isinstance(level, bool) or not isinstance(level, int) or level not in range(1, 6):
                raise DatasetPreparationError(
                    f"Source row {line_number} level must be an integer from 1 through 5"
                )
            record_id = row["unique_id"]
            if record_id in seen_ids:
                raise DatasetPreparationError(
                    f"Duplicate unique_id at line {line_number}: {record_id}"
                )
            if row["problem"] in seen_problems:
                raise DatasetPreparationError(f"Duplicate problem at line {line_number}")
            seen_ids.add(record_id)
            seen_problems.add(row["problem"])
            subject_counts[row["subject"]] += 1
            level_counts[level] += 1
            asymptote_count += "[asy]" in row["problem"]
            rows.append(row)

    if len(rows) != expected_rows:
        raise DatasetPreparationError(f"Expected {expected_rows} rows, found {len(rows)}")
    if expected_subjects is not None and dict(subject_counts) != dict(expected_subjects):
        raise DatasetPreparationError(
            f"Subject distribution mismatch: got {dict(sorted(subject_counts.items()))}"
        )
    if expected_levels is not None:
        normalized_levels = {str(key): value for key, value in level_counts.items()}
        if normalized_levels != dict(expected_levels):
            raise DatasetPreparationError(
                f"Level distribution mismatch: got {dict(sorted(normalized_levels.items()))}"
            )
    if expected_asymptote_count is not None and asymptote_count != expected_asymptote_count:
        raise DatasetPreparationError(
            f"Asymptote-problem count mismatch: got {asymptote_count}"
        )
    return rows


def build_views(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split verified rows into a strict question view and evaluation-only labels."""

    opaque_ids = [
        "math500-" + hashlib.sha256(row["unique_id"].encode("utf-8")).hexdigest()[:16]
        for row in rows
    ]
    if len(set(opaque_ids)) != len(opaque_ids):
        raise DatasetPreparationError("Opaque MATH-500 ID collision")
    questions = [
        {"id": record_id, "problem": row["problem"]}
        for record_id, row in zip(opaque_ids, rows, strict=True)
    ]
    labels = [
        {
            "id": record_id,
            "answer": row["answer"],
            "solution": row["solution"],
            "subject": row["subject"],
            "level": row["level"],
        }
        for record_id, row in zip(opaque_ids, rows, strict=True)
    ]
    return questions, labels


def canonical_jsonl(records: Iterable[Mapping[str, Any]]) -> bytes:
    """Serialize deterministic UTF-8 JSONL according to the checked-in lock."""

    text = "".join(
        json.dumps(
            dict(record),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for record in records
    )
    return text.encode("utf-8")


def _validate_output_payload(
    name: str,
    payload: bytes,
    record_count: int,
    spec: Mapping[str, Any],
) -> None:
    actual_sha = _sha256_bytes(payload)
    if (
        actual_sha != spec.get("sha256")
        or len(payload) != spec.get("bytes")
        or record_count != spec.get("rows")
    ):
        raise DatasetPreparationError(
            f"Derived {name} does not match the checked-in lock: "
            f"sha256={actual_sha}, bytes={len(payload)}, rows={record_count}"
        )


def _verified_rows(path: Path, lock: Mapping[str, Any]) -> list[dict[str, Any]]:
    source_spec = _require_mapping(lock.get("source"), "source")
    statistics = _require_mapping(lock.get("statistics"), "statistics")
    _check_file_contract(path, source_spec, "raw MATH-500 snapshot")
    return read_source_rows(
        path,
        expected_rows=int(source_spec["rows"]),
        expected_subjects=_require_mapping(statistics.get("subjects"), "statistics.subjects"),
        expected_levels=_require_mapping(statistics.get("levels"), "statistics.levels"),
        expected_asymptote_count=int(statistics["asymptote_problem_count"]),
    )


def prepare_dataset(
    repository_root: Path = REPOSITORY_ROOT,
    lock_path: Path = DEFAULT_LOCK_PATH,
    *,
    source_file: Path | None = None,
    offline: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Acquire, validate, split, and atomically publish the locked dataset."""

    lock = load_dataset_lock(lock_path)
    source_path = acquire_source(
        lock,
        repository_root,
        source_file=source_file,
        offline=offline,
        force=force,
    )
    rows = _verified_rows(source_path, lock)
    questions, labels = build_views(rows)
    payloads = {
        "questions": canonical_jsonl(questions),
        "labels": canonical_jsonl(labels),
    }
    outputs = _require_mapping(lock.get("outputs"), "outputs")

    destinations: dict[str, Path] = {}
    for name, payload in payloads.items():
        spec = _require_mapping(outputs.get(name), f"outputs.{name}")
        _validate_output_payload(name, payload, len(rows), spec)
        destination = _locked_path(repository_root, spec.get("path"))
        destinations[name] = destination
        _preflight_destination(destination, payload, force)

    for name, payload in payloads.items():
        destination = destinations[name]
        if not destination.exists() or destination.read_bytes() != payload:
            _atomic_write(destination, payload)

    return verify_dataset(repository_root, lock_path)


def verify_dataset(
    repository_root: Path = REPOSITORY_ROOT,
    lock_path: Path = DEFAULT_LOCK_PATH,
) -> dict[str, Any]:
    """Verify local raw and derived artifacts without using the network."""

    lock = load_dataset_lock(lock_path)
    source_spec = _require_mapping(lock.get("source"), "source")
    source_path = _locked_path(repository_root, source_spec.get("path"))
    rows = _verified_rows(source_path, lock)
    expected_questions, expected_labels = build_views(rows)
    expected_payloads = {
        "questions": canonical_jsonl(expected_questions),
        "labels": canonical_jsonl(expected_labels),
    }
    outputs = _require_mapping(lock.get("outputs"), "outputs")

    for name, expected_payload in expected_payloads.items():
        spec = _require_mapping(outputs.get(name), f"outputs.{name}")
        output_path = _locked_path(repository_root, spec.get("path"))
        _check_file_contract(output_path, spec, name)
        actual_payload = output_path.read_bytes()
        if actual_payload != expected_payload:
            raise DatasetPreparationError(
                f"{name} bytes do not match the deterministic source-derived view"
            )

    return {
        "revision": lock["dataset"]["resolved_revision"],
        "rows": len(rows),
        "source_sha256": source_spec["sha256"],
        "questions_sha256": outputs["questions"]["sha256"],
        "labels_sha256": outputs["labels"]["sha256"],
    }


def verify_locked_questions(
    repository_root: Path = REPOSITORY_ROOT,
    lock_path: Path = DEFAULT_LOCK_PATH,
) -> dict[str, Any]:
    """Verify only the locked model-facing questions artifact."""

    lock = load_dataset_lock(lock_path)
    outputs = _require_mapping(lock.get("outputs"), "outputs")
    question_spec = _require_mapping(outputs.get("questions"), "outputs.questions")
    questions_path = _locked_path(repository_root, question_spec.get("path"))
    _check_file_contract(
        questions_path,
        question_spec,
        "locked MATH-500 questions",
    )
    questions = _load_questions(
        questions_path,
        expected_rows=int(question_spec["rows"]),
    )
    return {
        "revision": lock["dataset"]["resolved_revision"],
        "rows": len(questions),
        "questions_sha256": question_spec["sha256"],
    }


def _load_questions(path: Path, *, expected_rows: int = 500) -> list[QuestionRecord]:
    """Parse exact `{id, problem}` records after an integrity check by the caller."""

    records: list[QuestionRecord] = []
    seen_ids: set[str] = set()
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise DatasetPreparationError(f"Cannot open question file {path}: {exc}") from exc

    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise DatasetPreparationError(f"Blank question row at line {line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetPreparationError(
                    f"Invalid question JSON at line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(row, dict) or set(row) != QUESTION_KEYS:
                keys = sorted(row) if isinstance(row, dict) else type(row).__name__
                raise DatasetPreparationError(
                    f"Question row {line_number} has unsafe schema {keys}; "
                    f"expected exactly {sorted(QUESTION_KEYS)}"
                )
            if not all(isinstance(row[key], str) and row[key].strip() for key in QUESTION_KEYS):
                raise DatasetPreparationError(
                    f"Question row {line_number} must contain nonempty string values"
                )
            if row["id"] in seen_ids:
                raise DatasetPreparationError(f"Duplicate question id at line {line_number}")
            seen_ids.add(row["id"])
            records.append(QuestionRecord(id=row["id"], problem=row["problem"]))
    if len(records) != expected_rows:
        raise DatasetPreparationError(
            f"Expected {expected_rows} question rows, found {len(records)}"
        )
    return records


def load_locked_questions(
    path: Path,
    *,
    lock_path: Path = DEFAULT_LOCK_PATH,
) -> list[QuestionRecord]:
    """Integrity-check and load the one locked model-facing MATH-500 view."""

    lock = load_dataset_lock(lock_path)
    outputs = _require_mapping(lock.get("outputs"), "outputs")
    question_spec = _require_mapping(outputs.get("questions"), "outputs.questions")
    _check_file_contract(path, question_spec, "locked MATH-500 questions")
    return _load_questions(path, expected_rows=int(question_spec["rows"]))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download, validate, and split the pinned MATH-500 snapshot."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPOSITORY_ROOT,
        help="Repository root used to resolve locked artifact paths.",
    )
    parser.add_argument(
        "--lock-file",
        type=Path,
        default=DEFAULT_LOCK_PATH,
        help="Checked-in dataset lock JSON.",
    )
    parser.add_argument(
        "--source-file",
        type=Path,
        help="Use a local mirror instead of downloading; checksum verification still applies.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Never use the network; require the locked raw snapshot locally.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify existing artifacts without downloading or writing.",
    )
    parser.add_argument(
        "--verify-questions-only",
        action="store_true",
        help="Verify only the locked model-facing questions; do not read raw data or labels.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Intentionally replace existing files that do not match the lock.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.verify_only and args.verify_questions_only:
        parser.error("Choose only one verification mode")
    if (args.verify_only or args.verify_questions_only) and (
        args.source_file is not None or args.force or args.offline
    ):
        parser.error(
            "Verification modes cannot be combined with --source-file, --force, or --offline"
        )

    try:
        if args.verify_questions_only:
            summary = verify_locked_questions(args.repo_root, args.lock_file)
            print(
                f"Verified model-facing MATH-500 questions: rows={summary['rows']} "
                f"revision={summary['revision']}"
            )
            print(f"questions sha256: {summary['questions_sha256']}")
            return 0
        if args.verify_only:
            summary = verify_dataset(args.repo_root, args.lock_file)
            action = "Verified"
        else:
            summary = prepare_dataset(
                args.repo_root,
                args.lock_file,
                source_file=args.source_file,
                offline=args.offline,
                force=args.force,
            )
            action = "Prepared and verified"
    except DatasetPreparationError as exc:
        parser.exit(1, f"error: {exc}\n")

    print(
        f"{action} MATH-500: rows={summary['rows']} "
        f"revision={summary['revision']}"
    )
    print(f"source sha256:    {summary['source_sha256']}")
    print(f"questions sha256: {summary['questions_sha256']}")
    print(f"labels sha256:    {summary['labels_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
