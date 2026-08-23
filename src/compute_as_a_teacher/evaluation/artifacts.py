"""Deterministic, atomic JSON artifact helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from .errors import EvaluationError


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"Non-finite JSON number is forbidden: {value}")


def canonical_json_bytes(value: Any) -> bytes:
    """Encode canonical UTF-8 JSON with one terminal newline."""

    try:
        serialized = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise EvaluationError(f"Value is not canonical JSON: {exc}") from exc
    return (serialized + "\n").encode("utf-8")


def canonical_jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) for row in rows)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def file_digest(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def artifact_reference(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    sha256, byte_count = file_digest(path)
    reference: dict[str, Any] = {
        "path": str(path),
        "sha256": sha256,
        "bytes": byte_count,
    }
    if rows is not None:
        reference["rows"] = rows
    return reference


def publish_bytes(path: Path, payload: bytes, *, force: bool = False) -> bool:
    """Atomically publish bytes; return True when the file changed."""

    if path.exists():
        if not path.is_file():
            raise EvaluationError(f"Artifact destination is not a file: {path}")
        if path.read_bytes() == payload:
            return False
        if not force:
            raise EvaluationError(
                f"Refusing to overwrite mismatched artifact {path}; inspect it or use --force"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return True


def publish_json(path: Path, value: Any, *, force: bool = False) -> bool:
    return publish_bytes(path, canonical_json_bytes(value), force=force)


def publish_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    force: bool = False,
) -> bool:
    return publish_bytes(path, canonical_jsonl_bytes(rows), force=force)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_nonfinite_json,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise EvaluationError(f"Cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvaluationError(f"Expected a JSON object in {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise EvaluationError(f"Cannot open JSONL artifact {path}: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise EvaluationError(f"Blank JSONL row at {path}:{line_number}")
            try:
                row = json.loads(line, parse_constant=_reject_nonfinite_json)
            except (json.JSONDecodeError, ValueError) as exc:
                raise EvaluationError(
                    f"Invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise EvaluationError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows
