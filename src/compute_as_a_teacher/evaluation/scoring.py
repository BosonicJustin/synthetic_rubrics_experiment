"""Labels-only scoring boundary and MATH-500 metric summaries."""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from compute_as_a_teacher.data.math500 import DatasetPreparationError, load_dataset_lock

from .artifacts import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
    file_digest,
    publish_bytes,
    read_jsonl,
    sha256_bytes,
    sha256_text,
)
from .config import RAW_BASELINE_SELECTION_METHOD, ScoringConfig
from .errors import EvaluationError
from .execution import verify_complete_execution
from .grading import grade_response
from .planning import EXECUTION_NAME, GENERATIONS_NAME, load_plan
from .schemas import (
    GENERATION_ROW_KEYS,
    GenerationRequest,
)


SCORES_NAME = "scores.jsonl"
PAIRED_SCORES_NAME = "paired_scores.jsonl"
SUMMARY_NAME = "summary.json"
SCORING_MANIFEST_NAME = "scoring_manifest.json"
SCORING_MANIFEST_SCHEMA_VERSION = 3
DATASET_LINEAGE_SCHEMA_VERSION = 1
RAW_BASELINE_SELECTION_DOMAIN = (
    "compute_as_a_teacher/raw_baseline/sha256_uniform_per_question_v1"
)
SAFE_ARTIFACT_REFERENCE_KEYS = frozenset({"path", "sha256", "bytes"})
SAFE_ROW_ARTIFACT_REFERENCE_KEYS = frozenset(
    {"path", "sha256", "bytes", "rows"}
)
SCORE_ROW_KEYS = {
    "schema_version",
    "stage",
    "task_id",
    "generation_output_sha256",
    "question_id",
    "rollout_index",
    "source_task_ids",
    "extraction",
    "grades",
    "primary_grader",
    "primary_correct",
    "subject",
    "level",
}


@dataclass(frozen=True, slots=True)
class EvalLabel:
    id: str
    answer: str
    subject: str
    level: int


@dataclass(frozen=True, slots=True)
class LockedScoringData:
    """References retained after crossing the evaluation-label boundary."""

    labels: tuple[EvalLabel, ...]
    labels_reference: Mapping[str, Any]
    dataset_lineage: Mapping[str, Any]


def _resolve_locked_path(repository_root: Path, relative_path: str) -> Path:
    root = repository_root.resolve()
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise EvaluationError(f"Path escapes repository root: {relative_path}") from exc
    return path


def _artifact_reference(
    path: Path,
    logical_path: str,
    *,
    rows: int | None = None,
) -> dict[str, Any]:
    digest, size = file_digest(path)
    reference: dict[str, Any] = {
        "path": logical_path,
        "sha256": digest,
        "bytes": size,
    }
    if rows is not None:
        reference["rows"] = rows
    return reference


def _payload_reference(
    logical_path: str,
    payload: bytes,
    *,
    rows: int | None = None,
) -> dict[str, Any]:
    reference: dict[str, Any] = {
        "path": logical_path,
        "sha256": sha256_bytes(payload),
        "bytes": len(payload),
    }
    if rows is not None:
        reference["rows"] = rows
    return reference


def _reference_matches(
    reference: Mapping[str, Any],
    path: Path,
    *,
    expected_path: str,
    rows: int | None = None,
) -> bool:
    digest, size = file_digest(path)
    return (
        reference.get("path") == expected_path
        and reference.get("sha256") == digest
        and reference.get("bytes") == size
        and (rows is None or reference.get("rows") == rows)
    )


def _validated_safe_reference(
    reference: Mapping[str, Any],
    *,
    name: str,
    expected_rows: int | None,
) -> dict[str, Any]:
    """Copy only path/digest/size/count fields into scoring artifacts."""

    expected_keys = (
        SAFE_ARTIFACT_REFERENCE_KEYS
        if expected_rows is None
        else SAFE_ROW_ARTIFACT_REFERENCE_KEYS
    )
    if not isinstance(reference, dict) or set(reference) != expected_keys:
        raise EvaluationError(f"{name} must use the exact safe artifact schema")
    path = reference.get("path")
    digest = reference.get("sha256")
    byte_count = reference.get("bytes")
    if not isinstance(path, str) or not path.strip():
        raise EvaluationError(f"{name} path must be nonempty")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or digest != digest.lower()
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise EvaluationError(f"{name} SHA-256 is malformed")
    if (
        isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count < 0
    ):
        raise EvaluationError(f"{name} byte count is invalid")
    safe: dict[str, Any] = {
        "path": path,
        "sha256": digest,
        "bytes": byte_count,
    }
    if expected_rows is not None:
        rows = reference.get("rows")
        if (
            isinstance(rows, bool)
            or not isinstance(rows, int)
            or rows != expected_rows
        ):
            raise EvaluationError(f"{name} row count is invalid")
        safe["rows"] = rows
    return safe


def _validated_dataset_lineage(
    lineage: Mapping[str, Any],
    *,
    expected_label_rows: int,
) -> dict[str, Any]:
    if not isinstance(lineage, dict) or set(lineage) != {
        "schema_version",
        "verification",
        "dataset_lock",
        "questions",
        "labels",
    }:
        raise EvaluationError("Dataset lineage must use the exact safe schema")
    if lineage.get("schema_version") != DATASET_LINEAGE_SCHEMA_VERSION:
        raise EvaluationError("Unsupported dataset-lineage schema")
    verification = lineage.get("verification")
    if verification not in {"locked_dataset", "test_fixture"}:
        raise EvaluationError("Invalid dataset-lineage verification mode")
    labels = _validated_safe_reference(
        lineage.get("labels"),
        name="Dataset-lineage labels reference",
        expected_rows=expected_label_rows,
    )
    questions = _validated_safe_reference(
        lineage.get("questions"),
        name="Dataset-lineage questions reference",
        expected_rows=expected_label_rows,
    )
    lock_value = lineage.get("dataset_lock")
    if verification == "locked_dataset":
        dataset_lock: dict[str, Any] | None = _validated_safe_reference(
            lock_value,
            name="Dataset-lock reference",
            expected_rows=None,
        )
    else:
        if lock_value is not None:
            raise EvaluationError("Test-fixture lineage cannot claim a dataset lock")
        dataset_lock = None
    return {
        "schema_version": DATASET_LINEAGE_SCHEMA_VERSION,
        "verification": verification,
        "dataset_lock": dataset_lock,
        "questions": questions,
        "labels": labels,
    }


def _load_locked_scoring_data(
    repository_root: Path,
    config: ScoringConfig,
) -> LockedScoringData:
    """Load labels and prove both dataset views against one immutable lock."""

    lock_path = _resolve_locked_path(repository_root, config.dataset_lock_path)
    labels_path = _resolve_locked_path(repository_root, config.labels_path)
    try:
        lock = load_dataset_lock(lock_path)
    except DatasetPreparationError as exc:
        raise EvaluationError(str(exc)) from exc
    outputs = lock["outputs"]
    label_spec = outputs["labels"]
    question_spec = outputs["questions"]
    if config.labels_path != label_spec["path"]:
        raise EvaluationError(
            "Scoring labels_path must exactly equal outputs.labels.path in the "
            "dataset lock"
        )
    expected_labels_path = _resolve_locked_path(
        repository_root, label_spec["path"]
    )
    if labels_path != expected_labels_path:
        raise EvaluationError(
            "Scoring labels path must be the locked evaluation artifact: "
            f"{expected_labels_path}"
        )
    actual_sha, actual_bytes = file_digest(labels_path)
    if actual_sha != label_spec["sha256"] or actual_bytes != label_spec["bytes"]:
        raise EvaluationError("Labels file does not match the dataset lock")

    firewall = lock.get("label_firewall")
    if not isinstance(firewall, dict) or set(firewall) != {
        "training_input",
        "training_keys",
        "evaluation_labels",
        "forbidden_training_keys",
    }:
        raise EvaluationError("Dataset lock has an invalid label-firewall schema")
    training_keys = firewall.get("training_keys")
    forbidden_training_keys = firewall.get("forbidden_training_keys")
    if (
        firewall.get("training_input") != question_spec["path"]
        or firewall.get("evaluation_labels") != label_spec["path"]
        or not isinstance(training_keys, list)
        or not all(isinstance(key, str) for key in training_keys)
        or set(training_keys) != {"id", "problem"}
        or not isinstance(forbidden_training_keys, list)
        or not all(isinstance(key, str) for key in forbidden_training_keys)
        or set(forbidden_training_keys)
        != {"answer", "solution", "subject", "level"}
    ):
        raise EvaluationError("Dataset lock label-firewall paths or keys are invalid")
    questions_path = _resolve_locked_path(repository_root, question_spec["path"])
    question_sha, question_bytes = file_digest(questions_path)
    if (
        question_sha != question_spec["sha256"]
        or question_bytes != question_spec["bytes"]
    ):
        raise EvaluationError("Questions file does not match the dataset lock")

    labels: list[EvalLabel] = []
    seen_ids: set[str] = set()
    for line_number, row in enumerate(read_jsonl(labels_path), start=1):
        expected_keys = {"id", "answer", "solution", "subject", "level"}
        if set(row) != expected_keys:
            raise EvaluationError(f"Unsafe label schema at row {line_number}")
        if not all(
            isinstance(row[key], str) and row[key].strip()
            for key in ("id", "answer", "solution", "subject")
        ):
            raise EvaluationError(f"Invalid label strings at row {line_number}")
        if (
            isinstance(row["level"], bool)
            or not isinstance(row["level"], int)
            or row["level"] not in range(1, 6)
        ):
            raise EvaluationError(f"Invalid label level at row {line_number}")
        if row["id"] in seen_ids:
            raise EvaluationError(f"Duplicate label ID: {row['id']}")
        seen_ids.add(row["id"])
        # Deliberately do not retain row["solution"] beyond schema validation.
        labels.append(
            EvalLabel(
                id=row["id"],
                answer=row["answer"],
                subject=row["subject"],
                level=row["level"],
            )
        )
    if len(labels) != label_spec["rows"]:
        raise EvaluationError(
            f"Expected {label_spec['rows']} labels, found {len(labels)}"
        )

    question_ids: list[str] = []
    seen_question_ids: set[str] = set()
    for line_number, row in enumerate(read_jsonl(questions_path), start=1):
        if set(row) != {"id", "problem"}:
            raise EvaluationError(f"Unsafe question schema at row {line_number}")
        if not all(
            isinstance(row[key], str) and row[key].strip()
            for key in ("id", "problem")
        ):
            raise EvaluationError(f"Invalid question strings at row {line_number}")
        if row["id"] in seen_question_ids:
            raise EvaluationError(f"Duplicate question ID: {row['id']}")
        seen_question_ids.add(row["id"])
        question_ids.append(row["id"])
    if len(question_ids) != question_spec["rows"]:
        raise EvaluationError(
            f"Expected {question_spec['rows']} questions, found {len(question_ids)}"
        )
    label_ids = [label.id for label in labels]
    if question_ids != label_ids:
        raise EvaluationError(
            "Locked question and label artifacts must have identical ordered IDs"
        )

    labels_reference = _artifact_reference(
        labels_path,
        config.labels_path,
        rows=len(labels),
    )
    questions_reference = _artifact_reference(
        questions_path,
        question_spec["path"],
        rows=len(question_ids),
    )
    dataset_lock_reference = _artifact_reference(
        lock_path,
        config.dataset_lock_path,
    )
    lineage = _validated_dataset_lineage(
        {
            "schema_version": DATASET_LINEAGE_SCHEMA_VERSION,
            "verification": "locked_dataset",
            "dataset_lock": dataset_lock_reference,
            "questions": questions_reference,
            "labels": labels_reference,
        },
        expected_label_rows=len(labels),
    )
    return LockedScoringData(
        labels=tuple(labels),
        labels_reference=labels_reference,
        dataset_lineage=lineage,
    )


def load_locked_labels(
    repository_root: Path,
    config: ScoringConfig,
) -> tuple[list[EvalLabel], dict[str, Any]]:
    """Load evaluation fields while retaining the original public boundary."""

    data = _load_locked_scoring_data(repository_root, config)
    return list(data.labels), dict(data.labels_reference)


def _validated_label_reference(
    reference: Mapping[str, Any],
    labels: Sequence[EvalLabel],
) -> dict[str, Any]:
    """Return the only label provenance fields that may enter an artifact."""

    return _validated_safe_reference(
        reference,
        name="Label reference",
        expected_rows=len(labels),
    )


def _validate_generation_row(row: Mapping[str, Any], expected_stage: str) -> None:
    if not isinstance(row, dict) or set(row) != GENERATION_ROW_KEYS:
        raise EvaluationError("Invalid generation schema")
    if row.get("schema_version") != 1 or row.get("stage") != expected_stage:
        raise EvaluationError(
            f"Expected a {expected_stage} generation, found {row.get('stage')}"
        )
    if not isinstance(row.get("text"), str):
        raise EvaluationError("Generation text must be a string")
    if sha256_text(row["text"]) != row.get("output_sha256"):
        raise EvaluationError(f"Generation output hash mismatch: {row.get('task_id')}")


def score_generation_rows(
    generations: Sequence[Mapping[str, Any]],
    labels: Sequence[EvalLabel],
    config: ScoringConfig,
    *,
    expected_stage: str,
) -> list[dict[str, Any]]:
    """Grade verified generation rows without serializing reference answers."""

    label_by_id = {label.id: label for label in labels}
    if len(label_by_id) != len(labels):
        raise EvaluationError("Labels must have unique IDs")
    generation_ids: set[str] = set()
    seen_tasks: set[str] = set()
    graders = (config.primary_grader, *config.diagnostic_graders)
    scores: list[dict[str, Any]] = []
    for generation in generations:
        _validate_generation_row(generation, expected_stage)
        task_id = str(generation["task_id"])
        question_id = str(generation["question_id"])
        if task_id in seen_tasks:
            raise EvaluationError(f"Duplicate generation task ID: {task_id}")
        seen_tasks.add(task_id)
        if question_id not in label_by_id:
            raise EvaluationError(f"No evaluation label for question {question_id}")
        generation_ids.add(question_id)
        label = label_by_id[question_id]
        graded = grade_response(
            str(generation["text"]),
            label.answer,
            graders=graders,
            timeout_seconds=config.parsing_timeout_seconds,
            max_answer_chars=config.max_answer_chars,
        )
        scores.append(
            {
                "schema_version": 1,
                "stage": expected_stage,
                "task_id": task_id,
                "generation_output_sha256": generation["output_sha256"],
                "question_id": question_id,
                "rollout_index": generation["rollout_index"],
                "source_task_ids": generation["source_task_ids"],
                "extraction": graded["extraction"],
                "grades": graded["grades"],
                "primary_grader": config.primary_grader,
                "primary_correct": graded["grades"][config.primary_grader]["correct"],
                "subject": label.subject,
                "level": label.level,
            }
        )
    if generation_ids != set(label_by_id):
        raise EvaluationError(
            "Generation/label coverage mismatch: "
            f"missing={sorted(set(label_by_id) - generation_ids)}, "
            f"extra={sorted(generation_ids - set(label_by_id))}"
        )
    return scores


def _validate_and_order_scores(
    scores: Sequence[Mapping[str, Any]],
    generations: Sequence[Mapping[str, Any]],
    config: ScoringConfig,
) -> list[dict[str, Any]]:
    """Bind score rows to predictions and reject any serialized reference fields."""

    score_by_id: dict[str, Mapping[str, Any]] = {}
    for index, score in enumerate(scores):
        if not isinstance(score, dict) or set(score) != SCORE_ROW_KEYS:
            raise EvaluationError(f"Invalid score schema at index {index}")
        task_id = score.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise EvaluationError(f"Invalid score task ID at index {index}")
        if task_id in score_by_id:
            raise EvaluationError(f"Duplicate score task ID: {task_id}")
        score_by_id[task_id] = score
    generation_ids = {str(row["task_id"]) for row in generations}
    if set(score_by_id) != generation_ids:
        raise EvaluationError("Score/generation task coverage mismatch")

    graders = {config.primary_grader, *config.diagnostic_graders}
    ordered: list[dict[str, Any]] = []
    for generation in generations:
        task_id = str(generation["task_id"])
        score = score_by_id[task_id]
        expected_fields = {
            "schema_version": 1,
            "stage": generation["stage"],
            "task_id": task_id,
            "generation_output_sha256": generation["output_sha256"],
            "question_id": generation["question_id"],
            "rollout_index": generation["rollout_index"],
            "source_task_ids": generation["source_task_ids"],
            "primary_grader": config.primary_grader,
        }
        for field, expected in expected_fields.items():
            if score.get(field) != expected:
                raise EvaluationError(
                    f"Score {task_id} does not match prediction field {field}"
                )
        extraction = score.get("extraction")
        if (
            not isinstance(extraction, dict)
            or set(extraction) != {"value", "status"}
            or not isinstance(extraction.get("status"), str)
            or (
                extraction.get("value") is not None
                and not isinstance(extraction.get("value"), str)
            )
        ):
            raise EvaluationError(f"Invalid extraction record for score {task_id}")
        grades = score.get("grades")
        if not isinstance(grades, dict) or set(grades) != graders:
            raise EvaluationError(f"Invalid grader coverage for score {task_id}")
        for grader, grade in grades.items():
            if (
                not isinstance(grade, dict)
                or set(grade) != {"correct", "status"}
                or not isinstance(grade.get("correct"), bool)
                or not isinstance(grade.get("status"), str)
            ):
                raise EvaluationError(f"Invalid {grader} grade for score {task_id}")
        if score.get("primary_correct") != grades[config.primary_grader]["correct"]:
            raise EvaluationError(f"Inconsistent primary grade for score {task_id}")
        if not isinstance(score.get("subject"), str) or not score["subject"].strip():
            raise EvaluationError(f"Invalid subject for score {task_id}")
        if (
            isinstance(score.get("level"), bool)
            or not isinstance(score.get("level"), int)
            or score["level"] not in range(1, 6)
        ):
            raise EvaluationError(f"Invalid level for score {task_id}")
        ordered.append(dict(score))
    return ordered


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _standard_error(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    return statistics.stdev(values) / math.sqrt(len(values))


def _accuracy(rows: Iterable[Mapping[str, Any]], grader: str) -> float | None:
    values = [float(bool(row["grades"][grader]["correct"])) for row in rows]
    return _mean(values)


def _literal_plurality_row(
    rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    valid = [
        row
        for row in rows
        if row["extraction"]["status"] == "ok"
        and row["extraction"]["value"] is not None
    ]
    if not valid:
        return None
    counts = Counter(str(row["extraction"]["value"]).strip() for row in valid)
    highest_count = max(counts.values())
    winners = {value for value, count in counts.items() if count == highest_count}
    # Deterministic tie rule: earliest rollout index among tied literal strings.
    return min(
        (
            row
            for row in valid
            if str(row["extraction"]["value"]).strip() in winners
        ),
        key=lambda row: int(row["rollout_index"]),
    )


def _breakdown(
    rows: Sequence[Mapping[str, Any]],
    field: str,
    grader: str,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(row)
    return {
        key: {"count": len(group), "accuracy": _accuracy(group, grader)}
        for key, group in sorted(grouped.items())
    }


def _extraction_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row["extraction"]["status"]) for row in rows)
    ok = counts.get("ok", 0)
    return {
        "boxed_extraction_rate": ok / len(rows) if rows else None,
        "status_counts": dict(sorted(counts.items())),
    }


def raw_baseline_selection(
    protocol_version: str,
    seed: int,
    question_id: str,
    *,
    rollouts: int = 8,
) -> tuple[int, str]:
    """Choose a raw rollout without relying on process-global randomness."""

    if not isinstance(protocol_version, str) or not protocol_version:
        raise EvaluationError("Selection protocol version must be nonempty")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise EvaluationError("Raw-baseline seed must be in [0, 2^63)")
    if not isinstance(question_id, str) or not question_id:
        raise EvaluationError("Selection question ID must be nonempty")
    if isinstance(rollouts, bool) or not isinstance(rollouts, int) or rollouts <= 0:
        raise EvaluationError("Raw-baseline rollout count must be positive")
    payload = canonical_json_bytes(
        [
            RAW_BASELINE_SELECTION_DOMAIN,
            protocol_version,
            seed,
            question_id,
        ]
    )
    digest = sha256_bytes(payload)
    return int(digest, 16) % rollouts, digest


def select_raw_baseline_generations(
    raw_generations: Sequence[Mapping[str, Any]],
    synthesis_generations: Sequence[Mapping[str, Any]],
    config: ScoringConfig,
    *,
    expected_rollouts: int = 8,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Select one verified existing raw generation for each synthesis result."""

    if config.raw_baseline_selection != RAW_BASELINE_SELECTION_METHOD:
        raise EvaluationError("Unsupported raw-baseline selection method")
    raw_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for generation in raw_generations:
        _validate_generation_row(generation, "raw")
        raw_groups[str(generation["question_id"])].append(generation)
    synth_by_id: dict[str, Mapping[str, Any]] = {}
    for generation in synthesis_generations:
        _validate_generation_row(generation, "synthesis")
        question_id = str(generation["question_id"])
        if question_id in synth_by_id:
            raise EvaluationError(
                f"Duplicate synthesis generation for question {question_id}"
            )
        synth_by_id[question_id] = generation
    if set(synth_by_id) != set(raw_groups):
        raise EvaluationError("Raw and synthesis generation coverage must match")

    selected: list[dict[str, Any]] = []
    selection_by_id: dict[str, dict[str, Any]] = {}
    for question_id in sorted(synth_by_id):
        rows = sorted(
            raw_groups[question_id], key=lambda row: int(row["rollout_index"])
        )
        indexes = [row["rollout_index"] for row in rows]
        if indexes != list(range(expected_rollouts)):
            raise EvaluationError(
                f"Raw generation group {question_id} does not contain indexes "
                f"0..{expected_rollouts - 1}"
            )
        expected_source_ids = [str(row["task_id"]) for row in rows]
        if synth_by_id[question_id]["source_task_ids"] != expected_source_ids:
            raise EvaluationError(
                f"Synthesis source lineage does not match raw group {question_id}"
            )
        rollout_index, digest = raw_baseline_selection(
            config.protocol_version,
            config.raw_baseline_seed,
            question_id,
            rollouts=expected_rollouts,
        )
        row = dict(rows[rollout_index])
        selected.append(row)
        selection_by_id[question_id] = {
            "method": config.raw_baseline_selection,
            "seed": config.raw_baseline_seed,
            "domain": RAW_BASELINE_SELECTION_DOMAIN,
            "digest_sha256": digest,
            "rollout_index": rollout_index,
            "task_id": row["task_id"],
        }
    return selected, selection_by_id


def build_paired_score_rows(
    synthesis_scores: Sequence[Mapping[str, Any]],
    raw_baseline_scores: Sequence[Mapping[str, Any]],
    selections: Mapping[str, Mapping[str, Any]],
    config: ScoringConfig,
) -> list[dict[str, Any]]:
    synth_by_id = {str(row["question_id"]): row for row in synthesis_scores}
    raw_by_id = {str(row["question_id"]): row for row in raw_baseline_scores}
    if (
        len(synth_by_id) != len(synthesis_scores)
        or len(raw_by_id) != len(raw_baseline_scores)
        or set(synth_by_id) != set(raw_by_id)
        or set(synth_by_id) != set(selections)
    ):
        raise EvaluationError("Paired score coverage must be one-to-one")
    graders = (config.primary_grader, *config.diagnostic_graders)
    pairs: list[dict[str, Any]] = []
    for question_id in sorted(synth_by_id):
        synthesis = synth_by_id[question_id]
        raw = raw_by_id[question_id]
        selection = selections[question_id]
        if (
            selection.get("task_id") != raw["task_id"]
            or selection.get("rollout_index") != raw["rollout_index"]
        ):
            raise EvaluationError(
                f"Raw score does not match the selection for {question_id}"
            )
        pairs.append(
            {
                "schema_version": 1,
                "question_id": question_id,
                "selection": dict(selection),
                "synthesis": {
                    "task_id": synthesis["task_id"],
                    "output_sha256": synthesis["generation_output_sha256"],
                    "extraction": synthesis["extraction"],
                    "grades": synthesis["grades"],
                },
                "raw_baseline": {
                    "task_id": raw["task_id"],
                    "rollout_index": raw["rollout_index"],
                    "output_sha256": raw["generation_output_sha256"],
                    "extraction": raw["extraction"],
                    "grades": raw["grades"],
                },
                "grader_deltas": {
                    grader: int(synthesis["grades"][grader]["correct"])
                    - int(raw["grades"][grader]["correct"])
                    for grader in graders
                },
                "subject": synthesis["subject"],
                "level": synthesis["level"],
            }
        )
    return pairs


def summarize_raw_scores(
    scores: Sequence[Mapping[str, Any]],
    config: ScoringConfig,
    *,
    expected_rollouts: int,
    non_reportable_reasons: Sequence[str] = (),
) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in scores:
        groups[str(row["question_id"])].append(row)
    expected_indexes = list(range(expected_rollouts))
    for question_id, rows in groups.items():
        indexes = sorted(int(row["rollout_index"]) for row in rows)
        if indexes != expected_indexes:
            raise EvaluationError(
                f"Raw score group {question_id} has indexes {indexes}, "
                f"expected {expected_indexes}"
            )

    ordered_groups = [
        sorted(groups[question_id], key=lambda row: int(row["rollout_index"]))
        for question_id in sorted(groups)
    ]
    rollout_zero = [rows[0] for rows in ordered_groups]
    graders = (config.primary_grader, *config.diagnostic_graders)
    grader_metrics: dict[str, Any] = {}
    for grader in graders:
        problem_means = [
            sum(bool(row["grades"][grader]["correct"]) for row in rows) / len(rows)
            for rows in ordered_groups
        ]
        any_correct = [
            float(any(bool(row["grades"][grader]["correct"]) for row in rows))
            for rows in ordered_groups
        ]
        plurality_rows = [_literal_plurality_row(rows) for rows in ordered_groups]
        plurality_values = [
            float(bool(row["grades"][grader]["correct"])) if row is not None else 0.0
            for row in plurality_rows
        ]
        zero_values = [
            float(bool(row["grades"][grader]["correct"])) for row in rollout_zero
        ]
        grader_metrics[grader] = {
            "rollout_index_0_accuracy": _mean(zero_values),
            "rollout_index_0_standard_error": _standard_error(zero_values),
            "mean_rollout_accuracy": _mean(problem_means),
            "mean_rollout_standard_error_across_problems": _standard_error(
                problem_means
            ),
            f"empirical_any_correct_at_{expected_rollouts}": _mean(any_correct),
            "literal_plurality_vote_accuracy": _mean(plurality_values),
        }

    reasons = sorted(set(non_reportable_reasons))
    return {
        "schema_version": 1,
        "stage": "raw",
        "reportable": not reasons,
        "non_reportable_reasons": reasons,
        "primary_metric": "rollout_index_0_accuracy",
        "primary_grader": config.primary_grader,
        "counts": {
            "problems": len(groups),
            "generations": len(scores),
            "rollouts_per_problem": expected_rollouts,
        },
        "extraction": _extraction_summary(scores),
        "graders": grader_metrics,
        "primary_breakdown": {
            "subject": _breakdown(rollout_zero, "subject", config.primary_grader),
            "level": _breakdown(rollout_zero, "level", config.primary_grader),
        },
        "protocol_notes": {
            "raw_primary": (
                "This optional legacy all-rollout diagnostic reports index 0 and "
                "the mean across eight. It does not define the paired primary; "
                "score-synthesis uses the preregistered raw-baseline selector."
            ),
            "plurality": (
                "Plurality groups literal extracted answer strings and breaks ties "
                "by the earliest rollout index."
            ),
        },
    }


def summarize_synthesis_scores(
    scores: Sequence[Mapping[str, Any]],
    raw_baseline_scores: Sequence[Mapping[str, Any]],
    paired_scores: Sequence[Mapping[str, Any]],
    config: ScoringConfig,
    *,
    raw_candidate_generations: int,
    non_reportable_reasons: Sequence[str] = (),
) -> dict[str, Any]:
    if len({row["question_id"] for row in scores}) != len(scores):
        raise EvaluationError("Synthesis scoring requires one result per question")
    synth_by_id = {str(row["question_id"]): row for row in scores}
    raw_by_id = {str(row["question_id"]): row for row in raw_baseline_scores}
    if len(raw_by_id) != len(raw_baseline_scores) or set(synth_by_id) != set(raw_by_id):
        raise EvaluationError("Raw-baseline and synthesis score coverage must match")
    if len(paired_scores) != len(scores):
        raise EvaluationError("Paired-score coverage must match synthesis scores")

    graders = (config.primary_grader, *config.diagnostic_graders)
    grader_metrics: dict[str, Any] = {}
    for grader in graders:
        synthesis_values: list[float] = []
        raw_values: list[float] = []
        paired_deltas: list[float] = []
        for question_id in sorted(synth_by_id):
            synthesis_row = synth_by_id[question_id]
            raw_row = raw_by_id[question_id]
            synthesis_correct = float(
                bool(synthesis_row["grades"][grader]["correct"])
            )
            raw_correct = float(bool(raw_row["grades"][grader]["correct"]))
            synthesis_values.append(synthesis_correct)
            raw_values.append(raw_correct)
            paired_deltas.append(synthesis_correct - raw_correct)
        grader_metrics[grader] = {
            "synthesis_accuracy": _mean(synthesis_values),
            "synthesis_standard_error": _standard_error(synthesis_values),
            "raw_baseline_accuracy": _mean(raw_values),
            "raw_baseline_standard_error": _standard_error(raw_values),
            "paired_delta_vs_raw_baseline": _mean(paired_deltas),
            "paired_delta_vs_raw_baseline_standard_error": _standard_error(
                paired_deltas
            ),
        }

    comparable = 0
    disagreements = 0
    correct_when_disagree = 0
    primary = config.primary_grader
    for question_id in sorted(synth_by_id):
        synthesis_row = synth_by_id[question_id]
        raw_row = raw_by_id[question_id]
        synth_answer = synthesis_row["extraction"]["value"]
        raw_answer = raw_row["extraction"]["value"]
        if synth_answer is not None and raw_answer is not None:
            comparable += 1
            if str(synth_answer).strip() != str(raw_answer).strip():
                disagreements += 1
                correct_when_disagree += int(
                    synthesis_row["grades"][primary]["correct"]
                )

    primary_rows = [synth_by_id[question_id] for question_id in sorted(synth_by_id)]
    selected_rows = [raw_by_id[question_id] for question_id in sorted(raw_by_id)]
    rollout_histogram = Counter(
        int(row["selection"]["rollout_index"]) for row in paired_scores
    )
    reasons = sorted(set(non_reportable_reasons))
    return {
        "schema_version": 2,
        "stage": "synthesis",
        "reportable": not reasons,
        "non_reportable_reasons": reasons,
        "primary_metric": "synthesis_accuracy",
        "primary_grader": primary,
        "counts": {
            "problems": len(scores),
            "synthesis_generations": len(scores),
            "raw_candidate_generations": raw_candidate_generations,
            "raw_baseline_generations_scored": len(raw_baseline_scores),
        },
        "extraction": {
            "synthesis": _extraction_summary(primary_rows),
            "raw_baseline": _extraction_summary(selected_rows),
        },
        "graders": grader_metrics,
        "primary_breakdown": {
            "subject": _breakdown(primary_rows, "subject", primary),
            "level": _breakdown(primary_rows, "level", primary),
        },
        "raw_baseline_selection": {
            "method": config.raw_baseline_selection,
            "seed": config.raw_baseline_seed,
            "domain": RAW_BASELINE_SELECTION_DOMAIN,
            "rollout_index_counts": {
                str(index): rollout_histogram.get(index, 0) for index in range(8)
            },
        },
        "synthesis_analysis": {
            "raw_baseline_comparable_problems": comparable,
            "disagrees_with_raw_baseline_count": disagreements,
            "disagrees_with_raw_baseline_rate": (
                disagreements / comparable if comparable else None
            ),
            "synthesis_accuracy_when_disagreeing_with_raw_baseline": (
                correct_when_disagree / disagreements if disagreements else None
            ),
        },
    }


def _execution_and_generations(
    run_dir: Path,
    manifest: Mapping[str, Any],
    requests: Sequence[GenerationRequest],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return verify_complete_execution(run_dir, manifest, requests)


def _scoring_manifest_fingerprint(value: Mapping[str, Any]) -> str:
    semantic = dict(value)
    semantic.pop("scoring_fingerprint", None)
    return sha256_bytes(canonical_json_bytes(semantic))


def _raw_plan_questions_reference(
    raw_manifest: Mapping[str, Any],
    *,
    expected_rows: int,
) -> dict[str, Any]:
    if raw_manifest.get("kind") != "raw":
        raise EvaluationError("Dataset lineage must bind to a raw plan")
    counts = raw_manifest.get("counts")
    if not isinstance(counts, dict) or counts.get("problems") != expected_rows:
        raise EvaluationError("Raw-plan problem count does not match dataset lineage")
    inputs = raw_manifest.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {"questions", "dataset_lock"}:
        raise EvaluationError("Raw-plan inputs do not use the exact dataset schema")
    return _validated_safe_reference(
        inputs.get("questions"),
        name="Raw-plan questions reference",
        expected_rows=expected_rows,
    )


def _bind_dataset_lineage_to_raw_plan(
    lineage: Mapping[str, Any],
    raw_manifest: Mapping[str, Any],
    config: ScoringConfig,
    *,
    expected_label_rows: int,
) -> dict[str, Any]:
    """Prove the scoring labels and raw questions share one dataset lineage."""

    safe = _validated_dataset_lineage(
        lineage,
        expected_label_rows=expected_label_rows,
    )
    questions = _raw_plan_questions_reference(
        raw_manifest,
        expected_rows=expected_label_rows,
    )
    if questions != safe["questions"]:
        raise EvaluationError(
            "Raw-plan questions do not match the scoring dataset lock"
        )
    if raw_manifest.get("protocol_version") != config.protocol_version:
        raise EvaluationError("Raw plan and scoring protocol versions do not match")
    firewall = raw_manifest.get("label_firewall")
    if not isinstance(firewall, dict):
        raise EvaluationError("Raw plan has no valid label-firewall record")

    if safe["verification"] == "locked_dataset":
        lock_reference = safe["dataset_lock"]
        assert isinstance(lock_reference, dict)
        raw_inputs = raw_manifest["inputs"]
        raw_lock_reference = _validated_safe_reference(
            raw_inputs.get("dataset_lock"),
            name="Raw-plan dataset-lock reference",
            expected_rows=None,
        )
        if raw_lock_reference != lock_reference:
            raise EvaluationError(
                "Raw plan and scoring labels do not use the same exact "
                "dataset-lock artifact"
            )
        raw_config = raw_manifest.get("config")
        if not isinstance(raw_config, dict):
            raise EvaluationError("Raw plan has no embedded config")
        if (
            raw_config.get("dataset_lock_path") != config.dataset_lock_path
            or raw_config.get("dataset_lock_path") != lock_reference["path"]
        ):
            raise EvaluationError(
                "Raw plan and scoring labels do not use the same exact dataset lock"
            )
        if raw_config.get("questions_path") != safe["questions"]["path"]:
            raise EvaluationError(
                "Raw config question path does not match the scoring dataset lock"
            )
        if safe["labels"]["path"] != config.labels_path:
            raise EvaluationError(
                "Scoring label path does not match the scoring dataset lock"
            )
        if firewall.get("locked_questions_verified") is not True:
            raise EvaluationError(
                "A reportable scoring run requires lock-verified raw questions"
            )
    else:
        if raw_manifest["inputs"].get("dataset_lock") is not None:
            raise EvaluationError("Test-fixture raw plans cannot claim a dataset lock")
        if firewall.get("locked_questions_verified") is not False:
            raise EvaluationError(
                "Test-fixture scoring requires an explicitly unlocked raw question fixture"
            )
    return safe


def _test_fixture_dataset_lineage(
    raw_manifest: Mapping[str, Any],
    labels_reference: Mapping[str, Any],
    *,
    label_rows: int,
) -> dict[str, Any]:
    questions_reference = _raw_plan_questions_reference(
        raw_manifest,
        expected_rows=label_rows,
    )
    return _validated_dataset_lineage(
        {
            "schema_version": DATASET_LINEAGE_SCHEMA_VERSION,
            "verification": "test_fixture",
            "dataset_lock": None,
            "questions": questions_reference,
            "labels": labels_reference,
        },
        expected_label_rows=label_rows,
    )


def _generation_totals(generations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    prompt_tokens = [row["usage"]["prompt_tokens"] for row in generations]
    completion_tokens = [row["usage"]["completion_tokens"] for row in generations]
    return {
        "calls": len(generations),
        "known_prompt_tokens": sum(value for value in prompt_tokens if value is not None),
        "unknown_prompt_token_calls": sum(value is None for value in prompt_tokens),
        "known_completion_tokens": sum(
            value for value in completion_tokens if value is not None
        ),
        "unknown_completion_token_calls": sum(
            value is None for value in completion_tokens
        ),
        "finish_reason_counts": dict(
            sorted(Counter(str(row["finish_reason"]) for row in generations).items())
        ),
    }


def _preflight_payloads(
    payloads: Sequence[tuple[Path, bytes]],
    *,
    force: bool,
) -> None:
    for path, payload in payloads:
        if path.exists() and (not path.is_file() or path.read_bytes() != payload) and not force:
            raise EvaluationError(f"Refusing to overwrite mismatched scoring artifact {path}")


def _score_run_with_labels(
    run_dir: Path,
    labels: Sequence[EvalLabel],
    labels_reference: Mapping[str, Any],
    config: ScoringConfig,
    *,
    dataset_lineage: Mapping[str, Any] | None = None,
    raw_run_dir: Path | None = None,
    label_non_reportable_reasons: Sequence[str] = (),
    force: bool = False,
) -> dict[str, Any]:
    """Internal scorer for a labels/reference pair established at one boundary."""

    safe_labels_reference = _validated_label_reference(labels_reference, labels)
    manifest, requests = load_plan(run_dir)
    stage = str(manifest["kind"])
    execution, generations = _execution_and_generations(run_dir, manifest, requests)
    scores = score_generation_rows(
        generations,
        labels,
        config,
        expected_stage=stage,
    )
    scores = _validate_and_order_scores(scores, generations, config)
    reportability_reasons = [
        *execution["non_reportable_reasons"],
        *label_non_reportable_reasons,
    ]

    paired_scores: list[dict[str, Any]] = []
    raw_generation_reference: dict[str, Any] | None = None
    raw_execution_reference: dict[str, Any] | None = None
    safe_dataset_lineage: dict[str, Any]
    if stage == "raw":
        candidate_lineage = dataset_lineage or _test_fixture_dataset_lineage(
            manifest,
            safe_labels_reference,
            label_rows=len(labels),
        )
        safe_dataset_lineage = _bind_dataset_lineage_to_raw_plan(
            candidate_lineage,
            manifest,
            config,
            expected_label_rows=len(labels),
        )
        if safe_dataset_lineage["labels"] != safe_labels_reference:
            raise EvaluationError(
                "Dataset lineage labels do not match the loaded label artifact"
            )
        summary = summarize_raw_scores(
            scores,
            config,
            expected_rollouts=int(manifest["counts"]["rollouts_per_problem"]),
            non_reportable_reasons=reportability_reasons,
        )
    else:
        if raw_run_dir is None:
            raise EvaluationError("Synthesis scoring requires --raw-run-dir")
        raw_manifest, raw_requests = load_plan(raw_run_dir, expected_kind="raw")
        if manifest["inputs"].get("raw_plan_fingerprint") != raw_manifest.get(
            "plan_fingerprint"
        ):
            raise EvaluationError("Synthesis plan is not derived from this raw run")
        raw_execution, raw_generations = _execution_and_generations(
            raw_run_dir,
            raw_manifest,
            raw_requests,
        )
        candidate_lineage = dataset_lineage or _test_fixture_dataset_lineage(
            raw_manifest,
            safe_labels_reference,
            label_rows=len(labels),
        )
        safe_dataset_lineage = _bind_dataset_lineage_to_raw_plan(
            candidate_lineage,
            raw_manifest,
            config,
            expected_label_rows=len(labels),
        )
        if safe_dataset_lineage["labels"] != safe_labels_reference:
            raise EvaluationError(
                "Dataset lineage labels do not match the loaded label artifact"
            )
        upstream_execution_reference = manifest["inputs"].get("raw_execution")
        if not isinstance(upstream_execution_reference, dict) or not _reference_matches(
            upstream_execution_reference,
            raw_run_dir / EXECUTION_NAME,
            expected_path="upstream/raw/execution.json",
        ):
            raise EvaluationError(
                "Raw execution provenance no longer matches the synthesis plan input"
            )
        if manifest["inputs"].get("raw_execution_non_reportable") != bool(
            raw_execution["non_reportable"]
        ):
            raise EvaluationError("Raw reportability lineage is inconsistent")
        upstream_generation_reference = manifest["inputs"].get("raw_generations")
        if not isinstance(upstream_generation_reference, dict) or not _reference_matches(
            upstream_generation_reference,
            raw_run_dir / GENERATIONS_NAME,
            expected_path="upstream/raw/generations.jsonl",
            rows=len(raw_generations),
        ):
            raise EvaluationError(
                "Raw generations no longer match the synthesis plan input"
            )
        selected_raw_generations, selections = select_raw_baseline_generations(
            raw_generations,
            generations,
            config,
        )
        raw_baseline_scores = score_generation_rows(
            selected_raw_generations,
            labels,
            config,
            expected_stage="raw",
        )
        raw_baseline_scores = _validate_and_order_scores(
            raw_baseline_scores,
            selected_raw_generations,
            config,
        )
        paired_scores = build_paired_score_rows(
            scores,
            raw_baseline_scores,
            selections,
            config,
        )
        raw_generation_reference = _artifact_reference(
            raw_run_dir / GENERATIONS_NAME,
            "upstream/raw/generations.jsonl",
            rows=len(raw_generations),
        )
        raw_execution_reference = _artifact_reference(
            raw_run_dir / EXECUTION_NAME,
            "upstream/raw/execution.json",
        )
        reportability_reasons.extend(
            f"raw_dependency:{reason}"
            for reason in raw_execution["non_reportable_reasons"]
        )
        summary = summarize_synthesis_scores(
            scores,
            raw_baseline_scores,
            paired_scores,
            config,
            raw_candidate_generations=len(raw_generations),
            non_reportable_reasons=reportability_reasons,
        )
        summary["execution_totals"] = {
            "synthesis": _generation_totals(generations),
            "raw_dependency": _generation_totals(raw_generations),
        }
    if stage == "raw":
        summary["execution_totals"] = _generation_totals(generations)

    scores_payload = canonical_jsonl_bytes(scores)
    paired_scores_payload = canonical_jsonl_bytes(paired_scores)
    summary_payload = canonical_json_bytes(summary)
    generations_reference = _artifact_reference(
        run_dir / GENERATIONS_NAME,
        GENERATIONS_NAME,
        rows=len(generations),
    )
    execution_reference = _artifact_reference(
        run_dir / EXECUTION_NAME,
        EXECUTION_NAME,
    )
    manifest_inputs = (
        {
            "generations": generations_reference,
            "execution": execution_reference,
            "raw_scores": None,
            "raw_scoring_manifest": None,
        }
        if stage == "raw"
        else {
            "generations": generations_reference,
            "execution": execution_reference,
            "raw_generations": raw_generation_reference,
            "raw_execution": raw_execution_reference,
        }
    )
    manifest_artifacts: dict[str, Any] = {
        "scores": _payload_reference(
            SCORES_NAME,
            scores_payload,
            rows=len(scores),
        ),
        "summary": _payload_reference(SUMMARY_NAME, summary_payload),
    }
    if stage == "synthesis":
        manifest_artifacts["paired_scores"] = _payload_reference(
            PAIRED_SCORES_NAME,
            paired_scores_payload,
            rows=len(paired_scores),
        )
    scoring_manifest: dict[str, Any] = {
        "schema_version": SCORING_MANIFEST_SCHEMA_VERSION,
        "stage": stage,
        "plan_fingerprint": manifest["plan_fingerprint"],
        "config": config.to_dict(),
        "labels": safe_labels_reference,
        "dataset_lineage": safe_dataset_lineage,
        "solution_retained": False,
        "inputs": manifest_inputs,
        "artifacts": manifest_artifacts,
        "reportable": summary["reportable"],
        "non_reportable_reasons": summary["non_reportable_reasons"],
    }
    scoring_manifest["scoring_fingerprint"] = _scoring_manifest_fingerprint(
        scoring_manifest
    )
    scoring_manifest_payload = canonical_json_bytes(scoring_manifest)
    payloads = [
        (run_dir / SCORES_NAME, scores_payload),
        (run_dir / SUMMARY_NAME, summary_payload),
    ]
    if stage == "synthesis":
        payloads.append((run_dir / PAIRED_SCORES_NAME, paired_scores_payload))
    payloads.append((run_dir / SCORING_MANIFEST_NAME, scoring_manifest_payload))
    _preflight_payloads(payloads, force=force)
    for path, payload in payloads:
        publish_bytes(path, payload, force=force)
    return summary


def score_run(
    run_dir: Path,
    config: ScoringConfig,
    *,
    repository_root: Path,
    raw_run_dir: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Load locked labels and score a complete run at one provenance boundary."""

    manifest, requests = load_plan(run_dir)
    # Cross the labels boundary only after model-facing generation is complete
    # and its execution provenance has been verified without reference access.
    verify_complete_execution(run_dir, manifest, requests)
    scoring_data = _load_locked_scoring_data(repository_root, config)
    return _score_run_with_labels(
        run_dir,
        scoring_data.labels,
        scoring_data.labels_reference,
        config,
        dataset_lineage=scoring_data.dataset_lineage,
        raw_run_dir=raw_run_dir,
        force=force,
    )


def score_test_fixture_run(
    run_dir: Path,
    labels: Sequence[EvalLabel],
    labels_reference: Mapping[str, Any],
    config: ScoringConfig,
    *,
    raw_run_dir: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Exercise scoring with synthetic labels, permanently non-reportably."""

    return _score_run_with_labels(
        run_dir,
        labels,
        labels_reference,
        config,
        raw_run_dir=raw_run_dir,
        label_non_reportable_reasons=("labels_are_test_fixture",),
        force=force,
    )
