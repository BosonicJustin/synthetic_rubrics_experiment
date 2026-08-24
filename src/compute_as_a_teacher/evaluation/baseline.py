"""Resumable, label-free baseline generation sequence."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from compute_as_a_teacher.data.math500 import load_locked_questions
from compute_as_a_teacher.openai_chat import SUPPORTED_FINISH_REASONS

from .artifacts import read_json
from .backend import RESULTS_DIRECTORY, GenerationBackend, execute_plan
from .config import load_raw_config, load_synthesis_config
from .errors import EvaluationError
from .execution import verify_complete_execution
from .grading import extract_last_boxed
from .openai_backend import OpenAICompatibleBackend
from .planning import (
    load_plan,
    validate_synthesis_anchor_relation,
    write_raw_plan,
    write_synthesis_plan,
)
from .prompts import load_prompt
from .schemas import GenerationResult


_LABEL_DERIVED_ARTIFACTS = (
    "scores.jsonl",
    "paired_scores.jsonl",
    "summary.json",
    "scoring_manifest.json",
    "final-experiment.json",
)


def _resolved(root: Path, path: Path | str) -> Path:
    value = Path(path)
    return (value if value.is_absolute() else root / value).resolve()


def _require_label_boundary_closed(*run_dirs: Path) -> None:
    found = [
        str(run_dir / name)
        for run_dir in run_dirs
        for name in _LABEL_DERIVED_ARTIFACTS
        if os.path.lexists(run_dir / name)
    ]
    if found:
        raise EvaluationError(
            "Baseline generation requires label-derived artifacts to be absent: "
            f"{found}"
        )


def _validate_canary_count(results: int, available: int) -> None:
    if type(results) is not int or results <= 0:
        raise EvaluationError("canary result count must be a positive integer")
    if results > available:
        raise EvaluationError("canary result count exceeds the planned request count")


def _audit_canary(
    run_dir: Path,
    backend: GenerationBackend,
    *,
    expected_kind: str,
    results: int,
) -> dict[str, Any]:
    """Audit the completed request prefix without consulting reference labels."""

    manifest, requests = load_plan(run_dir, expected_kind=expected_kind)
    _validate_canary_count(results, len(requests))

    execution = execute_plan(run_dir, backend, max_requests=0)
    descriptor = backend.descriptor
    completed = execution.get("completed_requests")
    if (
        execution.get("mode") != "injected_backend"
        or execution.get("plan_fingerprint") != manifest["plan_fingerprint"]
        or execution.get("backend_fingerprint") != descriptor.fingerprint
        or execution.get("backend") != descriptor.to_dict()
        or type(completed) is not int
        or completed < results
    ):
        raise EvaluationError("Canary execution provenance is incomplete or mismatched")

    finish_reasons: dict[str, int] = {}
    extraction_statuses: dict[str, int] = {}
    boxed_outputs = 0
    for request in requests[:results]:
        result = GenerationResult.from_dict(
            read_json(run_dir / RESULTS_DIRECTORY / f"{request.task_id}.json")
        )
        if (
            result.task_id != request.task_id
            or result.request_fingerprint != request.request_fingerprint
            or result.backend_fingerprint != descriptor.fingerprint
        ):
            raise EvaluationError(
                f"Canary result does not match its request: {request.task_id}"
            )
        if result.provider_metadata.get("response_model") != request.model.model_id:
            raise EvaluationError(
                f"Canary response model is not exact: {request.task_id}"
            )
        if result.finish_reason not in SUPPORTED_FINISH_REASONS:
            raise EvaluationError(
                f"Canary finish reason is unsupported: {request.task_id}"
            )
        extraction = extract_last_boxed(result.text)
        extraction_statuses[extraction.status] = (
            extraction_statuses.get(extraction.status, 0) + 1
        )
        boxed_outputs += int(extraction.status == "ok")
        finish_reasons[result.finish_reason] = (
            finish_reasons.get(result.finish_reason, 0) + 1
        )

    return {
        "audited_results": results,
        "response_model": requests[0].model.model_id,
        "finish_reasons": dict(sorted(finish_reasons.items())),
        "extraction_status_counts": dict(sorted(extraction_statuses.items())),
        "boxed_outputs": boxed_outputs,
    }


def _run_canary(
    run_dir: Path,
    backend: GenerationBackend,
    *,
    expected_kind: str,
    results: int,
    batch_size: int,
) -> dict[str, Any]:
    _, requests = load_plan(run_dir, expected_kind=expected_kind)
    _validate_canary_count(results, len(requests))
    checkpoint = execute_plan(run_dir, backend, batch_size=batch_size, max_requests=0)
    completed = checkpoint["completed_requests"]
    if completed < results:
        execute_plan(
            run_dir,
            backend,
            batch_size=batch_size,
            max_requests=results - completed,
        )
    return _audit_canary(
        run_dir,
        backend,
        expected_kind=expected_kind,
        results=results,
    )


def run_baseline_sequence(
    *,
    repository_root: Path,
    raw_config_path: Path,
    synthesis_config_path: Path,
    raw_run_dir: Path,
    synthesis_run_dir: Path,
    base_url: str,
    pilot: bool = False,
    preregistration_path: Path | None = None,
    training_run_dir: Path | None = None,
    api_key_env: str = "",
    workers: int = 8,
    batch_size: int = 16,
    canary_results: int = 16,
    synthesis_canary_results: int = 16,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    """Plan, canary, resume, and complete raw then synthesis generation."""

    repository_root = repository_root.resolve()
    raw_config_path = _resolved(repository_root, raw_config_path)
    synthesis_config_path = _resolved(repository_root, synthesis_config_path)
    raw_run_dir = _resolved(repository_root, raw_run_dir)
    synthesis_run_dir = _resolved(repository_root, synthesis_run_dir)
    if not isinstance(pilot, bool):
        raise EvaluationError("pilot must be a boolean")
    if pilot:
        if preregistration_path is not None or training_run_dir is not None:
            raise EvaluationError(
                "Pilot mode cannot accept canonical preregistration inputs"
            )
    elif preregistration_path is None or training_run_dir is None:
        raise EvaluationError(
            "Choose --pilot or provide both canonical preregistration inputs"
        )
    if raw_run_dir == synthesis_run_dir:
        raise EvaluationError("raw and synthesis run directories must be distinct")
    if type(workers) is not int or workers <= 0:
        raise EvaluationError("workers must be a positive integer")
    if type(batch_size) is not int or batch_size <= 0:
        raise EvaluationError("batch size must be a positive integer")
    _require_label_boundary_closed(raw_run_dir, synthesis_run_dir)

    raw_config = load_raw_config(raw_config_path)
    synthesis_config = load_synthesis_config(synthesis_config_path)
    questions_path = _resolved(repository_root, raw_config.questions_path)
    questions = load_locked_questions(
        questions_path,
        lock_path=_resolved(repository_root, raw_config.dataset_lock_path),
    )
    raw_template = load_prompt(repository_root, raw_config.prompt)
    synthesis_template = load_prompt(repository_root, synthesis_config.prompt)
    _validate_canary_count(canary_results, len(questions) * 8)
    _validate_canary_count(synthesis_canary_results, len(questions))
    write_raw_plan(
        raw_run_dir,
        questions,
        raw_config,
        raw_template,
        questions_path,
        repository_root=repository_root,
    )
    raw_manifest, raw_requests = load_plan(raw_run_dir, expected_kind="raw")
    if synthesis_config.anchor_relation != "same_as_raw":
        raise EvaluationError(
            "Baseline synthesis must use the same frozen initial policy"
        )
    validate_synthesis_anchor_relation(synthesis_config, raw_requests)

    if not pilot:
        preregistration_path = _resolved(repository_root, preregistration_path)
        training_run_dir = _resolved(repository_root, training_run_dir)
        try:
            from compute_as_a_teacher.training.errors import TrainingError
            from compute_as_a_teacher.training.experiment_registry import (
                verify_preregistered_training_stage,
            )

            preregistration = verify_preregistered_training_stage(
                preregistration_path,
                training_run_dir,
            )
        except TrainingError as exc:
            raise EvaluationError(
                f"Experiment preregistration is invalid: {exc}"
            ) from exc
        stages = preregistration.get("stages")
        registered_raw = (
            stages.get("initial_raw") if isinstance(stages, dict) else None
        )
        registered_synthesis = (
            stages.get("initial_synthesis_config")
            if isinstance(stages, dict)
            else None
        )
        if (
            not isinstance(registered_raw, dict)
            or registered_raw.get("run_dir") != str(raw_run_dir)
            or not isinstance(registered_synthesis, dict)
            or registered_synthesis.get("path") != str(synthesis_config_path)
        ):
            raise EvaluationError(
                "Baseline paths do not match the experiment preregistration"
            )

    raw_backend = OpenAICompatibleBackend.from_environment(
        model=raw_requests[0].model,
        base_url=base_url,
        api_key_env=api_key_env,
        timeout_seconds=timeout_seconds,
        max_workers=workers,
    )
    raw_canary = _run_canary(
        raw_run_dir,
        raw_backend,
        expected_kind="raw",
        results=canary_results,
        batch_size=batch_size,
    )
    raw_execution = execute_plan(raw_run_dir, raw_backend, batch_size=batch_size)
    verified_raw, _ = verify_complete_execution(
        raw_run_dir,
        raw_manifest,
        raw_requests,
    )

    _require_label_boundary_closed(raw_run_dir, synthesis_run_dir)
    write_synthesis_plan(
        synthesis_run_dir,
        raw_run_dir,
        synthesis_config,
        synthesis_template,
    )
    synthesis_manifest, synthesis_requests = load_plan(
        synthesis_run_dir,
        expected_kind="synthesis",
    )
    synthesis_canary = _run_canary(
        synthesis_run_dir,
        raw_backend,
        expected_kind="synthesis",
        results=synthesis_canary_results,
        batch_size=batch_size,
    )
    synthesis_execution = execute_plan(
        synthesis_run_dir,
        raw_backend,
        batch_size=batch_size,
    )
    verified_synthesis, _ = verify_complete_execution(
        synthesis_run_dir,
        synthesis_manifest,
        synthesis_requests,
    )
    _require_label_boundary_closed(raw_run_dir, synthesis_run_dir)
    non_reportable_reasons = sorted(
        {
            *verified_raw["non_reportable_reasons"],
            *verified_synthesis["non_reportable_reasons"],
            *(["pilot_without_preregistration"] if pilot else []),
        }
    )

    return {
        "mode": "baseline_sequence_complete",
        "registration_mode": "pilot" if pilot else "canonical_preregistered",
        "pilot": pilot,
        "preregistration_verified": not pilot,
        "reportable": not non_reportable_reasons,
        "non_reportable_reasons": non_reportable_reasons,
        "raw": {
            "run_dir": str(raw_run_dir),
            "plan_fingerprint": raw_manifest["plan_fingerprint"],
            "canary": raw_canary,
            "completed_requests": raw_execution["completed_requests"],
            "reportable": not verified_raw["non_reportable"],
            "non_reportable_reasons": verified_raw["non_reportable_reasons"],
        },
        "synthesis": {
            "run_dir": str(synthesis_run_dir),
            "plan_fingerprint": synthesis_manifest["plan_fingerprint"],
            "canary": synthesis_canary,
            "completed_requests": synthesis_execution["completed_requests"],
            "reportable": not verified_synthesis["non_reportable"],
            "non_reportable_reasons": verified_synthesis[
                "non_reportable_reasons"
            ],
        },
        "labels_loaded": False,
        "scored": False,
    }


__all__ = ["run_baseline_sequence"]
