"""Strict verification for complete generation executions.

This module is deliberately independent of planning and backend implementations so
both synthesis planning and scoring apply the same provenance checks.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import (
    canonical_json_bytes,
    file_digest,
    read_json,
    read_jsonl,
    sha256_bytes,
)
from .errors import EvaluationError
from .schemas import (
    GenerationRequest,
    GenerationResult,
    ModelSpec,
    validate_and_order_generations,
)


EXECUTION_NAME = "execution.json"
GENERATIONS_NAME = "generations.jsonl"
RESULTS_DIRECTORY = "results"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMON_KEYS = {
    "schema_version",
    "plan_fingerprint",
    "mode",
    "backend",
    "backend_fingerprint",
    "planned_requests",
    "completed_requests",
    "completed_results_fingerprint",
    "complete",
    "generations",
    "seed_reproducibility",
    "non_reportable",
    "non_reportable_reasons",
}
_INJECTED_KEYS = _COMMON_KEYS
_EXTERNAL_KEYS = _COMMON_KEYS | {"response_source"}
_BACKEND_KEYS = {
    "name",
    "version",
    "model",
    "supported_sampling_fields",
    "non_reportable",
}


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise EvaluationError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _nonempty_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationError(f"{name} must be a nonempty string")
    return value


def _verify_reference(
    reference: Any,
    path: Path,
    *,
    logical_path: str,
    rows: int | None = None,
) -> None:
    if not isinstance(reference, dict):
        raise EvaluationError(f"Missing artifact reference for {logical_path}")
    expected_keys = {"path", "sha256", "bytes"}
    if rows is not None:
        expected_keys.add("rows")
    if set(reference) != expected_keys:
        raise EvaluationError(f"Malformed artifact reference for {logical_path}")
    try:
        digest, byte_count = file_digest(path)
    except OSError as exc:
        raise EvaluationError(f"Cannot read execution artifact {path}: {exc}") from exc
    if (
        reference.get("path") != logical_path
        or reference.get("sha256") != digest
        or reference.get("bytes") != byte_count
        or (rows is not None and reference.get("rows") != rows)
    ):
        raise EvaluationError(f"Artifact reference no longer matches {logical_path}")


def _result_path(run_dir: Path, task_id: str) -> Path:
    if not task_id.replace("-", "").isalnum():
        raise EvaluationError(f"Unsafe task ID for result artifact: {task_id}")
    return run_dir / RESULTS_DIRECTORY / f"{task_id}.json"


def _verify_results(
    run_dir: Path,
    requests: Sequence[GenerationRequest],
    backend_fingerprint: str,
) -> list[GenerationResult]:
    results: list[GenerationResult] = []
    for request in requests:
        result = GenerationResult.from_dict(
            read_json(_result_path(run_dir, request.task_id))
        )
        if (
            result.task_id != request.task_id
            or result.request_fingerprint != request.request_fingerprint
            or result.backend_fingerprint != backend_fingerprint
        ):
            raise EvaluationError(
                f"Execution result does not match its request: {request.task_id}"
            )
        results.append(result)
    return results


def _verify_generation_results(
    generations: Sequence[Mapping[str, Any]],
    results: Sequence[GenerationResult],
) -> None:
    if len(generations) != len(results):
        raise EvaluationError("Generation and result counts differ")
    for generation, result in zip(generations, results, strict=True):
        expected = {
            "task_id": result.task_id,
            "request_fingerprint": result.request_fingerprint,
            "backend_fingerprint": result.backend_fingerprint,
            "text": result.text,
            "output_sha256": result.output_sha256,
            "finish_reason": result.finish_reason,
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
            },
            "provider_metadata": dict(result.provider_metadata),
        }
        if any(generation.get(key) != value for key, value in expected.items()):
            raise EvaluationError(
                f"Materialized generation differs from result: {result.task_id}"
            )


def _verify_backend(
    value: Any,
    requests: Sequence[GenerationRequest],
    backend_fingerprint: str,
) -> tuple[bool, str]:
    if not isinstance(value, dict) or set(value) != _BACKEND_KEYS:
        raise EvaluationError("Injected execution has an invalid backend descriptor")
    name = _nonempty_text(value.get("name"), "backend.name")
    _nonempty_text(value.get("version"), "backend.version")
    non_reportable = value.get("non_reportable")
    if not isinstance(non_reportable, bool):
        raise EvaluationError("backend.non_reportable must be a boolean")
    supported = value.get("supported_sampling_fields")
    if (
        not isinstance(supported, list)
        or not all(isinstance(item, str) and item for item in supported)
        or len(set(supported)) != len(supported)
        or supported != sorted(supported)
    ):
        raise EvaluationError("Backend sampling-field declaration is malformed")
    model_value = value.get("model")
    if not isinstance(model_value, dict):
        raise EvaluationError("Backend model descriptor is malformed")
    model = ModelSpec.from_dict(model_value)
    if name != model.provider:
        raise EvaluationError("Backend name does not match its model provider")
    if any(request.model != model for request in requests):
        raise EvaluationError("Backend model does not exactly match every request")
    required_fields = set().union(*(set(request.sampling) for request in requests))
    if not required_fields.issubset(set(supported)):
        raise EvaluationError("Backend omits a planned sampling field")
    if sha256_bytes(canonical_json_bytes(value)) != backend_fingerprint:
        raise EvaluationError("Backend descriptor fingerprint mismatch")
    return non_reportable, model.seed_support


def _expected_reasons(
    manifest: Mapping[str, Any],
    *,
    mode: str,
    backend_non_reportable: bool,
) -> list[str]:
    if mode == "external_response_ingest":
        return ["unattested_external_response_ingest"]
    reasons: list[str] = []
    if backend_non_reportable:
        reasons.append("backend_declared_non_reportable")
    label_firewall = manifest.get("label_firewall")
    if not isinstance(label_firewall, dict):
        raise EvaluationError("Plan label-firewall provenance is malformed")
    if label_firewall.get("locked_questions_verified") is not True:
        reasons.append("question_input_is_a_test_fixture")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        raise EvaluationError("Plan input provenance is malformed")
    if (
        manifest.get("kind") == "synthesis"
        and inputs.get("raw_execution_non_reportable") is True
    ):
        reasons.append("raw_dependency_is_non_reportable")
    return reasons


def verify_complete_execution(
    run_dir: Path,
    manifest: Mapping[str, Any],
    requests: Sequence[GenerationRequest],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return a complete execution and generations after semantic verification."""

    execution_path = run_dir / EXECUTION_NAME
    if not execution_path.is_file():
        raise EvaluationError(f"Missing complete execution provenance: {execution_path}")
    execution = read_json(execution_path)
    mode = execution.get("mode")
    expected_keys = (
        _INJECTED_KEYS if mode == "injected_backend" else _EXTERNAL_KEYS
        if mode == "external_response_ingest"
        else None
    )
    if expected_keys is None or set(execution) != expected_keys:
        raise EvaluationError("Execution has an unsupported mode or schema")
    if execution.get("schema_version") != 1:
        raise EvaluationError("Unsupported execution schema version")
    if execution.get("plan_fingerprint") != manifest.get("plan_fingerprint"):
        raise EvaluationError("Execution provenance does not match the plan")
    request_count = len(requests)
    if (
        execution.get("complete") is not True
        or execution.get("planned_requests") != request_count
        or execution.get("completed_requests") != request_count
        or isinstance(execution.get("planned_requests"), bool)
        or isinstance(execution.get("completed_requests"), bool)
    ):
        raise EvaluationError("Execution is incomplete or has invalid request counts")
    backend_fingerprint = _sha256(
        execution.get("backend_fingerprint"), "execution.backend_fingerprint"
    )
    completed_fingerprint = _sha256(
        execution.get("completed_results_fingerprint"),
        "execution.completed_results_fingerprint",
    )

    generations_path = run_dir / GENERATIONS_NAME
    _verify_reference(
        execution.get("generations"),
        generations_path,
        logical_path=GENERATIONS_NAME,
        rows=request_count,
    )
    generations = validate_and_order_generations(
        read_jsonl(generations_path),
        requests,
    )
    if any(
        generation["backend_fingerprint"] != backend_fingerprint
        for generation in generations
    ):
        raise EvaluationError("Generation backend fingerprints do not match execution")
    results = _verify_results(run_dir, requests, backend_fingerprint)
    if sha256_bytes(
        canonical_json_bytes([result.to_dict() for result in results])
    ) != completed_fingerprint:
        raise EvaluationError("Completed result-set fingerprint mismatch")
    _verify_generation_results(generations, results)

    if mode == "injected_backend":
        backend_non_reportable, seed_support = _verify_backend(
            execution.get("backend"),
            requests,
            backend_fingerprint,
        )
    else:
        if execution.get("backend") is not None:
            raise EvaluationError("External ingestion cannot attest a backend descriptor")
        backend_non_reportable = True
        seed_support = "unattested"
        response_source = execution.get("response_source")
        if not isinstance(response_source, dict) or set(response_source) != {
            "path",
            "sha256",
            "bytes",
            "rows",
        }:
            raise EvaluationError("External response-source provenance is malformed")
        _nonempty_text(response_source.get("path"), "response_source.path")
        _sha256(response_source.get("sha256"), "response_source.sha256")
        if (
            isinstance(response_source.get("bytes"), bool)
            or not isinstance(response_source.get("bytes"), int)
            or response_source["bytes"] < 0
            or isinstance(response_source.get("rows"), bool)
            or response_source.get("rows") != request_count
        ):
            raise EvaluationError("External response-source counts are malformed")
    if execution.get("seed_reproducibility") != seed_support:
        raise EvaluationError("Execution seed-support declaration is inconsistent")

    expected_reasons = _expected_reasons(
        manifest,
        mode=mode,
        backend_non_reportable=backend_non_reportable,
    )
    reasons = execution.get("non_reportable_reasons")
    if reasons != expected_reasons or execution.get("non_reportable") is not bool(
        expected_reasons
    ):
        raise EvaluationError("Execution reportability does not follow its provenance")
    return execution, generations
