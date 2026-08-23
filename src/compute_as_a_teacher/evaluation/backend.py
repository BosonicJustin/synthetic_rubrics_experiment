"""Backend protocol and resumable execution helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .artifacts import (
    artifact_reference,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    publish_bytes,
    publish_json,
    read_json,
    read_jsonl,
    sha256_bytes,
    sha256_text,
)
from .errors import EvaluationError
from .planning import EXECUTION_NAME, GENERATIONS_NAME, load_plan
from .schemas import (
    BackendOutput,
    GenerationRequest,
    GenerationResult,
    ModelSpec,
    validate_and_order_generations,
)


RESULTS_DIRECTORY = "results"
RESPONSES_NAME = "responses.jsonl"


@dataclass(frozen=True, slots=True)
class BackendDescriptor:
    name: str
    version: str
    model: ModelSpec
    supported_sampling_fields: frozenset[str]
    non_reportable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "model": self.model.to_dict(),
            "supported_sampling_fields": sorted(self.supported_sampling_fields),
            "non_reportable": self.non_reportable,
        }

    @property
    def fingerprint(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_dict()))


class GenerationBackend(Protocol):
    @property
    def descriptor(self) -> BackendDescriptor:
        ...

    def generate_batch(
        self,
        requests: Sequence[GenerationRequest],
    ) -> Sequence[BackendOutput]:
        ...


def _result_path(run_dir: Path, task_id: str) -> Path:
    if not task_id.replace("-", "").isalnum():
        raise EvaluationError(f"Unsafe task ID for artifact filename: {task_id}")
    return run_dir / RESULTS_DIRECTORY / f"{task_id}.json"


def _load_result_for_request(
    run_dir: Path,
    request: GenerationRequest,
    *,
    expected_backend_fingerprint: str | None = None,
) -> GenerationResult | None:
    path = _result_path(run_dir, request.task_id)
    if not path.exists():
        return None
    result = GenerationResult.from_dict(read_json(path))
    if (
        result.task_id != request.task_id
        or result.request_fingerprint != request.request_fingerprint
    ):
        raise EvaluationError(f"Cached result does not match request: {path}")
    if (
        expected_backend_fingerprint is not None
        and result.backend_fingerprint != expected_backend_fingerprint
    ):
        raise EvaluationError(
            f"Cached result came from a different backend execution: {path}"
        )
    return result


def _validate_backend(
    descriptor: BackendDescriptor,
    requests: Sequence[GenerationRequest],
) -> None:
    if not requests:
        raise EvaluationError("Cannot execute an empty plan")
    if not descriptor.name.strip() or not descriptor.version.strip():
        raise EvaluationError("Backend name and version must be nonempty")
    descriptor.model.assert_resolved()
    for request in requests:
        # Reparse just before dispatch so shallow mutation cannot bypass identity.
        GenerationRequest.from_dict(request.to_dict())
        if request.model != descriptor.model:
            raise EvaluationError("Backend model spec does not exactly match the plan")
        missing_fields = set(request.sampling) - set(
            descriptor.supported_sampling_fields
        )
        if missing_fields:
            raise EvaluationError(
                "Backend does not support required sampling fields: "
                f"{sorted(missing_fields)}"
            )
    if descriptor.name != descriptor.model.provider:
        raise EvaluationError("Backend name must match the planned model provider")


def _wrap_output(
    request: GenerationRequest,
    output: BackendOutput,
    backend_fingerprint: str,
) -> GenerationResult:
    if output.task_id != request.task_id:
        raise EvaluationError(
            f"Backend output task ID mismatch: expected {request.task_id}, "
            f"found {output.task_id}"
        )
    if output.request_fingerprint != request.request_fingerprint:
        raise EvaluationError(f"Backend output fingerprint mismatch: {request.task_id}")
    if not isinstance(output.text, str):
        raise EvaluationError("Backend output text must be a string")
    if not isinstance(output.finish_reason, str) or not output.finish_reason:
        raise EvaluationError("Backend finish_reason must be a nonempty string")
    result = GenerationResult(
        task_id=request.task_id,
        request_fingerprint=request.request_fingerprint,
        backend_fingerprint=backend_fingerprint,
        text=output.text,
        output_sha256=sha256_text(output.text),
        finish_reason=output.finish_reason,
        prompt_tokens=output.prompt_tokens,
        completion_tokens=output.completion_tokens,
        provider_metadata=dict(output.provider_metadata or {}),
    )
    return GenerationResult.from_dict(result.to_dict())


def _validated_batch_results(
    batch: Sequence[GenerationRequest],
    outputs: Sequence[BackendOutput],
    backend_fingerprint: str,
) -> list[GenerationResult]:
    output_by_id: dict[str, BackendOutput] = {}
    for output in outputs:
        if not isinstance(output, BackendOutput):
            raise EvaluationError("Backend returned an invalid output object")
        if output.task_id in output_by_id:
            raise EvaluationError(f"Backend returned duplicate task ID: {output.task_id}")
        output_by_id[output.task_id] = output
    expected_ids = {request.task_id for request in batch}
    if set(output_by_id) != expected_ids:
        raise EvaluationError(
            "Backend batch coverage mismatch: "
            f"missing={sorted(expected_ids - set(output_by_id))}, "
            f"extra={sorted(set(output_by_id) - expected_ids)}"
        )
    results: list[GenerationResult] = []
    for request in batch:
        # Also revalidate after the adapter returns: GenerationRequest is frozen,
        # but its nested dictionaries are intentionally JSON-shaped and mutable.
        GenerationRequest.from_dict(request.to_dict())
        results.append(
            _wrap_output(
                request,
                output_by_id[request.task_id],
                backend_fingerprint,
            )
        )
    return results


def _generation_row(
    request: GenerationRequest,
    result: GenerationResult,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stage": request.stage,
        "task_id": request.task_id,
        "request_fingerprint": request.request_fingerprint,
        "backend_fingerprint": result.backend_fingerprint,
        "question_id": request.question_id,
        "rollout_index": request.rollout_index,
        "source_task_ids": list(request.source_task_ids),
        "seed": request.sampling["seed"],
        "text": result.text,
        "output_sha256": result.output_sha256,
        "finish_reason": result.finish_reason,
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
        },
        "provider_metadata": dict(result.provider_metadata),
    }


def _materialized_rows(
    run_dir: Path,
    requests: Sequence[GenerationRequest],
    *,
    expected_backend_fingerprint: str | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for request in requests:
        result = _load_result_for_request(
            run_dir,
            request,
            expected_backend_fingerprint=expected_backend_fingerprint,
        )
        if result is None:
            raise EvaluationError(f"Missing completed result for task {request.task_id}")
        rows.append(_generation_row(request, result))
    backend_fingerprints = {row["backend_fingerprint"] for row in rows}
    if len(backend_fingerprints) != 1:
        raise EvaluationError("A run cannot mix results from different backends")
    return validate_and_order_generations(rows, requests)


def materialize_generations(
    run_dir: Path,
    *,
    expected_backend_fingerprint: str | None = None,
    force: bool = False,
) -> list[dict[str, Any]]:
    _, requests = load_plan(run_dir)
    rows = _materialized_rows(
        run_dir,
        requests,
        expected_backend_fingerprint=expected_backend_fingerprint,
    )
    publish_bytes(
        run_dir / GENERATIONS_NAME,
        canonical_jsonl_bytes(rows),
        force=force,
    )
    return rows


def _completed_prefix_results(
    run_dir: Path,
    requests: Sequence[GenerationRequest],
    backend_fingerprint: str,
) -> list[GenerationResult]:
    """Load the contiguous completed prefix, rejecting gaps and mixed provenance."""

    completed_results: list[GenerationResult] = []
    missing_result_seen = False
    for request in requests:
        result = _load_result_for_request(
            run_dir,
            request,
            expected_backend_fingerprint=backend_fingerprint,
        )
        if result is None:
            missing_result_seen = True
            continue
        if missing_result_seen:
            raise EvaluationError(
                "Cached results are not a contiguous prefix of the planned requests"
            )
        completed_results.append(result)
    return completed_results


def _completed_results_state(
    run_dir: Path,
    requests: Sequence[GenerationRequest],
    backend_fingerprint: str,
) -> tuple[int, str]:
    completed_results = _completed_prefix_results(
        run_dir,
        requests,
        backend_fingerprint,
    )
    return (
        len(completed_results),
        sha256_bytes(
            canonical_json_bytes([result.to_dict() for result in completed_results])
        ),
    )


def _non_reportable_reasons(
    manifest: Mapping[str, Any],
    descriptor: BackendDescriptor,
) -> list[str]:
    reasons: list[str] = []
    if descriptor.non_reportable:
        reasons.append("backend_declared_non_reportable")
    if not manifest["label_firewall"].get("locked_questions_verified", False):
        reasons.append("question_input_is_a_test_fixture")
    if (
        manifest["kind"] == "synthesis"
        and manifest["inputs"].get("raw_execution_non_reportable") is True
    ):
        reasons.append("raw_dependency_is_non_reportable")
    return reasons


def _execution_record(
    manifest: Mapping[str, Any],
    descriptor: BackendDescriptor,
    requests: Sequence[GenerationRequest],
    *,
    completed: int,
    completed_results_fingerprint: str,
    generations_reference: Mapping[str, Any] | None,
) -> dict[str, Any]:
    reasons = _non_reportable_reasons(manifest, descriptor)
    return {
        "schema_version": 1,
        "plan_fingerprint": manifest["plan_fingerprint"],
        "mode": "injected_backend",
        "backend": descriptor.to_dict(),
        "backend_fingerprint": descriptor.fingerprint,
        "planned_requests": len(requests),
        "completed_requests": completed,
        "completed_results_fingerprint": completed_results_fingerprint,
        "complete": completed == len(requests),
        "generations": dict(generations_reference) if generations_reference else None,
        "seed_reproducibility": descriptor.model.seed_support,
        "non_reportable": bool(reasons),
        "non_reportable_reasons": reasons,
    }


def _existing_execution_guard(
    run_dir: Path,
    manifest: Mapping[str, Any],
    descriptor: BackendDescriptor,
    requests: Sequence[GenerationRequest],
) -> None:
    path = run_dir / EXECUTION_NAME
    if not path.exists():
        return
    execution = read_json(path)
    backend_fingerprint = descriptor.fingerprint
    if execution.get("plan_fingerprint") != manifest["plan_fingerprint"]:
        raise EvaluationError("Existing execution provenance belongs to another plan")
    if execution.get("backend_fingerprint") != backend_fingerprint:
        raise EvaluationError("Refusing to resume with a different backend fingerprint")
    if execution.get("mode") != "injected_backend":
        raise EvaluationError("Refusing to mix external ingest and backend execution")
    previous_completed = execution.get("completed_requests")
    if (
        isinstance(previous_completed, bool)
        or not isinstance(previous_completed, int)
        or not 0 <= previous_completed <= len(requests)
    ):
        raise EvaluationError("Existing execution has an invalid completed-request count")

    completed_results = _completed_prefix_results(
        run_dir,
        requests,
        backend_fingerprint,
    )
    if len(completed_results) < previous_completed:
        raise EvaluationError("A checkpointed cached result is missing")
    previous_results_fingerprint = sha256_bytes(
        canonical_json_bytes(
            [result.to_dict() for result in completed_results[:previous_completed]]
        )
    )
    if execution.get("completed_results_fingerprint") != previous_results_fingerprint:
        raise EvaluationError("A checkpointed cached result changed")

    generations_reference: dict[str, Any] | None = None
    if previous_completed == len(requests):
        generations_path = run_dir / GENERATIONS_NAME
        if not generations_path.is_file():
            raise EvaluationError("Completed generation artifact changed before resume")
        generations_reference = artifact_reference(
            generations_path,
            rows=len(requests),
        )
        generations_reference["path"] = GENERATIONS_NAME

    expected_execution = _execution_record(
        manifest,
        descriptor,
        requests,
        completed=previous_completed,
        completed_results_fingerprint=previous_results_fingerprint,
        generations_reference=generations_reference,
    )
    if execution != expected_execution:
        raise EvaluationError("Existing execution checkpoint metadata changed")


def _checkpoint_execution(
    run_dir: Path,
    manifest: Mapping[str, Any],
    descriptor: BackendDescriptor,
    requests: Sequence[GenerationRequest],
    *,
    force: bool = True,
) -> dict[str, Any]:
    completed, completed_results_fingerprint = _completed_results_state(
        run_dir,
        requests,
        descriptor.fingerprint,
    )
    generations_reference: dict[str, Any] | None = None
    if completed == len(requests):
        rows = materialize_generations(
            run_dir,
            expected_backend_fingerprint=descriptor.fingerprint,
            force=True,
        )
        generations_reference = artifact_reference(
            run_dir / GENERATIONS_NAME,
            rows=len(rows),
        )
        generations_reference["path"] = GENERATIONS_NAME
    execution = _execution_record(
        manifest,
        descriptor,
        requests,
        completed=completed,
        completed_results_fingerprint=completed_results_fingerprint,
        generations_reference=generations_reference,
    )
    publish_json(run_dir / EXECUTION_NAME, execution, force=force)
    return execution


def execute_plan(
    run_dir: Path,
    backend: GenerationBackend,
    *,
    batch_size: int = 16,
    max_requests: int | None = None,
) -> dict[str, Any]:
    """Execute or resume a planned run with an identity-matched backend."""

    if batch_size <= 0:
        raise EvaluationError("batch_size must be positive")
    manifest, requests = load_plan(run_dir)
    descriptor = backend.descriptor
    _validate_backend(descriptor, requests)
    backend_fingerprint = descriptor.fingerprint
    _existing_execution_guard(
        run_dir,
        manifest,
        descriptor,
        requests,
    )
    execution = _checkpoint_execution(run_dir, manifest, descriptor, requests)

    pending = [
        request
        for request in requests
        if _load_result_for_request(
            run_dir,
            request,
            expected_backend_fingerprint=backend_fingerprint,
        )
        is None
    ]
    if max_requests is not None:
        if max_requests < 0:
            raise EvaluationError("max_requests must be nonnegative")
        pending = pending[:max_requests]

    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        outputs = list(backend.generate_batch(batch))
        results = _validated_batch_results(batch, outputs, backend_fingerprint)
        for request, result in zip(batch, results, strict=True):
            publish_json(_result_path(run_dir, request.task_id), result.to_dict())
        execution = _checkpoint_execution(run_dir, manifest, descriptor, requests)
    return execution


def _preflight_payload(path: Path, payload: bytes, force: bool) -> None:
    if path.exists() and (not path.is_file() or path.read_bytes() != payload) and not force:
        raise EvaluationError(f"Refusing to overwrite mismatched artifact {path}")


def ingest_responses(
    run_dir: Path,
    responses_path: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Ingest complete external responses as explicitly non-reportable artifacts."""

    manifest, requests = load_plan(run_dir)
    responses = [GenerationResult.from_dict(row) for row in read_jsonl(responses_path)]
    response_by_id: dict[str, GenerationResult] = {}
    for response in responses:
        if response.task_id in response_by_id:
            raise EvaluationError(f"Duplicate response task ID: {response.task_id}")
        response_by_id[response.task_id] = response

    expected_ids = {request.task_id for request in requests}
    actual_ids = set(response_by_id)
    if actual_ids != expected_ids:
        raise EvaluationError(
            f"Response coverage mismatch: missing={sorted(expected_ids - actual_ids)}, "
            f"extra={sorted(actual_ids - expected_ids)}"
        )
    backend_fingerprints = {
        response.backend_fingerprint for response in responses
    }
    if len(backend_fingerprints) != 1:
        raise EvaluationError("External responses must share one backend fingerprint")
    backend_fingerprint = next(iter(backend_fingerprints))

    ordered_results: list[GenerationResult] = []
    for request in requests:
        response = response_by_id[request.task_id]
        if response.request_fingerprint != request.request_fingerprint:
            raise EvaluationError(f"Response fingerprint mismatch: {request.task_id}")
        ordered_results.append(response)
    generations = validate_and_order_generations(
        [
            _generation_row(request, result)
            for request, result in zip(requests, ordered_results, strict=True)
        ],
        requests,
    )

    result_payloads = [
        (
            _result_path(run_dir, result.task_id),
            canonical_json_bytes(result.to_dict()),
        )
        for result in ordered_results
    ]
    responses_payload = canonical_jsonl_bytes(
        result.to_dict() for result in ordered_results
    )
    generations_payload = canonical_jsonl_bytes(generations)
    source_reference = artifact_reference(responses_path, rows=len(responses))
    execution = {
        "schema_version": 1,
        "plan_fingerprint": manifest["plan_fingerprint"],
        "mode": "external_response_ingest",
        "response_source": source_reference,
        "backend": None,
        "backend_fingerprint": backend_fingerprint,
        "planned_requests": len(requests),
        "completed_requests": len(generations),
        "completed_results_fingerprint": sha256_bytes(
            canonical_json_bytes([result.to_dict() for result in ordered_results])
        ),
        "complete": True,
        "generations": {
            "path": GENERATIONS_NAME,
            "sha256": sha256_bytes(generations_payload),
            "bytes": len(generations_payload),
            "rows": len(generations),
        },
        "seed_reproducibility": "unattested",
        "non_reportable": True,
        "non_reportable_reasons": ["unattested_external_response_ingest"],
    }
    execution_payload = canonical_json_bytes(execution)
    all_payloads = [
        *result_payloads,
        (run_dir / RESPONSES_NAME, responses_payload),
        (run_dir / GENERATIONS_NAME, generations_payload),
        (run_dir / EXECUTION_NAME, execution_payload),
    ]
    for path, payload in all_payloads:
        _preflight_payload(path, payload, force)
    for path, payload in all_payloads:
        publish_bytes(path, payload, force=force)
    return execution
