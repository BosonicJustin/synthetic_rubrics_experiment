"""Provider-neutral, integrity-checked request and response schemas."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from compute_as_a_teacher.openai_chat import SUPPORTED_FINISH_REASONS

from .artifacts import canonical_json_bytes, sha256_bytes, sha256_text
from .errors import EvaluationError


SCHEMA_VERSION = 1
Stage = Literal["raw", "synthesis"]
SeedSupport = Literal["strict", "best_effort", "none"]

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_UNRESOLVED_EXACT = {
    "dev",
    "development",
    "latest",
    "main",
    "master",
    "required_immutable_revision",
    "todo",
    "unknown",
    "unversioned",
}
_UNRESOLVED_PREFIXES = ("required_", "replace_", "todo_", "<")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise EvaluationError(f"{name} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise EvaluationError(
            f"{name} keys are {sorted(value)}, expected exactly {sorted(expected)}"
        )


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationError(f"{name} must be a nonempty string")
    return value


def _finish_reason(value: Any, name: str) -> str:
    if value not in SUPPORTED_FINISH_REASONS:
        raise EvaluationError(f"{name} is unsupported")
    return value


def _sha256(value: Any, name: str) -> str:
    value = _nonempty_string(value, name)
    if not _SHA256_RE.fullmatch(value):
        raise EvaluationError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _nonnegative_count(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvaluationError(f"{name} must be null or a nonnegative integer")
    return value


def _looks_unresolved(value: str) -> bool:
    normalized = value.strip().lower()
    return normalized in _UNRESOLVED_EXACT or normalized.startswith(
        _UNRESOLVED_PREFIXES
    )


@dataclass(frozen=True, slots=True)
class ModelSpec:
    provider: str
    model_id: str
    revision: str
    tokenizer_id: str
    tokenizer_revision: str
    chat_template_sha256: str
    adapter_version: str
    dtype: str
    quantization: str
    seed_support: SeedSupport

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        allow_unresolved: bool = False,
    ) -> "ModelSpec":
        value = _mapping(value, "model")
        expected = {
            "provider",
            "model_id",
            "revision",
            "tokenizer_id",
            "tokenizer_revision",
            "chat_template_sha256",
            "adapter_version",
            "dtype",
            "quantization",
            "seed_support",
        }
        _exact_keys(value, expected, "model")
        fields = {key: _nonempty_string(value[key], f"model.{key}") for key in expected}
        if fields["seed_support"] not in {"strict", "best_effort", "none"}:
            raise EvaluationError(
                "model.seed_support must be strict, best_effort, or none"
            )
        model = cls(**fields)  # type: ignore[arg-type]
        if not allow_unresolved:
            model.assert_resolved()
        return model

    def unresolved_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        identity_fields = {
            "provider": self.provider,
            "model_id": self.model_id,
            "revision": self.revision,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_revision": self.tokenizer_revision,
            "adapter_version": self.adapter_version,
        }
        for name, value in identity_fields.items():
            if _looks_unresolved(value):
                reasons.append(f"model.{name} is unresolved")
        if "/" in self.model_id:
            if not _GIT_COMMIT_RE.fullmatch(self.revision):
                reasons.append(
                    "model.revision must be a full commit SHA for a repository model ID"
                )
            if not _GIT_COMMIT_RE.fullmatch(self.tokenizer_revision):
                reasons.append(
                    "model.tokenizer_revision must be a full commit SHA for a repository tokenizer"
                )
        if not _SHA256_RE.fullmatch(self.chat_template_sha256):
            reasons.append("model.chat_template_sha256 is not a pinned SHA-256 digest")
        return tuple(reasons)

    def assert_resolved(self) -> None:
        reasons = self.unresolved_reasons()
        if reasons:
            raise EvaluationError(
                "Model configuration is not runnable: " + "; ".join(reasons)
            )

    def to_dict(self) -> dict[str, str]:
        return {
            "provider": self.provider,
            "model_id": self.model_id,
            "revision": self.revision,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_revision": self.tokenizer_revision,
            "chat_template_sha256": self.chat_template_sha256,
            "adapter_version": self.adapter_version,
            "dtype": self.dtype,
            "quantization": self.quantization,
            "seed_support": self.seed_support,
        }

    @property
    def fingerprint(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_dict()))


@dataclass(frozen=True, slots=True)
class SamplingSpec:
    do_sample: bool
    temperature: float
    top_p: float
    top_k: int
    max_new_tokens: int
    num_beams: int
    repetition_penalty: float
    stop: tuple[str, ...]
    base_seed: int

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SamplingSpec":
        value = _mapping(value, "sampling")
        expected = {
            "do_sample",
            "temperature",
            "top_p",
            "top_k",
            "max_new_tokens",
            "num_beams",
            "repetition_penalty",
            "stop",
            "base_seed",
        }
        _exact_keys(value, expected, "sampling")
        do_sample = value["do_sample"]
        temperature = value["temperature"]
        top_p = value["top_p"]
        top_k = value["top_k"]
        max_new_tokens = value["max_new_tokens"]
        num_beams = value["num_beams"]
        repetition_penalty = value["repetition_penalty"]
        base_seed = value["base_seed"]
        stop = value["stop"]
        if not isinstance(do_sample, bool):
            raise EvaluationError("sampling.do_sample must be a boolean")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or temperature < 0
        ):
            raise EvaluationError("sampling.temperature must be finite and nonnegative")
        if (
            isinstance(top_p, bool)
            or not isinstance(top_p, (int, float))
            or not math.isfinite(top_p)
            or not 0 < top_p <= 1
        ):
            raise EvaluationError("sampling.top_p must be finite and in (0, 1]")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise EvaluationError("sampling.top_k must be a nonnegative integer")
        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or max_new_tokens <= 0
        ):
            raise EvaluationError(
                "sampling.max_new_tokens must be a positive integer"
            )
        if isinstance(num_beams, bool) or not isinstance(num_beams, int):
            raise EvaluationError("sampling.num_beams must be an integer")
        if num_beams != 1:
            raise EvaluationError("This protocol requires sampling.num_beams = 1")
        if (
            isinstance(repetition_penalty, bool)
            or not isinstance(repetition_penalty, (int, float))
            or not math.isfinite(repetition_penalty)
            or repetition_penalty <= 0
        ):
            raise EvaluationError(
                "sampling.repetition_penalty must be finite and positive"
            )
        if (
            isinstance(base_seed, bool)
            or not isinstance(base_seed, int)
            or not 0 <= base_seed < 2**31
        ):
            raise EvaluationError("sampling.base_seed must be in [0, 2^31)")
        if not isinstance(stop, list) or not all(isinstance(item, str) for item in stop):
            raise EvaluationError("sampling.stop must be a list of strings")
        if len(set(stop)) != len(stop):
            raise EvaluationError("sampling.stop must not contain duplicates")
        return cls(
            do_sample=do_sample,
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            repetition_penalty=float(repetition_penalty),
            stop=tuple(stop),
            base_seed=base_seed,
        )

    def to_dict(self, *, seed: int | None = None) -> dict[str, Any]:
        value: dict[str, Any] = {
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_new_tokens": self.max_new_tokens,
            "num_beams": self.num_beams,
            "repetition_penalty": self.repetition_penalty,
            "stop": list(self.stop),
            "base_seed": self.base_seed,
        }
        if seed is not None:
            value["seed"] = seed
        return value


def _request_sampling(value: Any) -> dict[str, Any]:
    value = _mapping(value, "request sampling")
    expected = {
        "do_sample",
        "temperature",
        "top_p",
        "top_k",
        "max_new_tokens",
        "num_beams",
        "repetition_penalty",
        "stop",
        "seed",
    }
    _exact_keys(value, expected, "request sampling")
    seed = value["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31:
        raise EvaluationError("request sampling.seed must be in [0, 2^31)")
    full = dict(value)
    full.pop("seed")
    full["base_seed"] = 0
    normalized = SamplingSpec.from_dict(full).to_dict(seed=seed)
    normalized.pop("base_seed")
    return normalized


def _semantic_request_payload(
    *,
    stage: Stage,
    question_id: str,
    rollout_index: int | None,
    source_task_ids: Sequence[str],
    input_sha256: str,
    prompt_template_sha256: str,
    model: ModelSpec,
    messages: Sequence[Mapping[str, str]],
    sampling: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "question_id": question_id,
        "rollout_index": rollout_index,
        "source_task_ids": list(source_task_ids),
        "input_sha256": input_sha256,
        "prompt_template_sha256": prompt_template_sha256,
        "model": model.to_dict(),
        "messages": [dict(message) for message in messages],
        "sampling": dict(sampling),
    }


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    task_id: str
    request_fingerprint: str
    stage: Stage
    question_id: str
    rollout_index: int | None
    source_task_ids: tuple[str, ...]
    input_sha256: str
    prompt_template_sha256: str
    model: ModelSpec
    messages: tuple[dict[str, str], ...]
    sampling: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "task_id": self.task_id,
            "request_fingerprint": self.request_fingerprint,
            "stage": self.stage,
            "question_id": self.question_id,
            "rollout_index": self.rollout_index,
            "source_task_ids": list(self.source_task_ids),
            "input_sha256": self.input_sha256,
            "prompt_template_sha256": self.prompt_template_sha256,
            "model": self.model.to_dict(),
            "messages": list(self.messages),
            "sampling": self.sampling,
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        allow_unresolved_model: bool = False,
    ) -> "GenerationRequest":
        value = _mapping(value, "generation request")
        expected = {
            "schema_version",
            "task_id",
            "request_fingerprint",
            "stage",
            "question_id",
            "rollout_index",
            "source_task_ids",
            "input_sha256",
            "prompt_template_sha256",
            "model",
            "messages",
            "sampling",
        }
        _exact_keys(value, expected, "generation request")
        if value["schema_version"] != SCHEMA_VERSION:
            raise EvaluationError("Unsupported generation-request schema")
        stage = value["stage"]
        if stage not in {"raw", "synthesis"}:
            raise EvaluationError(f"Invalid request stage: {stage}")
        rollout_index = value["rollout_index"]
        if rollout_index is not None and (
            isinstance(rollout_index, bool)
            or not isinstance(rollout_index, int)
            or rollout_index < 0
        ):
            raise EvaluationError(
                "rollout_index must be null or a nonnegative integer"
            )
        sources = value["source_task_ids"]
        if not isinstance(sources, list) or not all(
            isinstance(item, str) and item for item in sources
        ):
            raise EvaluationError("source_task_ids must be a list of nonempty strings")
        if len(set(sources)) != len(sources):
            raise EvaluationError("source_task_ids must be unique")
        messages = value["messages"]
        if not isinstance(messages, list) or not messages:
            raise EvaluationError("messages must be a nonempty list")
        normalized_messages: list[dict[str, str]] = []
        for index, message in enumerate(messages):
            message = _mapping(message, f"messages[{index}]")
            _exact_keys(message, {"role", "content"}, f"messages[{index}]")
            role = _nonempty_string(message["role"], f"messages[{index}].role")
            if role not in {"system", "user", "assistant"}:
                raise EvaluationError(f"messages[{index}].role is unsupported: {role}")
            normalized_messages.append(
                {
                    "role": role,
                    "content": _nonempty_string(
                        message["content"], f"messages[{index}].content"
                    ),
                }
            )
        model = ModelSpec.from_dict(
            _mapping(value["model"], "model"),
            allow_unresolved=allow_unresolved_model,
        )
        sampling = _request_sampling(value["sampling"])
        request = cls(
            task_id=_nonempty_string(value["task_id"], "task_id"),
            request_fingerprint=_sha256(
                value["request_fingerprint"], "request_fingerprint"
            ),
            stage=stage,
            question_id=_nonempty_string(value["question_id"], "question_id"),
            rollout_index=rollout_index,
            source_task_ids=tuple(sources),
            input_sha256=_sha256(value["input_sha256"], "input_sha256"),
            prompt_template_sha256=_sha256(
                value["prompt_template_sha256"], "prompt_template_sha256"
            ),
            model=model,
            messages=tuple(normalized_messages),
            sampling=sampling,
        )
        if request.stage == "raw" and (
            request.rollout_index is None or request.source_task_ids
        ):
            raise EvaluationError(
                "Raw requests require rollout_index and no source_task_ids"
            )
        if request.stage == "synthesis" and (
            request.rollout_index is not None or not request.source_task_ids
        ):
            raise EvaluationError(
                "Synthesis requests require source_task_ids and no rollout_index"
            )
        semantic_payload = _semantic_request_payload(
            stage=request.stage,
            question_id=request.question_id,
            rollout_index=request.rollout_index,
            source_task_ids=request.source_task_ids,
            input_sha256=request.input_sha256,
            prompt_template_sha256=request.prompt_template_sha256,
            model=request.model,
            messages=request.messages,
            sampling=request.sampling,
        )
        expected_fingerprint = sha256_bytes(canonical_json_bytes(semantic_payload))
        expected_task_id = f"{request.stage}-{expected_fingerprint[:24]}"
        if request.request_fingerprint != expected_fingerprint:
            raise EvaluationError(
                f"Request fingerprint does not match semantic payload: {request.task_id}"
            )
        if request.task_id != expected_task_id:
            raise EvaluationError(
                f"Request task ID does not match semantic payload: {request.task_id}"
            )
        return request


@dataclass(frozen=True, slots=True)
class BackendOutput:
    task_id: str
    request_fingerprint: str
    text: str
    finish_reason: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    provider_metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class GenerationResult:
    task_id: str
    request_fingerprint: str
    backend_fingerprint: str
    text: str
    output_sha256: str
    finish_reason: str
    prompt_tokens: int | None
    completion_tokens: int | None
    provider_metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "task_id": self.task_id,
            "request_fingerprint": self.request_fingerprint,
            "backend_fingerprint": self.backend_fingerprint,
            "text": self.text,
            "output_sha256": self.output_sha256,
            "finish_reason": self.finish_reason,
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
            },
            "provider_metadata": dict(self.provider_metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GenerationResult":
        value = _mapping(value, "generation result")
        expected = {
            "schema_version",
            "task_id",
            "request_fingerprint",
            "backend_fingerprint",
            "text",
            "output_sha256",
            "finish_reason",
            "usage",
            "provider_metadata",
        }
        _exact_keys(value, expected, "generation result")
        if value["schema_version"] != SCHEMA_VERSION:
            raise EvaluationError("Unsupported generation-result schema")
        usage = _mapping(value["usage"], "usage")
        _exact_keys(usage, {"prompt_tokens", "completion_tokens"}, "usage")
        prompt_tokens = _nonnegative_count(usage["prompt_tokens"], "usage.prompt_tokens")
        completion_tokens = _nonnegative_count(
            usage["completion_tokens"], "usage.completion_tokens"
        )
        metadata = _mapping(value["provider_metadata"], "provider_metadata")
        if not isinstance(value["text"], str):
            raise EvaluationError("result text must be a string")
        output_sha256 = _sha256(value["output_sha256"], "output_sha256")
        if output_sha256 != sha256_text(value["text"]):
            raise EvaluationError("Generation-result output hash mismatch")
        return cls(
            task_id=_nonempty_string(value["task_id"], "task_id"),
            request_fingerprint=_sha256(
                value["request_fingerprint"], "request_fingerprint"
            ),
            backend_fingerprint=_sha256(
                value["backend_fingerprint"], "backend_fingerprint"
            ),
            text=value["text"],
            output_sha256=output_sha256,
            finish_reason=_finish_reason(value["finish_reason"], "finish_reason"),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            provider_metadata=dict(metadata),
        )


GENERATION_ROW_KEYS = {
    "schema_version",
    "stage",
    "task_id",
    "request_fingerprint",
    "backend_fingerprint",
    "question_id",
    "rollout_index",
    "source_task_ids",
    "seed",
    "text",
    "output_sha256",
    "finish_reason",
    "usage",
    "provider_metadata",
}


def validate_and_order_generations(
    rows: Sequence[Mapping[str, Any]],
    requests: Sequence[GenerationRequest],
) -> list[dict[str, Any]]:
    """Bind every materialized generation to its immutable planned request."""

    request_by_id = {request.task_id: request for request in requests}
    if len(request_by_id) != len(requests):
        raise EvaluationError("Planned request task IDs must be unique")
    row_by_id: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        row = _mapping(row, f"generation[{index}]")
        _exact_keys(row, GENERATION_ROW_KEYS, f"generation[{index}]")
        task_id = _nonempty_string(row["task_id"], f"generation[{index}].task_id")
        if task_id in row_by_id:
            raise EvaluationError(f"Duplicate generation task ID: {task_id}")
        row_by_id[task_id] = row
    if set(row_by_id) != set(request_by_id):
        raise EvaluationError(
            "Generation/request coverage mismatch: "
            f"missing={sorted(set(request_by_id) - set(row_by_id))}, "
            f"extra={sorted(set(row_by_id) - set(request_by_id))}"
        )

    ordered: list[dict[str, Any]] = []
    for request in requests:
        row = row_by_id[request.task_id]
        immutable_expected = {
            "schema_version": SCHEMA_VERSION,
            "stage": request.stage,
            "task_id": request.task_id,
            "request_fingerprint": request.request_fingerprint,
            "question_id": request.question_id,
            "rollout_index": request.rollout_index,
            "source_task_ids": list(request.source_task_ids),
            "seed": request.sampling["seed"],
        }
        for field, expected in immutable_expected.items():
            if row[field] != expected or (
                field in {"rollout_index", "seed"} and isinstance(row[field], bool)
            ):
                raise EvaluationError(
                    f"Generation {request.task_id} does not match request field {field}"
                )
        text = row["text"]
        if not isinstance(text, str):
            raise EvaluationError(f"Generation text must be a string: {request.task_id}")
        output_sha256 = _sha256(
            row["output_sha256"], f"generation {request.task_id}.output_sha256"
        )
        if output_sha256 != sha256_text(text):
            raise EvaluationError(
                f"Generation output hash mismatch: {request.task_id}"
            )
        _sha256(
            row["backend_fingerprint"],
            f"generation {request.task_id}.backend_fingerprint",
        )
        _finish_reason(
            row["finish_reason"], f"generation {request.task_id}.finish_reason"
        )
        usage = _mapping(row["usage"], f"generation {request.task_id}.usage")
        _exact_keys(
            usage,
            {"prompt_tokens", "completion_tokens"},
            f"generation {request.task_id}.usage",
        )
        _nonnegative_count(
            usage["prompt_tokens"],
            f"generation {request.task_id}.usage.prompt_tokens",
        )
        _nonnegative_count(
            usage["completion_tokens"],
            f"generation {request.task_id}.usage.completion_tokens",
        )
        _mapping(
            row["provider_metadata"],
            f"generation {request.task_id}.provider_metadata",
        )
        ordered.append(dict(row))
    return ordered


def make_request(
    *,
    stage: Stage,
    question_id: str,
    rollout_index: int | None,
    source_task_ids: Sequence[str],
    input_sha256: str,
    prompt_template_sha256: str,
    model: ModelSpec,
    messages: Sequence[dict[str, str]],
    sampling: Mapping[str, Any],
) -> GenerationRequest:
    semantic_payload = _semantic_request_payload(
        stage=stage,
        question_id=question_id,
        rollout_index=rollout_index,
        source_task_ids=source_task_ids,
        input_sha256=input_sha256,
        prompt_template_sha256=prompt_template_sha256,
        model=model,
        messages=messages,
        sampling=sampling,
    )
    fingerprint = sha256_bytes(canonical_json_bytes(semantic_payload))
    value = {
        **semantic_payload,
        "task_id": f"{stage}-{fingerprint[:24]}",
        "request_fingerprint": fingerprint,
    }
    return GenerationRequest.from_dict(value, allow_unresolved_model=True)
