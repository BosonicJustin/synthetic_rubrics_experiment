"""OpenAI-compatible chat-completions backend for evaluation plans."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from compute_as_a_teacher.openai_chat import (
    JsonTransport,
    OpenAIChatError,
    OpenAIChatTransport,
    SUPPORTED_FINISH_REASONS,
    api_key_from_environment,
)

from .artifacts import sha256_text
from .backend import BackendDescriptor
from .errors import EvaluationError
from .schemas import BackendOutput, GenerationRequest, ModelSpec


BACKEND_NAME = "openai-compatible"
BACKEND_VERSION = "openai-compatible-chat-v1"
SUPPORTED_SAMPLING_FIELDS = frozenset(
    {
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
)


class OpenAIBackendError(EvaluationError):
    pass


def _token_count(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise OpenAIBackendError(f"{name} must be a nonnegative integer")
    return value


@dataclass(frozen=True, slots=True)
class OpenAICompatibleBackend:
    model: ModelSpec
    base_url: str
    timeout_seconds: float = 120.0
    api_key: str | None = None
    max_workers: int = 8
    transport: JsonTransport | None = None
    _client: OpenAIChatTransport = field(init=False, repr=False)
    _descriptor: BackendDescriptor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.model, ModelSpec):
            raise OpenAIBackendError("model must be a ModelSpec")
        self.model.assert_resolved()
        if self.model.provider != BACKEND_NAME:
            raise OpenAIBackendError(f"model.provider must be {BACKEND_NAME!r}")
        if self.model.adapter_version != BACKEND_VERSION:
            raise OpenAIBackendError(
                f"model.adapter_version must be {BACKEND_VERSION!r}"
            )
        if type(self.max_workers) is not int or self.max_workers <= 0:
            raise OpenAIBackendError("max_workers must be a positive integer")
        kwargs = {
            "base_url": self.base_url,
            "timeout_seconds": self.timeout_seconds,
            "api_key": self.api_key,
        }
        if self.transport is not None:
            kwargs["transport"] = self.transport
        try:
            client = OpenAIChatTransport(**kwargs)
        except OpenAIChatError as exc:
            raise OpenAIBackendError(str(exc)) from exc
        object.__setattr__(self, "_client", client)
        object.__setattr__(
            self,
            "_descriptor",
            BackendDescriptor(
                name=BACKEND_NAME,
                version=(
                    f"{BACKEND_VERSION}+endpoint-sha256-"
                    f"{sha256_text(client.endpoint)}"
                ),
                model=self.model,
                supported_sampling_fields=SUPPORTED_SAMPLING_FIELDS,
                non_reportable=True,
            ),
        )

    @classmethod
    def from_environment(
        cls,
        *,
        model: ModelSpec,
        base_url: str,
        api_key_env: str = "",
        timeout_seconds: float = 120.0,
        max_workers: int = 8,
        transport: JsonTransport | None = None,
    ) -> "OpenAICompatibleBackend":
        try:
            api_key = api_key_from_environment(api_key_env)
        except OpenAIChatError as exc:
            raise OpenAIBackendError(str(exc)) from exc
        return cls(model, base_url, timeout_seconds, api_key, max_workers, transport)

    @property
    def descriptor(self) -> BackendDescriptor:
        return self._descriptor

    def _payload(self, request: GenerationRequest) -> dict[str, Any]:
        if not isinstance(request, GenerationRequest) or request.model != self.model:
            raise OpenAIBackendError("request model does not match the backend model")
        sampling = request.sampling
        payload: dict[str, Any] = {
            "model": self.model.model_id,
            "messages": [dict(message) for message in request.messages],
            "n": 1,
            "stream": False,
            "temperature": sampling["temperature"] if sampling["do_sample"] else 0.0,
            "top_p": sampling["top_p"],
            "top_k": sampling["top_k"],
            "max_tokens": sampling["max_new_tokens"],
            "repetition_penalty": sampling["repetition_penalty"],
            "seed": sampling["seed"],
        }
        if sampling["stop"]:
            payload["stop"] = list(sampling["stop"])
        return payload

    def _generate(self, request: GenerationRequest) -> BackendOutput:
        try:
            response = self._client.post(self._payload(request))
        except OpenAIChatError as exc:
            raise OpenAIBackendError(str(exc)) from exc
        if response.get("model") != self.model.model_id:
            raise OpenAIBackendError("response model does not match the planned model")
        choices = response.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise OpenAIBackendError("response must contain exactly one choice")
        choice = choices[0]
        if not isinstance(choice, Mapping) or choice.get("index") != 0:
            raise OpenAIBackendError("response choice must have index 0")
        message = choice.get("message")
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str):
            raise OpenAIBackendError("response choice must contain text content")
        finish_reason = choice.get("finish_reason")
        if finish_reason not in SUPPORTED_FINISH_REASONS:
            raise OpenAIBackendError("response has an unsupported finish_reason")
        usage = response.get("usage")
        if usage is None:
            usage = {}
        elif not isinstance(usage, Mapping):
            raise OpenAIBackendError("response usage must be an object")
        metadata: dict[str, Any] = {"response_model": response["model"]}
        for key in ("id", "created", "system_fingerprint"):
            value = response.get(key)
            if isinstance(value, (str, int)) and not isinstance(value, bool):
                metadata[key] = value
        return BackendOutput(
            task_id=request.task_id,
            request_fingerprint=request.request_fingerprint,
            text=content,
            finish_reason=finish_reason,
            prompt_tokens=_token_count(usage.get("prompt_tokens"), "prompt_tokens"),
            completion_tokens=_token_count(usage.get("completion_tokens"), "completion_tokens"),
            provider_metadata=metadata,
        )

    def generate_batch(
        self,
        requests: Sequence[GenerationRequest],
    ) -> Sequence[BackendOutput]:
        if not requests:
            return []
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(requests))) as pool:
            return list(pool.map(self._generate, requests))


__all__ = [
    "BACKEND_NAME",
    "BACKEND_VERSION",
    "JsonTransport",
    "OpenAIBackendError",
    "OpenAICompatibleBackend",
    "SUPPORTED_SAMPLING_FIELDS",
]
