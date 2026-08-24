"""OpenAI-compatible client used by the frozen synthesis anchor."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from compute_as_a_teacher.openai_chat import (
    JsonTransport,
    OpenAIChatError,
    OpenAIChatTransport,
    SUPPORTED_FINISH_REASONS,
    TransportResponse,
    api_key_from_environment,
)

from .errors import RewardContractError


class AnchorClientError(RewardContractError):
    pass


@dataclass(frozen=True, slots=True)
class AnchorCompletion:
    text: str
    finish_reason: str


class AnchorChatClient(Protocol):
    def complete(
        self,
        *,
        model: str,
        message: str,
        temperature: float,
        top_p: float,
        top_k: int,
        max_tokens: int,
        seed: int,
    ) -> str | AnchorCompletion:
        ...


def _number(value: Any, name: str, lower: float, upper: float, *, inclusive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnchorClientError(f"{name} is outside its supported range")
    normalized = float(value)
    lower_ok = normalized >= lower if inclusive else normalized > lower
    if not math.isfinite(normalized) or not lower_ok or normalized > upper:
        raise AnchorClientError(f"{name} is outside its supported range")
    return normalized


def _positive_integer(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise AnchorClientError(f"{name} must be a positive integer")
    return value


def _seed(value: Any) -> int:
    if type(value) is not int or not 0 <= value < 2**31:
        raise AnchorClientError("seed must be an integer in [0, 2**31)")
    return value


def _text_response(value: Mapping[str, Any], expected_model: str) -> AnchorCompletion:
    if value.get("model") != expected_model:
        raise AnchorClientError("anchor response model does not match the frozen anchor")
    choices = value.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise AnchorClientError("anchor response must contain exactly one choice")
    choice = choices[0]
    if not isinstance(choice, Mapping) or choice.get("index") != 0:
        raise AnchorClientError("anchor response choice must have index 0")
    finish_reason = choice.get("finish_reason")
    if finish_reason not in SUPPORTED_FINISH_REASONS:
        raise AnchorClientError("anchor response has an unsupported finish reason")
    message = choice.get("message") if isinstance(choice, Mapping) else None
    content = message.get("content") if isinstance(message, Mapping) else None
    if not isinstance(content, str) or not content.strip():
        raise AnchorClientError("anchor response must contain one text response")
    return AnchorCompletion(content, finish_reason)


def completion_text(value: str | AnchorCompletion) -> tuple[str, str]:
    if isinstance(value, AnchorCompletion):
        return value.text, value.finish_reason
    if isinstance(value, str):
        return value, "unknown"
    raise AnchorClientError("anchor client returned an invalid completion")


@dataclass(frozen=True, slots=True)
class OpenAIChatCompletionsClient:
    base_url: str
    timeout_seconds: float
    api_key: str | None = None
    transport: JsonTransport | None = None
    _client: OpenAIChatTransport = field(init=False, repr=False)

    def __post_init__(self) -> None:
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
            raise AnchorClientError(str(exc)) from exc
        object.__setattr__(self, "_client", client)

    @classmethod
    def from_environment(
        cls,
        *,
        base_url: str,
        api_key_env: str,
        timeout_seconds: float,
        transport: JsonTransport | None = None,
    ) -> "OpenAIChatCompletionsClient":
        try:
            api_key = api_key_from_environment(api_key_env)
        except OpenAIChatError as exc:
            raise AnchorClientError(str(exc)) from exc
        return cls(base_url, timeout_seconds, api_key, transport)

    @property
    def endpoint(self) -> str:
        return self._client.endpoint

    def complete(
        self,
        *,
        model: str,
        message: str,
        temperature: float,
        top_p: float,
        top_k: int,
        max_tokens: int,
        seed: int,
    ) -> AnchorCompletion:
        if not isinstance(model, str) or not model.strip():
            raise AnchorClientError("anchor model must be nonempty text")
        if not isinstance(message, str) or not message:
            raise AnchorClientError("anchor message must be nonempty text")
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": message}],
            "temperature": _number(temperature, "temperature", 0, 2, inclusive=True),
            "top_p": _number(top_p, "top_p", 0, 1, inclusive=False),
            "top_k": _positive_integer(top_k, "top_k"),
            "max_tokens": _positive_integer(max_tokens, "max_tokens"),
            "seed": _seed(seed),
        }
        try:
            return _text_response(self._client.post(payload), model)
        except OpenAIChatError as exc:
            raise AnchorClientError(str(exc)) from exc


__all__ = [
    "AnchorChatClient",
    "AnchorCompletion",
    "AnchorClientError",
    "JsonTransport",
    "OpenAIChatCompletionsClient",
    "TransportResponse",
    "completion_text",
]
