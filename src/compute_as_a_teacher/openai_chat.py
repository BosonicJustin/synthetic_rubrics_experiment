"""Small standard-library transport for OpenAI-compatible chat endpoints."""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


MAX_RESPONSE_BYTES = 16 * 1024 * 1024
SUPPORTED_FINISH_REASONS = frozenset({"stop", "length"})
_ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


class OpenAIChatError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class TransportResponse:
    status_code: int
    body: bytes | str | Mapping[str, Any]


class JsonTransport(Protocol):
    def __call__(
        self,
        *,
        url: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> TransportResponse | Mapping[str, Any]:
        ...


def chat_endpoint(base_url: str) -> str:
    if not isinstance(base_url, str) or not base_url.strip():
        raise OpenAIChatError("base URL must be nonempty text")
    normalized = base_url.rstrip("/")
    if any(character.isspace() or ord(character) < 32 for character in normalized):
        raise OpenAIChatError("base URL contains whitespace or control bytes")
    try:
        parsed = urlsplit(normalized)
        _ = parsed.port
    except ValueError as exc:
        raise OpenAIChatError("base URL is malformed") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise OpenAIChatError("base URL must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise OpenAIChatError("base URL must not contain credentials")
    if parsed.query or parsed.fragment:
        raise OpenAIChatError("base URL must not contain a query or fragment")
    if parsed.path.rstrip("/").endswith("/chat/completions"):
        return normalized
    return normalized + "/chat/completions"


def api_key_from_environment(name: str) -> str | None:
    if not isinstance(name, str) or (name and _ENVIRONMENT_NAME.fullmatch(name) is None):
        raise OpenAIChatError("API-key environment name is invalid")
    if not name:
        return None
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise OpenAIChatError(f"required API-key environment variable is unset: {name}")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_nonfinite(value: str) -> None:
    raise ValueError(value)


def _response_object(body: bytes | str | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(body, Mapping):
        return body
    if isinstance(body, bytes):
        if len(body) > MAX_RESPONSE_BYTES:
            raise OpenAIChatError("chat response is too large")
        try:
            serialized = body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise OpenAIChatError("chat response is not UTF-8 JSON") from exc
    elif isinstance(body, str):
        if len(body.encode("utf-8")) > MAX_RESPONSE_BYTES:
            raise OpenAIChatError("chat response is too large")
        serialized = body
    else:
        raise OpenAIChatError("chat transport returned an invalid body")
    try:
        value = json.loads(
            serialized,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise OpenAIChatError("chat response is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise OpenAIChatError("chat response must be an object")
    return value


def _http_transport(
    *,
    url: str,
    payload: Mapping[str, Any],
    headers: Mapping[str, str],
    timeout_seconds: float,
) -> TransportResponse:
    data = json.dumps(payload, allow_nan=False, ensure_ascii=False).encode("utf-8")
    request = Request(url=url, data=data, headers=dict(headers), method="POST")
    try:
        with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
            body = response.read(MAX_RESPONSE_BYTES + 1)
            status = response.status
    except (HTTPError, URLError, OSError, TimeoutError) as exc:
        raise OpenAIChatError("chat-completions request failed") from exc
    return TransportResponse(status, body)


@dataclass(frozen=True, slots=True)
class OpenAIChatTransport:
    base_url: str
    timeout_seconds: float
    api_key: str | None = None
    transport: JsonTransport = _http_transport
    endpoint: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise OpenAIChatError("timeout_seconds must be positive")
        if self.api_key is not None and (
            not isinstance(self.api_key, str)
            or not self.api_key.strip()
            or "\r" in self.api_key
            or "\n" in self.api_key
        ):
            raise OpenAIChatError("API key is invalid")
        if not callable(self.transport):
            raise OpenAIChatError("transport must be callable")
        object.__setattr__(self, "endpoint", chat_endpoint(self.base_url))

    def post(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.api_key is not None:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            response = self.transport(
                url=self.endpoint,
                payload=payload,
                headers=headers,
                timeout_seconds=float(self.timeout_seconds),
            )
        except OpenAIChatError:
            raise
        except Exception as exc:
            raise OpenAIChatError("chat-completions transport failed") from exc
        if isinstance(response, TransportResponse):
            if type(response.status_code) is not int or not 200 <= response.status_code < 300:
                raise OpenAIChatError("chat-completions endpoint returned an error")
            return _response_object(response.body)
        if not isinstance(response, Mapping):
            raise OpenAIChatError("chat transport returned an invalid response")
        return response


__all__ = [
    "JsonTransport",
    "OpenAIChatError",
    "OpenAIChatTransport",
    "SUPPORTED_FINISH_REASONS",
    "TransportResponse",
    "api_key_from_environment",
    "chat_endpoint",
]
