from __future__ import annotations

import os
import sys
import unittest
from collections.abc import Mapping
from pathlib import Path
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from compute_as_a_teacher.training.anchor_client import (  # noqa: E402
    AnchorClientError,
    OpenAIChatCompletionsClient,
    TransportResponse,
)


class RecordingTransport:
    def __init__(self, response: TransportResponse) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        *,
        url: str,
        payload: Mapping[str, object],
        headers: dict[str, str],
        timeout_seconds: float,
    ) -> TransportResponse:
        self.calls.append(
            {
                "url": url,
                "payload": dict(payload),
                "headers": dict(headers),
                "timeout_seconds": timeout_seconds,
            }
        )
        return self.response


class OpenAIChatCompletionsClientTests(unittest.TestCase):
    def test_builds_one_user_message_and_parses_one_text_choice(self) -> None:
        transport = RecordingTransport(
            TransportResponse(
                status_code=200,
                body={
                    "id": "answer-1",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": r"synthesis \boxed{42}",
                            },
                        }
                    ],
                },
            )
        )
        client = OpenAIChatCompletionsClient(
            base_url="http://127.0.0.1:8000/v1/",
            timeout_seconds=12.5,
            api_key="local-test-key",
            transport=transport,
        )

        text = client.complete(
            model="frozen-anchor",
            message="eight rollouts only",
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            max_tokens=1536,
            seed=123,
        )

        self.assertEqual(text, r"synthesis \boxed{42}")
        self.assertEqual(len(transport.calls), 1)
        call = transport.calls[0]
        self.assertEqual(call["url"], "http://127.0.0.1:8000/v1/chat/completions")
        self.assertEqual(call["timeout_seconds"], 12.5)
        self.assertEqual(
            call["payload"],
            {
                "model": "frozen-anchor",
                "messages": [{"role": "user", "content": "eight rollouts only"}],
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "max_tokens": 1536,
                "seed": 123,
            },
        )
        self.assertEqual(
            call["headers"]["Authorization"],  # type: ignore[index]
            "Bearer local-test-key",
        )

    def test_environment_key_is_read_at_construction_not_stored_by_name(self) -> None:
        transport = RecordingTransport(
            TransportResponse(
                200,
                {"choices": [{"message": {"content": r"\boxed{1}"}}]},
            )
        )
        with patch.dict(os.environ, {"CAT_ANCHOR_TEST_KEY": "secret"}, clear=False):
            client = OpenAIChatCompletionsClient.from_environment(
                base_url="https://anchor.example/v1",
                api_key_env="CAT_ANCHOR_TEST_KEY",
                timeout_seconds=5,
                transport=transport,
            )
        self.assertEqual(client.api_key, "secret")
        with self.assertRaisesRegex(AnchorClientError, "environment variable is unset"):
            OpenAIChatCompletionsClient.from_environment(
                base_url="https://anchor.example/v1",
                api_key_env="CAT_MISSING_TEST_KEY",
                timeout_seconds=5,
                transport=transport,
            )

    def test_invalid_endpoints_fail_before_transport(self) -> None:
        response = TransportResponse(
            200,
            {"choices": [{"message": {"content": r"\boxed{1}"}}]},
        )
        for endpoint in (
            "anchor.example/v1",
            "ftp://anchor.example/v1",
            "https://user:password@anchor.example/v1",
            "https://anchor.example/v1?api_key=secret",
            "http://anchor.example:99999/v1",
        ):
            with self.subTest(endpoint=endpoint):
                transport = RecordingTransport(response)
                with self.assertRaises(AnchorClientError):
                    OpenAIChatCompletionsClient(
                        base_url=endpoint,
                        timeout_seconds=5,
                        transport=transport,
                    )
                self.assertEqual(transport.calls, [])

    def test_malformed_or_multiple_responses_fail_closed(self) -> None:
        bad_responses = (
            TransportResponse(500, {"choices": [{"message": {"content": "x"}}]}),
            TransportResponse(200, b"not json"),
            TransportResponse(200, {"choices": []}),
            TransportResponse(
                200,
                {
                    "choices": [
                        {"message": {"content": "first"}},
                        {"message": {"content": "second"}},
                    ]
                },
            ),
            TransportResponse(200, {"choices": [{"message": {"content": None}}]}),
            TransportResponse(200, {"choices": [{"message": {"content": "  "}}]}),
        )
        for response in bad_responses:
            with self.subTest(response=response):
                client = OpenAIChatCompletionsClient(
                    base_url="http://127.0.0.1:8000/v1",
                    timeout_seconds=5,
                    transport=RecordingTransport(response),
                )
                with self.assertRaises(AnchorClientError):
                    client.complete(
                        model="anchor",
                        message="prompt",
                        temperature=0.7,
                        top_p=0.8,
                        top_k=20,
                        max_tokens=1536,
                        seed=1,
                    )


if __name__ == "__main__":
    unittest.main()
