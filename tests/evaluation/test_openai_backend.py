from __future__ import annotations

import os
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from compute_as_a_teacher.evaluation.artifacts import sha256_text  # noqa: E402
from compute_as_a_teacher.evaluation.openai_backend import (  # noqa: E402
    BACKEND_NAME,
    BACKEND_VERSION,
    OpenAIBackendError,
    OpenAICompatibleBackend,
    SUPPORTED_SAMPLING_FIELDS,
)
from compute_as_a_teacher.evaluation.schemas import (  # noqa: E402
    GenerationRequest,
    ModelSpec,
    make_request,
)


def model_spec(**overrides: str) -> ModelSpec:
    values = {
        "provider": BACKEND_NAME,
        "model_id": "trained-math500-final",
        "revision": "a" * 64,
        "tokenizer_id": "Qwen/Qwen3-4B",
        "tokenizer_revision": "b" * 40,
        "chat_template_sha256": "c" * 64,
        "adapter_version": BACKEND_VERSION,
        "dtype": "bfloat16",
        "quantization": "none",
        "seed_support": "best_effort",
    }
    values.update(overrides)
    return ModelSpec(**values)  # type: ignore[arg-type]


def raw_request(
    model: ModelSpec,
    *,
    question_id: str = "q-1",
    do_sample: bool = True,
    stop: tuple[str, ...] = ("</s>",),
) -> GenerationRequest:
    sampling = {
        "do_sample": do_sample,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "max_new_tokens": 1536,
        "num_beams": 1,
        "repetition_penalty": 1.05,
        "stop": list(stop),
        "seed": 1729,
    }
    return make_request(
        stage="raw",
        question_id=question_id,
        rollout_index=0,
        source_task_ids=(),
        input_sha256=sha256_text(question_id),
        prompt_template_sha256="d" * 64,
        model=model,
        messages=(
            {"role": "system", "content": "Solve carefully."},
            {"role": "user", "content": "What is 1 + 1?"},
        ),
        sampling=sampling,
    )


class RecordingTransport:
    def __init__(self, finish_reason: str = "stop") -> None:
        self.calls: list[dict[str, Any]] = []
        self.finish_reason = finish_reason

    def __call__(
        self,
        *,
        url: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        self.calls.append(
            {
                "url": url,
                "payload": dict(payload),
                "headers": dict(headers),
                "timeout_seconds": timeout_seconds,
            }
        )
        return {
            "id": "chatcmpl-test",
            "created": 123,
            "model": payload["model"],
            "system_fingerprint": "fp-test",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": r"\boxed{2}"},
                    "finish_reason": self.finish_reason,
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4},
        }


class OpenAICompatibleBackendTests(unittest.TestCase):
    def test_maps_plan_request_and_parses_response(self) -> None:
        model = model_spec()
        request = raw_request(model)
        transport = RecordingTransport()
        backend = OpenAICompatibleBackend(
            model=model,
            base_url="http://127.0.0.1:8000/v1/",
            timeout_seconds=45,
            api_key="test-key",
            max_workers=1,
            transport=transport,
        )

        output = backend.generate_batch([request])[0]

        self.assertEqual(backend.descriptor.name, BACKEND_NAME)
        self.assertEqual(
            backend.descriptor.version,
            f"{BACKEND_VERSION}+endpoint-sha256-"
            f"{sha256_text('http://127.0.0.1:8000/v1/chat/completions')}",
        )
        self.assertEqual(
            backend.descriptor.supported_sampling_fields,
            SUPPORTED_SAMPLING_FIELDS,
        )
        self.assertTrue(backend.descriptor.non_reportable)
        call = transport.calls[0]
        self.assertEqual(call["url"], "http://127.0.0.1:8000/v1/chat/completions")
        self.assertEqual(call["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(call["timeout_seconds"], 45.0)
        self.assertEqual(
            call["payload"],
            {
                "model": "trained-math500-final",
                "messages": [
                    {"role": "system", "content": "Solve carefully."},
                    {"role": "user", "content": "What is 1 + 1?"},
                ],
                "n": 1,
                "stream": False,
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "max_tokens": 1536,
                "repetition_penalty": 1.05,
                "seed": 1729,
                "stop": ["</s>"],
            },
        )
        self.assertEqual(output.task_id, request.task_id)
        self.assertEqual(output.request_fingerprint, request.request_fingerprint)
        self.assertEqual(output.text, r"\boxed{2}")
        self.assertEqual(output.finish_reason, "stop")
        self.assertEqual(output.prompt_tokens, 12)
        self.assertEqual(output.completion_tokens, 4)
        self.assertEqual(
            output.provider_metadata,
            {
                "response_model": "trained-math500-final",
                "id": "chatcmpl-test",
                "created": 123,
                "system_fingerprint": "fp-test",
            },
        )

    def test_greedy_request_uses_zero_temperature_and_omits_empty_stop(self) -> None:
        model = model_spec()
        request = raw_request(model, do_sample=False, stop=())
        transport = RecordingTransport()
        backend = OpenAICompatibleBackend(
            model=model,
            base_url="https://example.test/v1/chat/completions",
            max_workers=1,
            transport=transport,
        )

        backend.generate_batch([request])

        self.assertEqual(transport.calls[0]["payload"]["temperature"], 0.0)
        self.assertNotIn("stop", transport.calls[0]["payload"])

    def test_accepts_stop_or_length_and_rejects_other_finish_reasons(self) -> None:
        model = model_spec()
        for reason in ("stop", "length"):
            with self.subTest(reason=reason):
                backend = OpenAICompatibleBackend(
                    model=model,
                    base_url="http://localhost:8000/v1",
                    transport=RecordingTransport(reason),
                )
                self.assertEqual(
                    backend.generate_batch([raw_request(model)])[0].finish_reason,
                    reason,
                )
        backend = OpenAICompatibleBackend(
            model=model,
            base_url="http://localhost:8000/v1",
            transport=RecordingTransport("content_filter"),
        )
        with self.assertRaisesRegex(OpenAIBackendError, "finish_reason"):
            backend.generate_batch([raw_request(model)])

    def test_batch_output_order_matches_request_order(self) -> None:
        model = model_spec()
        requests = [raw_request(model, question_id=f"q-{index}") for index in range(3)]
        backend = OpenAICompatibleBackend(
            model=model,
            base_url="http://localhost:8000/v1",
            max_workers=3,
            transport=RecordingTransport(),
        )

        outputs = backend.generate_batch(requests)

        self.assertEqual(
            [output.task_id for output in outputs],
            [request.task_id for request in requests],
        )

    def test_rejects_wrong_backend_identity(self) -> None:
        with self.assertRaisesRegex(OpenAIBackendError, "provider"):
            OpenAICompatibleBackend(
                model=model_spec(provider="huggingface"),
                base_url="http://localhost:8000/v1",
                transport=RecordingTransport(),
            )
        with self.assertRaisesRegex(OpenAIBackendError, "adapter_version"):
            OpenAICompatibleBackend(
                model=model_spec(adapter_version="other-adapter"),
                base_url="http://localhost:8000/v1",
                transport=RecordingTransport(),
            )

    def test_rejects_request_model_mismatch_before_transport(self) -> None:
        planned_model = model_spec()
        transport = RecordingTransport()
        backend = OpenAICompatibleBackend(
            model=planned_model,
            base_url="http://localhost:8000/v1",
            transport=transport,
        )
        mismatched_request = raw_request(
            replace(planned_model, revision="e" * 64)
        )

        with self.assertRaisesRegex(OpenAIBackendError, "request model"):
            backend.generate_batch([mismatched_request])
        self.assertEqual(transport.calls, [])

    def test_rejects_response_for_a_different_served_model(self) -> None:
        def wrong_model_transport(**_: Any) -> Mapping[str, Any]:
            return {"model": "wrong-model", "choices": []}

        model = model_spec()
        backend = OpenAICompatibleBackend(
            model=model,
            base_url="http://localhost:8000/v1",
            transport=wrong_model_transport,
        )

        with self.assertRaisesRegex(OpenAIBackendError, "response model"):
            backend.generate_batch([raw_request(model)])

    def test_environment_key_is_required_only_when_named(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(OpenAIBackendError, "is unset"):
                OpenAICompatibleBackend.from_environment(
                    model=model_spec(),
                    base_url="http://localhost:8000/v1",
                    api_key_env="CAT_EVAL_API_KEY",
                    transport=RecordingTransport(),
                )
            backend = OpenAICompatibleBackend.from_environment(
                model=model_spec(),
                base_url="http://localhost:8000/v1",
                transport=RecordingTransport(),
            )
        self.assertIsNone(backend.api_key)

    def test_endpoint_identity_is_normalized_and_changes_resume_fingerprint(self) -> None:
        model = model_spec()
        first = OpenAICompatibleBackend(
            model=model,
            base_url="http://127.0.0.1:8000/v1/",
            api_key="first-secret",
            transport=RecordingTransport(),
        )
        equivalent = OpenAICompatibleBackend(
            model=model,
            base_url="http://127.0.0.1:8000/v1/chat/completions",
            api_key="different-secret",
            transport=RecordingTransport(),
        )
        other_endpoint = OpenAICompatibleBackend(
            model=model,
            base_url="http://127.0.0.1:9000/v1",
            transport=RecordingTransport(),
        )

        self.assertEqual(first.descriptor.fingerprint, equivalent.descriptor.fingerprint)
        self.assertNotEqual(
            first.descriptor.fingerprint,
            other_endpoint.descriptor.fingerprint,
        )
        self.assertNotIn("first-secret", first.descriptor.version)
        self.assertNotIn("127.0.0.1", first.descriptor.version)


if __name__ == "__main__":
    unittest.main()
