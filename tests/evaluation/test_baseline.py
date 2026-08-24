from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from typing import Sequence
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from compute_as_a_teacher.data.math500 import QuestionRecord  # noqa: E402
from compute_as_a_teacher.evaluation import baseline, cli  # noqa: E402
from compute_as_a_teacher.evaluation.backend import (  # noqa: E402
    BackendDescriptor,
    GenerationBackend,
    execute_plan,
)
from compute_as_a_teacher.evaluation.config import (  # noqa: E402
    MATH500_PROTOCOL_VERSION,
    PromptSpec,
    RawEvalConfig,
    SynthesisEvalConfig,
)
from compute_as_a_teacher.evaluation.errors import EvaluationError  # noqa: E402
from compute_as_a_teacher.evaluation.openai_backend import (  # noqa: E402
    SUPPORTED_SAMPLING_FIELDS,
)
from compute_as_a_teacher.evaluation.planning import (  # noqa: E402
    load_plan,
    write_raw_plan,
)
from compute_as_a_teacher.evaluation.prompts import load_prompt  # noqa: E402
from compute_as_a_teacher.evaluation.schemas import (  # noqa: E402
    BackendOutput,
    GenerationRequest,
    ModelSpec,
    SamplingSpec,
)
from compute_as_a_teacher.training.errors import TrainingError  # noqa: E402


def fixed_model() -> ModelSpec:
    return ModelSpec(
        provider="fake-backend",
        model_id="fixture-model-v1",
        revision="a" * 40,
        tokenizer_id="fixture-tokenizer-v1",
        tokenizer_revision="b" * 40,
        chat_template_sha256="c" * 64,
        adapter_version="fake-adapter-v1",
        dtype="float32",
        quantization="none",
        seed_support="strict",
    )


def fixed_sampling(base_seed: int) -> SamplingSpec:
    return SamplingSpec(
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        max_new_tokens=1536,
        num_beams=1,
        repetition_penalty=1.0,
        stop=(),
        base_seed=base_seed,
    )


def configs(questions_path: Path) -> tuple[RawEvalConfig, SynthesisEvalConfig]:
    model = fixed_model()
    raw = RawEvalConfig(
        schema_version=1,
        kind="raw",
        protocol_version=MATH500_PROTOCOL_VERSION,
        run_name="fixture-raw",
        questions_path=str(questions_path),
        dataset_lock_path="unused.lock.json",
        rollouts_per_problem=8,
        prompt=PromptSpec(
            path="prompts/math500/solve_v1.txt",
            version="raw_math500_local_v1",
            prefix="",
        ),
        model=model,
        sampling=fixed_sampling(1729),
    )
    synthesis = SynthesisEvalConfig(
        schema_version=2,
        kind="synthesis",
        protocol_version=MATH500_PROTOCOL_VERSION,
        run_name="fixture-synthesis",
        required_rollouts=8,
        anchor_relation="same_as_raw",
        prompt=PromptSpec(
            path="prompts/math500/synthesis_cot_appendix_f_literal.txt",
            version="paper_appendix_f_cot_literal_v1",
            prefix="",
        ),
        anchor=model,
        sampling=fixed_sampling(2718),
    )
    return raw, synthesis


class FakeBackend(GenerationBackend):
    def __init__(
        self,
        model: ModelSpec,
        *,
        missing_box: bool = False,
        response_text: str | None = None,
        wrong_response_model: bool = False,
        finish_reason: str = "stop",
    ) -> None:
        self.calls: list[GenerationRequest] = []
        self.missing_box = missing_box
        self.response_text = response_text
        self.wrong_response_model = wrong_response_model
        self.finish_reason = finish_reason
        self._descriptor = BackendDescriptor(
            name=model.provider,
            version="fake-v1",
            model=model,
            supported_sampling_fields=SUPPORTED_SAMPLING_FIELDS,
            non_reportable=True,
        )

    @property
    def descriptor(self) -> BackendDescriptor:
        return self._descriptor

    def generate_batch(
        self,
        requests: Sequence[GenerationRequest],
    ) -> Sequence[BackendOutput]:
        self.calls.extend(requests)
        outputs = []
        for request in requests:
            text = (
                self.response_text
                if self.response_text is not None
                else (
                    "missing final answer"
                    if self.missing_box
                    else r"reasoning \boxed{42}"
                )
            )
            response_model = (
                "wrong-model" if self.wrong_response_model else request.model.model_id
            )
            outputs.append(
                BackendOutput(
                    task_id=request.task_id,
                    request_fingerprint=request.request_fingerprint,
                    text=text,
                    finish_reason=self.finish_reason,
                    prompt_tokens=10,
                    completion_tokens=5,
                    provider_metadata={"response_model": response_model},
                )
            )
        return outputs


class BaselineSequencerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.questions = [
            QuestionRecord(f"fixture-{index}", f"What is {index} plus one?")
            for index in range(3)
        ]

    def _fixture_raw_writer(self, *args, **kwargs):
        return write_raw_plan(*args, **kwargs, allow_test_fixture=True)

    def _run(
        self,
        root: Path,
        backend: FakeBackend,
        *,
        canary_results: int = 16,
        synthesis_canary_results: int = 1,
        pilot: bool = True,
        preregistration_error: TrainingError | None = None,
    ):
        questions_path = root / "questions.jsonl"
        questions_path.write_text(
            "".join(
                f'{{"id":"{question.id}","problem":"{question.problem}"}}\n'
                for question in self.questions
            ),
            encoding="utf-8",
        )
        raw_config, synthesis_config = configs(questions_path)
        preregistration = {
            "stages": {
                "initial_raw": {"run_dir": str((root / "raw-run").resolve())},
                "initial_synthesis_config": {
                    "path": str((root / "synthesis.toml").resolve())
                },
            }
        }
        verify_kwargs = (
            {"side_effect": preregistration_error}
            if preregistration_error is not None
            else {"return_value": preregistration}
        )
        with (
            patch.object(baseline, "load_raw_config", return_value=raw_config),
            patch.object(
                baseline,
                "load_synthesis_config",
                return_value=synthesis_config,
            ),
            patch.object(
                baseline,
                "load_locked_questions",
                return_value=self.questions,
            ),
            patch.object(
                baseline,
                "write_raw_plan",
                side_effect=self._fixture_raw_writer,
            ),
            patch.object(
                baseline.OpenAICompatibleBackend,
                "from_environment",
                return_value=backend,
            ) as endpoint,
            patch(
                "compute_as_a_teacher.training.experiment_registry."
                "verify_preregistered_training_stage",
                **verify_kwargs,
            ),
        ):
            result = baseline.run_baseline_sequence(
                repository_root=REPOSITORY_ROOT,
                raw_config_path=root / "raw.toml",
                synthesis_config_path=root / "synthesis.toml",
                pilot=pilot,
                preregistration_path=(
                    None if pilot else root / "preregistration.json"
                ),
                training_run_dir=None if pilot else root / "training-run",
                raw_run_dir=root / "raw-run",
                synthesis_run_dir=root / "synthesis-run",
                base_url="http://unused.invalid/v1",
                workers=1,
                batch_size=4,
                canary_results=canary_results,
                synthesis_canary_results=synthesis_canary_results,
            )
        endpoint.assert_called_once()
        return result

    def test_resumes_raw_then_gates_synthesis_and_completes_both(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            questions_path = root / "questions.jsonl"
            questions_path.write_text(
                "".join(
                    f'{{"id":"{question.id}","problem":"{question.problem}"}}\n'
                    for question in self.questions
                ),
                encoding="utf-8",
            )
            raw_config, _ = configs(questions_path)
            raw_run = root / "raw-run"
            write_raw_plan(
                raw_run,
                self.questions,
                raw_config,
                load_prompt(REPOSITORY_ROOT, raw_config.prompt),
                questions_path,
                allow_test_fixture=True,
            )
            _, requests = load_plan(raw_run, expected_kind="raw")
            backend = FakeBackend(raw_config.model)
            execute_plan(raw_run, backend, batch_size=2, max_requests=5)
            completed_ids = {request.task_id for request in requests[:5]}
            backend.calls.clear()

            result = self._run(root, backend)

            sequence_ids = [request.task_id for request in backend.calls]
            self.assertTrue(completed_ids.isdisjoint(sequence_ids))
            self.assertEqual(
                [request.stage for request in backend.calls],
                ["raw"] * 19 + ["synthesis"] * 3,
            )
            self.assertEqual(result["raw"]["completed_requests"], 24)
            self.assertEqual(result["raw"]["canary"]["audited_results"], 16)
            self.assertEqual(result["synthesis"]["completed_requests"], 3)
            self.assertEqual(
                result["synthesis"]["canary"]["audited_results"],
                1,
            )
            self.assertFalse(result["labels_loaded"])
            self.assertFalse(result["scored"])
            self.assertTrue(result["pilot"])
            self.assertEqual(result["registration_mode"], "pilot")
            self.assertFalse(result["preregistration_verified"])
            self.assertFalse(result["reportable"])
            self.assertIn(
                "pilot_without_preregistration",
                result["non_reportable_reasons"],
            )
            self.assertFalse(result["raw"]["reportable"])
            self.assertFalse(result["synthesis"]["reportable"])
            self.assertIn(
                "raw_dependency_is_non_reportable",
                result["synthesis"]["non_reportable_reasons"],
            )

    def test_canary_failures_stop_before_synthesis(self) -> None:
        cases = (
            (FakeBackend(fixed_model(), wrong_response_model=True), "response model"),
            (
                FakeBackend(fixed_model(), finish_reason="content_filter"),
                "finish_reason",
            ),
        )
        for backend, message in cases:
            with (
                self.subTest(message=message),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                with self.assertRaisesRegex(EvaluationError, message):
                    self._run(root, backend, canary_results=2)
                self.assertFalse((root / "synthesis-run" / "manifest.json").exists())

    def test_unextractable_canaries_are_preserved_and_counted(self) -> None:
        cases = (
            ("missing final answer", "missing_box"),
            (r"\boxed{42", "malformed_box"),
            (r"\boxed{  }", "empty_box"),
            (r"\boxed{" + "x" * 50_001 + "}", "too_long"),
        )
        for response_text, status in cases:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as temporary:
                backend = FakeBackend(
                    fixed_model(),
                    response_text=response_text,
                    finish_reason="length",
                )
                result = self._run(Path(temporary), backend, canary_results=2)

                self.assertEqual(result["raw"]["canary"]["boxed_outputs"], 0)
                self.assertEqual(
                    result["raw"]["canary"]["extraction_status_counts"],
                    {status: 2},
                )
                self.assertEqual(
                    result["raw"]["canary"]["finish_reasons"],
                    {"length": 2},
                )
                self.assertEqual(
                    result["synthesis"]["canary"]["boxed_outputs"], 0
                )
                self.assertEqual(
                    result["synthesis"]["canary"]["extraction_status_counts"],
                    {status: 1},
                )
                self.assertEqual(len(backend.calls), 27)

    def test_preregistration_failure_dispatches_no_request(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            backend = FakeBackend(fixed_model())
            with self.assertRaisesRegex(EvaluationError, "preregistration"):
                self._run(
                    root,
                    backend,
                    pilot=False,
                    preregistration_error=TrainingError("mismatched stage"),
                )
            self.assertEqual(backend.calls, [])

    def test_canonical_mode_verifies_preregistration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = self._run(
                Path(temporary),
                FakeBackend(fixed_model()),
                pilot=False,
            )
        self.assertFalse(result["pilot"])
        self.assertTrue(result["preregistration_verified"])
        self.assertEqual(result["registration_mode"], "canonical_preregistered")

    def test_invalid_synthesis_canary_dispatches_no_request(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            backend = FakeBackend(fixed_model())
            with self.assertRaisesRegex(EvaluationError, "positive integer"):
                self._run(
                    Path(temporary),
                    backend,
                    synthesis_canary_results=0,
                )
            self.assertEqual(backend.calls, [])

    def test_synthesis_canary_executes_and_audits_sixteen_results(self) -> None:
        self.questions = [
            QuestionRecord(f"fixture-{index}", f"What is {index} plus one?")
            for index in range(16)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            backend = FakeBackend(fixed_model())
            result = self._run(
                Path(temporary),
                backend,
                synthesis_canary_results=16,
            )
        synthesis_calls = [
            request for request in backend.calls if request.stage == "synthesis"
        ]
        self.assertEqual(len(synthesis_calls), 16)
        self.assertEqual(result["synthesis"]["canary"]["audited_results"], 16)
        self.assertEqual(result["synthesis"]["canary"]["boxed_outputs"], 16)
        self.assertEqual(
            result["synthesis"]["canary"]["extraction_status_counts"],
            {"ok": 16},
        )

    def test_label_derived_artifact_fails_before_config_or_endpoint(self) -> None:
        for artifact_name in ("scores.jsonl", "paired_scores.jsonl"):
            with self.subTest(artifact_name=artifact_name):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    raw_run = root / "raw-run"
                    raw_run.mkdir()
                    (raw_run / artifact_name).write_text("{}\n", encoding="utf-8")
                    with (
                        patch.object(baseline, "load_raw_config") as load_config,
                        self.assertRaisesRegex(
                            EvaluationError, "label-derived artifacts"
                        ),
                    ):
                        baseline.run_baseline_sequence(
                            repository_root=REPOSITORY_ROOT,
                            raw_config_path=root / "raw.toml",
                            synthesis_config_path=root / "synthesis.toml",
                            raw_run_dir=raw_run,
                            synthesis_run_dir=root / "synthesis-run",
                            base_url="http://unused.invalid/v1",
                            pilot=True,
                        )
                    load_config.assert_not_called()

    def test_cli_exposes_one_gpu_defaults(self) -> None:
        args = cli.build_parser().parse_args(
            [
                "run-baseline",
                "--raw-config",
                "raw.toml",
                "--synthesis-config",
                "synthesis.toml",
                "--pilot",
                "--raw-run-dir",
                "raw-run",
                "--synthesis-run-dir",
                "synthesis-run",
                "--base-url",
                "http://127.0.0.1:8000/v1",
            ]
        )
        self.assertEqual(args.workers, 8)
        self.assertEqual(args.canary_results, 16)
        self.assertEqual(args.synthesis_canary_results, 16)


if __name__ == "__main__":
    unittest.main()
