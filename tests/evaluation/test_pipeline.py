from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from compute_as_a_teacher.data.math500 import QuestionRecord  # noqa: E402
from compute_as_a_teacher.evaluation.artifacts import (  # noqa: E402
    publish_json,
    publish_jsonl,
    read_json,
    read_jsonl,
    sha256_text,
)
from compute_as_a_teacher.evaluation.backend import (  # noqa: E402
    BackendDescriptor,
    GenerationBackend,
    execute_plan,
    ingest_responses,
)
from compute_as_a_teacher.evaluation.config import (  # noqa: E402
    MATH500_PROTOCOL_VERSION,
    PromptSpec,
    RawEvalConfig,
    ScoringConfig,
    SynthesisEvalConfig,
)
from compute_as_a_teacher.evaluation.errors import EvaluationError  # noqa: E402
from compute_as_a_teacher.evaluation.grading import PRIMARY_GRADER  # noqa: E402
from compute_as_a_teacher.evaluation.planning import (  # noqa: E402
    GENERATIONS_NAME,
    build_raw_requests,
    build_synthesis_requests,
    load_plan,
    write_raw_plan,
    write_synthesis_plan,
)
from compute_as_a_teacher.evaluation.prompts import load_prompt  # noqa: E402
from compute_as_a_teacher.evaluation.schemas import (  # noqa: E402
    BackendOutput,
    GenerationRequest,
    GenerationResult,
    ModelSpec,
    SamplingSpec,
)
from compute_as_a_teacher.evaluation.scoring import (  # noqa: E402
    EvalLabel,
    score_generation_rows,
    score_test_fixture_run,
)


LABEL_SENTINEL = "TOP_SECRET_REFERENCE_SOLUTION"
SAMPLING_FIELDS = frozenset(
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


def qwen_model() -> ModelSpec:
    return ModelSpec(
        provider="fake-backend",
        model_id="Qwen/Qwen3-4B",
        revision="1" * 40,
        tokenizer_id="Qwen/Qwen3-4B",
        tokenizer_revision="2" * 40,
        chat_template_sha256="3" * 64,
        adapter_version="fake-adapter-v1",
        dtype="bfloat16",
        quantization="none",
        seed_support="best_effort",
    )


def fixed_sampling() -> SamplingSpec:
    return SamplingSpec(
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        max_new_tokens=1536,
        num_beams=1,
        repetition_penalty=1.0,
        stop=(),
        base_seed=1729,
    )


def raw_config(model: ModelSpec | None = None) -> RawEvalConfig:
    return RawEvalConfig(
        schema_version=1,
        kind="raw",
        protocol_version=MATH500_PROTOCOL_VERSION,
        run_name="fixture-raw",
        questions_path="fixture/questions.jsonl",
        dataset_lock_path="fixture/unused.lock.json",
        rollouts_per_problem=8,
        prompt=PromptSpec(
            path="prompts/math500/solve_v1.txt",
            version="raw_math500_local_v1",
            prefix="",
        ),
        model=model or fixed_model(),
        sampling=fixed_sampling(),
    )


def synthesis_config(model: ModelSpec | None = None) -> SynthesisEvalConfig:
    return SynthesisEvalConfig(
        schema_version=1,
        kind="synthesis",
        protocol_version=MATH500_PROTOCOL_VERSION,
        run_name="fixture-synthesis",
        required_rollouts=8,
        require_same_model_as_raw=True,
        prompt=PromptSpec(
            path="prompts/math500/synthesis_cot_v1.txt",
            version="paper_appendix_f_cot_boxfix_v1",
            prefix="",
        ),
        anchor=model or fixed_model(),
        sampling=fixed_sampling(),
    )


def qwen_raw_config() -> RawEvalConfig:
    config = raw_config(qwen_model())
    return replace(config, prompt=replace(config.prompt, prefix="/no_think\n"))


def qwen_synthesis_config() -> SynthesisEvalConfig:
    config = synthesis_config(qwen_model())
    return replace(config, prompt=replace(config.prompt, prefix="/no_think\n"))


def scoring_config() -> ScoringConfig:
    return ScoringConfig(
        schema_version=1,
        kind="scoring",
        protocol_version=MATH500_PROTOCOL_VERSION,
        labels_path="fixture/labels.jsonl",
        dataset_lock_path="fixture/unused.lock.json",
        primary_grader=PRIMARY_GRADER,
        diagnostic_graders=(),
        parsing_timeout_seconds=5,
        max_answer_chars=1000,
    )


class ScriptedFakeBackend(GenerationBackend):
    def __init__(
        self,
        model: ModelSpec,
        *,
        supported_fields: frozenset[str] = SAMPLING_FIELDS,
        reverse_outputs: bool = True,
    ) -> None:
        self.calls: list[GenerationRequest] = []
        self.reverse_outputs = reverse_outputs
        self._descriptor = BackendDescriptor(
            name="fake-backend",
            version="fake-v1",
            model=model,
            supported_sampling_fields=supported_fields,
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
        outputs: list[BackendOutput] = []
        for request in requests:
            if request.stage == "synthesis":
                answer = "2" if request.question_id == "fixture-alpha" else "3"
            elif request.question_id == "fixture-alpha":
                answer = "2" if int(request.rollout_index) < 5 else "9"
            else:
                answer = "3" if int(request.rollout_index) in {1, 2, 3} else (
                    "0" if request.rollout_index == 0 else "4"
                )
            outputs.append(
                BackendOutput(
                    task_id=request.task_id,
                    request_fingerprint=request.request_fingerprint,
                    text=f"Synthetic test reasoning. Therefore \\boxed{{{answer}}}.",
                    finish_reason="stop",
                    prompt_tokens=10,
                    completion_tokens=5,
                    provider_metadata={"fixture": True},
                )
            )
        return list(reversed(outputs)) if self.reverse_outputs else outputs


class EvaluationPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.questions = [
            QuestionRecord("fixture-alpha", "QUESTION_ALPHA: What is one plus one?"),
            QuestionRecord("fixture-beta", "QUESTION_BETA: What is six divided by two?"),
        ]
        self.labels = [
            EvalLabel("fixture-alpha", "2", "Algebra", 1),
            EvalLabel("fixture-beta", "3", "Prealgebra", 2),
        ]
        self.labels_reference = {
            "path": "fixture/labels.jsonl",
            "sha256": "d" * 64,
            "bytes": 123,
            "rows": 2,
        }
        self.raw_template = load_prompt(
            REPOSITORY_ROOT,
            raw_config().prompt,
        )
        self.synthesis_template = load_prompt(
            REPOSITORY_ROOT,
            synthesis_config().prompt,
        )

    def _write_raw_fixture_plan(self, root: Path) -> Path:
        questions_path = root / "questions.jsonl"
        publish_jsonl(
            questions_path,
            ({"id": row.id, "problem": row.problem} for row in self.questions),
        )
        raw_run = root / "raw"
        write_raw_plan(
            raw_run,
            self.questions,
            raw_config(),
            self.raw_template,
            questions_path,
            allow_test_fixture=True,
        )
        return raw_run

    def test_fake_backend_full_raw_and_synthesis_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw_run = self._write_raw_fixture_plan(root)
            raw_manifest, raw_requests = load_plan(raw_run, expected_kind="raw")
            self.assertEqual(raw_manifest["counts"]["requests"], 16)
            self.assertEqual(
                [request.task_id for request in raw_requests],
                [
                    request.task_id
                    for request in build_raw_requests(
                        self.questions,
                        raw_config(),
                        self.raw_template,
                    )
                ],
            )
            for request in raw_requests:
                message = request.messages[0]["content"]
                self.assertNotIn(LABEL_SENTINEL, message)

            raw_backend = ScriptedFakeBackend(fixed_model())
            partial = execute_plan(
                raw_run,
                raw_backend,
                batch_size=2,
                max_requests=3,
            )
            self.assertFalse(partial["complete"])
            first_result_path = raw_run / "results" / f"{raw_requests[0].task_id}.json"
            first_result_bytes = first_result_path.read_bytes()
            completed = execute_plan(raw_run, raw_backend, batch_size=5)
            self.assertTrue(completed["complete"])
            self.assertEqual(len(raw_backend.calls), 16)
            self.assertEqual(first_result_path.read_bytes(), first_result_bytes)
            self.assertTrue(completed["non_reportable"])

            raw_summary = score_test_fixture_run(
                raw_run,
                self.labels,
                self.labels_reference,
                scoring_config(),
            )
            raw_metrics = raw_summary["graders"][PRIMARY_GRADER]
            self.assertEqual(raw_metrics["rollout_index_0_accuracy"], 0.5)
            self.assertEqual(raw_metrics["mean_rollout_accuracy"], 0.5)
            self.assertEqual(raw_metrics["empirical_any_correct_at_8"], 1.0)
            self.assertEqual(raw_metrics["literal_plurality_vote_accuracy"], 0.5)
            self.assertFalse(raw_summary["reportable"])

            synthesis_run = root / "synthesis"
            write_synthesis_plan(
                synthesis_run,
                raw_run,
                synthesis_config(),
                self.synthesis_template,
            )
            _, synthesis_requests = load_plan(
                synthesis_run,
                expected_kind="synthesis",
            )
            self.assertEqual(len(synthesis_requests), 2)
            for request in synthesis_requests:
                content = request.messages[0]["content"]
                self.assertNotIn("QUESTION_ALPHA", content)
                self.assertNotIn("QUESTION_BETA", content)
                self.assertNotIn(LABEL_SENTINEL, content)
                self.assertEqual(content.count("## RESPONSE "), 8)
                self.assertEqual(len(request.source_task_ids), 8)

            synthesis_backend = ScriptedFakeBackend(fixed_model())
            execution = execute_plan(synthesis_run, synthesis_backend, batch_size=2)
            self.assertTrue(execution["complete"])
            self.assertEqual(len(synthesis_backend.calls), 2)
            synthesis_summary = score_test_fixture_run(
                synthesis_run,
                self.labels,
                self.labels_reference,
                scoring_config(),
                raw_run_dir=raw_run,
            )
            metrics = synthesis_summary["graders"][PRIMARY_GRADER]
            self.assertEqual(metrics["synthesis_accuracy"], 1.0)
            self.assertEqual(metrics["paired_delta_vs_raw_index_0"], 0.5)
            self.assertEqual(metrics["paired_delta_vs_raw_mean"], 0.5)
            self.assertFalse(synthesis_summary["reportable"])
            self.assertTrue(
                any(
                    reason.startswith("raw_dependency:")
                    for reason in synthesis_summary["non_reportable_reasons"]
                )
            )

    def test_shuffled_verified_raw_rows_produce_same_synthesis_requests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw_run = self._write_raw_fixture_plan(root)
            execute_plan(raw_run, ScriptedFakeBackend(fixed_model()))
            rows = read_jsonl(raw_run / GENERATIONS_NAME)
            forward = build_synthesis_requests(
                rows,
                synthesis_config(),
                self.synthesis_template,
            )
            reverse = build_synthesis_requests(
                list(reversed(rows)),
                synthesis_config(),
                self.synthesis_template,
            )
            self.assertEqual(
                [request.to_dict() for request in forward],
                [request.to_dict() for request in reverse],
            )

    def test_unsupported_sampling_field_fails_before_backend_call(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            raw_run = self._write_raw_fixture_plan(Path(temporary_directory))
            backend = ScriptedFakeBackend(
                fixed_model(),
                supported_fields=SAMPLING_FIELDS - {"top_k"},
            )
            with self.assertRaisesRegex(EvaluationError, "top_k"):
                execute_plan(raw_run, backend)
            self.assertEqual(backend.calls, [])

    def test_request_and_cached_result_tampering_fail_closed(self) -> None:
        request = build_raw_requests(
            self.questions,
            raw_config(),
            self.raw_template,
        )[0]
        tampered_request = request.to_dict()
        tampered_request["messages"][0]["content"] += " altered"
        with self.assertRaisesRegex(EvaluationError, "fingerprint"):
            GenerationRequest.from_dict(tampered_request)

        with tempfile.TemporaryDirectory() as temporary_directory:
            raw_run = self._write_raw_fixture_plan(Path(temporary_directory))
            backend = ScriptedFakeBackend(fixed_model())
            execute_plan(raw_run, backend, max_requests=1)
            _, requests = load_plan(raw_run)
            result_path = raw_run / "results" / f"{requests[0].task_id}.json"
            result = read_json(result_path)
            result["text"] += " tampered"
            publish_json(result_path, result, force=True)
            with self.assertRaisesRegex(EvaluationError, "output hash mismatch"):
                execute_plan(raw_run, backend)

    def test_external_ingest_is_non_reportable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw_run = self._write_raw_fixture_plan(root)
            _, requests = load_plan(raw_run)
            response_rows = []
            for request in requests:
                text = r"fixture \boxed{0}"
                response_rows.append(
                    GenerationResult(
                        task_id=request.task_id,
                        request_fingerprint=request.request_fingerprint,
                        backend_fingerprint="e" * 64,
                        text=text,
                        output_sha256=sha256_text(text),
                        finish_reason="stop",
                        prompt_tokens=None,
                        completion_tokens=None,
                        provider_metadata={},
                    ).to_dict()
                )
            responses_path = root / "responses.jsonl"
            publish_jsonl(responses_path, response_rows)
            execution = ingest_responses(raw_run, responses_path)
            self.assertTrue(execution["complete"])
            self.assertTrue(execution["non_reportable"])
            self.assertIn(
                "unattested_external_response_ingest",
                execution["non_reportable_reasons"],
            )

    def test_reference_values_are_not_serialized_into_score_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            raw_run = self._write_raw_fixture_plan(Path(temporary_directory))
            execute_plan(raw_run, ScriptedFakeBackend(fixed_model()))
            generations = read_jsonl(raw_run / GENERATIONS_NAME)
            secret_labels = [
                EvalLabel("fixture-alpha", f"{LABEL_SENTINEL}_A", "Algebra", 1),
                EvalLabel("fixture-beta", f"{LABEL_SENTINEL}_B", "Prealgebra", 2),
            ]
            scores = score_generation_rows(
                generations,
                secret_labels,
                scoring_config(),
                expected_stage="raw",
            )
            self.assertNotIn(LABEL_SENTINEL, json.dumps(scores, sort_keys=True))
            self.assertNotIn("solution", EvalLabel.__dataclass_fields__)

    def test_direct_contract_bypasses_are_rejected(self) -> None:
        bad_raw = replace(raw_config(), rollouts_per_problem=7)
        with self.assertRaisesRegex(EvaluationError, "eight"):
            build_raw_requests(self.questions, bad_raw, self.raw_template)
        bad_synthesis = replace(
            synthesis_config(),
            require_same_model_as_raw=False,
        )
        with self.assertRaisesRegex(EvaluationError, "same frozen"):
            build_synthesis_requests([], bad_synthesis, self.synthesis_template)

        with self.assertRaisesRegex(EvaluationError, "registered SHA-256"):
            build_raw_requests(
                self.questions,
                raw_config(),
                self.raw_template + " ",
            )
        with self.assertRaisesRegex(EvaluationError, "registered SHA-256"):
            build_synthesis_requests(
                [],
                synthesis_config(),
                self.synthesis_template + " ",
            )
        aliased_prompt = replace(
            raw_config(),
            prompt=replace(
                raw_config().prompt,
                path="prompts/math500/synthesis_cot_v1.txt",
            ),
        )
        with self.assertRaisesRegex(EvaluationError, "must use"):
            build_raw_requests(
                self.questions,
                aliased_prompt,
                self.raw_template,
            )

    def test_qwen_paper_profile_is_enforced_without_loading_a_model(self) -> None:
        raw = qwen_raw_config()
        requests = build_raw_requests(self.questions, raw, self.raw_template)
        self.assertEqual(len(requests), 16)
        for request in requests:
            self.assertTrue(request.messages[0]["content"].startswith("/no_think\n"))
            self.assertEqual(request.sampling["temperature"], 0.7)
            self.assertEqual(request.sampling["top_p"], 0.8)
            self.assertEqual(request.sampling["top_k"], 20)
            self.assertEqual(request.sampling["max_new_tokens"], 1536)

        for field, value in (
            ("temperature", 0.6),
            ("top_p", 0.9),
            ("top_k", 19),
            ("max_new_tokens", 1024),
        ):
            with self.subTest(field=field):
                bad_sampling = replace(raw.sampling, **{field: value})
                with self.assertRaises(EvaluationError):
                    build_raw_requests(
                        self.questions,
                        replace(raw, sampling=bad_sampling),
                        self.raw_template,
                    )
        with self.assertRaisesRegex(EvaluationError, "/no_think"):
            build_raw_requests(
                self.questions,
                replace(raw, prompt=replace(raw.prompt, prefix="")),
                self.raw_template,
            )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            questions_path = root / "questions.jsonl"
            publish_jsonl(
                questions_path,
                ({"id": row.id, "problem": row.problem} for row in self.questions),
            )
            raw_run = root / "qwen-raw"
            write_raw_plan(
                raw_run,
                self.questions,
                raw,
                self.raw_template,
                questions_path,
                allow_test_fixture=True,
            )
            execute_plan(raw_run, ScriptedFakeBackend(qwen_model()))
            synthesis_run = root / "qwen-synthesis"
            write_synthesis_plan(
                synthesis_run,
                raw_run,
                qwen_synthesis_config(),
                self.synthesis_template,
            )
            _, synthesis_requests = load_plan(synthesis_run, expected_kind="synthesis")
            self.assertEqual(len(synthesis_requests), 2)
            self.assertTrue(
                synthesis_requests[0].messages[0]["content"].startswith(
                    "/no_think\n"
                )
            )

    def test_execution_reportability_and_backend_metadata_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            raw_run = self._write_raw_fixture_plan(Path(temporary_directory))
            execute_plan(raw_run, ScriptedFakeBackend(fixed_model()))
            execution_path = raw_run / "execution.json"
            original = read_json(execution_path)
            mutations = {
                "mode": lambda value: value.update(
                    {"mode": "external_response_ingest"}
                ),
                "reasons": lambda value: value.update(
                    {"non_reportable": False, "non_reportable_reasons": []}
                ),
                "backend": lambda value: value["backend"].update(
                    {"version": "tampered"}
                ),
                "seed": lambda value: value.update(
                    {"seed_reproducibility": "none"}
                ),
            }
            for name, mutate in mutations.items():
                with self.subTest(name=name):
                    changed = json.loads(json.dumps(original))
                    mutate(changed)
                    publish_json(execution_path, changed, force=True)
                    with self.assertRaises(EvaluationError):
                        score_test_fixture_run(
                            raw_run,
                            self.labels,
                            self.labels_reference,
                            scoring_config(),
                        )
            publish_json(execution_path, original, force=True)

    def test_label_reference_schema_rejects_injected_reference_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            raw_run = self._write_raw_fixture_plan(Path(temporary_directory))
            execute_plan(raw_run, ScriptedFakeBackend(fixed_model()))
            unsafe_reference = {
                **self.labels_reference,
                "solution": LABEL_SENTINEL,
            }
            with self.assertRaisesRegex(EvaluationError, "exact safe"):
                score_test_fixture_run(
                    raw_run,
                    self.labels,
                    unsafe_reference,
                    scoring_config(),
                )

    def test_replaced_raw_outputs_invalidate_stale_raw_scores(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw_run = self._write_raw_fixture_plan(root)
            execute_plan(raw_run, ScriptedFakeBackend(fixed_model()))
            score_test_fixture_run(
                raw_run,
                self.labels,
                self.labels_reference,
                scoring_config(),
            )

            _, raw_requests = load_plan(raw_run, expected_kind="raw")
            replacement_rows = []
            for request in raw_requests:
                text = r"Replacement output. Therefore \boxed{0}."
                replacement_rows.append(
                    GenerationResult(
                        task_id=request.task_id,
                        request_fingerprint=request.request_fingerprint,
                        backend_fingerprint="f" * 64,
                        text=text,
                        output_sha256=sha256_text(text),
                        finish_reason="stop",
                        prompt_tokens=None,
                        completion_tokens=None,
                        provider_metadata={"fixture": "replacement"},
                    ).to_dict()
                )
            responses_path = root / "replacement-responses.jsonl"
            publish_jsonl(responses_path, replacement_rows)
            ingest_responses(raw_run, responses_path, force=True)

            synthesis_run = root / "synthesis-after-replacement"
            write_synthesis_plan(
                synthesis_run,
                raw_run,
                synthesis_config(),
                self.synthesis_template,
            )
            execute_plan(synthesis_run, ScriptedFakeBackend(fixed_model()))
            with self.assertRaisesRegex(EvaluationError, "changed since"):
                score_test_fixture_run(
                    synthesis_run,
                    self.labels,
                    self.labels_reference,
                    scoring_config(),
                    raw_run_dir=raw_run,
                )

            score_test_fixture_run(
                raw_run,
                self.labels,
                self.labels_reference,
                scoring_config(),
                force=True,
            )
            summary = score_test_fixture_run(
                synthesis_run,
                self.labels,
                self.labels_reference,
                scoring_config(),
                raw_run_dir=raw_run,
            )
            self.assertFalse(summary["reportable"])

    def test_synthesis_scoring_rejects_a_different_raw_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw_run = self._write_raw_fixture_plan(root)
            execute_plan(raw_run, ScriptedFakeBackend(fixed_model()))
            score_test_fixture_run(
                raw_run,
                self.labels,
                self.labels_reference,
                scoring_config(),
            )
            synthesis_run = root / "synthesis"
            write_synthesis_plan(
                synthesis_run,
                raw_run,
                synthesis_config(),
                self.synthesis_template,
            )
            execute_plan(synthesis_run, ScriptedFakeBackend(fixed_model()))

            other_raw = root / "other-raw"
            questions_path = root / "other-questions.jsonl"
            publish_jsonl(
                questions_path,
                ({"id": row.id, "problem": row.problem} for row in self.questions),
            )
            other_config = replace(raw_config(), run_name="other-raw")
            write_raw_plan(
                other_raw,
                self.questions,
                other_config,
                self.raw_template,
                questions_path,
                allow_test_fixture=True,
            )
            execute_plan(other_raw, ScriptedFakeBackend(fixed_model()))
            score_test_fixture_run(
                other_raw,
                self.labels,
                self.labels_reference,
                scoring_config(),
            )
            with self.assertRaisesRegex(EvaluationError, "not derived"):
                score_test_fixture_run(
                    synthesis_run,
                    self.labels,
                    self.labels_reference,
                    scoring_config(),
                    raw_run_dir=other_raw,
                )


if __name__ == "__main__":
    unittest.main()
