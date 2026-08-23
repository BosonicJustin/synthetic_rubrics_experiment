from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compute_as_a_teacher.evaluation import backend as backend_module  # noqa: E402
from compute_as_a_teacher.evaluation.artifacts import (  # noqa: E402
    publish_json,
    read_json,
    sha256_text,
)
from compute_as_a_teacher.evaluation.backend import execute_plan  # noqa: E402
from compute_as_a_teacher.evaluation.errors import EvaluationError  # noqa: E402
from compute_as_a_teacher.evaluation.planning import load_plan  # noqa: E402
from compute_as_a_teacher.evaluation.schemas import GenerationResult  # noqa: E402
import test_pipeline as pipeline_fixtures  # noqa: E402


class ResumeRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = pipeline_fixtures.EvaluationPipelineTests()
        self.pipeline.setUp()

    def _write_result(
        self,
        run_dir: Path,
        request_index: int,
        *,
        backend_fingerprint: str | None = None,
    ) -> None:
        _, requests = load_plan(run_dir, expected_kind="raw")
        request = requests[request_index]
        backend = pipeline_fixtures.ScriptedFakeBackend(
            pipeline_fixtures.fixed_model(),
            reverse_outputs=False,
        )
        output = backend.generate_batch([request])[0]
        text = output.text
        result = GenerationResult(
            task_id=request.task_id,
            request_fingerprint=request.request_fingerprint,
            backend_fingerprint=(
                backend_fingerprint or backend.descriptor.fingerprint
            ),
            text=text,
            output_sha256=sha256_text(text),
            finish_reason=output.finish_reason,
            prompt_tokens=output.prompt_tokens,
            completion_tokens=output.completion_tokens,
            provider_metadata=dict(output.provider_metadata or {}),
        )
        publish_json(
            run_dir / "results" / f"{request.task_id}.json",
            result.to_dict(),
        )

    def test_valid_append_only_result_beyond_stale_checkpoint_is_recovered(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = self.pipeline._write_raw_fixture_plan(
                Path(temporary_directory)
            )
            backend = pipeline_fixtures.ScriptedFakeBackend(
                pipeline_fixtures.fixed_model()
            )
            first = execute_plan(run_dir, backend, batch_size=1, max_requests=1)
            self.assertEqual(first["completed_requests"], 1)

            real_checkpoint = backend_module._checkpoint_execution
            checkpoint_calls = 0

            def fail_after_result(*args: object, **kwargs: object) -> dict[str, object]:
                nonlocal checkpoint_calls
                checkpoint_calls += 1
                if checkpoint_calls == 2:
                    raise RuntimeError("simulated checkpoint failure")
                return real_checkpoint(*args, **kwargs)

            with patch.object(
                backend_module,
                "_checkpoint_execution",
                side_effect=fail_after_result,
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated"):
                    execute_plan(run_dir, backend, batch_size=1, max_requests=1)

            stale = read_json(run_dir / "execution.json")
            self.assertEqual(stale["completed_requests"], 1)
            self.assertEqual(len(list((run_dir / "results").glob("*.json"))), 2)

            completed = execute_plan(run_dir, backend, batch_size=3)
            self.assertTrue(completed["complete"])
            self.assertEqual(completed["completed_requests"], 16)

    def test_non_prefix_cached_results_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = self.pipeline._write_raw_fixture_plan(
                Path(temporary_directory)
            )
            backend = pipeline_fixtures.ScriptedFakeBackend(
                pipeline_fixtures.fixed_model()
            )
            execute_plan(run_dir, backend, batch_size=1, max_requests=1)
            self._write_result(run_dir, 2)

            with self.assertRaisesRegex(EvaluationError, "contiguous prefix"):
                execute_plan(run_dir, backend)

    def test_mixed_backend_append_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = self.pipeline._write_raw_fixture_plan(
                Path(temporary_directory)
            )
            backend = pipeline_fixtures.ScriptedFakeBackend(
                pipeline_fixtures.fixed_model()
            )
            execute_plan(run_dir, backend, batch_size=1, max_requests=1)
            self._write_result(run_dir, 1, backend_fingerprint="f" * 64)

            with self.assertRaisesRegex(EvaluationError, "different backend"):
                execute_plan(run_dir, backend)

    def test_changed_checkpointed_result_is_rejected_even_with_a_new_text_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = self.pipeline._write_raw_fixture_plan(
                Path(temporary_directory)
            )
            backend = pipeline_fixtures.ScriptedFakeBackend(
                pipeline_fixtures.fixed_model()
            )
            execute_plan(run_dir, backend, batch_size=1, max_requests=2)
            _, requests = load_plan(run_dir, expected_kind="raw")
            result_path = run_dir / "results" / f"{requests[0].task_id}.json"
            result = read_json(result_path)
            result["text"] = r"Self-consistent replacement \boxed{999}"
            result["output_sha256"] = sha256_text(result["text"])
            publish_json(result_path, result, force=True)

            with self.assertRaisesRegex(EvaluationError, "checkpointed cached result changed"):
                execute_plan(run_dir, backend)


if __name__ == "__main__":
    unittest.main()
