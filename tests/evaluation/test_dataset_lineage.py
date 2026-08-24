from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compute_as_a_teacher.evaluation import scoring as scoring_module  # noqa: E402
from compute_as_a_teacher.evaluation.artifacts import (  # noqa: E402
    file_digest,
    publish_json,
    read_json,
)
from compute_as_a_teacher.data.math500 import load_locked_questions  # noqa: E402
from compute_as_a_teacher.evaluation.backend import execute_plan  # noqa: E402
from compute_as_a_teacher.evaluation.config import (  # noqa: E402
    MATH500_PROTOCOL_VERSION,
    ScoringConfig,
)
from compute_as_a_teacher.evaluation.errors import EvaluationError  # noqa: E402
from compute_as_a_teacher.evaluation.planning import (  # noqa: E402
    write_raw_plan,
    write_synthesis_plan,
)
from compute_as_a_teacher.evaluation.prompts import load_prompt  # noqa: E402
from compute_as_a_teacher.evaluation.scoring import (  # noqa: E402
    SCORING_MANIFEST_NAME,
    score_test_fixture_run,
)
import test_pipeline as pipeline_fixtures  # noqa: E402


def locked_scoring_config() -> ScoringConfig:
    return ScoringConfig(
        schema_version=2,
        kind="scoring",
        protocol_version=MATH500_PROTOCOL_VERSION,
        labels_path="data/math500/labels.jsonl",
        dataset_lock_path="configs/datasets/math500.lock.json",
        raw_baseline_selection="sha256_uniform_per_question_v1",
        raw_baseline_seed=1729,
        primary_grader="last_boxed_string_exact_v1",
        diagnostic_graders=(),
        parsing_timeout_seconds=5,
        max_answer_chars=1000,
    )


def raw_manifest_for_lineage(
    lineage: dict[str, object],
    *,
    dataset_lock_path: str = "configs/datasets/math500.lock.json",
    questions: dict[str, object] | None = None,
    dataset_lock: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "kind": "raw",
        "protocol_version": MATH500_PROTOCOL_VERSION,
        "counts": {"problems": 500},
        "inputs": {
            "questions": questions or lineage["questions"],
            "dataset_lock": dataset_lock or lineage["dataset_lock"],
        },
        "config": {
            "dataset_lock_path": dataset_lock_path,
            "questions_path": "data/math500/questions.jsonl",
        },
        "label_firewall": {"locked_questions_verified": True},
    }


class DatasetLineageTests(unittest.TestCase):
    def _locked_scoring_data(self) -> scoring_module.LockedScoringData:
        labels_path = REPOSITORY_ROOT / "data/math500/labels.jsonl"
        questions_path = REPOSITORY_ROOT / "data/math500/questions.jsonl"
        if not labels_path.is_file() or not questions_path.is_file():
            self.skipTest("Run scripts/prepare_math500.py for locked-lineage checks")
        return scoring_module._load_locked_scoring_data(
            REPOSITORY_ROOT,
            locked_scoring_config(),
        )

    def test_raw_plan_persists_the_historical_dataset_lock_digest(self) -> None:
        questions_path = REPOSITORY_ROOT / "data/math500/questions.jsonl"
        lock_path = REPOSITORY_ROOT / "configs/datasets/math500.lock.json"
        if not questions_path.is_file():
            self.skipTest("Run scripts/prepare_math500.py for the locked-plan check")
        questions = load_locked_questions(questions_path, lock_path=lock_path)
        config = replace(
            pipeline_fixtures.raw_config(),
            run_name="locked-dataset-plan-fixture",
            questions_path="data/math500/questions.jsonl",
            dataset_lock_path="configs/datasets/math500.lock.json",
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest = write_raw_plan(
                Path(temporary_directory) / "raw",
                questions,
                config,
                load_prompt(REPOSITORY_ROOT, config.prompt),
                questions_path,
                repository_root=REPOSITORY_ROOT,
            )
        digest, byte_count = file_digest(lock_path)
        self.assertEqual(
            manifest["inputs"]["dataset_lock"],
            {
                "path": "configs/datasets/math500.lock.json",
                "sha256": digest,
                "bytes": byte_count,
            },
        )

    def test_locked_boundary_binds_exact_lock_questions_and_labels(self) -> None:
        data = self._locked_scoring_data()
        lineage = dict(data.dataset_lineage)
        self.assertEqual(len(data.labels), 500)
        self.assertEqual(lineage["verification"], "locked_dataset")
        self.assertEqual(
            set(lineage["dataset_lock"]),
            {"path", "sha256", "bytes"},
        )
        self.assertEqual(
            set(lineage["questions"]),
            {"path", "sha256", "bytes", "rows"},
        )
        self.assertEqual(
            set(lineage["labels"]),
            {"path", "sha256", "bytes", "rows"},
        )
        bound = scoring_module._bind_dataset_lineage_to_raw_plan(
            lineage,
            raw_manifest_for_lineage(lineage),
            locked_scoring_config(),
            expected_label_rows=500,
        )
        self.assertEqual(bound, lineage)

    def test_cross_lock_is_rejected_even_when_question_artifact_aligns(self) -> None:
        data = self._locked_scoring_data()
        lineage = dict(data.dataset_lineage)
        manifest = raw_manifest_for_lineage(
            lineage,
            dataset_lock_path="configs/datasets/different-but-aligned.lock.json",
        )
        with self.assertRaisesRegex(EvaluationError, "same exact dataset lock"):
            scoring_module._bind_dataset_lineage_to_raw_plan(
                lineage,
                manifest,
                locked_scoring_config(),
                expected_label_rows=500,
            )

    def test_historical_lock_digest_must_match_current_scoring_lock(self) -> None:
        data = self._locked_scoring_data()
        lineage = dict(data.dataset_lineage)
        changed_lock = dict(lineage["dataset_lock"])
        changed_lock["sha256"] = "0" * 64
        manifest = raw_manifest_for_lineage(
            lineage,
            dataset_lock=changed_lock,
        )
        with self.assertRaisesRegex(EvaluationError, "dataset-lock artifact"):
            scoring_module._bind_dataset_lineage_to_raw_plan(
                lineage,
                manifest,
                locked_scoring_config(),
                expected_label_rows=500,
            )

    def test_cross_question_spec_is_rejected_even_when_rows_and_ids_align(self) -> None:
        data = self._locked_scoring_data()
        lineage = dict(data.dataset_lineage)
        changed_questions = dict(lineage["questions"])
        changed_questions["sha256"] = "0" * 64
        manifest = raw_manifest_for_lineage(
            lineage,
            questions=changed_questions,
        )
        with self.assertRaisesRegex(EvaluationError, "questions do not match"):
            scoring_module._bind_dataset_lineage_to_raw_plan(
                lineage,
                manifest,
                locked_scoring_config(),
                expected_label_rows=500,
            )

    def test_synthesis_uses_raw_generation_lineage_not_raw_scores(self) -> None:
        pipeline = pipeline_fixtures.EvaluationPipelineTests()
        pipeline.setUp()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw_run = pipeline._write_raw_fixture_plan(root)
            execute_plan(
                raw_run,
                pipeline_fixtures.ScriptedFakeBackend(
                    pipeline_fixtures.fixed_model()
                ),
            )
            score_test_fixture_run(
                raw_run,
                pipeline.labels,
                pipeline.labels_reference,
                pipeline_fixtures.scoring_config(),
            )
            raw_scoring_path = raw_run / SCORING_MANIFEST_NAME
            raw_scoring = read_json(raw_scoring_path)
            self.assertEqual(raw_scoring["schema_version"], 3)
            self.assertFalse(raw_scoring["reportable"])
            self.assertEqual(
                raw_scoring["dataset_lineage"]["verification"],
                "test_fixture",
            )
            self.assertIsNone(raw_scoring["dataset_lineage"]["dataset_lock"])
            for name in ("questions", "labels"):
                self.assertEqual(
                    set(raw_scoring["dataset_lineage"][name]),
                    {"path", "sha256", "bytes", "rows"},
                )

            synthesis_run = root / "synthesis"
            write_synthesis_plan(
                synthesis_run,
                raw_run,
                pipeline_fixtures.synthesis_config(),
                pipeline.synthesis_template,
            )
            execute_plan(
                synthesis_run,
                pipeline_fixtures.ScriptedFakeBackend(
                    pipeline_fixtures.fixed_model()
                ),
            )

            changed = copy.deepcopy(raw_scoring)
            changed["dataset_lineage"]["questions"]["sha256"] = "0" * 64
            changed["scoring_fingerprint"] = (
                scoring_module._scoring_manifest_fingerprint(changed)
            )
            publish_json(raw_scoring_path, changed, force=True)
            summary = score_test_fixture_run(
                synthesis_run,
                pipeline.labels,
                pipeline.labels_reference,
                pipeline_fixtures.scoring_config(),
                raw_run_dir=raw_run,
            )
            self.assertEqual(summary["counts"]["raw_baseline_generations_scored"], 2)
            synthesis_scoring = read_json(
                synthesis_run / SCORING_MANIFEST_NAME
            )
            self.assertEqual(
                set(synthesis_scoring["inputs"]),
                {"generations", "execution", "raw_generations", "raw_execution"},
            )

    def test_score_run_verifies_execution_before_crossing_label_boundary(self) -> None:
        events: list[str] = []
        dummy_data = scoring_module.LockedScoringData(
            labels=(),
            labels_reference={},
            dataset_lineage={},
        )

        def verify(*args: object, **kwargs: object) -> tuple[dict, list]:
            events.append("verify_execution")
            return {}, []

        def load_labels(*args: object, **kwargs: object) -> object:
            events.append("load_labels")
            return dummy_data

        def score(*args: object, **kwargs: object) -> dict[str, bool]:
            events.append("score")
            return {"reportable": False}

        with (
            patch.object(scoring_module, "load_plan", return_value=({}, [])),
            patch.object(
                scoring_module,
                "verify_complete_execution",
                side_effect=verify,
            ),
            patch.object(
                scoring_module,
                "_load_locked_scoring_data",
                side_effect=load_labels,
            ),
            patch.object(
                scoring_module,
                "_score_run_with_labels",
                side_effect=score,
            ),
        ):
            scoring_module.score_run(
                Path("unused-run"),
                locked_scoring_config(),
                repository_root=REPOSITORY_ROOT,
            )
        self.assertEqual(events, ["verify_execution", "load_labels", "score"])


if __name__ == "__main__":
    unittest.main()
