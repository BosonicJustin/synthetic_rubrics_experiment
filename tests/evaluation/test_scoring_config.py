from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from compute_as_a_teacher.evaluation.config import (  # noqa: E402
    RAW_BASELINE_SELECTION_METHOD,
    load_scoring_config,
)
from compute_as_a_teacher.evaluation.errors import EvaluationError  # noqa: E402


SCORING_CONFIG = REPOSITORY_ROOT / "configs/evals/math500_scoring.toml"


class ScoringConfigTests(unittest.TestCase):
    def test_checked_in_config_locks_the_paired_raw_selector(self) -> None:
        config = load_scoring_config(SCORING_CONFIG)

        self.assertEqual(config.schema_version, 2)
        self.assertEqual(config.raw_baseline_selection, RAW_BASELINE_SELECTION_METHOD)
        self.assertEqual(config.raw_baseline_seed, 1729)
        self.assertEqual(
            config.to_dict()["raw_baseline_selection"],
            "sha256_uniform_per_question_v1",
        )

    def test_invalid_selector_contract_is_rejected(self) -> None:
        original = SCORING_CONFIG.read_text(encoding="utf-8")
        cases = (
            ("schema_version = 2", "schema_version = 1", "schema_version=2"),
            (
                'raw_baseline_selection = "sha256_uniform_per_question_v1"',
                'raw_baseline_selection = "random"',
                "raw_baseline_selection",
            ),
            ("raw_baseline_seed = 1729", "raw_baseline_seed = true", "raw_baseline_seed"),
            ("raw_baseline_seed = 1729", "raw_baseline_seed = -1", "raw_baseline_seed"),
        )
        for old, new, message in cases:
            with self.subTest(replacement=new):
                with tempfile.TemporaryDirectory() as temporary:
                    path = Path(temporary) / "scoring.toml"
                    path.write_text(original.replace(old, new, 1), encoding="utf-8")
                    with self.assertRaisesRegex(EvaluationError, message):
                        load_scoring_config(path)


if __name__ == "__main__":
    unittest.main()
