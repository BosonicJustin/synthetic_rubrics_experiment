from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from compute_as_a_teacher.training import cli  # noqa: E402
from compute_as_a_teacher.training.errors import TrainingError  # noqa: E402
from compute_as_a_teacher.training.metrics import (  # noqa: E402
    summarize_rollout_logs,
)


def _row(
    step: int,
    score: float,
    anchor: str,
    *,
    rollout_status: str = "ok",
    anchor_status: str = "ok",
    finish_reason: str = "stop",
    latency: float | None = 1.0,
) -> dict[str, object]:
    row: dict[str, object] = {
        "step": step,
        "score": score,
        "rollout_extraction_status": rollout_status,
        "anchor_prompt_sha256": anchor,
        "anchor_seed": 1,
        "anchor_extraction_status": anchor_status,
        "anchor_finish_reason": finish_reason,
    }
    if latency is not None:
        row["anchor_latency_seconds"] = latency
    return row


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


class TrainingMetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temporary.name)
        self.logs = self.run_dir / "rollout_logs"
        self.logs.mkdir()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_summarizes_rollouts_and_unique_anchor_calls(self) -> None:
        rows = [
            _row(1, 1.0, "anchor-a", latency=1.0),
            _row(1, 0.0, "anchor-a", rollout_status="missing_boxed", latency=1.0),
            _row(
                1,
                1.0,
                "anchor-b",
                anchor_status="missing_boxed",
                finish_reason="length",
                latency=3.0,
            ),
            _row(
                1,
                0.0,
                "anchor-b",
                anchor_status="missing_boxed",
                finish_reason="length",
                latency=3.0,
            ),
        ]
        _write_rows(self.logs / "1.jsonl", rows)

        result = summarize_rollout_logs(self.logs)

        self.assertFalse(result["gold_labels_loaded"])
        self.assertEqual(result["latest_step"], 1)
        metrics = result["steps"][0]
        self.assertEqual(
            metrics["task_reward"],
            {
                "count": 4,
                "mean": 0.5,
                "zero_fraction": 0.5,
                "one_fraction": 0.5,
            },
        )
        self.assertEqual(
            metrics["rollout_extraction_status_counts"],
            {"missing_boxed": 1, "ok": 3},
        )
        self.assertEqual(metrics["anchor"]["count"], 2)
        self.assertEqual(
            metrics["anchor"]["extraction_status_counts"],
            {"missing_boxed": 1, "ok": 1},
        )
        self.assertEqual(
            metrics["anchor"]["finish_reason_counts"],
            {"length": 1, "stop": 1},
        )
        self.assertEqual(
            metrics["anchor"]["latency_seconds"],
            {"count": 2, "mean": 2.0, "p50": 2.0, "p95": 2.9, "max": 3.0},
        )

    def test_latest_reads_only_the_highest_step(self) -> None:
        (self.logs / "1.jsonl").write_text("not json\n", encoding="utf-8")
        _write_rows(self.logs / "2.jsonl", [_row(2, 1.0, "anchor")])

        result = summarize_rollout_logs(self.logs, latest_only=True)

        self.assertEqual([step["step"] for step in result["steps"]], [2])

    def test_missing_latency_is_explicit(self) -> None:
        _write_rows(
            self.logs / "1.jsonl",
            [_row(1, 0.0, "anchor", latency=None)],
        )

        latency = summarize_rollout_logs(self.logs)["steps"][0]["anchor"][
            "latency_seconds"
        ]

        self.assertEqual(
            latency,
            {"count": 0, "mean": None, "p50": None, "p95": None, "max": None},
        )

    def test_rejects_malformed_or_inconsistent_audit_rows(self) -> None:
        _write_rows(
            self.logs / "1.jsonl",
            [
                _row(1, 1.0, "anchor", latency=1.0),
                _row(1, 1.0, "anchor", latency=2.0),
            ],
        )
        with self.assertRaisesRegex(TrainingError, "inconsistent audit"):
            summarize_rollout_logs(self.logs)

        (self.logs / "1.jsonl").write_text("{\n", encoding="utf-8")
        with self.assertRaisesRegex(TrainingError, "Invalid JSON"):
            summarize_rollout_logs(self.logs)

    def test_cli_prints_latest_metrics_as_json(self) -> None:
        _write_rows(self.logs / "1.jsonl", [_row(1, 1.0, "anchor-a")])
        _write_rows(self.logs / "2.jsonl", [_row(2, 0.0, "anchor-b")])
        output = StringIO()

        with redirect_stdout(output):
            return_code = cli.main(
                ["metrics", "--run-dir", str(self.run_dir), "--latest"]
            )

        self.assertEqual(return_code, 0)
        result = json.loads(output.getvalue())
        self.assertEqual(result["mode"], "label_free_training_metrics")
        self.assertEqual([step["step"] for step in result["steps"]], [2])


if __name__ == "__main__":
    unittest.main()
