"""Label-free summaries of Verl rollout logs."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .errors import TrainingError


ROLLOUT_LOGS_NAME = "rollout_logs"
_NOT_RECORDED = "not_recorded"


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def _number(value: Any, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0):
        qualifier = " nonnegative" if nonnegative else ""
        raise TrainingError(f"{name} must be a finite{qualifier} number")
    return result


def _status(value: Any) -> str:
    return value if isinstance(value, str) and value else _NOT_RECORDED


def _load_step(path: Path, expected_step: int) -> list[Mapping[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise TrainingError(f"Rollout log must be a regular file: {path}")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise TrainingError(f"Cannot read rollout log: {path}") from exc
    if not lines:
        raise TrainingError(f"Rollout log is empty: {path}")

    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TrainingError(
                f"Invalid JSON in {path} at line {line_number}"
            ) from exc
        if not isinstance(row, Mapping):
            raise TrainingError(
                f"Rollout row in {path} at line {line_number} must be an object"
            )
        step = row.get("step")
        if type(step) is not int or step != expected_step:
            raise TrainingError(
                f"Rollout row in {path} at line {line_number} has the wrong step"
            )
        rows.append(row)
    return rows


def _latency_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": sum(values) / len(values) if values else None,
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "max": max(values) if values else None,
    }


def summarize_step(
    rows: Sequence[Mapping[str, Any]],
    *,
    step: int,
) -> dict[str, Any]:
    if not rows:
        raise TrainingError(f"Rollout step {step} has no rows")

    rewards: list[float] = []
    rollout_statuses: Counter[str] = Counter()
    anchors: dict[tuple[str, int], tuple[str, str, float | None]] = {}
    for row_index, row in enumerate(rows):
        rewards.append(_number(row.get("score"), f"step {step} row {row_index} score"))
        rollout_statuses[_status(row.get("rollout_extraction_status"))] += 1

        anchor_id = row.get("anchor_prompt_sha256")
        if not isinstance(anchor_id, str) or not anchor_id:
            raise TrainingError(
                f"step {step} row {row_index} has no anchor prompt identity"
            )
        anchor_seed = row.get("anchor_seed")
        if type(anchor_seed) is not int:
            raise TrainingError(f"step {step} row {row_index} has no anchor seed")
        latency_value = row.get("anchor_latency_seconds")
        latency = (
            None
            if latency_value is None
            else _number(
                latency_value,
                f"step {step} row {row_index} anchor latency",
                nonnegative=True,
            )
        )
        anchor = (
            _status(row.get("anchor_extraction_status")),
            _status(row.get("anchor_finish_reason")),
            latency,
        )
        previous = anchors.setdefault((anchor_id, anchor_seed), anchor)
        if previous != anchor:
            raise TrainingError(
                f"step {step} has inconsistent audit fields for one anchor"
            )

    anchor_extraction = Counter(value[0] for value in anchors.values())
    anchor_finish = Counter(value[1] for value in anchors.values())
    latencies = [value[2] for value in anchors.values() if value[2] is not None]
    zero_count = sum(value == 0.0 for value in rewards)
    one_count = sum(value == 1.0 for value in rewards)
    count = len(rewards)
    return {
        "step": step,
        "task_reward": {
            "count": count,
            "mean": sum(rewards) / count,
            "zero_fraction": zero_count / count,
            "one_fraction": one_count / count,
        },
        "rollout_extraction_status_counts": dict(sorted(rollout_statuses.items())),
        "anchor": {
            "count": len(anchors),
            "extraction_status_counts": dict(sorted(anchor_extraction.items())),
            "finish_reason_counts": dict(sorted(anchor_finish.items())),
            "latency_seconds": _latency_summary(latencies),
        },
    }


def summarize_rollout_logs(
    rollout_logs: Path,
    *,
    latest_only: bool = False,
) -> dict[str, Any]:
    root = rollout_logs.resolve()
    if not root.is_dir():
        raise TrainingError(f"Rollout log directory is missing: {root}")

    files: dict[int, Path] = {}
    for path in root.glob("*.jsonl"):
        if not path.stem.isdecimal():
            continue
        step = int(path.stem)
        if step <= 0 or step in files:
            raise TrainingError(f"Rollout log has an invalid or duplicate step: {path}")
        files[step] = path
    if not files:
        raise TrainingError(f"No numeric Verl rollout logs found in: {root}")

    selected = [max(files)] if latest_only else sorted(files)
    summaries = [
        summarize_step(_load_step(files[step], step), step=step)
        for step in selected
    ]
    return {
        "mode": "label_free_training_metrics",
        "rollout_logs": str(root),
        "latest_step": max(files),
        "steps": summaries,
        "gold_labels_loaded": False,
    }


__all__ = ["ROLLOUT_LOGS_NAME", "summarize_rollout_logs", "summarize_step"]
