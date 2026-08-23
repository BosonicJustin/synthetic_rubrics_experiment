"""Model-free command line interface for MATH-500 evaluation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from compute_as_a_teacher.data.math500 import (
    DatasetPreparationError,
    load_locked_questions,
)

from .artifacts import read_jsonl
from .backend import ingest_responses
from .config import (
    load_raw_config,
    load_scoring_config,
    load_synthesis_config,
)
from .errors import EvaluationError
from .planning import (
    GENERATIONS_NAME,
    build_raw_requests,
    build_synthesis_requests,
    load_plan,
    write_raw_plan,
    write_synthesis_plan,
)
from .prompts import load_prompt
from .schemas import validate_and_order_generations
from .scoring import score_run


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _repo_path(value: Path) -> Path:
    path = value if value.is_absolute() else REPOSITORY_ROOT / value
    return path.resolve()


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _raw_plan(args: argparse.Namespace) -> dict[str, Any]:
    config_path = _repo_path(args.config)
    config = load_raw_config(
        config_path,
        allow_unresolved_model=args.dry_run,
    )
    questions_path = _repo_path(Path(config.questions_path))
    lock_path = _repo_path(Path(config.dataset_lock_path))
    questions = load_locked_questions(questions_path, lock_path=lock_path)
    template = load_prompt(REPOSITORY_ROOT, config.prompt)
    if args.dry_run:
        requests = build_raw_requests(questions, config, template)
        return {
            "mode": "dry_run",
            "kind": "raw",
            "would_write": False,
            "model_loaded": False,
            "labels_loaded": False,
            "problems": len(questions),
            "requests": len(requests),
            "rollouts_per_problem": config.rollouts_per_problem,
            "model_runnable": not config.model.unresolved_reasons(),
            "unresolved_model_reasons": list(config.model.unresolved_reasons()),
        }
    manifest = write_raw_plan(
        _repo_path(args.run_dir),
        questions,
        config,
        template,
        questions_path,
        repository_root=REPOSITORY_ROOT,
        force=args.force,
    )
    return {
        "mode": "plan_written",
        "kind": "raw",
        "run_dir": str(_repo_path(args.run_dir)),
        "plan_fingerprint": manifest["plan_fingerprint"],
        "counts": manifest["counts"],
        "model_loaded": False,
        "labels_loaded": False,
    }


def _synthesis_plan(args: argparse.Namespace) -> dict[str, Any]:
    config = load_synthesis_config(
        _repo_path(args.config),
        allow_unresolved_model=args.dry_run,
    )
    template = load_prompt(REPOSITORY_ROOT, config.prompt)
    raw_run_dir = _repo_path(args.raw_run_dir)
    if args.dry_run:
        raw_manifest, raw_requests = load_plan(raw_run_dir, expected_kind="raw")
        raw_generations = validate_and_order_generations(
            read_jsonl(raw_run_dir / GENERATIONS_NAME),
            raw_requests,
        )
        requests = build_synthesis_requests(raw_generations, config, template)
        same_anchor = all(request.model == config.anchor for request in raw_requests)
        return {
            "mode": "dry_run",
            "kind": "synthesis",
            "would_write": False,
            "model_loaded": False,
            "labels_loaded": False,
            "raw_plan_fingerprint": raw_manifest["plan_fingerprint"],
            "problems": len(requests),
            "raw_rollouts_consumed": len(raw_generations),
            "synthesis_requests": len(requests),
            "same_frozen_anchor": same_anchor,
            "model_runnable": not config.anchor.unresolved_reasons(),
            "unresolved_model_reasons": list(config.anchor.unresolved_reasons()),
        }
    manifest = write_synthesis_plan(
        _repo_path(args.run_dir),
        raw_run_dir,
        config,
        template,
        force=args.force,
    )
    return {
        "mode": "plan_written",
        "kind": "synthesis",
        "run_dir": str(_repo_path(args.run_dir)),
        "plan_fingerprint": manifest["plan_fingerprint"],
        "counts": manifest["counts"],
        "model_loaded": False,
        "labels_loaded": False,
    }


def _ingest(args: argparse.Namespace, expected_kind: str) -> dict[str, Any]:
    run_dir = _repo_path(args.run_dir)
    load_plan(run_dir, expected_kind=expected_kind)
    execution = ingest_responses(
        run_dir,
        _repo_path(args.responses),
        force=args.force,
    )
    return {
        "mode": execution["mode"],
        "kind": expected_kind,
        "complete": execution["complete"],
        "completed_requests": execution["completed_requests"],
        "reportable": not execution["non_reportable"],
        "non_reportable_reasons": execution["non_reportable_reasons"],
        "note": (
            "External ingestion is deliberately non-reportable until a real "
            "backend adapter can attest full model and runtime provenance."
        ),
    }


def _score(args: argparse.Namespace, expected_kind: str) -> dict[str, Any]:
    run_dir = _repo_path(args.run_dir)
    load_plan(run_dir, expected_kind=expected_kind)
    config = load_scoring_config(_repo_path(args.config))
    summary = score_run(
        run_dir,
        config,
        repository_root=REPOSITORY_ROOT,
        raw_run_dir=(
            _repo_path(args.raw_run_dir) if expected_kind == "synthesis" else None
        ),
        force=args.force,
    )
    return {
        "mode": "scored",
        "kind": expected_kind,
        "run_dir": str(run_dir),
        "reportable": summary["reportable"],
        "primary_metric": summary["primary_metric"],
        "primary_grader": summary["primary_grader"],
        "summary": str(run_dir / "summary.json"),
    }


def _inspect(args: argparse.Namespace) -> dict[str, Any]:
    manifest, requests = load_plan(_repo_path(args.run_dir))
    return {
        "kind": manifest["kind"],
        "protocol_version": manifest["protocol_version"],
        "run_name": manifest["run_name"],
        "plan_fingerprint": manifest["plan_fingerprint"],
        "counts": manifest["counts"],
        "request_rows_verified": len(requests),
        "label_firewall": manifest["label_firewall"],
    }


def _add_force(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace mismatched derived artifacts only after full preflight.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan, ingest, and score raw/synthesis MATH-500 evaluations. "
            "This milestone has no command that loads or runs a model."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    raw = subparsers.add_parser(
        "plan-raw",
        help="Plan eight raw policy rollouts per locked MATH-500 problem.",
    )
    raw.add_argument("--config", type=Path, required=True)
    raw.add_argument("--run-dir", type=Path, required=True)
    raw.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and count requests without writing or resolving a model.",
    )
    _add_force(raw)

    synthesis = subparsers.add_parser(
        "plan-synthesis",
        help="Plan one rollout-only synthesis request per completed raw group.",
    )
    synthesis.add_argument("--config", type=Path, required=True)
    synthesis.add_argument("--raw-run-dir", type=Path, required=True)
    synthesis.add_argument("--run-dir", type=Path, required=True)
    synthesis.add_argument("--dry-run", action="store_true")
    _add_force(synthesis)

    for kind in ("raw", "synthesis"):
        ingest = subparsers.add_parser(
            f"ingest-{kind}",
            help=f"Validate a complete external {kind} response JSONL.",
        )
        ingest.add_argument("--run-dir", type=Path, required=True)
        ingest.add_argument("--responses", type=Path, required=True)
        _add_force(ingest)

    score_raw = subparsers.add_parser(
        "score-raw",
        help="Load locked labels only now and score a complete raw run.",
    )
    score_raw.add_argument("--run-dir", type=Path, required=True)
    score_raw.add_argument("--config", type=Path, required=True)
    _add_force(score_raw)

    score_synthesis = subparsers.add_parser(
        "score-synthesis",
        help="Score synthesis and its exact paired raw dependency.",
    )
    score_synthesis.add_argument("--run-dir", type=Path, required=True)
    score_synthesis.add_argument("--raw-run-dir", type=Path, required=True)
    score_synthesis.add_argument("--config", type=Path, required=True)
    _add_force(score_synthesis)

    inspect = subparsers.add_parser(
        "inspect-plan",
        help="Verify and display a plan without loading labels or a model.",
    )
    inspect.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "plan-raw":
            result = _raw_plan(args)
        elif args.command == "plan-synthesis":
            result = _synthesis_plan(args)
        elif args.command == "ingest-raw":
            result = _ingest(args, "raw")
        elif args.command == "ingest-synthesis":
            result = _ingest(args, "synthesis")
        elif args.command == "score-raw":
            result = _score(args, "raw")
        elif args.command == "score-synthesis":
            result = _score(args, "synthesis")
        else:
            result = _inspect(args)
    except (EvaluationError, DatasetPreparationError) as exc:
        parser.exit(2, f"error: {exc}\n")
    _print_json(result)
    return 0
