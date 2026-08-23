"""CLI for planning and launching the MATH-500 CaT training run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from compute_as_a_teacher.data.math500 import DatasetPreparationError, load_locked_questions
from compute_as_a_teacher.evaluation.errors import EvaluationError
from compute_as_a_teacher.evaluation.prompts import load_prompt

from .config import load_training_config
from .checkpoints import register_final_checkpoint
from .eval_handoff import write_eval_handoff
from .errors import TrainingError
from .planning import (
    TRAINING_DATA_NAME,
    build_training_rows,
    dry_run_summary,
    load_training_plan,
    write_training_plan,
)
from .verl_adapter import checkpointed_step, launch_verl, merge_command


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _repo_path(path: Path) -> Path:
    return (path if path.is_absolute() else REPOSITORY_ROOT / path).resolve()


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _prepare(args: argparse.Namespace) -> dict[str, Any]:
    config = load_training_config(
        _repo_path(args.config),
        allow_unresolved=args.dry_run,
    )
    questions_path = _repo_path(Path(config.questions_path))
    lock_path = _repo_path(Path(config.dataset_lock_path))
    questions = load_locked_questions(questions_path, lock_path=lock_path)
    raw_prompt = load_prompt(REPOSITORY_ROOT, config.rollouts.prompt)
    synthesis_prompt = load_prompt(REPOSITORY_ROOT, config.synthesis.prompt)
    if args.dry_run:
        build_training_rows(questions, config, raw_prompt)
        return dry_run_summary(config, len(questions))
    run_dir = _repo_path(args.run_dir)
    manifest = write_training_plan(
        run_dir,
        questions,
        config,
        raw_prompt,
        synthesis_prompt,
        questions_path=questions_path,
        dataset_lock_path=lock_path,
        repository_root=REPOSITORY_ROOT,
        force=args.force,
    )
    return {
        "mode": "plan_written",
        "run_dir": str(run_dir),
        "plan_fingerprint": manifest["plan_fingerprint"],
        "counts": manifest["counts"],
        "training_data": str(run_dir / TRAINING_DATA_NAME),
        "model_loaded": False,
        "labels_loaded": False,
        "framework_imported": False,
    }


def _inspect(args: argparse.Namespace) -> dict[str, Any]:
    manifest, command = load_training_plan(_repo_path(args.run_dir))
    return {
        "run_name": manifest["run_name"],
        "protocol_version": manifest["protocol_version"],
        "plan_fingerprint": manifest["plan_fingerprint"],
        "counts": manifest["counts"],
        "label_firewall": manifest["label_firewall"],
        "framework_revision": command.framework_revision,
        "command_fingerprint": command.fingerprint,
    }


def _launch(args: argparse.Namespace) -> dict[str, Any]:
    config = load_training_config(_repo_path(args.config))
    run_dir = _repo_path(args.run_dir)
    manifest, command = load_training_plan(run_dir)
    if manifest["config_fingerprint"] != config.fingerprint:
        raise TrainingError("Config no longer matches the prepared training plan")
    completed_step = checkpointed_step(run_dir, config.grpo.max_steps)
    if completed_step == config.grpo.max_steps:
        return {
            "mode": "training_already_complete",
            "completed_step": completed_step,
            "would_execute": False,
        }
    if not args.execute:
        return {
            "mode": "launch_preflight",
            "would_execute": False,
            "argv": list(command.argv),
            "cwd": command.cwd,
            "environment": dict(command.environment),
            "checkpointed_step": completed_step,
            "note": "Add --execute only in the prepared verl/GPU environment.",
        }
    return_code = launch_verl(command, config, run_dir)
    if return_code:
        raise TrainingError(f"verl exited with status {return_code}")
    return {"mode": "training_finished", "return_code": return_code}


def _merge(args: argparse.Namespace) -> dict[str, Any]:
    config = load_training_config(_repo_path(args.config))
    manifest, _ = load_training_plan(_repo_path(args.run_dir))
    if manifest["config_fingerprint"] != config.fingerprint:
        raise TrainingError("Config no longer matches the prepared training plan")
    argv = merge_command(
        config,
        run_dir=_repo_path(args.run_dir),
        export_directory=_repo_path(args.export_dir),
    )
    return {
        "mode": "merge_command",
        "would_execute": False,
        "argv": list(argv),
        "note": "Run this after fixed-step training; it is not executed by this command.",
    }


def _register_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    manifest, completion = register_final_checkpoint(
        _repo_path(args.run_dir),
        _repo_path(args.export_dir),
        force=args.force,
    )
    return {
        "mode": "checkpoint_registered",
        "step": manifest["step"],
        "export_tree_sha256": manifest["export"]["tree_sha256"],
        "reportable": completion["reportable"],
        "non_reportable_reasons": completion["non_reportable_reasons"],
    }


def _plan_eval(args: argparse.Namespace) -> dict[str, Any]:
    handoff = write_eval_handoff(
        _repo_path(args.run_dir),
        _repo_path(args.output_dir),
        args.served_model,
        force=args.force,
    )
    return {
        "mode": "trained_eval_ready",
        "checkpoint_tree_sha256": handoff["checkpoint_tree_sha256"],
        "raw_config": handoff["raw_config"]["path"],
        "synthesis_config": handoff["synthesis_config"]["path"],
        "labels_loaded": False,
        "reportable": handoff["reportable"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and launch reference-free CaT GRPO training on MATH-500."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Build the locked label-free verl input and launch plan.")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--run-dir", type=Path, required=True)
    prepare.add_argument("--dry-run", action="store_true")
    prepare.add_argument("--force", action="store_true")

    inspect = commands.add_parser("inspect", help="Verify and summarize a prepared training plan.")
    inspect.add_argument("--run-dir", type=Path, required=True)

    launch = commands.add_parser("launch", help="Show or explicitly execute the pinned verl command.")
    launch.add_argument("--config", type=Path, required=True)
    launch.add_argument("--run-dir", type=Path, required=True)
    launch.add_argument("--execute", action="store_true")

    merge = commands.add_parser("merge-command", help="Print the pinned verl final-checkpoint export command.")
    merge.add_argument("--config", type=Path, required=True)
    merge.add_argument("--run-dir", type=Path, required=True)
    merge.add_argument("--export-dir", type=Path, required=True)

    register = commands.add_parser(
        "register-checkpoint",
        help="Content-address the merged fixed final checkpoint.",
    )
    register.add_argument("--run-dir", type=Path, required=True)
    register.add_argument("--export-dir", type=Path, required=True)
    register.add_argument("--force", action="store_true")

    plan_eval = commands.add_parser(
        "plan-trained-eval",
        help="Create raw and synthesis eval configs for the registered checkpoint.",
    )
    plan_eval.add_argument("--run-dir", type=Path, required=True)
    plan_eval.add_argument("--output-dir", type=Path, required=True)
    plan_eval.add_argument("--served-model", required=True)
    plan_eval.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = _prepare(args)
        elif args.command == "inspect":
            result = _inspect(args)
        elif args.command == "launch":
            result = _launch(args)
        elif args.command == "merge-command":
            result = _merge(args)
        elif args.command == "register-checkpoint":
            result = _register_checkpoint(args)
        else:
            result = _plan_eval(args)
    except (TrainingError, EvaluationError, DatasetPreparationError) as exc:
        parser.error(str(exc))
    _print(result)
    return 0
