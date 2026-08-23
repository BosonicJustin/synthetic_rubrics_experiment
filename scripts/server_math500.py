#!/usr/bin/env python3
"""Entry point for server readiness and guarded MATH-500 phases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from compute_as_a_teacher.server import (  # noqa: E402
    PHASE_EXECUTION_SCOPES,
    QUALIFICATION_PROFILES,
    ServerWorkflowError,
    execute_phase,
    load_server_workflow,
    preview_phase,
    provision_export_parent,
    readiness_report,
)
from compute_as_a_teacher.evaluation.errors import EvaluationError  # noqa: E402
from compute_as_a_teacher.training.errors import TrainingError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check a no-download server and delegate guarded experiment phases."
    )
    parser.add_argument("--workflow", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "readiness",
        help="Run read-only/no-download identity, GPU, storage, and anchor checks.",
    )
    commands.add_parser(
        "provision-export-parent",
        help="Create only the private parent required by guarded checkpoint export.",
    )
    phase = commands.add_parser(
        "phase", help="Preview a phase, or explicitly delegate it with --execute."
    )
    phase.add_argument(
        "name",
        choices=tuple(PHASE_EXECUTION_SCOPES),
    )
    phase.add_argument("--profile", choices=QUALIFICATION_PROFILES)
    phase.add_argument("--resume-action", choices=("prepare", "initial", "restart"))
    phase.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        workflow = load_server_workflow(
            args.workflow, repository_root=REPOSITORY_ROOT
        )
        if args.command == "readiness":
            result = readiness_report(workflow)
            status = 0 if result["ready"] else 2
        elif args.command == "provision-export-parent":
            result = provision_export_parent(workflow)
            status = 0
        elif args.execute:
            result = execute_phase(
                workflow,
                args.name,
                qualification_profile=args.profile,
                resume_action=args.resume_action,
                replace_final_process=True,
            )
            status = 0
        else:
            result = preview_phase(
                workflow,
                args.name,
                qualification_profile=args.profile,
                resume_action=args.resume_action,
            )
            status = 0
    except (ServerWorkflowError, EvaluationError, TrainingError) as exc:
        result = {
            "schema_version": 1,
            "kind": "cat_server_error",
            "error": str(exc),
        }
        status = 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
