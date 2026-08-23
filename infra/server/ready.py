#!/usr/bin/env python3
"""Run container and experiment readiness gates as one no-download check."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from compute_as_a_teacher.server import (  # noqa: E402
    ServerWorkflowError,
    load_server_workflow,
    readiness_report,
)
from infra.server.doctor import check_runtime, load_contract  # noqa: E402


def combined_report(
    workflow_path: Path,
    *,
    doctor_runner: Callable[[], dict[str, Any]] | None = None,
    workflow_runner: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    checks = []
    try:
        detail = (doctor_runner or (lambda: check_runtime(load_contract())))()
    except Exception as exc:
        checks.append({"name": "container_doctor", "ok": False, "error": str(exc)})
    else:
        checks.append({"name": "container_doctor", "ok": True, "detail": detail})
    try:
        if workflow_runner is None:
            workflow = load_server_workflow(workflow_path, repository_root=ROOT)
            detail = readiness_report(workflow)
        else:
            detail = workflow_runner()
        if detail.get("ready") is not True:
            raise ServerWorkflowError("experiment workflow readiness did not pass")
    except Exception as exc:
        checks.append({"name": "experiment_readiness", "ok": False, "error": str(exc)})
    else:
        checks.append({"name": "experiment_readiness", "ok": True, "detail": detail})
    return {
        "schema_version": 1,
        "kind": "cat_combined_server_readiness",
        "ready": all(item["ok"] for item in checks),
        "no_download": True,
        "model_weights_loaded": False,
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workflow",
        type=Path,
        default=Path(os.environ.get("CAT_WORKFLOW_PATH", "/mnt/config/workflow.toml")),
    )
    args = parser.parse_args(argv)
    result = combined_report(args.workflow.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
