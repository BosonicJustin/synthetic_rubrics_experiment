#!/usr/bin/env python3
"""Small, explicit command router for the trainer image."""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
COMMANDS = {
    "doctor": (sys.executable, str(ROOT / "infra/server/doctor.py")),
    "ready": (sys.executable, str(ROOT / "infra/server/ready.py")),
    "train": (sys.executable, str(ROOT / "scripts/train_math500.py")),
    "workflow": (sys.executable, str(ROOT / "scripts/server_math500.py")),
}
ALLOWED_COMMANDS = {
    "trainer": frozenset({"doctor", "ready", "train", "workflow"}),
    "evaluator": frozenset({"workflow"}),
    "scorer": frozenset({"workflow"}),
}


def _serve_command(role: str, name: str) -> tuple[str, ...]:
    model_dir = os.environ.get("CAT_MODEL_DIR", "")
    if not model_dir or not Path(model_dir).is_absolute():
        raise SystemExit("entrypoint: CAT_MODEL_DIR must be an absolute path")
    common = (
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        "0.0.0.0",
        "--generation-config",
        "vllm",
    )
    if role == "anchor" and name == "serve-anchor":
        return (
            *common,
            "--port",
            "8001",
            "--model",
            model_dir,
            "--served-model-name",
            "Qwen/Qwen3-4B",
            "cat-frozen-qwen3-4b",
            "--dtype",
            "bfloat16",
            "--tensor-parallel-size",
            "1",
            "--max-model-len",
            "16384",
        )
    if role == "trained-policy" and name == "serve-trained-policy":
        return (
            *common,
            "--port",
            "8002",
            "--model",
            "/mnt/trained-model",
            "--tokenizer",
            model_dir,
            "--served-model-name",
            "math500-cat-final",
            "--dtype",
            "bfloat16",
            "--tensor-parallel-size",
            "1",
            "--max-model-len",
            "16384",
        )
    raise SystemExit(f"entrypoint: {name} is not allowed for role {role or '<unset>'}")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        args = ["ready"]
    role = os.environ.get("CAT_SERVICE_ROLE", "").strip()
    name = args.pop(0)
    if name in {"serve-anchor", "serve-trained-policy"}:
        if args:
            raise SystemExit(f"entrypoint: {name} accepts no arguments")
        command = list(_serve_command(role, name))
    elif name in COMMANDS and name in ALLOWED_COMMANDS.get(role, frozenset()):
        command = [*COMMANDS[name], *args]
    else:
        raise SystemExit(
            f"entrypoint: {name} is not allowed for role {role or '<unset>'}"
        )
    os.execvp(command[0], command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
