#!/usr/bin/env python3
"""Validate host mounts and immutable identities before starting Compose."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Mapping


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from compute_as_a_teacher.data.math500 import (  # noqa: E402
    DatasetPreparationError,
    verify_dataset,
)


IMAGE_REFERENCE = re.compile(r"[^@\s]+@(sha256:[0-9a-f]{64})")
IMAGE_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
MODEL_REVISION = re.compile(r"[0-9a-f]{40}")
GPU_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*")
SEMVER = re.compile(r"(?<![0-9])(\d+)\.(\d+)\.(\d+)(?![0-9])")
TRAINED_EXPORT = Path("exports/qwen3-4b-math500-cat-step-1000")
DATASET_LOCK = ROOT / "configs/datasets/math500.lock.json"
MINIMUM_TOOL_VERSIONS = {
    "docker": (24, 0, 0),
    "compose": (2, 30, 0),
    "buildx": (0, 14, 0),
}


class EnvironmentError(RuntimeError):
    pass


def _command_version(argv: tuple[str, ...], name: str) -> tuple[str, tuple[int, int, int]]:
    try:
        completed = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise EnvironmentError(f"cannot inspect {name} version: {exc}") from exc
    output = (completed.stdout or completed.stderr).strip()
    match = SEMVER.search(output)
    if completed.returncode or match is None:
        raise EnvironmentError(f"cannot inspect {name} version: {output}")
    return output, tuple(map(int, match.groups()))


def validate_container_tooling() -> dict[str, str]:
    commands = {
        "docker": ("docker", "--version"),
        "compose": ("docker", "compose", "version", "--short"),
        "buildx": ("docker", "buildx", "version"),
    }
    result = {}
    for name, argv in commands.items():
        output, version = _command_version(argv, name)
        if version < MINIMUM_TOOL_VERSIONS[name]:
            minimum = ".".join(map(str, MINIMUM_TOOL_VERSIONS[name]))
            raise EnvironmentError(f"{name} must be at least {minimum}: {output}")
        result[name] = ".".join(map(str, version))
    return result


def _required(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name, "")
    if not value:
        raise EnvironmentError(f"{name} is required")
    return value


def _directory(environment: Mapping[str, str], name: str) -> Path:
    path = Path(_required(environment, name)).expanduser().resolve()
    if not path.is_dir():
        raise EnvironmentError(f"{name} directory is missing: {path}")
    return path


def _materialized_model_directory(environment: Mapping[str, str]) -> Path:
    configured = Path(_required(environment, "CAT_MODEL_HOST_DIR")).expanduser()
    if configured.is_symlink():
        raise EnvironmentError("CAT_MODEL_HOST_DIR must not be a symlink")
    model_dir = configured.resolve()
    if not model_dir.is_dir():
        raise EnvironmentError(f"CAT_MODEL_HOST_DIR directory is missing: {model_dir}")
    pending = [model_dir]
    try:
        while pending:
            directory = pending.pop()
            with os.scandir(directory) as entries:
                for entry in entries:
                    path = Path(entry.path)
                    if entry.is_symlink():
                        raise EnvironmentError(
                            f"model snapshot must be materialized without symlinks: {path}"
                        )
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(path)
                    elif not entry.is_file(follow_symlinks=False):
                        raise EnvironmentError(
                            f"model snapshot contains a non-file entry: {path}"
                        )
    except OSError as exc:
        raise EnvironmentError(f"cannot inspect model snapshot: {exc}") from exc
    return model_dir


def _validate_isolated_roots(roots: Mapping[str, Path]) -> None:
    items = tuple(roots.items())
    for index, (left_name, left) in enumerate(items):
        for right_name, right in items[index + 1 :]:
            if left == right or left in right.parents or right in left.parents:
                raise EnvironmentError(
                    f"{left_name} and {right_name} must be distinct, non-nested directories"
                )


def _verify_full_dataset(data_dir: Path) -> dict:
    if data_dir.name != "data":
        raise EnvironmentError("CAT_DATA_HOST_DIR basename must be data")
    try:
        return verify_dataset(data_dir.parent, DATASET_LOCK)
    except DatasetPreparationError as exc:
        raise EnvironmentError(f"locked host MATH-500 verification failed: {exc}") from exc


def validate(
    environment: Mapping[str, str], *, require_trained_policy: bool = False
) -> dict:
    trainer_ids = [
        _required(environment, f"CAT_TRAINER_GPU_DEVICE_{index}")
        for index in range(8)
    ]
    if (
        len(set(trainer_ids)) != 8
        or any(GPU_ID.fullmatch(item) is None for item in trainer_ids)
    ):
        raise EnvironmentError("trainer GPU devices must contain exactly eight distinct IDs")
    anchor_mode = _required(environment, "CAT_ANCHOR_MODE")
    anchor_value = environment.get("CAT_ANCHOR_GPU_DEVICE", "")
    if anchor_mode == "local":
        if GPU_ID.fullmatch(anchor_value) is None:
            raise EnvironmentError("local anchor GPU device must contain exactly one ID")
        if anchor_value in set(trainer_ids):
            raise EnvironmentError("trainer and anchor GPU IDs must be disjoint")
        anchor_gpu_id = anchor_value
    elif anchor_mode == "remote":
        if anchor_value:
            raise EnvironmentError("remote anchor mode must not set a local anchor GPU ID")
        anchor_gpu_id = None
    else:
        raise EnvironmentError("CAT_ANCHOR_MODE must be local or remote")

    revision = _required(environment, "CAT_MODEL_REVISION")
    if MODEL_REVISION.fullmatch(revision) is None or len(set(revision)) == 1:
        raise EnvironmentError("CAT_MODEL_REVISION must be a full non-placeholder commit")
    image = _required(environment, "CAT_TRAINER_IMAGE")
    digest = _required(environment, "CAT_TRAINER_IMAGE_DIGEST")
    match = IMAGE_REFERENCE.fullmatch(image)
    if (
        match is None
        or IMAGE_DIGEST.fullmatch(digest) is None
        or len(set(digest.removeprefix("sha256:"))) == 1
        or match.group(1) != digest
    ):
        raise EnvironmentError("trainer image must be name@digest matching CAT_TRAINER_IMAGE_DIGEST")

    model_dir = _materialized_model_directory(environment)
    data_dir = _directory(environment, "CAT_DATA_HOST_DIR")
    output_dir = _directory(environment, "CAT_OUTPUT_HOST_DIR")
    config_dir = _directory(environment, "CAT_CONFIG_HOST_DIR")
    _validate_isolated_roots(
        {
            "CAT_MODEL_HOST_DIR": model_dir,
            "CAT_DATA_HOST_DIR": data_dir,
            "CAT_OUTPUT_HOST_DIR": output_dir,
            "CAT_CONFIG_HOST_DIR": config_dir,
        }
    )
    if model_dir.name != revision:
        raise EnvironmentError("model host directory basename must equal CAT_MODEL_REVISION")
    trained_policy_value = environment.get(
        "CAT_TRAINED_POLICY_GPU_DEVICE", ""
    )
    if require_trained_policy or trained_policy_value:
        if GPU_ID.fullmatch(trained_policy_value) is None:
            raise EnvironmentError("trained-policy GPU device must contain exactly one ID")
        if anchor_gpu_id is not None and trained_policy_value == anchor_gpu_id:
            raise EnvironmentError("trained-policy and local anchor GPU IDs must be disjoint")
        trained_policy_gpu_id = trained_policy_value
    else:
        trained_policy_gpu_id = None
    required_data = (
        data_dir / "raw/math500-test.jsonl",
        data_dir / "math500/questions.jsonl",
        data_dir / "math500/labels.jsonl",
    )
    required_configs = tuple(
        config_dir / name
        for name in ("workflow.toml", "training.toml", "raw.toml", "synthesis.toml", "scoring.toml")
    )
    missing = [str(path) for path in (*required_data, *required_configs) if not path.is_file()]
    if require_trained_policy:
        export_dir = output_dir / TRAINED_EXPORT
        export_files = (export_dir / "config.json",)
        missing.extend(str(path) for path in export_files if not path.is_file())
        if not any(path.is_file() for path in export_dir.glob("*.safetensors")):
            missing.append(f"{export_dir}/*.safetensors")
    if missing:
        raise EnvironmentError(f"required server files are missing: {missing}")
    dataset = _verify_full_dataset(data_dir)
    try:
        with tempfile.NamedTemporaryFile(dir=output_dir, prefix=".cat-host-check-", delete=True):
            pass
    except OSError as exc:
        raise EnvironmentError(f"output directory is not writable: {exc}") from exc
    return {
        "schema_version": 1,
        "kind": "cat_server_host_environment",
        "ready": True,
        "trainer_gpu_ids": trainer_ids,
        "anchor_mode": anchor_mode,
        "anchor_gpu_id": anchor_gpu_id,
        "trained_policy_required": require_trained_policy,
        "trained_policy_gpu_id": trained_policy_gpu_id,
        "model_revision": revision,
        "trainer_image": image,
        "trainer_image_digest": digest,
        "model_dir": str(model_dir),
        "data_dir": str(data_dir),
        "dataset": dataset,
        "output_dir": str(output_dir),
        "config_dir": str(config_dir),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-trained-policy", action="store_true")
    args = parser.parse_args(argv)
    try:
        tooling = validate_container_tooling()
        result = validate(
            os.environ, require_trained_policy=args.require_trained_policy
        )
    except EnvironmentError as exc:
        print(f"server environment validation failed: {exc}", file=sys.stderr)
        return 1
    result["container_tooling"] = tooling
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
