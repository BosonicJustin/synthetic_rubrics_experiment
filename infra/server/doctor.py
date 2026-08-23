#!/usr/bin/env python3
"""Fail-closed image and server checks without loading model weights."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

try:
    from infra.server.adapter_smoke import run as run_adapter_smoke
    from infra.server.write_image_metadata import source_tree
except ModuleNotFoundError:  # Direct execution from infra/server.
    from adapter_smoke import run as run_adapter_smoke
    from write_image_metadata import source_tree


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = Path(__file__).with_name("runtime-contract.json")
SOURCE_METADATA_PATH = Path("/opt/cat/image-metadata/source.json")
IMAGE_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
IMAGE_REFERENCE = re.compile(r"[^@\s]+@(sha256:[0-9a-f]{64})")


class DoctorError(RuntimeError):
    pass


def load_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DoctorError(f"cannot read runtime contract {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise DoctorError("runtime contract must be a schema-version-1 object")
    return value


def package_inventory() -> tuple[list[list[str]], str]:
    packages = sorted(
        [dist.metadata.get("Name", "").lower(), dist.version]
        for dist in importlib.metadata.distributions()
        if dist.metadata.get("Name")
    )
    payload = json.dumps(packages, separators=(",", ":")).encode()
    return packages, hashlib.sha256(payload).hexdigest()


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise DoctorError(message)


def _check_readable_file(path: Path, name: str) -> None:
    try:
        with path.open("rb") as handle:
            handle.read(1)
    except OSError as exc:
        raise DoctorError(f"{name} is not readable by runtime UID: {path}: {exc}") from exc


def check_adapter_smoke() -> dict[str, Any]:
    try:
        return run_adapter_smoke(ROOT)
    except Exception as exc:
        raise DoctorError(f"pinned Verl adapter smoke failed: {exc}") from exc


def validate_image_identity(reference: str, digest: str) -> None:
    _check(IMAGE_DIGEST.fullmatch(digest) is not None, "CAT_TRAINER_IMAGE_DIGEST is invalid")
    _check(len(set(digest.removeprefix("sha256:"))) > 1, "trainer image digest is a placeholder")
    match = IMAGE_REFERENCE.fullmatch(reference)
    _check(match is not None, "CAT_TRAINER_IMAGE_REFERENCE must be name@sha256")
    _check(match.group(1) == digest, "trainer image reference and digest disagree")


def _git_revision(path: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise DoctorError(f"cannot verify Verl checkout: {completed.stderr.strip()}")
    return completed.stdout.strip()


def check_image(contract: dict[str, Any]) -> dict[str, Any]:
    python = contract["python"]
    minimum = tuple(map(int, python["minimum"].split(".")))
    maximum = tuple(map(int, python["maximum_exclusive"].split(".")))
    current = sys.version_info[:2]
    _check(minimum <= current < maximum, f"Python {current} violates the runtime contract")
    if hasattr(os, "geteuid"):
        _check(
            os.geteuid() == contract["runtime_uid"],
            f"trainer runtime must run as UID {contract['runtime_uid']}",
        )
    for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        _check(os.environ.get(name) == "1", f"{name}=1 is required")

    source = Path(os.environ.get("CAT_VERL_SOURCE", "/opt/verl/source")).resolve()
    _check(source.is_dir(), f"Verl source checkout is missing: {source}")
    revision = _git_revision(source)
    _check(revision == contract["verl_revision"], "Verl checkout revision mismatch")
    try:
        source_metadata = json.loads(SOURCE_METADATA_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DoctorError(f"cannot read immutable image source metadata: {exc}") from exc
    _check(
        isinstance(source_metadata, dict)
        and source_metadata.get("schema_version") == 1
        and re.fullmatch(r"[0-9a-f]{40}", str(source_metadata.get("source_revision", "")))
        and source_metadata.get("source_layout") == "explicit_allowlist_v1",
        "immutable image source metadata is invalid",
    )
    base_metadata = source_metadata.get("base_image")
    _check(
        isinstance(base_metadata, dict)
        and isinstance(base_metadata.get("name"), str)
        and IMAGE_DIGEST.fullmatch(str(base_metadata.get("digest", ""))) is not None,
        "immutable base-image metadata is invalid",
    )
    _check(
        len(set(str(base_metadata["digest"]).removeprefix("sha256:"))) > 1,
        "immutable base-image digest is a placeholder",
    )
    _check(
        source_metadata.get("verl_revision") == revision,
        "image metadata and Verl checkout revisions disagree",
    )
    inventory, source_tree_sha256 = source_tree(ROOT)
    _check(
        source_metadata.get("source_inventory") == inventory
        and source_metadata.get("source_tree_sha256") == source_tree_sha256,
        "copied source tree does not match immutable image metadata",
    )
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    spec = importlib.util.find_spec("verl")
    _check(spec is not None and spec.origin is not None, "cannot resolve the Verl module")
    try:
        Path(spec.origin).resolve().relative_to(source)
    except ValueError as exc:
        raise DoctorError("Verl import does not resolve inside the pinned checkout") from exc

    actual_packages: dict[str, str] = {}
    for name, expected in contract["packages"].items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise DoctorError(f"required package is missing: {name}=={expected}") from exc
        _check(actual == expected, f"{name} must be {expected}, found {actual}")
        actual_packages[name] = actual

    import antlr4
    import latex2sympy2_extended
    import math_verify
    import mpmath
    import sympy
    import tomli
    import torch
    import wandb

    _check(antlr4 is not None, "antlr4 import failed")
    _check(latex2sympy2_extended is not None, "latex2sympy2_extended import failed")
    _check(math_verify is not None, "math_verify import failed")
    _check(mpmath.__version__ == contract["packages"]["mpmath"], "mpmath import version mismatch")
    _check(sympy.__version__ == contract["packages"]["sympy"], "SymPy import version mismatch")
    _check(tomli.__version__ == contract["packages"]["tomli"], "tomli import version mismatch")
    _check(wandb.__version__ == contract["packages"]["wandb"], "W&B import version mismatch")
    _check(torch.version.cuda == contract["cuda"], "PyTorch CUDA version mismatch")
    _check(torch.backends.cudnn.version() == contract["cudnn"], "cuDNN version mismatch")
    adapter_smoke = check_adapter_smoke()
    inventory, inventory_sha256 = package_inventory()
    return {
        "python": ".".join(map(str, sys.version_info[:3])),
        "packages": actual_packages,
        "package_inventory": inventory,
        "package_inventory_sha256": inventory_sha256,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "runtime_uid": os.geteuid() if hasattr(os, "geteuid") else None,
        "verl_revision": revision,
        "verl_module": str(Path(spec.origin).resolve()),
        "adapter_smoke": adapter_smoke,
        "image_source": source_metadata,
    }


def _required_directory(env_name: str, default: str) -> Path:
    path = Path(os.environ.get(env_name, default)).resolve()
    _check(path.is_dir(), f"{env_name} directory is missing: {path}")
    return path


def is_read_only_mount(path: Path) -> bool:
    return bool(os.statvfs(path).f_flag & os.ST_RDONLY)


def check_runtime(contract: dict[str, Any]) -> dict[str, Any]:
    result = check_image(contract)
    import torch

    gpu = contract["gpu"]
    visible = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")]
    _check(
        len(visible) == gpu["count"] and all(visible) and len(set(visible)) == len(visible),
        "CUDA_VISIBLE_DEVICES must identify exactly eight distinct trainer GPUs",
    )
    _check(torch.cuda.is_available(), "CUDA is not available inside the container")
    _check(torch.cuda.device_count() == gpu["count"], "GPU count violates the runtime contract")
    devices = []
    for index in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(index)
        _check(gpu["name_contains"].lower() in name.lower(), f"GPU {index} is not an H100")
        if gpu["bf16_required"]:
            _check(torch.cuda.is_bf16_supported(), f"GPU {index} lacks BF16 support")
        devices.append(name)

    model_dir = _required_directory(
        "CAT_MODEL_DIR", "/mnt/models/required_full_model_commit_sha"
    )
    _check(
        re.fullmatch(r"[0-9a-f]{40}", model_dir.name) is not None,
        "model mount basename must be the full model revision SHA",
    )
    data_dir = _required_directory("CAT_DATA_DIR", "/mnt/data")
    output_dir = _required_directory("CAT_OUTPUT_DIR", "/mnt/outputs")
    config_path = Path(os.environ.get("CAT_CONFIG_PATH", "/mnt/config/training.toml")).resolve()
    _check(config_path.is_file(), f"training config is missing: {config_path}")
    for filename in ("config.json", "tokenizer_config.json"):
        path = model_dir / filename
        _check(path.is_file(), f"model snapshot lacks {filename}")
        _check_readable_file(path, f"model {filename}")
    weights = tuple(model_dir.glob("*.safetensors"))
    _check(bool(weights), "model snapshot has no safetensors weights")
    for path in weights:
        _check_readable_file(path, "model weight shard")
    questions = data_dir / "math500/questions.jsonl"
    _check(questions.is_file(), "MATH-500 questions are missing")
    _check_readable_file(questions, "MATH-500 questions")
    for forbidden in (
        data_dir / "raw/math500-test.jsonl",
        data_dir / "math500/labels.jsonl",
    ):
        _check(
            not os.path.lexists(forbidden),
            f"label-free trainer must not expose {forbidden}",
        )
    _check_readable_file(config_path, "training config")
    _check(is_read_only_mount(model_dir), "model mount must be read-only")
    _check(is_read_only_mount(questions), "questions mount must be read-only")
    _check(is_read_only_mount(config_path.parent), "config mount must be read-only")
    _check(os.access(output_dir, os.W_OK), "output mount is not writable")
    try:
        with tempfile.NamedTemporaryFile(dir=output_dir, prefix=".cat-doctor-", delete=True):
            pass
    except OSError as exc:
        raise DoctorError(f"cannot create an output probe: {exc}") from exc

    image_digest = os.environ.get("CAT_TRAINER_IMAGE_DIGEST", "")
    image_reference = os.environ.get("CAT_TRAINER_IMAGE_REFERENCE", "")
    validate_image_identity(image_reference, image_digest)
    result.update(
        {
            "gpus": devices,
            "cuda_visible_devices": visible,
            "model_dir": str(model_dir),
            "data_dir": str(data_dir),
            "output_dir": str(output_dir),
            "config_path": str(config_path),
            "trainer_image_digest": image_digest,
            "trainer_image_reference": image_reference,
        }
    )
    return result


def check_health(contract: dict[str, Any]) -> dict[str, Any]:
    if hasattr(os, "geteuid"):
        _check(
            os.geteuid() == contract["runtime_uid"],
            f"trainer runtime must run as UID {contract['runtime_uid']}",
        )
    for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        _check(os.environ.get(name) == "1", f"{name}=1 is required")
    try:
        metadata = json.loads(SOURCE_METADATA_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DoctorError(f"cannot read immutable image source metadata: {exc}") from exc
    inventory, tree_sha256 = source_tree(ROOT)
    _check(
        isinstance(metadata, dict)
        and metadata.get("source_inventory") == inventory
        and metadata.get("source_tree_sha256") == tree_sha256
        and metadata.get("verl_revision") == contract["verl_revision"],
        "immutable source health check failed",
    )
    validate_image_identity(
        os.environ.get("CAT_TRAINER_IMAGE_REFERENCE", ""),
        os.environ.get("CAT_TRAINER_IMAGE_DIGEST", ""),
    )
    config_path = Path(os.environ.get("CAT_CONFIG_PATH", "/mnt/config/training.toml")).resolve()
    output_dir = Path(os.environ.get("CAT_OUTPUT_DIR", "/mnt/outputs")).resolve()
    _check(config_path.is_file(), f"training config is missing: {config_path}")
    _check(output_dir.is_dir(), f"output directory is missing: {output_dir}")
    _check(not is_read_only_mount(output_dir), "output mount is read-only")
    return {
        "source_revision": metadata.get("source_revision"),
        "source_tree_sha256": tree_sha256,
        "config_path": str(config_path),
        "output_dir": str(output_dir),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("image", "runtime", "health"), default="runtime")
    parser.add_argument("--inventory-out", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        contract = load_contract()
        if args.mode == "image":
            result = check_image(contract)
        elif args.mode == "health":
            result = check_health(contract)
        else:
            result = check_runtime(contract)
        if args.inventory_out:
            payload = {
                "schema_version": 1,
                "packages": result["package_inventory"],
                "sha256": result["package_inventory_sha256"],
            }
            args.inventory_out.parent.mkdir(parents=True, exist_ok=True)
            args.inventory_out.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        if not args.quiet:
            print(json.dumps({"ok": True, "mode": args.mode, **result}, sort_keys=True))
        return 0
    except DoctorError as exc:
        print(f"server doctor failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
