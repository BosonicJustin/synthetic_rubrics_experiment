#!/usr/bin/env python3
"""Build a content-addressed trainer image from a reviewed base digest."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

try:
    from infra.server.write_image_metadata import source_tree
except ModuleNotFoundError:  # Direct execution from infra/server.
    from write_image_metadata import source_tree


ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = Path(__file__).with_name("Dockerfile")
VERL_REVISION = "8fdc4d3f202f41461f4de9f42a637228e342668b"
BASE_REFERENCE = re.compile(r"(?P<name>[^@\s]+)@(?P<digest>sha256:[0-9a-f]{64})")
SHA256 = re.compile(r"sha256:[0-9a-f]{64}")


class BuildError(RuntimeError):
    pass


def parse_base_reference(value: str) -> tuple[str, str]:
    match = BASE_REFERENCE.fullmatch(value)
    if match is None:
        raise BuildError(
            "--base-image must be a reviewed registry name@sha256:<64 lowercase hex> reference"
        )
    digest = match.group("digest")
    if len(set(digest.removeprefix("sha256:"))) == 1:
        raise BuildError("placeholder or uniform base-image digests are not accepted")
    return match.group("name"), digest


def _run(arguments: list[str], *, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(arguments, cwd=cwd, check=False, capture_output=True, text=True)


def _git_revision() -> str:
    completed = _run(["git", "rev-parse", "HEAD"])
    revision = completed.stdout.strip()
    if completed.returncode or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise BuildError("cannot resolve a full repository commit")
    return revision


def _require_clean_repository() -> None:
    completed = _run(["git", "status", "--porcelain", "--untracked-files=all"])
    if completed.returncode:
        raise BuildError("cannot inspect repository status")
    if completed.stdout.strip():
        raise BuildError("refusing to label a dirty build context; build from a clean clone")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-image", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--metadata-out", type=Path, required=True)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--push", action="store_true")
    destination.add_argument("--load", action="store_true")
    parser.add_argument(
        "--accept-reviewed-base",
        action="store_true",
        help="Confirm that the base digest and its CUDA/PyTorch contents were reviewed.",
    )
    return parser.parse_args(argv)


def build(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    if not args.accept_reviewed_base:
        raise BuildError("--accept-reviewed-base is required after reviewing the immutable base")
    base_name, base_digest = parse_base_reference(args.base_image)
    if args.metadata_out.exists():
        raise BuildError(f"metadata output already exists: {args.metadata_out}")
    if shutil.which("docker") is None:
        raise BuildError("docker is not installed")
    _require_clean_repository()
    source_revision = _git_revision()
    _, source_tree_sha256 = source_tree(ROOT)

    with tempfile.TemporaryDirectory(prefix="cat-image-build-") as temporary:
        temp = Path(temporary)
        iid_path = temp / "iid.txt"
        buildkit_path = temp / "buildkit.json"
        command = [
            "docker",
            "buildx",
            "build",
            "--file",
            str(DOCKERFILE),
            "--tag",
            args.tag,
            "--build-arg",
            f"TRAINER_BASE_IMAGE={args.base_image}",
            "--build-arg",
            f"TRAINER_BASE_NAME={base_name}",
            "--build-arg",
            f"TRAINER_BASE_DIGEST={base_digest}",
            "--build-arg",
            f"SOURCE_REVISION={source_revision}",
            "--build-arg",
            f"SOURCE_TREE_SHA256={source_tree_sha256}",
            "--build-arg",
            f"VERL_REVISION={VERL_REVISION}",
            "--iidfile",
            str(iid_path),
            "--metadata-file",
            str(buildkit_path),
            "--provenance=mode=max",
            "--sbom=true",
            "--push" if args.push else "--load",
            str(ROOT),
        ]
        completed = _run(command)
        if completed.returncode:
            detail = (completed.stderr or completed.stdout).strip()[-4000:]
            raise BuildError(f"docker buildx failed:\n{detail}")
        image_id = iid_path.read_text(encoding="utf-8").strip()
        if SHA256.fullmatch(image_id) is None:
            raise BuildError("buildx did not emit a content-addressed image ID")
        buildkit = json.loads(buildkit_path.read_text(encoding="utf-8"))

    registry_digest = buildkit.get("containerimage.digest")
    if registry_digest is not None and SHA256.fullmatch(registry_digest) is None:
        raise BuildError("buildx emitted an invalid registry digest")
    if args.push and registry_digest is None:
        raise BuildError("pushed build did not emit an immutable registry digest")
    result = {
        "schema_version": 1,
        "built_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "base_image": {"reference": args.base_image, "name": base_name, "digest": base_digest},
        "source_revision": source_revision,
        "source_tree_sha256": source_tree_sha256,
        "verl_revision": VERL_REVISION,
        "tag": args.tag,
        "destination": "push" if args.push else "load",
        "image_id": image_id,
        "registry_digest": registry_digest,
        "buildkit": buildkit,
    }
    args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    try:
        result = build(argv)
    except (BuildError, OSError, json.JSONDecodeError) as exc:
        print(f"trainer image build failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
