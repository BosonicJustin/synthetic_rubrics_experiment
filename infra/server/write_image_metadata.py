#!/usr/bin/env python3
"""Write validated immutable-source metadata during the image build."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


COMMIT = re.compile(r"[0-9a-f]{40}")
DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
SOURCE_FILES = (
    "README.md",
    "pyproject.toml",
    "uv.lock",
    "scripts/evaluate_math500.py",
    "scripts/prepare_math500.py",
    "scripts/server_math500.py",
    "scripts/train_math500.py",
)
SOURCE_DIRECTORIES = (
    "configs/datasets",
    "configs/evals",
    "configs/server",
    "configs/training",
    "infra/server",
    "prompts",
    "src",
)


def source_tree(root: Path) -> tuple[list[dict], str]:
    paths = [root / relative for relative in SOURCE_FILES]
    for relative in SOURCE_DIRECTORIES:
        directory = root / relative
        if not directory.is_dir():
            raise ValueError(f"source allowlist directory is missing: {relative}")
        paths.extend(
            path
            for path in directory.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".pyo"}
            and path.name != ".DS_Store"
        )
    inventory = []
    for path in sorted(set(paths), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"source allowlist file is missing or a symlink: {path}")
        payload = path.read_bytes()
        inventory.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
        )
    encoded = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    return inventory, hashlib.sha256(encoded).hexdigest()


def metadata(
    source_revision: str,
    base_name: str,
    base_digest: str,
    verl_revision: str,
    source_root: Path,
) -> dict:
    if COMMIT.fullmatch(source_revision) is None or len(set(source_revision)) == 1:
        raise ValueError("source revision must be a full Git commit")
    if not base_name or "@" in base_name or any(character.isspace() for character in base_name):
        raise ValueError("base image name is invalid")
    if (
        DIGEST.fullmatch(base_digest) is None
        or len(set(base_digest.removeprefix("sha256:"))) == 1
    ):
        raise ValueError("base image digest is invalid")
    if COMMIT.fullmatch(verl_revision) is None or len(set(verl_revision)) == 1:
        raise ValueError("Verl revision must be a full Git commit")
    inventory, tree_sha256 = source_tree(source_root)
    return {
        "schema_version": 1,
        "source_revision": source_revision,
        "source_layout": "explicit_allowlist_v1",
        "source_tree_sha256": tree_sha256,
        "source_inventory": inventory,
        "base_image": {"name": base_name, "digest": base_digest},
        "verl_revision": verl_revision,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--base-name", required=True)
    parser.add_argument("--base-digest", required=True)
    parser.add_argument("--verl-revision", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-source-tree-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        value = metadata(
            args.source_revision,
            args.base_name,
            args.base_digest,
            args.verl_revision,
            args.source_root.resolve(),
        )
        if value["source_tree_sha256"] != args.expected_source_tree_sha256:
            raise ValueError("copied source tree does not match the clean host build context")
        args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
