"""Content-addressed registration of the fixed final verl checkpoint."""

from __future__ import annotations

import os
import re
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Mapping

from compute_as_a_teacher.evaluation.artifacts import (
    artifact_reference,
    canonical_json_bytes,
    file_digest,
    publish_json,
    read_json,
    sha256_bytes,
)
from compute_as_a_teacher.evaluation.errors import EvaluationError

from .config import TrainingConfig
from .errors import TrainingError
from .planning import MANIFEST_NAME as TRAINING_MANIFEST_NAME, load_training_plan
from .preflight import hash_model_snapshot_tree
from .verl_adapter import (
    LaunchLease,
    exclusive_launch,
    merge_command,
    run_command_with_log,
    validate_export_destination,
    validate_verl_checkpoint,
    verify_verl_checkout,
)


CHECKPOINT_MANIFEST_NAME = "checkpoint_manifest.json"
COMPLETION_NAME = "completion.json"
MERGE_RECEIPT_NAME = "merge_receipt.json"
MERGE_RECEIPT_KIND = "cat_guarded_merge_receipt"
_MERGE_ENVIRONMENT_NAMES = (
    "CUDA_HOME",
    "CUDA_VISIBLE_DEVICES",
    "LD_LIBRARY_PATH",
    "NVIDIA_VISIBLE_DEVICES",
    "PATH",
    "TMPDIR",
)


def directory_inventory(root: Path) -> tuple[list[dict[str, Any]], str]:
    root = root.resolve()
    if not root.is_dir():
        raise TrainingError(f"Checkpoint directory is missing: {root}")
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise TrainingError(f"Checkpoint inventory rejects symlinks: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise TrainingError(f"Unsupported checkpoint entry: {path}")
        digest, size = file_digest(path)
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": digest,
                "bytes": size,
            }
        )
    if not entries:
        raise TrainingError(f"Checkpoint directory is empty: {root}")
    return entries, sha256_bytes(canonical_json_bytes(entries))


def _validate_huggingface_export(export_dir: Path, inventory: list[dict[str, Any]]) -> None:
    paths = {entry["path"] for entry in inventory}
    if "config.json" not in paths:
        raise TrainingError("Merged Hugging Face checkpoint is missing config.json")
    if not any("/" not in path and path.endswith(".safetensors") for path in paths):
        raise TrainingError(
            "Merged Hugging Face checkpoint has no top-level safetensors weight file"
        )


def _artifact(path: Path, *, relative_name: str | None = None) -> dict[str, Any]:
    value = artifact_reference(path)
    if relative_name is not None:
        value["path"] = relative_name
    return value


def _receipt_fingerprint(value: Mapping[str, Any]) -> str:
    unsigned = dict(value)
    unsigned.pop("receipt_fingerprint", None)
    return sha256_bytes(canonical_json_bytes(unsigned))


def _merge_environment(source: Path) -> dict[str, str]:
    environment = {
        name: os.environ[name]
        for name in _MERGE_ENVIRONMENT_NAMES
        if os.environ.get(name)
    }
    if any(
        any(character.isspace() and character not in {" "} for character in value)
        for value in environment.values()
    ):
        raise TrainingError("Merge environment contains control whitespace")
    environment.update(
        {
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": str(source),
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    return dict(sorted(environment.items()))


def _expected_merge_argv(
    plan: Mapping[str, Any], run_dir: Path, export_dir: Path
) -> tuple[str, ...]:
    config = plan["config"]
    final_step = config["grpo"]["max_steps"]
    actor_dir = run_dir / "checkpoints" / f"global_step_{final_step}" / "actor"
    return (
        config["runtime"]["python_executable"],
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        str(actor_dir.resolve()),
        "--target_dir",
        str(export_dir.resolve()),
    )


def _verify_reference(reference: Any, path: Path, name: str) -> None:
    if not isinstance(reference, dict):
        raise TrainingError(f"{name} reference is missing")
    try:
        canonical = path.resolve(strict=True)
    except OSError as exc:
        raise TrainingError(f"{name} artifact is missing or unreadable") from exc
    if path.is_symlink() or canonical != path or not path.is_file():
        raise TrainingError(f"{name} must be a regular file")
    try:
        expected = artifact_reference(path)
    except EvaluationError as exc:
        raise TrainingError(f"{name} artifact is missing or unreadable") from exc
    if reference.get("path") not in {str(path), path.name}:
        raise TrainingError(f"{name} path changed")
    if reference.get("sha256") != expected["sha256"] or reference.get(
        "bytes"
    ) != expected["bytes"]:
        raise TrainingError(f"{name} changed")


def _verify_merge_receipt(
    run_dir: Path,
    export_dir: Path,
    plan: Mapping[str, Any],
    actor_inventory: list[dict[str, Any]],
    actor_digest: str,
    export_inventory: list[dict[str, Any]],
    export_digest: str,
    *,
    verify_live_base: bool = True,
) -> dict[str, Any]:
    receipt_path = run_dir / MERGE_RECEIPT_NAME
    try:
        receipt = read_json(receipt_path)
    except EvaluationError as exc:
        raise TrainingError(
            "A verified guarded merge receipt is required for registration"
        ) from exc
    expected_keys = {
        "schema_version",
        "kind",
        "training_plan_fingerprint",
        "config_fingerprint",
        "training_plan",
        "step",
        "tracker",
        "merge",
        "runtime_identity",
        "base_model",
        "actor",
        "export",
        "log",
        "receipt_fingerprint",
    }
    if set(receipt) != expected_keys or receipt.get("schema_version") != 1:
        raise TrainingError("Guarded merge receipt schema is invalid")
    if receipt.get("kind") != MERGE_RECEIPT_KIND:
        raise TrainingError("Guarded merge receipt kind is invalid")
    if receipt.get("receipt_fingerprint") != _receipt_fingerprint(receipt):
        raise TrainingError("Guarded merge receipt fingerprint mismatch")
    config = plan["config"]
    runtime = config["runtime"]
    if (
        receipt.get("training_plan_fingerprint") != plan["plan_fingerprint"]
        or receipt.get("config_fingerprint") != plan["config_fingerprint"]
        or receipt.get("step") != config["grpo"]["max_steps"]
    ):
        raise TrainingError("Guarded merge receipt belongs to another training plan")
    _verify_reference(
        receipt.get("training_plan"),
        run_dir / TRAINING_MANIFEST_NAME,
        "Merge training plan",
    )
    tracker = run_dir / "checkpoints/latest_checkpointed_iteration.txt"
    _verify_reference(receipt.get("tracker"), tracker, "Merge tracker")
    expected_runtime = {
        "framework_revision": runtime["framework_revision"],
        "package_inventory_sha256": runtime["package_inventory_sha256"],
        "trainer_image_digest": runtime["trainer_image_digest"],
    }
    if receipt.get("runtime_identity") != expected_runtime:
        raise TrainingError("Guarded merge runtime identity mismatch")
    base_model = receipt.get("base_model")
    if not isinstance(base_model, dict) or set(base_model) != {
        "model_path",
        "tree_sha256",
        "files",
        "bytes",
        "inventory",
        "policy_fingerprint",
    }:
        raise TrainingError("Guarded merge base-model identity mismatch")
    inventory = base_model.get("inventory")
    if not isinstance(inventory, list) or not inventory:
        raise TrainingError("Guarded merge base-model inventory is invalid")
    normalized_inventory: list[dict[str, Any]] = []
    for entry in inventory:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256", "bytes"}:
            raise TrainingError("Guarded merge base-model inventory is invalid")
        relative = entry.get("path")
        digest = entry.get("sha256")
        size = entry.get("bytes")
        pure = PurePosixPath(relative) if isinstance(relative, str) else None
        if (
            pure is None
            or pure.is_absolute()
            or not pure.parts
            or ".." in pure.parts
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or type(size) is not int
            or size < 0
        ):
            raise TrainingError("Guarded merge base-model inventory is invalid")
        normalized_inventory.append(dict(entry))
    expected_tree = sha256_bytes(canonical_json_bytes(normalized_inventory))
    if (
        [entry["path"] for entry in normalized_inventory]
        != sorted(entry["path"] for entry in normalized_inventory)
        or len({entry["path"] for entry in normalized_inventory})
        != len(normalized_inventory)
        or base_model.get("model_path") != str(Path(runtime["model_path"]).resolve())
        or base_model.get("tree_sha256") != runtime["model_snapshot_tree_sha256"]
        or base_model.get("tree_sha256") != expected_tree
        or base_model.get("files") != len(normalized_inventory)
        or base_model.get("bytes")
        != sum(entry["bytes"] for entry in normalized_inventory)
        or base_model.get("policy_fingerprint")
        != sha256_bytes(canonical_json_bytes(config["policy"]))
    ):
        raise TrainingError("Guarded merge base-model identity mismatch")
    if verify_live_base:
        live_base = hash_model_snapshot_tree(runtime["model_path"])
        expected_base = {
            "model_path": str(Path(runtime["model_path"]).resolve()),
            "tree_sha256": live_base["tree_sha256"],
            "files": live_base["files"],
            "bytes": live_base["bytes"],
            "inventory": live_base["inventory"],
            "policy_fingerprint": sha256_bytes(
                canonical_json_bytes(config["policy"])
            ),
        }
        if live_base["tree_sha256"] != runtime["model_snapshot_tree_sha256"] or (
            base_model != expected_base
        ):
            raise TrainingError("Guarded merge base-model identity mismatch")
    merge = receipt.get("merge")
    expected_argv = list(_expected_merge_argv(plan, run_dir, export_dir))
    expected_cwd = str(Path(runtime["verl_source_path"]).resolve())
    if (
        not isinstance(merge, dict)
        or set(merge) != {"argv", "cwd", "environment"}
        or merge.get("argv") != expected_argv
        or merge.get("cwd") != expected_cwd
    ):
        raise TrainingError("Guarded merge invocation mismatch")
    environment = merge.get("environment")
    allowed_environment = set(_MERGE_ENVIRONMENT_NAMES) | {
        "HF_DATASETS_OFFLINE",
        "HF_HUB_OFFLINE",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONHASHSEED",
        "PYTHONNOUSERSITE",
        "PYTHONPATH",
        "TRANSFORMERS_OFFLINE",
    }
    if (
        not isinstance(environment, dict)
        or not set(environment).issubset(allowed_environment)
        or environment.get("HF_DATASETS_OFFLINE") != "1"
        or environment.get("HF_HUB_OFFLINE") != "1"
        or environment.get("TRANSFORMERS_OFFLINE") != "1"
        or environment.get("PYTHONPATH") != expected_cwd
        or not all(isinstance(value, str) for value in environment.values())
    ):
        raise TrainingError("Guarded merge environment is invalid")
    fixed_environment = {
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": expected_cwd,
        "TRANSFORMERS_OFFLINE": "1",
    }
    if any(environment.get(name) != value for name, value in fixed_environment.items()):
        raise TrainingError("Guarded merge environment is invalid")
    expected_actor = {
        "path": expected_argv[7],
        "tree_sha256": actor_digest,
        "files": actor_inventory,
    }
    expected_export = {
        "path": str(export_dir),
        "tree_sha256": export_digest,
        "files": export_inventory,
    }
    if receipt.get("actor") != expected_actor:
        raise TrainingError("Guarded merge actor checkpoint changed")
    if receipt.get("export") != expected_export:
        raise TrainingError("Guarded merge export changed")
    if export_digest == runtime["model_snapshot_tree_sha256"]:
        raise TrainingError(
            "Merged export must differ from the immutable initial model snapshot"
        )
    expected_log = run_dir / f"logs/export-step-{config['grpo']['max_steps']}.log"
    _verify_reference(receipt.get("log"), expected_log, "Guarded merge log")
    return receipt


def register_final_checkpoint(
    run_dir: Path,
    export_dir: Path,
    *,
    force: bool = False,
    _lease: LaunchLease | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    run_dir = run_dir.resolve()
    export_dir = export_dir.resolve()
    if _lease is None:
        with exclusive_launch(run_dir) as lease:
            return register_final_checkpoint(
                run_dir,
                export_dir,
                force=force,
                _lease=lease,
            )
    _lease.assert_for(run_dir)
    plan, _ = load_training_plan(run_dir)
    final_step = plan["config"]["grpo"]["max_steps"]
    selected = plan["config"]["checkpointing"]["selected_checkpoint"]
    if selected != "fixed_final_step":
        raise TrainingError("Only the predeclared fixed final checkpoint may be registered")
    runtime = plan["config"]["runtime"]
    world_size = runtime["nodes"] * runtime["gpus_per_node"]
    tracker = run_dir / "checkpoints/latest_checkpointed_iteration.txt"
    try:
        tracked_step = int(tracker.read_text(encoding="utf-8").strip())
    except (OSError, ValueError) as exc:
        raise TrainingError(f"Cannot verify verl final-step tracker {tracker}: {exc}") from exc
    if tracked_step != final_step:
        raise TrainingError(f"Expected completed step {final_step}, found {tracked_step}")
    actor_dir = validate_verl_checkpoint(
        run_dir,
        final_step,
        expected_world_size=world_size,
    )
    export_dir = validate_export_destination(run_dir, actor_dir, export_dir)
    actor_inventory, actor_digest = directory_inventory(actor_dir)
    export_inventory, export_digest = directory_inventory(export_dir)
    _validate_huggingface_export(export_dir, export_inventory)
    _verify_merge_receipt(
        run_dir,
        export_dir,
        plan,
        actor_inventory,
        actor_digest,
        export_inventory,
        export_digest,
    )
    receipt_reference = _artifact(
        run_dir / MERGE_RECEIPT_NAME,
        relative_name=MERGE_RECEIPT_NAME,
    )
    checkpoint_manifest = {
        "schema_version": 2,
        "format": "huggingface",
        "training_plan_fingerprint": plan["plan_fingerprint"],
        "selected_by": "fixed_final_step",
        "step": final_step,
        "base_model_fingerprint": sha256_bytes(
            canonical_json_bytes(plan["config"]["policy"])
        ),
        "merge_receipt": receipt_reference,
        "verl_actor_checkpoint": {
            "path": str(actor_dir),
            "tree_sha256": actor_digest,
            "files": actor_inventory,
        },
        "export": {
            "path": str(export_dir),
            "tree_sha256": export_digest,
            "files": export_inventory,
        },
    }
    manifest_path = run_dir / CHECKPOINT_MANIFEST_NAME
    publish_json(manifest_path, checkpoint_manifest, force=force)
    manifest_reference = artifact_reference(manifest_path)
    manifest_reference["path"] = CHECKPOINT_MANIFEST_NAME
    reasons = [
        "external_anchor_service_not_content_attested",
        "distributed_runtime_not_content_attested",
        "checkpoint_export_lineage_not_content_attested",
    ]
    completion = {
        "schema_version": 2,
        "training_plan_fingerprint": plan["plan_fingerprint"],
        "completed_step": final_step,
        "selection": "fixed_final_step_without_labels",
        "checkpoint_manifest": manifest_reference,
        "merge_receipt": receipt_reference,
        "export_tree_sha256": export_digest,
        "labels_loaded": False,
        "reportable": False,
        "non_reportable_reasons": reasons,
    }
    publish_json(run_dir / COMPLETION_NAME, completion, force=force)
    return checkpoint_manifest, completion


def export_and_register_final_checkpoint(
    config: TrainingConfig,
    run_dir: Path,
    export_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    run_dir = run_dir.resolve()
    lexical_export = Path(os.path.abspath(export_dir))
    with exclusive_launch(run_dir) as lease:
        plan, _ = load_training_plan(run_dir)
        if plan["config_fingerprint"] != config.fingerprint:
            raise TrainingError("Config no longer matches the prepared training plan")
        config.assert_runnable()
        if os.path.lexists(lexical_export):
            raise TrainingError(
                "Guarded export requires a fresh nonexistent export directory"
            )
        parent = lexical_export.parent
        if (
            not parent.is_dir()
            or parent.is_symlink()
            or parent.resolve(strict=True) != parent
        ):
            raise TrainingError(
                "Guarded export requires an existing canonical parent directory"
            )
        python = Path(config.runtime.python_executable).resolve()
        source = Path(config.runtime.verl_source_path).resolve()
        base_model = Path(config.runtime.model_path).resolve()
        if not python.is_file() or not os.access(python, os.X_OK):
            raise TrainingError(f"Configured Python is not executable: {python}")
        verify_verl_checkout(source, config.runtime.framework_revision)
        base_identity = hash_model_snapshot_tree(base_model)
        if base_identity["tree_sha256"] != config.runtime.model_snapshot_tree_sha256:
            raise TrainingError("Initial model snapshot changed before guarded export")
        final_step = config.grpo.max_steps
        actor_dir = validate_verl_checkpoint(
            run_dir,
            final_step,
            expected_world_size=config.runtime.nodes * config.runtime.gpus_per_node,
        )
        export_dir = validate_export_destination(run_dir, actor_dir, lexical_export)
        if (
            export_dir == base_model
            or export_dir in base_model.parents
            or base_model in export_dir.parents
        ):
            raise TrainingError("Export directory must not overlap the base model")
        argv = merge_command(
            config,
            run_dir=run_dir,
            export_directory=export_dir,
        )
        expected_argv = _expected_merge_argv(plan, run_dir, export_dir)
        if argv != expected_argv:
            raise TrainingError("Generated merge argv differs from the pinned contract")
        actor_before, actor_digest_before = directory_inventory(actor_dir)
        tracker = run_dir / "checkpoints/latest_checkpointed_iteration.txt"
        tracker_before = artifact_reference(tracker)
        receipt_path = run_dir / MERGE_RECEIPT_NAME
        manifest_path = run_dir / CHECKPOINT_MANIFEST_NAME
        completion_path = run_dir / COMPLETION_NAME
        log_path = run_dir / f"logs/export-step-{final_step}.log"
        if any(
            os.path.lexists(path)
            for path in (receipt_path, manifest_path, completion_path, log_path)
        ):
            raise TrainingError(
                "Guarded export artifacts already exist; verify them instead of rerunning"
            )
        environment = _merge_environment(source)
        os.mkdir(lexical_export, mode=0o700)
        export_dir = lexical_export.resolve(strict=True)
        if export_dir != Path(expected_argv[-1]):
            raise TrainingError("Atomic export directory identity changed")
        return_code = run_command_with_log(
            argv,
            cwd=source,
            environment=environment,
            log_path=log_path,
        )
        if return_code:
            raise TrainingError(f"Verl model merger exited with status {return_code}")
        actor_after, actor_digest_after = directory_inventory(actor_dir)
        if (
            actor_after != actor_before
            or actor_digest_after != actor_digest_before
            or artifact_reference(tracker) != tracker_before
        ):
            raise TrainingError("Actor checkpoint changed during guarded export")
        export_inventory, export_digest = directory_inventory(export_dir)
        _validate_huggingface_export(export_dir, export_inventory)
        if export_digest == config.runtime.model_snapshot_tree_sha256:
            raise TrainingError(
                "Merged export must differ from the immutable initial model snapshot"
            )
        receipt = {
            "schema_version": 1,
            "kind": MERGE_RECEIPT_KIND,
            "training_plan_fingerprint": plan["plan_fingerprint"],
            "config_fingerprint": plan["config_fingerprint"],
            "training_plan": _artifact(
                run_dir / TRAINING_MANIFEST_NAME,
                relative_name=TRAINING_MANIFEST_NAME,
            ),
            "step": final_step,
            "tracker": tracker_before,
            "merge": {
                "argv": list(argv),
                "cwd": str(source),
                "environment": environment,
            },
            "runtime_identity": {
                "framework_revision": config.runtime.framework_revision,
                "package_inventory_sha256": config.runtime.package_inventory_sha256,
                "trainer_image_digest": config.runtime.trainer_image_digest,
            },
            "base_model": {
                "model_path": str(base_model),
                "tree_sha256": base_identity["tree_sha256"],
                "files": base_identity["files"],
                "bytes": base_identity["bytes"],
                "inventory": base_identity["inventory"],
                "policy_fingerprint": sha256_bytes(
                    canonical_json_bytes(plan["config"]["policy"])
                ),
            },
            "actor": {
                "path": str(actor_dir),
                "tree_sha256": actor_digest_after,
                "files": actor_after,
            },
            "export": {
                "path": str(export_dir),
                "tree_sha256": export_digest,
                "files": export_inventory,
            },
            "log": artifact_reference(log_path),
        }
        receipt["receipt_fingerprint"] = _receipt_fingerprint(receipt)
        publish_json(receipt_path, receipt)
        manifest, completion = register_final_checkpoint(
            run_dir,
            export_dir,
            _lease=lease,
        )
        return manifest, completion, receipt


def _load_registered_checkpoint(
    run_dir: Path, *, verify_live_base: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    run_dir = run_dir.resolve()
    plan, _ = load_training_plan(run_dir)
    manifest_path = run_dir / CHECKPOINT_MANIFEST_NAME
    completion_path = run_dir / COMPLETION_NAME
    manifest = read_json(manifest_path)
    completion = read_json(completion_path)
    if manifest.get("schema_version") != 2 or completion.get("schema_version") != 2:
        raise TrainingError("Registered checkpoint schema is obsolete or invalid")
    if manifest.get("training_plan_fingerprint") != plan["plan_fingerprint"]:
        raise TrainingError("Checkpoint manifest belongs to another training plan")
    if completion.get("training_plan_fingerprint") != plan["plan_fingerprint"]:
        raise TrainingError("Completion artifact belongs to another training plan")
    reference = completion.get("checkpoint_manifest")
    if not isinstance(reference, dict):
        raise TrainingError("Completion is missing its checkpoint-manifest reference")
    digest, size = file_digest(manifest_path)
    if reference.get("sha256") != digest or reference.get("bytes") != size:
        raise TrainingError("Registered checkpoint manifest changed")
    export = manifest.get("export")
    actor = manifest.get("verl_actor_checkpoint")
    if not isinstance(export, dict) or not isinstance(actor, dict):
        raise TrainingError("Checkpoint manifest has invalid directory references")
    for name, spec in (("export", export), ("actor", actor)):
        inventory, tree_digest = directory_inventory(Path(spec["path"]))
        if inventory != spec.get("files") or tree_digest != spec.get("tree_sha256"):
            raise TrainingError(f"Registered {name} checkpoint changed")
    receipt_path = run_dir / MERGE_RECEIPT_NAME
    for owner, reference in (
        ("manifest", manifest.get("merge_receipt")),
        ("completion", completion.get("merge_receipt")),
    ):
        _verify_reference(reference, receipt_path, f"{owner} merge receipt")
    _verify_merge_receipt(
        run_dir,
        Path(export["path"]),
        plan,
        actor["files"],
        actor["tree_sha256"],
        export["files"],
        export["tree_sha256"],
        verify_live_base=verify_live_base,
    )
    if completion.get("export_tree_sha256") != export["tree_sha256"]:
        raise TrainingError("Completion export digest mismatch")
    return manifest, completion


def load_registered_checkpoint(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify a registered checkpoint, including the live immutable base model."""

    return _load_registered_checkpoint(run_dir, verify_live_base=True)


def load_registered_checkpoint_artifacts(
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify checkpoint artifacts and live actor/export trees without opening the base model."""

    return _load_registered_checkpoint(run_dir, verify_live_base=False)
