"""Content-addressed registration of the fixed final verl checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from compute_as_a_teacher.evaluation.artifacts import (
    artifact_reference,
    canonical_json_bytes,
    file_digest,
    publish_json,
    read_json,
    sha256_bytes,
)

from .errors import TrainingError
from .planning import load_training_plan
from .verl_adapter import (
    LaunchLease,
    exclusive_launch,
    validate_export_destination,
    validate_verl_checkpoint,
)


CHECKPOINT_MANIFEST_NAME = "checkpoint_manifest.json"
COMPLETION_NAME = "completion.json"


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
    if not any(path.endswith((".safetensors", ".bin")) for path in paths):
        raise TrainingError("Merged Hugging Face checkpoint has no model weight file")


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
    checkpoint_manifest = {
        "schema_version": 1,
        "format": "huggingface",
        "training_plan_fingerprint": plan["plan_fingerprint"],
        "selected_by": "fixed_final_step",
        "step": final_step,
        "base_model_fingerprint": sha256_bytes(
            canonical_json_bytes(plan["config"]["policy"])
        ),
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
        "schema_version": 1,
        "training_plan_fingerprint": plan["plan_fingerprint"],
        "completed_step": final_step,
        "selection": "fixed_final_step_without_labels",
        "checkpoint_manifest": manifest_reference,
        "export_tree_sha256": export_digest,
        "labels_loaded": False,
        "reportable": False,
        "non_reportable_reasons": reasons,
    }
    publish_json(run_dir / COMPLETION_NAME, completion, force=force)
    return checkpoint_manifest, completion


def load_registered_checkpoint(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    run_dir = run_dir.resolve()
    plan, _ = load_training_plan(run_dir)
    manifest_path = run_dir / CHECKPOINT_MANIFEST_NAME
    completion_path = run_dir / COMPLETION_NAME
    manifest = read_json(manifest_path)
    completion = read_json(completion_path)
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
    if completion.get("export_tree_sha256") != export["tree_sha256"]:
        raise TrainingError("Completion export digest mismatch")
    return manifest, completion
