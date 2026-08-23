"""Derived, non-reportable qualification plans for the canonical training run."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from compute_as_a_teacher.evaluation.artifacts import (
    artifact_reference,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    publish_bytes,
    read_json,
    read_jsonl,
    sha256_bytes,
)

from .config import TrainingConfig
from .errors import TrainingError
from .planning import MANIFEST_NAME, TRAINING_DATA_NAME, load_training_plan
from .verl_adapter import (
    LaunchLease,
    VerlCommand,
    build_process_environment,
    checkpointed_step,
    command_from_dict,
    exclusive_launches,
    qualification_wandb_group,
    qualification_wandb_run_id,
    require_label_free_training_outputs,
    run_command_with_log,
    verify_verl_checkout,
)


QUALIFICATION_MANIFEST_NAME = "qualification_manifest.json"
QUALIFICATION_COMMAND_NAME = "verl_command.json"
QUALIFICATION_DATA_NAME = "math500_train.jsonl"


@dataclass(frozen=True, slots=True)
class QualificationProfile:
    name: str
    prompt_count: int
    prompt_batch_size: int
    max_steps: int
    save_every_steps: int


QUALIFICATION_PROFILES = {
    profile.name: profile
    for profile in (
        QualificationProfile("one_step", 8, 8, 1, 1),
        QualificationProfile("resume_three_step", 8, 8, 3, 1),
        QualificationProfile("full_shape_five_step", 500, 256, 5, 5),
    )
}


def _require_disjoint_run_directories(source: Path, qualification: Path) -> None:
    if source == qualification or source in qualification.parents or qualification in source.parents:
        raise TrainingError(
            "Qualification and canonical run directories must not contain each other"
        )


def _replace_override(argv: list[str], key: str, value: str) -> None:
    prefix = f"{key}="
    indexes = [index for index, item in enumerate(argv) if item.startswith(prefix)]
    if len(indexes) != 1:
        raise TrainingError(f"Expected exactly one planned override for {key}")
    argv[indexes[0]] = f"{key}={value}"


def derive_qualification_command(
    source: VerlCommand,
    profile: QualificationProfile,
    qualification_dir: Path,
    *,
    run_name: str,
) -> VerlCommand:
    qualification_dir = qualification_dir.resolve()
    data_path = qualification_dir / QUALIFICATION_DATA_NAME
    hydra_dir = qualification_dir / "hydra"
    checkpoint_dir = qualification_dir / "checkpoints"
    rollout_dir = qualification_dir / "rollout_logs"
    argv = list(source.argv)
    replacements = {
        "hydra.run.dir": json.dumps(str(hydra_dir), ensure_ascii=False),
        "data.train_files": json.dumps(str(data_path), ensure_ascii=False),
        "data.val_files": json.dumps(str(data_path), ensure_ascii=False),
        "data.train_batch_size": str(profile.prompt_batch_size),
        "data.val_batch_size": str(profile.prompt_batch_size),
        "actor_rollout_ref.actor.ppo_mini_batch_size": str(
            profile.prompt_batch_size
        ),
        "trainer.total_training_steps": str(profile.max_steps),
        "trainer.total_epochs": str(profile.max_steps),
        "trainer.save_freq": str(profile.save_every_steps),
        "trainer.default_local_dir": json.dumps(
            str(checkpoint_dir), ensure_ascii=False
        ),
        "trainer.experiment_name": json.dumps(
            f"{run_name}-{profile.name}-nonreportable", ensure_ascii=False
        ),
    }
    for key, value in replacements.items():
        _replace_override(argv, key, value)
    if any(item.startswith("trainer.rollout_data_dir=") for item in argv):
        _replace_override(
            argv,
            "trainer.rollout_data_dir",
            json.dumps(str(rollout_dir), ensure_ascii=False),
        )
    else:
        argv.append(
            "trainer.rollout_data_dir="
            + json.dumps(str(rollout_dir), ensure_ascii=False)
        )
    environment = dict(source.environment)
    canonical_wandb_id = environment.get("WANDB_RUN_ID")
    if canonical_wandb_id is not None:
        environment["WANDB_RUN_ID"] = qualification_wandb_run_id(
            canonical_wandb_id,
            profile.name,
        )
        environment["WANDB_DIR"] = str((qualification_dir / "wandb").resolve())
        environment["WANDB_RUN_GROUP"] = qualification_wandb_group(
            environment.get("WANDB_RUN_GROUP", ""),
            canonical_wandb_id,
            profile.name,
        )
        tags = [tag for tag in environment.get("WANDB_TAGS", "").split(",") if tag]
        for tag in ("qualification", "nonreportable", profile.name):
            if tag not in tags:
                tags.append(tag)
        environment["WANDB_TAGS"] = ",".join(tags)
    return VerlCommand(
        argv=tuple(argv),
        cwd=source.cwd,
        environment=environment,
        framework_revision=source.framework_revision,
        adapter_version=source.adapter_version,
    )


def _artifact(path: Path, payload: bytes, *, rows: int | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "path": path.name,
        "sha256": sha256_bytes(payload),
        "bytes": len(payload),
    }
    if rows is not None:
        value["rows"] = rows
    return value


def write_qualification_plan(
    source_run_dir: Path,
    qualification_dir: Path,
    profile_name: str,
    *,
    force: bool = False,
    _leases: Mapping[Path, LaunchLease] | None = None,
) -> dict[str, Any]:
    source_run_dir = source_run_dir.resolve()
    qualification_dir = qualification_dir.resolve()
    _require_disjoint_run_directories(source_run_dir, qualification_dir)
    if _leases is None:
        qualification_dir.mkdir(parents=True, exist_ok=True)
        with exclusive_launches((source_run_dir, qualification_dir)) as leases:
            return write_qualification_plan(
                source_run_dir,
                qualification_dir,
                profile_name,
                force=force,
                _leases=leases,
            )
    _leases[source_run_dir].assert_for(source_run_dir)
    _leases[qualification_dir].assert_for(qualification_dir)
    profile = QUALIFICATION_PROFILES.get(profile_name)
    if profile is None:
        raise TrainingError(f"Unknown qualification profile: {profile_name}")
    source_manifest, source_command = load_training_plan(source_run_dir)
    rows = read_jsonl(source_run_dir / TRAINING_DATA_NAME)
    if len(rows) != 500:
        raise TrainingError("The canonical training plan must contain 500 rows")
    selected_rows = rows[: profile.prompt_count]
    if any(row.get("reward_model", {}).get("ground_truth") is not None for row in selected_rows):
        raise TrainingError("Qualification data must remain reference-free")
    data_payload = canonical_jsonl_bytes(selected_rows)
    command = derive_qualification_command(
        source_command,
        profile,
        qualification_dir,
        run_name=source_manifest["run_name"],
    )
    command_payload = canonical_json_bytes(command.to_dict())
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "cat_grpo_qualification",
        "profile": {
            "name": profile.name,
            "prompt_count": profile.prompt_count,
            "prompt_batch_size": profile.prompt_batch_size,
            "max_steps": profile.max_steps,
            "save_every_steps": profile.save_every_steps,
        },
        "source": {
            "run_dir": str(source_run_dir),
            "training_plan": artifact_reference(source_run_dir / MANIFEST_NAME),
            "plan_fingerprint": source_manifest["plan_fingerprint"],
            "config_fingerprint": source_manifest["config_fingerprint"],
            "command_fingerprint": source_command.fingerprint,
        },
        "artifacts": {
            "training_data": _artifact(
                qualification_dir / QUALIFICATION_DATA_NAME,
                data_payload,
                rows=len(selected_rows),
            ),
            "verl_command": {
                **_artifact(
                    qualification_dir / QUALIFICATION_COMMAND_NAME,
                    command_payload,
                ),
                "fingerprint": command.fingerprint,
            },
        },
        "counts": {
            "problems": len(selected_rows),
            "rollouts_per_prompt": 8,
            "trajectories_per_update": profile.prompt_batch_size * 8,
            "anchor_calls_per_update": profile.prompt_batch_size,
            "max_steps": profile.max_steps,
        },
        "label_firewall": source_manifest["label_firewall"],
        "reportable": False,
        "non_reportable_reasons": [
            "derived_qualification_profile",
            "not_the_canonical_paper_run",
        ],
    }
    manifest["qualification_fingerprint"] = sha256_bytes(
        canonical_json_bytes(manifest)
    )
    manifest_payload = canonical_json_bytes(manifest)
    payloads = {
        QUALIFICATION_DATA_NAME: data_payload,
        QUALIFICATION_COMMAND_NAME: command_payload,
        QUALIFICATION_MANIFEST_NAME: manifest_payload,
    }
    descendants = [
        qualification_dir / name
        for name in ("checkpoints", "rollout_logs")
        if (qualification_dir / name).exists()
    ]
    mismatched = [
        qualification_dir / name
        for name, payload in payloads.items()
        if (qualification_dir / name).exists()
        and (
            not (qualification_dir / name).is_file()
            or (qualification_dir / name).read_bytes() != payload
        )
    ]
    if mismatched and descendants:
        raise TrainingError("Refusing to replace a qualification plan with run artifacts")
    if mismatched and not force:
        raise TrainingError(f"Refusing to replace qualification plan: {mismatched}")
    for name, payload in payloads.items():
        publish_bytes(qualification_dir / name, payload, force=force)
    return manifest


def load_qualification_plan(
    qualification_dir: Path,
) -> tuple[dict[str, Any], VerlCommand]:
    qualification_dir = qualification_dir.resolve()
    manifest = read_json(qualification_dir / QUALIFICATION_MANIFEST_NAME)
    fingerprint = manifest.get("qualification_fingerprint")
    unsigned = dict(manifest)
    unsigned.pop("qualification_fingerprint", None)
    if (
        set(manifest)
        != {
            "schema_version",
            "kind",
            "profile",
            "source",
            "artifacts",
            "counts",
            "label_firewall",
            "reportable",
            "non_reportable_reasons",
            "qualification_fingerprint",
        }
        or manifest.get("reportable") is not False
        or manifest.get("non_reportable_reasons")
        != ["derived_qualification_profile", "not_the_canonical_paper_run"]
        or manifest.get("schema_version") != 1
        or manifest.get("kind") != "cat_grpo_qualification"
        or fingerprint != sha256_bytes(canonical_json_bytes(unsigned))
    ):
        raise TrainingError("Invalid qualification manifest")
    profile_value = manifest.get("profile")
    if not isinstance(profile_value, dict):
        raise TrainingError("Qualification profile is missing")
    profile = QUALIFICATION_PROFILES.get(profile_value.get("name"))
    if profile is None or profile_value != {
        "name": profile.name,
        "prompt_count": profile.prompt_count,
        "prompt_batch_size": profile.prompt_batch_size,
        "max_steps": profile.max_steps,
        "save_every_steps": profile.save_every_steps,
    }:
        raise TrainingError("Qualification profile changed")
    source = manifest.get("source")
    if (
        not isinstance(source, dict)
        or set(source)
        != {
            "run_dir",
            "training_plan",
            "plan_fingerprint",
            "config_fingerprint",
            "command_fingerprint",
        }
        or not isinstance(source.get("run_dir"), str)
    ):
        raise TrainingError("Qualification source plan is missing")
    source_dir = Path(source["run_dir"]).resolve()
    _require_disjoint_run_directories(source_dir, qualification_dir)
    source_manifest, source_command = load_training_plan(source_dir)
    if (
        source.get("plan_fingerprint") != source_manifest["plan_fingerprint"]
        or source.get("config_fingerprint") != source_manifest["config_fingerprint"]
        or source.get("command_fingerprint") != source_command.fingerprint
        or source.get("training_plan")
        != artifact_reference(source_dir / MANIFEST_NAME)
    ):
        raise TrainingError("Qualification source lineage changed")
    expected_counts = {
        "problems": profile.prompt_count,
        "rollouts_per_prompt": 8,
        "trajectories_per_update": profile.prompt_batch_size * 8,
        "anchor_calls_per_update": profile.prompt_batch_size,
        "max_steps": profile.max_steps,
    }
    if (
        manifest.get("counts") != expected_counts
        or manifest.get("label_firewall") != source_manifest["label_firewall"]
    ):
        raise TrainingError("Qualification counts or label firewall changed")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "training_data",
        "verl_command",
    }:
        raise TrainingError("Qualification artifact registry changed")
    for key, filename in (
        ("training_data", QUALIFICATION_DATA_NAME),
        ("verl_command", QUALIFICATION_COMMAND_NAME),
    ):
        reference = artifacts.get(key)
        path = qualification_dir / filename
        expected_keys = (
            {"path", "sha256", "bytes", "rows"}
            if key == "training_data"
            else {"path", "sha256", "bytes", "fingerprint"}
        )
        if (
            not isinstance(reference, dict)
            or set(reference) != expected_keys
            or reference.get("path") != filename
            or (key == "training_data" and reference.get("rows") != profile.prompt_count)
            or not path.is_file()
        ):
            raise TrainingError(f"Missing qualification artifact: {filename}")
        payload = path.read_bytes()
        if (
            reference.get("sha256") != sha256_bytes(payload)
            or reference.get("bytes") != len(payload)
        ):
            raise TrainingError(f"Qualification artifact changed: {filename}")
    rows = read_jsonl(qualification_dir / QUALIFICATION_DATA_NAME)
    if len(rows) != profile.prompt_count or any(
        row.get("reward_model", {}).get("ground_truth") is not None for row in rows
    ):
        raise TrainingError("Qualification data violates its label-free profile")
    command = command_from_dict(read_json(qualification_dir / QUALIFICATION_COMMAND_NAME))
    if command.fingerprint != manifest["artifacts"]["verl_command"]["fingerprint"]:
        raise TrainingError("Qualification command fingerprint mismatch")
    expected_command = derive_qualification_command(
        source_command,
        profile,
        qualification_dir,
        run_name=source_manifest["run_name"],
    )
    if command != expected_command:
        raise TrainingError("Qualification command is not the registered derivation")
    return manifest, command


def launch_qualification(
    qualification_dir: Path,
    command: VerlCommand,
    config: TrainingConfig,
    *,
    lease: LaunchLease | None = None,
    _locked_source_run_dir: Path | None = None,
) -> int:
    qualification_dir = qualification_dir.resolve()
    if lease is None:
        initial_manifest, _ = load_qualification_plan(qualification_dir)
        source_run_dir = Path(initial_manifest["source"]["run_dir"]).resolve()
        with exclusive_launches((source_run_dir, qualification_dir)) as leases:
            return launch_qualification(
                qualification_dir,
                command,
                config,
                lease=leases[qualification_dir],
                _locked_source_run_dir=source_run_dir,
            )
    lease.assert_for(qualification_dir)
    require_label_free_training_outputs()
    manifest, loaded_command = load_qualification_plan(qualification_dir)
    current_source = Path(manifest["source"]["run_dir"]).resolve()
    if (
        _locked_source_run_dir is not None
        and current_source != _locked_source_run_dir.resolve()
    ):
        raise TrainingError("Qualification source changed during lock acquisition")
    if loaded_command != command:
        raise TrainingError("Qualification command no longer matches its plan")
    if manifest["source"]["config_fingerprint"] != config.fingerprint:
        raise TrainingError("Config no longer matches the qualification source plan")
    profile = QUALIFICATION_PROFILES[manifest["profile"]["name"]]
    world_size = config.runtime.nodes * config.runtime.gpus_per_node
    if checkpointed_step(
        qualification_dir,
        profile.max_steps,
        expected_world_size=world_size,
    ) == profile.max_steps:
        raise TrainingError("Qualification run already reached its terminal step")
    python = Path(config.runtime.python_executable).resolve()
    model = Path(config.runtime.model_path).resolve()
    source = Path(config.runtime.verl_source_path).resolve()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise TrainingError(f"Configured Python is not executable: {python}")
    if not model.is_dir():
        raise TrainingError(f"Configured local model snapshot is missing: {model}")
    if config.runtime.download_allowed:
        raise TrainingError("Qualification runs require runtime.download_allowed=false")
    if config.runtime.anchor_api_key_env not in os.environ:
        raise TrainingError(
            f"Anchor API key environment variable is unset: {config.runtime.anchor_api_key_env}"
        )
    verify_verl_checkout(source, command.framework_revision)
    if command.argv[0] != config.runtime.python_executable or Path(command.cwd).resolve() != source:
        raise TrainingError("Qualification command runtime no longer matches the config")
    environment = build_process_environment(
        config,
        command,
        qualification_dir,
        qualification_profile=profile.name,
    )
    return run_command_with_log(
        command.argv,
        cwd=source,
        environment=environment,
        log_path=qualification_dir / "logs" / "trainer.log",
    )
