"""Content-addressed gate for launching the canonical MATH-500 training run."""

from __future__ import annotations

from datetime import datetime
import math
from pathlib import Path
import re
from typing import Any, Mapping

from compute_as_a_teacher.evaluation.artifacts import (
    artifact_reference,
    canonical_json_bytes,
    publish_json,
    read_json,
    sha256_bytes,
)
from compute_as_a_teacher.evaluation.errors import EvaluationError

from .checkpoints import directory_inventory
from .errors import TrainingError
from .experiment_registry import verify_preregistered_training_stage
from .planning import MANIFEST_NAME, load_training_plan
from .preflight import PREFLIGHT_NAME, validate_anchor_probe_result
from .qualification import (
    QUALIFICATION_MANIFEST_NAME,
    QUALIFICATION_PROFILES,
    load_qualification_plan,
)
from .verl_adapter import (
    LaunchLease,
    checkpointed_step,
    exclusive_launch,
    exclusive_launches,
)


LAUNCH_APPROVAL_KIND = "cat_canonical_launch_approval"
LAUNCH_EVIDENCE_KIND = "cat_canonical_launch_evidence"
MANUAL_ATTESTATION_KIND = "cat_canonical_launch_manual_attestation"
SCHEMA_VERSION = 1
_SHA256 = re.compile(r"[0-9a-f]{64}")
_PLACEHOLDER = re.compile(
    r"(?:replace[_ -]?with|placeholder|todo|tbd|n/?a|none)(?![A-Za-z0-9])",
    re.I,
)
REQUIRED_QUALIFICATION_PROFILES = (
    "one_step",
    "resume_three_step",
    "full_shape_five_step",
)
REQUIRED_MANUAL_ATTESTATIONS = (
    "qualification_metrics_reviewed_and_accepted",
    "kill_and_resume_verified",
    "frozen_anchor_identity_and_hardware_disjointness_verified",
    "anchor_context_sampling_and_seed_contract_verified",
    "time_cost_and_storage_budget_approved",
)
REQUIRED_BUDGET_LIMITS = (
    "max_wall_time_seconds",
    "max_trainer_gpu_hours",
    "max_anchor_gpu_hours",
    "max_storage_bytes",
    "max_total_cost",
    "currency",
    "trainer_gpu_hour_rate",
    "anchor_gpu_hour_rate",
    "storage_and_network_cost",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TrainingError(message)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise TrainingError(f"{name} must be an object")
    return value


def _artifact(path: Path, name: str, *, nonempty: bool = True) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise TrainingError(f"{name} is missing: {path}")
    reference = artifact_reference(path)
    if nonempty and reference["bytes"] <= 0:
        raise TrainingError(f"{name} is empty: {path}")
    return reference


def _directory(path: Path, name: str) -> dict[str, Any]:
    path = path.resolve()
    try:
        inventory, digest = directory_inventory(path)
    except TrainingError:
        raise
    return {
        "path": str(path),
        "tree_sha256": digest,
        "files": inventory,
    }


def _number(value: Any, name: str, *, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingError(f"{name} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized) or (normalized <= 0 if positive else normalized < 0):
        qualifier = "positive" if positive else "nonnegative"
        raise TrainingError(f"{name} must be a finite {qualifier} number")
    return normalized


def _budget_limits(value: Any) -> dict[str, Any]:
    limits = _mapping(value, "budget_limits")
    if set(limits) != set(REQUIRED_BUDGET_LIMITS):
        raise TrainingError("Manual launch attestation has invalid budget limits")
    wall_time = limits.get("max_wall_time_seconds")
    storage = limits.get("max_storage_bytes")
    if type(wall_time) is not int or wall_time <= 0:
        raise TrainingError("max_wall_time_seconds must be a positive integer")
    if type(storage) is not int or storage <= 0:
        raise TrainingError("max_storage_bytes must be a positive integer")
    trainer_hours = _number(
        limits.get("max_trainer_gpu_hours"),
        "max_trainer_gpu_hours",
        positive=True,
    )
    anchor_hours = _number(
        limits.get("max_anchor_gpu_hours"),
        "max_anchor_gpu_hours",
        positive=True,
    )
    total_cost = _number(
        limits.get("max_total_cost"), "max_total_cost", positive=False
    )
    trainer_rate = _number(
        limits.get("trainer_gpu_hour_rate"),
        "trainer_gpu_hour_rate",
        positive=False,
    )
    anchor_rate = _number(
        limits.get("anchor_gpu_hour_rate"),
        "anchor_gpu_hour_rate",
        positive=False,
    )
    fixed_cost = _number(
        limits.get("storage_and_network_cost"),
        "storage_and_network_cost",
        positive=False,
    )
    currency = limits.get("currency")
    if not isinstance(currency, str) or not re.fullmatch(r"[A-Z]{3}", currency):
        raise TrainingError("budget currency must be a three-letter uppercase code")
    priced_ceiling = trainer_hours * trainer_rate + anchor_hours * anchor_rate + fixed_cost
    if total_cost + 1e-9 < priced_ceiling:
        raise TrainingError("max_total_cost does not cover the declared resource ceilings")
    return dict(limits)


def _manual_attestation(
    path: Path,
    *,
    reviewed_evidence_fingerprint: str,
) -> dict[str, Any]:
    path = path.resolve()
    try:
        value = read_json(path)
    except EvaluationError as exc:
        raise TrainingError(str(exc)) from exc
    expected = {
        "schema_version",
        "kind",
        "attested_by",
        "attested_at_utc",
        "reviewed_evidence_fingerprint",
        "attestations",
        "evidence",
        "budget_limits",
    }
    if set(value) != expected:
        raise TrainingError("Manual launch attestation has an invalid schema")
    attested_by = value.get("attested_by")
    attested_at = value.get("attested_at_utc")
    if (
        not isinstance(attested_by, str)
        or len(attested_by.strip()) < 3
        or _PLACEHOLDER.search(attested_by.strip())
    ):
        raise TrainingError("Manual launch attestation requires attested_by")
    if not isinstance(attested_at, str) or not attested_at.endswith("Z"):
        raise TrainingError("Manual launch attestation requires a UTC timestamp")
    try:
        datetime.fromisoformat(attested_at[:-1] + "+00:00")
    except ValueError as exc:
        raise TrainingError("Manual launch attestation timestamp is invalid") from exc
    attestations = _mapping(value.get("attestations"), "manual attestations")
    evidence = _mapping(value.get("evidence"), "manual attestation evidence")
    if value.get("reviewed_evidence_fingerprint") != reviewed_evidence_fingerprint:
        raise TrainingError(
            "Manual launch attestation reviewed_evidence_fingerprint does not match"
        )
    required = set(REQUIRED_MANUAL_ATTESTATIONS)
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("kind") != MANUAL_ATTESTATION_KIND
        or set(attestations) != required
        or any(attestations.get(name) is not True for name in required)
        or set(evidence) != required
        or any(
            not isinstance(evidence.get(name), str)
            or len(evidence[name].strip()) < 16
            or _PLACEHOLDER.search(evidence[name].strip())
            for name in required
        )
    ):
        raise TrainingError(
            "Every required manual launch attestation and evidence note must be present"
        )
    budget_limits = _budget_limits(value.get("budget_limits"))
    return {
        "artifact": _artifact(path, "manual launch attestation"),
        "semantic_fingerprint": sha256_bytes(canonical_json_bytes(value)),
        "attested_by": attested_by,
        "attested_at_utc": attested_at,
        "reviewed_evidence_fingerprint": reviewed_evidence_fingerprint,
        "attestations": dict(attestations),
        "evidence": dict(evidence),
        "budget_limits": budget_limits,
    }


def _preflight(
    qualification_dir: Path,
    qualification_manifest: Mapping[str, Any],
    command_fingerprint: str,
    training_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    path = qualification_dir / PREFLIGHT_NAME
    try:
        receipt = read_json(path)
    except EvaluationError as exc:
        raise TrainingError(str(exc)) from exc
    fingerprint = receipt.get("preflight_fingerprint")
    unsigned = dict(receipt)
    unsigned.pop("preflight_fingerprint", None)
    lineage = {
        "profile": qualification_manifest["profile"],
        "qualification_fingerprint": qualification_manifest[
            "qualification_fingerprint"
        ],
        "training_data_sha256": qualification_manifest["artifacts"][
            "training_data"
        ]["sha256"],
        "command_sha256": qualification_manifest["artifacts"]["verl_command"][
            "sha256"
        ],
    }
    checks = receipt.get("checks")
    anchor_check = checks.get("anchor") if isinstance(checks, dict) else None
    tokenizer_check = checks.get("tokenizer") if isinstance(checks, dict) else None
    model_tree_sha256 = (
        checks.get("model_snapshot", {}).get("all_files_tree_sha256")
        if isinstance(checks, dict)
        and isinstance(checks.get("model_snapshot"), dict)
        else None
    )
    if (
        receipt.get("schema_version") != 2
        or receipt.get("kind") != "cat_training_preflight"
        or fingerprint != sha256_bytes(canonical_json_bytes(unsigned))
        or receipt.get("training_plan_fingerprint")
        != training_manifest["plan_fingerprint"]
        or receipt.get("config_fingerprint")
        != training_manifest["config_fingerprint"]
        or receipt.get("command_fingerprint") != command_fingerprint
        or receipt.get("qualification_lineage") != lineage
        or receipt.get("operationally_ready_to_launch") is not True
        or receipt.get("missing_gates") != []
        or receipt.get("scientifically_attested") is not False
        or not isinstance(checks, dict)
        or set(checks)
        != {"model_snapshot", "runtime", "hydra_composition", "tokenizer", "anchor"}
        or not isinstance(anchor_check, dict)
        or not isinstance(tokenizer_check, dict)
        or not isinstance(checks.get("model_snapshot"), dict)
        or not isinstance(model_tree_sha256, str)
        or not _SHA256.fullmatch(model_tree_sha256)
    ):
        raise TrainingError("Qualification preflight receipt is incomplete or changed")
    runtime = _mapping(training_manifest["config"]["runtime"], "training runtime")
    validate_anchor_probe_result(
        anchor_check,
        expected_model=str(runtime.get("anchor_model")),
        require_long_context=True,
    )
    canary_required = tokenizer_check.get("anchor_context_canary_required_tokens")
    model_context = tokenizer_check.get("model_context_tokens")
    if (
        type(canary_required) is not int
        or canary_required <= 0
        or type(model_context) is not int
        or model_context < canary_required
    ):
        raise TrainingError("Qualification tokenizer context evidence is incomplete")
    return {
        "artifact": _artifact(path, "qualification preflight receipt"),
        "fingerprint": fingerprint,
    }


def _qualification_stage(
    profile_name: str,
    qualification_dir: Path,
    training_run_dir: Path,
    training_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    qualification_dir = qualification_dir.resolve()
    manifest, command = load_qualification_plan(qualification_dir)
    profile = QUALIFICATION_PROFILES[profile_name]
    _require(
        manifest.get("profile")
        == {
            "name": profile.name,
            "prompt_count": profile.prompt_count,
            "prompt_batch_size": profile.prompt_batch_size,
            "max_steps": profile.max_steps,
            "save_every_steps": profile.save_every_steps,
        },
        f"Qualification directory does not contain the {profile_name} profile",
    )
    source = _mapping(manifest.get("source"), "qualification source")
    _require(
        Path(str(source.get("run_dir"))).resolve() == training_run_dir.resolve()
        and source.get("plan_fingerprint") == training_manifest["plan_fingerprint"]
        and source.get("config_fingerprint") == training_manifest["config_fingerprint"],
        f"{profile_name} qualification belongs to another canonical plan",
    )
    runtime = _mapping(training_manifest["config"]["runtime"], "training runtime")
    world_size = runtime["nodes"] * runtime["gpus_per_node"]
    completed = checkpointed_step(
        qualification_dir,
        profile.max_steps,
        expected_world_size=world_size,
    )
    _require(
        completed == profile.max_steps,
        f"{profile_name} qualification has no verified terminal checkpoint",
    )
    tracker = qualification_dir / "checkpoints/latest_checkpointed_iteration.txt"
    terminal = (
        qualification_dir / "checkpoints" / f"global_step_{profile.max_steps}"
    )
    trainer_log = qualification_dir / "logs/trainer.log"
    rollout_logs = qualification_dir / "rollout_logs"
    return {
        "run_dir": str(qualification_dir),
        "profile": dict(manifest["profile"]),
        "manifest": _artifact(
            qualification_dir / QUALIFICATION_MANIFEST_NAME,
            f"{profile_name} qualification manifest",
        ),
        "qualification_fingerprint": manifest["qualification_fingerprint"],
        "command_fingerprint": command.fingerprint,
        "preflight": _preflight(
            qualification_dir,
            manifest,
            command.fingerprint,
            training_manifest,
        ),
        "terminal_checkpoint_step": completed,
        "checkpoint_tracker": _artifact(
            tracker, f"{profile_name} checkpoint tracker"
        ),
        "terminal_checkpoint": _directory(
            terminal, f"{profile_name} terminal checkpoint"
        ),
        "trainer_log": _artifact(trainer_log, f"{profile_name} trainer log"),
        "rollout_logs": _directory(
            rollout_logs, f"{profile_name} rollout logs"
        ),
        "reportable": False,
    }


def build_launch_evidence(
    preregistration_path: Path,
    training_run_dir: Path,
    one_step_dir: Path,
    resume_three_step_dir: Path,
    full_shape_five_step_dir: Path,
) -> dict[str, Any]:
    preregistration_path = preregistration_path.resolve()
    training_run_dir = training_run_dir.resolve()
    qualification_paths = {
        "one_step": one_step_dir,
        "resume_three_step": resume_three_step_dir,
        "full_shape_five_step": full_shape_five_step_dir,
    }
    resolved = {name: path.resolve() for name, path in qualification_paths.items()}
    _require(
        len(set(resolved.values())) == len(REQUIRED_QUALIFICATION_PROFILES),
        "Every qualification profile requires a distinct run directory",
    )
    for name, path in resolved.items():
        for other_name, other in resolved.items():
            if name >= other_name:
                continue
            _require(
                path not in other.parents and other not in path.parents,
                "Qualification run directories must be pairwise disjoint",
            )
    with exclusive_launches(tuple(resolved.values())):
        preregistration = verify_preregistered_training_stage(
            preregistration_path, training_run_dir
        )
        training_manifest, _ = load_training_plan(training_run_dir)
        value: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "kind": LAUNCH_EVIDENCE_KIND,
            "preregistration": {
                "artifact": _artifact(
                    preregistration_path, "experiment preregistration"
                ),
                "fingerprint": preregistration["preregistration_fingerprint"],
            },
            "canonical_training": {
                "run_dir": str(training_run_dir),
                "manifest": _artifact(
                    training_run_dir / MANIFEST_NAME, "canonical training manifest"
                ),
                "plan_fingerprint": training_manifest["plan_fingerprint"],
                "config_fingerprint": training_manifest["config_fingerprint"],
            },
            "qualifications": {
                name: _qualification_stage(
                    name, resolved[name], training_run_dir, training_manifest
                )
                for name in REQUIRED_QUALIFICATION_PROFILES
            },
        }
        value["reviewed_evidence_fingerprint"] = sha256_bytes(
            canonical_json_bytes(value)
        )
        return value


def build_launch_approval(
    preregistration_path: Path,
    training_run_dir: Path,
    one_step_dir: Path,
    resume_three_step_dir: Path,
    full_shape_five_step_dir: Path,
    manual_attestation_path: Path,
) -> dict[str, Any]:
    evidence = build_launch_evidence(
        preregistration_path,
        training_run_dir,
        one_step_dir,
        resume_three_step_dir,
        full_shape_five_step_dir,
    )
    evidence_fingerprint = evidence["reviewed_evidence_fingerprint"]
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": LAUNCH_APPROVAL_KIND,
        "state": "approved_for_canonical_launch",
        "preregistration": evidence["preregistration"],
        "canonical_training": evidence["canonical_training"],
        "qualifications": evidence["qualifications"],
        "reviewed_evidence_fingerprint": evidence_fingerprint,
        "manual_attestation": _manual_attestation(
            manual_attestation_path,
            reviewed_evidence_fingerprint=evidence_fingerprint,
        ),
        "approved_for_canonical_launch": True,
        "scientifically_attested": False,
        "attestation_limitations": [
            "manual_attestations_are_recorded_but_not_independently_verified",
            "resource_ceilings_are_approval_metadata_not_runtime_enforced",
        ],
    }
    value["launch_approval_fingerprint"] = sha256_bytes(
        canonical_json_bytes(value)
    )
    return value


def write_launch_approval(
    output_path: Path,
    preregistration_path: Path,
    training_run_dir: Path,
    one_step_dir: Path,
    resume_three_step_dir: Path,
    full_shape_five_step_dir: Path,
    manual_attestation_path: Path,
    *,
    force: bool = False,
    _lease: LaunchLease | None = None,
) -> dict[str, Any]:
    training_run_dir = training_run_dir.resolve()
    if _lease is None:
        with exclusive_launch(training_run_dir) as lease:
            return write_launch_approval(
                output_path,
                preregistration_path,
                training_run_dir,
                one_step_dir,
                resume_three_step_dir,
                full_shape_five_step_dir,
                manual_attestation_path,
                force=force,
                _lease=lease,
            )
    _lease.assert_for(training_run_dir)
    output = output_path.resolve()
    preregistration = verify_preregistered_training_stage(
        preregistration_path.resolve(), training_run_dir.resolve()
    )
    stages = _mapping(preregistration.get("stages"), "preregistered stages")
    initial_raw = _mapping(
        stages.get("initial_raw"), "preregistered initial raw stage"
    )
    initial_synthesis = _mapping(
        stages.get("initial_synthesis_config"),
        "preregistered initial synthesis config",
    )
    initial_raw_run_dir = initial_raw.get("run_dir")
    initial_synthesis_config_path = initial_synthesis.get("path")
    _require(
        isinstance(initial_raw_run_dir, str)
        and isinstance(initial_synthesis_config_path, str),
        "Preregistered initial evaluation paths are invalid",
    )
    qualification_roots = tuple(
        path.resolve()
        for path in (
            one_step_dir,
            resume_three_step_dir,
            full_shape_five_step_dir,
        )
    )
    bound_roots = (
        Path(initial_raw_run_dir).resolve(),
        training_run_dir.resolve(),
        *qualification_roots,
    )
    if any(root == output or root in output.parents for root in bound_roots):
        raise TrainingError(
            "Launch approval output must be outside every bound run directory"
        )
    protected = {
        preregistration_path.resolve(),
        manual_attestation_path.resolve(),
        Path(initial_synthesis_config_path).resolve(),
        (training_run_dir / MANIFEST_NAME).resolve(),
        *((path / QUALIFICATION_MANIFEST_NAME).resolve() for path in qualification_roots),
    }
    if output in protected:
        raise TrainingError("Launch approval output would replace a source artifact")
    value = build_launch_approval(
        preregistration_path,
        training_run_dir,
        one_step_dir,
        resume_three_step_dir,
        full_shape_five_step_dir,
        manual_attestation_path,
    )
    try:
        publish_json(output, value, force=force)
    except EvaluationError as exc:
        raise TrainingError(str(exc)) from exc
    return value


def load_launch_approval(path: Path) -> dict[str, Any]:
    try:
        value = read_json(path.resolve())
    except EvaluationError as exc:
        raise TrainingError(str(exc)) from exc
    expected = {
        "schema_version",
        "kind",
        "state",
        "preregistration",
        "canonical_training",
        "qualifications",
        "reviewed_evidence_fingerprint",
        "manual_attestation",
        "approved_for_canonical_launch",
        "scientifically_attested",
        "attestation_limitations",
        "launch_approval_fingerprint",
    }
    unsigned = dict(value)
    fingerprint = unsigned.pop("launch_approval_fingerprint", None)
    if (
        set(value) != expected
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("kind") != LAUNCH_APPROVAL_KIND
        or value.get("state") != "approved_for_canonical_launch"
        or value.get("approved_for_canonical_launch") is not True
        or value.get("scientifically_attested") is not False
        or not isinstance(value.get("reviewed_evidence_fingerprint"), str)
        or not _SHA256.fullmatch(value["reviewed_evidence_fingerprint"])
        or value.get("attestation_limitations")
        != [
            "manual_attestations_are_recorded_but_not_independently_verified",
            "resource_ceilings_are_approval_metadata_not_runtime_enforced",
        ]
        or fingerprint != sha256_bytes(canonical_json_bytes(unsigned))
    ):
        raise TrainingError("Invalid canonical launch approval")
    qualifications = _mapping(value.get("qualifications"), "launch qualifications")
    if set(qualifications) != set(REQUIRED_QUALIFICATION_PROFILES):
        raise TrainingError("Launch approval does not contain all qualification profiles")
    return value


def verify_launch_approval(
    approval_path: Path,
    preregistration_path: Path,
    training_run_dir: Path,
) -> dict[str, Any]:
    registered = load_launch_approval(approval_path)
    preregistration = _mapping(
        registered.get("preregistration"), "launch preregistration"
    )
    preregistration_artifact = _mapping(
        preregistration.get("artifact"), "launch preregistration artifact"
    )
    training = _mapping(
        registered.get("canonical_training"), "launch canonical training"
    )
    if (
        Path(str(preregistration_artifact.get("path"))).resolve()
        != preregistration_path.resolve()
        or Path(str(training.get("run_dir"))).resolve() != training_run_dir.resolve()
    ):
        raise TrainingError("Launch approval belongs to another preregistration or run")
    qualifications = _mapping(registered["qualifications"], "launch qualifications")
    qualification_stages = {
        name: _mapping(qualifications.get(name), f"{name} launch qualification")
        for name in REQUIRED_QUALIFICATION_PROFILES
    }
    manual = _mapping(registered["manual_attestation"], "manual attestation")
    artifact = _mapping(manual.get("artifact"), "manual attestation artifact")
    current = build_launch_approval(
        preregistration_path,
        training_run_dir,
        Path(str(qualification_stages["one_step"].get("run_dir"))),
        Path(str(qualification_stages["resume_three_step"].get("run_dir"))),
        Path(str(qualification_stages["full_shape_five_step"].get("run_dir"))),
        Path(str(artifact.get("path"))),
    )
    if registered != current:
        raise TrainingError("Canonical launch approval no longer matches its evidence")
    return registered


__all__ = [
    "LAUNCH_APPROVAL_KIND",
    "LAUNCH_EVIDENCE_KIND",
    "MANUAL_ATTESTATION_KIND",
    "REQUIRED_BUDGET_LIMITS",
    "REQUIRED_MANUAL_ATTESTATIONS",
    "REQUIRED_QUALIFICATION_PROFILES",
    "build_launch_approval",
    "build_launch_evidence",
    "load_launch_approval",
    "verify_launch_approval",
    "write_launch_approval",
]
