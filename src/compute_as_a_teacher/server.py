"""No-download server checks and guarded experiment phase orchestration."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit

from compute_as_a_teacher._toml import tomllib
from compute_as_a_teacher.evaluation.config import (
    load_raw_config,
    load_scoring_config,
    load_synthesis_config,
)
from compute_as_a_teacher.evaluation.execution import verify_complete_execution
from compute_as_a_teacher.evaluation.errors import EvaluationError
from compute_as_a_teacher.evaluation.planning import load_plan
from compute_as_a_teacher.evaluation.scoring import (
    SCORES_NAME,
    SCORING_MANIFEST_NAME,
    SUMMARY_NAME,
    score_run,
)
from compute_as_a_teacher.openai_chat import OpenAIChatError, chat_endpoint
from compute_as_a_teacher.training.checkpoints import (
    load_registered_checkpoint,
    load_registered_checkpoint_artifacts,
)
from compute_as_a_teacher.training.config import load_training_config
from compute_as_a_teacher.training.errors import TrainingError
from compute_as_a_teacher.training.experiment_registry import (
    verify_preregistered_training_stage,
)
from compute_as_a_teacher.training.eval_handoff import (
    RAW_CONFIG_NAME as TRAINED_RAW_CONFIG_NAME,
    SYNTHESIS_CONFIG_NAME as TRAINED_SYNTHESIS_CONFIG_NAME,
    verify_eval_handoff,
    verify_eval_handoff_artifacts,
)
from compute_as_a_teacher.training.planning import load_training_plan
from compute_as_a_teacher.training.verl_adapter import (
    checkpointed_step,
    validate_tracking_readiness,
)


SERVER_WORKFLOW_KIND = "cat_math500_server_workflow"
SERVER_WORKFLOW_SCHEMA_VERSION = 2
TRAINED_EXPORT_RELATIVE_PATH = Path(
    "exports/qwen3-4b-math500-cat-step-1000"
)
TRAINED_POLICY_BASE_URL = "http://trained-policy:8002/v1"
TRAINED_POLICY_SERVED_MODEL = "math500-cat-final"
QUALIFICATION_PROFILES = (
    "one_step",
    "resume_three_step",
    "full_shape_five_step",
)
PHASE_EXECUTION_SCOPES = {
    "prepare": "evaluator",
    "anchor": "host",
    "preregister": "evaluator",
    "baseline-generation": "evaluator",
    "baseline-scoring": "scorer",
    "qualification": "trainer",
    "approval": "trainer",
    "canonical": "trainer",
    "handoff": "trainer",
    "trained-policy": "host",
    "trained-eval-generation": "evaluator",
    "trained-eval-scoring": "scorer",
    "finalize": "scorer",
}
_ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_IMAGE_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_Runner = Callable[..., subprocess.CompletedProcess[str]]


class ServerWorkflowError(RuntimeError):
    """Raised when a server workflow cannot be checked or safely delegated."""


@dataclass(frozen=True, slots=True)
class ServerWorkflow:
    repository_root: Path
    config_path: Path
    training_config: Path
    raw_config: Path
    synthesis_config: Path
    scoring_config: Path
    output_root: Path
    training_run: Path
    initial_raw_run: Path
    initial_synthesis_run: Path
    preregistration: Path
    launch_approval: Path
    manual_attestation: Path
    checkpoint_export: Path
    trained_eval_config_dir: Path
    trained_raw_run: Path
    trained_synthesis_run: Path
    final_registry: Path
    qualification_dirs: Mapping[str, Path]
    evaluation_base_url: str
    evaluation_api_key_env: str
    evaluation_workers: int
    evaluation_batch_size: int
    anchor_start_command: tuple[str, ...]
    trained_policy_base_url: str
    trained_policy_api_key_env: str
    trained_policy_served_model: str
    trained_policy_start_command: tuple[str, ...]
    required_commands: tuple[str, ...]
    required_environment: tuple[str, ...]
    minimum_free_bytes: int


@dataclass(frozen=True, slots=True)
class PhaseCommand:
    name: str
    argv: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "argv": list(self.argv)}


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ServerWorkflowError(f"{name} must be a TOML table")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ServerWorkflowError(
            f"{name} keys differ: missing={missing}, extra={extra}"
        )


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ServerWorkflowError(f"{name} must be nonempty text")
    return value


def _positive_integer(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ServerWorkflowError(f"{name} must be a positive integer")
    return value


def _repo_path(root: Path, value: Any, name: str) -> Path:
    text = _text(value, name)
    path = Path(text)
    return (path if path.is_absolute() else root / path).resolve()


def _safe_base_url(value: Any, name: str) -> str:
    url = _text(value, name).rstrip("/")
    try:
        chat_endpoint(url)
    except OpenAIChatError as exc:
        raise ServerWorkflowError(
            f"{name} is not a safe HTTP(S) base URL"
        ) from exc
    if urlsplit(url).path not in {"", "/v1"}:
        raise ServerWorkflowError(f"{name} path must be empty or /v1")
    return url


def _compose_start_command(
    root: Path,
    value: Any,
    name: str,
    *,
    profile: str,
    service: str,
) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not all(isinstance(item, str) and item for item in value)
        or len(value) != 12
        or Path(value[0]).name != "docker"
        or value[1:3] != ["compose", "-f"]
        or value[4:11]
        != [
            "--profile",
            profile,
            "up",
            "-d",
            "--wait",
            "--wait-timeout",
            "600",
        ]
        or value[11] != service
    ):
        raise ServerWorkflowError(
            f"{name} must be the exact health-waiting {service} Compose argv"
        )
    compose_path = _repo_path(root, value[3], f"{name} compose file")
    expected = (root / "infra/server/compose.yaml").resolve()
    if compose_path != expected or not compose_path.is_file():
        raise ServerWorkflowError(f"{name} must use infra/server/compose.yaml")
    normalized = [*value]
    normalized[3] = str(compose_path)
    return tuple(normalized)


def _require_output_child(root: Path, path: Path, name: str) -> None:
    if path == root:
        raise ServerWorkflowError(f"{name} must be below output_root")
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ServerWorkflowError(f"{name} must be below output_root") from exc


def load_server_workflow(
    path: Path, *, repository_root: Path | None = None
) -> ServerWorkflow:
    config_path = path.resolve()
    root = (
        repository_root.resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    try:
        value = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ServerWorkflowError(f"Cannot read server workflow {config_path}: {exc}") from exc
    _exact_keys(
        value,
        {"schema_version", "kind", "configs", "outputs", "services", "readiness"},
        "server workflow",
    )
    if (
        value["schema_version"] != SERVER_WORKFLOW_SCHEMA_VERSION
        or value["kind"] != SERVER_WORKFLOW_KIND
    ):
        raise ServerWorkflowError("Unsupported server workflow schema or kind")

    configs = _mapping(value["configs"], "configs")
    _exact_keys(configs, {"training", "raw", "synthesis", "scoring"}, "configs")
    outputs = _mapping(value["outputs"], "outputs")
    _exact_keys(
        outputs,
        {
            "root",
            "training_run",
            "initial_raw_run",
            "initial_synthesis_run",
            "preregistration",
            "launch_approval",
            "manual_attestation",
            "checkpoint_export",
            "trained_eval_config_dir",
            "trained_raw_run",
            "trained_synthesis_run",
            "final_registry",
            "one_step_dir",
            "resume_three_step_dir",
            "full_shape_five_step_dir",
        },
        "outputs",
    )
    services = _mapping(value["services"], "services")
    _exact_keys(
        services,
        {
            "evaluation_base_url",
            "evaluation_api_key_env",
            "evaluation_workers",
            "evaluation_batch_size",
            "anchor_start_command",
            "trained_policy_base_url",
            "trained_policy_api_key_env",
            "trained_policy_served_model",
            "trained_policy_start_command",
        },
        "services",
    )
    readiness = _mapping(value["readiness"], "readiness")
    _exact_keys(
        readiness,
        {"required_commands", "required_environment", "minimum_free_bytes"},
        "readiness",
    )

    output_root = _repo_path(root, outputs["root"], "outputs.root")
    if output_root == Path(output_root.anchor) or output_root == root:
        raise ServerWorkflowError("output_root must be a dedicated directory")
    output_paths = {
        name: _repo_path(root, outputs[name], f"outputs.{name}")
        for name in (
            "training_run",
            "initial_raw_run",
            "initial_synthesis_run",
            "preregistration",
            "launch_approval",
            "manual_attestation",
            "checkpoint_export",
            "trained_eval_config_dir",
            "trained_raw_run",
            "trained_synthesis_run",
            "final_registry",
            "one_step_dir",
            "resume_three_step_dir",
            "full_shape_five_step_dir",
        )
    }
    for name, output_path in output_paths.items():
        _require_output_child(output_root, output_path, f"outputs.{name}")
    if len(set(output_paths.values())) != len(output_paths):
        raise ServerWorkflowError("Server workflow output paths must be distinct")
    stage_names = (
        "training_run",
        "initial_raw_run",
        "initial_synthesis_run",
        "one_step_dir",
        "resume_three_step_dir",
        "full_shape_five_step_dir",
        "checkpoint_export",
        "trained_eval_config_dir",
        "trained_raw_run",
        "trained_synthesis_run",
    )
    for index, left_name in enumerate(stage_names):
        left = output_paths[left_name]
        for right_name in stage_names[index + 1 :]:
            right = output_paths[right_name]
            if left in right.parents or right in left.parents:
                raise ServerWorkflowError(
                    f"outputs.{left_name} and outputs.{right_name} must not be nested"
                )
    for registry_name in (
        "preregistration",
        "launch_approval",
        "manual_attestation",
        "final_registry",
    ):
        registry_path = output_paths[registry_name]
        if any(
            registry_path == output_paths[name]
            or output_paths[name] in registry_path.parents
            or registry_path in output_paths[name].parents
            for name in stage_names
        ):
            raise ServerWorkflowError(
                f"outputs.{registry_name} must be outside every run directory"
            )
    if output_paths["checkpoint_export"] != (
        output_root / TRAINED_EXPORT_RELATIVE_PATH
    ).resolve():
        raise ServerWorkflowError(
            "outputs.checkpoint_export must match the trained-policy Compose export"
        )

    normalized_url = _safe_base_url(
        services["evaluation_base_url"], "services.evaluation_base_url"
    )
    api_key_env = _text(
        services["evaluation_api_key_env"], "services.evaluation_api_key_env"
    )
    if not _ENVIRONMENT_NAME.fullmatch(api_key_env):
        raise ServerWorkflowError("services.evaluation_api_key_env is invalid")
    anchor_command = _compose_start_command(
        root,
        services["anchor_start_command"],
        "services.anchor_start_command",
        profile="local-anchor",
        service="anchor",
    )
    trained_policy_url = _safe_base_url(
        services["trained_policy_base_url"],
        "services.trained_policy_base_url",
    )
    if trained_policy_url != TRAINED_POLICY_BASE_URL:
        raise ServerWorkflowError(
            "services.trained_policy_base_url must match the internal Compose service"
        )
    trained_api_key_env = _text(
        services["trained_policy_api_key_env"],
        "services.trained_policy_api_key_env",
    )
    if not _ENVIRONMENT_NAME.fullmatch(trained_api_key_env):
        raise ServerWorkflowError("services.trained_policy_api_key_env is invalid")
    trained_served_model = _text(
        services["trained_policy_served_model"],
        "services.trained_policy_served_model",
    )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", trained_served_model):
        raise ServerWorkflowError("services.trained_policy_served_model is invalid")
    if trained_served_model != TRAINED_POLICY_SERVED_MODEL:
        raise ServerWorkflowError(
            "services.trained_policy_served_model must match the Compose alias"
        )
    trained_start_command = _compose_start_command(
        root,
        services["trained_policy_start_command"],
        "services.trained_policy_start_command",
        profile="trained-policy",
        service="trained-policy",
    )

    commands = readiness["required_commands"]
    environments = readiness["required_environment"]
    if (
        not isinstance(commands, list)
        or not commands
        or not all(isinstance(item, str) and item and "/" not in item for item in commands)
        or len(commands) != len(set(commands))
    ):
        raise ServerWorkflowError("readiness.required_commands must be unique command names")
    if (
        not isinstance(environments, list)
        or not all(isinstance(item, str) and _ENVIRONMENT_NAME.fullmatch(item) for item in environments)
        or len(environments) != len(set(environments))
    ):
        raise ServerWorkflowError(
            "readiness.required_environment must contain unique environment names"
        )
    return ServerWorkflow(
        repository_root=root,
        config_path=config_path,
        training_config=_repo_path(root, configs["training"], "configs.training"),
        raw_config=_repo_path(root, configs["raw"], "configs.raw"),
        synthesis_config=_repo_path(root, configs["synthesis"], "configs.synthesis"),
        scoring_config=_repo_path(root, configs["scoring"], "configs.scoring"),
        output_root=output_root,
        training_run=output_paths["training_run"],
        initial_raw_run=output_paths["initial_raw_run"],
        initial_synthesis_run=output_paths["initial_synthesis_run"],
        preregistration=output_paths["preregistration"],
        launch_approval=output_paths["launch_approval"],
        manual_attestation=output_paths["manual_attestation"],
        checkpoint_export=output_paths["checkpoint_export"],
        trained_eval_config_dir=output_paths["trained_eval_config_dir"],
        trained_raw_run=output_paths["trained_raw_run"],
        trained_synthesis_run=output_paths["trained_synthesis_run"],
        final_registry=output_paths["final_registry"],
        qualification_dirs={
            name: output_paths[f"{name}_dir"] for name in QUALIFICATION_PROFILES
        },
        evaluation_base_url=normalized_url,
        evaluation_api_key_env=api_key_env,
        evaluation_workers=_positive_integer(
            services["evaluation_workers"], "services.evaluation_workers"
        ),
        evaluation_batch_size=_positive_integer(
            services["evaluation_batch_size"], "services.evaluation_batch_size"
        ),
        anchor_start_command=anchor_command,
        trained_policy_base_url=trained_policy_url,
        trained_policy_api_key_env=trained_api_key_env,
        trained_policy_served_model=trained_served_model,
        trained_policy_start_command=trained_start_command,
        required_commands=tuple(commands),
        required_environment=tuple(environments),
        minimum_free_bytes=_positive_integer(
            readiness["minimum_free_bytes"], "readiness.minimum_free_bytes"
        ),
    )


def _script(workflow: ServerWorkflow, name: str) -> str:
    return str(workflow.repository_root / "scripts" / name)


def _dataset_repository_root(workflow: ServerWorkflow) -> Path:
    data_dir = Path(
        os.environ.get(
            "CAT_DATA_DIR",
            str(workflow.repository_root / "data"),
        )
    ).resolve()
    if data_dir.name != "data":
        raise ServerWorkflowError("CAT_DATA_DIR basename must be data")
    return data_dir.parent


def _train(workflow: ServerWorkflow, *arguments: str) -> tuple[str, ...]:
    return (sys.executable, _script(workflow, "train_math500.py"), *arguments)


def _eval(workflow: ServerWorkflow, *arguments: str) -> tuple[str, ...]:
    return (sys.executable, _script(workflow, "evaluate_math500.py"), *arguments)


def _qualification_inputs(workflow: ServerWorkflow) -> tuple[str, ...]:
    return (
        "--one-step-dir",
        str(workflow.qualification_dirs["one_step"]),
        "--resume-three-step-dir",
        str(workflow.qualification_dirs["resume_three_step"]),
        "--full-shape-five-step-dir",
        str(workflow.qualification_dirs["full_shape_five_step"]),
    )


def _final_registry_inputs(workflow: ServerWorkflow) -> tuple[str, ...]:
    return (
        "--preregistration",
        str(workflow.preregistration),
        "--initial-raw-run-dir",
        str(workflow.initial_raw_run),
        "--initial-synthesis-run-dir",
        str(workflow.initial_synthesis_run),
        "--training-run-dir",
        str(workflow.training_run),
        "--trained-raw-run-dir",
        str(workflow.trained_raw_run),
        "--trained-synthesis-run-dir",
        str(workflow.trained_synthesis_run),
    )


def phase_commands(
    workflow: ServerWorkflow,
    phase: str,
    *,
    execute: bool = False,
    qualification_profile: str | None = None,
    resume_action: str | None = None,
) -> tuple[PhaseCommand, ...]:
    training_config = str(workflow.training_config)
    training_run = str(workflow.training_run)
    if phase != "qualification" and (
        qualification_profile is not None or resume_action is not None
    ):
        raise ServerWorkflowError(
            "Qualification options apply only to the qualification phase"
        )
    if phase == "prepare":
        return (
            PhaseCommand(
                "verify_locked_math500_questions",
                (
                    sys.executable,
                    _script(workflow, "prepare_math500.py"),
                    "--repo-root",
                    str(_dataset_repository_root(workflow)),
                    "--verify-questions-only",
                ),
            ),
            PhaseCommand(
                "provision_checkpoint_export_parent",
                (
                    sys.executable,
                    _script(workflow, "server_math500.py"),
                    "--workflow",
                    str(workflow.config_path),
                    "provision-export-parent",
                ),
            ),
            PhaseCommand(
                "prepare_canonical_training_plan",
                _train(
                    workflow,
                    "prepare",
                    "--config",
                    training_config,
                    "--run-dir",
                    training_run,
                ),
            ),
            PhaseCommand(
                "prepare_initial_raw_plan",
                _eval(
                    workflow,
                    "plan-raw",
                    "--config",
                    str(workflow.raw_config),
                    "--run-dir",
                    str(workflow.initial_raw_run),
                ),
            ),
        )
    if phase == "anchor":
        return (PhaseCommand("start_frozen_anchor", workflow.anchor_start_command),)
    if phase == "trained-policy":
        compose = workflow.trained_policy_start_command[:4]
        return (
            PhaseCommand(
                "verify_registered_checkpoint_in_trainer",
                (
                    *compose,
                    "run",
                    "--rm",
                    "trainer",
                    "train",
                    "inspect-checkpoint",
                    "--run-dir",
                    str(workflow.training_run),
                    "--export-dir",
                    str(workflow.checkpoint_export),
                ),
            ),
            PhaseCommand(
                "verify_trained_handoff_in_trainer",
                (
                    *compose,
                    "run",
                    "--rm",
                    "trainer",
                    "train",
                    "inspect-trained-eval",
                    "--run-dir",
                    str(workflow.training_run),
                    "--output-dir",
                    str(workflow.trained_eval_config_dir),
                    "--served-model",
                    workflow.trained_policy_served_model,
                ),
            ),
            PhaseCommand(
                "start_trained_policy",
                workflow.trained_policy_start_command,
            ),
        )
    if phase == "preregister":
        return (
            PhaseCommand(
                "preregister_experiment",
                _train(
                    workflow,
                    "preregister-experiment",
                    "--output",
                    str(workflow.preregistration),
                    "--initial-raw-run-dir",
                    str(workflow.initial_raw_run),
                    "--initial-synthesis-config",
                    str(workflow.synthesis_config),
                    "--training-run-dir",
                    training_run,
                ),
            ),
        )
    if phase == "baseline-generation":
        endpoint = (
            "--base-url",
            workflow.evaluation_base_url,
            "--api-key-env",
            workflow.evaluation_api_key_env,
            "--workers",
            str(workflow.evaluation_workers),
            "--batch-size",
            str(workflow.evaluation_batch_size),
        )
        return (
            PhaseCommand(
                "raw_endpoint_canary",
                _eval(
                    workflow,
                    "run-openai",
                    "--run-dir",
                    str(workflow.initial_raw_run),
                    *endpoint,
                    "--max-requests",
                    "16",
                ),
            ),
            PhaseCommand(
                "complete_initial_raw",
                _eval(
                    workflow,
                    "run-openai",
                    "--run-dir",
                    str(workflow.initial_raw_run),
                    *endpoint,
                ),
            ),
            PhaseCommand(
                "prepare_initial_synthesis_plan",
                _eval(
                    workflow,
                    "plan-synthesis",
                    "--config",
                    str(workflow.synthesis_config),
                    "--raw-run-dir",
                    str(workflow.initial_raw_run),
                    "--run-dir",
                    str(workflow.initial_synthesis_run),
                ),
            ),
            PhaseCommand(
                "complete_initial_synthesis",
                _eval(
                    workflow,
                    "run-openai",
                    "--run-dir",
                    str(workflow.initial_synthesis_run),
                    *endpoint,
                ),
            ),
        )
    if phase == "baseline-scoring":
        return (
            PhaseCommand(
                "score_initial_raw",
                _eval(
                    workflow,
                    "score-raw",
                    "--run-dir",
                    str(workflow.initial_raw_run),
                    "--config",
                    str(workflow.scoring_config),
                ),
            ),
            PhaseCommand(
                "score_initial_synthesis",
                _eval(
                    workflow,
                    "score-synthesis",
                    "--run-dir",
                    str(workflow.initial_synthesis_run),
                    "--raw-run-dir",
                    str(workflow.initial_raw_run),
                    "--config",
                    str(workflow.scoring_config),
                ),
            ),
        )
    if phase == "qualification":
        if qualification_profile not in QUALIFICATION_PROFILES:
            raise ServerWorkflowError(
                "qualification requires --profile with one registered profile"
            )
        qualification_dir = str(workflow.qualification_dirs[qualification_profile])
        if qualification_profile != "resume_three_step" and resume_action is not None:
            raise ServerWorkflowError(
                "--resume-action applies only to resume_three_step"
            )
        launch = list(
            _train(
                workflow,
                "launch-qualification",
                "--config",
                training_config,
                "--qualification-dir",
                qualification_dir,
            )
        )
        if execute:
            launch.append("--execute")
        prepare = PhaseCommand(
            f"prepare_{qualification_profile}",
            _train(
                workflow,
                "prepare-qualification",
                "--run-dir",
                training_run,
                "--qualification-dir",
                qualification_dir,
                "--profile",
                qualification_profile,
            ),
        )
        launch_command = PhaseCommand(
            f"launch_{qualification_profile}", tuple(launch)
        )
        if qualification_profile == "resume_three_step" and execute:
            if resume_action == "prepare":
                return (prepare,)
            if resume_action in {"initial", "restart"}:
                return (launch_command,)
            raise ServerWorkflowError(
                "resume_three_step execution requires --resume-action prepare, initial, "
                "or restart; initial must be supervised and interrupted after checkpoint 1"
            )
        return (prepare, launch_command)
    evidence_inputs = (
        "--preregistration",
        str(workflow.preregistration),
        "--training-run-dir",
        training_run,
        *_qualification_inputs(workflow),
    )
    if phase == "approval":
        return (
            PhaseCommand(
                "inspect_launch_evidence",
                _train(workflow, "inspect-launch-evidence", *evidence_inputs),
            ),
            PhaseCommand(
                "write_launch_approval",
                _train(
                    workflow,
                    "write-launch-approval",
                    "--output",
                    str(workflow.launch_approval),
                    *evidence_inputs,
                    "--manual-attestation",
                    str(workflow.manual_attestation),
                ),
            ),
            PhaseCommand(
                "verify_launch_approval",
                _train(
                    workflow,
                    "inspect-launch-approval",
                    "--launch-approval",
                    str(workflow.launch_approval),
                    "--preregistration",
                    str(workflow.preregistration),
                    "--training-run-dir",
                    training_run,
                ),
            ),
        )
    if phase == "canonical":
        launch = list(
            _train(
                workflow,
                "launch",
                "--config",
                training_config,
                "--run-dir",
                training_run,
                "--preregistration",
                str(workflow.preregistration),
                "--launch-approval",
                str(workflow.launch_approval),
            )
        )
        if execute:
            launch.append("--execute")
        return (
            PhaseCommand(
                "verify_launch_approval",
                _train(
                    workflow,
                    "inspect-launch-approval",
                    "--launch-approval",
                    str(workflow.launch_approval),
                    "--preregistration",
                    str(workflow.preregistration),
                    "--training-run-dir",
                    training_run,
                ),
            ),
            PhaseCommand("launch_canonical", tuple(launch)),
        )
    if phase == "handoff":
        export = list(
            _train(
                workflow,
                "export-register",
                "--config",
                training_config,
                "--run-dir",
                training_run,
                "--export-dir",
                str(workflow.checkpoint_export),
            )
        )
        if execute:
            export.append("--execute")
        return (
            PhaseCommand(
                "verify_launch_approval",
                _train(
                    workflow,
                    "inspect-launch-approval",
                    "--launch-approval",
                    str(workflow.launch_approval),
                    "--preregistration",
                    str(workflow.preregistration),
                    "--training-run-dir",
                    training_run,
                ),
            ),
            PhaseCommand("export_and_register_fixed_checkpoint", tuple(export)),
            PhaseCommand(
                "plan_trained_evaluation",
                _train(
                    workflow,
                    "plan-trained-eval",
                    "--run-dir",
                    training_run,
                    "--output-dir",
                    str(workflow.trained_eval_config_dir),
                    "--served-model",
                    workflow.trained_policy_served_model,
                ),
            ),
        )
    if phase == "trained-eval-generation":
        trained_raw_config = (
            workflow.trained_eval_config_dir / TRAINED_RAW_CONFIG_NAME
        )
        trained_synthesis_config = (
            workflow.trained_eval_config_dir / TRAINED_SYNTHESIS_CONFIG_NAME
        )
        trained_endpoint = (
            "--base-url",
            workflow.trained_policy_base_url,
            "--api-key-env",
            workflow.trained_policy_api_key_env,
            "--workers",
            str(workflow.evaluation_workers),
            "--batch-size",
            str(workflow.evaluation_batch_size),
        )
        anchor_endpoint = (
            "--base-url",
            workflow.evaluation_base_url,
            "--api-key-env",
            workflow.evaluation_api_key_env,
            "--workers",
            str(workflow.evaluation_workers),
            "--batch-size",
            str(workflow.evaluation_batch_size),
        )
        return (
            PhaseCommand(
                "verify_trained_evaluation_handoff",
                _train(
                    workflow,
                    "inspect-trained-eval",
                    "--run-dir",
                    training_run,
                    "--output-dir",
                    str(workflow.trained_eval_config_dir),
                    "--served-model",
                    workflow.trained_policy_served_model,
                ),
            ),
            PhaseCommand(
                "prepare_trained_raw_plan",
                _eval(
                    workflow,
                    "plan-raw",
                    "--config",
                    str(trained_raw_config),
                    "--run-dir",
                    str(workflow.trained_raw_run),
                ),
            ),
            PhaseCommand(
                "trained_raw_endpoint_canary",
                _eval(
                    workflow,
                    "run-openai",
                    "--run-dir",
                    str(workflow.trained_raw_run),
                    *trained_endpoint,
                    "--max-requests",
                    "16",
                ),
            ),
            PhaseCommand(
                "complete_trained_raw",
                _eval(
                    workflow,
                    "run-openai",
                    "--run-dir",
                    str(workflow.trained_raw_run),
                    *trained_endpoint,
                ),
            ),
            PhaseCommand(
                "prepare_trained_synthesis_plan",
                _eval(
                    workflow,
                    "plan-synthesis",
                    "--config",
                    str(trained_synthesis_config),
                    "--raw-run-dir",
                    str(workflow.trained_raw_run),
                    "--run-dir",
                    str(workflow.trained_synthesis_run),
                ),
            ),
            PhaseCommand(
                "complete_trained_synthesis",
                _eval(
                    workflow,
                    "run-openai",
                    "--run-dir",
                    str(workflow.trained_synthesis_run),
                    *anchor_endpoint,
                ),
            ),
        )
    if phase == "trained-eval-scoring":
        return (
            PhaseCommand(
                "score_trained_raw",
                _eval(
                    workflow,
                    "score-raw",
                    "--run-dir",
                    str(workflow.trained_raw_run),
                    "--config",
                    str(workflow.scoring_config),
                ),
            ),
            PhaseCommand(
                "score_trained_synthesis",
                _eval(
                    workflow,
                    "score-synthesis",
                    "--run-dir",
                    str(workflow.trained_synthesis_run),
                    "--raw-run-dir",
                    str(workflow.trained_raw_run),
                    "--config",
                    str(workflow.scoring_config),
                ),
            ),
        )
    if phase == "finalize":
        registry_inputs = _final_registry_inputs(workflow)
        return (
            PhaseCommand(
                "finalize_experiment_registry",
                _train(
                    workflow,
                    "finalize-experiment",
                    "--output",
                    str(workflow.final_registry),
                    *registry_inputs,
                ),
            ),
            PhaseCommand(
                "verify_experiment_registry",
                _train(
                    workflow,
                    "verify-experiment",
                    "--registry",
                    str(workflow.final_registry),
                    *registry_inputs,
                ),
            ),
        )
    raise ServerWorkflowError(f"Unsupported server phase: {phase}")


def provision_export_parent(workflow: ServerWorkflow) -> dict[str, Any]:
    root = workflow.output_root
    parent = workflow.checkpoint_export.parent
    if (
        not root.is_dir()
        or root.is_symlink()
        or root.resolve(strict=True) != root
        or parent.parent != root
    ):
        raise ServerWorkflowError(
            "Checkpoint export parent requires one canonical output-root child"
        )
    if os.path.lexists(workflow.checkpoint_export):
        raise ServerWorkflowError(
            "Canonical checkpoint export target must remain nonexistent before merge"
        )
    if os.path.lexists(parent):
        if (
            not parent.is_dir()
            or parent.is_symlink()
            or parent.resolve(strict=True) != parent
        ):
            raise ServerWorkflowError("Checkpoint export parent is unsafe")
    else:
        os.mkdir(parent, mode=0o700)
    mode = parent.stat().st_mode & 0o777
    if mode & 0o077:
        raise ServerWorkflowError(
            "Checkpoint export parent must not grant group or world access"
        )
    return {
        "schema_version": 1,
        "kind": "cat_checkpoint_export_parent",
        "path": str(parent),
        "mode": f"{mode:03o}",
        "target_nonexistent": True,
    }


def preview_phase(
    workflow: ServerWorkflow,
    phase: str,
    *,
    qualification_profile: str | None = None,
    resume_action: str | None = None,
) -> dict[str, Any]:
    anchor_mode = os.environ.get("CAT_ANCHOR_MODE", "").strip()
    if phase == "anchor" and anchor_mode == "remote":
        if os.environ.get("CAT_ANCHOR_GPU_DEVICE", "").strip():
            raise ServerWorkflowError(
                "Remote anchor mode forbids CAT_ANCHOR_GPU_DEVICE"
            )
        return {
            "schema_version": 1,
            "kind": "cat_server_phase",
            "phase": "anchor",
            "anchor_mode": "remote",
            "execution_scope": "external",
            "would_execute": False,
            "skipped": True,
            "commands": [],
            "note": (
                "Remote anchor lifecycle is external. Readiness still verifies its "
                "configured URL, identity alias, sampling behavior, and canaries."
            ),
        }
    if phase == "anchor" and anchor_mode != "local":
        raise ServerWorkflowError("CAT_ANCHOR_MODE must be 'local' or 'remote'")
    commands = phase_commands(
        workflow,
        phase,
        execute=False,
        qualification_profile=qualification_profile,
        resume_action=resume_action,
    )
    note = (
        "Preview only. Add --execute to delegate these existing CLIs; their "
        "preregistration, qualification, approval, and preflight gates remain active."
    )
    if phase == "qualification" and qualification_profile == "resume_three_step":
        note += (
            " Execute --resume-action prepare, then --resume-action initial under a "
            "scheduler that interrupts it only after checkpoint 1, and finally execute "
            "--resume-action restart. An uninterrupted three-step launch is not accepted."
        )
    if phase in {"anchor", "trained-policy"}:
        note += (
            " This service phase is host-only. Do not run it inside the trainer or "
            "mount a Docker socket into that container."
        )
    return {
        "schema_version": 1,
        "kind": "cat_server_phase",
        "phase": phase,
        "qualification_profile": qualification_profile,
        "resume_action": resume_action,
        "execution_scope": PHASE_EXECUTION_SCOPES.get(phase, "unknown"),
        "would_execute": False,
        "commands": [command.to_dict() for command in commands],
        "note": note,
    }


def _validate_host_environment(
    workflow: ServerWorkflow,
    *,
    runner: _Runner,
    require_trained_policy: bool,
) -> dict[str, Any]:
    validator = workflow.repository_root / "infra/server/validate_server_env.py"
    if not validator.is_file():
        raise ServerWorkflowError(f"Host environment validator is missing: {validator}")
    argv = [sys.executable, str(validator)]
    if require_trained_policy:
        argv.append("--require-trained-policy")
    try:
        completed = runner(
            argv,
            cwd=str(workflow.repository_root),
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ServerWorkflowError(f"Cannot validate the host environment: {exc}") from exc
    validation = _json_from_completed(completed, validator.name)
    if validation.get("ready") is not True:
        raise ServerWorkflowError("Host environment validator did not approve launch")
    if require_trained_policy and validation.get("trained_policy_required") is not True:
        raise ServerWorkflowError(
            "Host environment validator did not approve trained-policy mode"
        )
    return validation


def _require_clean_host_checkout(
    workflow: ServerWorkflow, *, runner: _Runner
) -> None:
    if not (workflow.repository_root / ".git").exists():
        raise ServerWorkflowError(
            "Host service launch requires a Git checkout"
        )
    repository = _repository_check(workflow, runner=runner)
    if repository.get("mode") != "git_checkout":
        raise ServerWorkflowError(
            "Host service launch requires a clean Git checkout"
        )


def _verify_trained_handoff(
    workflow: ServerWorkflow, *, artifact_only: bool = False
) -> dict[str, Any]:
    if artifact_only:
        verify_eval_handoff_artifacts(
            workflow.training_run,
            workflow.trained_eval_config_dir,
            workflow.trained_policy_served_model,
        )
        checkpoint, _ = load_registered_checkpoint_artifacts(
            workflow.training_run
        )
    else:
        verify_eval_handoff(
            workflow.training_run,
            workflow.trained_eval_config_dir,
            workflow.trained_policy_served_model,
        )
        checkpoint, _ = load_registered_checkpoint(workflow.training_run)
    export = checkpoint.get("export")
    if (
        not isinstance(export, dict)
        or Path(export.get("path", "")).resolve() != workflow.checkpoint_export
    ):
        raise ServerWorkflowError(
            "Registered checkpoint export does not match the server workflow"
        )
    raw_path = workflow.trained_eval_config_dir / TRAINED_RAW_CONFIG_NAME
    synthesis_path = (
        workflow.trained_eval_config_dir / TRAINED_SYNTHESIS_CONFIG_NAME
    )
    raw = load_raw_config(raw_path)
    synthesis = load_synthesis_config(synthesis_path)
    training = load_training_config(workflow.training_config)
    if (
        raw.model.model_id != workflow.trained_policy_served_model
        or raw.model.revision != export.get("tree_sha256")
        or synthesis.anchor_relation != "frozen_initial_for_trained_raw"
        or synthesis.anchor.model_id != training.runtime.anchor_model
    ):
        raise ServerWorkflowError(
            "Trained-evaluation handoff does not bind pi_T raw and frozen pi_0 synthesis"
        )
    return {"export_tree_sha256": export["tree_sha256"]}


def _verify_scored_experiment(workflow: ServerWorkflow) -> None:
    scoring = load_scoring_config(workflow.scoring_config)
    for raw_run, synthesis_run in (
        (workflow.initial_raw_run, workflow.initial_synthesis_run),
        (workflow.trained_raw_run, workflow.trained_synthesis_run),
    ):
        score_run(
            raw_run,
            scoring,
            repository_root=workflow.repository_root,
        )
        score_run(
            synthesis_run,
            scoring,
            repository_root=workflow.repository_root,
            raw_run_dir=raw_run,
        )


def _verify_all_generation_complete(workflow: ServerWorkflow) -> None:
    try:
        for run_dir in (
            workflow.initial_raw_run,
            workflow.initial_synthesis_run,
            workflow.trained_raw_run,
            workflow.trained_synthesis_run,
        ):
            manifest, requests = load_plan(run_dir)
            verify_complete_execution(run_dir, manifest, requests)
    except EvaluationError as exc:
        raise ServerWorkflowError(
            f"All generation phases must be complete before scoring: {exc}"
        ) from exc


def _require_label_derived_artifacts_absent(workflow: ServerWorkflow) -> None:
    found = [
        str(run_dir / name)
        for run_dir in (
            workflow.initial_raw_run,
            workflow.initial_synthesis_run,
            workflow.trained_raw_run,
            workflow.trained_synthesis_run,
        )
        for name in (SCORES_NAME, SUMMARY_NAME, SCORING_MANIFEST_NAME)
        if os.path.lexists(run_dir / name)
    ]
    if found:
        raise ServerWorkflowError(
            "Training and generation phases require label-derived artifacts to be "
            f"absent: {found}"
        )


def _phase_precondition(
    workflow: ServerWorkflow,
    phase: str,
    *,
    qualification_profile: str | None,
    resume_action: str | None,
    runner: _Runner,
) -> None:
    if phase in {
        "prepare",
        "preregister",
        "baseline-generation",
        "qualification",
        "approval",
        "canonical",
        "handoff",
        "trained-eval-generation",
    }:
        _require_label_derived_artifacts_absent(workflow)
    if phase == "anchor":
        mode = os.environ.get("CAT_ANCHOR_MODE", "").strip()
        if mode != "local":
            raise ServerWorkflowError(
                "Local anchor execution requires CAT_ANCHOR_MODE=local"
            )
        validation = _validate_host_environment(
            workflow,
            runner=runner,
            require_trained_policy=False,
        )
        if (
            validation.get("anchor_mode") != "local"
        ):
            raise ServerWorkflowError(
                "Host environment validator did not approve local anchor mode"
            )
        _require_clean_host_checkout(workflow, runner=runner)
    if phase == "trained-policy":
        _validate_host_environment(
            workflow,
            runner=runner,
            require_trained_policy=True,
        )
        _require_clean_host_checkout(workflow, runner=runner)
    if phase in {
        "baseline-generation",
        "baseline-scoring",
        "qualification",
        "handoff",
        "trained-eval-generation",
        "trained-eval-scoring",
        "finalize",
    }:
        verify_preregistered_training_stage(
            workflow.preregistration, workflow.training_run
        )
    if phase == "baseline-generation" and not os.environ.get(
        workflow.evaluation_api_key_env
    ):
        raise ServerWorkflowError(
            f"Evaluation API key is unset: {workflow.evaluation_api_key_env}"
        )
    if phase == "baseline-scoring":
        _verify_trained_handoff(workflow, artifact_only=True)
        _verify_all_generation_complete(workflow)
    if phase == "handoff":
        config = load_training_config(workflow.training_config)
        step = checkpointed_step(
            workflow.training_run,
            config.grpo.max_steps,
            expected_world_size=config.runtime.nodes * config.runtime.gpus_per_node,
        )
        if step != config.grpo.max_steps:
            raise ServerWorkflowError(
                f"Post-training handoff requires terminal step {config.grpo.max_steps}"
            )
    if phase == "trained-eval-generation":
        missing = [
            name
            for name in (
                workflow.trained_policy_api_key_env,
                workflow.evaluation_api_key_env,
            )
            if not os.environ.get(name)
        ]
        if missing:
            raise ServerWorkflowError(
                f"Trained evaluation API key variables are unset: {sorted(set(missing))}"
            )
        _verify_trained_handoff(workflow, artifact_only=True)
    if phase == "trained-eval-scoring":
        _verify_trained_handoff(workflow, artifact_only=True)
        _verify_all_generation_complete(workflow)
    if phase == "finalize":
        _verify_trained_handoff(workflow, artifact_only=True)
        _verify_scored_experiment(workflow)
    if (
        phase == "qualification"
        and qualification_profile == "resume_three_step"
        and resume_action in {"initial", "restart"}
    ):
        config = load_training_config(workflow.training_config)
        step = checkpointed_step(
            workflow.qualification_dirs["resume_three_step"],
            3,
            expected_world_size=config.runtime.nodes * config.runtime.gpus_per_node,
        )
        expected = 0 if resume_action == "initial" else 1
        if step != expected:
            raise ServerWorkflowError(
                f"resume_three_step {resume_action} requires verified step {expected}, found {step}"
            )


def execute_phase(
    workflow: ServerWorkflow,
    phase: str,
    *,
    qualification_profile: str | None = None,
    resume_action: str | None = None,
    runner: _Runner = subprocess.run,
    replace_final_process: bool = False,
) -> dict[str, Any]:
    expected_scope = PHASE_EXECUTION_SCOPES.get(phase)
    if expected_scope is None:
        raise ServerWorkflowError(f"Unsupported server phase: {phase}")
    service_role = os.environ.get("CAT_SERVICE_ROLE", "").strip()
    if expected_scope == "host":
        if service_role:
            raise ServerWorkflowError(
                f"Phase {phase} is host-only, not {service_role}"
            )
    elif service_role != expected_scope:
        raise ServerWorkflowError(
            f"Phase {phase} requires CAT_SERVICE_ROLE={expected_scope}"
        )
    if phase == "anchor" and os.environ.get("CAT_ANCHOR_MODE", "").strip() == "remote":
        return preview_phase(
            workflow,
            phase,
            qualification_profile=qualification_profile,
            resume_action=resume_action,
        )
    _phase_precondition(
        workflow,
        phase,
        qualification_profile=qualification_profile,
        resume_action=resume_action,
        runner=runner,
    )
    commands = phase_commands(
        workflow,
        phase,
        execute=True,
        qualification_profile=qualification_profile,
        resume_action=resume_action,
    )
    completed_names: list[str] = []
    for index, command in enumerate(commands):
        if (
            replace_final_process
            and index == len(commands) - 1
            and command.name.startswith("launch_")
        ):
            os.chdir(workflow.repository_root)
            os.execvpe(command.argv[0], list(command.argv), os.environ.copy())
        try:
            completed = runner(
                list(command.argv),
                cwd=str(workflow.repository_root),
                check=False,
                text=True,
            )
        except OSError as exc:
            raise ServerWorkflowError(f"Cannot execute {command.name}: {exc}") from exc
        if completed.returncode:
            raise ServerWorkflowError(
                f"Phase command {command.name} failed with status {completed.returncode}"
            )
        completed_names.append(command.name)
    return {
        "schema_version": 1,
        "kind": "cat_server_phase",
        "phase": phase,
        "qualification_profile": qualification_profile,
        "resume_action": resume_action,
        "executed": True,
        "completed_commands": completed_names,
        "complete": True,
    }


def _json_from_completed(
    completed: subprocess.CompletedProcess[str], name: str
) -> dict[str, Any]:
    if completed.returncode:
        detail = (completed.stderr or completed.stdout or "").strip()[-2000:]
        raise ServerWorkflowError(
            f"{name} failed with status {completed.returncode}: {detail}"
        )
    try:
        value = json.loads(completed.stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ServerWorkflowError(f"{name} did not return one JSON object") from exc
    if not isinstance(value, dict):
        raise ServerWorkflowError(f"{name} did not return one JSON object")
    return value


def _run_offline_command(
    argv: Sequence[str],
    *,
    workflow: ServerWorkflow,
    runner: _Runner,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    try:
        completed = runner(
            list(argv),
            cwd=str(workflow.repository_root),
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ServerWorkflowError(f"Cannot run readiness command: {exc}") from exc
    return completed


def _run_json_command(
    argv: Sequence[str],
    *,
    workflow: ServerWorkflow,
    runner: _Runner,
    timeout: float,
) -> dict[str, Any]:
    completed = _run_offline_command(
        argv, workflow=workflow, runner=runner, timeout=timeout
    )
    return _json_from_completed(completed, Path(argv[1]).name)


def _run_text_command(
    argv: Sequence[str],
    *,
    workflow: ServerWorkflow,
    runner: _Runner,
    timeout: float,
) -> dict[str, Any]:
    completed = _run_offline_command(
        argv, workflow=workflow, runner=runner, timeout=timeout
    )
    if completed.returncode:
        detail = (completed.stderr or completed.stdout or "").strip()[-2000:]
        raise ServerWorkflowError(
            f"{Path(argv[1]).name} failed with status {completed.returncode}: {detail}"
        )
    output = completed.stdout.strip()
    if not output:
        raise ServerWorkflowError(f"{Path(argv[1]).name} returned no verification output")
    return {"verified": True, "output": output.splitlines()}


def _storage_check(workflow: ServerWorkflow) -> dict[str, Any]:
    root = workflow.output_root
    if not root.is_dir():
        raise ServerWorkflowError(f"Output root is missing: {root}")
    usage = shutil.disk_usage(root)
    if usage.free < workflow.minimum_free_bytes:
        raise ServerWorkflowError(
            f"Output storage has {usage.free} free bytes; "
            f"{workflow.minimum_free_bytes} are required"
        )
    try:
        with tempfile.NamedTemporaryFile(
            dir=root, prefix=".cat-readiness-", delete=True
        ) as handle:
            handle.write(b"compute-as-a-teacher readiness\n")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise ServerWorkflowError(f"Output root is not writable: {root}: {exc}") from exc
    return {
        "path": str(root),
        "free_bytes": usage.free,
        "minimum_free_bytes": workflow.minimum_free_bytes,
        "temporary_write_verified": True,
    }


def _repository_check(
    workflow: ServerWorkflow, *, runner: _Runner
) -> dict[str, Any]:
    required = (
        workflow.repository_root / "pyproject.toml",
        workflow.repository_root / "scripts" / "prepare_math500.py",
        workflow.repository_root / "scripts" / "evaluate_math500.py",
        workflow.repository_root / "scripts" / "train_math500.py",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise ServerWorkflowError(f"Repository files are missing: {missing}")
    git_directory = workflow.repository_root / ".git"
    if not git_directory.exists():
        receipt_path = workflow.repository_root.parent / "image-metadata" / "source.json"
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            training_value = tomllib.loads(
                workflow.training_config.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
            raise ServerWorkflowError(
                f"Cannot verify immutable image source metadata: {exc}"
            ) from exc
        expected_keys = {
            "schema_version",
            "source_revision",
            "source_layout",
            "source_tree_sha256",
            "source_inventory",
            "base_image",
            "verl_revision",
        }
        base_image = receipt.get("base_image") if isinstance(receipt, dict) else None
        runtime = training_value.get("runtime") if isinstance(training_value, dict) else None
        if (
            not isinstance(receipt, dict)
            or set(receipt) != expected_keys
            or receipt.get("schema_version") != 1
            or receipt.get("source_layout") != "explicit_allowlist_v1"
            or not isinstance(base_image, dict)
            or set(base_image) != {"name", "digest"}
            or not isinstance(base_image.get("name"), str)
            or not base_image["name"]
            or "@" in base_image["name"]
            or any(character.isspace() for character in base_image["name"])
            or not isinstance(runtime, dict)
            or receipt.get("verl_revision") != runtime.get("framework_revision")
        ):
            raise ServerWorkflowError("Immutable image source metadata is invalid")
        source_revision = receipt.get("source_revision")
        verl_revision = receipt.get("verl_revision")
        base_digest = base_image.get("digest")
        source_tree_sha256 = receipt.get("source_tree_sha256")
        inventory = receipt.get("source_inventory")
        if (
            not isinstance(source_revision, str)
            or not _COMMIT.fullmatch(source_revision)
            or len(set(source_revision)) == 1
            or not isinstance(verl_revision, str)
            or not _COMMIT.fullmatch(verl_revision)
            or len(set(verl_revision)) == 1
            or not isinstance(base_digest, str)
            or not _IMAGE_DIGEST.fullmatch(base_digest)
            or len(set(base_digest.removeprefix("sha256:"))) == 1
            or not isinstance(source_tree_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", source_tree_sha256)
            or not isinstance(inventory, list)
            or not inventory
        ):
            raise ServerWorkflowError("Immutable image source metadata is invalid")
        normalized_inventory: list[dict[str, Any]] = []
        for entry in inventory:
            if not isinstance(entry, dict) or set(entry) != {"path", "sha256", "bytes"}:
                raise ServerWorkflowError("Immutable image source inventory is invalid")
            relative = entry.get("path")
            digest = entry.get("sha256")
            size = entry.get("bytes")
            pure = PurePosixPath(relative) if isinstance(relative, str) else None
            if (
                pure is None
                or pure.is_absolute()
                or ".." in pure.parts
                or not pure.parts
                or not isinstance(digest, str)
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
                or type(size) is not int
                or size < 0
            ):
                raise ServerWorkflowError("Immutable image source inventory is invalid")
            source_path = workflow.repository_root.joinpath(*pure.parts)
            if source_path.is_symlink() or not source_path.is_file():
                raise ServerWorkflowError(
                    f"Immutable image source file is missing or unsafe: {relative}"
                )
            payload = source_path.read_bytes()
            if len(payload) != size or hashlib.sha256(payload).hexdigest() != digest:
                raise ServerWorkflowError(
                    f"Immutable image source file changed: {relative}"
                )
            normalized_inventory.append(dict(entry))
        if [entry["path"] for entry in normalized_inventory] != sorted(
            entry["path"] for entry in normalized_inventory
        ) or len({entry["path"] for entry in normalized_inventory}) != len(
            normalized_inventory
        ):
            raise ServerWorkflowError("Immutable image source inventory is invalid")
        encoded_inventory = json.dumps(
            normalized_inventory, sort_keys=True, separators=(",", ":")
        ).encode()
        if hashlib.sha256(encoded_inventory).hexdigest() != source_tree_sha256:
            raise ServerWorkflowError("Immutable image source tree digest is invalid")
        return {
            "mode": "immutable_image_receipt",
            "source_revision": source_revision,
            "source_layout": receipt["source_layout"],
            "source_tree_sha256": source_tree_sha256,
            "source_files": len(normalized_inventory),
            "base_image": dict(base_image),
            "verl_revision": verl_revision,
            "receipt": str(receipt_path),
        }
    try:
        revision = runner(
            ["git", "-C", str(workflow.repository_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        status = runner(
            [
                "git",
                "-C",
                str(workflow.repository_root),
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ServerWorkflowError(f"Cannot inspect repository identity: {exc}") from exc
    if revision.returncode or not re.fullmatch(r"[0-9a-f]{40}\n?", revision.stdout):
        raise ServerWorkflowError("Repository HEAD is unavailable")
    if status.returncode or status.stdout.strip():
        raise ServerWorkflowError("Repository has tracked or untracked changes")
    return {
        "mode": "git_checkout",
        "commit": revision.stdout.strip(),
        "worktree_clean": True,
    }


def _commands_check(workflow: ServerWorkflow) -> dict[str, Any]:
    resolved = {name: shutil.which(name) for name in workflow.required_commands}
    missing = sorted(name for name, path in resolved.items() if path is None)
    if missing:
        raise ServerWorkflowError(f"Required commands are missing: {missing}")
    return {"resolved": dict(sorted(resolved.items()))}


def readiness_report(
    workflow: ServerWorkflow,
    *,
    runner: _Runner = subprocess.run,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    state = SimpleNamespace(training_config=None)

    def check(name: str, operation: Callable[[], Any]) -> None:
        try:
            detail = operation()
        except Exception as exc:
            checks.append({"name": name, "ok": False, "error": str(exc)})
        else:
            checks.append({"name": name, "ok": True, "detail": detail})

    check("required_commands", lambda: _commands_check(workflow))
    check("repository", lambda: _repository_check(workflow, runner=runner))

    def source_paths() -> dict[str, Any]:
        paths = {
            "server_workflow": workflow.config_path,
            "training_config": workflow.training_config,
            "raw_config": workflow.raw_config,
            "synthesis_config": workflow.synthesis_config,
            "scoring_config": workflow.scoring_config,
        }
        missing = [name for name, path in paths.items() if not path.is_file()]
        if missing:
            raise ServerWorkflowError(f"Required configuration files are missing: {missing}")
        return {name: str(path) for name, path in sorted(paths.items())}

    check("source_paths", source_paths)

    def configs() -> dict[str, Any]:
        training = load_training_config(workflow.training_config)
        raw = load_raw_config(workflow.raw_config)
        synthesis = load_synthesis_config(workflow.synthesis_config)
        load_scoring_config(workflow.scoring_config)
        state.training_config = training
        return {
            "training_fingerprint": training.fingerprint,
            "raw_fingerprint": raw.fingerprint,
            "synthesis_fingerprint": synthesis.fingerprint,
            "download_allowed": training.runtime.download_allowed,
        }

    check("resolved_configs", configs)

    def environment() -> dict[str, Any]:
        names = list(workflow.required_environment)
        if state.training_config is not None:
            names.append(state.training_config.runtime.anchor_api_key_env)
        names.append(workflow.evaluation_api_key_env)
        unique = sorted(set(names))
        missing = [name for name in unique if not os.environ.get(name)]
        if missing:
            raise ServerWorkflowError(f"Required environment variables are unset: {missing}")
        return {"present": unique}

    check("required_environment", environment)
    check("storage", lambda: _storage_check(workflow))
    check(
        "locked_math500",
        lambda: _run_text_command(
            (
                sys.executable,
                _script(workflow, "prepare_math500.py"),
                "--repo-root",
                str(_dataset_repository_root(workflow)),
                "--verify-questions-only",
            ),
            workflow=workflow,
            runner=runner,
            timeout=120,
        ),
    )

    def require_training_config() -> Any:
        if state.training_config is None:
            raise ServerWorkflowError("Resolved training config is unavailable")
        return state.training_config

    check(
        "model_identity",
        lambda: _run_json_command(
            _train(
                workflow,
                "model-identity",
                "--model-path",
                require_training_config().runtime.model_path,
            ),
            workflow=workflow,
            runner=runner,
            timeout=1800,
        ),
    )
    check(
        "runtime_identity",
        lambda: _run_json_command(
            _train(
                workflow,
                "runtime-identity",
                "--python",
                require_training_config().runtime.python_executable,
                "--verl-source",
                require_training_config().runtime.verl_source_path,
            ),
            workflow=workflow,
            runner=runner,
            timeout=180,
        ),
    )
    def training_preflight() -> dict[str, Any]:
        require_training_config()
        value = _run_json_command(
            _train(
                workflow,
                "preflight",
                "--config",
                str(workflow.training_config),
                "--run-dir",
                str(workflow.training_run),
                "--hash-model",
                "--check-anchor",
            ),
            workflow=workflow,
            runner=runner,
            timeout=2400,
        )
        if (
            value.get("operationally_ready_to_launch") is not True
            or value.get("missing_gates") != []
        ):
            raise ServerWorkflowError("Training preflight did not pass every gate")
        return value

    check("training_preflight", training_preflight)
    def tracking() -> dict[str, Any]:
        config = require_training_config()
        _, command = load_training_plan(workflow.training_run)
        return validate_tracking_readiness(config, command, workflow.training_run)

    check("tracking", tracking)
    ready = all(item["ok"] for item in checks)
    preflight_passed = any(
        item["name"] == "training_preflight" and item["ok"] for item in checks
    )
    return {
        "schema_version": 1,
        "kind": "cat_server_readiness",
        "ready": ready,
        "no_download": True,
        "model_weights_loaded": False,
        "models_launched": False,
        "anchor_canary_requested": True,
        "anchor_called": preflight_passed,
        "checks": checks,
        "failed_checks": [item["name"] for item in checks if not item["ok"]],
    }
