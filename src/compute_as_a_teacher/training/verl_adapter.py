"""Pinned verl v0.5.0 command compilation and launch preflight.

Nothing in this module imports verl, Torch, Ray, Transformers, or vLLM.  Planning
therefore cannot initialize a model or a GPU runtime.
"""

from __future__ import annotations

import errno
import fcntl
import json
import os
import re
import signal
import subprocess
import threading
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from compute_as_a_teacher.evaluation.artifacts import canonical_json_bytes, sha256_bytes

from .config import SUPPORTED_VERL_REVISION, TrainingConfig
from .errors import TrainingError


LAUNCH_LOCK_NAME = ".launch.lock"
_PROCESS_EXIT_GRACE_SECONDS = 0.25
_PROCESS_GROUP_STOP_SECONDS = 10.0
_WANDB_RUN_ID = re.compile(r"cat-[0-9a-f]{32}(?:-q-[a-z0-9_]+)?")
_WANDB_SECRET_ENV = "WANDB_API_KEY"
_LABEL_DERIVED_ARTIFACT_NAMES = frozenset(
    {"scores.jsonl", "summary.json", "scoring_manifest.json", "final-experiment.json"}
)


class LaunchLease:
    __slots__ = ("_active", "_descriptor", "_owner_pid", "run_dir")

    def __init__(self, run_dir: Path, descriptor: int) -> None:
        self.run_dir = run_dir.resolve()
        self._descriptor = descriptor
        self._owner_pid = os.getpid()
        self._active = True

    def assert_for(self, run_dir: Path) -> None:
        if (
            not self._active
            or self._owner_pid != os.getpid()
            or self.run_dir != run_dir.resolve()
        ):
            raise TrainingError("A live launch lease for this run directory is required")

    def _release(self) -> None:
        self._active = False


def _hydra_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _hydra_bool(value: bool) -> str:
    return "True" if value else "False"


@dataclass(frozen=True, slots=True)
class VerlCommand:
    argv: tuple[str, ...]
    cwd: str
    environment: Mapping[str, str]
    framework_revision: str
    adapter_version: str

    def __post_init__(self) -> None:
        if _WANDB_SECRET_ENV in self.environment:
            raise TrainingError("W&B credentials must never be serialized in a command")

    def to_dict(self) -> dict[str, Any]:
        return {
            "argv": list(self.argv),
            "cwd": self.cwd,
            "environment": dict(sorted(self.environment.items())),
            "framework_revision": self.framework_revision,
            "adapter_version": self.adapter_version,
        }

    @property
    def fingerprint(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_dict()))


def require_label_free_training_outputs(
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    source = os.environ if environment is None else environment
    if source.get("CAT_SERVICE_ROLE", "").strip() != "trainer":
        return {"enforced": False}
    configured = source.get("CAT_OUTPUT_DIR", "").strip()
    if not configured:
        raise TrainingError("Trainer label-isolation guard requires CAT_OUTPUT_DIR")
    root = Path(configured)
    try:
        canonical = root.resolve(strict=True)
    except OSError as exc:
        raise TrainingError("Trainer output root is missing or unreadable") from exc
    if root.is_symlink() or canonical != root or not root.is_dir():
        raise TrainingError("Trainer output root must be a canonical directory")
    try:
        found = sorted(
            str(path)
            for path in root.rglob("*")
            if path.name in _LABEL_DERIVED_ARTIFACT_NAMES and os.path.lexists(path)
        )
    except OSError as exc:
        raise TrainingError("Cannot inspect trainer output root for label artifacts") from exc
    if found:
        raise TrainingError(
            "Trainer refuses an output root containing label-derived artifacts: "
            f"{found}"
        )
    return {"enforced": True, "output_root": str(root), "artifacts_found": []}


def canonical_wandb_run_id(config: TrainingConfig) -> str:
    return f"cat-{config.fingerprint[:32]}"


def qualification_wandb_run_id(canonical_run_id: str, profile_name: str) -> str:
    if not re.fullmatch(r"cat-[0-9a-f]{32}", canonical_run_id):
        raise TrainingError("Canonical W&B run ID has an invalid shape")
    if not re.fullmatch(r"[a-z0-9_]+", profile_name):
        raise TrainingError("Qualification profile name is invalid for W&B")
    run_id = f"{canonical_run_id}-q-{profile_name}"
    if len(run_id) > 128:
        raise TrainingError("Qualification W&B run ID is too long")
    return run_id


def qualification_wandb_group(
    source_group: str,
    canonical_run_id: str,
    profile_name: str,
) -> str:
    prefix = source_group or canonical_run_id
    candidate = f"{prefix}-qual-{profile_name}"
    if len(candidate) <= 128:
        return candidate
    suffix = sha256_bytes(canonical_json_bytes([prefix, profile_name]))[:12]
    return f"{prefix[:80]}-qual-{profile_name[:24]}-{suffix}"


def _wandb_tags(config: TrainingConfig, qualification_profile: str | None) -> tuple[str, ...]:
    tags = list(config.tracking.wandb.tags)
    if qualification_profile is not None:
        for tag in ("qualification", "nonreportable", qualification_profile):
            if tag not in tags:
                tags.append(tag)
    return tuple(tags)


def _wandb_environment(
    config: TrainingConfig,
    run_dir: Path,
    *,
    qualification_profile: str | None = None,
) -> dict[str, str]:
    wandb = config.tracking.wandb
    if not wandb.enabled:
        return {}
    canonical_id = canonical_wandb_run_id(config)
    run_id = (
        qualification_wandb_run_id(canonical_id, qualification_profile)
        if qualification_profile is not None
        else canonical_id
    )
    group = (
        qualification_wandb_group(wandb.group, canonical_id, qualification_profile)
        if qualification_profile is not None
        else wandb.group
    )
    environment = {
        "WANDB_DIR": str((run_dir.resolve() / "wandb").resolve()),
        "WANDB_ENTITY": wandb.entity,
        "WANDB_MODE": wandb.mode,
        "WANDB_RESUME": wandb.resume,
        "WANDB_RUN_ID": run_id,
    }
    if group:
        environment["WANDB_RUN_GROUP"] = group
    tags = _wandb_tags(config, qualification_profile)
    if tags:
        environment["WANDB_TAGS"] = ",".join(tags)
    return environment


def _override(command: VerlCommand, name: str) -> str:
    prefix = f"{name}="
    matches = [item[len(prefix) :] for item in command.argv[3:] if item.startswith(prefix)]
    if len(matches) != 1:
        raise TrainingError(f"Expected exactly one planned override for {name}")
    return matches[0]


def validate_tracking_readiness(
    config: TrainingConfig,
    command: VerlCommand,
    run_dir: Path,
    *,
    qualification_profile: str | None = None,
    source_environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    wandb = config.tracking.wandb
    planned_wandb = {
        key: value for key, value in command.environment.items() if key.startswith("WANDB_")
    }
    if (
        _WANDB_SECRET_ENV in planned_wandb
        or wandb.api_key_env in command.environment
    ):
        raise TrainingError("W&B API key must not appear in the planned command")
    if not wandb.enabled:
        if planned_wandb:
            raise TrainingError("Disabled W&B tracking must not plan W&B environment variables")
        return {"console": True, "wandb": {"enabled": False}}

    expected_environment = _wandb_environment(
        config,
        run_dir,
        qualification_profile=qualification_profile,
    )
    if planned_wandb != expected_environment:
        raise TrainingError("Planned W&B environment does not match the immutable config")
    run_id = planned_wandb["WANDB_RUN_ID"]
    if not _WANDB_RUN_ID.fullmatch(run_id):
        raise TrainingError("Planned W&B run ID is invalid")
    logger = _override(command, "trainer.logger")
    project = _override(command, "trainer.project_name")
    experiment = _override(command, "trainer.experiment_name")
    expected_name = (
        f"{config.run_name}-{qualification_profile}-nonreportable"
        if qualification_profile is not None
        else config.run_name
    )
    if logger != "[console,wandb]":
        raise TrainingError("Enabled W&B tracking requires console and wandb Verl loggers")
    if project != _hydra_string(wandb.project):
        raise TrainingError("Planned Verl W&B project does not match the config")
    if experiment != _hydra_string(expected_name):
        raise TrainingError("Planned W&B experiment name does not match the run stage")

    source = os.environ if source_environment is None else source_environment
    credential = source.get(wandb.api_key_env)
    credential_present = isinstance(credential, str) and bool(credential) and credential == credential.strip()
    if wandb.mode == "online" and not credential_present:
        raise TrainingError(
            f"W&B credential environment variable is unset or blank: {wandb.api_key_env}"
        )
    return {
        "console": True,
        "wandb": {
            "enabled": True,
            "project": wandb.project,
            "entity": wandb.entity,
            "mode": wandb.mode,
            "sdk_version": wandb.sdk_version,
            "run_id": run_id,
            "resume": wandb.resume,
            "group": planned_wandb.get("WANDB_RUN_GROUP"),
            "tags": list(_wandb_tags(config, qualification_profile)),
            "api_key_env": wandb.api_key_env,
            "credential_present": credential_present,
        },
    }


def build_process_environment(
    config: TrainingConfig,
    command: VerlCommand,
    run_dir: Path,
    *,
    qualification_profile: str | None = None,
    source_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    source = dict(os.environ if source_environment is None else source_environment)
    validate_tracking_readiness(
        config,
        command,
        run_dir,
        qualification_profile=qualification_profile,
        source_environment=source,
    )
    wandb = config.tracking.wandb
    credential = source.get(wandb.api_key_env) if wandb.enabled else None
    environment = {
        key: value for key, value in source.items() if not key.startswith("WANDB_")
    }
    if wandb.enabled and wandb.api_key_env != _WANDB_SECRET_ENV:
        environment.pop(wandb.api_key_env, None)
    environment.update(command.environment)
    if wandb.enabled and wandb.mode == "online":
        assert isinstance(credential, str)
        environment[_WANDB_SECRET_ENV] = credential
    return environment


def build_verl_command(
    config: TrainingConfig,
    *,
    repository_root: Path,
    run_dir: Path,
    training_data_path: Path,
) -> VerlCommand:
    """Translate the semantic contract into pinned verl Hydra overrides."""

    repository_root = repository_root.resolve()
    run_dir = run_dir.resolve()
    training_data_path = training_data_path.resolve()
    reward_module = (
        repository_root
        / "src/compute_as_a_teacher/training/verl_reward.py"
    ).resolve()
    hydra_dir = (run_dir / "hydra").resolve()
    checkpoints = (run_dir / "checkpoints").resolve()
    rollout_logs = (run_dir / "rollout_logs").resolve()
    rollout = config.rollouts.sampling
    synthesis = config.synthesis.sampling
    runtime = config.runtime
    wandb = config.tracking.wandb
    loggers = ("console", "wandb") if wandb.enabled else ("console",)
    project_name = wandb.project if wandb.enabled else "compute-as-a-teacher"

    overrides = [
        f"hydra.run.dir={_hydra_string(str(hydra_dir))}",
        f"hydra.output_subdir={_hydra_string('.hydra')}",
        "hydra.job.chdir=False",
        f"data.train_files={_hydra_string(str(training_data_path))}",
        f"data.val_files={_hydra_string(str(training_data_path))}",
        "data.prompt_key=prompt",
        "data.reward_fn_key=data_source",
        f"data.max_prompt_length={runtime.max_prompt_tokens}",
        f"data.max_response_length={rollout.max_new_tokens}",
        f"data.train_batch_size={config.grpo.global_batch_size}",
        f"data.val_batch_size={config.grpo.global_batch_size}",
        "data.shuffle=True",
        f"+data.seed={runtime.seed}",
        "data.validation_shuffle=False",
        "data.filter_overlong_prompts=False",
        "data.truncation=error",
        "data.return_raw_chat=True",
        f"data.dataloader_num_workers={runtime.dataloader_workers}",
        "data.trust_remote_code=False",
        "data.custom_cls.path="
        f"{_hydra_string('pkg://compute_as_a_teacher.training.verl_dataset')}",
        "data.custom_cls.name=JsonlRLHFDataset",
        f"actor_rollout_ref.model.path={_hydra_string(runtime.model_path)}",
        "actor_rollout_ref.model.trust_remote_code=False",
        f"actor_rollout_ref.actor.strategy={runtime.strategy}",
        f"actor_rollout_ref.ref.strategy={runtime.strategy}",
        f"actor_rollout_ref.actor.optim.lr={config.grpo.learning_rate}",
        f"actor_rollout_ref.actor.optim.lr_warmup_steps={config.grpo.warmup_steps}",
        "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0",
        f"actor_rollout_ref.actor.optim.warmup_style={config.grpo.lr_scheduler}",
        f"actor_rollout_ref.actor.optim.weight_decay={config.optimizer.weight_decay}",
        "+actor_rollout_ref.actor.optim.betas="
        f"[{config.optimizer.betas[0]},{config.optimizer.betas[1]}]",
        f"actor_rollout_ref.actor.grad_clip={config.optimizer.max_grad_norm}",
        f"actor_rollout_ref.actor.clip_ratio={config.grpo.clip_epsilon}",
        f"actor_rollout_ref.actor.clip_ratio_low={config.grpo.clip_epsilon}",
        f"actor_rollout_ref.actor.clip_ratio_high={config.grpo.clip_epsilon}",
        f"actor_rollout_ref.actor.ppo_epochs={config.grpo.ppo_epochs}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={config.grpo.ppo_mini_batch_size}",
        "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean",
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={runtime.max_tokens_per_gpu}",
        "actor_rollout_ref.actor.shuffle=False",
        "actor_rollout_ref.actor.use_kl_loss=False",
        f"actor_rollout_ref.rollout.name={runtime.rollout_engine}",
        "actor_rollout_ref.rollout.mode=sync",
        f"actor_rollout_ref.rollout.n={config.rollouts.group_size}",
        f"actor_rollout_ref.rollout.do_sample={_hydra_bool(rollout.do_sample)}",
        f"actor_rollout_ref.rollout.temperature={rollout.temperature}",
        f"actor_rollout_ref.rollout.top_p={rollout.top_p}",
        f"actor_rollout_ref.rollout.top_k={rollout.top_k}",
        f"+actor_rollout_ref.rollout.seed={rollout.base_seed}",
        f"actor_rollout_ref.rollout.dtype={runtime.dtype}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={runtime.tensor_parallel_size}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={runtime.gpu_memory_utilization}",
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True",
        f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={runtime.max_tokens_per_gpu}",
        "actor_rollout_ref.rollout.ignore_eos=False",
        "actor_rollout_ref.rollout.enforce_eager=True",
        "actor_rollout_ref.rollout.free_cache_engine=True",
        "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True",
        f"actor_rollout_ref.ref.log_prob_max_token_len_per_gpu={runtime.max_tokens_per_gpu}",
        "reward_model.enable=False",
        "reward_model.reward_manager=batch",
        "reward_model.launch_reward_fn_async=False",
        f"custom_reward_function.path={_hydra_string(str(reward_module))}",
        "custom_reward_function.name=compute_score",
        "+custom_reward_function.reward_kwargs.repository_root="
        f"{_hydra_string(str(repository_root))}",
        "+custom_reward_function.reward_kwargs.prompt_path="
        f"{_hydra_string(config.synthesis.prompt.path)}",
        "+custom_reward_function.reward_kwargs.prompt_version="
        f"{_hydra_string(config.synthesis.prompt.version)}",
        "+custom_reward_function.reward_kwargs.prompt_prefix="
        f"{_hydra_string(config.synthesis.prompt.prefix)}",
        "+custom_reward_function.reward_kwargs.anchor_base_url="
        f"{_hydra_string(runtime.anchor_base_url)}",
        "+custom_reward_function.reward_kwargs.anchor_model="
        f"{_hydra_string(runtime.anchor_model)}",
        "+custom_reward_function.reward_kwargs.anchor_api_key_env="
        f"{_hydra_string(runtime.anchor_api_key_env)}",
        "+custom_reward_function.reward_kwargs.anchor_timeout_seconds="
        f"{runtime.anchor_timeout_seconds}",
        "+custom_reward_function.reward_kwargs.anchor_max_concurrency="
        f"{runtime.anchor_max_concurrency}",
        "+custom_reward_function.reward_kwargs.anchor_temperature="
        f"{synthesis.temperature}",
        f"+custom_reward_function.reward_kwargs.anchor_top_p={synthesis.top_p}",
        f"+custom_reward_function.reward_kwargs.anchor_top_k={synthesis.top_k}",
        "+custom_reward_function.reward_kwargs.anchor_max_tokens="
        f"{synthesis.max_new_tokens}",
        f"+custom_reward_function.reward_kwargs.base_seed={synthesis.base_seed}",
        "+custom_reward_function.reward_kwargs.max_answer_chars="
        f"{config.reward.max_answer_chars}",
        "+custom_reward_function.reward_kwargs.anchor_failure_policy="
        f"{_hydra_string(config.reward.invalid_anchor)}",
        "algorithm.adv_estimator=grpo",
        f"algorithm.norm_adv_by_std_in_grpo={_hydra_bool(config.grpo.normalize_advantages)}",
        "algorithm.use_kl_in_reward=True",
        "algorithm.kl_penalty=kl",
        "algorithm.kl_ctrl.type=fixed",
        f"algorithm.kl_ctrl.kl_coef={config.grpo.kl_coefficient}",
        f"trainer.total_training_steps={config.grpo.max_steps}",
        f"trainer.total_epochs={config.grpo.max_steps}",
        f"trainer.project_name={_hydra_string(project_name)}",
        f"trainer.experiment_name={_hydra_string(config.run_name)}",
        "trainer.balance_batch=False",
        "trainer.val_before_train=False",
        "trainer.test_freq=-1",
        "trainer.log_val_generations=0",
        f"trainer.save_freq={config.checkpointing.save_every_steps}",
        f"trainer.resume_mode={config.checkpointing.resume_mode}",
        f"trainer.default_local_dir={_hydra_string(str(checkpoints))}",
        f"trainer.rollout_data_dir={_hydra_string(str(rollout_logs))}",
        f"trainer.max_actor_ckpt_to_keep={config.checkpointing.max_checkpoints}",
        f"trainer.nnodes={runtime.nodes}",
        f"trainer.n_gpus_per_node={runtime.gpus_per_node}",
        f"trainer.logger=[{','.join(loggers)}]",
    ]
    command = (
        runtime.python_executable,
        "-m",
        "verl.trainer.main_ppo",
        *overrides,
    )
    environment = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HYDRA_FULL_ERROR": "1",
        "PYTHONPATH": str((repository_root / "src").resolve()),
        "PYTHONDONTWRITEBYTECODE": "1",
        "TOKENIZERS_PARALLELISM": "true",
    }
    environment.update(_wandb_environment(config, run_dir))
    return VerlCommand(
        argv=command,
        cwd=runtime.verl_source_path,
        environment=environment,
        framework_revision=runtime.framework_revision,
        adapter_version=runtime.adapter_version,
    )


def command_from_dict(value: Mapping[str, Any]) -> VerlCommand:
    expected = {
        "argv", "cwd", "environment", "framework_revision", "adapter_version"
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise TrainingError("Invalid verl command artifact schema")
    argv = value["argv"]
    environment = value["environment"]
    if not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv):
        raise TrainingError("verl command argv must be a nonempty string list")
    if not isinstance(environment, dict) or not all(
        isinstance(key, str) and isinstance(item, str)
        for key, item in environment.items()
    ):
        raise TrainingError("verl command environment must map strings to strings")
    command = VerlCommand(
        argv=tuple(argv),
        cwd=str(value["cwd"]),
        environment=dict(environment),
        framework_revision=str(value["framework_revision"]),
        adapter_version=str(value["adapter_version"]),
    )
    if command.framework_revision != SUPPORTED_VERL_REVISION:
        raise TrainingError("Command targets an unsupported verl revision")
    return command


def verify_verl_checkout(path: Path, expected_revision: str) -> None:
    path = path.resolve()
    if not path.is_dir():
        raise TrainingError(f"verl_source_path is not a directory: {path}")
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise TrainingError(f"Cannot verify verl checkout at {path}: {exc}") from exc
    actual = completed.stdout.strip()
    if actual != expected_revision:
        raise TrainingError(
            f"verl checkout revision mismatch: expected {expected_revision}, found {actual}"
        )
    try:
        status = subprocess.run(
            [
                "git",
                "-C",
                str(path),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise TrainingError(f"Cannot verify verl checkout at {path}: {exc}") from exc
    if status.stdout.strip():
        raise TrainingError(f"verl checkout must be clean: {path}")


_SHARD_NAME = re.compile(
    r"(?P<kind>model|optim|extra_state)_world_size_(?P<world_size>\d+)_rank_(?P<rank>\d+)\.pt"
)


def _checkpoint_file(path: Path, name: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise TrainingError(f"verl checkpoint is missing {name}: {path}")
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise TrainingError(f"Cannot inspect verl checkpoint file {path}: {exc}") from exc
    if size == 0:
        raise TrainingError(f"verl checkpoint file is empty: {path}")


def validate_verl_checkpoint(
    run_dir: Path,
    step: int,
    *,
    expected_world_size: int | None = None,
) -> Path:
    """Validate the on-disk checkpoint contract emitted by pinned verl v0.5.0."""

    if type(step) is not int or step <= 0:
        raise TrainingError("A saved verl checkpoint step must be a positive integer")
    if expected_world_size is not None and (
        type(expected_world_size) is not int or expected_world_size <= 0
    ):
        raise TrainingError("expected_world_size must be a positive integer")

    step_dir = run_dir.resolve() / "checkpoints" / f"global_step_{step}"
    actor_dir = step_dir / "actor"
    if step_dir.is_symlink() or not step_dir.is_dir():
        raise TrainingError(f"verl checkpoint step directory is missing: {step_dir}")
    if actor_dir.is_symlink() or not actor_dir.is_dir():
        raise TrainingError(f"verl actor checkpoint directory is missing: {actor_dir}")

    _checkpoint_file(step_dir / "data.pt", "dataloader state")
    fsdp_config_path = actor_dir / "fsdp_config.json"
    _checkpoint_file(fsdp_config_path, "actor fsdp_config.json")
    _checkpoint_file(
        actor_dir / "huggingface" / "config.json",
        "actor Hugging Face config.json",
    )
    try:
        fsdp_config = json.loads(fsdp_config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrainingError(
            f"Cannot read verl actor FSDP config {fsdp_config_path}: {exc}"
        ) from exc
    if not isinstance(fsdp_config, dict):
        raise TrainingError("verl actor FSDP config must be a JSON object")
    if set(fsdp_config) != {"FSDP_version", "world_size"}:
        raise TrainingError("verl actor FSDP config has unexpected fields")
    if (
        type(fsdp_config["FSDP_version"]) is not int
        or fsdp_config["FSDP_version"] != 1
    ):
        raise TrainingError("verl actor checkpoint must use FSDP version 1")
    world_size = fsdp_config.get("world_size")
    if type(world_size) is not int or world_size <= 0:
        raise TrainingError("verl actor FSDP world_size must be a positive integer")
    if expected_world_size is not None and world_size != expected_world_size:
        raise TrainingError(
            "verl actor FSDP world_size mismatch: "
            f"expected {expected_world_size}, found {world_size}"
        )

    expected_names = {
        f"{kind}_world_size_{world_size}_rank_{rank}.pt"
        for kind in ("model", "optim", "extra_state")
        for rank in range(world_size)
    }
    actual_names = {
        path.name
        for path in actor_dir.iterdir()
        if path.is_file() and _SHARD_NAME.fullmatch(path.name)
    }
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        raise TrainingError(
            "verl actor checkpoint shard set is invalid: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for name in sorted(expected_names):
        _checkpoint_file(actor_dir / name, f"actor shard {name}")
    return actor_dir


def checkpointed_step(
    run_dir: Path,
    max_steps: int,
    *,
    expected_world_size: int | None = None,
) -> int:
    tracker = run_dir.resolve() / "checkpoints/latest_checkpointed_iteration.txt"
    if not tracker.exists():
        return 0
    try:
        step = int(tracker.read_text(encoding="utf-8").strip())
    except (OSError, ValueError) as exc:
        raise TrainingError(f"Cannot read checkpoint tracker {tracker}: {exc}") from exc
    if not 0 <= step <= max_steps:
        raise TrainingError(f"Checkpoint step {step} is outside [0, {max_steps}]")
    if step:
        validate_verl_checkpoint(
            run_dir,
            step,
            expected_world_size=expected_world_size,
        )
    return step


@contextmanager
def exclusive_launch(run_dir: Path) -> Iterator[LaunchLease]:
    root = run_dir.resolve()
    lock = root / LAUNCH_LOCK_NAME
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
    except OSError as exc:
        raise TrainingError(f"Cannot open training launch lock {lock}: {exc}") from exc
    acquired = False
    lease: LaunchLease | None = None
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise TrainingError(
                    f"Training launch is already locked: {lock}"
                ) from exc
            raise TrainingError(
                f"Cannot acquire training launch lock {lock}: {exc}"
            ) from exc
        acquired = True
        try:
            os.ftruncate(descriptor, 0)
            os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
            os.fsync(descriptor)
        except OSError as exc:
            raise TrainingError(f"Cannot record training lock owner: {exc}") from exc
        lease = LaunchLease(root, descriptor)
        yield lease
    finally:
        try:
            if acquired:
                if lease is not None:
                    lease._release()
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


@contextmanager
def exclusive_launches(run_dirs: Sequence[Path]) -> Iterator[Mapping[Path, LaunchLease]]:
    """Acquire multiple launch locks in the caller's declared global order."""

    roots = tuple(dict.fromkeys(path.resolve() for path in run_dirs))
    with ExitStack() as stack:
        leases = {
            root: stack.enter_context(exclusive_launch(root)) for root in roots
        }
        yield leases


def _signal_process_group(group_id: int, signal_number: int) -> bool:
    try:
        os.killpg(group_id, signal_number)
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        raise TrainingError(
            f"Cannot signal trainer process group {group_id}: {exc}"
        ) from exc
    return True


def _wait_for_group_exit(group_id: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.killpg(group_id, 0)
        except KeyboardInterrupt:
            continue
        except OSError as exc:
            if exc.errno == errno.ESRCH:
                return True
            if exc.errno == errno.EPERM:
                try:
                    time.sleep(0.05)
                except KeyboardInterrupt:
                    pass
                continue
            raise TrainingError(
                f"Cannot inspect trainer process group {group_id}: {exc}"
            ) from exc
        try:
            time.sleep(0.05)
        except KeyboardInterrupt:
            pass
    return False


def _wait_for_process(process: subprocess.Popen[str], timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return process.poll() is not None
        try:
            process.wait(timeout=remaining)
            return True
        except subprocess.TimeoutExpired:
            return False
        except KeyboardInterrupt:
            continue


def _stop_and_wait_once(process: subprocess.Popen[str]) -> None:
    group_id = process.pid
    if not _wait_for_process(process, _PROCESS_EXIT_GRACE_SECONDS):
        _signal_process_group(group_id, signal.SIGTERM)
        if not _wait_for_process(process, _PROCESS_GROUP_STOP_SECONDS):
            _signal_process_group(group_id, signal.SIGKILL)
            if not _wait_for_process(process, _PROCESS_GROUP_STOP_SECONDS):
                raise TrainingError(f"Trainer leader process {group_id} did not exit")
    if not _signal_process_group(group_id, signal.SIGTERM):
        return
    if _wait_for_group_exit(group_id, _PROCESS_GROUP_STOP_SECONDS):
        return
    _signal_process_group(group_id, signal.SIGKILL)
    if not _wait_for_group_exit(group_id, _PROCESS_GROUP_STOP_SECONDS):
        raise TrainingError(f"Trainer process group {group_id} did not exit")


def _stop_and_wait(process: subprocess.Popen[str]) -> None:
    while True:
        try:
            _stop_and_wait_once(process)
            return
        except (KeyboardInterrupt, SystemExit):
            continue


@contextmanager
def _forward_termination_signals(process: subprocess.Popen[str]) -> Iterator[None]:
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = {
        signal_number: signal.getsignal(signal_number)
        for signal_number in (signal.SIGINT, signal.SIGTERM)
    }

    def forward(signal_number: int, _frame: Any) -> None:
        _signal_process_group(process.pid, signal_number)
        if signal_number == signal.SIGINT:
            raise KeyboardInterrupt
        raise SystemExit(128 + signal_number)

    try:
        for signal_number in previous:
            signal.signal(signal_number, forward)
        yield
    finally:
        for signal_number, handler in previous.items():
            signal.signal(signal_number, handler)


def launch_verl(
    command: VerlCommand,
    config: TrainingConfig,
    run_dir: Path,
    *,
    lease: LaunchLease | None = None,
) -> int:
    """Run the already planned command after a no-download preflight."""

    run_dir = run_dir.resolve()
    if lease is None:
        with exclusive_launch(run_dir) as acquired:
            return launch_verl(command, config, run_dir, lease=acquired)
    lease.assert_for(run_dir)

    require_label_free_training_outputs()
    config.assert_runnable()
    world_size = config.runtime.nodes * config.runtime.gpus_per_node
    if checkpointed_step(
        run_dir,
        config.grpo.max_steps,
        expected_world_size=world_size,
    ) == config.grpo.max_steps:
        raise TrainingError("Training already reached the fixed final step")
    python = Path(config.runtime.python_executable).resolve()
    model = Path(config.runtime.model_path).resolve()
    source = Path(config.runtime.verl_source_path).resolve()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise TrainingError(f"Configured Python is not executable: {python}")
    if not model.is_dir():
        raise TrainingError(f"Configured local model snapshot is missing: {model}")
    if config.runtime.download_allowed:
        raise TrainingError("This adapter requires runtime.download_allowed=false")
    verify_verl_checkout(source, command.framework_revision)
    if command.argv[0] != config.runtime.python_executable:
        raise TrainingError("Planned command Python no longer matches the config")
    environment = build_process_environment(config, command, run_dir)
    return run_command_with_log(
        command.argv,
        cwd=source,
        environment=environment,
        log_path=run_dir / "logs" / "trainer.log",
    )


def run_command_with_log(
    argv: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    log_path: Path,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                list(argv),
                cwd=cwd,
                env=dict(environment),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            with _forward_termination_signals(process):
                try:
                    if process.stdout is None:
                        raise TrainingError("Cannot capture the Verl process output")
                    with process.stdout:
                        for line in process.stdout:
                            print(line, end="", flush=True)
                            log.write(line)
                            log.flush()
                    return process.wait()
                finally:
                    _stop_and_wait(process)
    except OSError as exc:
        raise TrainingError(f"Cannot launch the Verl process: {exc}") from exc


def validate_export_destination(
    run_dir: Path,
    actor_checkpoint: Path,
    export_directory: Path,
) -> Path:
    run_root = run_dir.resolve()
    actor_root = actor_checkpoint.resolve()
    export_root = export_directory.resolve()
    if (
        export_root == run_root
        or export_root in run_root.parents
        or run_root in export_root.parents
    ):
        raise TrainingError(
            "Export directory must not overlap the training run directory"
        )
    if (
        export_root == actor_root
        or export_root in actor_root.parents
        or actor_root in export_root.parents
    ):
        raise TrainingError(
            "Export directory must not overlap the actor checkpoint directory"
        )
    return export_root


def merge_command(
    config: TrainingConfig,
    *,
    run_dir: Path,
    export_directory: Path,
) -> tuple[str, ...]:
    world_size = config.runtime.nodes * config.runtime.gpus_per_node
    step = checkpointed_step(
        run_dir,
        config.grpo.max_steps,
        expected_world_size=world_size,
    )
    if step != config.grpo.max_steps:
        raise TrainingError(
            f"Expected completed step {config.grpo.max_steps}, found {step}"
        )
    actor_checkpoint = (
        run_dir.resolve()
        / "checkpoints"
        / f"global_step_{config.grpo.max_steps}"
        / "actor"
    )
    export_directory = validate_export_destination(
        run_dir,
        actor_checkpoint,
        export_directory,
    )
    return (
        config.runtime.python_executable,
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        str(actor_checkpoint),
        "--target_dir",
        str(export_directory),
    )
