"""Pinned verl v0.5.0 command compilation and launch preflight.

Nothing in this module imports verl, Torch, Ray, Transformers, or vLLM.  Planning
therefore cannot initialize a model or a GPU runtime.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from compute_as_a_teacher.evaluation.artifacts import canonical_json_bytes, sha256_bytes

from .config import SUPPORTED_VERL_REVISION, TrainingConfig
from .errors import TrainingError


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
    dataset_module = (
        repository_root
        / "src/compute_as_a_teacher/training/verl_dataset.py"
    ).resolve()
    checkpoints = (run_dir / "checkpoints").resolve()
    rollout = config.rollouts.sampling
    synthesis = config.synthesis.sampling
    runtime = config.runtime

    overrides = [
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
        f"data.custom_cls.path={_hydra_string(str(dataset_module))}",
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
        f"trainer.project_name={_hydra_string('compute-as-a-teacher')}",
        f"trainer.experiment_name={_hydra_string(config.run_name)}",
        "trainer.balance_batch=False",
        "trainer.val_before_train=False",
        "trainer.test_freq=-1",
        "trainer.log_val_generations=0",
        f"trainer.save_freq={config.checkpointing.save_every_steps}",
        f"trainer.resume_mode={config.checkpointing.resume_mode}",
        f"trainer.default_local_dir={_hydra_string(str(checkpoints))}",
        f"trainer.max_actor_ckpt_to_keep={config.checkpointing.max_checkpoints}",
        f"trainer.nnodes={runtime.nodes}",
        f"trainer.n_gpus_per_node={runtime.gpus_per_node}",
        f"trainer.logger=[{','.join(runtime.logger)}]",
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
        "TOKENIZERS_PARALLELISM": "true",
    }
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


def checkpointed_step(run_dir: Path, max_steps: int) -> int:
    tracker = run_dir.resolve() / "checkpoints/latest_checkpointed_iteration.txt"
    if not tracker.exists():
        return 0
    try:
        step = int(tracker.read_text(encoding="utf-8").strip())
    except (OSError, ValueError) as exc:
        raise TrainingError(f"Cannot read checkpoint tracker {tracker}: {exc}") from exc
    if not 0 <= step <= max_steps:
        raise TrainingError(f"Checkpoint step {step} is outside [0, {max_steps}]")
    return step


def launch_verl(command: VerlCommand, config: TrainingConfig, run_dir: Path) -> int:
    """Run the already planned command after a no-download preflight."""

    config.assert_runnable()
    if checkpointed_step(run_dir, config.grpo.max_steps) == config.grpo.max_steps:
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
    environment = os.environ.copy()
    environment.update(command.environment)
    completed = subprocess.run(
        list(command.argv),
        cwd=source,
        env=environment,
        check=False,
    )
    return completed.returncode


def merge_command(
    config: TrainingConfig,
    *,
    run_dir: Path,
    export_directory: Path,
) -> tuple[str, ...]:
    actor_checkpoint = (
        run_dir.resolve()
        / "checkpoints"
        / f"global_step_{config.grpo.max_steps}"
        / "actor"
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
        str(export_directory.resolve()),
    )
