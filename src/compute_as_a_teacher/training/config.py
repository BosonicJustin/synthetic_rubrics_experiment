"""Strict semantic configuration for the MATH-500 CaT GRPO experiment.

The TOML file is the experiment contract.  Framework-specific Hydra overrides are
derived from it; they are never the source of truth.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from compute_as_a_teacher._toml import tomllib

from compute_as_a_teacher.evaluation.artifacts import canonical_json_bytes, sha256_bytes
from compute_as_a_teacher.evaluation.config import PromptSpec
from compute_as_a_teacher.evaluation.grading import PRIMARY_GRADER
from compute_as_a_teacher.evaluation.schemas import ModelSpec, SamplingSpec

from .errors import TrainingError


TRAINING_PROTOCOL_VERSION = "cat_math500_grpo_verl_v1"
TRAINING_KIND = "cat_grpo"
TRAINING_SCHEMA_VERSION = 2
SUPPORTED_VERL_REVISION = "8fdc4d3f202f41461f4de9f42a637228e342668b"
SUPPORTED_VERL_RELEASE = "0.5.0"
SUPPORTED_ADAPTER_VERSION = "cat-verl-batch-reward-v1"
SUPPORTED_WANDB_SDK_VERSION = "0.21.1"

_FORBIDDEN_CONFIG_KEYS = frozenset(
    {
        "answer",
        "answers",
        "ground_truth",
        "label",
        "labels",
        "labels_path",
        "reference",
        "reference_answer",
        "solution",
        "solutions",
    }
)
_UNRESOLVED_PREFIXES = ("required_", "replace_", "todo_", "<")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise TrainingError(f"{name} must be a TOML table")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise TrainingError(
            f"{name} keys are {sorted(value)}, expected exactly {sorted(expected)}"
        )


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrainingError(f"{name} must be a nonempty string")
    return value


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TrainingError(f"{name} must be a boolean")
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TrainingError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingError(f"{name} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized) or (
        minimum is not None and normalized < minimum
    ):
        suffix = f" >= {minimum}" if minimum is not None else ""
        raise TrainingError(f"{name} must be a finite number{suffix}")
    return normalized


def _reject_forbidden_keys(value: Any, *, path: str = "config") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in _FORBIDDEN_CONFIG_KEYS:
                raise TrainingError(
                    f"Training config cannot contain evaluation-only field {path}.{key}"
                )
            _reject_forbidden_keys(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_keys(child, path=f"{path}[{index}]")


def _looks_unresolved(value: str) -> bool:
    return value.strip().lower().startswith(_UNRESOLVED_PREFIXES)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class AnchorSpec:
    source: str
    frozen: bool
    model: ModelSpec


@dataclass(frozen=True, slots=True)
class RolloutSpec:
    group_size: int
    fresh_per_step: bool
    prompt: PromptSpec
    sampling: SamplingSpec

@dataclass(frozen=True, slots=True)
class SynthesisTrainingSpec:
    required_rollouts: int
    rollout_text_only: bool
    anchor_role: str
    prompt: PromptSpec
    sampling: SamplingSpec

@dataclass(frozen=True, slots=True)
class RewardSpec:
    kind: str
    extractor: str
    labels_allowed: bool
    max_answer_chars: int
    invalid_anchor: str

@dataclass(frozen=True, slots=True)
class AdvantageSpec:
    kind: str
    std_ddof: int
    epsilon: float
    zero_variance: str

@dataclass(frozen=True, slots=True)
class GrpoSpec:
    algorithm: str
    global_batch_size: int
    batch_size_unit: str
    learning_rate: float
    lr_scheduler: str
    warmup_steps: int
    max_steps: int
    kl_placement: str
    kl_coefficient: float
    clip_epsilon: float
    ppo_epochs: int
    ppo_mini_batch_size: int
    normalize_advantages: bool

@dataclass(frozen=True, slots=True)
class OptimizerSpec:
    name: str
    betas: tuple[float, float]
    epsilon: float
    weight_decay: float
    max_grad_norm: float

@dataclass(frozen=True, slots=True)
class CheckpointingSpec:
    save_every_steps: int
    selected_checkpoint: str
    resume_mode: str
    max_checkpoints: int

@dataclass(frozen=True, slots=True)
class WandbSpec:
    enabled: bool
    project: str
    entity: str
    mode: str
    sdk_version: str
    api_key_env: str
    resume: str
    group: str
    tags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TrackingSpec:
    console: bool
    wandb: WandbSpec


@dataclass(frozen=True, slots=True)
class RuntimeSpec:
    framework: str
    framework_release: str
    framework_revision: str
    adapter_version: str
    python_executable: str
    verl_source_path: str
    model_path: str
    model_snapshot_tree_sha256: str
    trainer_image_digest: str
    package_inventory_sha256: str
    anchor_base_url: str
    anchor_model: str
    anchor_api_key_env: str
    anchor_timeout_seconds: int
    anchor_max_concurrency: int
    strategy: str
    nodes: int
    gpus_per_node: int
    minimum_gpu_free_memory_fraction: float
    dtype: str
    rollout_engine: str
    tensor_parallel_size: int
    gpu_memory_utilization: float
    max_prompt_tokens: int
    max_tokens_per_gpu: int
    dataloader_workers: int
    seed: int
    download_allowed: bool

    def unresolved_reasons(self) -> tuple[str, ...]:
        reasons = []
        for field_name in ("python_executable", "verl_source_path", "model_path"):
            value = getattr(self, field_name)
            if _looks_unresolved(value):
                reasons.append(f"runtime.{field_name} is unresolved")
            elif not Path(value).is_absolute():
                reasons.append(f"runtime.{field_name} must be an absolute path")
        if not re.fullmatch(r"[0-9a-f]{40}", self.framework_revision):
            reasons.append("runtime.framework_revision is not a full commit SHA")
        for field_name in (
            "model_snapshot_tree_sha256",
            "package_inventory_sha256",
        ):
            value = getattr(self, field_name)
            if _looks_unresolved(value):
                reasons.append(f"runtime.{field_name} is unresolved")
            elif not re.fullmatch(r"[0-9a-f]{64}", value):
                reasons.append(f"runtime.{field_name} must be a lowercase SHA-256")
        if _looks_unresolved(self.trainer_image_digest):
            reasons.append("runtime.trainer_image_digest is unresolved")
        elif not re.fullmatch(r"sha256:[0-9a-f]{64}", self.trainer_image_digest):
            reasons.append(
                "runtime.trainer_image_digest must be an immutable sha256 digest"
            )
        return tuple(reasons)

@dataclass(frozen=True, slots=True)
class TrainingConfig:
    schema_version: int
    kind: str
    protocol_version: str
    run_name: str
    questions_path: str
    dataset_lock_path: str
    policy: ModelSpec
    anchor: AnchorSpec
    rollouts: RolloutSpec
    synthesis: SynthesisTrainingSpec
    reward: RewardSpec
    advantage: AdvantageSpec
    grpo: GrpoSpec
    optimizer: OptimizerSpec
    checkpointing: CheckpointingSpec
    tracking: TrackingSpec
    runtime: RuntimeSpec

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    @property
    def fingerprint(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_dict()))

    def unresolved_reasons(self) -> tuple[str, ...]:
        reasons = list(self.policy.unresolved_reasons())
        reasons.extend(
            reason.replace("model.", "anchor.model.", 1)
            for reason in self.anchor.model.unresolved_reasons()
        )
        reasons.extend(self.runtime.unresolved_reasons())
        return tuple(reasons)

    def assert_runnable(self) -> None:
        reasons = self.unresolved_reasons()
        if reasons:
            raise TrainingError(
                "Training configuration is not runnable: " + "; ".join(reasons)
            )


def _prompt(value: Any, name: str) -> PromptSpec:
    try:
        return PromptSpec.from_dict(_mapping(value, name))
    except Exception as exc:
        if isinstance(exc, TrainingError):
            raise
        raise TrainingError(str(exc)) from exc


def _sampling(value: Any, name: str) -> SamplingSpec:
    try:
        return SamplingSpec.from_dict(_mapping(value, name))
    except Exception as exc:
        raise TrainingError(str(exc)) from exc


def _model(
    value: Any,
    name: str,
    *,
    allow_unresolved: bool,
) -> ModelSpec:
    try:
        return ModelSpec.from_dict(
            _mapping(value, name),
            allow_unresolved=allow_unresolved,
        )
    except Exception as exc:
        raise TrainingError(str(exc)) from exc


def _parse_anchor(value: Any, *, allow_unresolved: bool) -> AnchorSpec:
    table = _mapping(value, "anchor")
    _exact_keys(table, {"source", "frozen", "model"}, "anchor")
    return AnchorSpec(
        source=_text(table["source"], "anchor.source"),
        frozen=_bool(table["frozen"], "anchor.frozen"),
        model=_model(table["model"], "anchor.model", allow_unresolved=allow_unresolved),
    )


def _parse_rollouts(value: Any) -> RolloutSpec:
    table = _mapping(value, "rollouts")
    _exact_keys(table, {"group_size", "fresh_per_step", "prompt", "sampling"}, "rollouts")
    return RolloutSpec(
        group_size=_integer(table["group_size"], "rollouts.group_size", minimum=1),
        fresh_per_step=_bool(table["fresh_per_step"], "rollouts.fresh_per_step"),
        prompt=_prompt(table["prompt"], "rollouts.prompt"),
        sampling=_sampling(table["sampling"], "rollouts.sampling"),
    )


def _parse_synthesis(value: Any) -> SynthesisTrainingSpec:
    table = _mapping(value, "synthesis")
    _exact_keys(
        table,
        {"required_rollouts", "rollout_text_only", "anchor_role", "prompt", "sampling"},
        "synthesis",
    )
    return SynthesisTrainingSpec(
        required_rollouts=_integer(table["required_rollouts"], "synthesis.required_rollouts", minimum=1),
        rollout_text_only=_bool(table["rollout_text_only"], "synthesis.rollout_text_only"),
        anchor_role=_text(table["anchor_role"], "synthesis.anchor_role"),
        prompt=_prompt(table["prompt"], "synthesis.prompt"),
        sampling=_sampling(table["sampling"], "synthesis.sampling"),
    )


def _parse_reward(value: Any) -> RewardSpec:
    table = _mapping(value, "reward")
    _exact_keys(
        table,
        {"kind", "extractor", "labels_allowed", "max_answer_chars", "invalid_anchor"},
        "reward",
    )
    return RewardSpec(
        kind=_text(table["kind"], "reward.kind"),
        extractor=_text(table["extractor"], "reward.extractor"),
        labels_allowed=_bool(table["labels_allowed"], "reward.labels_allowed"),
        max_answer_chars=_integer(table["max_answer_chars"], "reward.max_answer_chars", minimum=1),
        invalid_anchor=_text(table["invalid_anchor"], "reward.invalid_anchor"),
    )


def _parse_advantage(value: Any) -> AdvantageSpec:
    table = _mapping(value, "advantage")
    _exact_keys(table, {"kind", "std_ddof", "epsilon", "zero_variance"}, "advantage")
    return AdvantageSpec(
        kind=_text(table["kind"], "advantage.kind"),
        std_ddof=_integer(table["std_ddof"], "advantage.std_ddof"),
        epsilon=_number(table["epsilon"], "advantage.epsilon", minimum=0.0),
        zero_variance=_text(table["zero_variance"], "advantage.zero_variance"),
    )


def _parse_grpo(value: Any) -> GrpoSpec:
    table = _mapping(value, "grpo")
    expected = {
        "algorithm", "global_batch_size", "batch_size_unit", "learning_rate",
        "lr_scheduler", "warmup_steps", "max_steps", "kl_placement",
        "kl_coefficient", "clip_epsilon", "ppo_epochs", "ppo_mini_batch_size",
        "normalize_advantages",
    }
    _exact_keys(table, expected, "grpo")
    return GrpoSpec(
        algorithm=_text(table["algorithm"], "grpo.algorithm"),
        global_batch_size=_integer(table["global_batch_size"], "grpo.global_batch_size", minimum=1),
        batch_size_unit=_text(table["batch_size_unit"], "grpo.batch_size_unit"),
        learning_rate=_number(table["learning_rate"], "grpo.learning_rate", minimum=0.0),
        lr_scheduler=_text(table["lr_scheduler"], "grpo.lr_scheduler"),
        warmup_steps=_integer(table["warmup_steps"], "grpo.warmup_steps"),
        max_steps=_integer(table["max_steps"], "grpo.max_steps", minimum=1),
        kl_placement=_text(table["kl_placement"], "grpo.kl_placement"),
        kl_coefficient=_number(table["kl_coefficient"], "grpo.kl_coefficient", minimum=0.0),
        clip_epsilon=_number(table["clip_epsilon"], "grpo.clip_epsilon", minimum=0.0),
        ppo_epochs=_integer(table["ppo_epochs"], "grpo.ppo_epochs", minimum=1),
        ppo_mini_batch_size=_integer(table["ppo_mini_batch_size"], "grpo.ppo_mini_batch_size", minimum=1),
        normalize_advantages=_bool(table["normalize_advantages"], "grpo.normalize_advantages"),
    )


def _parse_optimizer(value: Any) -> OptimizerSpec:
    table = _mapping(value, "optimizer")
    _exact_keys(table, {"name", "betas", "epsilon", "weight_decay", "max_grad_norm"}, "optimizer")
    betas = table["betas"]
    if not isinstance(betas, list) or len(betas) != 2:
        raise TrainingError("optimizer.betas must contain exactly two numbers")
    normalized_betas = tuple(_number(item, f"optimizer.betas[{index}]", minimum=0.0) for index, item in enumerate(betas))
    if any(beta >= 1 for beta in normalized_betas):
        raise TrainingError("optimizer.betas values must be < 1")
    return OptimizerSpec(
        name=_text(table["name"], "optimizer.name"),
        betas=(normalized_betas[0], normalized_betas[1]),
        epsilon=_number(table["epsilon"], "optimizer.epsilon", minimum=0.0),
        weight_decay=_number(table["weight_decay"], "optimizer.weight_decay", minimum=0.0),
        max_grad_norm=_number(table["max_grad_norm"], "optimizer.max_grad_norm", minimum=0.0),
    )


def _parse_checkpointing(value: Any) -> CheckpointingSpec:
    table = _mapping(value, "checkpointing")
    _exact_keys(table, {"save_every_steps", "selected_checkpoint", "resume_mode", "max_checkpoints"}, "checkpointing")
    return CheckpointingSpec(
        save_every_steps=_integer(table["save_every_steps"], "checkpointing.save_every_steps", minimum=1),
        selected_checkpoint=_text(table["selected_checkpoint"], "checkpointing.selected_checkpoint"),
        resume_mode=_text(table["resume_mode"], "checkpointing.resume_mode"),
        max_checkpoints=_integer(table["max_checkpoints"], "checkpointing.max_checkpoints", minimum=1),
    )


def _optional_text(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TrainingError(f"{name} must be a string")
    if value != value.strip() or "\0" in value or "\n" in value or "\r" in value:
        raise TrainingError(f"{name} must not contain surrounding or control whitespace")
    return value


def _parse_tracking(value: Any) -> TrackingSpec:
    table = _mapping(value, "tracking")
    _exact_keys(table, {"console", "wandb"}, "tracking")
    console = _bool(table["console"], "tracking.console")
    if not console:
        raise TrainingError("tracking.console must remain enabled")

    wandb = _mapping(table["wandb"], "tracking.wandb")
    _exact_keys(
        wandb,
        {
            "enabled",
            "project",
            "entity",
            "mode",
            "sdk_version",
            "api_key_env",
            "resume",
            "group",
            "tags",
        },
        "tracking.wandb",
    )
    enabled = _bool(wandb["enabled"], "tracking.wandb.enabled")
    project = _optional_text(wandb["project"], "tracking.wandb.project")
    entity = _optional_text(wandb["entity"], "tracking.wandb.entity")
    mode = _text(wandb["mode"], "tracking.wandb.mode")
    sdk_version = _text(wandb["sdk_version"], "tracking.wandb.sdk_version")
    api_key_env = _text(wandb["api_key_env"], "tracking.wandb.api_key_env")
    resume = _text(wandb["resume"], "tracking.wandb.resume")
    group = _optional_text(wandb["group"], "tracking.wandb.group")
    tags = wandb["tags"]
    if not isinstance(tags, list) or not all(isinstance(item, str) for item in tags):
        raise TrainingError("tracking.wandb.tags must be a list of strings")
    if len(tags) != len(set(tags)):
        raise TrainingError("tracking.wandb.tags must not contain duplicates")
    if any(
        not item
        or item != item.strip()
        or len(item) > 64
        or any(character in item for character in ",\0\n\r")
        for item in tags
    ):
        raise TrainingError(
            "tracking.wandb.tags entries must be nonempty, comma-free strings of at most 64 characters"
        )
    if mode != "online":
        raise TrainingError(
            "tracking.wandb.mode must be 'online'; W&B 0.21.1 ignores resume "
            "in offline mode"
        )
    if resume != "allow":
        raise TrainingError("tracking.wandb.resume must be 'allow' for restart-safe identity")
    if sdk_version != SUPPORTED_WANDB_SDK_VERSION:
        raise TrainingError(
            f"tracking.wandb.sdk_version must be {SUPPORTED_WANDB_SDK_VERSION!r}"
        )
    if not re.fullmatch(r"[A-Z_][A-Z0-9_]{0,127}", api_key_env) or "WANDB" not in api_key_env:
        raise TrainingError(
            "tracking.wandb.api_key_env must name an uppercase W&B environment variable"
        )
    slug = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
    if enabled and (not slug.fullmatch(project) or not slug.fullmatch(entity)):
        raise TrainingError(
            "enabled W&B tracking requires explicit project and entity slugs"
        )
    if group and not slug.fullmatch(group):
        raise TrainingError("tracking.wandb.group must be empty or a valid slug")
    return TrackingSpec(
        console=True,
        wandb=WandbSpec(
            enabled=enabled,
            project=project,
            entity=entity,
            mode=mode,
            sdk_version=sdk_version,
            api_key_env=api_key_env,
            resume=resume,
            group=group,
            tags=tuple(tags),
        ),
    )


def _parse_runtime(value: Any) -> RuntimeSpec:
    table = _mapping(value, "runtime")
    expected = {
        "framework", "framework_release", "framework_revision", "adapter_version",
        "python_executable", "verl_source_path", "model_path",
        "model_snapshot_tree_sha256", "trainer_image_digest",
        "package_inventory_sha256", "anchor_base_url",
        "anchor_model", "anchor_api_key_env", "anchor_timeout_seconds",
        "anchor_max_concurrency", "strategy", "nodes", "gpus_per_node",
        "minimum_gpu_free_memory_fraction", "dtype",
        "rollout_engine", "tensor_parallel_size", "gpu_memory_utilization",
        "max_prompt_tokens", "max_tokens_per_gpu", "dataloader_workers", "seed",
        "download_allowed",
    }
    _exact_keys(table, expected, "runtime")
    gpu_memory = _number(table["gpu_memory_utilization"], "runtime.gpu_memory_utilization", minimum=0.0)
    if not 0 < gpu_memory < 1:
        raise TrainingError("runtime.gpu_memory_utilization must be in (0, 1)")
    minimum_free_memory = _number(
        table["minimum_gpu_free_memory_fraction"],
        "runtime.minimum_gpu_free_memory_fraction",
        minimum=0.0,
    )
    if not 0 < minimum_free_memory <= 1:
        raise TrainingError(
            "runtime.minimum_gpu_free_memory_fraction must be in (0, 1]"
        )
    seed = _integer(table["seed"], "runtime.seed")
    if seed >= 2**31:
        raise TrainingError("runtime.seed must be below 2^31")
    return RuntimeSpec(
        framework=_text(table["framework"], "runtime.framework"),
        framework_release=_text(table["framework_release"], "runtime.framework_release"),
        framework_revision=_text(table["framework_revision"], "runtime.framework_revision"),
        adapter_version=_text(table["adapter_version"], "runtime.adapter_version"),
        python_executable=_text(table["python_executable"], "runtime.python_executable"),
        verl_source_path=_text(table["verl_source_path"], "runtime.verl_source_path"),
        model_path=_text(table["model_path"], "runtime.model_path"),
        model_snapshot_tree_sha256=_text(
            table["model_snapshot_tree_sha256"],
            "runtime.model_snapshot_tree_sha256",
        ),
        trainer_image_digest=_text(
            table["trainer_image_digest"], "runtime.trainer_image_digest"
        ),
        package_inventory_sha256=_text(
            table["package_inventory_sha256"],
            "runtime.package_inventory_sha256",
        ),
        anchor_base_url=_text(table["anchor_base_url"], "runtime.anchor_base_url"),
        anchor_model=_text(table["anchor_model"], "runtime.anchor_model"),
        anchor_api_key_env=_text(table["anchor_api_key_env"], "runtime.anchor_api_key_env"),
        anchor_timeout_seconds=_integer(table["anchor_timeout_seconds"], "runtime.anchor_timeout_seconds", minimum=1),
        anchor_max_concurrency=_integer(table["anchor_max_concurrency"], "runtime.anchor_max_concurrency", minimum=1),
        strategy=_text(table["strategy"], "runtime.strategy"),
        nodes=_integer(table["nodes"], "runtime.nodes", minimum=1),
        gpus_per_node=_integer(table["gpus_per_node"], "runtime.gpus_per_node", minimum=1),
        minimum_gpu_free_memory_fraction=minimum_free_memory,
        dtype=_text(table["dtype"], "runtime.dtype"),
        rollout_engine=_text(table["rollout_engine"], "runtime.rollout_engine"),
        tensor_parallel_size=_integer(table["tensor_parallel_size"], "runtime.tensor_parallel_size", minimum=1),
        gpu_memory_utilization=gpu_memory,
        max_prompt_tokens=_integer(table["max_prompt_tokens"], "runtime.max_prompt_tokens", minimum=1),
        max_tokens_per_gpu=_integer(table["max_tokens_per_gpu"], "runtime.max_tokens_per_gpu", minimum=1),
        dataloader_workers=_integer(table["dataloader_workers"], "runtime.dataloader_workers"),
        seed=seed,
        download_allowed=_bool(table["download_allowed"], "runtime.download_allowed"),
    )


def _validate_paper_profile(config: TrainingConfig) -> None:
    failures: list[str] = []
    if config.policy != config.anchor.model:
        failures.append("anchor.model must exactly equal the initial policy")
    model_expected = {
        "provider": "huggingface",
        "model_id": "Qwen/Qwen3-4B",
        "tokenizer_id": "Qwen/Qwen3-4B",
        "adapter_version": "transformers-vllm-pinned-by-verl-v0.5.0",
        "seed_support": "best_effort",
    }
    for name, expected in model_expected.items():
        if getattr(config.policy, name) != expected:
            failures.append(f"policy.{name} must be {expected!r}")
    if config.policy.dtype != "bfloat16":
        failures.append("policy.dtype must be 'bfloat16'")
    if config.policy.quantization != "none":
        failures.append("policy.quantization must be 'none'")
    if config.policy.dtype != config.runtime.dtype:
        failures.append("policy.dtype must equal runtime.dtype")
    if config.anchor.source != "initial_policy" or not config.anchor.frozen:
        failures.append("anchor must be the frozen initial_policy")
    if config.rollouts.group_size != 8 or not config.rollouts.fresh_per_step:
        failures.append("rollouts must be 8 fresh current-policy samples per step")
    if config.synthesis.required_rollouts != 8:
        failures.append("synthesis must consume exactly 8 rollouts")
    if not config.synthesis.rollout_text_only or config.synthesis.anchor_role != "initial_policy":
        failures.append("synthesis must send rollout text only to initial_policy")
    prompt_expected = {
        "rollouts": PromptSpec(
            path="prompts/math500/solve_v1.txt",
            version="raw_math500_local_v1",
            prefix="/no_think\n",
        ),
        "synthesis": PromptSpec(
            path="prompts/math500/synthesis_cot_appendix_f_literal.txt",
            version="paper_appendix_f_cot_literal_v1",
            prefix="/no_think\n",
        ),
    }
    for role, expected in prompt_expected.items():
        if getattr(config, role).prompt != expected:
            failures.append(f"{role}.prompt must match the registered local choice")
    if config.reward != RewardSpec(
        kind="pseudo_reference_boxed_exact",
        extractor=PRIMARY_GRADER,
        labels_allowed=False,
        max_answer_chars=50_000,
        invalid_anchor="fail_closed",
    ):
        failures.append(
            "reward must use the locked 50,000-character, label-free exact boxed "
            "agreement contract and fail closed"
        )
    if config.advantage != AdvantageSpec(
        kind="group_zscore",
        std_ddof=1,
        epsilon=1e-6,
        zero_variance="zero_advantages",
    ):
        failures.append("advantage must match the pinned verl v0.5.0 GRPO implementation")
    expected_grpo = {
        "algorithm": "grpo",
        "global_batch_size": 256,
        "batch_size_unit": "prompts",
        "learning_rate": 5e-7,
        "lr_scheduler": "constant",
        "warmup_steps": 0,
        "max_steps": 1000,
        "kl_placement": "reward",
        "kl_coefficient": 1e-3,
        "clip_epsilon": 0.2,
        "ppo_epochs": 1,
        "ppo_mini_batch_size": 256,
        "normalize_advantages": True,
    }
    for name, expected in expected_grpo.items():
        if getattr(config.grpo, name) != expected:
            failures.append(f"grpo.{name} must be {expected!r}")
    if config.optimizer != OptimizerSpec(
        name="adamw",
        betas=(0.9, 0.999),
        epsilon=1e-8,
        weight_decay=0.01,
        max_grad_norm=1.0,
    ):
        failures.append("optimizer must match the declared verl v0.5.0 AdamW defaults")
    runtime_expected = {
        "framework": "verl",
        "framework_release": SUPPORTED_VERL_RELEASE,
        "framework_revision": SUPPORTED_VERL_REVISION,
        "adapter_version": SUPPORTED_ADAPTER_VERSION,
        "strategy": "fsdp",
        "nodes": 1,
        "gpus_per_node": 8,
        "dtype": "bfloat16",
        "rollout_engine": "vllm",
        "seed": 42,
        "download_allowed": False,
    }
    for name, expected in runtime_expected.items():
        if getattr(config.runtime, name) != expected:
            failures.append(f"runtime.{name} must be {expected!r}")
    if config.runtime.max_tokens_per_gpu < (
        config.runtime.max_prompt_tokens + config.rollouts.sampling.max_new_tokens
    ):
        failures.append("runtime.max_tokens_per_gpu must cover one prompt and response")
    total_gpus = config.runtime.nodes * config.runtime.gpus_per_node
    if total_gpus % config.runtime.tensor_parallel_size:
        failures.append("total GPUs must be divisible by runtime.tensor_parallel_size")
    if config.grpo.global_batch_size < config.grpo.ppo_mini_batch_size:
        failures.append("global prompt batch must cover one PPO mini-batch")
    trajectory_batch = config.grpo.global_batch_size * config.rollouts.group_size
    if trajectory_batch % total_gpus:
        failures.append("trajectory batch must be divisible by total GPUs")
    for role, sampling, base_seed in (
        ("rollouts", config.rollouts.sampling, 1729),
        ("synthesis", config.synthesis.sampling, 2718),
    ):
        if (
            not sampling.do_sample
            or sampling.temperature != 0.7
            or sampling.top_p != 0.8
            or sampling.top_k != 20
            or sampling.max_new_tokens != 1536
            or sampling.num_beams != 1
            or sampling.repetition_penalty != 1.0
            or sampling.stop
            or sampling.base_seed != base_seed
        ):
            failures.append(
                f"{role}.sampling must match the registered Qwen3-4B profile"
            )
    if "qwen3-4b" in config.policy.model_id.lower():
        if config.rollouts.prompt.prefix != "/no_think\n" or config.synthesis.prompt.prefix != "/no_think\n":
            failures.append("Qwen3-4B requires the exact '/no_think\\n' prefix")
    checkpoint_expected = {
        "save_every_steps": 100,
        "selected_checkpoint": "fixed_final_step",
        "resume_mode": "auto",
        "max_checkpoints": 3,
    }
    for name, expected in checkpoint_expected.items():
        if getattr(config.checkpointing, name) != expected:
            failures.append(f"checkpointing.{name} must be {expected!r}")
    if failures:
        raise TrainingError("Training config violates the registered protocol: " + "; ".join(failures))


def load_training_config(
    path: Path,
    *,
    allow_unresolved: bool = False,
) -> TrainingConfig:
    try:
        with path.open("rb") as handle:
            value = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise TrainingError(f"Cannot read TOML config {path}: {exc}") from exc
    _reject_forbidden_keys(value)
    if (
        value.get("schema_version") != TRAINING_SCHEMA_VERSION
        or value.get("kind") != TRAINING_KIND
    ):
        raise TrainingError(
            f"Training config must use schema_version={TRAINING_SCHEMA_VERSION} "
            f"and kind={TRAINING_KIND!r}"
        )
    expected = {
        "schema_version", "kind", "protocol_version", "run_name", "questions_path",
        "dataset_lock_path", "policy", "anchor", "rollouts", "synthesis",
        "reward", "advantage", "grpo", "optimizer", "checkpointing", "tracking",
        "runtime",
    }
    _exact_keys(value, expected, "training config")
    protocol = _text(value["protocol_version"], "protocol_version")
    if protocol != TRAINING_PROTOCOL_VERSION:
        raise TrainingError(f"protocol_version must be {TRAINING_PROTOCOL_VERSION!r}")
    config = TrainingConfig(
        schema_version=TRAINING_SCHEMA_VERSION,
        kind=TRAINING_KIND,
        protocol_version=protocol,
        run_name=_text(value["run_name"], "run_name"),
        questions_path=_text(value["questions_path"], "questions_path"),
        dataset_lock_path=_text(value["dataset_lock_path"], "dataset_lock_path"),
        policy=_model(value["policy"], "policy", allow_unresolved=allow_unresolved),
        anchor=_parse_anchor(value["anchor"], allow_unresolved=allow_unresolved),
        rollouts=_parse_rollouts(value["rollouts"]),
        synthesis=_parse_synthesis(value["synthesis"]),
        reward=_parse_reward(value["reward"]),
        advantage=_parse_advantage(value["advantage"]),
        grpo=_parse_grpo(value["grpo"]),
        optimizer=_parse_optimizer(value["optimizer"]),
        checkpointing=_parse_checkpointing(value["checkpointing"]),
        tracking=_parse_tracking(value["tracking"]),
        runtime=_parse_runtime(value["runtime"]),
    )
    _validate_paper_profile(config)
    if not allow_unresolved:
        config.assert_runnable()
    return config
