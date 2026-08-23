"""Generate raw and inference-time-synthesis eval configs for a trained checkpoint."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from compute_as_a_teacher.evaluation.artifacts import (
    artifact_reference,
    canonical_json_bytes,
    publish_bytes,
    publish_json,
    read_json,
    sha256_bytes,
)
from compute_as_a_teacher.evaluation.config import MATH500_PROTOCOL_VERSION
from compute_as_a_teacher.evaluation.errors import EvaluationError
from compute_as_a_teacher.evaluation.schemas import ModelSpec, SamplingSpec

from .checkpoints import (
    load_registered_checkpoint,
    load_registered_checkpoint_artifacts,
)
from .errors import TrainingError
from .planning import load_training_plan
from .verl_adapter import LaunchLease, exclusive_launch


RAW_CONFIG_NAME = "math500_trained_raw.toml"
SYNTHESIS_CONFIG_NAME = "math500_trained_synthesis.toml"
EVAL_HANDOFF_NAME = "eval_handoff.json"


def _quoted(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _model_toml(model: ModelSpec) -> str:
    return "\n".join(
        f"{name} = {_quoted(value)}" for name, value in model.to_dict().items()
    )


def _sampling_toml(sampling: SamplingSpec) -> str:
    value = sampling.to_dict()
    lines = [
        f"do_sample = {'true' if value['do_sample'] else 'false'}",
        f"temperature = {value['temperature']}",
        f"top_p = {value['top_p']}",
        f"top_k = {value['top_k']}",
        f"max_new_tokens = {value['max_new_tokens']}",
        f"num_beams = {value['num_beams']}",
        f"repetition_penalty = {value['repetition_penalty']}",
        "stop = [" + ", ".join(_quoted(item) for item in value["stop"]) + "]",
        f"base_seed = {value['base_seed']}",
    ]
    return "\n".join(lines)


def _served_model_id(value: str, name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise TrainingError(
            f"{name} must use only letters, digits, dot, underscore, and hyphen"
        )
    return value


def _endpoint_model(base: dict[str, Any], model_id: str, revision: str) -> ModelSpec:
    return ModelSpec(
        provider="openai-compatible",
        model_id=model_id,
        revision=revision,
        tokenizer_id=base["tokenizer_id"],
        tokenizer_revision=base["tokenizer_revision"],
        chat_template_sha256=base["chat_template_sha256"],
        adapter_version="openai-compatible-chat-v1",
        dtype=base["dtype"],
        quantization=base["quantization"],
        seed_support="best_effort",
    )


def _trained_model(
    plan: dict[str, Any],
    checkpoint: dict[str, Any],
    served_model: str,
) -> ModelSpec:
    base = plan["config"]["policy"]
    return _endpoint_model(
        base,
        _served_model_id(served_model, "served_model"),
        checkpoint["export"]["tree_sha256"],
    )


def _initial_anchor_model(plan: dict[str, Any]) -> ModelSpec:
    config = plan["config"]
    anchor = config["anchor"]
    base = anchor["model"]
    if (
        anchor.get("source") != "initial_policy"
        or anchor.get("frozen") is not True
        or base != config["policy"]
    ):
        raise TrainingError("Training plan does not bind a frozen initial-policy anchor")
    return _endpoint_model(
        base,
        _served_model_id(config["runtime"]["anchor_model"], "runtime.anchor_model"),
        base["revision"],
    )


def _raw_toml(plan: dict[str, Any], model: ModelSpec) -> bytes:
    config = plan["config"]
    prompt = config["rollouts"]["prompt"]
    sampling = SamplingSpec.from_dict(config["rollouts"]["sampling"])
    text = f"""schema_version = 1
kind = "raw"
protocol_version = {_quoted(MATH500_PROTOCOL_VERSION)}
run_name = {_quoted(config['run_name'] + '-trained-final-raw')}
questions_path = {_quoted(config['questions_path'])}
dataset_lock_path = {_quoted(config['dataset_lock_path'])}
rollouts_per_problem = 8

[prompt]
path = {_quoted(prompt['path'])}
version = {_quoted(prompt['version'])}
prefix = {_quoted(prompt['prefix'])}

[model]
{_model_toml(model)}

[sampling]
{_sampling_toml(sampling)}
"""
    return text.encode("utf-8")


def _synthesis_toml(plan: dict[str, Any], anchor: ModelSpec) -> bytes:
    config = plan["config"]
    prompt = config["synthesis"]["prompt"]
    sampling = SamplingSpec.from_dict(config["synthesis"]["sampling"])
    text = f"""schema_version = 2
kind = "synthesis"
protocol_version = {_quoted(MATH500_PROTOCOL_VERSION)}
run_name = {_quoted(config['run_name'] + '-trained-final-synthesis')}
required_rollouts = 8
anchor_relation = "frozen_initial_for_trained_raw"

[prompt]
path = {_quoted(prompt['path'])}
version = {_quoted(prompt['version'])}
prefix = {_quoted(prompt['prefix'])}

[anchor]
{_model_toml(anchor)}

[sampling]
{_sampling_toml(sampling)}
"""
    return text.encode("utf-8")


def _handoff_artifacts(
    run_dir: Path,
    output_dir: Path,
    served_model: str,
    plan: dict[str, Any],
    checkpoint: dict[str, Any],
    completion: dict[str, Any],
) -> tuple[bytes, bytes, dict[str, Any]]:
    bound_checkpoint_roots = tuple(
        Path(checkpoint[name]["path"]).resolve()
        for name in ("verl_actor_checkpoint", "export")
    )
    if any(
        output_dir == root or root in output_dir.parents
        for root in bound_checkpoint_roots
    ):
        raise TrainingError(
            "Evaluation handoff output must be outside registered checkpoint trees"
        )
    if completion.get("completed_step") != plan["config"]["grpo"]["max_steps"]:
        raise TrainingError("Only the completed fixed final checkpoint can be evaluated")
    raw_model = _trained_model(plan, checkpoint, served_model)
    synthesis_anchor = _initial_anchor_model(plan)
    raw_model.assert_resolved()
    synthesis_anchor.assert_resolved()
    if raw_model == synthesis_anchor:
        raise TrainingError("Trained raw policy must differ from its initial anchor")
    raw_payload = _raw_toml(plan, raw_model)
    synthesis_payload = _synthesis_toml(plan, synthesis_anchor)
    raw_path = output_dir / RAW_CONFIG_NAME
    synthesis_path = output_dir / SYNTHESIS_CONFIG_NAME
    handoff = {
        "schema_version": 2,
        "training_plan_fingerprint": plan["plan_fingerprint"],
        "completion": artifact_reference(run_dir / "completion.json"),
        "checkpoint_manifest": artifact_reference(
            run_dir / "checkpoint_manifest.json"
        ),
        "checkpoint_tree_sha256": checkpoint["export"]["tree_sha256"],
        "checkpoint_path": checkpoint["export"]["path"],
        "raw_policy_model": raw_model.to_dict(),
        "synthesis_anchor_model": synthesis_anchor.to_dict(),
        "synthesis_anchor_relation": "frozen_initial_for_trained_raw",
        "raw_config": {
            "path": str(raw_path),
            "sha256": sha256_bytes(raw_payload),
            "bytes": len(raw_payload),
        },
        "synthesis_config": {
            "path": str(synthesis_path),
            "sha256": sha256_bytes(synthesis_payload),
            "bytes": len(synthesis_payload),
        },
        "selection": "fixed_final_step_without_labels",
        "labels_loaded": False,
        "reportable": completion["reportable"],
        "non_reportable_reasons": completion["non_reportable_reasons"],
    }
    handoff["handoff_fingerprint"] = sha256_bytes(canonical_json_bytes(handoff))
    return raw_payload, synthesis_payload, handoff


def write_eval_handoff(
    run_dir: Path,
    output_dir: Path,
    served_model: str,
    *,
    force: bool = False,
    _lease: LaunchLease | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    output_dir = output_dir.resolve()
    if _lease is None:
        with exclusive_launch(run_dir) as lease:
            return write_eval_handoff(
                run_dir,
                output_dir,
                served_model,
                force=force,
                _lease=lease,
            )
    _lease.assert_for(run_dir)
    plan, _ = load_training_plan(run_dir)
    checkpoint, completion = load_registered_checkpoint(run_dir)
    raw_payload, synthesis_payload, handoff = _handoff_artifacts(
        run_dir,
        output_dir,
        served_model,
        plan,
        checkpoint,
        completion,
    )
    raw_path = output_dir / RAW_CONFIG_NAME
    synthesis_path = output_dir / SYNTHESIS_CONFIG_NAME
    publish_bytes(raw_path, raw_payload, force=force)
    publish_bytes(synthesis_path, synthesis_payload, force=force)
    publish_json(output_dir / EVAL_HANDOFF_NAME, handoff, force=force)
    return handoff


def verify_eval_handoff(
    run_dir: Path, output_dir: Path, served_model: str
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    paths = tuple(
        output_dir / name
        for name in (RAW_CONFIG_NAME, SYNTHESIS_CONFIG_NAME, EVAL_HANDOFF_NAME)
    )
    if not all(path.is_file() and not path.is_symlink() for path in paths):
        raise TrainingError("Trained-evaluation handoff is incomplete or unsafe")
    return write_eval_handoff(run_dir, output_dir, served_model)


def verify_eval_handoff_artifacts(
    run_dir: Path, output_dir: Path, served_model: str
) -> dict[str, Any]:
    """Verify handoff, receipt, actor, and export artifacts without opening the base model."""

    run_dir = run_dir.resolve()
    output_dir = output_dir.resolve()
    raw_path = output_dir / RAW_CONFIG_NAME
    synthesis_path = output_dir / SYNTHESIS_CONFIG_NAME
    handoff_path = output_dir / EVAL_HANDOFF_NAME
    paths = (raw_path, synthesis_path, handoff_path)
    if not all(path.is_file() and not path.is_symlink() for path in paths):
        raise TrainingError("Trained-evaluation handoff is incomplete or unsafe")
    plan, _ = load_training_plan(run_dir)
    checkpoint, completion = load_registered_checkpoint_artifacts(run_dir)
    raw_payload, synthesis_payload, expected_handoff = _handoff_artifacts(
        run_dir,
        output_dir,
        served_model,
        plan,
        checkpoint,
        completion,
    )
    if raw_path.read_bytes() != raw_payload or synthesis_path.read_bytes() != synthesis_payload:
        raise TrainingError("Trained-evaluation config changed")
    try:
        handoff = read_json(handoff_path)
    except EvaluationError as exc:
        raise TrainingError("Cannot read trained-evaluation handoff") from exc
    if handoff != expected_handoff:
        raise TrainingError("Trained-evaluation handoff changed")
    return handoff
