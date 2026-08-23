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
    sha256_bytes,
)
from compute_as_a_teacher.evaluation.config import MATH500_PROTOCOL_VERSION
from compute_as_a_teacher.evaluation.schemas import ModelSpec, SamplingSpec

from .checkpoints import load_registered_checkpoint
from .errors import TrainingError
from .planning import load_training_plan


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


def _trained_model(plan: dict[str, Any], checkpoint: dict[str, Any], served_model: str) -> ModelSpec:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", served_model):
        raise TrainingError("served_model must use only letters, digits, dot, underscore, and hyphen")
    base = plan["config"]["policy"]
    return ModelSpec(
        provider="openai-compatible",
        model_id=served_model,
        revision=checkpoint["export"]["tree_sha256"],
        tokenizer_id=base["tokenizer_id"],
        tokenizer_revision=base["tokenizer_revision"],
        chat_template_sha256=base["chat_template_sha256"],
        adapter_version="openai-compatible-chat-v1",
        dtype=base["dtype"],
        quantization=base["quantization"],
        seed_support="best_effort",
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


def _synthesis_toml(plan: dict[str, Any], model: ModelSpec) -> bytes:
    config = plan["config"]
    prompt = config["synthesis"]["prompt"]
    sampling = SamplingSpec.from_dict(config["synthesis"]["sampling"])
    text = f"""schema_version = 1
kind = "synthesis"
protocol_version = {_quoted(MATH500_PROTOCOL_VERSION)}
run_name = {_quoted(config['run_name'] + '-trained-final-synthesis')}
required_rollouts = 8
require_same_model_as_raw = true

[prompt]
path = {_quoted(prompt['path'])}
version = {_quoted(prompt['version'])}
prefix = {_quoted(prompt['prefix'])}

[anchor]
{_model_toml(model)}

[sampling]
{_sampling_toml(sampling)}
"""
    return text.encode("utf-8")


def write_eval_handoff(
    run_dir: Path,
    output_dir: Path,
    served_model: str,
    *,
    force: bool = False,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    output_dir = output_dir.resolve()
    plan, _ = load_training_plan(run_dir)
    checkpoint, completion = load_registered_checkpoint(run_dir)
    if completion.get("completed_step") != plan["config"]["grpo"]["max_steps"]:
        raise TrainingError("Only the completed fixed final checkpoint can be evaluated")
    model = _trained_model(plan, checkpoint, served_model)
    model.assert_resolved()
    raw_payload = _raw_toml(plan, model)
    synthesis_payload = _synthesis_toml(plan, model)
    raw_path = output_dir / RAW_CONFIG_NAME
    synthesis_path = output_dir / SYNTHESIS_CONFIG_NAME
    publish_bytes(raw_path, raw_payload, force=force)
    publish_bytes(synthesis_path, synthesis_payload, force=force)
    handoff = {
        "schema_version": 1,
        "training_plan_fingerprint": plan["plan_fingerprint"],
        "completion": artifact_reference(run_dir / "completion.json"),
        "checkpoint_manifest": artifact_reference(run_dir / "checkpoint_manifest.json"),
        "checkpoint_tree_sha256": checkpoint["export"]["tree_sha256"],
        "checkpoint_path": checkpoint["export"]["path"],
        "evaluation_model": model.to_dict(),
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
    publish_json(output_dir / EVAL_HANDOFF_NAME, handoff, force=force)
    return handoff
