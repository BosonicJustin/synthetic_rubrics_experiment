"""Label-free training dataset and immutable launch-plan construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from compute_as_a_teacher.data.math500 import QuestionRecord
from compute_as_a_teacher.evaluation.artifacts import (
    artifact_reference,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    file_digest,
    publish_bytes,
    read_json,
    sha256_bytes,
)
from compute_as_a_teacher.evaluation.prompts import (
    prompt_contract_sha256,
    render_raw_prompt,
    validate_prompt_template,
)

from .config import TrainingConfig
from .errors import TrainingError
from .verl_adapter import VerlCommand, build_verl_command, command_from_dict


MANIFEST_NAME = "manifest.json"
TRAINING_DATA_NAME = "math500_train.jsonl"
VERL_COMMAND_NAME = "verl_command.json"


def _source_tree_reference(root: Path) -> dict[str, Any]:
    root = root.resolve()
    files = []
    for path in sorted(root.rglob("*.py"), key=lambda item: item.relative_to(root).as_posix()):
        digest, size = file_digest(path)
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": digest,
                "bytes": size,
            }
        )
    if not files:
        raise TrainingError(f"Python source tree is empty: {root}")
    return {
        "root": str(root),
        "tree_sha256": sha256_bytes(canonical_json_bytes(files)),
        "files": files,
    }


def build_training_rows(
    questions: Sequence[QuestionRecord],
    config: TrainingConfig,
    raw_prompt_template: str,
) -> list[dict[str, Any]]:
    validate_prompt_template(raw_prompt_template, config.rollouts.prompt)
    if len(questions) != 500:
        raise TrainingError(f"The locked MATH-500 training view must contain 500 rows, found {len(questions)}")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for question in questions:
        if question.id in seen:
            raise TrainingError(f"Duplicate training question ID: {question.id}")
        seen.add(question.id)
        rows.append(
            {
                "data_source": "cat_math500_reference_free_v1",
                "prompt": [
                    {
                        "role": "user",
                        "content": render_raw_prompt(
                            raw_prompt_template,
                            config.rollouts.prompt,
                            question.problem,
                        ),
                    }
                ],
                "reward_model": {"style": "reference_free", "ground_truth": None},
                "extra_info": {"question_id": question.id},
            }
        )
    return rows


def dry_run_summary(config: TrainingConfig, problem_count: int) -> dict[str, Any]:
    return {
        "mode": "dry_run",
        "would_write": False,
        "model_loaded": False,
        "labels_loaded": False,
        "framework_imported": False,
        "problems": problem_count,
        "prompt_batch_size": config.grpo.global_batch_size,
        "rollouts_per_prompt": config.rollouts.group_size,
        "trajectories_per_update": config.grpo.global_batch_size * config.rollouts.group_size,
        "anchor_calls_per_update": config.grpo.global_batch_size,
        "max_steps": config.grpo.max_steps,
        "planned_policy_trajectories": (
            config.grpo.global_batch_size
            * config.rollouts.group_size
            * config.grpo.max_steps
        ),
        "planned_anchor_calls": config.grpo.global_batch_size * config.grpo.max_steps,
        "config_runnable": not config.unresolved_reasons(),
        "unresolved_reasons": list(config.unresolved_reasons()),
    }


def _preflight(run_dir: Path, payloads: Mapping[str, bytes], *, force: bool) -> None:
    mismatches = []
    for name, payload in payloads.items():
        path = run_dir / name
        if path.exists() and (not path.is_file() or path.read_bytes() != payload):
            mismatches.append(path)
    descendants = [
        path
        for path in (
            run_dir / "checkpoints",
            run_dir / "rollout_logs",
            run_dir / "exports",
            run_dir / "completion.json",
            run_dir / "checkpoint_manifest.json",
            run_dir / "eval_handoff.json",
        )
        if path.exists()
    ]
    if mismatches and descendants:
        raise TrainingError("Refusing to re-plan a training run with descendant artifacts")
    if mismatches and not force:
        raise TrainingError(f"Refusing to replace mismatched training plan: {mismatches}")


def write_training_plan(
    run_dir: Path,
    questions: Sequence[QuestionRecord],
    config: TrainingConfig,
    raw_prompt_template: str,
    synthesis_prompt_template: str,
    *,
    questions_path: Path,
    dataset_lock_path: Path,
    repository_root: Path,
    force: bool = False,
) -> dict[str, Any]:
    config.assert_runnable()
    validate_prompt_template(synthesis_prompt_template, config.synthesis.prompt)
    rows = build_training_rows(questions, config, raw_prompt_template)
    data_payload = canonical_jsonl_bytes(rows)
    command = build_verl_command(
        config,
        repository_root=repository_root,
        run_dir=run_dir,
        training_data_path=run_dir / TRAINING_DATA_NAME,
    )
    command_payload = canonical_json_bytes(command.to_dict())
    question_reference = artifact_reference(questions_path, rows=len(questions))
    lock_reference = artifact_reference(dataset_lock_path)
    source_reference = _source_tree_reference(
        repository_root / "src/compute_as_a_teacher"
    )
    manifest_without_fingerprint = {
        "schema_version": 1,
        "kind": config.kind,
        "protocol_version": config.protocol_version,
        "run_name": config.run_name,
        "config": config.to_dict(),
        "config_fingerprint": config.fingerprint,
        "inputs": {
            "questions": question_reference,
            "dataset_lock": lock_reference,
            "raw_prompt": artifact_reference(
                repository_root / config.rollouts.prompt.path
            ),
            "synthesis_prompt": artifact_reference(
                repository_root / config.synthesis.prompt.path
            ),
            "python_source": source_reference,
            "raw_prompt_contract_sha256": prompt_contract_sha256(
                raw_prompt_template, config.rollouts.prompt
            ),
            "synthesis_prompt_contract_sha256": prompt_contract_sha256(
                synthesis_prompt_template, config.synthesis.prompt
            ),
        },
        "artifacts": {
            "training_data": {
                "path": TRAINING_DATA_NAME,
                "sha256": sha256_bytes(data_payload),
                "bytes": len(data_payload),
                "rows": len(rows),
            },
            "verl_command": {
                "path": VERL_COMMAND_NAME,
                "sha256": sha256_bytes(command_payload),
                "bytes": len(command_payload),
                "fingerprint": command.fingerprint,
            },
        },
        "counts": {
            "problems": len(rows),
            "prompt_batch_size": config.grpo.global_batch_size,
            "rollouts_per_prompt": config.rollouts.group_size,
            "trajectories_per_update": config.grpo.global_batch_size * config.rollouts.group_size,
            "max_steps": config.grpo.max_steps,
        },
        "label_firewall": {
            "labels_loaded": False,
            "reference_answers_loaded": False,
            "reference_solutions_loaded": False,
            "reward_uses_synthesized_answer_only": True,
            "checkpoint_selection_uses_labels": False,
        },
        "paper_contract": {
            "online_current_policy_rollouts": True,
            "frozen_initial_policy_anchor": True,
            "anchor_receives_question": False,
            "anchor_receives_ordered_rollout_texts": True,
            "reward": "boxed_answer_agreement_with_synthesis",
            "kl_reference": "initial_policy",
            "fixed_final_checkpoint": config.grpo.max_steps,
        },
    }
    manifest_without_fingerprint["plan_fingerprint"] = sha256_bytes(
        canonical_json_bytes(manifest_without_fingerprint)
    )
    manifest_payload = canonical_json_bytes(manifest_without_fingerprint)
    payloads = {
        TRAINING_DATA_NAME: data_payload,
        VERL_COMMAND_NAME: command_payload,
        MANIFEST_NAME: manifest_payload,
    }
    _preflight(run_dir, payloads, force=force)
    for name, payload in payloads.items():
        publish_bytes(run_dir / name, payload, force=force)
    return manifest_without_fingerprint


def load_training_plan(run_dir: Path) -> tuple[dict[str, Any], VerlCommand]:
    manifest = read_json(run_dir / MANIFEST_NAME)
    if manifest.get("schema_version") != 1 or manifest.get("kind") != "cat_grpo":
        raise TrainingError("Invalid training manifest")
    stored_fingerprint = manifest.get("plan_fingerprint")
    unsigned = dict(manifest)
    unsigned.pop("plan_fingerprint", None)
    if stored_fingerprint != sha256_bytes(canonical_json_bytes(unsigned)):
        raise TrainingError("Training manifest fingerprint mismatch")
    inputs = manifest.get("inputs", {})
    for name in ("raw_prompt", "synthesis_prompt"):
        reference = inputs.get(name)
        if not isinstance(reference, dict) or not isinstance(reference.get("path"), str):
            raise TrainingError(f"Training plan is missing its {name} reference")
        if artifact_reference(Path(reference["path"])) != reference:
            raise TrainingError(f"Planned {name} changed")
    source = inputs.get("python_source")
    if not isinstance(source, dict) or not isinstance(source.get("root"), str):
        raise TrainingError("Training plan is missing its Python source reference")
    if _source_tree_reference(Path(source["root"])) != source:
        raise TrainingError("Planned Python source changed")
    for key, name in (("training_data", TRAINING_DATA_NAME), ("verl_command", VERL_COMMAND_NAME)):
        reference = manifest.get("artifacts", {}).get(key)
        path = run_dir / name
        if not isinstance(reference, dict) or not path.is_file():
            raise TrainingError(f"Missing planned artifact: {name}")
        payload = path.read_bytes()
        if reference.get("sha256") != sha256_bytes(payload) or reference.get("bytes") != len(payload):
            raise TrainingError(f"Planned artifact changed: {name}")
    command = command_from_dict(read_json(run_dir / VERL_COMMAND_NAME))
    if command.fingerprint != manifest["artifacts"]["verl_command"]["fingerprint"]:
        raise TrainingError("verl command fingerprint mismatch")
    return manifest, command
