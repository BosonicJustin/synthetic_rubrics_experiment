"""Two-phase, content-addressed registry for the MATH-500 CaT experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from compute_as_a_teacher.evaluation.artifacts import (
    artifact_reference,
    canonical_json_bytes,
    file_digest,
    publish_json,
    read_json,
    sha256_bytes,
)
from compute_as_a_teacher.evaluation.config import (
    MATH500_PROTOCOL_VERSION,
    ScoringConfig,
    SynthesisEvalConfig,
    load_scoring_config,
    load_synthesis_config,
)
from compute_as_a_teacher.evaluation.errors import EvaluationError
from compute_as_a_teacher.evaluation.planning import load_plan
from compute_as_a_teacher.evaluation.schemas import ModelSpec

from .checkpoints import load_registered_checkpoint
from .config import TRAINING_PROTOCOL_VERSION
from .errors import TrainingError
from .planning import load_training_plan
from .verl_adapter import LaunchLease, exclusive_launch


PREREGISTRATION_KIND = "cat_math500_experiment_preregistration"
REGISTRY_KIND = "cat_math500_experiment_registry"
SCHEMA_VERSION = 2

_EVAL_RESULT_NAMES = (
    "results",
    "generations.jsonl",
    "execution.json",
    "responses.jsonl",
    "scores.jsonl",
    "paired_scores.jsonl",
    "summary.json",
    "scoring_manifest.json",
)
_TRAINING_RESULT_NAMES = (
    "checkpoints",
    "rollout_logs",
    "logs",
    "checkpoint_manifest.json",
    "completion.json",
    "eval_handoff.json",
    "exports",
)
_SNAPSHOT_FIELDS = (
    "revision",
    "tokenizer_id",
    "tokenizer_revision",
    "chat_template_sha256",
    "dtype",
    "quantization",
)
_ENDPOINT_ATTESTATION_LIMITATION = (
    "endpoint_model_aliases_do_not_content_attest_weights_or_hardware"
)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise TrainingError(f"{name} must be an object")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TrainingError(message)


def _content_identity(value: Any, name: str) -> dict[str, Any]:
    reference = _mapping(value, name)
    sha256 = reference.get("sha256")
    byte_count = reference.get("bytes")
    rows = reference.get("rows")
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
        or isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count <= 0
        or (
            rows is not None
            and (isinstance(rows, bool) or not isinstance(rows, int) or rows < 0)
        )
    ):
        raise TrainingError(f"{name} has an invalid content reference")
    return {"sha256": sha256, "bytes": byte_count, "rows": rows}


def _model(value: Any, name: str) -> ModelSpec:
    try:
        return ModelSpec.from_dict(_mapping(value, name))
    except EvaluationError as exc:
        raise TrainingError(f"Invalid {name}: {exc}") from exc


def _snapshot(model: ModelSpec) -> dict[str, str]:
    value = model.to_dict()
    return {field: value[field] for field in _SNAPSHOT_FIELDS}


def _semantic_fingerprint(value: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(dict(value)))


def _fingerprint(value: Mapping[str, Any], field: str) -> str:
    unsigned = dict(value)
    stored = unsigned.pop(field, None)
    expected = sha256_bytes(canonical_json_bytes(unsigned))
    if stored != expected:
        raise TrainingError(f"{field} mismatch")
    return expected


def _artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise TrainingError(f"Required artifact is missing: {path}")
    return artifact_reference(path.resolve())


def _verify_artifact(value: Any, name: str) -> None:
    reference = _mapping(value, name)
    if set(reference) != {"path", "sha256", "bytes"}:
        raise TrainingError(f"{name} has an invalid artifact schema")
    path_value = reference.get("path")
    if not isinstance(path_value, str):
        raise TrainingError(f"{name}.path must be text")
    path = Path(path_value)
    if not path.is_file():
        raise TrainingError(f"{name} is missing: {path}")
    digest, byte_count = file_digest(path)
    if digest != reference.get("sha256") or byte_count != reference.get("bytes"):
        raise TrainingError(f"{name} changed after registration")


def _ensure_preresult(run_dir: Path, names: Sequence[str], name: str) -> None:
    descendants = [run_dir / item for item in names if (run_dir / item).exists()]
    if descendants:
        raise TrainingError(
            f"{name} must be preregistered before result artifacts exist: {descendants}"
        )


def _reject_output_collision(
    output_path: Path,
    protected: Sequence[Path],
    *,
    bound_roots: Sequence[Path] = (),
) -> None:
    output = output_path.resolve()
    roots = tuple(path.resolve() for path in bound_roots)
    if any(output == root or root in output.parents for root in roots):
        raise TrainingError(
            f"Registry output must be outside every bound run directory: {output}"
        )
    collisions = {path.resolve() for path in protected}
    if output in collisions:
        raise TrainingError(f"Registry output would replace a source artifact: {output}")


def _require_disjoint_run_directories(run_dirs: Mapping[str, Path]) -> None:
    resolved = {name: path.resolve() for name, path in run_dirs.items()}
    for name, path in resolved.items():
        for other_name, other in resolved.items():
            if name >= other_name:
                continue
            if path == other or path in other.parents or other in path.parents:
                raise TrainingError(
                    "Experiment stage run directories must be disjoint: "
                    f"{name}={path}, {other_name}={other}"
                )


def _eval_stage(run_dir: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "run_dir": str(run_dir.resolve()),
        "manifest": _artifact(run_dir / "manifest.json"),
        "plan_fingerprint": manifest["plan_fingerprint"],
    }


def _training_stage(run_dir: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "run_dir": str(run_dir.resolve()),
        "manifest": _artifact(run_dir / "manifest.json"),
        "plan_fingerprint": manifest["plan_fingerprint"],
        "config_fingerprint": manifest["config_fingerprint"],
    }


def _require_eval_firewall(manifest: Mapping[str, Any], name: str) -> None:
    firewall = _mapping(manifest.get("label_firewall"), f"{name} label firewall")
    _require(firewall.get("labels_loaded") is False, f"{name} loaded labels")
    _require(
        firewall.get("reference_answers_loaded") is False,
        f"{name} loaded reference answers",
    )
    _require(
        firewall.get("reference_solutions_loaded") is False,
        f"{name} loaded reference solutions",
    )
    _require(
        firewall.get("locked_questions_verified") is True,
        f"{name} does not use lock-verified questions",
    )
    if manifest.get("kind") == "synthesis":
        _require(
            firewall.get("question_field_supplied_to_synthesis") is False,
            f"{name} supplied the question to synthesis",
        )


def _require_training_firewall(manifest: Mapping[str, Any]) -> None:
    firewall = _mapping(manifest.get("label_firewall"), "training label firewall")
    expected = {
        "labels_loaded": False,
        "reference_answers_loaded": False,
        "reference_solutions_loaded": False,
        "reward_uses_synthesized_answer_only": True,
        "checkpoint_selection_uses_labels": False,
    }
    for key, value in expected.items():
        _require(firewall.get(key) is value, f"Training label firewall violates {key}")


def _require_prompt_lineage(
    manifest: Mapping[str, Any],
    training_manifest: Mapping[str, Any],
    role: str,
) -> None:
    prompt = _mapping(manifest.get("prompt"), f"{role} evaluation prompt")
    inputs = _mapping(training_manifest.get("inputs"), "training inputs")
    prompt_artifact = _mapping(inputs.get(f"{role}_prompt"), f"training {role} prompt")
    _require(
        prompt.get("template_sha256") == prompt_artifact.get("sha256"),
        f"{role} prompt bytes differ between evaluation and training",
    )
    _require(
        prompt.get("contract_sha256") == inputs.get(f"{role}_prompt_contract_sha256"),
        f"{role} prompt contract differs between evaluation and training",
    )


def _validate_preregistration_contract(
    raw_manifest: Mapping[str, Any],
    synthesis_config: SynthesisEvalConfig,
    scoring_config: ScoringConfig,
    training_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    _require(raw_manifest.get("kind") == "raw", "Initial evaluation must be raw")
    _require(
        raw_manifest.get("protocol_version") == MATH500_PROTOCOL_VERSION,
        "Initial raw plan uses the wrong evaluation protocol",
    )
    _require(
        training_manifest.get("protocol_version") == TRAINING_PROTOCOL_VERSION,
        "Canonical training plan uses the wrong training protocol",
    )
    _require_eval_firewall(raw_manifest, "Initial raw plan")
    _require_training_firewall(training_manifest)

    raw_config = _mapping(raw_manifest.get("config"), "initial raw config")
    training_config = _mapping(training_manifest.get("config"), "training config")
    rollouts = _mapping(training_config.get("rollouts"), "training rollouts")
    synthesis = _mapping(training_config.get("synthesis"), "training synthesis")
    anchor = _mapping(training_config.get("anchor"), "training anchor")
    grpo = _mapping(training_config.get("grpo"), "training GRPO")
    checkpointing = _mapping(
        training_config.get("checkpointing"), "training checkpointing"
    )
    reward = _mapping(training_config.get("reward"), "training reward")

    raw_model = _model(raw_manifest.get("model"), "initial raw model")
    policy_model = _model(training_config.get("policy"), "training policy")
    anchor_model = _model(anchor.get("model"), "training anchor model")
    _require(policy_model == anchor_model, "Training anchor is not the initial policy")
    _require(
        anchor.get("source") == "initial_policy" and anchor.get("frozen") is True,
        "Training anchor is not frozen initial policy pi0",
    )
    _require(
        _snapshot(raw_model) == _snapshot(policy_model),
        "Initial raw model does not match the training pi0 snapshot",
    )

    synthesis_value = synthesis_config.to_dict()
    scoring_value = scoring_config.to_dict()
    _require(
        synthesis_config.protocol_version == MATH500_PROTOCOL_VERSION,
        "Initial synthesis config uses the wrong evaluation protocol",
    )
    _require(
        synthesis_config.anchor_relation == "same_as_raw",
        "Initial synthesis must use the raw pi0 as its anchor",
    )
    _require(
        synthesis_config.anchor == raw_model,
        "Initial synthesis anchor does not exactly equal the initial raw policy",
    )
    _require(
        synthesis_config.required_rollouts == 8,
        "Initial synthesis must consume eight rollouts",
    )
    _require(
        scoring_config.protocol_version == MATH500_PROTOCOL_VERSION,
        "Scoring config uses the wrong evaluation protocol",
    )

    _require(
        raw_config.get("questions_path") == training_config.get("questions_path")
        and raw_config.get("dataset_lock_path")
        == training_config.get("dataset_lock_path"),
        "Evaluation and training dataset paths differ",
    )
    _require(
        scoring_config.dataset_lock_path == raw_config.get("dataset_lock_path"),
        "Scoring and generation dataset locks differ",
    )
    raw_inputs = _mapping(raw_manifest.get("inputs"), "initial raw inputs")
    training_inputs = _mapping(training_manifest.get("inputs"), "training inputs")
    _require(
        _content_identity(raw_inputs.get("questions"), "initial questions")
        == _content_identity(training_inputs.get("questions"), "training questions"),
        "Evaluation and training question artifacts differ",
    )
    _require(
        _content_identity(raw_inputs.get("dataset_lock"), "initial dataset lock")
        == _content_identity(training_inputs.get("dataset_lock"), "training dataset lock"),
        "Evaluation and training dataset locks differ",
    )

    raw_prompt = _mapping(raw_config.get("prompt"), "initial raw prompt")
    rollout_prompt = _mapping(rollouts.get("prompt"), "training rollout prompt")
    synthesis_prompt = _mapping(synthesis.get("prompt"), "training synthesis prompt")
    _require(raw_prompt == rollout_prompt, "Raw prompt spec differs from training")
    _require(
        synthesis_value["prompt"] == synthesis_prompt,
        "Synthesis prompt spec differs from training",
    )
    _require_prompt_lineage(raw_manifest, training_manifest, "raw")

    raw_sampling = _mapping(raw_config.get("sampling"), "initial raw sampling")
    rollout_sampling = _mapping(rollouts.get("sampling"), "training rollout sampling")
    synthesis_sampling = _mapping(
        synthesis.get("sampling"), "training synthesis sampling"
    )
    _require(raw_sampling == rollout_sampling, "Raw sampling differs from training")
    _require(
        synthesis_value["sampling"] == synthesis_sampling,
        "Synthesis sampling differs from training",
    )

    _require(
        raw_config.get("rollouts_per_problem") == 8
        and rollouts.get("group_size") == 8
        and synthesis.get("required_rollouts") == 8,
        "Every CaT stage must use eight rollouts",
    )
    _require(
        raw_manifest.get("counts", {}).get("problems") == 500
        and raw_manifest.get("counts", {}).get("requests") == 4000
        and training_manifest.get("counts", {}).get("problems") == 500,
        "Canonical plans must cover all 500 MATH-500 problems",
    )
    _require(
        reward.get("labels_allowed") is False,
        "Canonical reward must not allow labels",
    )
    _require(grpo.get("max_steps") == 1000, "Canonical training must use 1000 steps")
    _require(
        checkpointing.get("selected_checkpoint") == "fixed_final_step",
        "Checkpoint selection must be the fixed final step",
    )

    raw_prompt_manifest = _mapping(raw_manifest.get("prompt"), "initial raw prompt")
    return {
        "protocols": {
            "evaluation": MATH500_PROTOCOL_VERSION,
            "training": TRAINING_PROTOCOL_VERSION,
        },
        "dataset": {
            "questions_path": raw_config["questions_path"],
            "dataset_lock_path": raw_config["dataset_lock_path"],
            "questions": _content_identity(raw_inputs["questions"], "initial questions"),
            "dataset_lock": _content_identity(
                raw_inputs["dataset_lock"], "initial dataset lock"
            ),
        },
        "pi0": {
            "snapshot": _snapshot(policy_model),
            "snapshot_fingerprint": _semantic_fingerprint(_snapshot(policy_model)),
            "initial_raw_model_fingerprint": raw_model.fingerprint,
        },
        "prompts": {
            "raw": {
                "spec": dict(raw_prompt),
                "template_sha256": raw_prompt_manifest["template_sha256"],
                "contract_sha256": raw_prompt_manifest["contract_sha256"],
            },
            "synthesis": {
                "spec": dict(synthesis_prompt),
                "template_sha256": training_inputs["synthesis_prompt"]["sha256"],
                "contract_sha256": training_inputs[
                    "synthesis_prompt_contract_sha256"
                ],
            },
        },
        "sampling": {
            "raw": dict(raw_sampling),
            "synthesis": dict(synthesis_sampling),
            "common_eval_seeds": {
                "raw": raw_sampling["base_seed"],
                "synthesis": synthesis_sampling["base_seed"],
            },
        },
        "fixed_checkpoint": {
            "step": 1000,
            "selection": "fixed_final_step_without_labels",
        },
        "scoring": {
            "semantic_fingerprint": _semantic_fingerprint(scoring_value),
            "raw_baseline_selection": scoring_config.raw_baseline_selection,
            "raw_baseline_seed": scoring_config.raw_baseline_seed,
            "primary_grader": scoring_config.primary_grader,
            "diagnostic_graders": list(scoring_config.diagnostic_graders),
        },
        "initial_synthesis_config_fingerprint": _semantic_fingerprint(
            synthesis_value
        ),
        "scoring_config_fingerprint": _semantic_fingerprint(scoring_value),
        "labels_loaded": False,
    }


def _assemble_preregistration(
    initial_raw_run_dir: Path,
    initial_synthesis_config_path: Path,
    scoring_config_path: Path,
    training_run_dir: Path,
    *,
    enforce_preresult: bool,
) -> dict[str, Any]:
    initial_raw_run_dir = initial_raw_run_dir.resolve()
    initial_synthesis_config_path = initial_synthesis_config_path.resolve()
    scoring_config_path = scoring_config_path.resolve()
    training_run_dir = training_run_dir.resolve()
    if enforce_preresult:
        _ensure_preresult(
            initial_raw_run_dir, _EVAL_RESULT_NAMES, "Initial raw plan"
        )
        _ensure_preresult(
            training_run_dir, _TRAINING_RESULT_NAMES, "Canonical training plan"
        )
    raw_manifest, _ = load_plan(initial_raw_run_dir, expected_kind="raw")
    synthesis_config = load_synthesis_config(initial_synthesis_config_path)
    scoring_config = load_scoring_config(scoring_config_path)
    training_manifest, _ = load_training_plan(training_run_dir)
    contract = _validate_preregistration_contract(
        raw_manifest,
        synthesis_config,
        scoring_config,
        training_manifest,
    )
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": PREREGISTRATION_KIND,
        "state": "preregistered",
        "stages": {
            "initial_raw": _eval_stage(initial_raw_run_dir, raw_manifest),
            "initial_synthesis_config": {
                "path": str(initial_synthesis_config_path),
                "artifact": _artifact(initial_synthesis_config_path),
                "semantic_fingerprint": contract[
                    "initial_synthesis_config_fingerprint"
                ],
            },
            "scoring_config": {
                "path": str(scoring_config_path),
                "artifact": _artifact(scoring_config_path),
                "semantic_fingerprint": contract[
                    "scoring_config_fingerprint"
                ],
            },
            "canonical_training": _training_stage(
                training_run_dir, training_manifest
            ),
        },
        "contract": contract,
        "results_included": False,
        "labels_loaded": False,
        "scientifically_attested": False,
        "attestation_limitations": [_ENDPOINT_ATTESTATION_LIMITATION],
    }
    value["preregistration_fingerprint"] = sha256_bytes(
        canonical_json_bytes(value)
    )
    return value


def build_experiment_preregistration(
    initial_raw_run_dir: Path,
    initial_synthesis_config_path: Path,
    scoring_config_path: Path,
    training_run_dir: Path,
) -> dict[str, Any]:
    """Build the frozen pre-result experiment contract without writing it."""

    try:
        return _assemble_preregistration(
            initial_raw_run_dir,
            initial_synthesis_config_path,
            scoring_config_path,
            training_run_dir,
            enforce_preresult=True,
        )
    except EvaluationError as exc:
        raise TrainingError(str(exc)) from exc


def write_experiment_preregistration(
    output_path: Path,
    initial_raw_run_dir: Path,
    initial_synthesis_config_path: Path,
    scoring_config_path: Path,
    training_run_dir: Path,
    *,
    force: bool = False,
    _lease: LaunchLease | None = None,
) -> dict[str, Any]:
    training_run_dir = training_run_dir.resolve()
    if _lease is None:
        with exclusive_launch(training_run_dir) as lease:
            return write_experiment_preregistration(
                output_path,
                initial_raw_run_dir,
                initial_synthesis_config_path,
                scoring_config_path,
                training_run_dir,
                force=force,
                _lease=lease,
            )
    _lease.assert_for(training_run_dir)
    _reject_output_collision(
        output_path,
        (
            initial_raw_run_dir / "manifest.json",
            initial_synthesis_config_path,
            scoring_config_path,
            training_run_dir / "manifest.json",
        ),
        bound_roots=(initial_raw_run_dir, training_run_dir),
    )
    value = build_experiment_preregistration(
        initial_raw_run_dir,
        initial_synthesis_config_path,
        scoring_config_path,
        training_run_dir,
    )
    try:
        publish_json(output_path.resolve(), value, force=force)
    except EvaluationError as exc:
        raise TrainingError(str(exc)) from exc
    return value


def load_experiment_preregistration(path: Path) -> dict[str, Any]:
    try:
        value = read_json(path.resolve())
    except EvaluationError as exc:
        raise TrainingError(str(exc)) from exc
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("kind") != PREREGISTRATION_KIND
        or value.get("state") != "preregistered"
        or value.get("results_included") is not False
        or value.get("labels_loaded") is not False
        or value.get("scientifically_attested") is not False
        or value.get("attestation_limitations")
        != [_ENDPOINT_ATTESTATION_LIMITATION]
    ):
        raise TrainingError("Invalid experiment preregistration schema")
    _fingerprint(value, "preregistration_fingerprint")
    return value


def _same_registered_stage(
    registered: Any,
    current: Mapping[str, Any],
    name: str,
) -> None:
    expected = _mapping(registered, f"registered {name}")
    _verify_artifact(expected.get("manifest"), f"registered {name} manifest")
    _require(
        expected.get("run_dir") == current.get("run_dir")
        and expected.get("plan_fingerprint") == current.get("plan_fingerprint"),
        f"{name} no longer matches its preregistration",
    )
    if "config_fingerprint" in current:
        _require(
            expected.get("config_fingerprint") == current.get("config_fingerprint"),
            f"{name} config no longer matches its preregistration",
        )


def verify_preregistered_training_stage(
    preregistration_path: Path,
    training_run_dir: Path,
) -> dict[str, Any]:
    """Verify every preregistered source and the current canonical plan."""

    preregistration = load_experiment_preregistration(preregistration_path.resolve())
    stages = _mapping(preregistration.get("stages"), "preregistered stages")
    initial_raw = _mapping(stages.get("initial_raw"), "preregistered initial raw")
    initial_synthesis = _mapping(
        stages.get("initial_synthesis_config"),
        "preregistered initial synthesis config",
    )
    _verify_artifact(
        initial_raw.get("manifest"), "preregistered initial raw manifest"
    )
    _verify_artifact(
        initial_synthesis.get("artifact"),
        "preregistered initial synthesis config",
    )
    scoring = _mapping(
        stages.get("scoring_config"),
        "preregistered scoring config",
    )
    _verify_artifact(
        scoring.get("artifact"),
        "preregistered scoring config",
    )
    registered = _mapping(
        stages.get("canonical_training"), "preregistered canonical training"
    )
    _verify_artifact(
        registered.get("manifest"), "preregistered canonical training manifest"
    )
    manifest, _ = load_training_plan(training_run_dir.resolve())
    _same_registered_stage(
        registered,
        _training_stage(training_run_dir.resolve(), manifest),
        "canonical training plan",
    )
    return preregistration


def _validate_synthesis_plan(
    manifest: Mapping[str, Any],
    *,
    name: str,
    expected_raw_fingerprint: str,
    expected_prompt: Mapping[str, Any],
    expected_prompt_manifest: Mapping[str, Any],
    expected_sampling: Mapping[str, Any],
) -> Mapping[str, Any]:
    _require(manifest.get("kind") == "synthesis", f"{name} is not synthesis")
    _require(
        manifest.get("protocol_version") == MATH500_PROTOCOL_VERSION,
        f"{name} uses the wrong protocol",
    )
    _require_eval_firewall(manifest, name)
    config = _mapping(manifest.get("config"), f"{name} config")
    inputs = _mapping(manifest.get("inputs"), f"{name} inputs")
    _require(
        inputs.get("raw_plan_fingerprint") == expected_raw_fingerprint,
        f"{name} is linked to the wrong raw plan",
    )
    _require(config.get("prompt") == expected_prompt, f"{name} prompt changed")
    _require(
        config.get("sampling") == expected_sampling,
        f"{name} sampling changed",
    )
    _require(
        manifest.get("prompt") == expected_prompt_manifest,
        f"{name} prompt bytes changed",
    )
    _require(
        manifest.get("counts", {}).get("problems") == 500
        and manifest.get("counts", {}).get("requests") == 500,
        f"{name} must contain 500 synthesis requests",
    )
    return config


def _validate_final_contract(
    preregistration: Mapping[str, Any],
    initial_raw_manifest: Mapping[str, Any],
    initial_synthesis_manifest: Mapping[str, Any],
    training_manifest: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    completion: Mapping[str, Any],
    trained_raw_manifest: Mapping[str, Any],
    trained_synthesis_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    contract = _mapping(preregistration.get("contract"), "preregistered contract")
    stages = _mapping(preregistration.get("stages"), "preregistered stages")
    raw_stage = _mapping(stages.get("initial_raw"), "preregistered initial raw")
    training_stage = _mapping(
        stages.get("canonical_training"), "preregistered training"
    )
    synthesis_stage = _mapping(
        stages.get("initial_synthesis_config"),
        "preregistered synthesis config",
    )

    initial_synthesis_config = _validate_synthesis_plan(
        initial_synthesis_manifest,
        name="Initial synthesis plan",
        expected_raw_fingerprint=raw_stage["plan_fingerprint"],
        expected_prompt=contract["prompts"]["synthesis"]["spec"],
        expected_prompt_manifest={
            "version": contract["prompts"]["synthesis"]["spec"]["version"],
            "template_sha256": contract["prompts"]["synthesis"][
                "template_sha256"
            ],
            "contract_sha256": contract["prompts"]["synthesis"][
                "contract_sha256"
            ],
        },
        expected_sampling=contract["sampling"]["synthesis"],
    )
    _require(
        _semantic_fingerprint(initial_synthesis_config)
        == synthesis_stage["semantic_fingerprint"],
        "Initial synthesis plan differs from its preregistered config",
    )
    initial_raw_model = _model(
        initial_raw_manifest.get("model"), "initial raw model"
    )
    initial_anchor = _model(
        initial_synthesis_manifest.get("model"), "initial synthesis anchor"
    )
    _require(
        initial_synthesis_config.get("anchor_relation") == "same_as_raw"
        and initial_anchor == initial_raw_model,
        "Initial synthesis is not pi0 over pi0 rollouts",
    )

    training_config = _mapping(training_manifest.get("config"), "training config")
    max_steps = training_config["grpo"]["max_steps"]
    policy = _model(training_config.get("policy"), "training policy")
    export = _mapping(checkpoint.get("export"), "checkpoint export")
    export_digest = export.get("tree_sha256")
    _require(
        checkpoint.get("training_plan_fingerprint")
        == training_stage["plan_fingerprint"],
        "Checkpoint belongs to another training plan",
    )
    _require(
        checkpoint.get("selected_by") == "fixed_final_step"
        and checkpoint.get("step") == max_steps
        and max_steps == contract["fixed_checkpoint"]["step"],
        "Checkpoint is not the preregistered fixed final step",
    )
    _require(
        checkpoint.get("base_model_fingerprint")
        == sha256_bytes(canonical_json_bytes(policy.to_dict())),
        "Checkpoint base model does not match pi0",
    )
    _require(
        isinstance(export_digest, str)
        and completion.get("training_plan_fingerprint")
        == training_stage["plan_fingerprint"]
        and completion.get("completed_step") == max_steps
        and completion.get("selection") == "fixed_final_step_without_labels"
        and completion.get("labels_loaded") is False
        and completion.get("export_tree_sha256") == export_digest,
        "Completion did not use label-free fixed-step checkpoint selection",
    )

    _require(trained_raw_manifest.get("kind") == "raw", "Trained raw plan is not raw")
    _require(
        trained_raw_manifest.get("protocol_version") == MATH500_PROTOCOL_VERSION,
        "Trained raw plan uses the wrong protocol",
    )
    _require_eval_firewall(trained_raw_manifest, "Trained raw plan")
    trained_raw_config = _mapping(
        trained_raw_manifest.get("config"), "trained raw config"
    )
    initial_raw_config = _mapping(
        initial_raw_manifest.get("config"), "initial raw config"
    )
    _require(
        trained_raw_config.get("questions_path")
        == contract["dataset"]["questions_path"]
        and trained_raw_config.get("dataset_lock_path")
        == contract["dataset"]["dataset_lock_path"],
        "Trained raw plan uses a different dataset",
    )
    trained_inputs = _mapping(trained_raw_manifest.get("inputs"), "trained raw inputs")
    _require(
        _content_identity(trained_inputs.get("questions"), "trained questions")
        == contract["dataset"]["questions"]
        and _content_identity(trained_inputs.get("dataset_lock"), "trained lock")
        == contract["dataset"]["dataset_lock"],
        "Trained raw plan uses different dataset artifacts",
    )
    _require(
        trained_raw_config.get("prompt") == contract["prompts"]["raw"]["spec"]
        and trained_raw_manifest.get("prompt")
        == {
            "version": contract["prompts"]["raw"]["spec"]["version"],
            "template_sha256": contract["prompts"]["raw"]["template_sha256"],
            "contract_sha256": contract["prompts"]["raw"]["contract_sha256"],
        },
        "Trained raw prompt differs from initial raw",
    )
    _require(
        trained_raw_config.get("sampling") == contract["sampling"]["raw"]
        and initial_raw_config.get("sampling") == contract["sampling"]["raw"],
        "Initial and trained raw evaluation seeds or sampling differ",
    )
    _require(
        trained_raw_manifest.get("counts", {}).get("problems") == 500
        and trained_raw_manifest.get("counts", {}).get("requests") == 4000,
        "Trained raw plan must contain 4000 requests",
    )
    trained_policy = _model(trained_raw_manifest.get("model"), "trained policy")
    _require(
        trained_policy.revision == export_digest,
        "Trained raw model is not the registered checkpoint export",
    )
    for field in _SNAPSHOT_FIELDS[1:]:
        _require(
            getattr(trained_policy, field) == getattr(policy, field),
            f"Trained raw model changed base {field}",
        )
    _require(trained_policy != initial_raw_model, "Trained policy must differ from pi0")

    trained_synthesis_config = _validate_synthesis_plan(
        trained_synthesis_manifest,
        name="Trained synthesis plan",
        expected_raw_fingerprint=trained_raw_manifest["plan_fingerprint"],
        expected_prompt=contract["prompts"]["synthesis"]["spec"],
        expected_prompt_manifest={
            "version": contract["prompts"]["synthesis"]["spec"]["version"],
            "template_sha256": contract["prompts"]["synthesis"][
                "template_sha256"
            ],
            "contract_sha256": contract["prompts"]["synthesis"][
                "contract_sha256"
            ],
        },
        expected_sampling=contract["sampling"]["synthesis"],
    )
    trained_anchor = _model(
        trained_synthesis_manifest.get("model"), "trained synthesis anchor"
    )
    _require(
        trained_synthesis_config.get("anchor_relation")
        == "frozen_initial_for_trained_raw"
        and _snapshot(trained_anchor) == _snapshot(policy)
        and trained_anchor.model_id == training_config["runtime"]["anchor_model"]
        and trained_anchor != trained_policy,
        "Trained synthesis is not frozen pi0 over piT rollouts",
    )
    _require(
        initial_synthesis_config.get("sampling", {}).get("base_seed")
        == trained_synthesis_config.get("sampling", {}).get("base_seed")
        == contract["sampling"]["common_eval_seeds"]["synthesis"],
        "Initial and trained synthesis evaluation seeds differ",
    )

    return {
        "pi0_snapshot_fingerprint": contract["pi0"]["snapshot_fingerprint"],
        "piT_export_tree_sha256": export_digest,
        "initial_chain": {
            "raw_plan_fingerprint": initial_raw_manifest["plan_fingerprint"],
            "synthesis_plan_fingerprint": initial_synthesis_manifest[
                "plan_fingerprint"
            ],
            "declared_relation": "pi0_rollouts_synthesized_by_pi0",
        },
        "training_chain": {
            "plan_fingerprint": training_manifest["plan_fingerprint"],
            "checkpoint_step": max_steps,
            "checkpoint_selection": "fixed_final_step_without_labels",
        },
        "trained_chain": {
            "raw_plan_fingerprint": trained_raw_manifest["plan_fingerprint"],
            "synthesis_plan_fingerprint": trained_synthesis_manifest[
                "plan_fingerprint"
            ],
            "declared_relation": "piT_rollouts_synthesized_by_frozen_pi0",
        },
        "labels_loaded": False,
    }


def build_final_experiment_registry(
    preregistration_path: Path,
    scoring_config_path: Path,
    initial_raw_run_dir: Path,
    initial_synthesis_run_dir: Path,
    training_run_dir: Path,
    trained_raw_run_dir: Path,
    trained_synthesis_run_dir: Path,
) -> dict[str, Any]:
    """Validate and join every planned stage after fixed-step checkpointing."""

    try:
        preregistration_path = preregistration_path.resolve()
        preregistration = load_experiment_preregistration(preregistration_path)
        registered_stages = _mapping(
            preregistration.get("stages"), "preregistered stages"
        )
        for stage_name in ("initial_raw", "canonical_training"):
            registered_stage = _mapping(
                registered_stages.get(stage_name), f"preregistered {stage_name}"
            )
            _verify_artifact(
                registered_stage.get("manifest"),
                f"preregistered {stage_name} manifest",
            )
        synthesis_registration = _mapping(
            registered_stages.get("initial_synthesis_config"),
            "preregistered synthesis config",
        )
        _verify_artifact(
            synthesis_registration.get("artifact"),
            "preregistered synthesis config",
        )
        scoring_registration = _mapping(
            registered_stages.get("scoring_config"),
            "preregistered scoring config",
        )
        _verify_artifact(
            scoring_registration.get("artifact"),
            "preregistered scoring config",
        )
        scoring_config_path = scoring_config_path.resolve()
        current_preregistration = _assemble_preregistration(
            initial_raw_run_dir,
            Path(
                preregistration["stages"]["initial_synthesis_config"]["path"]
            ),
            scoring_config_path,
            training_run_dir,
            enforce_preresult=False,
        )
        _require(
            current_preregistration == preregistration,
            "Preregistered inputs or contract changed before finalization",
        )

        initial_raw_run_dir = initial_raw_run_dir.resolve()
        initial_synthesis_run_dir = initial_synthesis_run_dir.resolve()
        training_run_dir = training_run_dir.resolve()
        trained_raw_run_dir = trained_raw_run_dir.resolve()
        trained_synthesis_run_dir = trained_synthesis_run_dir.resolve()
        _require_disjoint_run_directories(
            {
                "initial_raw": initial_raw_run_dir,
                "initial_synthesis": initial_synthesis_run_dir,
                "canonical_training": training_run_dir,
                "trained_raw": trained_raw_run_dir,
                "trained_synthesis": trained_synthesis_run_dir,
            }
        )

        initial_raw_manifest, _ = load_plan(
            initial_raw_run_dir, expected_kind="raw"
        )
        initial_synthesis_manifest, _ = load_plan(
            initial_synthesis_run_dir, expected_kind="synthesis"
        )
        training_manifest, _ = load_training_plan(training_run_dir)
        checkpoint, completion = load_registered_checkpoint(training_run_dir)
        trained_raw_manifest, _ = load_plan(
            trained_raw_run_dir, expected_kind="raw"
        )
        trained_synthesis_manifest, _ = load_plan(
            trained_synthesis_run_dir, expected_kind="synthesis"
        )

        _same_registered_stage(
            preregistration["stages"]["initial_raw"],
            _eval_stage(initial_raw_run_dir, initial_raw_manifest),
            "initial raw plan",
        )
        _same_registered_stage(
            preregistration["stages"]["canonical_training"],
            _training_stage(training_run_dir, training_manifest),
            "canonical training plan",
        )
        lineage = _validate_final_contract(
            preregistration,
            initial_raw_manifest,
            initial_synthesis_manifest,
            training_manifest,
            checkpoint,
            completion,
            trained_raw_manifest,
            trained_synthesis_manifest,
        )

        value: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "kind": REGISTRY_KIND,
            "state": "finalized",
            "preregistration": {
                "artifact": _artifact(preregistration_path),
                "fingerprint": preregistration["preregistration_fingerprint"],
            },
            "contract": preregistration["contract"],
            "stages": {
                "initial_raw": _eval_stage(
                    initial_raw_run_dir, initial_raw_manifest
                ),
                "initial_synthesis": _eval_stage(
                    initial_synthesis_run_dir, initial_synthesis_manifest
                ),
                "scoring_config": {
                    "path": str(scoring_config_path),
                    "artifact": _artifact(scoring_config_path),
                    "semantic_fingerprint": preregistration["contract"][
                        "scoring_config_fingerprint"
                    ],
                },
                "canonical_training": _training_stage(
                    training_run_dir, training_manifest
                ),
                "fixed_final_checkpoint": {
                    "manifest": _artifact(
                        training_run_dir / "checkpoint_manifest.json"
                    ),
                    "completion": _artifact(training_run_dir / "completion.json"),
                    "step": checkpoint["step"],
                    "export_tree_sha256": checkpoint["export"]["tree_sha256"],
                },
                "trained_raw": _eval_stage(
                    trained_raw_run_dir, trained_raw_manifest
                ),
                "trained_synthesis": _eval_stage(
                    trained_synthesis_run_dir, trained_synthesis_manifest
                ),
            },
            "lineage": lineage,
            "results_included": False,
            "labels_loaded": False,
            "scientifically_attested": False,
            "attestation_limitations": [_ENDPOINT_ATTESTATION_LIMITATION],
        }
        value["registry_fingerprint"] = sha256_bytes(canonical_json_bytes(value))
        return value
    except EvaluationError as exc:
        raise TrainingError(str(exc)) from exc


def write_final_experiment_registry(
    output_path: Path,
    preregistration_path: Path,
    scoring_config_path: Path,
    initial_raw_run_dir: Path,
    initial_synthesis_run_dir: Path,
    training_run_dir: Path,
    trained_raw_run_dir: Path,
    trained_synthesis_run_dir: Path,
    *,
    force: bool = False,
    _lease: LaunchLease | None = None,
) -> dict[str, Any]:
    training_run_dir = training_run_dir.resolve()
    if _lease is None:
        with exclusive_launch(training_run_dir) as lease:
            return write_final_experiment_registry(
                output_path,
                preregistration_path,
                scoring_config_path,
                initial_raw_run_dir,
                initial_synthesis_run_dir,
                training_run_dir,
                trained_raw_run_dir,
                trained_synthesis_run_dir,
                force=force,
                _lease=lease,
            )
    _lease.assert_for(training_run_dir)
    _reject_output_collision(
        output_path,
        (
            preregistration_path,
            scoring_config_path,
            initial_raw_run_dir / "manifest.json",
            initial_synthesis_run_dir / "manifest.json",
            training_run_dir / "manifest.json",
            training_run_dir / "checkpoint_manifest.json",
            training_run_dir / "completion.json",
            trained_raw_run_dir / "manifest.json",
            trained_synthesis_run_dir / "manifest.json",
        ),
        bound_roots=(
            initial_raw_run_dir,
            initial_synthesis_run_dir,
            training_run_dir,
            trained_raw_run_dir,
            trained_synthesis_run_dir,
        ),
    )
    value = build_final_experiment_registry(
        preregistration_path,
        scoring_config_path,
        initial_raw_run_dir,
        initial_synthesis_run_dir,
        training_run_dir,
        trained_raw_run_dir,
        trained_synthesis_run_dir,
    )
    checkpoint, _ = load_registered_checkpoint(training_run_dir.resolve())
    export_root = Path(checkpoint["export"]["path"]).resolve()
    output = output_path.resolve()
    if output == export_root or export_root in output.parents:
        raise TrainingError(
            "Registry output must be outside the registered checkpoint export"
        )
    try:
        publish_json(output, value, force=force)
    except EvaluationError as exc:
        raise TrainingError(str(exc)) from exc
    return value


def load_final_experiment_registry(path: Path) -> dict[str, Any]:
    try:
        value = read_json(path.resolve())
    except EvaluationError as exc:
        raise TrainingError(str(exc)) from exc
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("kind") != REGISTRY_KIND
        or value.get("state") != "finalized"
        or value.get("results_included") is not False
        or value.get("labels_loaded") is not False
        or value.get("scientifically_attested") is not False
        or value.get("attestation_limitations")
        != [_ENDPOINT_ATTESTATION_LIMITATION]
    ):
        raise TrainingError("Invalid final experiment registry schema")
    _fingerprint(value, "registry_fingerprint")
    return value


def verify_final_experiment_registry(
    registry_path: Path,
    preregistration_path: Path,
    scoring_config_path: Path,
    initial_raw_run_dir: Path,
    initial_synthesis_run_dir: Path,
    training_run_dir: Path,
    trained_raw_run_dir: Path,
    trained_synthesis_run_dir: Path,
) -> dict[str, Any]:
    registered = load_final_experiment_registry(registry_path)
    current = build_final_experiment_registry(
        preregistration_path,
        scoring_config_path,
        initial_raw_run_dir,
        initial_synthesis_run_dir,
        training_run_dir,
        trained_raw_run_dir,
        trained_synthesis_run_dir,
    )
    _require(registered == current, "Final experiment registry no longer matches its stages")
    return registered


__all__ = [
    "PREREGISTRATION_KIND",
    "REGISTRY_KIND",
    "build_experiment_preregistration",
    "build_final_experiment_registry",
    "load_experiment_preregistration",
    "load_final_experiment_registry",
    "verify_final_experiment_registry",
    "verify_preregistered_training_stage",
    "write_experiment_preregistration",
    "write_final_experiment_registry",
]
