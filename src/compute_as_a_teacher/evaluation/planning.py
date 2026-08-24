"""Pure request planning for raw rollouts and rollout-only synthesis."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from compute_as_a_teacher.data.math500 import (
    DatasetPreparationError,
    QuestionRecord,
    load_dataset_lock,
)

from .artifacts import (
    artifact_reference,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    file_digest,
    publish_bytes,
    read_json,
    read_jsonl,
    sha256_bytes,
    sha256_text,
)
from .config import (
    MATH500_PROTOCOL_VERSION,
    RawEvalConfig,
    SYNTHESIS_ANCHOR_RELATIONS,
    SynthesisEvalConfig,
)
from .errors import EvaluationError
from .execution import verify_complete_execution
from .prompts import (
    REQUIRED_SYNTHESIS_ROLLOUTS,
    prompt_contract_sha256,
    render_raw_prompt,
    render_synthesis_prompt,
    validate_prompt_template,
)
from .schemas import (
    GENERATION_ROW_KEYS,
    GenerationRequest,
    ModelSpec,
    SamplingSpec,
    make_request,
)


MANIFEST_NAME = "manifest.json"
REQUESTS_NAME = "requests.jsonl"
GENERATIONS_NAME = "generations.jsonl"
EXECUTION_NAME = "execution.json"
RAW_PROMPT_VERSION = "raw_math500_local_v1"
SYNTHESIS_PROMPT_VERSION = "paper_appendix_f_cot_literal_v1"


def derive_seed(base_seed: int, question_id: str, model_role: str, index: int) -> int:
    payload = canonical_json_bytes(
        {
            "base_seed": base_seed,
            "question_id": question_id,
            "model_role": model_role,
            "index": index,
        }
    )
    digest = bytes.fromhex(sha256_bytes(payload))
    return int.from_bytes(digest[:8], "big") % (2**31)


def _request_sampling(spec: SamplingSpec, seed: int) -> dict[str, Any]:
    value = spec.to_dict(seed=seed)
    value.pop("base_seed")
    return value


def _validate_sampling_profile(model: ModelSpec, sampling: SamplingSpec) -> None:
    if not sampling.do_sample:
        raise EvaluationError("The paper-aligned profile requires do_sample=true")
    if sampling.max_new_tokens != 1536:
        raise EvaluationError(
            "The paper-aligned MATH-500 profile requires max_new_tokens=1536"
        )
    if "qwen3-4b" in model.model_id.lower():
        actual = (sampling.temperature, sampling.top_p, sampling.top_k)
        if actual != (0.7, 0.8, 20):
            raise EvaluationError(
                "The Qwen3-4B paper profile requires temperature=0.7, "
                "top_p=0.8, and top_k=20"
            )


def _validate_raw_contract(
    config: RawEvalConfig,
    *,
    require_resolved_model: bool,
) -> None:
    if config.protocol_version != MATH500_PROTOCOL_VERSION:
        raise EvaluationError("Raw config does not use the registered paper protocol")
    if type(config.rollouts_per_problem) is not int or config.rollouts_per_problem != 8:
        raise EvaluationError("Raw planning requires exactly eight rollouts per problem")
    if config.prompt.version != RAW_PROMPT_VERSION:
        raise EvaluationError(f"Raw planning requires prompt {RAW_PROMPT_VERSION}")
    if "qwen3-4b" in config.model.model_id.lower() and config.prompt.prefix != "/no_think\n":
        raise EvaluationError("Qwen3-4B prompts must begin with exactly '/no_think\\n'")
    _validate_sampling_profile(config.model, config.sampling)
    if require_resolved_model:
        config.model.assert_resolved()


def _validate_synthesis_contract(
    config: SynthesisEvalConfig,
    *,
    require_resolved_model: bool,
) -> None:
    if config.schema_version != 2 or config.kind != "synthesis":
        raise EvaluationError("Synthesis config must use schema version 2")
    if config.protocol_version != MATH500_PROTOCOL_VERSION:
        raise EvaluationError(
            "Synthesis config does not use the registered paper protocol"
        )
    if type(config.required_rollouts) is not int or config.required_rollouts != 8:
        raise EvaluationError("Synthesis planning requires exactly eight rollouts")
    if config.anchor_relation not in SYNTHESIS_ANCHOR_RELATIONS:
        raise EvaluationError("Synthesis config has an invalid anchor relation")
    if config.prompt.version != SYNTHESIS_PROMPT_VERSION:
        raise EvaluationError(
            f"Synthesis planning requires prompt {SYNTHESIS_PROMPT_VERSION}"
        )
    if "qwen3-4b" in config.anchor.model_id.lower() and config.prompt.prefix != "/no_think\n":
        raise EvaluationError("Qwen3-4B prompts must begin with exactly '/no_think\\n'")
    _validate_sampling_profile(config.anchor, config.sampling)
    if require_resolved_model:
        config.anchor.assert_resolved()


def build_raw_requests(
    questions: Sequence[QuestionRecord],
    config: RawEvalConfig,
    prompt_template: str,
) -> list[GenerationRequest]:
    """Build eight requests per question without loading any evaluation labels."""

    _validate_raw_contract(config, require_resolved_model=False)
    validate_prompt_template(prompt_template, config.prompt)
    if not questions:
        raise EvaluationError("Raw planning requires at least one question")
    if len({question.id for question in questions}) != len(questions):
        raise EvaluationError("Raw planning requires unique question IDs")
    if not all(question.id.strip() and question.problem.strip() for question in questions):
        raise EvaluationError("Question IDs and problem text must be nonempty")
    prompt_sha256 = prompt_contract_sha256(prompt_template, config.prompt)
    requests: list[GenerationRequest] = []
    for question in questions:
        content = render_raw_prompt(prompt_template, config.prompt, question.problem)
        for rollout_index in range(config.rollouts_per_problem):
            seed = derive_seed(
                config.sampling.base_seed,
                question.id,
                "policy",
                rollout_index,
            )
            requests.append(
                make_request(
                    stage="raw",
                    question_id=question.id,
                    rollout_index=rollout_index,
                    source_task_ids=(),
                    input_sha256=sha256_text(question.problem),
                    prompt_template_sha256=prompt_sha256,
                    model=config.model,
                    messages=({"role": "user", "content": content},),
                    sampling=_request_sampling(config.sampling, seed),
                )
            )
    if len({request.task_id for request in requests}) != len(requests):
        raise EvaluationError("Raw planning produced duplicate task IDs")
    return requests


def _validate_basic_raw_generation(row: Mapping[str, Any], index: int) -> None:
    if not isinstance(row, dict) or set(row) != GENERATION_ROW_KEYS:
        raise EvaluationError(f"Invalid verified raw generation row at index {index}")
    if row.get("schema_version") != 1 or row.get("stage") != "raw":
        raise EvaluationError(f"Expected a raw generation row at index {index}")
    if not isinstance(row.get("text"), str):
        raise EvaluationError(f"Raw generation text must be a string at index {index}")
    if sha256_text(row["text"]) != row.get("output_sha256"):
        raise EvaluationError(f"Raw generation output hash mismatch at index {index}")
    rollout_index = row.get("rollout_index")
    if isinstance(rollout_index, bool) or not isinstance(rollout_index, int):
        raise EvaluationError(f"Raw rollout index must be an integer at index {index}")
    if row.get("source_task_ids") != []:
        raise EvaluationError("Raw generations must not have source task IDs")


def build_synthesis_requests(
    raw_generations: Sequence[Mapping[str, Any]],
    config: SynthesisEvalConfig,
    prompt_template: str,
) -> list[GenerationRequest]:
    """Build one rollout-only synthesis request for each complete raw group."""

    _validate_synthesis_contract(config, require_resolved_model=False)
    validate_prompt_template(prompt_template, config.prompt)
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for index, row in enumerate(raw_generations):
        _validate_basic_raw_generation(row, index)
        groups[str(row["question_id"])].append(row)

    prompt_sha256 = prompt_contract_sha256(prompt_template, config.prompt)
    requests: list[GenerationRequest] = []
    for question_id in sorted(groups):
        rows = sorted(groups[question_id], key=lambda row: int(row["rollout_index"]))
        indexes = [row["rollout_index"] for row in rows]
        expected_indexes = list(range(REQUIRED_SYNTHESIS_ROLLOUTS))
        if indexes != expected_indexes:
            raise EvaluationError(
                f"Question {question_id} has rollout indexes {indexes}, "
                f"expected {expected_indexes}"
            )
        texts = [str(row["text"]) for row in rows]
        content = render_synthesis_prompt(prompt_template, config.prompt, texts)
        ordered_inputs = [
            {"task_id": row["task_id"], "output_sha256": row["output_sha256"]}
            for row in rows
        ]
        seed = derive_seed(config.sampling.base_seed, question_id, "anchor", 0)
        requests.append(
            make_request(
                stage="synthesis",
                question_id=question_id,
                rollout_index=None,
                source_task_ids=[str(row["task_id"]) for row in rows],
                input_sha256=sha256_bytes(canonical_json_bytes(ordered_inputs)),
                prompt_template_sha256=prompt_sha256,
                model=config.anchor,
                messages=({"role": "user", "content": content},),
                sampling=_request_sampling(config.sampling, seed),
            )
        )
    if not requests:
        raise EvaluationError("No complete raw generation groups were available for synthesis")
    if len({request.task_id for request in requests}) != len(requests):
        raise EvaluationError("Synthesis planning produced duplicate task IDs")
    return requests


def _logical_reference(
    path: Path,
    logical_path: str,
    *,
    rows: int | None = None,
) -> dict[str, Any]:
    reference = artifact_reference(path, rows=rows)
    reference["path"] = logical_path
    return reference


def _safe_repository_path(repository_root: Path, relative_path: str) -> Path:
    root = repository_root.resolve()
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise EvaluationError(f"Path escapes repository root: {relative_path}") from exc
    return path


def _question_payload(questions: Sequence[QuestionRecord]) -> bytes:
    return canonical_jsonl_bytes(
        {"id": question.id, "problem": question.problem} for question in questions
    )


def _verified_question_reference(
    questions: Sequence[QuestionRecord],
    questions_path: Path,
    config: RawEvalConfig,
    *,
    repository_root: Path | None,
    allow_test_fixture: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None, bool]:
    payload = _question_payload(questions)
    try:
        actual_payload = questions_path.read_bytes()
    except OSError as exc:
        raise EvaluationError(f"Cannot read question artifact {questions_path}: {exc}") from exc
    if payload != actual_payload:
        raise EvaluationError(
            "Supplied QuestionRecord values do not exactly match the question artifact"
        )
    if allow_test_fixture:
        return (
            _logical_reference(
                questions_path,
                "test-fixture/questions.jsonl",
                rows=len(questions),
            ),
            None,
            False,
        )
    if repository_root is None:
        raise EvaluationError("A repository root is required to verify locked questions")
    lock_path = _safe_repository_path(repository_root, config.dataset_lock_path)
    expected_questions_path = _safe_repository_path(repository_root, config.questions_path)
    if questions_path.resolve() != expected_questions_path:
        raise EvaluationError("questions_path is not the config's locked model-facing path")
    try:
        lock = load_dataset_lock(lock_path)
    except DatasetPreparationError as exc:
        raise EvaluationError(str(exc)) from exc
    firewall_path = lock["label_firewall"]["training_input"]
    question_spec = lock["outputs"]["questions"]
    if firewall_path != question_spec["path"] or firewall_path != config.questions_path:
        raise EvaluationError("Config does not point to the locked label-free training input")
    actual_sha, actual_bytes = file_digest(questions_path)
    if (
        actual_sha != question_spec["sha256"]
        or actual_bytes != question_spec["bytes"]
        or len(questions) != question_spec["rows"]
    ):
        raise EvaluationError("Question artifact does not match the dataset lock")
    return (
        _logical_reference(
            questions_path,
            config.questions_path,
            rows=len(questions),
        ),
        _logical_reference(
            lock_path,
            config.dataset_lock_path,
        ),
        True,
    )


def _plan_manifest(
    *,
    kind: str,
    protocol_version: str,
    run_name: str,
    config: Mapping[str, Any],
    model: ModelSpec,
    prompt_template: str,
    prompt_version: str,
    prompt_prefix: str,
    input_reference: Mapping[str, Any],
    requests_payload: bytes,
    request_count: int,
    problem_count: int,
    rollouts_per_problem: int,
    locked_questions_verified: bool,
) -> dict[str, Any]:
    manifest_without_fingerprint: dict[str, Any] = {
        "schema_version": 1,
        "kind": kind,
        "protocol_version": protocol_version,
        "run_name": run_name,
        "model": model.to_dict(),
        "config": dict(config),
        "prompt": {
            "version": prompt_version,
            "template_sha256": sha256_text(prompt_template),
            "contract_sha256": sha256_text(
                f"version={prompt_version}\nprefix={prompt_prefix}\n{prompt_template}"
            ),
        },
        "inputs": dict(input_reference),
        "counts": {
            "problems": problem_count,
            "requests": request_count,
            "rollouts_per_problem": rollouts_per_problem,
        },
        "artifacts": {
            "requests": {
                "path": REQUESTS_NAME,
                "sha256": sha256_bytes(requests_payload),
                "bytes": len(requests_payload),
                "rows": request_count,
            }
        },
        "label_firewall": {
            "labels_loaded": False,
            "reference_answers_loaded": False,
            "reference_solutions_loaded": False,
            "locked_questions_verified": locked_questions_verified,
            "question_field_supplied_to_synthesis": False if kind == "synthesis" else None,
        },
        "paper_contract": {
            "rollouts_per_problem": 8,
            "raw_primary_protocol_decision": (
                "rollout_index_0" if kind == "raw" else None
            ),
            "synthesis_receives_rollout_text_only": kind == "synthesis",
            "synthesis_anchor_relation": (
                config.get("anchor_relation") if kind == "synthesis" else None
            ),
        },
    }
    manifest_without_fingerprint["plan_fingerprint"] = sha256_bytes(
        canonical_json_bytes(manifest_without_fingerprint)
    )
    return manifest_without_fingerprint


def _descendant_artifacts(run_dir: Path) -> list[Path]:
    names = (
        "results",
        GENERATIONS_NAME,
        EXECUTION_NAME,
        "responses.jsonl",
        "scores.jsonl",
        "paired_scores.jsonl",
        "summary.json",
        "scoring_manifest.json",
    )
    return [run_dir / name for name in names if (run_dir / name).exists()]


def _preflight_plan(
    run_dir: Path,
    requests_payload: bytes,
    manifest_payload: bytes,
    force: bool,
) -> None:
    mismatched = []
    for path, payload in (
        (run_dir / REQUESTS_NAME, requests_payload),
        (run_dir / MANIFEST_NAME, manifest_payload),
    ):
        if path.exists() and (not path.is_file() or path.read_bytes() != payload):
            mismatched.append(path)
    if mismatched and not force:
        raise EvaluationError(
            f"Refusing to replace mismatched plan artifacts: {mismatched}; "
            "use --force intentionally"
        )
    descendants = _descendant_artifacts(run_dir)
    if mismatched and descendants:
        raise EvaluationError(
            "Refusing to re-plan a run directory with descendant artifacts; "
            "choose a new run directory"
        )


def write_raw_plan(
    run_dir: Path,
    questions: Sequence[QuestionRecord],
    config: RawEvalConfig,
    prompt_template: str,
    questions_path: Path,
    *,
    repository_root: Path | None = None,
    allow_test_fixture: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    _validate_raw_contract(config, require_resolved_model=True)
    (
        question_reference,
        dataset_lock_reference,
        locked_questions_verified,
    ) = _verified_question_reference(
        questions,
        questions_path,
        config,
        repository_root=repository_root,
        allow_test_fixture=allow_test_fixture,
    )
    requests = build_raw_requests(questions, config, prompt_template)
    requests_payload = canonical_jsonl_bytes(request.to_dict() for request in requests)
    manifest = _plan_manifest(
        kind="raw",
        protocol_version=config.protocol_version,
        run_name=config.run_name,
        config=config.to_dict(),
        model=config.model,
        prompt_template=prompt_template,
        prompt_version=config.prompt.version,
        prompt_prefix=config.prompt.prefix,
        input_reference={
            "questions": question_reference,
            "dataset_lock": dataset_lock_reference,
        },
        requests_payload=requests_payload,
        request_count=len(requests),
        problem_count=len(questions),
        rollouts_per_problem=config.rollouts_per_problem,
        locked_questions_verified=locked_questions_verified,
    )
    manifest_payload = canonical_json_bytes(manifest)
    _preflight_plan(run_dir, requests_payload, manifest_payload, force)
    publish_bytes(run_dir / REQUESTS_NAME, requests_payload, force=force)
    publish_bytes(run_dir / MANIFEST_NAME, manifest_payload, force=force)
    return manifest


def write_synthesis_plan(
    run_dir: Path,
    raw_run_dir: Path,
    config: SynthesisEvalConfig,
    prompt_template: str,
    *,
    force: bool = False,
) -> dict[str, Any]:
    _validate_synthesis_contract(config, require_resolved_model=True)
    raw_manifest, raw_requests = load_plan(raw_run_dir, expected_kind="raw")
    if raw_manifest["protocol_version"] != config.protocol_version:
        raise EvaluationError("Raw and synthesis protocol versions must match")
    validate_synthesis_anchor_relation(config, raw_requests)
    raw_sampling = dict(raw_manifest["config"]["sampling"])
    synthesis_sampling = config.sampling.to_dict()
    raw_sampling.pop("base_seed", None)
    synthesis_sampling.pop("base_seed", None)
    if raw_sampling != synthesis_sampling:
        raise EvaluationError(
            "Raw policy and frozen synthesis anchor must use the same decoding profile"
        )
    raw_execution, raw_generations = verify_complete_execution(
        raw_run_dir,
        raw_manifest,
        raw_requests,
    )
    generations_path = raw_run_dir / GENERATIONS_NAME
    requests = build_synthesis_requests(raw_generations, config, prompt_template)
    requests_payload = canonical_jsonl_bytes(request.to_dict() for request in requests)
    locked_questions_verified = bool(
        raw_manifest["label_firewall"].get("locked_questions_verified", False)
    )
    input_reference = {
        "raw_plan_fingerprint": raw_manifest["plan_fingerprint"],
        "raw_generations": _logical_reference(
            generations_path,
            "upstream/raw/generations.jsonl",
            rows=len(raw_generations),
        ),
        "raw_execution": _logical_reference(
            raw_run_dir / EXECUTION_NAME,
            "upstream/raw/execution.json",
        ),
        "raw_execution_non_reportable": bool(
            raw_execution.get("non_reportable", True)
        ),
    }
    manifest = _plan_manifest(
        kind="synthesis",
        protocol_version=config.protocol_version,
        run_name=config.run_name,
        config=config.to_dict(),
        model=config.anchor,
        prompt_template=prompt_template,
        prompt_version=config.prompt.version,
        prompt_prefix=config.prompt.prefix,
        input_reference=input_reference,
        requests_payload=requests_payload,
        request_count=len(requests),
        problem_count=len(requests),
        rollouts_per_problem=config.required_rollouts,
        locked_questions_verified=locked_questions_verified,
    )
    manifest_payload = canonical_json_bytes(manifest)
    _preflight_plan(run_dir, requests_payload, manifest_payload, force)
    publish_bytes(run_dir / REQUESTS_NAME, requests_payload, force=force)
    publish_bytes(run_dir / MANIFEST_NAME, manifest_payload, force=force)
    return manifest


def validate_synthesis_anchor_relation(
    config: SynthesisEvalConfig,
    raw_requests: Sequence[GenerationRequest],
) -> bool:
    same_anchor = all(request.model == config.anchor for request in raw_requests)
    if config.anchor_relation == "same_as_raw" and not same_anchor:
        raise EvaluationError("same_as_raw synthesis requires the exact raw policy")
    if config.anchor_relation == "frozen_initial_for_trained_raw" and same_anchor:
        raise EvaluationError(
            "trained-policy synthesis requires a distinct frozen initial-policy anchor"
        )
    return same_anchor


def _validate_manifest(
    manifest: Mapping[str, Any],
    run_dir: Path,
) -> None:
    expected_keys = {
        "schema_version",
        "kind",
        "protocol_version",
        "run_name",
        "model",
        "config",
        "prompt",
        "inputs",
        "counts",
        "artifacts",
        "label_firewall",
        "paper_contract",
        "plan_fingerprint",
    }
    if set(manifest) != expected_keys or manifest.get("schema_version") != 1:
        raise EvaluationError(f"Invalid plan-manifest schema in {run_dir}")
    fingerprint = manifest.get("plan_fingerprint")
    if not isinstance(fingerprint, str):
        raise EvaluationError("Plan fingerprint must be a string")
    semantic_manifest = dict(manifest)
    semantic_manifest.pop("plan_fingerprint")
    expected_fingerprint = sha256_bytes(canonical_json_bytes(semantic_manifest))
    if fingerprint != expected_fingerprint:
        raise EvaluationError("Plan manifest fingerprint mismatch")


def load_plan(
    run_dir: Path,
    *,
    expected_kind: str | None = None,
) -> tuple[dict[str, Any], list[GenerationRequest]]:
    manifest = read_json(run_dir / MANIFEST_NAME)
    _validate_manifest(manifest, run_dir)
    kind = manifest.get("kind")
    if kind not in {"raw", "synthesis"}:
        raise EvaluationError(f"Invalid plan kind in {run_dir}: {kind}")
    if expected_kind is not None and kind != expected_kind:
        raise EvaluationError(f"Expected a {expected_kind} plan, found {kind}")
    if manifest.get("protocol_version") != MATH500_PROTOCOL_VERSION:
        raise EvaluationError("Plan does not use the registered MATH-500 protocol")

    request_rows = read_jsonl(run_dir / REQUESTS_NAME)
    requests = [GenerationRequest.from_dict(row) for row in request_rows]
    request_spec = manifest.get("artifacts", {}).get("requests", {})
    actual_sha, actual_bytes = file_digest(run_dir / REQUESTS_NAME)
    if (
        request_spec.get("path") != REQUESTS_NAME
        or actual_sha != request_spec.get("sha256")
        or actual_bytes != request_spec.get("bytes")
        or len(requests) != request_spec.get("rows")
    ):
        raise EvaluationError("Request artifact does not match its plan manifest")
    if len({request.task_id for request in requests}) != len(requests):
        raise EvaluationError("Plan contains duplicate task IDs")
    if not requests or any(request.stage != kind for request in requests):
        raise EvaluationError("Request stages do not match the plan kind")

    manifest_model = ModelSpec.from_dict(manifest["model"])
    if any(request.model != manifest_model for request in requests):
        raise EvaluationError("Request model identity does not match the plan manifest")
    config_value = manifest.get("config")
    if not isinstance(config_value, dict):
        raise EvaluationError("Plan config must be an object")
    expected_config_keys = (
        {
            "schema_version",
            "kind",
            "protocol_version",
            "run_name",
            "questions_path",
            "dataset_lock_path",
            "rollouts_per_problem",
            "prompt",
            "model",
            "sampling",
        }
        if kind == "raw"
        else {
            "schema_version",
            "kind",
            "protocol_version",
            "run_name",
            "required_rollouts",
            "anchor_relation",
            "prompt",
            "anchor",
            "sampling",
        }
    )
    if set(config_value) != expected_config_keys:
        raise EvaluationError("Plan contains an invalid embedded config schema")
    model_key = "model" if kind == "raw" else "anchor"
    config_schema_version = 1 if kind == "raw" else 2
    if (
        config_value.get("schema_version") != config_schema_version
        or config_value.get("kind") != kind
        or config_value.get("protocol_version") != manifest["protocol_version"]
        or config_value.get("run_name") != manifest["run_name"]
        or config_value.get(model_key) != manifest["model"]
    ):
        raise EvaluationError("Embedded plan config does not match the manifest")
    if kind == "raw" and config_value.get("rollouts_per_problem") != 8:
        raise EvaluationError("Embedded raw config does not require eight rollouts")
    if kind == "synthesis" and (
        config_value.get("schema_version") != 2
        or config_value.get("required_rollouts") != 8
        or config_value.get("anchor_relation") not in SYNTHESIS_ANCHOR_RELATIONS
    ):
        raise EvaluationError("Embedded synthesis config violates the paper contract")
    paper_contract = manifest.get("paper_contract")
    if not isinstance(paper_contract, dict) or set(paper_contract) != {
        "rollouts_per_problem",
        "raw_primary_protocol_decision",
        "synthesis_receives_rollout_text_only",
        "synthesis_anchor_relation",
    }:
        raise EvaluationError("Plan contains an invalid paper-contract schema")
    expected_relation = (
        config_value.get("anchor_relation") if kind == "synthesis" else None
    )
    if (
        paper_contract.get("rollouts_per_problem") != 8
        or paper_contract.get("raw_primary_protocol_decision")
        != ("rollout_index_0" if kind == "raw" else None)
        or paper_contract.get("synthesis_receives_rollout_text_only")
        is not (kind == "synthesis")
        or paper_contract.get("synthesis_anchor_relation") != expected_relation
    ):
        raise EvaluationError("Plan paper contract does not match its config")
    embedded_prompt = config_value.get("prompt")
    if (
        not isinstance(embedded_prompt, dict)
        or embedded_prompt.get("version") != manifest.get("prompt", {}).get("version")
    ):
        raise EvaluationError("Embedded prompt config does not match the manifest")
    embedded_sampling = config_value.get("sampling")
    if not isinstance(embedded_sampling, dict):
        raise EvaluationError("Embedded sampling config must be an object")
    sampling_spec = SamplingSpec.from_dict(embedded_sampling)
    base_seed = sampling_spec.base_seed
    request_sampling = sampling_spec.to_dict()
    request_sampling.pop("base_seed")
    for request in requests:
        actual_sampling = dict(request.sampling)
        actual_seed = actual_sampling.pop("seed")
        if actual_sampling != request_sampling:
            raise EvaluationError("Request decoding parameters do not match the plan config")
        expected_seed = derive_seed(
            base_seed,
            request.question_id,
            "policy" if kind == "raw" else "anchor",
            int(request.rollout_index) if kind == "raw" else 0,
        )
        if actual_seed != expected_seed:
            raise EvaluationError("Request seed does not match the plan seed contract")
    contract_sha = manifest.get("prompt", {}).get("contract_sha256")
    if any(request.prompt_template_sha256 != contract_sha for request in requests):
        raise EvaluationError("Request prompt contract does not match the plan manifest")

    groups: dict[str, list[GenerationRequest]] = defaultdict(list)
    for request in requests:
        groups[request.question_id].append(request)
    counts = manifest.get("counts", {})
    if (
        counts.get("problems") != len(groups)
        or counts.get("requests") != len(requests)
        or counts.get("rollouts_per_problem") != 8
    ):
        raise EvaluationError("Plan counts do not match its requests")
    for question_id, group in groups.items():
        if kind == "raw":
            indexes = sorted(request.rollout_index for request in group)
            if indexes != list(range(8)):
                raise EvaluationError(
                    f"Raw request group {question_id} does not contain indexes 0..7"
                )
        elif len(group) != 1 or len(group[0].source_task_ids) != 8:
            raise EvaluationError(
                f"Synthesis request group {question_id} is not one request over eight sources"
            )
    return manifest, requests
