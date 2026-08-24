"""verl v0.5.0 batch reward hook for label-free MATH CaT training.

``BatchRewardManager`` calls :func:`compute_score` once with parallel vectors.
Rows are grouped by ``extra_info["question_id"]`` even when question groups are
interleaved.  Within a group, rollout order is the encounter order in the batch:
the paper treats the eight samples as exchangeable, and verl creates
``extra_info`` before repeating each prompt, so no rollout index is available.

This module imports neither torch nor verl.  It is repo-owned glue between the
frozen anchor endpoint and the pure reward contract in :mod:`.rewards`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from compute_as_a_teacher.evaluation.artifacts import sha256_text
from compute_as_a_teacher.evaluation.config import PromptSpec
from compute_as_a_teacher.evaluation.planning import derive_seed
from compute_as_a_teacher.evaluation.prompts import load_prompt

from compute_as_a_teacher.training.anchor_client import (
    AnchorChatClient,
    OpenAIChatCompletionsClient,
    completion_text,
)
from compute_as_a_teacher.training.errors import RewardContractError
from compute_as_a_teacher.training.rewards import (
    ANCHOR_FAILURE_POLICIES,
    DEFAULT_MAX_ANSWER_CHARS,
    AnchorFailurePolicy,
    compute_math_rewards,
    render_anchor_prompt,
)


REWARD_CONTRACT_VERSION = "cat_math_boxed_agreement_v2"


@dataclass(frozen=True, slots=True)
class _QuestionGroup:
    question_id: str
    row_indices: tuple[int, ...]
    rollouts: tuple[str, ...]


def _vector(value: Any, name: str) -> list[Any]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise RewardContractError(f"{name} must be a batch vector")
    if not isinstance(value, Iterable):
        raise RewardContractError(f"{name} must be a batch vector")
    try:
        return list(value)
    except (TypeError, ValueError) as exc:
        raise RewardContractError(f"{name} must be a batch vector") from exc


def _nonempty_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RewardContractError(f"{name} must be nonempty text")
    return value


def _positive_integer(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise RewardContractError(f"{name} must be a positive integer")
    return value


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RewardContractError(f"{name} must be a finite positive number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0:
        raise RewardContractError(f"{name} must be a finite positive number")
    return normalized


def _validate_sampling(
    *,
    temperature: Any,
    top_p: Any,
    top_k: Any,
    max_tokens: Any,
) -> tuple[float, float, int, int]:
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
        or not 0 <= float(temperature) <= 2
    ):
        raise RewardContractError("anchor_temperature must be in [0, 2]")
    if (
        isinstance(top_p, bool)
        or not isinstance(top_p, (int, float))
        or not math.isfinite(float(top_p))
        or not 0 < float(top_p) <= 1
    ):
        raise RewardContractError("anchor_top_p must be in (0, 1]")
    return (
        float(temperature),
        float(top_p),
        _positive_integer(top_k, "anchor_top_k"),
        _positive_integer(max_tokens, "anchor_max_tokens"),
    )


def _validate_and_group(
    data_sources: Any,
    solution_strs: Any,
    ground_truths: Any,
    extra_infos: Any,
) -> tuple[list[str], tuple[_QuestionGroup, ...]]:
    sources = _vector(data_sources, "data_sources")
    solutions = _vector(solution_strs, "solution_strs")
    references = _vector(ground_truths, "ground_truths")
    infos = _vector(extra_infos, "extra_infos")
    lengths = {len(sources), len(solutions), len(references), len(infos)}
    if len(lengths) != 1:
        raise RewardContractError("verl reward vectors must have identical lengths")
    if not solutions:
        raise RewardContractError("verl reward batch must not be empty")

    for row_index, data_source in enumerate(sources):
        if not isinstance(data_source, str) or not data_source.strip():
            raise RewardContractError(
                f"data_sources[{row_index}] must be nonempty text"
            )
    for row_index, solution in enumerate(solutions):
        if not isinstance(solution, str):
            raise RewardContractError(f"solution_strs[{row_index}] must be text")
    for row_index, reference in enumerate(references):
        if reference is not None:
            raise RewardContractError(
                "ground_truths must contain only None; gold labels are forbidden "
                f"during CaT training (row {row_index})"
            )

    grouped_indices: dict[str, list[int]] = {}
    for row_index, extra_info in enumerate(infos):
        if not isinstance(extra_info, Mapping):
            raise RewardContractError(f"extra_infos[{row_index}] must be an object")
        # Deliberately read no problem, answer, label, or other dataset field.
        question_id = extra_info.get("question_id")
        if not isinstance(question_id, str) or not question_id.strip():
            raise RewardContractError(
                f"extra_infos[{row_index}].question_id must be nonempty text"
            )
        grouped_indices.setdefault(question_id, []).append(row_index)

    groups: list[_QuestionGroup] = []
    for question_id, row_indices in grouped_indices.items():
        if len(row_indices) != 8:
            raise RewardContractError(
                f"question {question_id!r} has {len(row_indices)} rollouts; "
                "CaT requires exactly 8"
            )
        indices = tuple(row_indices)
        groups.append(
            _QuestionGroup(
                question_id=question_id,
                row_indices=indices,
                rollouts=tuple(solutions[index] for index in indices),
            )
        )
    return solutions, tuple(groups)


def _score_group(
    group: _QuestionGroup,
    *,
    client: AnchorChatClient,
    template: str,
    prompt: PromptSpec,
    model: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    base_seed: int,
    max_answer_chars: int,
    anchor_failure_policy: AnchorFailurePolicy,
) -> tuple[tuple[int, dict[str, Any]], ...]:
    anchor_prompt = render_anchor_prompt(template, prompt, group.rollouts)
    seed = derive_seed(base_seed, group.question_id, "anchor", 0)
    started = perf_counter()
    try:
        completion = client.complete(
            model=model,
            message=anchor_prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            seed=seed,
        )
    except RewardContractError:
        raise
    except Exception as exc:
        raise RewardContractError(
            f"anchor client failed for question {group.question_id!r}"
        ) from exc
    latency_seconds = perf_counter() - started
    synthesized, finish_reason = completion_text(completion)
    if not synthesized.strip():
        raise RewardContractError("anchor client returned an invalid text response")

    reward_result = compute_math_rewards(
        group.rollouts,
        synthesized,
        max_answer_chars=max_answer_chars,
        anchor_failure_policy=anchor_failure_policy,
    )
    extracted_anchor = reward_result.anchor_extraction.value
    anchor_answer_sha256 = (
        sha256_text(extracted_anchor) if extracted_anchor is not None else None
    )
    shared_audit = {
        "reward_contract_version": REWARD_CONTRACT_VERSION,
        "anchor_seed": seed,
        "anchor_prompt_sha256": sha256_text(anchor_prompt),
        "anchor_response_sha256": sha256_text(synthesized),
        "anchor_answer_sha256": anchor_answer_sha256,
        "anchor_extraction_status": reward_result.anchor_status,
        "anchor_failure_policy": reward_result.anchor_failure_policy,
        "anchor_finish_reason": finish_reason,
        "anchor_latency_seconds": latency_seconds,
    }
    rows: list[tuple[int, dict[str, Any]]] = []
    for position, row_index in enumerate(group.row_indices):
        score = {
            "score": reward_result.rewards[position],
            **shared_audit,
            "rollout_extraction_status": reward_result.rollout_statuses[position],
        }
        rows.append((row_index, score))
    return tuple(rows)


def compute_score(
    data_sources: Any,
    solution_strs: Any,
    ground_truths: Any,
    extra_infos: Any,
    *,
    repository_root: str | Path,
    prompt_path: str,
    prompt_version: str,
    prompt_prefix: str,
    anchor_base_url: str,
    anchor_model: str,
    anchor_api_key_env: str,
    anchor_timeout_seconds: float,
    anchor_max_concurrency: int,
    anchor_temperature: float,
    anchor_top_p: float,
    anchor_top_k: int,
    anchor_max_tokens: int,
    base_seed: int,
    max_answer_chars: int = DEFAULT_MAX_ANSWER_CHARS,
    anchor_failure_policy: AnchorFailurePolicy = "fail_closed",
    anchor_client: AnchorChatClient | None = None,
) -> list[dict[str, Any]]:
    """Return position-aligned score dictionaries for verl BatchRewardManager.

    All kwargs except ``anchor_client`` are JSON/Hydra-friendly values emitted by
    the checked-in training config compiler.  ``anchor_client`` is an in-process
    test seam; normal runs construct the strict HTTP client from endpoint config.
    Unknown kwargs are rejected by this explicit signature.
    """

    solutions, groups = _validate_and_group(
        data_sources,
        solution_strs,
        ground_truths,
        extra_infos,
    )
    root = Path(repository_root).resolve()
    normalized_prompt_path = _nonempty_text(prompt_path, "prompt_path")
    normalized_prompt_version = _nonempty_text(prompt_version, "prompt_version")
    if not isinstance(prompt_prefix, str):
        raise RewardContractError("prompt_prefix must be text")
    prompt = PromptSpec(
        path=normalized_prompt_path,
        version=normalized_prompt_version,
        prefix=prompt_prefix,
    )
    template = load_prompt(root, prompt)
    normalized_model = _nonempty_text(anchor_model, "anchor_model")
    temperature, top_p, top_k, max_tokens = _validate_sampling(
        temperature=anchor_temperature,
        top_p=anchor_top_p,
        top_k=anchor_top_k,
        max_tokens=anchor_max_tokens,
    )
    concurrency = _positive_integer(
        anchor_max_concurrency,
        "anchor_max_concurrency",
    )
    timeout_seconds = _positive_number(
        anchor_timeout_seconds,
        "anchor_timeout_seconds",
    )
    if type(base_seed) is not int or not 0 <= base_seed < 2**31:
        raise RewardContractError("base_seed must be an integer in [0, 2**31)")
    normalized_max_answer_chars = _positive_integer(
        max_answer_chars,
        "max_answer_chars",
    )
    if anchor_failure_policy not in ANCHOR_FAILURE_POLICIES:
        raise RewardContractError(
            "anchor_failure_policy must be 'fail_closed' or 'reward_zero'"
        )
    if not isinstance(anchor_api_key_env, str):
        raise RewardContractError("anchor_api_key_env must be text")

    # Validate endpoint syntax even when tests inject a fake client.  Constructing
    # a client is side-effect free; no request is made until ``complete``.
    OpenAIChatCompletionsClient(
        base_url=anchor_base_url,
        timeout_seconds=timeout_seconds,
    )

    if anchor_client is None:
        client: AnchorChatClient = OpenAIChatCompletionsClient.from_environment(
            base_url=anchor_base_url,
            api_key_env=anchor_api_key_env,
            timeout_seconds=timeout_seconds,
        )
    else:
        if not callable(getattr(anchor_client, "complete", None)):
            raise RewardContractError("anchor_client must implement complete()")
        client = anchor_client

    score_rows: list[dict[str, Any] | None] = [None] * len(solutions)
    common = {
        "client": client,
        "template": template,
        "prompt": prompt,
        "model": normalized_model,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_tokens": max_tokens,
        "base_seed": base_seed,
        "max_answer_chars": normalized_max_answer_chars,
        "anchor_failure_policy": anchor_failure_policy,
    }

    if concurrency == 1 or len(groups) == 1:
        completed = [_score_group(group, **common) for group in groups]
    else:
        worker_count = min(concurrency, len(groups))
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="cat-anchor",
        ) as executor:
            futures: list[Future[tuple[tuple[int, dict[str, Any]], ...]]] = [
                executor.submit(_score_group, group, **common) for group in groups
            ]
            # Consume in stable group order; the final mapping is by original row.
            completed = [future.result() for future in futures]

    for group_rows in completed:
        for row_index, score in group_rows:
            if score_rows[row_index] is not None:  # defensive invariant
                raise RewardContractError("duplicate score produced for a batch row")
            score_rows[row_index] = score
    if any(score is None for score in score_rows):  # defensive invariant
        raise RewardContractError("reward hook did not cover every batch row")
    return [score for score in score_rows if score is not None]


__all__ = ["REWARD_CONTRACT_VERSION", "compute_score"]
