"""Pure, label-free reward computation for MATH CaT rollout groups."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import fsum, isfinite, sqrt
from typing import Any, Literal

from compute_as_a_teacher.evaluation.config import PromptSpec
from compute_as_a_teacher.evaluation.grading import (
    ExtractedAnswer,
    ExtractionStatus,
    extract_last_boxed,
)
from compute_as_a_teacher.evaluation.prompts import (
    REQUIRED_SYNTHESIS_ROLLOUTS,
    render_synthesis_prompt,
)

from .errors import InvalidAnchorAnswerError, RewardContractError


GROUP_SIZE = REQUIRED_SYNTHESIS_ROLLOUTS
DEFAULT_EPSILON = 1e-6
DEFAULT_STD_DDOF = 1
DEFAULT_MAX_ANSWER_CHARS = 50_000

AnchorFailurePolicy = Literal["fail_closed", "reward_zero"]
ANCHOR_FAILURE_POLICIES = frozenset({"fail_closed", "reward_zero"})


def _require_ordered_sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RewardContractError(f"{name} must be an ordered sequence")
    return value


def _validate_epsilon(epsilon: Any) -> float:
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise RewardContractError("epsilon must be a finite positive number")
    normalized = float(epsilon)
    if not isfinite(normalized) or normalized <= 0:
        raise RewardContractError("epsilon must be a finite positive number")
    return normalized


def _validate_anchor_failure_policy(value: Any) -> AnchorFailurePolicy:
    if not isinstance(value, str) or value not in ANCHOR_FAILURE_POLICIES:
        raise RewardContractError(
            "anchor_failure_policy must be 'fail_closed' or 'reward_zero'"
        )
    return value


@dataclass(frozen=True, slots=True)
class OrderedRolloutGroup:
    """Exactly eight rollout texts whose tuple position is their paper order."""

    rollouts: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.rollouts, tuple):
            raise RewardContractError("rollouts must be stored as an immutable tuple")
        if len(self.rollouts) != GROUP_SIZE:
            raise RewardContractError(
                f"CaT reward groups require exactly {GROUP_SIZE} ordered rollouts"
            )
        for position, rollout in enumerate(self.rollouts, start=1):
            if not isinstance(rollout, str):
                raise RewardContractError(f"rollout {position} must be text")

    @classmethod
    def from_sequence(cls, rollouts: Sequence[str]) -> "OrderedRolloutGroup":
        ordered = _require_ordered_sequence(rollouts, "rollouts")
        return cls(tuple(ordered))

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_size": GROUP_SIZE,
            "rollouts": [
                {"position": position, "text": rollout}
                for position, rollout in enumerate(self.rollouts, start=1)
            ],
        }


@dataclass(frozen=True, slots=True)
class MathRewardResult:
    """Position-aligned extraction and reward results."""

    rollout_extractions: tuple[ExtractedAnswer, ...]
    anchor_extraction: ExtractedAnswer
    rewards: tuple[int, ...]
    anchor_failure_policy: AnchorFailurePolicy

    def __post_init__(self) -> None:
        if not isinstance(self.rollout_extractions, tuple):
            raise RewardContractError("rollout_extractions must be an immutable tuple")
        if len(self.rollout_extractions) != GROUP_SIZE:
            raise RewardContractError(
                f"rollout_extractions must contain exactly {GROUP_SIZE} entries"
            )
        if not all(
            isinstance(extraction, ExtractedAnswer)
            for extraction in self.rollout_extractions
        ):
            raise RewardContractError(
                "rollout_extractions must contain ExtractedAnswer values"
            )
        if not isinstance(self.anchor_extraction, ExtractedAnswer):
            raise RewardContractError("anchor_extraction must be an ExtractedAnswer")
        if not isinstance(self.rewards, tuple):
            raise RewardContractError("rewards must be an immutable tuple")
        if len(self.rewards) != GROUP_SIZE or any(
            type(reward) is not int or reward not in (0, 1)
            for reward in self.rewards
        ):
            raise RewardContractError(
                f"rewards must contain exactly {GROUP_SIZE} binary integers"
            )
        _validate_anchor_failure_policy(self.anchor_failure_policy)

    @property
    def rollout_statuses(self) -> tuple[ExtractionStatus, ...]:
        return tuple(extraction.status for extraction in self.rollout_extractions)

    @property
    def anchor_status(self) -> ExtractionStatus:
        return self.anchor_extraction.status

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_extraction": self.anchor_extraction.to_dict(),
            "rollout_extractions": [
                extraction.to_dict() for extraction in self.rollout_extractions
            ],
            "rewards": list(self.rewards),
            "anchor_failure_policy": self.anchor_failure_policy,
        }


def render_anchor_prompt(
    template: str,
    prompt: PromptSpec,
    ordered_rollouts: Sequence[str],
) -> str:
    """Render the anchor input from only the eight rollouts, never the question."""

    if not isinstance(template, str):
        raise RewardContractError("template must be text")
    if not isinstance(prompt, PromptSpec):
        raise RewardContractError("prompt must be a PromptSpec")
    group = OrderedRolloutGroup.from_sequence(ordered_rollouts)
    return render_synthesis_prompt(template, prompt, group.rollouts)


def group_normalized_advantages(
    rewards: Sequence[int | float],
    *,
    epsilon: float = DEFAULT_EPSILON,
    std_ddof: int = DEFAULT_STD_DDOF,
) -> tuple[float, ...]:
    """Mirror verl's GRPO group normalization for model-free tests."""

    ordered = _require_ordered_sequence(rewards, "rewards")
    if len(ordered) != GROUP_SIZE:
        raise RewardContractError(
            f"reward normalization requires exactly {GROUP_SIZE} rewards"
        )
    values: list[float] = []
    for position, reward in enumerate(ordered, start=1):
        if isinstance(reward, bool) or not isinstance(reward, (int, float)):
            raise RewardContractError(f"reward {position} must be numeric and binary")
        normalized = float(reward)
        if not isfinite(normalized) or normalized not in (0.0, 1.0):
            raise RewardContractError(f"reward {position} must be numeric and binary")
        values.append(normalized)

    normalized_epsilon = _validate_epsilon(epsilon)
    if type(std_ddof) is not int or not 0 <= std_ddof < GROUP_SIZE:
        raise RewardContractError(f"std_ddof must be an integer in [0, {GROUP_SIZE})")
    mean = fsum(values) / GROUP_SIZE
    variance = fsum((reward - mean) ** 2 for reward in values) / (
        GROUP_SIZE - std_ddof
    )
    std = sqrt(variance)
    if std == 0.0:
        return (0.0,) * GROUP_SIZE
    denominator = std + normalized_epsilon
    return tuple((reward - mean) / denominator for reward in values)


def compute_math_rewards(
    ordered_rollouts: Sequence[str],
    synthesized: str,
    *,
    max_answer_chars: int = DEFAULT_MAX_ANSWER_CHARS,
    anchor_failure_policy: AnchorFailurePolicy = "fail_closed",
) -> MathRewardResult:
    r"""Return label-free boxed-answer agreement rewards for one CaT group.

    This API deliberately accepts no problem, gold label, or reference answer.
    The synthesized response is the only reward reference. Rollout extraction
    failures receive zero; an invalid synthesized anchor raises by default.
    """

    group = OrderedRolloutGroup.from_sequence(ordered_rollouts)
    if not isinstance(synthesized, str):
        raise RewardContractError("synthesized must be text")
    if type(max_answer_chars) is not int or max_answer_chars <= 0:
        raise RewardContractError("max_answer_chars must be a positive integer")
    policy = _validate_anchor_failure_policy(anchor_failure_policy)

    anchor_extraction = extract_last_boxed(
        synthesized,
        max_answer_chars=max_answer_chars,
    )
    rollout_extractions = tuple(
        extract_last_boxed(rollout, max_answer_chars=max_answer_chars)
        for rollout in group.rollouts
    )

    if anchor_extraction.status != "ok" or anchor_extraction.value is None:
        if policy == "fail_closed":
            raise InvalidAnchorAnswerError(anchor_extraction.status)
        rewards = (0,) * GROUP_SIZE
    else:
        rewards = tuple(
            int(
                extraction.status == "ok"
                and extraction.value is not None
                and extraction.value == anchor_extraction.value
            )
            for extraction in rollout_extractions
        )

    return MathRewardResult(
        rollout_extractions=rollout_extractions,
        anchor_extraction=anchor_extraction,
        rewards=rewards,
        anchor_failure_policy=policy,
    )
