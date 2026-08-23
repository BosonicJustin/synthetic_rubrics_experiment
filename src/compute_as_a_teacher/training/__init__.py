"""Model-free contracts used by compute-as-a-teacher training adapters."""

from .errors import InvalidAnchorAnswerError, RewardContractError, TrainingError
from .rewards import (
    AnchorFailurePolicy,
    DEFAULT_EPSILON,
    DEFAULT_MAX_ANSWER_CHARS,
    DEFAULT_STD_DDOF,
    GROUP_SIZE,
    MathRewardResult,
    OrderedRolloutGroup,
    compute_math_rewards,
    group_normalized_advantages,
    render_anchor_prompt,
)

__all__ = [
    "AnchorFailurePolicy",
    "DEFAULT_EPSILON",
    "DEFAULT_MAX_ANSWER_CHARS",
    "DEFAULT_STD_DDOF",
    "GROUP_SIZE",
    "InvalidAnchorAnswerError",
    "MathRewardResult",
    "OrderedRolloutGroup",
    "RewardContractError",
    "TrainingError",
    "compute_math_rewards",
    "group_normalized_advantages",
    "render_anchor_prompt",
]
