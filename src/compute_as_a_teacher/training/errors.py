"""Training-contract exceptions."""

from __future__ import annotations

from typing import Any


class TrainingError(RuntimeError):
    """Raised when a training artifact violates its declared contract."""


class RewardContractError(TrainingError):
    """Raised when a CaT reward group is invalid."""


class InvalidAnchorAnswerError(RewardContractError):
    """Raised when fail-closed reward computation cannot extract the anchor."""

    def __init__(self, status: str) -> None:
        self.status = status
        super().__init__(
            "Cannot compute MATH rewards from synthesized anchor: "
            f"boxed-answer extraction status is {status!r}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {"error": type(self).__name__, "anchor_status": self.status}
