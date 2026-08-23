"""Paper-aligned MATH-500 raw and synthesis evaluation infrastructure."""

from .config import MATH500_PROTOCOL_VERSION
from .errors import EvaluationError
from .grading import MATH_VERIFY_GRADER, PRIMARY_GRADER

__all__ = [
    "EvaluationError",
    "MATH500_PROTOCOL_VERSION",
    "MATH_VERIFY_GRADER",
    "PRIMARY_GRADER",
]
