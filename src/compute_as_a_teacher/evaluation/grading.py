"""Strict boxed-answer extraction and versioned math graders."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Literal, Sequence

from .errors import EvaluationError


PRIMARY_GRADER = "last_boxed_string_exact_v1"
MATH_VERIFY_GRADER = "math_verify_v0.9.0"
SUPPORTED_GRADERS = frozenset({PRIMARY_GRADER, MATH_VERIFY_GRADER})

ExtractionStatus = Literal["ok", "missing_box", "malformed_box", "empty_box", "too_long"]


@dataclass(frozen=True, slots=True)
class ExtractedAnswer:
    value: str | None
    status: ExtractionStatus

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "status": self.status}


@dataclass(frozen=True, slots=True)
class GradeResult:
    correct: bool
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {"correct": self.correct, "status": self.status}


def extract_last_boxed(text: str, *, max_answer_chars: int = 50_000) -> ExtractedAnswer:
    r"""Extract the final syntactically opened ``\boxed{...}`` expression."""

    if (
        isinstance(max_answer_chars, bool)
        or not isinstance(max_answer_chars, int)
        or max_answer_chars <= 0
    ):
        raise EvaluationError("max_answer_chars must be a positive integer")

    marker = "\\boxed"
    last_marker: tuple[int, int] | None = None
    search_start = 0
    while True:
        marker_start = text.find(marker, search_start)
        if marker_start < 0:
            break
        cursor = marker_start + len(marker)
        # A following letter would make this a different LaTeX command, e.g.
        # ``\boxedness``. Whitespace before the opening brace is accepted.
        if cursor >= len(text) or not text[cursor].isalpha():
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            if cursor < len(text) and text[cursor] == "{":
                last_marker = (marker_start, cursor)
        search_start = marker_start + len(marker)
    if last_marker is None:
        return ExtractedAnswer(None, "missing_box")

    _, cursor = last_marker
    content_start = cursor + 1
    depth = 1
    cursor = content_start
    consecutive_backslashes = 0
    first_non_whitespace: int | None = None
    last_non_whitespace_end: int | None = None
    while cursor < len(text) and depth:
        character = text[cursor]
        escaped = consecutive_backslashes % 2 == 1
        if character == "{" and not escaped:
            depth += 1
        elif character == "}" and not escaped:
            depth -= 1

        # The outer closing brace is syntax rather than answer content. For all
        # other characters, the span from the first through the current
        # non-whitespace character is exactly the length that ``strip()`` would
        # eventually retain. Once that span exceeds the limit it cannot shrink,
        # so reject before scanning or allocating the rest of an adversarial
        # response. Leading and trailing whitespace remain free, as before.
        if depth and not character.isspace():
            if first_non_whitespace is None:
                first_non_whitespace = cursor
            last_non_whitespace_end = cursor + 1
            if last_non_whitespace_end - first_non_whitespace > max_answer_chars:
                return ExtractedAnswer(None, "too_long")

        if character == "\\":
            consecutive_backslashes += 1
        else:
            consecutive_backslashes = 0
        cursor += 1
    if depth:
        return ExtractedAnswer(None, "malformed_box")

    if first_non_whitespace is None or last_non_whitespace_end is None:
        return ExtractedAnswer(None, "empty_box")
    value = text[first_non_whitespace:last_non_whitespace_end]
    return ExtractedAnswer(value, "ok")


def _grade_exact(extracted: ExtractedAnswer, reference: str) -> GradeResult:
    if extracted.status != "ok" or extracted.value is None:
        return GradeResult(False, extracted.status)
    correct = extracted.value.strip() == reference.strip()
    return GradeResult(correct, "correct" if correct else "incorrect")


def _grade_math_verify(
    extracted: ExtractedAnswer,
    reference: str,
    *,
    timeout_seconds: int,
) -> GradeResult:
    if extracted.status != "ok" or extracted.value is None:
        return GradeResult(False, extracted.status)
    parse, verify, MathVerifyTimeout = _math_verify_api()

    try:
        parsed_reference = _parse_math_verify_reference(reference, timeout_seconds)
    except (Exception, MathVerifyTimeout) as exc:
        raise EvaluationError(
            f"Reference answer could not be parsed by {MATH_VERIFY_GRADER}: {reference!r}"
        ) from exc
    if not parsed_reference:
        raise EvaluationError(
            f"Reference answer produced no parse under {MATH_VERIFY_GRADER}: {reference!r}"
        )

    try:
        parsed_prediction = parse(
            f"\\boxed{{{extracted.value}}}",
            parsing_timeout=timeout_seconds,
        )
        if not parsed_prediction:
            return GradeResult(False, "unparseable")
    except MathVerifyTimeout:
        return GradeResult(False, "timeout")
    except Exception:
        return GradeResult(False, "unparseable")
    try:
        correct = bool(
            verify(
                parsed_reference,
                parsed_prediction,
                strict=True,
                timeout_seconds=timeout_seconds,
            )
        )
    except MathVerifyTimeout:
        return GradeResult(False, "timeout")
    except Exception:
        return GradeResult(False, "unparseable")
    return GradeResult(correct, "correct" if correct else "incorrect")


@lru_cache(maxsize=1)
def _math_verify_api() -> tuple[Any, Any, Any]:
    try:
        installed_version = version("math-verify")
        if installed_version != "0.9.0":
            raise EvaluationError(
                f"{MATH_VERIFY_GRADER} requires math-verify==0.9.0, "
                f"found {installed_version}"
            )
        from math_verify import parse, verify
        from math_verify.errors import TimeoutException as MathVerifyTimeout
    except (ImportError, PackageNotFoundError) as exc:
        raise EvaluationError(
            "math_verify_v0.9.0 was requested but is not installed; "
            "run `uv sync --extra evaluation --frozen`"
        ) from exc

    return parse, verify, MathVerifyTimeout


@lru_cache(maxsize=2048)
def _parse_math_verify_reference(reference: str, timeout_seconds: int) -> Any:
    parse, _, _ = _math_verify_api()
    return parse(
        f"\\boxed{{{reference}}}",
        parsing_timeout=timeout_seconds,
    )


def grade_response(
    response: str,
    reference: str,
    *,
    graders: Sequence[str],
    timeout_seconds: int,
    max_answer_chars: int,
) -> dict[str, Any]:
    if not graders:
        raise EvaluationError("At least one grader is required")
    unknown = set(graders) - SUPPORTED_GRADERS
    if unknown:
        raise EvaluationError(f"Unknown graders: {sorted(unknown)}")
    extracted = extract_last_boxed(response, max_answer_chars=max_answer_chars)
    grades: dict[str, Any] = {}
    for grader in graders:
        if grader == PRIMARY_GRADER:
            result = _grade_exact(extracted, reference)
        else:
            result = _grade_math_verify(
                extracted,
                reference,
                timeout_seconds=timeout_seconds,
            )
        grades[grader] = result.to_dict()
    return {"extraction": extracted.to_dict(), "grades": grades}
