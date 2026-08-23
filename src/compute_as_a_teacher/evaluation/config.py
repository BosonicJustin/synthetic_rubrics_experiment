"""TOML configuration for MATH-500 evaluation stages."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .errors import EvaluationError
from .grading import SUPPORTED_GRADERS
from .schemas import ModelSpec, SamplingSpec


MATH500_PROTOCOL_VERSION = "cat_math500_paper_v1"

def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise EvaluationError(f"{name} must be a TOML table")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise EvaluationError(
            f"{name} keys are {sorted(value)}, expected exactly {sorted(expected)}"
        )


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationError(f"{name} must be a nonempty string")
    return value


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            value = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise EvaluationError(f"Cannot read TOML config {path}: {exc}") from exc
    return value


@dataclass(frozen=True, slots=True)
class PromptSpec:
    path: str
    version: str
    prefix: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PromptSpec":
        _exact_keys(value, {"path", "version", "prefix"}, "prompt")
        prefix = value["prefix"]
        if not isinstance(prefix, str):
            raise EvaluationError("prompt.prefix must be a string")
        return cls(
            path=_text(value["path"], "prompt.path"),
            version=_text(value["version"], "prompt.version"),
            prefix=prefix,
        )

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "version": self.version, "prefix": self.prefix}


@dataclass(frozen=True, slots=True)
class RawEvalConfig:
    schema_version: int
    kind: str
    protocol_version: str
    run_name: str
    questions_path: str
    dataset_lock_path: str
    rollouts_per_problem: int
    prompt: PromptSpec
    model: ModelSpec
    sampling: SamplingSpec

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_version": self.protocol_version,
            "run_name": self.run_name,
            "questions_path": self.questions_path,
            "dataset_lock_path": self.dataset_lock_path,
            "rollouts_per_problem": self.rollouts_per_problem,
            "prompt": self.prompt.to_dict(),
            "model": self.model.to_dict(),
            "sampling": self.sampling.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class SynthesisEvalConfig:
    schema_version: int
    kind: str
    protocol_version: str
    run_name: str
    required_rollouts: int
    require_same_model_as_raw: bool
    prompt: PromptSpec
    anchor: ModelSpec
    sampling: SamplingSpec

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_version": self.protocol_version,
            "run_name": self.run_name,
            "required_rollouts": self.required_rollouts,
            "require_same_model_as_raw": self.require_same_model_as_raw,
            "prompt": self.prompt.to_dict(),
            "anchor": self.anchor.to_dict(),
            "sampling": self.sampling.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ScoringConfig:
    schema_version: int
    kind: str
    protocol_version: str
    labels_path: str
    dataset_lock_path: str
    primary_grader: str
    diagnostic_graders: tuple[str, ...]
    parsing_timeout_seconds: int
    max_answer_chars: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_version": self.protocol_version,
            "labels_path": self.labels_path,
            "dataset_lock_path": self.dataset_lock_path,
            "primary_grader": self.primary_grader,
            "diagnostic_graders": list(self.diagnostic_graders),
            "parsing_timeout_seconds": self.parsing_timeout_seconds,
            "max_answer_chars": self.max_answer_chars,
        }


def load_raw_config(path: Path, *, allow_unresolved_model: bool = False) -> RawEvalConfig:
    value = _load_toml(path)
    expected = {
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
    _exact_keys(value, expected, "raw config")
    if value["schema_version"] != 1 or value["kind"] != "raw":
        raise EvaluationError("Raw config must use schema_version=1 and kind='raw'")
    if type(value["rollouts_per_problem"]) is not int or value["rollouts_per_problem"] != 8:
        raise EvaluationError("The paper-aligned raw plan requires 8 rollouts per problem")
    protocol_version = _text(value["protocol_version"], "protocol_version")
    if protocol_version != MATH500_PROTOCOL_VERSION:
        raise EvaluationError(
            f"protocol_version must be {MATH500_PROTOCOL_VERSION!r}"
        )
    return RawEvalConfig(
        schema_version=1,
        kind="raw",
        protocol_version=protocol_version,
        run_name=_text(value["run_name"], "run_name"),
        questions_path=_text(value["questions_path"], "questions_path"),
        dataset_lock_path=_text(value["dataset_lock_path"], "dataset_lock_path"),
        rollouts_per_problem=8,
        prompt=PromptSpec.from_dict(_mapping(value["prompt"], "prompt")),
        model=ModelSpec.from_dict(
            _mapping(value["model"], "model"),
            allow_unresolved=allow_unresolved_model,
        ),
        sampling=SamplingSpec.from_dict(_mapping(value["sampling"], "sampling")),
    )


def load_synthesis_config(
    path: Path,
    *,
    allow_unresolved_model: bool = False,
) -> SynthesisEvalConfig:
    value = _load_toml(path)
    expected = {
        "schema_version",
        "kind",
        "protocol_version",
        "run_name",
        "required_rollouts",
        "require_same_model_as_raw",
        "prompt",
        "anchor",
        "sampling",
    }
    _exact_keys(value, expected, "synthesis config")
    if value["schema_version"] != 1 or value["kind"] != "synthesis":
        raise EvaluationError(
            "Synthesis config must use schema_version=1 and kind='synthesis'"
        )
    if type(value["required_rollouts"]) is not int or value["required_rollouts"] != 8:
        raise EvaluationError("The paper-aligned synthesis plan requires exactly 8 rollouts")
    if not isinstance(value["require_same_model_as_raw"], bool):
        raise EvaluationError("require_same_model_as_raw must be a boolean")
    if value["require_same_model_as_raw"] is not True:
        raise EvaluationError(
            "The paper-aligned protocol requires the frozen raw policy as synthesis anchor"
        )
    protocol_version = _text(value["protocol_version"], "protocol_version")
    if protocol_version != MATH500_PROTOCOL_VERSION:
        raise EvaluationError(
            f"protocol_version must be {MATH500_PROTOCOL_VERSION!r}"
        )
    return SynthesisEvalConfig(
        schema_version=1,
        kind="synthesis",
        protocol_version=protocol_version,
        run_name=_text(value["run_name"], "run_name"),
        required_rollouts=8,
        require_same_model_as_raw=value["require_same_model_as_raw"],
        prompt=PromptSpec.from_dict(_mapping(value["prompt"], "prompt")),
        anchor=ModelSpec.from_dict(
            _mapping(value["anchor"], "anchor"),
            allow_unresolved=allow_unresolved_model,
        ),
        sampling=SamplingSpec.from_dict(_mapping(value["sampling"], "sampling")),
    )


def load_scoring_config(path: Path) -> ScoringConfig:
    value = _load_toml(path)
    expected = {
        "schema_version",
        "kind",
        "protocol_version",
        "labels_path",
        "dataset_lock_path",
        "primary_grader",
        "diagnostic_graders",
        "parsing_timeout_seconds",
        "max_answer_chars",
    }
    _exact_keys(value, expected, "scoring config")
    if value["schema_version"] != 1 or value["kind"] != "scoring":
        raise EvaluationError("Scoring config must use schema_version=1 and kind='scoring'")
    diagnostics = value["diagnostic_graders"]
    if not isinstance(diagnostics, list) or not all(
        isinstance(item, str) and item for item in diagnostics
    ):
        raise EvaluationError("diagnostic_graders must be a list of names")
    if len(set(diagnostics)) != len(diagnostics):
        raise EvaluationError("diagnostic_graders must not contain duplicates")
    timeout = value["parsing_timeout_seconds"]
    max_chars = value["max_answer_chars"]
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
        raise EvaluationError("parsing_timeout_seconds must be positive")
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars <= 0:
        raise EvaluationError("max_answer_chars must be positive")
    primary = _text(value["primary_grader"], "primary_grader")
    if primary in diagnostics:
        raise EvaluationError("primary_grader must not be repeated in diagnostic_graders")
    unknown_graders = {primary, *diagnostics} - SUPPORTED_GRADERS
    if unknown_graders:
        raise EvaluationError(f"Unknown graders: {sorted(unknown_graders)}")
    protocol_version = _text(value["protocol_version"], "protocol_version")
    if protocol_version != MATH500_PROTOCOL_VERSION:
        raise EvaluationError(
            f"protocol_version must be {MATH500_PROTOCOL_VERSION!r}"
        )
    return ScoringConfig(
        schema_version=1,
        kind="scoring",
        protocol_version=protocol_version,
        labels_path=_text(value["labels_path"], "labels_path"),
        dataset_lock_path=_text(value["dataset_lock_path"], "dataset_lock_path"),
        primary_grader=primary,
        diagnostic_graders=tuple(diagnostics),
        parsing_timeout_seconds=timeout,
        max_answer_chars=max_chars,
    )
