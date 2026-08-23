"""Versioned prompt loading and paper-aligned rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .artifacts import sha256_text
from .config import PromptSpec
from .errors import EvaluationError


RAW_PLACEHOLDER = "{problem}"
SYNTHESIS_PLACEHOLDER = "{rollouts}"
REQUIRED_SYNTHESIS_ROLLOUTS = 8

# Hashes turn the human-readable version names into immutable prompt contracts.
# The literal Appendix F text is canonical. The one-character boxfix remains
# registered for an explicitly separate sensitivity protocol.
PROMPT_REGISTRY = {
    "raw_math500_local_v1": {
        "path": "prompts/math500/solve_v1.txt",
        "sha256": "1fe83bb37db7308ceaccd1fc407f3eefc7038fef511fb8c96b9b1f30266a27c4",
    },
    "paper_appendix_f_cot_boxfix_v1": {
        "path": "prompts/math500/synthesis_cot_v1.txt",
        "sha256": "24f46cf2b510e589acffda4094ad16a586f8287486ebf983a41f0649e4e80a8c",
    },
    "paper_appendix_f_cot_literal_v1": {
        "path": "prompts/math500/synthesis_cot_appendix_f_literal.txt",
        "sha256": "c1efb453ae280787445a907b97bdc2ff8b2094c20788686cc6c27f27c289374e",
    },
}


def validate_prompt_template(template: str, spec: PromptSpec) -> None:
    registered = PROMPT_REGISTRY.get(spec.version)
    if registered is None:
        raise EvaluationError(f"Unknown prompt version: {spec.version}")
    if spec.path != registered["path"]:
        raise EvaluationError(
            f"Prompt version {spec.version} must use {registered['path']}"
        )
    actual_sha256 = sha256_text(template)
    if actual_sha256 != registered["sha256"]:
        raise EvaluationError(
            f"Prompt {spec.version} does not match its registered SHA-256"
        )


def load_prompt(repository_root: Path, spec: PromptSpec) -> str:
    path = (repository_root / spec.path).resolve()
    try:
        path.relative_to(repository_root.resolve())
    except ValueError as exc:
        raise EvaluationError(f"Prompt path escapes repository root: {spec.path}") from exc
    try:
        template = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EvaluationError(f"Cannot read prompt {path}: {exc}") from exc
    validate_prompt_template(template, spec)
    return template


def prompt_contract_sha256(template: str, spec: PromptSpec) -> str:
    return sha256_text(f"version={spec.version}\nprefix={spec.prefix}\n{template}")


def render_raw_prompt(template: str, spec: PromptSpec, problem: str) -> str:
    if template.count(RAW_PLACEHOLDER) != 1:
        raise EvaluationError("Raw prompt must contain exactly one {problem} placeholder")
    if SYNTHESIS_PLACEHOLDER in template:
        raise EvaluationError("Raw prompt must not contain {rollouts}")
    return spec.prefix + template.replace(RAW_PLACEHOLDER, problem)


def serialize_rollouts(rollouts: Sequence[str]) -> str:
    if len(rollouts) != REQUIRED_SYNTHESIS_ROLLOUTS:
        raise EvaluationError(
            f"Synthesis requires exactly {REQUIRED_SYNTHESIS_ROLLOUTS} rollouts"
        )
    if not all(isinstance(text, str) for text in rollouts):
        raise EvaluationError("Every synthesis rollout must be text")
    blocks = [f"## RESPONSE {index}\n{text}" for index, text in enumerate(rollouts, start=1)]
    return "\n\n".join(blocks)


def render_synthesis_prompt(
    template: str,
    spec: PromptSpec,
    rollouts: Sequence[str],
) -> str:
    if template.count(SYNTHESIS_PLACEHOLDER) != 1:
        raise EvaluationError(
            "Synthesis prompt must contain exactly one {rollouts} placeholder"
        )
    if RAW_PLACEHOLDER in template:
        raise EvaluationError(
            "Synthesis prompt must not contain {problem}; the paper omits the question"
        )
    return spec.prefix + template.replace(SYNTHESIS_PLACEHOLDER, serialize_rollouts(rollouts))
