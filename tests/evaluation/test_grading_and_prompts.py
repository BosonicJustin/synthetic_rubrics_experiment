from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from compute_as_a_teacher.evaluation.config import (  # noqa: E402
    PromptSpec,
    load_raw_config,
    load_synthesis_config,
)
from compute_as_a_teacher.evaluation.errors import EvaluationError  # noqa: E402
from compute_as_a_teacher.evaluation.grading import (  # noqa: E402
    MATH_VERIFY_GRADER,
    PRIMARY_GRADER,
    extract_last_boxed,
    grade_response,
)
from compute_as_a_teacher.evaluation.prompts import (  # noqa: E402
    load_prompt,
    render_raw_prompt,
    render_synthesis_prompt,
)


class BoxedAnswerTests(unittest.TestCase):
    def test_nested_braces_and_last_box_win(self) -> None:
        result = extract_last_boxed(
            r"First \boxed{wrong}; finally \boxed{\frac{1}{2}}."
        )
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.value, r"\frac{1}{2}")

    def test_escaped_braces_do_not_change_group_depth(self) -> None:
        result = extract_last_boxed(r"Answer: \boxed{\text{\{}}}")
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.value, r"\text{\{}")

    def test_odd_and_even_backslash_runs_control_brace_escaping(self) -> None:
        for run_length in (1, 3, 10_001):
            with self.subTest(parity="odd", run_length=run_length):
                slashes = "\\" * run_length
                result = extract_last_boxed(
                    "prefix \\boxed{x" + slashes + "}y}",
                    max_answer_chars=20_000,
                )
                self.assertEqual(result.status, "ok")
                self.assertEqual(result.value, "x" + slashes + "}y")

        for run_length in (2, 4, 10_000):
            with self.subTest(parity="even", run_length=run_length):
                slashes = "\\" * run_length
                result = extract_last_boxed(
                    "prefix \\boxed{x" + slashes + "}ignored",
                    max_answer_chars=20_000,
                )
                self.assertEqual(result.status, "ok")
                self.assertEqual(result.value, "x" + slashes)

    def test_valid_box_survives_later_bare_command_mention(self) -> None:
        result = extract_last_boxed(r"\boxed{42}; remember to use \boxed")
        self.assertEqual(result.value, "42")
        result = extract_last_boxed(r"\boxed{42}; not \boxedness{7}")
        self.assertEqual(result.value, "42")

    def test_whitespace_missing_malformed_empty_and_length_limit(self) -> None:
        self.assertEqual(extract_last_boxed(r"\boxed {42}").value, "42")
        self.assertEqual(extract_last_boxed("no final answer").status, "missing_box")
        self.assertEqual(extract_last_boxed(r"\boxed{42").status, "malformed_box")
        self.assertEqual(extract_last_boxed(r"\boxed{  }").status, "empty_box")
        self.assertEqual(
            extract_last_boxed(r"\boxed{123}", max_answer_chars=2).status,
            "too_long",
        )

    def test_length_limit_is_early_but_preserves_outer_whitespace(self) -> None:
        self.assertEqual(
            extract_last_boxed(r"\boxed{   123   }", max_answer_chars=3).value,
            "123",
        )
        self.assertEqual(
            extract_last_boxed(r"\boxed{a  b}", max_answer_chars=3).status,
            "too_long",
        )
        # The limit is irreversible after the sixth retained character, so an
        # unclosed adversarial response is rejected without scanning for a brace.
        self.assertEqual(
            extract_last_boxed("\\boxed{" + ("x" * 100_000), max_answer_chars=5).status,
            "too_long",
        )

    def test_strict_grader_is_exact_after_outer_whitespace_only(self) -> None:
        exact = grade_response(
            r"Therefore \boxed{ \frac{1}{2} }",
            r"\frac{1}{2}",
            graders=(PRIMARY_GRADER,),
            timeout_seconds=5,
            max_answer_chars=100,
        )
        different = grade_response(
            r"Therefore \boxed{0.5}",
            r"\frac{1}{2}",
            graders=(PRIMARY_GRADER,),
            timeout_seconds=5,
            max_answer_chars=100,
        )
        self.assertTrue(exact["grades"][PRIMARY_GRADER]["correct"])
        self.assertFalse(different["grades"][PRIMARY_GRADER]["correct"])

    @unittest.skipUnless(
        importlib.util.find_spec("math_verify") is not None,
        "Install the optional evaluation extra for math-verify diagnostics",
    )
    def test_math_verify_is_a_separate_equivalence_diagnostic(self) -> None:
        result = grade_response(
            r"Therefore \boxed{0.5}",
            r"\frac{1}{2}",
            graders=(PRIMARY_GRADER, MATH_VERIFY_GRADER),
            timeout_seconds=5,
            max_answer_chars=100,
        )
        self.assertFalse(result["grades"][PRIMARY_GRADER]["correct"])
        self.assertTrue(result["grades"][MATH_VERIFY_GRADER]["correct"])


class PromptContractTests(unittest.TestCase):
    def test_registered_raw_prompt_is_a_versioned_local_choice(self) -> None:
        spec = PromptSpec(
            path="prompts/math500/solve_v1.txt",
            version="raw_math500_local_v1",
            prefix="/no_think\n",
        )
        template = load_prompt(REPOSITORY_ROOT, spec)
        rendered = render_raw_prompt(template, spec, "QUESTION_SENTINEL")
        self.assertTrue(rendered.startswith("/no_think\nQUESTION_SENTINEL"))
        self.assertIn(r"\boxed{}", rendered)

    def test_appendix_f_synthesis_prompt_uses_only_eight_ordered_rollouts(self) -> None:
        spec = PromptSpec(
            path="prompts/math500/synthesis_cot_v1.txt",
            version="paper_appendix_f_cot_boxfix_v1",
            prefix="/no_think\n",
        )
        template = load_prompt(REPOSITORY_ROOT, spec)
        rollouts = [f"ROLLOUT_{index}" for index in range(8)]
        rendered = render_synthesis_prompt(template, spec, rollouts)
        self.assertTrue(rendered.startswith("/no_think\n"))
        self.assertIn("# SUMMARY", rendered)
        self.assertIn("# UNIFIED RESPONSE", rendered)
        self.assertIn(r"\boxed{answer}", rendered)
        self.assertNotIn("QUESTION_SENTINEL", rendered)
        positions = [rendered.index(f"ROLLOUT_{index}") for index in range(8)]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(rendered.count("## RESPONSE "), 8)

    def test_synthesis_prompt_rejects_any_non_eight_group(self) -> None:
        spec = PromptSpec(
            path="prompts/math500/synthesis_cot_v1.txt",
            version="paper_appendix_f_cot_boxfix_v1",
            prefix="",
        )
        template = load_prompt(REPOSITORY_ROOT, spec)
        with self.assertRaisesRegex(EvaluationError, "exactly 8"):
            render_synthesis_prompt(template, spec, ["x"] * 7)


class ConfigurationContractTests(unittest.TestCase):
    def test_example_configs_are_lintable_but_not_runnable_until_pinned(self) -> None:
        raw_path = REPOSITORY_ROOT / "configs/evals/math500_raw.example.toml"
        synthesis_path = (
            REPOSITORY_ROOT / "configs/evals/math500_synthesis.example.toml"
        )
        raw = load_raw_config(raw_path, allow_unresolved_model=True)
        synthesis = load_synthesis_config(
            synthesis_path,
            allow_unresolved_model=True,
        )
        self.assertTrue(raw.model.unresolved_reasons())
        self.assertTrue(synthesis.anchor.unresolved_reasons())
        with self.assertRaisesRegex(EvaluationError, "not runnable"):
            load_raw_config(raw_path)
        with self.assertRaisesRegex(EvaluationError, "not runnable"):
            load_synthesis_config(synthesis_path)

    def test_same_anchor_toggle_cannot_disable_the_paper_contract(self) -> None:
        source = (
            REPOSITORY_ROOT / "configs/evals/math500_synthesis.example.toml"
        ).read_text(encoding="utf-8")
        source = source.replace(
            "require_same_model_as_raw = true",
            "require_same_model_as_raw = false",
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "bad.toml"
            path.write_text(source, encoding="utf-8")
            with self.assertRaisesRegex(EvaluationError, "frozen raw policy"):
                load_synthesis_config(path, allow_unresolved_model=True)

    def test_rollout_count_must_be_an_integer_eight(self) -> None:
        source = (
            REPOSITORY_ROOT / "configs/evals/math500_raw.example.toml"
        ).read_text(encoding="utf-8")
        source = source.replace("rollouts_per_problem = 8", "rollouts_per_problem = 8.0")
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "bad.toml"
            path.write_text(source, encoding="utf-8")
            with self.assertRaisesRegex(EvaluationError, "requires 8"):
                load_raw_config(path, allow_unresolved_model=True)


if __name__ == "__main__":
    unittest.main()
