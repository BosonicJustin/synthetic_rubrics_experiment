from __future__ import annotations

import inspect
import math
import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from compute_as_a_teacher.evaluation.config import PromptSpec  # noqa: E402
from compute_as_a_teacher.training.errors import (  # noqa: E402
    InvalidAnchorAnswerError,
    RewardContractError,
)
from compute_as_a_teacher.training.rewards import (  # noqa: E402
    DEFAULT_EPSILON,
    DEFAULT_STD_DDOF,
    OrderedRolloutGroup,
    compute_math_rewards,
    group_normalized_advantages,
    render_anchor_prompt,
)


def _boxed_group(answer: str = "42") -> list[str]:
    return [rf"rollout {index}: \boxed{{{answer}}}" for index in range(1, 9)]


class OrderedRolloutGroupTests(unittest.TestCase):
    def test_group_requires_exactly_eight_ordered_texts(self) -> None:
        rollouts = [f"ROLL_{index}" for index in range(1, 9)]
        group = OrderedRolloutGroup.from_sequence(rollouts)
        self.assertEqual(group.rollouts, tuple(rollouts))
        self.assertEqual(
            [entry["position"] for entry in group.to_dict()["rollouts"]],
            list(range(1, 9)),
        )
        with self.assertRaisesRegex(RewardContractError, "exactly 8"):
            OrderedRolloutGroup.from_sequence(rollouts[:7])
        with self.assertRaisesRegex(RewardContractError, "exactly 8"):
            OrderedRolloutGroup.from_sequence(rollouts + ["ROLL_9"])
        with self.assertRaisesRegex(RewardContractError, "ordered sequence"):
            OrderedRolloutGroup.from_sequence("not-a-group")
        with self.assertRaisesRegex(RewardContractError, "rollout 4"):
            OrderedRolloutGroup.from_sequence(rollouts[:3] + [4] + rollouts[4:])

    def test_anchor_prompt_reuses_question_free_ordered_renderer(self) -> None:
        rollouts = [f"ROLL_{index}" for index in range(1, 9)]
        prompt = PromptSpec(path="unused.txt", version="test", prefix="PREFIX\n")
        rendered = render_anchor_prompt("{rollouts}", prompt, rollouts)
        self.assertTrue(rendered.startswith("PREFIX\n## RESPONSE 1\nROLL_1"))
        positions = [rendered.index(rollout) for rollout in rollouts]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(rendered.count("## RESPONSE "), 8)
        with self.assertRaisesRegex(Exception, "must not contain.*problem"):
            render_anchor_prompt("{problem}\n{rollouts}", prompt, rollouts)


class MathRewardTests(unittest.TestCase):
    def test_exact_agreement_nested_boxes_and_last_box_win(self) -> None:
        rollouts = _boxed_group("wrong")
        rollouts[0] = r"draft \boxed{wrong}; final \boxed{\frac{1}{2}}"
        rollouts[1] = r"final \boxed{\frac{1}{2}}"
        rollouts[2] = r"final \boxed{0.5}"
        result = compute_math_rewards(
            rollouts,
            r"summary \boxed{wrong}; unified \boxed{\frac{1}{2}}",
        )
        self.assertEqual(result.anchor_status, "ok")
        self.assertEqual(result.rewards, (1, 1, 0, 0, 0, 0, 0, 0))
        self.assertEqual(result.rollout_statuses, ("ok",) * 8)

    def test_rollout_extraction_failures_are_visible_and_reward_zero(self) -> None:
        rollouts = _boxed_group()
        rollouts[0] = "no box"
        rollouts[1] = r"\boxed{42"
        rollouts[2] = r"\boxed{ }"
        rollouts[3] = r"\boxed{123}"
        result = compute_math_rewards(
            rollouts,
            r"\boxed{42}",
            max_answer_chars=2,
        )
        self.assertEqual(
            result.rollout_statuses[:4],
            ("missing_box", "malformed_box", "empty_box", "too_long"),
        )
        self.assertEqual(result.rewards, (0, 0, 0, 0, 1, 1, 1, 1))

    def test_invalid_anchor_is_fail_closed_by_default(self) -> None:
        invalid = {
            "missing_box": "no box",
            "malformed_box": r"\boxed{42",
            "empty_box": r"\boxed{ }",
            "too_long": r"\boxed{123}",
        }
        for expected_status, synthesized in invalid.items():
            with self.subTest(expected_status=expected_status):
                kwargs = {"max_answer_chars": 2} if expected_status == "too_long" else {}
                with self.assertRaises(InvalidAnchorAnswerError) as caught:
                    compute_math_rewards(_boxed_group(), synthesized, **kwargs)
                self.assertEqual(caught.exception.status, expected_status)

    def test_invalid_anchor_reward_zero_policy_is_explicit_and_auditable(self) -> None:
        result = compute_math_rewards(
            _boxed_group(),
            "no box",
            anchor_failure_policy="reward_zero",
        )
        self.assertEqual(result.anchor_status, "missing_box")
        self.assertEqual(result.anchor_failure_policy, "reward_zero")
        self.assertEqual(result.rewards, (0,) * 8)
        with self.assertRaisesRegex(RewardContractError, "fail_closed.*reward_zero"):
            compute_math_rewards(
                _boxed_group(),
                r"\boxed{42}",
                anchor_failure_policy="ignore",  # type: ignore[arg-type]
            )

    def test_api_has_no_problem_or_gold_label_input(self) -> None:
        parameters = inspect.signature(compute_math_rewards).parameters
        self.assertNotIn("problem", parameters)
        self.assertNotIn("question", parameters)
        self.assertNotIn("label", parameters)
        self.assertNotIn("reference", parameters)
        with self.assertRaises(TypeError):
            compute_math_rewards(  # type: ignore[call-arg]
                _boxed_group(),
                r"\boxed{42}",
                reference_answer="42",
            )

    def test_input_contracts_reject_ambiguous_values(self) -> None:
        with self.assertRaisesRegex(RewardContractError, "synthesized must be text"):
            compute_math_rewards(_boxed_group(), 42)  # type: ignore[arg-type]
        with self.assertRaisesRegex(RewardContractError, "positive integer"):
            compute_math_rewards(_boxed_group(), r"\boxed{42}", max_answer_chars=True)
        with self.assertRaisesRegex(RewardContractError, "finite positive"):
            group_normalized_advantages((0,) * 8, epsilon=float("nan"))


class AdvantageTests(unittest.TestCase):
    def test_verl_sample_std_and_epsilon_are_used(self) -> None:
        rewards = (1, 0, 0, 0, 0, 0, 0, 0)
        advantages = group_normalized_advantages(rewards)
        mean = 1 / 8
        sample_variance = (
            (1 - mean) ** 2 + 7 * (0 - mean) ** 2
        ) / (8 - DEFAULT_STD_DDOF)
        denominator = math.sqrt(sample_variance) + DEFAULT_EPSILON
        self.assertAlmostEqual(advantages[0], (1 - mean) / denominator)
        for advantage in advantages[1:]:
            self.assertAlmostEqual(advantage, (0 - mean) / denominator)
        self.assertAlmostEqual(sum(advantages), 0.0)

    def test_zero_variance_produces_exact_zero_advantages(self) -> None:
        self.assertEqual(group_normalized_advantages((0,) * 8), (0.0,) * 8)
        self.assertEqual(group_normalized_advantages((1,) * 8), (0.0,) * 8)

    def test_normalization_requires_eight_binary_numeric_rewards(self) -> None:
        with self.assertRaisesRegex(RewardContractError, "exactly 8"):
            group_normalized_advantages((0,) * 7)
        with self.assertRaisesRegex(RewardContractError, "numeric and binary"):
            group_normalized_advantages((0, 0, 0, 0, 0, 0, 0, 2))
        with self.assertRaisesRegex(RewardContractError, "numeric and binary"):
            group_normalized_advantages((0, 0, 0, 0, 0, 0, 0, True))
        with self.assertRaisesRegex(RewardContractError, "std_ddof"):
            group_normalized_advantages((0,) * 8, std_ddof=8)


if __name__ == "__main__":
    unittest.main()
