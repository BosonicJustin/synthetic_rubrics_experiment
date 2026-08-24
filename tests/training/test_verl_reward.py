from __future__ import annotations

import sys
import unittest
from pathlib import Path
from threading import Barrier, Lock
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from compute_as_a_teacher.evaluation.artifacts import sha256_text  # noqa: E402
from compute_as_a_teacher.evaluation.planning import derive_seed  # noqa: E402
from compute_as_a_teacher.training.errors import (  # noqa: E402
    InvalidAnchorAnswerError,
    RewardContractError,
)
from compute_as_a_teacher.training.verl_reward import compute_score  # noqa: E402


class FakeAnchorClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def complete(self, **request: Any) -> str:
        self.calls.append(dict(request))
        message = request["message"]
        if "A_ROLLOUT_" in message:
            return r"A synthesis: \boxed{42}"
        if "B_ROLLOUT_" in message:
            return r"B synthesis: \boxed{7}"
        raise AssertionError("unexpected fake prompt")


class InvalidAnchorClient:
    def complete(self, **request: Any) -> str:
        return "no boxed answer"


class ConcurrentAnchorClient:
    def __init__(self, *, failing_group: str | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.failing_group = failing_group
        self.barrier = Barrier(2)
        self.lock = Lock()

    def complete(self, **request: Any) -> str:
        message = request["message"]
        if "A_ROLLOUT_" in message:
            group, answer = "A", "42"
        elif "B_ROLLOUT_" in message:
            group, answer = "B", "7"
        else:
            raise AssertionError("unexpected fake prompt")
        with self.lock:
            self.calls.append({**request, "group": group})
        self.barrier.wait(timeout=5)
        if group == self.failing_group:
            raise RuntimeError(f"synthetic {group} failure")
        return rf"{group} synthesis: \boxed{{{answer}}}"


def _kwargs(client: Any) -> dict[str, Any]:
    return {
        "repository_root": REPOSITORY_ROOT,
        "prompt_path": "prompts/math500/synthesis_cot_appendix_f_literal.txt",
        "prompt_version": "paper_appendix_f_cot_literal_v1",
        "prompt_prefix": "/no_think\n",
        "anchor_base_url": "http://127.0.0.1:8000/v1",
        "anchor_model": "frozen-anchor",
        "anchor_api_key_env": "",
        "anchor_timeout_seconds": 10,
        "anchor_max_concurrency": 1,
        "anchor_temperature": 0.7,
        "anchor_top_p": 0.8,
        "anchor_top_k": 20,
        "anchor_max_tokens": 1536,
        "base_seed": 8675309,
        "max_answer_chars": 50_000,
        "anchor_failure_policy": "fail_closed",
        "anchor_client": client,
    }


def _two_interleaved_groups() -> tuple[
    list[str],
    list[str],
    list[None],
    list[dict[str, str]],
]:
    a_answers = ("42", "0", "42", "0", "42", "0", "42", "0")
    b_answers = ("7", "7", "8", "8", "7", "8", "8", "7")
    a = [
        rf"A_ROLLOUT_{index} \boxed{{{answer}}}"
        for index, answer in enumerate(a_answers)
    ]
    b = [
        rf"B_ROLLOUT_{index} \boxed{{{answer}}}"
        for index, answer in enumerate(b_answers)
    ]
    # Groups are noncontiguous and begin in a deliberately mixed order.  Each
    # group's encounter order is the local exchangeable rollout order.
    permutation = [
        ("B", 0),
        ("A", 0),
        ("B", 1),
        ("A", 1),
        ("A", 2),
        ("B", 2),
        ("A", 3),
        ("B", 3),
        ("B", 4),
        ("A", 4),
        ("B", 5),
        ("A", 5),
        ("A", 6),
        ("B", 6),
        ("A", 7),
        ("B", 7),
    ]
    solutions = [a[index] if group == "A" else b[index] for group, index in permutation]
    infos = [
        {
            "question_id": f"question-{group.lower()}",
            "problem": f"SECRET_PROBLEM_{group}",
            "gold_answer": f"SECRET_GOLD_{group}",
        }
        for group, _ in permutation
    ]
    return ["math500"] * 16, solutions, [None] * 16, infos


class VerlBatchRewardTests(unittest.TestCase):
    def test_two_shuffled_groups_use_one_anchor_call_and_align_scores(self) -> None:
        data_sources, solutions, ground_truths, extra_infos = _two_interleaved_groups()
        client = FakeAnchorClient()

        result = compute_score(
            data_sources,
            solutions,
            ground_truths,
            extra_infos,
            **_kwargs(client),
        )

        self.assertEqual(len(client.calls), 2)
        self.assertEqual(len(result), len(solutions))
        expected = [
            int(
                ("A_ROLLOUT" in solution and r"\boxed{42}" in solution)
                or ("B_ROLLOUT" in solution and r"\boxed{7}" in solution)
            )
            for solution in solutions
        ]
        self.assertEqual([row["score"] for row in result], expected)

        messages = [call["message"] for call in client.calls]
        for message in messages:
            self.assertEqual(message.count("## RESPONSE "), 8)
            self.assertNotIn("SECRET_PROBLEM", message)
            self.assertNotIn("SECRET_GOLD", message)
            self.assertNotIn("question-a", message)
            self.assertNotIn("question-b", message)
        self.assertIn("A_ROLLOUT_", messages[1])
        self.assertTrue("B_ROLLOUT_" in messages[0])

        expected_seed_b = derive_seed(8675309, "question-b", "anchor", 0)
        expected_seed_a = derive_seed(8675309, "question-a", "anchor", 0)
        self.assertEqual(
            [call["seed"] for call in client.calls],
            [expected_seed_b, expected_seed_a],
        )
        self.assertNotEqual(expected_seed_a, expected_seed_b)

        anchor_hashes = {
            sha256_text(r"A synthesis: \boxed{42}"),
            sha256_text(r"B synthesis: \boxed{7}"),
        }
        for row in result:
            self.assertEqual(row["anchor_extraction_status"], "ok")
            self.assertEqual(row["anchor_failure_policy"], "fail_closed")
            self.assertEqual(row["anchor_finish_reason"], "unknown")
            self.assertEqual(row["rollout_extraction_status"], "ok")
            self.assertIn(row["anchor_response_sha256"], anchor_hashes)
            self.assertRegex(row["anchor_answer_sha256"], r"^[0-9a-f]{64}$")
            self.assertGreaterEqual(row["anchor_latency_seconds"], 0.0)
            self.assertNotIn("anchor_answer", row)
            self.assertNotIn(r"\boxed", str(row))

    def test_concurrent_groups_keep_alignment_and_call_anchor_once_each(self) -> None:
        data_sources, solutions, ground_truths, extra_infos = _two_interleaved_groups()
        client = ConcurrentAnchorClient()
        kwargs = _kwargs(client)
        kwargs["anchor_max_concurrency"] = 32

        result = compute_score(
            data_sources,
            solutions,
            ground_truths,
            extra_infos,
            **kwargs,
        )

        expected = [
            int(
                ("A_ROLLOUT" in solution and r"\boxed{42}" in solution)
                or ("B_ROLLOUT" in solution and r"\boxed{7}" in solution)
            )
            for solution in solutions
        ]
        self.assertEqual([row["score"] for row in result], expected)
        self.assertEqual(len(client.calls), 2)
        self.assertCountEqual([call["group"] for call in client.calls], ["A", "B"])

    def test_concurrent_anchor_failure_returns_no_partial_scores(self) -> None:
        data_sources, solutions, ground_truths, extra_infos = _two_interleaved_groups()
        client = ConcurrentAnchorClient(failing_group="A")
        kwargs = _kwargs(client)
        kwargs["anchor_max_concurrency"] = 32
        result = None

        with self.assertRaisesRegex(
            RewardContractError,
            "anchor client failed for question 'question-a'",
        ):
            result = compute_score(
                data_sources,
                solutions,
                ground_truths,
                extra_infos,
                **kwargs,
            )

        self.assertIsNone(result)
        self.assertEqual(len(client.calls), 2)
        self.assertCountEqual([call["group"] for call in client.calls], ["A", "B"])

    def test_invalid_group_sizes_and_shapes_fail_before_anchor_call(self) -> None:
        data_sources, solutions, ground_truths, extra_infos = _two_interleaved_groups()
        malformed = (
            (
                data_sources[:-1],
                solutions,
                ground_truths,
                extra_infos,
                "identical lengths",
            ),
            (
                data_sources[:-1],
                solutions[:-1],
                ground_truths[:-1],
                extra_infos[:-1],
                "exactly 8",
            ),
            ([], [], [], [], "must not be empty"),
        )
        for sources, outputs, labels, infos, message in malformed:
            with self.subTest(message=message):
                client = FakeAnchorClient()
                with self.assertRaisesRegex(RewardContractError, message):
                    compute_score(
                        sources,
                        outputs,
                        labels,
                        infos,
                        **_kwargs(client),
                    )
                self.assertEqual(client.calls, [])

    def test_gold_labels_are_forbidden_even_if_otherwise_valid(self) -> None:
        data_sources, solutions, ground_truths, extra_infos = _two_interleaved_groups()
        ground_truths[3] = "42"  # type: ignore[list-item]
        client = FakeAnchorClient()
        with self.assertRaisesRegex(RewardContractError, "gold labels are forbidden"):
            compute_score(
                data_sources,
                solutions,
                ground_truths,
                extra_infos,
                **_kwargs(client),
            )
        self.assertEqual(client.calls, [])

    def test_missing_question_id_and_non_text_solution_fail_closed(self) -> None:
        data_sources, solutions, ground_truths, extra_infos = _two_interleaved_groups()
        bad_infos = list(extra_infos)
        bad_infos[0] = {"problem": "must never be used"}
        with self.assertRaisesRegex(RewardContractError, "question_id"):
            compute_score(
                data_sources,
                solutions,
                ground_truths,
                bad_infos,
                **_kwargs(FakeAnchorClient()),
            )

        bad_solutions: list[Any] = list(solutions)
        bad_solutions[0] = 123
        with self.assertRaisesRegex(RewardContractError, "must be text"):
            compute_score(
                data_sources,
                bad_solutions,
                ground_truths,
                extra_infos,
                **_kwargs(FakeAnchorClient()),
            )

    def test_invalid_anchor_answer_uses_explicit_failure_policy(self) -> None:
        data_sources = ["math500"] * 8
        solutions = [rf"rollout {index} \boxed{{42}}" for index in range(8)]
        ground_truths = [None] * 8
        extra_infos = [{"question_id": "question-a"}] * 8
        with self.assertRaises(InvalidAnchorAnswerError):
            compute_score(
                data_sources,
                solutions,
                ground_truths,
                extra_infos,
                **_kwargs(InvalidAnchorClient()),
            )

        kwargs = _kwargs(InvalidAnchorClient())
        kwargs["anchor_failure_policy"] = "reward_zero"
        rows = compute_score(
            data_sources,
            solutions,
            ground_truths,
            extra_infos,
            **kwargs,
        )
        self.assertEqual([row["score"] for row in rows], [0] * 8)
        self.assertTrue(
            all(row["anchor_extraction_status"] == "missing_box" for row in rows)
        )
        self.assertTrue(all(row["anchor_answer_sha256"] is None for row in rows))

    def test_unknown_configured_kwarg_is_rejected_by_batch_hook_signature(self) -> None:
        data_sources = ["math500"] * 8
        solutions = [rf"A_ROLLOUT_{index} \boxed{{42}}" for index in range(8)]
        ground_truths = [None] * 8
        extra_infos = [{"question_id": "question-a"}] * 8
        kwargs = _kwargs(FakeAnchorClient())
        kwargs["gold_answer"] = "42"
        with self.assertRaises(TypeError):
            compute_score(
                data_sources,
                solutions,
                ground_truths,
                extra_infos,
                **kwargs,
            )

    def test_invalid_endpoint_fails_even_with_injected_client(self) -> None:
        client = FakeAnchorClient()
        kwargs = _kwargs(client)
        kwargs["anchor_base_url"] = "not-an-absolute-endpoint"
        with self.assertRaisesRegex(RewardContractError, "absolute HTTP"):
            compute_score(
                ["math500"] * 8,
                [rf"A_ROLLOUT_{index} \boxed{{42}}" for index in range(8)],
                [None] * 8,
                [{"question_id": "question-a"}] * 8,
                **kwargs,
            )
        self.assertEqual(client.calls, [])


if __name__ == "__main__":
    unittest.main()
