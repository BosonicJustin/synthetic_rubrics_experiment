from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from compute_as_a_teacher.data.math500 import QuestionRecord  # noqa: E402
from compute_as_a_teacher.evaluation.prompts import load_prompt  # noqa: E402
from compute_as_a_teacher.training.config import (  # noqa: E402
    SUPPORTED_VERL_REVISION,
    TrainingError,
    load_training_config,
)
from compute_as_a_teacher.training.planning import (  # noqa: E402
    build_training_rows,
    write_training_plan,
)
from compute_as_a_teacher.training.verl_adapter import (  # noqa: E402
    build_verl_command,
    checkpointed_step,
    exclusive_launch,
)


EXAMPLE = REPOSITORY_ROOT / "configs/training/math500_cat_grpo.example.toml"


def resolved_config_text() -> str:
    return (
        EXAMPLE.read_text(encoding="utf-8")
        .replace("required_full_model_commit_sha", "a" * 40)
        .replace("required_full_tokenizer_commit_sha", "b" * 40)
        .replace("required_chat_template_sha256", "c" * 64)
        .replace("required_absolute_python_from_verl_environment", "/opt/verl/bin/python")
        .replace("required_absolute_verl_checkout", "/opt/verl/source")
        .replace("required_absolute_local_model_snapshot", "/models/qwen3-4b")
        .replace("required_model_snapshot_tree_sha256", "d" * 64)
        .replace("required_trainer_image_digest", "sha256:" + "e" * 64)
        .replace("required_target_package_inventory_sha256", "f" * 64)
    )


def load_text(text: str, *, allow_unresolved: bool = False):
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "config.toml"
        path.write_text(text, encoding="utf-8")
        return load_training_config(path, allow_unresolved=allow_unresolved)


class TrainingConfigTests(unittest.TestCase):
    def test_example_is_paper_profile_but_intentionally_unresolved(self) -> None:
        config = load_training_config(EXAMPLE, allow_unresolved=True)
        self.assertEqual(config.runtime.framework_revision, SUPPORTED_VERL_REVISION)
        self.assertEqual(config.grpo.global_batch_size, 256)
        self.assertEqual(config.rollouts.group_size, 8)
        self.assertEqual(config.grpo.max_steps, 1000)
        self.assertTrue(config.unresolved_reasons())

    def test_resolved_config_is_runnable_without_loading_paths(self) -> None:
        config = load_text(resolved_config_text())
        self.assertFalse(config.unresolved_reasons())
        self.assertEqual(config.policy, config.anchor.model)

    def test_label_fields_and_protocol_changes_fail_closed(self) -> None:
        with self.assertRaisesRegex(TrainingError, "evaluation-only.*labels_path"):
            load_text(resolved_config_text() + '\nlabels_path = "data/math500/labels.jsonl"\n')
        with self.assertRaisesRegex(TrainingError, "rollouts must be 8"):
            load_text(resolved_config_text().replace("group_size = 8", "group_size = 7"))
        with self.assertRaisesRegex(TrainingError, "anchor.model must exactly equal"):
            text = resolved_config_text().replace("revision = \"a" + "a" * 39 + "\"", "revision = \"d" + "d" * 39 + "\"", 1)
            load_text(text)

    def test_ignored_or_location_dependent_settings_fail_closed(self) -> None:
        with self.assertRaisesRegex(TrainingError, "absolute path"):
            load_text(resolved_config_text().replace("/opt/verl/bin/python", "python"))
        with self.assertRaisesRegex(TrainingError, "AdamW defaults"):
            load_text(resolved_config_text().replace("epsilon = 1e-8", "epsilon = 2e-8"))
        with self.assertRaisesRegex(TrainingError, "model_snapshot_tree_sha256"):
            load_text(resolved_config_text().replace("d" * 64, "not-a-digest"))
        with self.assertRaisesRegex(TrainingError, "trainer_image_digest"):
            load_text(
                resolved_config_text().replace(
                    "sha256:" + "e" * 64,
                    "image:latest",
                )
            )

    def test_policy_precision_matches_the_training_runtime(self) -> None:
        with self.assertRaisesRegex(TrainingError, "policy.dtype must be 'bfloat16'"):
            load_text(
                resolved_config_text().replace(
                    'dtype = "bfloat16"',
                    'dtype = "float16"',
                    2,
                )
            )
        with self.assertRaisesRegex(TrainingError, "policy.quantization must be 'none'"):
            load_text(
                resolved_config_text().replace(
                    'quantization = "none"',
                    'quantization = "awq"',
                    2,
                )
            )
        with self.assertRaisesRegex(TrainingError, "policy.dtype must equal runtime.dtype"):
            load_text(
                resolved_config_text().replace(
                    'minimum_gpu_free_memory_fraction = 0.9\ndtype = "bfloat16"',
                    'minimum_gpu_free_memory_fraction = 0.9\ndtype = "float16"',
                )
            )

    def test_documented_canonical_local_choices_are_immutable(self) -> None:
        mutations = (
            ("clip_epsilon = 0.2", "clip_epsilon = 0.3", "grpo.clip_epsilon"),
            ("ppo_epochs = 1", "ppo_epochs = 2", "grpo.ppo_epochs"),
            (
                "ppo_mini_batch_size = 256",
                "ppo_mini_batch_size = 128",
                "grpo.ppo_mini_batch_size",
            ),
            ("base_seed = 1729", "base_seed = 1730", "rollouts.sampling"),
            ("base_seed = 2718", "base_seed = 2719", "synthesis.sampling"),
            ("seed = 42", "seed = 43", "runtime.seed"),
            ("max_answer_chars = 50000", "max_answer_chars = 1", "reward"),
            (
                "save_every_steps = 100",
                "save_every_steps = 50",
                "checkpointing.save_every_steps",
            ),
            (
                "max_checkpoints = 3",
                "max_checkpoints = 4",
                "checkpointing.max_checkpoints",
            ),
        )
        source = resolved_config_text()
        for old, new, error in mutations:
            with self.subTest(setting=old):
                self.assertIn(old, source)
                with self.assertRaisesRegex(TrainingError, error):
                    load_text(source.replace(old, new))

    def test_registered_model_and_prompt_choices_are_immutable(self) -> None:
        source = resolved_config_text()
        mutations = (
            (
                'model_id = "Qwen/Qwen3-4B"',
                'model_id = "Other/Model"',
                "policy.model_id",
            ),
            (
                'path = "prompts/math500/solve_v1.txt"',
                'path = "prompts/math500/other.txt"',
                "rollouts.prompt",
            ),
            (
                'path = "prompts/math500/synthesis_cot_appendix_f_literal.txt"',
                'path = "prompts/math500/other.txt"',
                "synthesis.prompt",
            ),
        )
        for old, new, error in mutations:
            with self.subTest(setting=old):
                self.assertIn(old, source)
                with self.assertRaisesRegex(TrainingError, error):
                    load_text(source.replace(old, new))


class TrainingPlanningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_text(resolved_config_text())
        self.raw_prompt = load_prompt(REPOSITORY_ROOT, self.config.rollouts.prompt)
        self.questions = [
            QuestionRecord(f"question-{index:03d}", f"PROBLEM_{index:03d}")
            for index in range(500)
        ]

    def test_training_rows_have_only_reference_free_metadata(self) -> None:
        rows = build_training_rows(self.questions, self.config, self.raw_prompt)
        self.assertEqual(len(rows), 500)
        self.assertEqual(rows[0]["extra_info"], {"question_id": "question-000"})
        self.assertIsNone(rows[0]["reward_model"]["ground_truth"])
        serialized = "\n".join(json.dumps(row) for row in rows)
        self.assertNotIn("labels_path", serialized)
        self.assertNotIn("reference_answer", serialized)
        self.assertIn("PROBLEM_000", rows[0]["prompt"][0]["content"])

    def test_verl_translation_keeps_group_and_kl_contract(self) -> None:
        command = build_verl_command(
            self.config,
            repository_root=REPOSITORY_ROOT,
            run_dir=Path("/tmp/cat-test-run"),
            training_data_path=Path("/tmp/cat-test-run/math500_train.jsonl"),
        )
        overrides = set(command.argv[3:])
        required = {
            "data.train_batch_size=256",
            "actor_rollout_ref.rollout.n=8",
            "+actor_rollout_ref.rollout.seed=1729",
            "algorithm.adv_estimator=grpo",
            "algorithm.norm_adv_by_std_in_grpo=True",
            "algorithm.use_kl_in_reward=True",
            "actor_rollout_ref.actor.use_kl_loss=False",
            "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean",
            "algorithm.kl_ctrl.kl_coef=0.001",
            "trainer.total_training_steps=1000",
            "trainer.val_before_train=False",
            "trainer.test_freq=-1",
            "trainer.balance_batch=False",
            "reward_model.reward_manager=batch",
            'data.custom_cls.path="pkg://compute_as_a_teacher.training.verl_dataset"',
            "data.custom_cls.name=JsonlRLHFDataset",
            "actor_rollout_ref.actor.use_dynamic_bsz=True",
            "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384",
        }
        self.assertTrue(required.issubset(overrides))
        self.assertNotIn("critic.enable=False", overrides)
        self.assertIn(
            "trainer.rollout_data_dir="
            + json.dumps(str(Path("/tmp/cat-test-run/rollout_logs").resolve())),
            overrides,
        )
        self.assertEqual(
            command.environment["PYTHONPATH"],
            str(REPOSITORY_ROOT / "src"),
        )
        serialized = json.dumps(command.to_dict())
        self.assertNotIn("labels.jsonl", serialized)
        self.assertNotIn("solution", serialized)

    def test_checkpoint_step_zero_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            self.assertEqual(checkpointed_step(run_dir, 1000), 0)
            tracker = run_dir / "checkpoints/latest_checkpointed_iteration.txt"
            tracker.parent.mkdir()
            tracker.write_text("0\n", encoding="utf-8")
            self.assertEqual(checkpointed_step(run_dir, 1000), 0)
            tracker.write_text("1001\n", encoding="utf-8")
            with self.assertRaisesRegex(TrainingError, "outside"):
                checkpointed_step(run_dir, 1000)

    def test_training_plan_writer_respects_the_live_launch_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            run_dir.mkdir()
            questions_path = root / "questions.jsonl"
            questions_path.write_text("fixture\n", encoding="utf-8")
            lock_path = root / "lock.json"
            lock_path.write_text("{}\n", encoding="utf-8")
            synthesis_prompt = load_prompt(
                REPOSITORY_ROOT, self.config.synthesis.prompt
            )
            with exclusive_launch(run_dir), self.assertRaisesRegex(
                TrainingError, "already locked"
            ):
                write_training_plan(
                    run_dir,
                    self.questions,
                    self.config,
                    self.raw_prompt,
                    synthesis_prompt,
                    questions_path=questions_path,
                    dataset_lock_path=lock_path,
                    repository_root=REPOSITORY_ROOT,
                    force=True,
                )
            self.assertFalse((run_dir / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
