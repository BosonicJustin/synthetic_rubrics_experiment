from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from compute_as_a_teacher.training.config import SUPPORTED_VERL_REVISION  # noqa: E402
from compute_as_a_teacher.training.errors import TrainingError  # noqa: E402
from compute_as_a_teacher.training.qualification import (  # noqa: E402
    QUALIFICATION_PROFILES,
    derive_qualification_command,
    load_qualification_plan,
    write_qualification_plan,
)
from compute_as_a_teacher.training.verl_adapter import (  # noqa: E402
    VerlCommand,
    exclusive_launch,
)


def source_command() -> VerlCommand:
    argv = (
        "/opt/verl/bin/python",
        "-m",
        "verl.trainer.main_ppo",
        'data.train_files="/canonical/math500_train.jsonl"',
        'data.val_files="/canonical/math500_train.jsonl"',
        "data.train_batch_size=256",
        "data.val_batch_size=256",
        "actor_rollout_ref.actor.ppo_mini_batch_size=256",
        "actor_rollout_ref.actor.optim.lr=5e-7",
        "actor_rollout_ref.rollout.temperature=0.7",
        "actor_rollout_ref.rollout.top_p=0.8",
        "actor_rollout_ref.rollout.top_k=20",
        "actor_rollout_ref.rollout.dtype=bfloat16",
        "actor_rollout_ref.rollout.n=8",
        "algorithm.use_kl_in_reward=True",
        "algorithm.kl_ctrl.kl_coef=0.001",
        "+custom_reward_function.reward_kwargs.anchor_model=\"pi0\"",
        "+custom_reward_function.reward_kwargs.anchor_temperature=0.7",
        "+custom_reward_function.reward_kwargs.anchor_top_p=0.8",
        "+custom_reward_function.reward_kwargs.anchor_top_k=20",
        "trainer.n_gpus_per_node=8",
        "trainer.total_training_steps=1000",
        "trainer.total_epochs=1000",
        "trainer.save_freq=100",
        'trainer.default_local_dir="/canonical/checkpoints"',
        'trainer.experiment_name="canonical"',
    )
    return VerlCommand(
        argv=argv,
        cwd="/opt/verl/source",
        environment={"HF_HUB_OFFLINE": "1"},
        framework_revision=SUPPORTED_VERL_REVISION,
        adapter_version="cat-verl-batch-reward-v1",
    )


def source_manifest() -> dict[str, object]:
    return {
        "run_name": "qwen3-4b-math500-cat",
        "plan_fingerprint": "a" * 64,
        "config_fingerprint": "b" * 64,
        "label_firewall": {
            "labels_loaded": False,
            "reference_answers_loaded": False,
            "reference_solutions_loaded": False,
            "reward_uses_synthesized_answer_only": True,
            "checkpoint_selection_uses_labels": False,
        },
    }


def write_source_files(source_dir: Path) -> None:
    source_dir.mkdir()
    (source_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    with (source_dir / "math500_train.jsonl").open("w", encoding="utf-8") as handle:
        for index in range(500):
            row = {
                "data_source": "cat_math500_reference_free_v1",
                "prompt": [{"role": "user", "content": f"problem {index}"}],
                "reward_model": {"style": "reference_free", "ground_truth": None},
                "extra_info": {"question_id": f"question-{index:03d}"},
            }
            handle.write(json.dumps(row) + "\n")


class QualificationPlanTests(unittest.TestCase):
    def test_profiles_change_only_operational_scale_and_are_non_reportable(self) -> None:
        command = source_command()
        allowed_changes = {
            "data.train_files",
            "data.val_files",
            "data.train_batch_size",
            "data.val_batch_size",
            "actor_rollout_ref.actor.ppo_mini_batch_size",
            "trainer.total_training_steps",
            "trainer.total_epochs",
            "trainer.save_freq",
            "trainer.default_local_dir",
            "trainer.experiment_name",
            "trainer.rollout_data_dir",
        }
        frozen_science = {
            "actor_rollout_ref.actor.optim.lr",
            "actor_rollout_ref.rollout.temperature",
            "actor_rollout_ref.rollout.top_p",
            "actor_rollout_ref.rollout.top_k",
            "actor_rollout_ref.rollout.dtype",
            "actor_rollout_ref.rollout.n",
            "algorithm.use_kl_in_reward",
            "algorithm.kl_ctrl.kl_coef",
            "+custom_reward_function.reward_kwargs.anchor_model",
            "+custom_reward_function.reward_kwargs.anchor_temperature",
            "+custom_reward_function.reward_kwargs.anchor_top_p",
            "+custom_reward_function.reward_kwargs.anchor_top_k",
            "trainer.n_gpus_per_node",
        }

        def override_map(argv):
            return {
                item.split("=", 1)[0]: item.split("=", 1)[1]
                for item in argv[3:]
            }

        source_overrides = override_map(command.argv)
        for profile in QUALIFICATION_PROFILES.values():
            with self.subTest(profile=profile.name):
                derived = derive_qualification_command(
                    command,
                    profile,
                    Path("/tmp/cat-qualification"),
                    run_name="canonical",
                )
                override_items = set(derived.argv[3:])
                self.assertIn("actor_rollout_ref.rollout.n=8", override_items)
                self.assertIn(
                    f"data.train_batch_size={profile.prompt_batch_size}", override_items
                )
                self.assertIn(
                    f"trainer.total_training_steps={profile.max_steps}", override_items
                )
                self.assertIn(
                    f"trainer.save_freq={profile.save_every_steps}", override_items
                )
                self.assertTrue(
                    any("nonreportable" in item for item in derived.argv)
                )
                derived_overrides = override_map(derived.argv)
                changed = {
                    key
                    for key in set(source_overrides) | set(derived_overrides)
                    if source_overrides.get(key) != derived_overrides.get(key)
                }
                self.assertLessEqual(changed, allowed_changes)
                self.assertEqual(
                    changed,
                    {
                        key
                        for key in allowed_changes
                        if source_overrides.get(key) != derived_overrides.get(key)
                    },
                )
                self.assertEqual(
                    {key: derived_overrides[key] for key in frozen_science},
                    {key: source_overrides[key] for key in frozen_science},
                )
                self.assertEqual(command, source_command())

    def test_qualification_directory_must_be_disjoint_from_canonical(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir = root / "canonical"
            write_source_files(source_dir)
            with patch(
                "compute_as_a_teacher.training.qualification.load_training_plan",
                return_value=(source_manifest(), source_command()),
            ):
                with self.assertRaisesRegex(TrainingError, "must not contain"):
                    write_qualification_plan(
                        source_dir,
                        source_dir / "qualification",
                        "one_step",
                    )

    def test_plan_is_lineage_bound_label_free_and_tamper_evident(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir = root / "canonical"
            qualification_dir = root / "qualification"
            write_source_files(source_dir)
            expected = (source_manifest(), source_command())
            with patch(
                "compute_as_a_teacher.training.qualification.load_training_plan",
                return_value=expected,
            ):
                manifest = write_qualification_plan(
                    source_dir,
                    qualification_dir,
                    "one_step",
                )
                loaded, command = load_qualification_plan(qualification_dir)

            self.assertEqual(loaded, manifest)
            self.assertFalse(loaded["reportable"])
            self.assertEqual(loaded["counts"]["problems"], 8)
            self.assertEqual(loaded["counts"]["trajectories_per_update"], 64)
            self.assertEqual(loaded["counts"]["anchor_calls_per_update"], 8)
            self.assertEqual(command.framework_revision, SUPPORTED_VERL_REVISION)
            rows = [
                json.loads(line)
                for line in (qualification_dir / "math500_train.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(rows), 8)
            self.assertTrue(
                all(row["reward_model"]["ground_truth"] is None for row in rows)
            )

            with (qualification_dir / "math500_train.jsonl").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write("{}\n")
            with patch(
                "compute_as_a_teacher.training.qualification.load_training_plan",
                return_value=expected,
            ):
                with self.assertRaisesRegex(TrainingError, "artifact changed"):
                    load_qualification_plan(qualification_dir)

    def test_plan_writer_cannot_race_a_live_run_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir = root / "canonical"
            qualification_dir = root / "qualification"
            write_source_files(source_dir)
            qualification_dir.mkdir()
            expected = (source_manifest(), source_command())
            with patch(
                "compute_as_a_teacher.training.qualification.load_training_plan",
                return_value=expected,
            ), exclusive_launch(qualification_dir), self.assertRaisesRegex(
                TrainingError, "already locked"
            ):
                write_qualification_plan(
                    source_dir,
                    qualification_dir,
                    "one_step",
                    force=True,
                )

    def test_labeled_source_fails_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir = root / "canonical"
            qualification_dir = root / "qualification"
            write_source_files(source_dir)
            path = source_dir / "math500_train.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]["reward_model"]["ground_truth"] = "42"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            with patch(
                "compute_as_a_teacher.training.qualification.load_training_plan",
                return_value=(source_manifest(), source_command()),
            ):
                with self.assertRaisesRegex(TrainingError, "reference-free"):
                    write_qualification_plan(
                        source_dir,
                        qualification_dir,
                        "one_step",
                    )
            self.assertFalse((qualification_dir / "qualification_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
