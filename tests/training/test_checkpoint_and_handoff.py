from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from compute_as_a_teacher.data.math500 import QuestionRecord  # noqa: E402
from compute_as_a_teacher.evaluation.config import (  # noqa: E402
    load_raw_config,
    load_synthesis_config,
)
from compute_as_a_teacher.evaluation.prompts import load_prompt  # noqa: E402
from compute_as_a_teacher.training.checkpoints import (  # noqa: E402
    load_registered_checkpoint,
    register_final_checkpoint,
)
from compute_as_a_teacher.training.config import (  # noqa: E402
    TrainingError,
    load_training_config,
)
from compute_as_a_teacher.training.eval_handoff import (  # noqa: E402
    RAW_CONFIG_NAME,
    SYNTHESIS_CONFIG_NAME,
    write_eval_handoff,
)
from compute_as_a_teacher.training.planning import write_training_plan  # noqa: E402


EXAMPLE = REPOSITORY_ROOT / "configs/training/math500_cat_grpo.example.toml"


def _resolved_config_text() -> str:
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


def _write_verl_checkpoint(checkpoint_root: Path, step: int) -> None:
    step_dir = checkpoint_root / f"global_step_{step}"
    actor_dir = step_dir / "actor"
    huggingface_dir = actor_dir / "huggingface"
    huggingface_dir.mkdir(parents=True)
    (step_dir / "data.pt").write_bytes(b"dataloader-state")
    (actor_dir / "fsdp_config.json").write_text(
        json.dumps({"FSDP_version": 1, "world_size": 8}),
        encoding="utf-8",
    )
    (huggingface_dir / "config.json").write_text("{}\n", encoding="utf-8")
    for kind in ("model", "optim", "extra_state"):
        for rank in range(8):
            (actor_dir / f"{kind}_world_size_8_rank_{rank}.pt").write_bytes(
                f"{kind}-{rank}".encode("ascii")
            )


class CheckpointAndHandoffTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        config_path = self.root / "training.toml"
        config_path.write_text(_resolved_config_text(), encoding="utf-8")
        self.config = load_training_config(config_path)

        questions = [
            QuestionRecord(f"question-{index:03d}", f"PROBLEM_{index:03d}")
            for index in range(500)
        ]
        questions_path = self.root / "questions.jsonl"
        questions_path.write_text(
            "".join(
                json.dumps({"id": item.id, "problem": item.problem}) + "\n"
                for item in questions
            ),
            encoding="utf-8",
        )
        lock_path = self.root / "dataset.lock.json"
        lock_path.write_text("{}\n", encoding="utf-8")

        raw_prompt = load_prompt(REPOSITORY_ROOT, self.config.rollouts.prompt)
        synthesis_prompt = load_prompt(REPOSITORY_ROOT, self.config.synthesis.prompt)
        self.run_dir = self.root / "run"
        write_training_plan(
            self.run_dir,
            questions,
            self.config,
            raw_prompt,
            synthesis_prompt,
            questions_path=questions_path,
            dataset_lock_path=lock_path,
            repository_root=REPOSITORY_ROOT,
        )

        checkpoint_root = self.run_dir / "checkpoints"
        _write_verl_checkpoint(checkpoint_root, 1000)
        (checkpoint_root / "latest_checkpointed_iteration.txt").write_text(
            "1000\n", encoding="utf-8"
        )

        self.export_dir = self.root / "export"
        self.export_dir.mkdir()
        (self.export_dir / "config.json").write_text("{}\n", encoding="utf-8")
        (self.export_dir / "model.safetensors").write_bytes(b"merged-weights")
        (self.export_dir / "tokenizer_config.json").write_text(
            "{}\n", encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_registers_only_the_fixed_terminal_checkpoint(self) -> None:
        tracker = self.run_dir / "checkpoints/latest_checkpointed_iteration.txt"
        tracker.write_text("999\n", encoding="utf-8")

        with self.assertRaisesRegex(TrainingError, "Expected completed step 1000"):
            register_final_checkpoint(self.run_dir, self.export_dir)

        self.assertFalse((self.run_dir / "checkpoint_manifest.json").exists())
        self.assertFalse((self.run_dir / "completion.json").exists())

    def test_registration_records_integrity_and_detects_tampering(self) -> None:
        manifest, completion = register_final_checkpoint(
            self.run_dir, self.export_dir
        )

        self.assertEqual(manifest["step"], 1000)
        self.assertEqual(manifest["selected_by"], "fixed_final_step")
        self.assertEqual(completion["completed_step"], 1000)
        self.assertEqual(completion["selection"], "fixed_final_step_without_labels")
        self.assertFalse(completion["labels_loaded"])
        loaded_manifest, loaded_completion = load_registered_checkpoint(self.run_dir)
        self.assertEqual(loaded_manifest, manifest)
        self.assertEqual(loaded_completion, completion)

        (self.export_dir / "model.safetensors").write_bytes(b"tampered")
        with self.assertRaisesRegex(TrainingError, "Registered export checkpoint changed"):
            load_registered_checkpoint(self.run_dir)

    def test_registration_rejects_export_paths_that_overlap_sources(self) -> None:
        actor_dir = (
            self.run_dir / "checkpoints" / "global_step_1000" / "actor"
        )
        unsafe = (
            self.run_dir,
            self.root,
            actor_dir,
            actor_dir.parent,
            actor_dir / "merged",
        )
        for export_dir in unsafe:
            with self.subTest(export_dir=export_dir), self.assertRaisesRegex(
                TrainingError,
                "Export directory must not",
            ):
                register_final_checkpoint(self.run_dir, export_dir)
        self.assertFalse((self.run_dir / "checkpoint_manifest.json").exists())
        self.assertFalse((self.run_dir / "completion.json").exists())

    def test_handoff_configs_load_and_target_the_registered_export(self) -> None:
        manifest, _ = register_final_checkpoint(self.run_dir, self.export_dir)
        output_dir = self.root / "evaluation"
        handoff = write_eval_handoff(
            self.run_dir,
            output_dir,
            "cat-trained-final",
        )

        raw = load_raw_config(output_dir / RAW_CONFIG_NAME)
        synthesis = load_synthesis_config(output_dir / SYNTHESIS_CONFIG_NAME)
        export_digest = manifest["export"]["tree_sha256"]
        self.assertEqual(raw.model.model_id, "cat-trained-final")
        self.assertEqual(raw.model.revision, export_digest)
        self.assertEqual(synthesis.anchor_relation, "frozen_initial_for_trained_raw")
        self.assertEqual(synthesis.anchor.model_id, "cat-frozen-qwen3-4b")
        self.assertEqual(synthesis.anchor.revision, "a" * 40)
        self.assertNotEqual(raw.model, synthesis.anchor)
        self.assertEqual(handoff["schema_version"], 2)
        self.assertEqual(handoff["raw_policy_model"], raw.model.to_dict())
        self.assertEqual(
            handoff["synthesis_anchor_model"],
            synthesis.anchor.to_dict(),
        )
        self.assertEqual(
            handoff["synthesis_anchor_relation"],
            "frozen_initial_for_trained_raw",
        )
        self.assertEqual(handoff["checkpoint_tree_sha256"], export_digest)
        self.assertEqual(Path(handoff["checkpoint_path"]), self.export_dir.resolve())
        self.assertEqual(handoff["selection"], "fixed_final_step_without_labels")
        self.assertFalse(handoff["labels_loaded"])

    def test_handoff_cannot_write_inside_a_registered_checkpoint(self) -> None:
        register_final_checkpoint(self.run_dir, self.export_dir)
        with self.assertRaisesRegex(TrainingError, "outside registered checkpoint"):
            write_eval_handoff(
                self.run_dir,
                self.export_dir,
                "cat-trained-final",
                force=True,
            )
        load_registered_checkpoint(self.run_dir)


if __name__ == "__main__":
    unittest.main()
