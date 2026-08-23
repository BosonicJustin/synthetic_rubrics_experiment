from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from compute_as_a_teacher.data.math500 import QuestionRecord  # noqa: E402
from compute_as_a_teacher.evaluation.config import (  # noqa: E402
    load_raw_config,
    load_synthesis_config,
)
from compute_as_a_teacher.evaluation.artifacts import (  # noqa: E402
    artifact_reference,
    canonical_json_bytes,
    publish_json,
    sha256_bytes,
)
from compute_as_a_teacher.evaluation.prompts import load_prompt  # noqa: E402
from compute_as_a_teacher.training.checkpoints import (  # noqa: E402
    MERGE_RECEIPT_KIND,
    MERGE_RECEIPT_NAME,
    _expected_merge_argv,
    _receipt_fingerprint,
    directory_inventory,
    export_and_register_final_checkpoint,
    load_registered_checkpoint,
    load_registered_checkpoint_artifacts,
    register_final_checkpoint,
)
from compute_as_a_teacher.training.config import (  # noqa: E402
    TrainingError,
    load_training_config,
)
from compute_as_a_teacher.training.eval_handoff import (  # noqa: E402
    RAW_CONFIG_NAME,
    SYNTHESIS_CONFIG_NAME,
    verify_eval_handoff_artifacts,
    write_eval_handoff,
)
from compute_as_a_teacher.training.planning import (  # noqa: E402
    MANIFEST_NAME,
    load_training_plan,
    write_training_plan,
)
from compute_as_a_teacher.training.preflight import (  # noqa: E402
    hash_model_snapshot_tree,
)
from compute_as_a_teacher.training import cli as training_cli  # noqa: E402


EXAMPLE = REPOSITORY_ROOT / "configs/training/math500_cat_grpo.example.toml"


def _resolved_config_text(model_path: Path, model_tree_sha256: str) -> str:
    return (
        EXAMPLE.read_text(encoding="utf-8")
        .replace("required_full_model_commit_sha", "a" * 40)
        .replace("required_full_tokenizer_commit_sha", "b" * 40)
        .replace("required_chat_template_sha256", "c" * 64)
        .replace("required_absolute_python_from_verl_environment", sys.executable)
        .replace("required_absolute_verl_checkout", "/opt/verl/source")
        .replace("required_absolute_local_model_snapshot", str(model_path))
        .replace("required_model_snapshot_tree_sha256", model_tree_sha256)
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
        self.root = Path(self.temporary.name).resolve()
        self.base_model = self.root / "base-model"
        self.base_model.mkdir()
        (self.base_model / "config.json").write_text("{}\n", encoding="utf-8")
        (self.base_model / "model.safetensors").write_bytes(b"initial-weights")
        base_identity = hash_model_snapshot_tree(self.base_model)
        self.config_path = self.root / "training.toml"
        self.config_path.write_text(
            _resolved_config_text(self.base_model, base_identity["tree_sha256"]),
            encoding="utf-8",
        )
        self.config = load_training_config(self.config_path)

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
        self._write_guarded_receipt()

    def _write_guarded_receipt(self) -> None:
        plan, _ = load_training_plan(self.run_dir)
        actor = self.run_dir / "checkpoints/global_step_1000/actor"
        actor_files, actor_digest = directory_inventory(actor)
        export_files, export_digest = directory_inventory(self.export_dir)
        base = hash_model_snapshot_tree(self.base_model)
        log = self.run_dir / "logs/export-step-1000.log"
        log.parent.mkdir(exist_ok=True)
        log.write_text("synthetic successful merger log\n", encoding="utf-8")
        source = str(Path(self.config.runtime.verl_source_path).resolve())
        receipt = {
            "schema_version": 1,
            "kind": MERGE_RECEIPT_KIND,
            "training_plan_fingerprint": plan["plan_fingerprint"],
            "config_fingerprint": plan["config_fingerprint"],
            "training_plan": {
                **artifact_reference(self.run_dir / MANIFEST_NAME),
                "path": MANIFEST_NAME,
            },
            "step": 1000,
            "tracker": artifact_reference(
                self.run_dir / "checkpoints/latest_checkpointed_iteration.txt"
            ),
            "merge": {
                "argv": list(
                    _expected_merge_argv(plan, self.run_dir, self.export_dir)
                ),
                "cwd": source,
                "environment": {
                    "HF_DATASETS_OFFLINE": "1",
                    "HF_HUB_OFFLINE": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONHASHSEED": "0",
                    "PYTHONNOUSERSITE": "1",
                    "PYTHONPATH": source,
                    "TRANSFORMERS_OFFLINE": "1",
                },
            },
            "runtime_identity": {
                "framework_revision": self.config.runtime.framework_revision,
                "package_inventory_sha256": self.config.runtime.package_inventory_sha256,
                "trainer_image_digest": self.config.runtime.trainer_image_digest,
            },
            "base_model": {
                "model_path": str(self.base_model),
                "tree_sha256": base["tree_sha256"],
                "files": base["files"],
                "bytes": base["bytes"],
                "inventory": base["inventory"],
                "policy_fingerprint": sha256_bytes(
                    canonical_json_bytes(plan["config"]["policy"])
                ),
            },
            "actor": {
                "path": str(actor),
                "tree_sha256": actor_digest,
                "files": actor_files,
            },
            "export": {
                "path": str(self.export_dir),
                "tree_sha256": export_digest,
                "files": export_files,
            },
            "log": artifact_reference(log),
        }
        receipt["receipt_fingerprint"] = _receipt_fingerprint(receipt)
        publish_json(self.run_dir / MERGE_RECEIPT_NAME, receipt)

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

    def test_registration_rejects_an_unreceipted_or_tampered_export(self) -> None:
        receipt_path = self.run_dir / MERGE_RECEIPT_NAME
        original = receipt_path.read_text(encoding="utf-8")
        receipt_path.unlink()
        with self.assertRaisesRegex(TrainingError, "guarded merge receipt"):
            register_final_checkpoint(self.run_dir, self.export_dir)

        receipt_path.write_text(original, encoding="utf-8")
        value = json.loads(original)
        value["merge"]["argv"][-1] = str(self.root / "unrelated-export")
        receipt_path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(TrainingError, "fingerprint mismatch"):
            register_final_checkpoint(self.run_dir, self.export_dir)

    def test_registration_rejects_a_base_model_copy_even_with_a_receipt(self) -> None:
        for path in self.export_dir.iterdir():
            path.unlink()
        for path in self.base_model.iterdir():
            (self.export_dir / path.name).write_bytes(path.read_bytes())
        (self.run_dir / MERGE_RECEIPT_NAME).unlink()
        (self.run_dir / "logs/export-step-1000.log").unlink()
        self._write_guarded_receipt()
        with self.assertRaisesRegex(TrainingError, "must differ from the immutable"):
            register_final_checkpoint(self.run_dir, self.export_dir)

    def test_registration_rejects_bin_only_or_nested_weight_exports(self) -> None:
        weights = self.export_dir / "model.safetensors"
        weights.rename(self.export_dir / "model.bin")
        (self.run_dir / MERGE_RECEIPT_NAME).unlink()
        (self.run_dir / "logs/export-step-1000.log").unlink()
        self._write_guarded_receipt()
        with self.assertRaisesRegex(TrainingError, "top-level safetensors"):
            register_final_checkpoint(self.run_dir, self.export_dir)

    def test_export_register_recovers_only_from_a_complete_receipt(self) -> None:
        args = SimpleNamespace(
            config=self.config_path,
            run_dir=self.run_dir,
            export_dir=self.export_dir,
            execute=True,
        )
        result = training_cli._export_register(args)
        self.assertEqual(result["mode"], "checkpoint_export_registration_verified")
        load_registered_checkpoint(self.run_dir)

    def test_export_register_recovers_after_manifest_publication(self) -> None:
        register_final_checkpoint(self.run_dir, self.export_dir)
        (self.run_dir / "completion.json").unlink()
        args = SimpleNamespace(
            config=self.config_path,
            run_dir=self.run_dir,
            export_dir=self.export_dir,
            execute=True,
        )
        result = training_cli._export_register(args)
        self.assertEqual(result["mode"], "checkpoint_export_registration_verified")
        load_registered_checkpoint(self.run_dir)

    def test_registration_rejects_a_repointed_merge_log(self) -> None:
        receipt_path = self.run_dir / MERGE_RECEIPT_NAME
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        decoy = self.root / "decoy.log"
        decoy.write_text("unrelated log\n", encoding="utf-8")
        receipt["log"] = artifact_reference(decoy)
        receipt["receipt_fingerprint"] = _receipt_fingerprint(receipt)
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        with self.assertRaisesRegex(TrainingError, "log path changed"):
            register_final_checkpoint(self.run_dir, self.export_dir)

    def test_registered_checkpoint_revalidates_actor_base_and_receipt(self) -> None:
        register_final_checkpoint(self.run_dir, self.export_dir)
        actor_file = (
            self.run_dir
            / "checkpoints/global_step_1000/actor/model_world_size_8_rank_0.pt"
        )
        actor_file.write_bytes(b"mutated")
        with self.assertRaisesRegex(TrainingError, "Registered actor checkpoint changed"):
            load_registered_checkpoint(self.run_dir)

    def test_guarded_export_uses_fresh_private_destination_and_secret_free_env(self) -> None:
        (self.run_dir / MERGE_RECEIPT_NAME).unlink()
        (self.run_dir / "logs/export-step-1000.log").unlink()
        destination = self.root / "fresh-export"
        observed_environment = {}

        def merger(argv, *, cwd, environment, log_path):
            observed_environment.update(environment)
            target = Path(argv[-1])
            (target / "config.json").write_text("{}\n", encoding="utf-8")
            (target / "model.safetensors").write_bytes(b"fresh-merged-weights")
            log_path.parent.mkdir(exist_ok=True)
            log_path.write_text("merge complete\n", encoding="utf-8")
            return 0

        with (
            patch.object(type(self.config), "assert_runnable"),
            patch(
                "compute_as_a_teacher.training.checkpoints.verify_verl_checkout"
            ),
            patch(
                "compute_as_a_teacher.training.checkpoints.run_command_with_log",
                side_effect=merger,
            ),
            patch.dict(
                os.environ,
                {
                    "PATH": "/usr/bin",
                    "WANDB_API_KEY": "must-not-leak",
                    "CAT_ANCHOR_API_KEY": "must-not-leak",
                },
                clear=True,
            ),
        ):
            manifest, _, receipt = export_and_register_final_checkpoint(
                self.config, self.run_dir, destination
            )
        self.assertEqual(destination.stat().st_mode & 0o777, 0o700)
        self.assertNotIn("WANDB_API_KEY", observed_environment)
        self.assertNotIn("CAT_ANCHOR_API_KEY", observed_environment)
        self.assertNotIn("must-not-leak", json.dumps(receipt))
        self.assertEqual(manifest["export"]["path"], str(destination))
        load_registered_checkpoint(self.run_dir)

    def test_guarded_export_preflight_failure_leaves_target_absent(self) -> None:
        (self.run_dir / MERGE_RECEIPT_NAME).unlink()
        (self.run_dir / "logs/export-step-1000.log").unlink()
        destination = self.root / "preflight-failed-export"
        with (
            patch.object(type(self.config), "assert_runnable"),
            patch(
                "compute_as_a_teacher.training.checkpoints.verify_verl_checkout",
                side_effect=TrainingError("pinned checkout mismatch"),
            ),
            self.assertRaisesRegex(TrainingError, "pinned checkout mismatch"),
        ):
            export_and_register_final_checkpoint(
                self.config,
                self.run_dir,
                destination,
            )
        self.assertFalse(destination.exists())

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

    def test_scorer_handoff_verification_does_not_open_the_base_model(self) -> None:
        manifest, completion = register_final_checkpoint(
            self.run_dir, self.export_dir
        )
        output_dir = self.root / "evaluation"
        expected_handoff = write_eval_handoff(
            self.run_dir,
            output_dir,
            "cat-trained-final",
        )
        unavailable_base = self.root / "base-model-unmounted"
        self.base_model.rename(unavailable_base)

        loaded_manifest, loaded_completion = load_registered_checkpoint_artifacts(
            self.run_dir
        )
        self.assertEqual(loaded_manifest, manifest)
        self.assertEqual(loaded_completion, completion)
        self.assertEqual(
            verify_eval_handoff_artifacts(
                self.run_dir,
                output_dir,
                "cat-trained-final",
            ),
            expected_handoff,
        )
        with self.assertRaisesRegex(TrainingError, "Model snapshot is not a directory"):
            load_registered_checkpoint(self.run_dir)

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
