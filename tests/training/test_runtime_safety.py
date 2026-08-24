from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from compute_as_a_teacher.training.config import (  # noqa: E402
    TrainingError,
    load_training_config,
)
from compute_as_a_teacher.training.verl_adapter import (  # noqa: E402
    checkpointed_step,
    exclusive_launch,
    merge_command,
    require_label_free_training_outputs,
    run_command_with_log,
    validate_verl_checkpoint,
    verify_verl_checkout,
)


EXAMPLE = REPOSITORY_ROOT / "configs/training/math500_cat_grpo.example.toml"


def _resolved_config():
    text = (
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
    temporary = tempfile.TemporaryDirectory()
    path = Path(temporary.name) / "config.toml"
    path.write_text(text, encoding="utf-8")
    return temporary, load_training_config(path)


def _write_checkpoint(
    run_dir: Path,
    step: int,
    world_size: int,
    fsdp_version: int = 1,
) -> Path:
    checkpoint_root = run_dir / "checkpoints"
    actor_dir = checkpoint_root / f"global_step_{step}" / "actor"
    (actor_dir / "huggingface").mkdir(parents=True)
    (checkpoint_root / "latest_checkpointed_iteration.txt").write_text(
        f"{step}\n",
        encoding="utf-8",
    )
    (actor_dir.parent / "data.pt").write_bytes(b"dataloader-state")
    (actor_dir / "fsdp_config.json").write_text(
        json.dumps({"FSDP_version": fsdp_version, "world_size": world_size}),
        encoding="utf-8",
    )
    (actor_dir / "huggingface/config.json").write_text("{}\n", encoding="utf-8")
    for kind in ("model", "optim", "extra_state"):
        for rank in range(world_size):
            (actor_dir / f"{kind}_world_size_{world_size}_rank_{rank}.pt").write_bytes(
                f"{kind}-{rank}".encode()
            )
    return actor_dir


class CheckpointSafetyTests(unittest.TestCase):
    def test_trainer_rejects_label_derived_artifacts_in_its_output_mount(self) -> None:
        for artifact_name in ("scores.jsonl", "paired_scores.jsonl"):
            with self.subTest(artifact_name=artifact_name):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary).resolve()
                    scored = root / "evals/initial-raw"
                    scored.mkdir(parents=True)
                    (scored / artifact_name).write_text("\n", encoding="utf-8")
                    environment = {
                        "CAT_SERVICE_ROLE": "trainer",
                        "CAT_OUTPUT_DIR": str(root),
                    }
                    with self.assertRaisesRegex(
                        TrainingError, "label-derived artifacts"
                    ):
                        require_label_free_training_outputs(environment)
                    (scored / artifact_name).unlink()
                    self.assertTrue(
                        require_label_free_training_outputs(environment)["enforced"]
                    )

    def test_training_cli_import_does_not_load_gpu_or_model_frameworks(self) -> None:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(REPOSITORY_ROOT / "src")
        code = (
            "import sys; import compute_as_a_teacher.training.cli; "
            "blocked={'torch','transformers','datasets','verl','vllm','ray'}; "
            "loaded={name.split('.')[0] for name in sys.modules}; "
            "assert not blocked & loaded, sorted(blocked & loaded)"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPOSITORY_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_process_output_is_streamed_to_a_durable_log(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log_path = root / "logs" / "trainer.log"
            return_code = run_command_with_log(
                (sys.executable, "-c", "print('step:1 - actor/pg_loss:0.5')"),
                cwd=root,
                environment=os.environ.copy(),
                log_path=log_path,
            )
            self.assertEqual(return_code, 0)
            self.assertEqual(
                log_path.read_text(encoding="utf-8"),
                "step:1 - actor/pg_loss:0.5\n",
            )

    def test_accepts_pinned_fsdp_checkpoint_versions(self) -> None:
        for fsdp_version in (1, 2):
            with (
                self.subTest(fsdp_version=fsdp_version),
                tempfile.TemporaryDirectory() as temporary,
            ):
                run_dir = Path(temporary)
                actor_dir = _write_checkpoint(
                    run_dir,
                    step=20,
                    world_size=2,
                    fsdp_version=fsdp_version,
                )
                self.assertEqual(
                    validate_verl_checkpoint(run_dir, 20, expected_world_size=2),
                    actor_dir.resolve(),
                )
                self.assertEqual(
                    checkpointed_step(run_dir, 1000, expected_world_size=2),
                    20,
                )

    def test_launch_lock_rejects_concurrent_writers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            with exclusive_launch(run_dir):
                with self.assertRaisesRegex(TrainingError, "already locked"):
                    with exclusive_launch(run_dir):
                        self.fail("a second launch acquired the same run directory")
            self.assertTrue((run_dir / ".launch.lock").is_file())

    def test_rejects_missing_step_data_and_actor_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            tracker = run_dir / "checkpoints/latest_checkpointed_iteration.txt"
            tracker.parent.mkdir()
            tracker.write_text("20\n", encoding="utf-8")
            with self.assertRaisesRegex(TrainingError, "step directory is missing"):
                checkpointed_step(run_dir, 1000)

            actor_dir = _write_checkpoint(run_dir, step=20, world_size=2)
            (actor_dir.parent / "data.pt").unlink()
            with self.assertRaisesRegex(TrainingError, "dataloader state"):
                checkpointed_step(run_dir, 1000)

            (actor_dir.parent / "data.pt").write_bytes(b"state")
            (actor_dir / "huggingface/config.json").unlink()
            with self.assertRaisesRegex(TrainingError, "Hugging Face config"):
                checkpointed_step(run_dir, 1000)

    def test_rejects_world_size_and_shard_rank_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            actor_dir = _write_checkpoint(run_dir, step=100, world_size=2)
            with self.assertRaisesRegex(TrainingError, "world_size mismatch"):
                checkpointed_step(run_dir, 1000, expected_world_size=8)

            (actor_dir / "optim_world_size_2_rank_1.pt").unlink()
            (actor_dir / "model_world_size_2_rank_2.pt").write_bytes(b"extra-rank")
            with self.assertRaisesRegex(TrainingError, "shard set is invalid"):
                checkpointed_step(run_dir, 1000, expected_world_size=2)

    def test_rejects_an_unsupported_fsdp_version(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            _write_checkpoint(
                run_dir,
                step=100,
                world_size=2,
                fsdp_version=3,
            )
            with self.assertRaisesRegex(TrainingError, "FSDP version 1 or 2"):
                checkpointed_step(run_dir, 1000, expected_world_size=2)

    def test_merge_requires_the_valid_fixed_final_checkpoint(self) -> None:
        config_owner, config = _resolved_config()
        self.addCleanup(config_owner.cleanup)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            run_dir.mkdir()
            export_dir = root / "export"
            with self.assertRaisesRegex(TrainingError, "Expected completed step 1000"):
                merge_command(
                    config,
                    run_dir=run_dir,
                    export_directory=export_dir,
                )
            actor_dir = _write_checkpoint(run_dir, step=1000, world_size=8)
            command = merge_command(
                config,
                run_dir=run_dir,
                export_directory=export_dir,
            )
            self.assertEqual(
                command[command.index("--local_dir") + 1],
                str(actor_dir.resolve()),
            )

    def test_merge_rejects_export_paths_that_overlap_sources(self) -> None:
        config_owner, config = _resolved_config()
        self.addCleanup(config_owner.cleanup)
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            actor_dir = _write_checkpoint(run_dir, step=1000, world_size=8)
            unsafe = (
                run_dir,
                run_dir.parent,
                actor_dir,
                actor_dir.parent,
                actor_dir / "merged",
            )
            for export_dir in unsafe:
                with self.subTest(export_dir=export_dir), self.assertRaisesRegex(
                    TrainingError,
                    "Export directory must not",
                ):
                    merge_command(
                        config,
                        run_dir=run_dir,
                        export_directory=export_dir,
                    )


class VerlCheckoutSafetyTests(unittest.TestCase):
    def test_requires_the_expected_clean_git_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary)
            subprocess.run(["git", "init", "--quiet", str(checkout)], check=True)
            tracked = checkout / "tracked.py"
            tracked.write_text("VALUE = 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(checkout), "add", "tracked.py"], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(checkout),
                    "-c",
                    "user.name=Test",
                    "-c",
                    "user.email=test@example.invalid",
                    "-c",
                    "commit.gpgsign=false",
                    "commit",
                    "--quiet",
                    "-m",
                    "initial",
                ],
                check=True,
            )
            revision = subprocess.run(
                ["git", "-C", str(checkout), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            verify_verl_checkout(checkout, revision)

            tracked.write_text("VALUE = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(TrainingError, "must be clean"):
                verify_verl_checkout(checkout, revision)

            tracked.write_text("VALUE = 1\n", encoding="utf-8")
            (checkout / "untracked.py").write_text("VALUE = 3\n", encoding="utf-8")
            with self.assertRaisesRegex(TrainingError, "must be clean"):
                verify_verl_checkout(checkout, revision)

            with self.assertRaisesRegex(TrainingError, "revision mismatch"):
                verify_verl_checkout(checkout, "0" * 40)


if __name__ == "__main__":
    unittest.main()
