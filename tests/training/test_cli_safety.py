from __future__ import annotations

import argparse
import sys
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from compute_as_a_teacher.training import cli  # noqa: E402
from compute_as_a_teacher.training.errors import TrainingError  # noqa: E402
from compute_as_a_teacher.training.verl_adapter import VerlCommand  # noqa: E402
from tests.training.test_config_and_planning import (  # noqa: E402
    load_text,
    resolved_config_text,
)


def command() -> VerlCommand:
    return VerlCommand(
        argv=("/opt/verl/bin/python", "-m", "verl.trainer.main_ppo"),
        cwd="/opt/verl/source",
        environment={"HF_HUB_OFFLINE": "1"},
        framework_revision="8fdc4d3f202f41461f4de9f42a637228e342668b",
        adapter_version="cat-verl-batch-reward-v1",
    )


class LaunchPostconditionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_text(resolved_config_text())

    def test_canonical_zero_exit_without_terminal_checkpoint_fails(self) -> None:
        args = argparse.Namespace(
            config=Path("config.toml"),
            run_dir=Path("canonical"),
            execute=True,
            preregistration=Path("preregistration.json"),
            launch_approval=Path("launch-approval.json"),
        )
        manifest = {"config_fingerprint": self.config.fingerprint}
        with (
            patch.object(cli, "load_training_config", return_value=self.config),
            patch.object(cli, "load_training_plan", return_value=(manifest, command())),
            patch.object(cli, "checkpointed_step", side_effect=[0, 0, 0]),
            patch.object(cli, "exclusive_launch", return_value=nullcontext(object())),
            patch.object(
                cli,
                "verify_launch_approval",
                return_value={"launch_approval_fingerprint": "f" * 64},
            ),
            patch.object(
                cli,
                "run_preflight",
                return_value={
                    "preflight_fingerprint": "a" * 64,
                    "operationally_ready_to_launch": True,
                    "missing_gates": [],
                },
            ),
            patch.object(cli, "write_preflight_receipt"),
            patch.object(cli, "launch_verl", return_value=0),
        ):
            with self.assertRaisesRegex(TrainingError, "terminal checkpoint"):
                cli._launch(args)

    def test_canonical_execution_requires_preregistration_and_approval(self) -> None:
        args = argparse.Namespace(
            config=Path("config.toml"),
            run_dir=Path("canonical"),
            execute=True,
            preregistration=None,
            launch_approval=None,
        )
        manifest = {"config_fingerprint": self.config.fingerprint}
        with (
            patch.object(cli, "load_training_config", return_value=self.config),
            patch.object(cli, "load_training_plan", return_value=(manifest, command())),
            patch.object(cli, "checkpointed_step", return_value=0),
            patch.object(cli, "exclusive_launch", return_value=nullcontext(object())),
            patch.object(cli, "run_preflight") as preflight,
        ):
            with self.assertRaisesRegex(TrainingError, "--preregistration"):
                cli._launch(args)
        preflight.assert_not_called()

    def test_canonical_preview_does_not_require_approval(self) -> None:
        args = argparse.Namespace(
            config=Path("config.toml"),
            run_dir=Path("canonical"),
            execute=False,
            preregistration=None,
            launch_approval=None,
        )
        manifest = {"config_fingerprint": self.config.fingerprint}
        with (
            patch.object(cli, "load_training_config", return_value=self.config),
            patch.object(cli, "load_training_plan", return_value=(manifest, command())),
            patch.object(cli, "checkpointed_step", return_value=0),
        ):
            result = cli._launch(args)
        self.assertFalse(result["would_execute"])

    def test_canonical_preflight_failure_never_spawns_verl(self) -> None:
        args = argparse.Namespace(
            config=Path("config.toml"),
            run_dir=Path("canonical"),
            execute=True,
            preregistration=Path("preregistration.json"),
            launch_approval=Path("launch-approval.json"),
        )
        manifest = {"config_fingerprint": self.config.fingerprint}
        with (
            patch.object(cli, "load_training_config", return_value=self.config),
            patch.object(cli, "load_training_plan", return_value=(manifest, command())),
            patch.object(cli, "checkpointed_step", return_value=0),
            patch.object(cli, "exclusive_launch", return_value=nullcontext(object())),
            patch.object(
                cli,
                "verify_launch_approval",
                return_value={"launch_approval_fingerprint": "f" * 64},
            ),
            patch.object(
                cli,
                "run_preflight",
                return_value={
                    "operationally_ready_to_launch": False,
                    "missing_gates": ["anchor canary"],
                },
            ),
            patch.object(cli, "write_preflight_receipt") as write_receipt,
            patch.object(cli, "launch_verl") as launch,
        ):
            with self.assertRaisesRegex(TrainingError, "operational preflight"):
                cli._launch(args)
        write_receipt.assert_not_called()
        launch.assert_not_called()

    def test_canonical_lock_spans_preflight_receipt_and_process(self) -> None:
        args = argparse.Namespace(
            config=Path("config.toml"),
            run_dir=Path("canonical"),
            execute=True,
            preregistration=Path("preregistration.json"),
            launch_approval=Path("launch-approval.json"),
        )
        manifest = {"config_fingerprint": self.config.fingerprint}
        state = {"locked": False, "checkpoints": 0}
        lease = object()

        @contextmanager
        def locked():
            self.assertFalse(state["locked"])
            state["locked"] = True
            try:
                yield lease
            finally:
                state["locked"] = False

        def checkpoint(*_args, **_kwargs):
            state["checkpoints"] += 1
            if state["checkpoints"] == 1:
                self.assertFalse(state["locked"])
                return 0
            self.assertTrue(state["locked"])
            return self.config.grpo.max_steps if state["checkpoints"] == 3 else 0

        def require_lock(*_args, **_kwargs):
            self.assertTrue(state["locked"])

        def approval(*_args, **_kwargs):
            require_lock()
            return {"launch_approval_fingerprint": "f" * 64}

        def preflight(*_args, **_kwargs):
            require_lock()
            return {
                "preflight_fingerprint": "a" * 64,
                "operationally_ready_to_launch": True,
                "missing_gates": [],
            }

        def launch(*_args, **kwargs):
            require_lock()
            self.assertIs(kwargs["lease"], lease)
            return 0

        with (
            patch.object(cli, "load_training_config", return_value=self.config),
            patch.object(cli, "load_training_plan", return_value=(manifest, command())),
            patch.object(cli, "checkpointed_step", side_effect=checkpoint),
            patch.object(cli, "exclusive_launch", side_effect=lambda _path: locked()),
            patch.object(cli, "verify_launch_approval", side_effect=approval),
            patch.object(cli, "run_preflight", side_effect=preflight),
            patch.object(cli, "write_preflight_receipt", side_effect=require_lock),
            patch.object(cli, "launch_verl", side_effect=launch),
        ):
            result = cli._launch(args)
        self.assertTrue(result["terminal_checkpoint_verified"])
        self.assertFalse(state["locked"])

    def test_qualification_zero_exit_without_terminal_checkpoint_fails(self) -> None:
        args = argparse.Namespace(
            config=Path("config.toml"),
            qualification_dir=Path("qualification"),
            execute=True,
        )
        manifest = {
            "source": {
                "config_fingerprint": self.config.fingerprint,
                "run_dir": "/tmp/canonical-source",
            },
            "profile": {"name": "one_step"},
            "qualification_fingerprint": "c" * 64,
            "artifacts": {
                "training_data": {"sha256": "d" * 64},
                "verl_command": {"sha256": "e" * 64},
            },
        }
        source_dir = Path("/tmp/canonical-source").resolve()
        qualification_dir = (REPOSITORY_ROOT / "qualification").resolve()
        leases = {source_dir: object(), qualification_dir: object()}
        with (
            patch.object(cli, "load_training_config", return_value=self.config),
            patch.object(cli, "load_qualification_plan", return_value=(manifest, command())),
            patch.object(cli, "checkpointed_step", side_effect=[0, 0, 0]),
            patch.object(
                cli,
                "exclusive_launches",
                return_value=nullcontext(leases),
            ),
            patch.object(cli, "load_training_plan", return_value=({"plan_fingerprint": "b" * 64}, command())),
            patch.object(
                cli,
                "run_preflight",
                return_value={
                    "preflight_fingerprint": "a" * 64,
                    "operationally_ready_to_launch": True,
                    "missing_gates": [],
                },
            ),
            patch.object(cli, "write_preflight_receipt"),
            patch.object(cli, "launch_qualification", return_value=0),
        ):
            with self.assertRaisesRegex(TrainingError, "terminal checkpoint"):
                cli._launch_qualification(args)


if __name__ == "__main__":
    unittest.main()
