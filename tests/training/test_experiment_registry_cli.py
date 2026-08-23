from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from compute_as_a_teacher.training import cli  # noqa: E402
from compute_as_a_teacher.training.errors import TrainingError  # noqa: E402


CLI_ROOT = Path("/registry-cli-test-root")


def _final_stage_arguments() -> list[str]:
    return [
        "--preregistration",
        "artifacts/preregistration.json",
        "--initial-raw-run-dir",
        "runs/initial-raw",
        "--initial-synthesis-run-dir",
        "runs/initial-synthesis",
        "--training-run-dir",
        "runs/training",
        "--trained-raw-run-dir",
        "runs/trained-raw",
        "--trained-synthesis-run-dir",
        "runs/trained-synthesis",
    ]


class ExperimentRegistryCliTests(unittest.TestCase):
    def test_preregister_routes_resolved_paths_force_and_registry_flags(self) -> None:
        registered = {
            "preregistration_fingerprint": "a" * 64,
            "results_included": True,
            "labels_loaded": True,
            "scientifically_attested": False,
        }
        with (
            patch.object(cli, "REPOSITORY_ROOT", CLI_ROOT),
            patch.object(
                cli,
                "write_experiment_preregistration",
                return_value=registered,
            ) as write,
            patch.object(cli, "_print") as output,
        ):
            status = cli.main(
                [
                    "preregister-experiment",
                    "--output",
                    "artifacts/preregistration.json",
                    "--initial-raw-run-dir",
                    "runs/initial-raw",
                    "--initial-synthesis-config",
                    "configs/initial-synthesis.toml",
                    "--training-run-dir",
                    "runs/training",
                    "--force",
                ]
            )

        self.assertEqual(status, 0)
        write.assert_called_once_with(
            CLI_ROOT / "artifacts/preregistration.json",
            CLI_ROOT / "runs/initial-raw",
            CLI_ROOT / "configs/initial-synthesis.toml",
            CLI_ROOT / "runs/training",
            force=True,
        )
        self.assertEqual(
            output.call_args.args[0],
            {
                "mode": "experiment_preregistered",
                "output": str(CLI_ROOT / "artifacts/preregistration.json"),
                "preregistration_fingerprint": "a" * 64,
                "results_included": True,
                "labels_loaded": True,
                "scientifically_attested": False,
            },
        )

    def test_finalize_routes_every_stage_and_force(self) -> None:
        finalized = {
            "registry_fingerprint": "b" * 64,
            "results_included": True,
            "labels_loaded": True,
            "scientifically_attested": False,
        }
        with (
            patch.object(cli, "REPOSITORY_ROOT", CLI_ROOT),
            patch.object(
                cli,
                "write_final_experiment_registry",
                return_value=finalized,
            ) as write,
            patch.object(cli, "_print") as output,
        ):
            status = cli.main(
                [
                    "finalize-experiment",
                    "--output",
                    "artifacts/registry.json",
                    *_final_stage_arguments(),
                    "--force",
                ]
            )

        self.assertEqual(status, 0)
        write.assert_called_once_with(
            CLI_ROOT / "artifacts/registry.json",
            CLI_ROOT / "artifacts/preregistration.json",
            CLI_ROOT / "runs/initial-raw",
            CLI_ROOT / "runs/initial-synthesis",
            CLI_ROOT / "runs/training",
            CLI_ROOT / "runs/trained-raw",
            CLI_ROOT / "runs/trained-synthesis",
            force=True,
        )
        self.assertEqual(
            output.call_args.args[0],
            {
                "mode": "experiment_registry_finalized",
                "output": str(CLI_ROOT / "artifacts/registry.json"),
                "registry_fingerprint": "b" * 64,
                "results_included": True,
                "labels_loaded": True,
                "scientifically_attested": False,
            },
        )

    def test_verify_routes_registry_and_every_registered_stage(self) -> None:
        verified = {
            "registry_fingerprint": "c" * 64,
            "results_included": True,
            "labels_loaded": True,
            "scientifically_attested": False,
        }
        with (
            patch.object(cli, "REPOSITORY_ROOT", CLI_ROOT),
            patch.object(
                cli,
                "verify_final_experiment_registry",
                return_value=verified,
            ) as verify,
            patch.object(cli, "_print") as output,
        ):
            status = cli.main(
                [
                    "verify-experiment",
                    "--registry",
                    "artifacts/registry.json",
                    *_final_stage_arguments(),
                ]
            )

        self.assertEqual(status, 0)
        verify.assert_called_once_with(
            CLI_ROOT / "artifacts/registry.json",
            CLI_ROOT / "artifacts/preregistration.json",
            CLI_ROOT / "runs/initial-raw",
            CLI_ROOT / "runs/initial-synthesis",
            CLI_ROOT / "runs/training",
            CLI_ROOT / "runs/trained-raw",
            CLI_ROOT / "runs/trained-synthesis",
        )
        self.assertEqual(
            output.call_args.args[0],
            {
                "mode": "experiment_registry_verified",
                "registry_fingerprint": "c" * 64,
                "results_included": True,
                "labels_loaded": True,
                "scientifically_attested": False,
            },
        )

    def test_registry_contract_error_is_an_argparse_error_without_json(self) -> None:
        stderr = io.StringIO()
        with (
            patch.object(cli, "REPOSITORY_ROOT", CLI_ROOT),
            patch.object(
                cli,
                "write_experiment_preregistration",
                side_effect=TrainingError("result artifacts already exist"),
            ),
            patch.object(cli, "_print") as output,
            redirect_stderr(stderr),
            self.assertRaisesRegex(SystemExit, "2"),
        ):
            cli.main(
                [
                    "preregister-experiment",
                    "--output",
                    "artifacts/preregistration.json",
                    "--initial-raw-run-dir",
                    "runs/initial-raw",
                    "--initial-synthesis-config",
                    "configs/initial-synthesis.toml",
                    "--training-run-dir",
                    "runs/training",
                ]
            )

        output.assert_not_called()
        self.assertIn("error: result artifacts already exist", stderr.getvalue())

    def test_finalize_parser_requires_all_six_registered_inputs(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaisesRegex(SystemExit, "2"):
            cli.build_parser().parse_args(
                [
                    "finalize-experiment",
                    "--output",
                    "registry.json",
                    "--preregistration",
                    "preregistration.json",
                ]
            )
        self.assertIn("--initial-raw-run-dir", stderr.getvalue())
        self.assertIn("--trained-synthesis-run-dir", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
