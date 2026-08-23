from __future__ import annotations

import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from compute_as_a_teacher.training import cli  # noqa: E402


CLI_ROOT = Path("/launch-approval-cli-root")


class LaunchApprovalCliTests(unittest.TestCase):
    def test_inspect_evidence_routes_all_qualification_inputs(self) -> None:
        evidence = {
            "reviewed_evidence_fingerprint": "e" * 64,
            "qualifications": {
                "one_step": {},
                "resume_three_step": {},
                "full_shape_five_step": {},
            },
        }
        with (
            patch.object(cli, "REPOSITORY_ROOT", CLI_ROOT),
            patch.object(
                cli, "build_launch_evidence", return_value=evidence
            ) as inspect,
            patch.object(cli, "exclusive_launch", return_value=nullcontext()),
            patch.object(cli, "_print") as output,
        ):
            status = cli.main(
                [
                    "inspect-launch-evidence",
                    "--preregistration", "preregistration.json",
                    "--training-run-dir", "training",
                    "--one-step-dir", "one-step",
                    "--resume-three-step-dir", "resume-three-step",
                    "--full-shape-five-step-dir", "full-shape-five-step",
                ]
            )
        self.assertEqual(status, 0)
        inspect.assert_called_once_with(
            CLI_ROOT / "preregistration.json",
            CLI_ROOT / "training",
            CLI_ROOT / "one-step",
            CLI_ROOT / "resume-three-step",
            CLI_ROOT / "full-shape-five-step",
        )
        self.assertEqual(
            output.call_args.args[0]["reviewed_evidence_fingerprint"],
            "e" * 64,
        )

    def test_write_routes_all_bound_evidence(self) -> None:
        approval = {
            "launch_approval_fingerprint": "a" * 64,
            "approved_for_canonical_launch": True,
            "scientifically_attested": False,
        }
        with (
            patch.object(cli, "REPOSITORY_ROOT", CLI_ROOT),
            patch.object(cli, "write_launch_approval", return_value=approval) as write,
            patch.object(cli, "_print") as output,
        ):
            status = cli.main(
                [
                    "write-launch-approval",
                    "--output", "approval.json",
                    "--preregistration", "preregistration.json",
                    "--training-run-dir", "training",
                    "--one-step-dir", "one-step",
                    "--resume-three-step-dir", "resume-three-step",
                    "--full-shape-five-step-dir", "full-shape-five-step",
                    "--manual-attestation", "manual.json",
                    "--force",
                ]
            )
        self.assertEqual(status, 0)
        write.assert_called_once_with(
            CLI_ROOT / "approval.json",
            CLI_ROOT / "preregistration.json",
            CLI_ROOT / "training",
            CLI_ROOT / "one-step",
            CLI_ROOT / "resume-three-step",
            CLI_ROOT / "full-shape-five-step",
            CLI_ROOT / "manual.json",
            force=True,
        )
        self.assertEqual(
            output.call_args.args[0]["mode"],
            "canonical_launch_approval_written",
        )

    def test_inspect_reverifies_current_artifacts(self) -> None:
        approval = {
            "launch_approval_fingerprint": "b" * 64,
            "approved_for_canonical_launch": True,
            "scientifically_attested": False,
            "qualifications": {
                "one_step": {},
                "resume_three_step": {},
                "full_shape_five_step": {},
            },
        }
        with (
            patch.object(cli, "REPOSITORY_ROOT", CLI_ROOT),
            patch.object(cli, "verify_launch_approval", return_value=approval) as verify,
            patch.object(cli, "exclusive_launch", return_value=nullcontext()),
            patch.object(cli, "_print") as output,
        ):
            status = cli.main(
                [
                    "inspect-launch-approval",
                    "--launch-approval", "approval.json",
                    "--preregistration", "preregistration.json",
                    "--training-run-dir", "training",
                ]
            )
        self.assertEqual(status, 0)
        verify.assert_called_once_with(
            CLI_ROOT / "approval.json",
            CLI_ROOT / "preregistration.json",
            CLI_ROOT / "training",
        )
        self.assertEqual(
            output.call_args.args[0]["qualification_profiles"],
            ["full_shape_five_step", "one_step", "resume_three_step"],
        )


if __name__ == "__main__":
    unittest.main()
