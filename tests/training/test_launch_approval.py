from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from compute_as_a_teacher.evaluation.artifacts import (  # noqa: E402
    canonical_json_bytes,
    sha256_bytes,
)
from compute_as_a_teacher.training.errors import TrainingError  # noqa: E402
from compute_as_a_teacher.training.launch_approval import (  # noqa: E402
    MANUAL_ATTESTATION_KIND,
    REQUIRED_MANUAL_ATTESTATIONS,
    REQUIRED_QUALIFICATION_PROFILES,
    build_launch_evidence,
    verify_launch_approval,
    write_launch_approval,
)
from compute_as_a_teacher.training.qualification import (  # noqa: E402
    QUALIFICATION_PROFILES,
)
from compute_as_a_teacher.training.verl_adapter import VerlCommand  # noqa: E402
from compute_as_a_teacher.training.verl_adapter import exclusive_launch  # noqa: E402


MODULE = "compute_as_a_teacher.training.launch_approval"


class LaunchApprovalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.training_dir = self.root / "training"
        self.training_dir.mkdir()
        (self.training_dir / "manifest.json").write_text(
            '{"training":true}\n', encoding="utf-8"
        )
        self.preregistration_path = self.root / "preregistration.json"
        self.preregistration_path.write_text(
            '{"preregistered":true}\n', encoding="utf-8"
        )
        self.initial_raw_dir = self.root / "initial-raw"
        self.initial_raw_dir.mkdir()
        self.initial_synthesis_config_path = self.root / "initial-synthesis.toml"
        self.initial_synthesis_config_path.write_text(
            'kind = "synthesis"\n', encoding="utf-8"
        )
        self.approval_path = self.root / "launch-approval.json"
        self.attestation_path = self.root / "manual-attestation.json"
        self.training_manifest = {
            "plan_fingerprint": "a" * 64,
            "config_fingerprint": "b" * 64,
            "config": {
                "runtime": {
                    "nodes": 1,
                    "gpus_per_node": 8,
                    "anchor_model": "cat-frozen-qwen3-4b",
                }
            },
        }
        self.preregistration = {
            "preregistration_fingerprint": "c" * 64,
            "stages": {
                "initial_raw": {"run_dir": str(self.initial_raw_dir)},
                "initial_synthesis_config": {
                    "path": str(self.initial_synthesis_config_path)
                },
                "canonical_training": {"run_dir": str(self.training_dir)},
            },
        }
        self.qualification_dirs = {
            name: self.root / name for name in REQUIRED_QUALIFICATION_PROFILES
        }
        self.manifests: dict[str, dict[str, object]] = {}
        self.commands: dict[str, VerlCommand] = {}
        for index, name in enumerate(REQUIRED_QUALIFICATION_PROFILES):
            profile = QUALIFICATION_PROFILES[name]
            run_dir = self.qualification_dirs[name]
            run_dir.mkdir()
            command = VerlCommand(
                argv=("/opt/verl/bin/python", "-m", "verl.trainer.main_ppo"),
                cwd="/opt/verl/source",
                environment={"PYTHONPATH": "/repo/src"},
                framework_revision="d" * 40,
                adapter_version="cat-verl-batch-reward-v2",
            )
            manifest: dict[str, object] = {
                "profile": {
                    "name": profile.name,
                    "prompt_count": profile.prompt_count,
                    "prompt_batch_size": profile.prompt_batch_size,
                    "max_steps": profile.max_steps,
                    "save_every_steps": profile.save_every_steps,
                },
                "source": {
                    "run_dir": str(self.training_dir.resolve()),
                    "plan_fingerprint": self.training_manifest["plan_fingerprint"],
                    "config_fingerprint": self.training_manifest[
                        "config_fingerprint"
                    ],
                },
                "artifacts": {
                    "training_data": {"sha256": f"{index + 1}" * 64},
                    "verl_command": {"sha256": f"{index + 4}" * 64},
                },
                "qualification_fingerprint": f"{index + 7}" * 64,
            }
            self.manifests[name] = manifest
            self.commands[name] = command
            (run_dir / "qualification_manifest.json").write_text(
                json.dumps({"profile": name}), encoding="utf-8"
            )
            (run_dir / "logs").mkdir()
            (run_dir / "logs/trainer.log").write_text(
                "finite metrics\n", encoding="utf-8"
            )
            (run_dir / "rollout_logs").mkdir()
            (run_dir / "rollout_logs/step.jsonl").write_text(
                '{"reward":1}\n', encoding="utf-8"
            )
            checkpoint_root = run_dir / "checkpoints"
            terminal = checkpoint_root / f"global_step_{profile.max_steps}"
            terminal.mkdir(parents=True)
            (terminal / "state.pt").write_bytes(b"checkpoint")
            (checkpoint_root / "latest_checkpointed_iteration.txt").write_text(
                f"{profile.max_steps}\n", encoding="utf-8"
            )
            receipt = {
                "schema_version": 2,
                "kind": "cat_training_preflight",
                "training_plan_fingerprint": self.training_manifest[
                    "plan_fingerprint"
                ],
                "config_fingerprint": self.training_manifest["config_fingerprint"],
                "command_fingerprint": command.fingerprint,
                "qualification_lineage": {
                    "profile": manifest["profile"],
                    "qualification_fingerprint": manifest[
                        "qualification_fingerprint"
                    ],
                    "training_data_sha256": manifest["artifacts"][
                        "training_data"
                    ]["sha256"],
                    "command_sha256": manifest["artifacts"]["verl_command"][
                        "sha256"
                    ],
                },
                "checks": {
                    "model_snapshot": {"all_files_tree_sha256": "e" * 64},
                    "runtime": {},
                    "hydra_composition": {},
                    "tokenizer": {
                        "anchor_context_canary_required_tokens": 14_000,
                        "model_context_tokens": 32_768,
                    },
                    "anchor": {
                        "model": "cat-frozen-qwen3-4b",
                        "endpoint_sha256": "1" * 64,
                        "prompt_sha256": "2" * 64,
                        "response_sha256": "3" * 64,
                        "finish_reason": "stop",
                        "anchor_extraction_status": "ok",
                        "unanimous_agreement_rewards": [1] * 8,
                        "long_context_request_accepted": True,
                        "long_context_tail_answer_preserved": True,
                        "context_prompt_sha256": "4" * 64,
                        "context_response_sha256": "5" * 64,
                        "context_finish_reason": "stop",
                        "context_expected_answer_sha256": "6" * 64,
                    },
                },
                "operationally_ready_to_launch": True,
                "missing_gates": [],
                "scientifically_attested": False,
                "attestation_limitations": ["external"],
            }
            receipt["preflight_fingerprint"] = sha256_bytes(
                canonical_json_bytes(receipt)
            )
            (run_dir / "preflight.json").write_bytes(
                canonical_json_bytes(receipt)
            )
        with self._patched_runtime():
            evidence = build_launch_evidence(
                self.preregistration_path,
                self.training_dir,
                self.qualification_dirs["one_step"],
                self.qualification_dirs["resume_three_step"],
                self.qualification_dirs["full_shape_five_step"],
            )
        self.reviewed_evidence_fingerprint = evidence[
            "reviewed_evidence_fingerprint"
        ]
        self._write_attestation()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_attestation(self, *, accepted: bool = True) -> None:
        value = {
            "schema_version": 1,
            "kind": MANUAL_ATTESTATION_KIND,
            "attested_by": "experiment-owner",
            "attested_at_utc": "2026-08-23T12:00:00Z",
            "reviewed_evidence_fingerprint": self.reviewed_evidence_fingerprint,
            "attestations": {
                name: accepted for name in REQUIRED_MANUAL_ATTESTATIONS
            },
            "evidence": {
                name: f"Reviewed bound evidence for {name}."
                for name in REQUIRED_MANUAL_ATTESTATIONS
            },
            "budget_limits": {
                "max_wall_time_seconds": 86_400,
                "max_trainer_gpu_hours": 192.0,
                "max_anchor_gpu_hours": 24.0,
                "max_storage_bytes": 2_000_000_000_000,
                "max_total_cost": 1_500.0,
                "currency": "USD",
                "trainer_gpu_hour_rate": 5.0,
                "anchor_gpu_hour_rate": 2.0,
                "storage_and_network_cost": 100.0,
            },
        }
        self.attestation_path.write_text(json.dumps(value), encoding="utf-8")

    def _load_qualification(self, run_dir: Path):
        name = Path(run_dir).name
        return copy.deepcopy(self.manifests[name]), self.commands[name]

    def _checkpointed_step(self, run_dir: Path, max_steps: int, **_: object) -> int:
        self.assertEqual(max_steps, QUALIFICATION_PROFILES[Path(run_dir).name].max_steps)
        return max_steps

    @contextmanager
    def _patched_runtime(self):
        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    f"{MODULE}.verify_preregistered_training_stage",
                    return_value=copy.deepcopy(self.preregistration),
                )
            )
            stack.enter_context(
                patch(
                    f"{MODULE}.load_training_plan",
                    return_value=(copy.deepcopy(self.training_manifest), object()),
                )
            )
            stack.enter_context(
                patch(f"{MODULE}.load_qualification_plan", self._load_qualification)
            )
            stack.enter_context(
                patch(f"{MODULE}.checkpointed_step", self._checkpointed_step)
            )
            yield

    def _write(self) -> dict[str, object]:
        return write_launch_approval(
            self.approval_path,
            self.preregistration_path,
            self.training_dir,
            self.qualification_dirs["one_step"],
            self.qualification_dirs["resume_three_step"],
            self.qualification_dirs["full_shape_five_step"],
            self.attestation_path,
        )

    def test_approval_binds_every_completed_profile_and_manual_gate(self) -> None:
        with self._patched_runtime():
            written = self._write()
            verified = verify_launch_approval(
                self.approval_path,
                self.preregistration_path,
                self.training_dir,
            )
        self.assertEqual(written, verified)
        self.assertTrue(written["approved_for_canonical_launch"])
        self.assertFalse(written["scientifically_attested"])
        self.assertEqual(
            set(written["qualifications"]), set(REQUIRED_QUALIFICATION_PROFILES)
        )
        self.assertTrue(
            all(written["manual_attestation"]["attestations"].values())
        )
        self.assertEqual(
            written["manual_attestation"]["budget_limits"]["max_wall_time_seconds"],
            86_400,
        )

    def test_false_manual_attestation_fails_closed(self) -> None:
        self._write_attestation(accepted=False)
        with self._patched_runtime(), self.assertRaisesRegex(
            TrainingError, "Every required manual"
        ):
            self._write()
        self.assertFalse(self.approval_path.exists())

    def test_incomplete_qualification_fails_closed(self) -> None:
        def incomplete(run_dir: Path, max_steps: int, **_: object) -> int:
            return 0 if Path(run_dir).name == "resume_three_step" else max_steps

        with self._patched_runtime(), patch(
            f"{MODULE}.checkpointed_step", incomplete
        ), self.assertRaisesRegex(TrainingError, "no verified terminal checkpoint"):
            self._write()

    def test_missing_anchor_context_canary_fails_closed(self) -> None:
        path = self.qualification_dirs["one_step"] / "preflight.json"
        receipt = json.loads(path.read_text(encoding="utf-8"))
        receipt["checks"]["anchor"]["long_context_tail_answer_preserved"] = False
        receipt.pop("preflight_fingerprint")
        receipt["preflight_fingerprint"] = sha256_bytes(
            canonical_json_bytes(receipt)
        )
        path.write_bytes(canonical_json_bytes(receipt))
        with self._patched_runtime(), self.assertRaisesRegex(
            TrainingError, "long-context tail canary"
        ):
            self._write()

    def test_underfunded_numeric_budget_fails_closed(self) -> None:
        value = json.loads(self.attestation_path.read_text(encoding="utf-8"))
        value["budget_limits"]["max_total_cost"] = 1.0
        self.attestation_path.write_text(json.dumps(value), encoding="utf-8")
        with self._patched_runtime(), self.assertRaisesRegex(
            TrainingError, "does not cover"
        ):
            self._write()

    def test_placeholder_evidence_fails_closed(self) -> None:
        value = json.loads(self.attestation_path.read_text(encoding="utf-8"))
        gate = REQUIRED_MANUAL_ATTESTATIONS[0]
        value["evidence"][gate] = "TBD after the GPU run"
        self.attestation_path.write_text(json.dumps(value), encoding="utf-8")
        with self._patched_runtime(), self.assertRaisesRegex(
            TrainingError, "Every required manual"
        ):
            self._write()

    def test_underscore_placeholder_identity_fails_closed(self) -> None:
        value = json.loads(self.attestation_path.read_text(encoding="utf-8"))
        value["attested_by"] = "  replace_with_reviewer_identity  "
        self.attestation_path.write_text(json.dumps(value), encoding="utf-8")
        with self._patched_runtime(), self.assertRaisesRegex(
            TrainingError, "requires attested_by"
        ):
            self._write()

    def test_nonpositive_resource_ceiling_fails_closed(self) -> None:
        value = json.loads(self.attestation_path.read_text(encoding="utf-8"))
        value["budget_limits"]["max_anchor_gpu_hours"] = 0
        self.attestation_path.write_text(json.dumps(value), encoding="utf-8")
        with self._patched_runtime(), self.assertRaisesRegex(
            TrainingError, "finite positive"
        ):
            self._write()

    def test_bound_log_tampering_invalidates_approval(self) -> None:
        with self._patched_runtime():
            self._write()
        (self.qualification_dirs["one_step"] / "logs/trainer.log").write_text(
            "changed\n", encoding="utf-8"
        )
        with self._patched_runtime(), self.assertRaisesRegex(
            TrainingError, "reviewed_evidence_fingerprint"
        ):
            verify_launch_approval(
                self.approval_path,
                self.preregistration_path,
                self.training_dir,
            )

    def test_evidence_rejects_a_running_qualification(self) -> None:
        with exclusive_launch(self.qualification_dirs["one_step"]):
            with self._patched_runtime(), self.assertRaisesRegex(
                TrainingError, "already locked"
            ):
                build_launch_evidence(
                    self.preregistration_path,
                    self.training_dir,
                    self.qualification_dirs["one_step"],
                    self.qualification_dirs["resume_three_step"],
                    self.qualification_dirs["full_shape_five_step"],
                )

    def test_approval_output_must_be_outside_bound_runs(self) -> None:
        output = self.training_dir / "math500_train.jsonl"
        with self._patched_runtime(), self.assertRaisesRegex(
            TrainingError, "outside every bound run"
        ):
            write_launch_approval(
                output,
                self.preregistration_path,
                self.training_dir,
                self.qualification_dirs["one_step"],
                self.qualification_dirs["resume_three_step"],
                self.qualification_dirs["full_shape_five_step"],
                self.attestation_path,
                force=True,
            )

    def test_qualification_evidence_directories_must_not_be_nested(self) -> None:
        nested = self.qualification_dirs["one_step"] / "resume_three_step"
        with self._patched_runtime(), self.assertRaisesRegex(
            TrainingError, "pairwise disjoint"
        ):
            build_launch_evidence(
                self.preregistration_path,
                self.training_dir,
                self.qualification_dirs["one_step"],
                nested,
                self.qualification_dirs["full_shape_five_step"],
            )

    def test_approval_output_cannot_replace_initial_evaluation_artifacts(self) -> None:
        outputs = (
            self.initial_raw_dir / "manifest.json",
            self.initial_synthesis_config_path,
        )
        for output in outputs:
            with self.subTest(output=output), self._patched_runtime(), self.assertRaises(
                TrainingError
            ):
                write_launch_approval(
                    output,
                    self.preregistration_path,
                    self.training_dir,
                    self.qualification_dirs["one_step"],
                    self.qualification_dirs["resume_three_step"],
                    self.qualification_dirs["full_shape_five_step"],
                    self.attestation_path,
                    force=True,
                )


if __name__ == "__main__":
    unittest.main()
