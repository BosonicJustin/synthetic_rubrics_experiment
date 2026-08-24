from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from compute_as_a_teacher import server  # noqa: E402
from compute_as_a_teacher.training.errors import TrainingError  # noqa: E402


EXAMPLE = REPOSITORY_ROOT / "configs/server/math500_server.example.toml"


def completed(argv, value=None, *, returncode=0):
    return subprocess.CompletedProcess(
        argv,
        returncode,
        json.dumps(value or {"mode": "ok"}) + "\n",
        "",
    )


def registered_scoring_config(workflow):
    return {
        "stages": {
            "scoring_config": {"path": str(workflow.scoring_config.resolve())}
        }
    }


class ServerWorkflowTests(unittest.TestCase):
    def test_phase_is_preview_only_by_default(self) -> None:
        workflow = server.load_server_workflow(
            EXAMPLE, repository_root=REPOSITORY_ROOT
        )
        result = server.preview_phase(workflow, "canonical")
        self.assertFalse(result["would_execute"])
        commands = result["commands"]
        self.assertEqual(
            [item["name"] for item in commands],
            ["verify_launch_approval", "launch_canonical"],
        )
        self.assertNotIn("--execute", commands[-1]["argv"])
        self.assertIn("--preregistration", commands[-1]["argv"])
        self.assertIn("--launch-approval", commands[-1]["argv"])

    def test_registry_commands_bind_the_scoring_config(self) -> None:
        workflow = server.load_server_workflow(
            EXAMPLE, repository_root=REPOSITORY_ROOT
        )
        commands = (
            *server.phase_commands(workflow, "preregister"),
            *server.phase_commands(workflow, "finalize"),
        )
        self.assertEqual(len(commands), 3)
        for command in commands:
            with self.subTest(command=command.name):
                self.assertEqual(command.argv.count("--scoring-config"), 1)
                index = command.argv.index("--scoring-config")
                self.assertEqual(
                    command.argv[index + 1], str(workflow.scoring_config)
                )

    def test_execute_adds_flag_only_to_existing_launch_cli(self) -> None:
        workflow = server.load_server_workflow(
            EXAMPLE, repository_root=REPOSITORY_ROOT
        )
        observed = []

        def runner(argv, **kwargs):
            observed.append((argv, kwargs))
            return completed(argv)

        with patch.dict(os.environ, {"CAT_SERVICE_ROLE": "trainer"}, clear=True):
            result = server.execute_phase(workflow, "canonical", runner=runner)
        self.assertTrue(result["complete"])
        self.assertEqual(len(observed), 2)
        self.assertNotIn("--execute", observed[0][0])
        self.assertIn("--execute", observed[1][0])
        self.assertEqual(observed[1][0][2], "launch")

    def test_real_canonical_wrapper_execs_the_guarded_cli_for_signal_delivery(self) -> None:
        workflow = server.load_server_workflow(
            EXAMPLE, repository_root=REPOSITORY_ROOT
        )
        observed = []

        def runner(argv, **_kwargs):
            observed.append(argv)
            return completed(argv)

        with patch.object(
            server.os,
            "execvpe",
            side_effect=RuntimeError("process replaced"),
        ) as replace:
            with self.assertRaisesRegex(RuntimeError, "process replaced"):
                with patch.dict(
                    os.environ, {"CAT_SERVICE_ROLE": "trainer"}, clear=True
                ):
                    server.execute_phase(
                        workflow,
                        "canonical",
                        runner=runner,
                        replace_final_process=True,
                    )
        self.assertEqual(len(observed), 1)
        self.assertIn("--execute", replace.call_args.args[1])

    def test_baseline_generation_requires_current_preregistration_before_endpoint_calls(self) -> None:
        workflow = server.load_server_workflow(
            EXAMPLE, repository_root=REPOSITORY_ROOT
        )
        with patch.object(
            server,
            "verify_preregistered_training_stage",
            side_effect=TrainingError("preregistration missing"),
        ):
            with self.assertRaisesRegex(TrainingError, "preregistration missing"):
                with patch.dict(
                    os.environ,
                    {
                        "CAT_SERVICE_ROLE": "evaluator",
                        "CAT_ANCHOR_API_KEY": "anchor",
                    },
                    clear=True,
                ):
                    server.execute_phase(
                        workflow,
                        "baseline-generation",
                        runner=lambda *_args, **_kwargs: self.fail("must not execute"),
                    )

    def test_baseline_scoring_requires_terminal_artifact_handoff(self) -> None:
        workflow = server.load_server_workflow(
            EXAMPLE, repository_root=REPOSITORY_ROOT
        )
        with (
            patch.object(
                server,
                "verify_preregistered_training_stage",
                return_value=registered_scoring_config(workflow),
            ),
            patch.object(
                server,
                "_verify_trained_handoff",
                side_effect=TrainingError("terminal handoff missing"),
            ) as verify_handoff,
            patch.dict(os.environ, {"CAT_SERVICE_ROLE": "scorer"}, clear=True),
            self.assertRaisesRegex(TrainingError, "terminal handoff missing"),
        ):
            server.execute_phase(
                workflow,
                "baseline-scoring",
                runner=lambda *_args, **_kwargs: self.fail("must not execute"),
            )
        verify_handoff.assert_called_once_with(workflow, artifact_only=True)

    def test_scorer_precondition_uses_artifacts_without_a_model_probe(self) -> None:
        workflow = server.load_server_workflow(
            EXAMPLE, repository_root=REPOSITORY_ROOT
        )
        with (
            patch.object(
                server,
                "verify_preregistered_training_stage",
                return_value=registered_scoring_config(workflow),
            ),
            patch.object(server, "_verify_trained_handoff") as verify_handoff,
            patch.object(server, "_verify_all_generation_complete"),
            patch.dict(os.environ, {"CAT_SERVICE_ROLE": "scorer"}, clear=True),
        ):
            result = server.execute_phase(
                workflow,
                "baseline-scoring",
                runner=lambda argv, **_kwargs: completed(argv),
            )
        self.assertTrue(result["complete"])
        verify_handoff.assert_called_once_with(workflow, artifact_only=True)

    def test_scorer_requires_the_preregistered_scoring_config_path(self) -> None:
        workflow = server.load_server_workflow(
            EXAMPLE, repository_root=REPOSITORY_ROOT
        )
        with (
            patch.object(
                server,
                "verify_preregistered_training_stage",
                return_value={
                    "stages": {
                        "scoring_config": {"path": "/different/scoring.toml"}
                    }
                },
            ),
            patch.object(server, "_verify_trained_handoff") as verify_handoff,
            patch.dict(os.environ, {"CAT_SERVICE_ROLE": "scorer"}, clear=True),
            self.assertRaisesRegex(
                server.ServerWorkflowError,
                "does not match the experiment preregistration",
            ),
        ):
            server.execute_phase(
                workflow,
                "baseline-scoring",
                runner=lambda *_args, **_kwargs: self.fail("must not score"),
            )
        verify_handoff.assert_not_called()

    def test_baseline_endpoint_argv_has_one_safe_base_url(self) -> None:
        workflow = server.load_server_workflow(
            EXAMPLE, repository_root=REPOSITORY_ROOT
        )
        commands = server.phase_commands(workflow, "baseline-generation")
        endpoint_commands = (
            commands[0],
            commands[1],
            commands[3],
        )
        for command in endpoint_commands:
            with self.subTest(command=command.name):
                self.assertEqual(command.argv.count("--base-url"), 1)
                index = command.argv.index("--base-url")
                self.assertEqual(
                    command.argv[index + 1], "http://anchor:8001/v1"
                )
                self.assertEqual(command.argv.count("--api-key-env"), 1)

    def test_generation_and_scoring_phases_have_disjoint_service_scopes(self) -> None:
        workflow = server.load_server_workflow(
            EXAMPLE, repository_root=REPOSITORY_ROOT
        )
        baseline_generation = server.phase_commands(
            workflow, "baseline-generation"
        )
        baseline_scoring = server.phase_commands(workflow, "baseline-scoring")
        trained_generation = server.phase_commands(
            workflow, "trained-eval-generation"
        )
        trained_scoring = server.phase_commands(
            workflow, "trained-eval-scoring"
        )
        self.assertFalse(
            any(command.name.startswith("score_") for command in baseline_generation)
        )
        self.assertTrue(
            all(command.name.startswith("score_") for command in baseline_scoring)
        )
        self.assertFalse(
            any(command.name.startswith("score_") for command in trained_generation)
        )
        self.assertTrue(
            all(command.name.startswith("score_") for command in trained_scoring)
        )
        self.assertEqual(
            [command.name for command in baseline_scoring],
            ["score_initial_synthesis"],
        )
        self.assertEqual(
            [command.name for command in trained_scoring],
            ["score_trained_synthesis"],
        )
        for command in (*baseline_scoring, *trained_scoring):
            self.assertIn("score-synthesis", command.argv)
            self.assertNotIn("score-raw", command.argv)
        self.assertEqual(
            server.preview_phase(workflow, "baseline-generation")["execution_scope"],
            "evaluator",
        )
        self.assertEqual(
            server.preview_phase(workflow, "baseline-scoring")["execution_scope"],
            "scorer",
        )
        self.assertEqual(
            server.preview_phase(workflow, "finalize")["execution_scope"],
            "scorer",
        )
        with (
            patch.dict(os.environ, {"CAT_SERVICE_ROLE": "trainer"}, clear=True),
            self.assertRaisesRegex(
                server.ServerWorkflowError,
                "CAT_SERVICE_ROLE=evaluator",
            ),
        ):
            server.execute_phase(
                workflow,
                "baseline-generation",
                runner=lambda *_args, **_kwargs: self.fail("must not execute"),
            )

    def test_resume_qualification_stays_a_registered_profile(self) -> None:
        workflow = server.load_server_workflow(
            EXAMPLE, repository_root=REPOSITORY_ROOT
        )
        commands = server.phase_commands(
            workflow,
            "qualification",
            qualification_profile="resume_three_step",
        )
        self.assertEqual(len(commands), 2)
        self.assertIn("resume_three_step", commands[0].argv)
        self.assertNotIn("--execute", commands[1].argv)
        with self.assertRaisesRegex(
            server.ServerWorkflowError, "supervised and interrupted"
        ):
            server.phase_commands(
                workflow,
                "qualification",
                execute=True,
                qualification_profile="resume_three_step",
            )
        prepare = server.phase_commands(
            workflow,
            "qualification",
            execute=True,
            qualification_profile="resume_three_step",
            resume_action="prepare",
        )
        self.assertEqual([item.name for item in prepare], ["prepare_resume_three_step"])
        restart = server.phase_commands(
            workflow,
            "qualification",
            execute=True,
            qualification_profile="resume_three_step",
            resume_action="restart",
        )
        self.assertEqual([item.name for item in restart], ["launch_resume_three_step"])
        self.assertIn("--execute", restart[0].argv)
        initial = server.phase_commands(
            workflow,
            "qualification",
            execute=True,
            qualification_profile="resume_three_step",
            resume_action="initial",
        )
        self.assertEqual([item.name for item in initial], ["launch_resume_three_step"])
        with self.assertRaisesRegex(server.ServerWorkflowError, "--profile"):
            server.phase_commands(workflow, "qualification")

    def test_restart_requires_a_verified_step_one_checkpoint(self) -> None:
        workflow = server.load_server_workflow(
            EXAMPLE, repository_root=REPOSITORY_ROOT
        )
        config = SimpleNamespace(
            runtime=SimpleNamespace(nodes=1, gpus_per_node=8)
        )
        with (
            patch.object(server, "verify_preregistered_training_stage"),
            patch.object(server, "load_training_config", return_value=config),
            patch.object(server, "checkpointed_step", return_value=0),
        ):
            with self.assertRaisesRegex(server.ServerWorkflowError, "step 1"):
                with patch.dict(
                    os.environ, {"CAT_SERVICE_ROLE": "trainer"}, clear=True
                ):
                    server.execute_phase(
                        workflow,
                        "qualification",
                        qualification_profile="resume_three_step",
                        resume_action="restart",
                        runner=lambda *_args, **_kwargs: self.fail("must not execute"),
                    )
        with (
            patch.object(server, "verify_preregistered_training_stage"),
            patch.object(server, "load_training_config", return_value=config),
            patch.object(server, "checkpointed_step", return_value=1),
        ):
            with self.assertRaisesRegex(server.ServerWorkflowError, "step 0"):
                with patch.dict(
                    os.environ, {"CAT_SERVICE_ROLE": "trainer"}, clear=True
                ):
                    server.execute_phase(
                        workflow,
                        "qualification",
                        qualification_profile="resume_three_step",
                        resume_action="initial",
                        runner=lambda *_args, **_kwargs: self.fail("must not execute"),
                    )

    def test_scoring_waits_until_every_generation_is_complete(self) -> None:
        workflow = server.load_server_workflow(
            EXAMPLE, repository_root=REPOSITORY_ROOT
        )
        with (
            patch.object(
                server,
                "verify_preregistered_training_stage",
                return_value=registered_scoring_config(workflow),
            ),
            patch.object(server, "_verify_trained_handoff"),
            patch.object(
                server,
                "_verify_all_generation_complete",
                side_effect=server.ServerWorkflowError("generation incomplete"),
            ),
            patch.dict(os.environ, {"CAT_SERVICE_ROLE": "scorer"}, clear=True),
            self.assertRaisesRegex(server.ServerWorkflowError, "generation incomplete"),
        ):
            server.execute_phase(
                workflow,
                "baseline-scoring",
                runner=lambda *_args, **_kwargs: self.fail("must not score"),
            )

    def test_finalization_verifies_only_paired_synthesis_scores(self) -> None:
        workflow = server.load_server_workflow(
            EXAMPLE, repository_root=REPOSITORY_ROOT
        )
        scoring_config = object()
        with (
            patch.object(
                server,
                "load_scoring_config",
                return_value=scoring_config,
            ),
            patch.object(server, "score_run") as score_run,
        ):
            server._verify_scored_experiment(workflow)

        self.assertEqual(score_run.call_count, 2)
        expected = (
            (workflow.initial_raw_run, workflow.initial_synthesis_run),
            (workflow.trained_raw_run, workflow.trained_synthesis_run),
        )
        for invocation, (raw_run, synthesis_run) in zip(
            score_run.call_args_list, expected, strict=True
        ):
            self.assertEqual(invocation.args, (synthesis_run, scoring_config))
            self.assertEqual(
                invocation.kwargs,
                {
                    "repository_root": workflow.repository_root,
                    "raw_run_dir": raw_run,
                },
            )

    def test_generation_rejects_existing_label_derived_artifacts(self) -> None:
        for artifact_name in (server.SCORES_NAME, server.PAIRED_SCORES_NAME):
            with self.subTest(artifact_name=artifact_name):
                workflow = server.load_server_workflow(
                    EXAMPLE, repository_root=REPOSITORY_ROOT
                )
                with tempfile.TemporaryDirectory() as temporary:
                    initial_raw = Path(temporary) / "initial-raw"
                    initial_raw.mkdir()
                    (initial_raw / artifact_name).write_text("\n", encoding="utf-8")
                    workflow = replace(workflow, initial_raw_run=initial_raw)
                    with (
                        patch.dict(
                            os.environ,
                            {
                                "CAT_SERVICE_ROLE": "evaluator",
                                "CAT_ANCHOR_API_KEY": "anchor",
                            },
                            clear=True,
                        ),
                        self.assertRaisesRegex(
                            server.ServerWorkflowError,
                            "label-derived artifacts",
                        ),
                    ):
                        server.execute_phase(
                            workflow,
                            "baseline-generation",
                            runner=lambda *_args, **_kwargs: self.fail(
                                "must not generate"
                            ),
                        )

    def test_generation_verification_uses_server_error_contract(self) -> None:
        workflow = server.load_server_workflow(
            EXAMPLE, repository_root=REPOSITORY_ROOT
        )
        with patch.object(
            server,
            "load_plan",
            side_effect=server.EvaluationError("missing plan"),
        ):
            with self.assertRaisesRegex(
                server.ServerWorkflowError,
                "generation phases must be complete",
            ):
                server._verify_all_generation_complete(workflow)

    def test_prepare_is_question_only_and_anchor_preview_contains_no_secret(self) -> None:
        workflow = server.load_server_workflow(
            EXAMPLE, repository_root=REPOSITORY_ROOT
        )
        prepare = server.preview_phase(workflow, "prepare")
        self.assertIn("--verify-questions-only", prepare["commands"][0]["argv"])
        self.assertNotIn("--offline", prepare["commands"][0]["argv"])
        self.assertEqual(prepare["execution_scope"], "evaluator")
        with patch.dict(os.environ, {"CAT_ANCHOR_MODE": "local"}, clear=False):
            anchor = server.preview_phase(workflow, "anchor")
        serialized = json.dumps(anchor)
        self.assertNotIn("CAT_ANCHOR_API_KEY", serialized)
        self.assertFalse(anchor["would_execute"])
        self.assertEqual(anchor["execution_scope"], "host")
        self.assertIn("--wait", anchor["commands"][0]["argv"])
        self.assertIn("--wait-timeout", anchor["commands"][0]["argv"])
        self.assertEqual(workflow.evaluation_base_url, "http://anchor:8001/v1")
        self.assertEqual(workflow.evaluation_api_key_env, "CAT_ANCHOR_API_KEY")

    def test_post_training_phases_bind_distinct_policy_endpoints(self) -> None:
        workflow = server.load_server_workflow(
            EXAMPLE, repository_root=REPOSITORY_ROOT
        )
        handoff = server.phase_commands(workflow, "handoff", execute=True)
        self.assertEqual(
            [command.name for command in handoff],
            [
                "verify_launch_approval",
                "export_and_register_fixed_checkpoint",
                "plan_trained_evaluation",
            ],
        )
        self.assertIn("export-register", handoff[1].argv)
        self.assertEqual(handoff[1].argv.count("--execute"), 1)

        trained_service = server.preview_phase(workflow, "trained-policy")
        self.assertEqual(trained_service["execution_scope"], "host")
        service_commands = trained_service["commands"]
        self.assertEqual(
            [command["name"] for command in service_commands],
            [
                "verify_registered_checkpoint_in_trainer",
                "verify_trained_handoff_in_trainer",
                "start_trained_policy",
            ],
        )
        self.assertIn("inspect-checkpoint", service_commands[0]["argv"])
        self.assertIn("inspect-trained-eval", service_commands[1]["argv"])
        self.assertEqual(service_commands[2]["argv"][-1], "trained-policy")
        self.assertIn("--wait", service_commands[2]["argv"])

        trained_eval = server.phase_commands(workflow, "trained-eval-generation")
        endpoint_commands = {command.name: command for command in trained_eval}
        handoff_check = endpoint_commands["verify_trained_evaluation_handoff"].argv
        self.assertIn("inspect-trained-eval", handoff_check)
        self.assertNotIn("plan-trained-eval", handoff_check)
        raw = endpoint_commands["trained_raw_endpoint_canary"].argv
        synthesis = endpoint_commands["complete_trained_synthesis"].argv
        self.assertEqual(raw[raw.index("--base-url") + 1], server.TRAINED_POLICY_BASE_URL)
        self.assertEqual(
            raw[raw.index("--api-key-env") + 1], "CAT_TRAINED_POLICY_API_KEY"
        )
        self.assertIn("--max-requests", raw)
        self.assertEqual(
            synthesis[synthesis.index("--base-url") + 1], "http://anchor:8001/v1"
        )
        self.assertEqual(
            synthesis[synthesis.index("--api-key-env") + 1], "CAT_ANCHOR_API_KEY"
        )
        final = server.phase_commands(workflow, "finalize")
        self.assertEqual(
            [command.name for command in final],
            ["finalize_experiment_registry", "verify_experiment_registry"],
        )

    def test_trained_eval_requires_both_keys_before_handoff_or_plan_writes(self) -> None:
        workflow = server.load_server_workflow(
            EXAMPLE, repository_root=REPOSITORY_ROOT
        )
        with (
            patch.object(server, "verify_preregistered_training_stage"),
            patch.object(
                server,
                "_verify_trained_handoff",
                side_effect=AssertionError("must not inspect after missing keys"),
            ),
            patch.dict(os.environ, {"CAT_ANCHOR_API_KEY": "anchor"}, clear=True),
        ):
            with self.assertRaisesRegex(
                server.ServerWorkflowError, "CAT_TRAINED_POLICY_API_KEY"
            ):
                with patch.dict(
                    os.environ,
                    {
                        "CAT_SERVICE_ROLE": "evaluator",
                        "CAT_ANCHOR_API_KEY": "anchor",
                    },
                    clear=True,
                ):
                    server.execute_phase(
                        workflow,
                        "trained-eval-generation",
                        runner=lambda *_args, **_kwargs: self.fail("must not execute"),
                    )

    def test_provision_export_parent_is_private_and_keeps_target_absent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            workflow = SimpleNamespace(
                output_root=root,
                checkpoint_export=(
                    root / "exports/qwen3-4b-math500-cat-step-1000"
                ),
            )
            result = server.provision_export_parent(workflow)
            self.assertEqual(result["mode"], "700")
            self.assertTrue((root / "exports").is_dir())
            self.assertFalse(workflow.checkpoint_export.exists())
            self.assertEqual(server.provision_export_parent(workflow), result)

    def test_remote_anchor_is_external_and_local_uses_host_validator(self) -> None:
        workflow = server.load_server_workflow(
            EXAMPLE, repository_root=REPOSITORY_ROOT
        )
        with patch.dict(
            os.environ,
            {"CAT_ANCHOR_MODE": "remote"},
            clear=True,
        ):
            result = server.execute_phase(
                workflow,
                "anchor",
                runner=lambda *_args, **_kwargs: self.fail("must not execute"),
            )
        self.assertTrue(result["skipped"])
        self.assertEqual(result["commands"], [])
        with patch.dict(
            os.environ,
            {
                "CAT_ANCHOR_MODE": "remote",
                "CAT_ANCHOR_GPU_DEVICE": "8",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(server.ServerWorkflowError, "forbids"):
                server.preview_phase(workflow, "anchor")

        with patch.dict(os.environ, {"CAT_ANCHOR_MODE": "local"}, clear=True):
            with self.assertRaisesRegex(
                server.ServerWorkflowError, "validate_server_env.py failed"
            ):
                server.execute_phase(
                    workflow,
                    "anchor",
                    runner=lambda argv, **_kwargs: subprocess.CompletedProcess(
                        argv,
                        1,
                        "",
                        "server environment validation failed: topology mismatch\n",
                    ),
                )

        observed = []

        def runner(argv, **_kwargs):
            observed.append(argv)
            if Path(argv[1]).name == "validate_server_env.py":
                return completed(
                    argv,
                    {"ready": True, "anchor_mode": "local"},
                )
            if argv[0] == "git" and "rev-parse" in argv:
                return subprocess.CompletedProcess(argv, 0, "a" * 40 + "\n", "")
            if argv[0] == "git":
                return subprocess.CompletedProcess(argv, 0, "", "")
            return subprocess.CompletedProcess(argv, 0, "", "")

        with patch.dict(os.environ, {"CAT_ANCHOR_MODE": "local"}, clear=True):
            result = server.execute_phase(workflow, "anchor", runner=runner)
        self.assertTrue(result["complete"])
        self.assertEqual(Path(observed[0][1]).name, "validate_server_env.py")
        self.assertEqual(observed[-1][0:2], ["docker", "compose"])

    def test_loader_rejects_output_escape(self) -> None:
        text = EXAMPLE.read_text(encoding="utf-8").replace(
            'training_run = "/mnt/outputs/training/qwen3-4b-math500-cat"',
            'training_run = "/tmp/outside-training"',
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "server.toml"
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(server.ServerWorkflowError, "below output_root"):
                server.load_server_workflow(path, repository_root=REPOSITORY_ROOT)

    def test_loader_rejects_anchor_argv_that_could_carry_a_secret(self) -> None:
        text = EXAMPLE.read_text(encoding="utf-8").replace(
            '["docker", "compose", "-f", "infra/server/compose.yaml", "--profile", "local-anchor", "up", "-d", "--wait", "--wait-timeout", "600", "anchor"]',
            '["python", "serve.py", "--api-key", "do-not-store-this"]',
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "server.toml"
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(server.ServerWorkflowError, "exact health-waiting"):
                server.load_server_workflow(path, repository_root=REPOSITORY_ROOT)

    def test_loader_rejects_endpoint_secrets_before_preview(self) -> None:
        unsafe_urls = (
            "http://operator:credential@anchor:8001/v1",
            "http://anchor:8001/v1?api_key=credential",
            "http://anchor:8001/v1#credential",
            "http://anchor:8001/private/credential",
            "http://anchor:8001/v1\ncredential",
        )
        for unsafe in unsafe_urls:
            with self.subTest(unsafe=unsafe), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "server.toml"
                path.write_text(
                    EXAMPLE.read_text(encoding="utf-8").replace(
                        "http://anchor:8001/v1", unsafe
                    ),
                    encoding="utf-8",
                )
                with self.assertRaises(server.ServerWorkflowError) as raised:
                    server.load_server_workflow(
                        path, repository_root=REPOSITORY_ROOT
                    )
                self.assertNotIn("credential", str(raised.exception))

    def test_repository_check_accepts_strict_immutable_image_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "repo"
            scripts = root / "scripts"
            metadata = parent / "image-metadata"
            scripts.mkdir(parents=True)
            metadata.mkdir()
            (root / "pyproject.toml").write_text("", encoding="utf-8")
            for name in (
                "prepare_math500.py",
                "evaluate_math500.py",
                "train_math500.py",
            ):
                (scripts / name).write_text("", encoding="utf-8")
            training = root / "training.toml"
            training.write_text(
                "[runtime]\nframework_revision = "
                '"8fdc4d3f202f41461f4de9f42a637228e342668b"\n',
                encoding="utf-8",
            )
            inventory = []
            for source_path in sorted(
                [root / "pyproject.toml", *scripts.iterdir()],
                key=lambda item: item.relative_to(root).as_posix(),
            ):
                payload = source_path.read_bytes()
                inventory.append(
                    {
                        "path": source_path.relative_to(root).as_posix(),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "bytes": len(payload),
                    }
                )
            encoded_inventory = json.dumps(
                inventory, sort_keys=True, separators=(",", ":")
            ).encode()
            receipt = {
                "schema_version": 1,
                "source_revision": "0123456789abcdef" * 2 + "01234567",
                "source_layout": "explicit_allowlist_v1",
                "source_tree_sha256": hashlib.sha256(encoded_inventory).hexdigest(),
                "source_inventory": inventory,
                "base_image": {
                    "name": "registry.example/trainer:cuda",
                    "digest": "sha256:" + "0123456789abcdef" * 4,
                },
                "verl_revision": "8fdc4d3f202f41461f4de9f42a637228e342668b",
            }
            (metadata / "source.json").write_text(
                json.dumps(receipt), encoding="utf-8"
            )
            workflow = SimpleNamespace(
                repository_root=root,
                training_config=training,
            )
            result = server._repository_check(
                workflow,
                runner=lambda *_args, **_kwargs: self.fail("git must not run"),
            )
            self.assertEqual(result["mode"], "immutable_image_receipt")
            receipt["unexpected"] = True
            (metadata / "source.json").write_text(
                json.dumps(receipt), encoding="utf-8"
            )
            with self.assertRaisesRegex(server.ServerWorkflowError, "metadata is invalid"):
                server._repository_check(
                    workflow,
                    runner=lambda *_args, **_kwargs: self.fail("git must not run"),
                )

    def test_repository_check_rejects_tracked_or_untracked_changes(self) -> None:
        workflow = server.load_server_workflow(
            EXAMPLE, repository_root=REPOSITORY_ROOT
        )
        observed = []

        def clean_runner(argv, **_kwargs):
            observed.append(argv)
            output = "a" * 40 + "\n" if "rev-parse" in argv else ""
            return subprocess.CompletedProcess(argv, 0, output, "")

        result = server._repository_check(workflow, runner=clean_runner)
        self.assertTrue(result["worktree_clean"])
        self.assertIn("--untracked-files=all", observed[1])

        def dirty_runner(argv, **_kwargs):
            output = "a" * 40 + "\n" if "rev-parse" in argv else " M README.md\n"
            return subprocess.CompletedProcess(argv, 0, output, "")

        with self.assertRaisesRegex(server.ServerWorkflowError, "tracked or untracked"):
            server._repository_check(workflow, runner=dirty_runner)

    def test_readiness_is_no_download_and_reports_wandb(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".git").mkdir()
            (root / "scripts").mkdir()
            (root / "configs").mkdir()
            (root / "outputs").mkdir()
            (root / "infra/server").mkdir(parents=True)
            (root / "infra/server/compose.yaml").write_text(
                "services: {}\n", encoding="utf-8"
            )
            (root / "pyproject.toml").write_text("", encoding="utf-8")
            for name in (
                "prepare_math500.py",
                "evaluate_math500.py",
                "train_math500.py",
            ):
                (root / "scripts" / name).write_text("", encoding="utf-8")
            for name in ("training.toml", "raw.toml", "synthesis.toml", "scoring.toml"):
                (root / "configs" / name).write_text("", encoding="utf-8")
            workflow_path = root / "server.toml"
            workflow_path.write_text(
                """
schema_version = 2
kind = "cat_math500_server_workflow"
[configs]
training = "configs/training.toml"
raw = "configs/raw.toml"
synthesis = "configs/synthesis.toml"
scoring = "configs/scoring.toml"
[outputs]
root = "outputs"
training_run = "outputs/training"
initial_raw_run = "outputs/raw"
initial_synthesis_run = "outputs/synthesis"
preregistration = "outputs/registry/prereg.json"
launch_approval = "outputs/registry/approval.json"
manual_attestation = "outputs/registry/manual.json"
checkpoint_export = "outputs/exports/qwen3-4b-math500-cat-step-1000"
trained_eval_config_dir = "outputs/trained-configs"
trained_raw_run = "outputs/trained-raw"
trained_synthesis_run = "outputs/trained-synthesis"
final_registry = "outputs/registry/final.json"
one_step_dir = "outputs/qualifications/one"
resume_three_step_dir = "outputs/qualifications/resume"
full_shape_five_step_dir = "outputs/qualifications/full"
[services]
evaluation_base_url = "http://anchor:8001/v1"
evaluation_api_key_env = "CAT_ANCHOR_API_KEY"
evaluation_workers = 2
evaluation_batch_size = 4
anchor_start_command = ["docker", "compose", "-f", "infra/server/compose.yaml", "--profile", "local-anchor", "up", "-d", "--wait", "--wait-timeout", "600", "anchor"]
trained_policy_base_url = "http://trained-policy:8002/v1"
trained_policy_api_key_env = "CAT_TRAINED_POLICY_API_KEY"
trained_policy_served_model = "math500-cat-final"
trained_policy_start_command = ["docker", "compose", "-f", "infra/server/compose.yaml", "--profile", "trained-policy", "up", "-d", "--wait", "--wait-timeout", "600", "trained-policy"]
[readiness]
required_commands = ["git", "nvidia-smi"]
required_environment = []
minimum_free_bytes = 1
""".strip()
                + "\n",
                encoding="utf-8",
            )
            workflow = server.load_server_workflow(
                workflow_path, repository_root=root
            )
            training = SimpleNamespace(
                fingerprint="t" * 64,
                runtime=SimpleNamespace(
                    anchor_api_key_env="CAT_ANCHOR_API_KEY",
                    download_allowed=False,
                    model_path="/models/qwen",
                    python_executable="/opt/verl/python",
                    verl_source_path="/opt/verl/source",
                ),
                tracking=SimpleNamespace(
                    wandb=SimpleNamespace(
                        enabled=True,
                        mode="online",
                        api_key_env="WANDB_API_KEY",
                        project="cat",
                        entity="research",
                        sdk_version="0.21.1",
                    )
                ),
            )
            evaluation = SimpleNamespace(fingerprint="e" * 64)
            observed_environments = []
            observed_argv = []

            def runner(argv, **kwargs):
                if argv[0] == "git":
                    if "rev-parse" in argv:
                        return subprocess.CompletedProcess(argv, 0, "a" * 40 + "\n", "")
                    return subprocess.CompletedProcess(argv, 0, "", "")
                observed_argv.append(argv)
                observed_environments.append(kwargs["env"])
                if "prepare_math500.py" in argv[1]:
                    return subprocess.CompletedProcess(
                        argv, 0, "Verified MATH-500: rows=500\n", ""
                    )
                if "preflight" in argv:
                    return completed(
                        argv,
                        {
                            "mode": "ok",
                            "operationally_ready_to_launch": True,
                            "missing_gates": [],
                        },
                    )
                return completed(argv, {"mode": "ok"})

            environment = {
                "CAT_ANCHOR_API_KEY": "secret",
                "WANDB_API_KEY": "secret",
            }
            with (
                patch.object(server.shutil, "which", side_effect=lambda name: f"/bin/{name}"),
                patch.object(server, "load_training_config", return_value=training),
                patch.object(server, "load_raw_config", return_value=evaluation),
                patch.object(server, "load_synthesis_config", return_value=evaluation),
                patch.object(server, "load_scoring_config", return_value=evaluation),
                patch.object(
                    server,
                    "load_training_plan",
                    return_value=({}, SimpleNamespace()),
                ),
                patch.object(
                    server,
                    "validate_tracking_readiness",
                    return_value={
                        "console": True,
                        "wandb": {
                            "enabled": True,
                            "sdk_version": "0.21.1",
                            "credential_present": True,
                        },
                    },
                ),
                patch.dict(os.environ, environment, clear=True),
            ):
                result = server.readiness_report(workflow, runner=runner)
            self.assertTrue(result["ready"], result)
            self.assertTrue(result["no_download"])
            self.assertFalse(result["models_launched"])
            tracking = next(
                item for item in result["checks"] if item["name"] == "tracking"
            )
            self.assertTrue(tracking["detail"]["wandb"]["credential_present"])
            self.assertTrue(observed_environments)
            self.assertTrue(
                all(env["HF_HUB_OFFLINE"] == "1" for env in observed_environments)
            )
            self.assertEqual(len(observed_argv), 4)
            self.assertIn("--verify-questions-only", observed_argv[0])
            self.assertFalse(any("--execute" in argv for argv in observed_argv))


if __name__ == "__main__":
    unittest.main()
