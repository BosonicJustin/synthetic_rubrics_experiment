from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from compute_as_a_teacher.training.errors import TrainingError  # noqa: E402
from compute_as_a_teacher.training.preflight import validate_wandb_runtime  # noqa: E402
from compute_as_a_teacher.training.qualification import (  # noqa: E402
    QUALIFICATION_PROFILES,
    derive_qualification_command,
)
from compute_as_a_teacher.training.verl_adapter import (  # noqa: E402
    VerlCommand,
    build_process_environment,
    build_verl_command,
    canonical_wandb_run_id,
    validate_tracking_readiness,
)
from tests.training.test_config_and_planning import (  # noqa: E402
    load_text,
    resolved_config_text,
)


def wandb_config_text(*, mode: str = "online") -> str:
    return (
        resolved_config_text()
        .replace("enabled = false", "enabled = true", 1)
        .replace('entity = ""', 'entity = "bosonic-justin"', 1)
        .replace('mode = "online"', f'mode = "{mode}"', 1)
    )


def command_for(config, run_dir: Path = Path("/tmp/cat-wandb")) -> VerlCommand:
    return build_verl_command(
        config,
        repository_root=REPOSITORY_ROOT,
        run_dir=run_dir,
        training_data_path=run_dir / "math500_train.jsonl",
    )


class WandbIntegrationTests(unittest.TestCase):
    def test_stale_v1_config_gets_an_explicit_version_error(self) -> None:
        with self.assertRaisesRegex(TrainingError, "schema_version=2"):
            load_text(
                resolved_config_text()
                .replace("schema_version = 2", "schema_version = 1", 1)
                .replace("[tracking]\n", "", 1)
                .replace("console = true\n\n[tracking.wandb]\n", "", 1)
            )

    def test_default_is_console_only(self) -> None:
        config = load_text(resolved_config_text())
        command = command_for(config)
        self.assertFalse(config.tracking.wandb.enabled)
        self.assertIn("trainer.logger=[console]", command.argv)
        self.assertFalse(
            any(name.startswith("WANDB_") for name in command.environment)
        )

    def test_enabled_command_has_stable_nonsecret_identity(self) -> None:
        config = load_text(wandb_config_text())
        first = command_for(config)
        second = command_for(config, Path("/tmp/another-copy-of-the-same-run"))
        expected_id = canonical_wandb_run_id(config)
        self.assertEqual(first.environment["WANDB_RUN_ID"], expected_id)
        self.assertEqual(second.environment["WANDB_RUN_ID"], expected_id)
        self.assertEqual(first.environment["WANDB_RESUME"], "allow")
        self.assertEqual(first.environment["WANDB_MODE"], "online")
        self.assertEqual(first.environment["WANDB_ENTITY"], "bosonic-justin")
        self.assertIn("trainer.logger=[console,wandb]", first.argv)
        self.assertIn(
            'trainer.project_name="synthetic-rubrics-experiment"', first.argv
        )
        serialized = json.dumps(first.to_dict(), sort_keys=True)
        self.assertNotIn("WANDB_API_KEY", serialized)
        self.assertNotIn("secret-value", serialized)

        changed = load_text(
            wandb_config_text().replace(
                'run_name = "qwen3-4b-math500-cat"',
                'run_name = "qwen3-4b-math500-cat-replica"',
            )
        )
        self.assertNotEqual(canonical_wandb_run_id(changed), expected_id)

    def test_online_readiness_requires_key_and_redacts_it(self) -> None:
        config = load_text(wandb_config_text())
        run_dir = Path("/tmp/cat-wandb")
        command = command_for(config, run_dir)
        with self.assertRaisesRegex(TrainingError, "unset or blank"):
            validate_tracking_readiness(
                config,
                command,
                run_dir,
                source_environment={},
            )
        with self.assertRaisesRegex(TrainingError, "unset or blank"):
            validate_tracking_readiness(
                config,
                command,
                run_dir,
                source_environment={"WANDB_API_KEY": "   "},
            )

        readiness = validate_tracking_readiness(
            config,
            command,
            run_dir,
            source_environment={"WANDB_API_KEY": "secret-value"},
        )
        self.assertTrue(readiness["wandb"]["credential_present"])
        self.assertNotIn("secret-value", json.dumps(readiness, sort_keys=True))
        environment = build_process_environment(
            config,
            command,
            run_dir,
            source_environment={
                "WANDB_API_KEY": "secret-value",
                "WANDB_PROJECT": "host-drift",
                "KEEP_ME": "yes",
            },
        )
        self.assertEqual(environment["WANDB_API_KEY"], "secret-value")
        self.assertEqual(environment["KEEP_ME"], "yes")
        self.assertNotIn("WANDB_PROJECT", environment)

    def test_offline_mode_is_rejected_because_resume_is_ignored(self) -> None:
        with self.assertRaisesRegex(TrainingError, "ignores resume"):
            load_text(wandb_config_text(mode="offline"))

    def test_custom_credential_variable_is_copied_only_at_process_launch(self) -> None:
        config = load_text(
            wandb_config_text().replace(
                'api_key_env = "WANDB_API_KEY"',
                'api_key_env = "CAT_WANDB_API_KEY"',
            )
        )
        run_dir = Path("/tmp/cat-wandb-custom-key")
        command = command_for(config, run_dir)
        self.assertNotIn("CAT_WANDB_API_KEY", command.environment)
        environment = build_process_environment(
            config,
            command,
            run_dir,
            source_environment={
                "CAT_WANDB_API_KEY": "custom-secret",
                "WANDB_API_KEY": "unplanned-secret",
            },
        )
        self.assertEqual(environment["WANDB_API_KEY"], "custom-secret")
        self.assertNotIn("CAT_WANDB_API_KEY", environment)

    def test_qualification_runs_have_distinct_ids_names_groups_and_tags(self) -> None:
        config = load_text(wandb_config_text())
        canonical = command_for(config)
        observed_ids = {canonical.environment["WANDB_RUN_ID"]}
        observed_groups = {canonical.environment["WANDB_RUN_GROUP"]}
        for profile in QUALIFICATION_PROFILES.values():
            with self.subTest(profile=profile.name):
                run_dir = Path("/tmp") / profile.name
                command = derive_qualification_command(
                    canonical,
                    profile,
                    run_dir,
                    run_name=config.run_name,
                )
                validate_tracking_readiness(
                    config,
                    command,
                    run_dir,
                    qualification_profile=profile.name,
                    source_environment={"WANDB_API_KEY": "secret-value"},
                )
                run_id = command.environment["WANDB_RUN_ID"]
                group = command.environment["WANDB_RUN_GROUP"]
                self.assertNotIn(run_id, observed_ids)
                self.assertNotIn(group, observed_groups)
                observed_ids.add(run_id)
                observed_groups.add(group)
                self.assertIn(
                    f'trainer.experiment_name="{config.run_name}-{profile.name}-nonreportable"',
                    command.argv,
                )
                tags = command.environment["WANDB_TAGS"].split(",")
                self.assertIn("nonreportable", tags)
                self.assertIn(profile.name, tags)

    def test_config_and_command_reject_unsafe_tracking_values(self) -> None:
        invalid = (
            ('entity = "bosonic-justin"', 'entity = ""', "project and entity"),
            ('resume = "allow"', 'resume = "auto"', "restart-safe"),
            ('sdk_version = "0.21.1"', 'sdk_version = "latest"', "sdk_version"),
            ('api_key_env = "WANDB_API_KEY"', 'api_key_env = "secret"', "environment variable"),
            ('tags = ["math500",', 'tags = ["bad,tag",', "comma-free"),
        )
        source = wandb_config_text()
        for old, new, error in invalid:
            with self.subTest(setting=old), self.assertRaisesRegex(
                TrainingError, error
            ):
                load_text(source.replace(old, new, 1))
        with self.assertRaisesRegex(TrainingError, "never be serialized"):
            VerlCommand(
                argv=("python",),
                cwd="/tmp",
                environment={"WANDB_API_KEY": "secret-value"},
                framework_revision="8fdc4d3f202f41461f4de9f42a637228e342668b",
                adapter_version="cat-verl-batch-reward-v1",
            )

    def test_runtime_contract_requires_exact_sdk_and_verl_call_shape(self) -> None:
        config = load_text(wandb_config_text())
        contract = {
            "python_prefix": "/opt/verl",
            "packages": {"wandb": "0.21.1"},
            "wandb_imported": True,
            "wandb_module_version": "0.21.1",
            "wandb_module_origin": (
                "/opt/verl/lib/python3.11/site-packages/wandb/__init__.py"
            ),
            "wandb_tracking": {
                "source": "/opt/verl/source/verl/utils/tracking.py",
                "source_sha256": "a" * 64,
                "tracking_class_found": True,
                "supported_backend_has_wandb": True,
                "init_call_count": 1,
                "init_keyword_names": ["config", "name", "project", "settings"],
                "init_keyword_values": {
                    "project": "project_name",
                    "name": "experiment_name",
                    "config": "config",
                    "settings": "settings",
                },
                "init_positional_args": 0,
            },
        }
        result = validate_wandb_runtime(config, contract)
        self.assertTrue(result["enabled"])
        self.assertEqual(result["sdk_version"], "0.21.1")

        contract["packages"]["wandb"] = "0.22.0"
        with self.assertRaisesRegex(TrainingError, "SDK does not match"):
            validate_wandb_runtime(config, contract)
        contract["packages"]["wandb"] = "0.21.1"
        contract["wandb_tracking"]["init_keyword_names"].append("id")
        with self.assertRaisesRegex(TrainingError, "contract no longer matches"):
            validate_wandb_runtime(config, contract)
        contract["wandb_tracking"]["init_keyword_names"].remove("id")
        contract["wandb_module_origin"] = "/tmp/shadowed/wandb/__init__.py"
        with self.assertRaisesRegex(TrainingError, "outside"):
            validate_wandb_runtime(config, contract)


if __name__ == "__main__":
    unittest.main()
