from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_image = _module("cat_build_image", ROOT / "infra/server/build_image.py")
doctor = _module("cat_server_doctor", ROOT / "infra/server/doctor.py")
entrypoint = _module("cat_server_entrypoint", ROOT / "infra/server/entrypoint.py")
write_image_metadata = _module(
    "cat_write_image_metadata", ROOT / "infra/server/write_image_metadata.py"
)
validate_server_env = _module(
    "cat_validate_server_env", ROOT / "infra/server/validate_server_env.py"
)
ready = _module("cat_server_ready", ROOT / "infra/server/ready.py")


def _validate_host(environment: dict[str, str], **kwargs):
    with patch.object(
        validate_server_env,
        "_verify_full_dataset",
        return_value={"rows": 500, "questions_sha256": "q", "labels_sha256": "l"},
    ):
        return validate_server_env.validate(environment, **kwargs)


def _gpu_devices() -> dict[str, str]:
    return {
        **{f"CAT_TRAINER_GPU_DEVICE_{index}": str(index) for index in range(8)},
        "CAT_ANCHOR_GPU_DEVICE": "8",
    }


class ContainerContractTests(unittest.TestCase):
    def test_host_tooling_versions_fail_closed(self) -> None:
        current = iter(
            (
                ("Docker version 24.0.0", (24, 0, 0)),
                ("2.30.0", (2, 30, 0)),
                ("github.com/docker/buildx v0.14.0", (0, 14, 0)),
            )
        )
        with patch.object(
            validate_server_env,
            "_command_version",
            side_effect=lambda *_args: next(current),
        ):
            self.assertEqual(
                validate_server_env.validate_container_tooling(),
                {"docker": "24.0.0", "compose": "2.30.0", "buildx": "0.14.0"},
            )
        with patch.object(
            validate_server_env,
            "_command_version",
            return_value=("Docker version 23.0.0", (23, 0, 0)),
        ):
            with self.assertRaisesRegex(
                validate_server_env.EnvironmentError,
                "docker must be at least",
            ):
                validate_server_env.validate_container_tooling()

    def test_entrypoint_enforces_service_roles_and_fixed_servers(self) -> None:
        with patch.dict(
            entrypoint.os.environ,
            {"CAT_SERVICE_ROLE": "trainer"},
            clear=True,
        ):
            with self.assertRaisesRegex(SystemExit, "not allowed"):
                entrypoint.main(["evaluate", "--help"])
        with patch.dict(
            entrypoint.os.environ,
            {"CAT_SERVICE_ROLE": "scorer"},
            clear=True,
        ):
            with self.assertRaisesRegex(SystemExit, "not allowed"):
                entrypoint.main(["evaluate", "score-raw"])
        with patch.dict(
            entrypoint.os.environ,
            {"CAT_SERVICE_ROLE": "anchor", "CAT_MODEL_DIR": "/mnt/models/revision"},
            clear=True,
        ), patch.object(entrypoint.os, "execvp") as execute:
            self.assertEqual(entrypoint.main(["serve-anchor"]), 0)
        argv = execute.call_args.args[1]
        self.assertIn("vllm.entrypoints.openai.api_server", argv)
        self.assertIn("cat-frozen-qwen3-4b", argv)
        self.assertEqual(argv[argv.index("--generation-config") + 1], "vllm")
        self.assertNotIn("exec", entrypoint.COMMANDS)

    def test_base_image_requires_a_nonplaceholder_digest(self) -> None:
        digest = "0123456789abcdef" * 4
        self.assertEqual(
            build_image.parse_base_reference(f"registry.example/trainer:cuda@sha256:{digest}"),
            ("registry.example/trainer:cuda", f"sha256:{digest}"),
        )
        for invalid in (
            "registry.example/trainer:latest",
            "registry.example/trainer@sha256:REPLACE_ME",
            "registry.example/trainer@sha256:" + "0" * 64,
            "registry.example/trainer@sha256:" + "A" * 64,
        ):
            with self.subTest(invalid=invalid), self.assertRaises(build_image.BuildError):
                build_image.parse_base_reference(invalid)

    def test_runtime_image_reference_and_digest_must_agree(self) -> None:
        first = "sha256:" + "0123456789abcdef" * 4
        second = "sha256:" + "fedcba9876543210" * 4
        doctor.validate_image_identity(f"registry.example/cat@{first}", first)
        with self.assertRaises(doctor.DoctorError):
            doctor.validate_image_identity("registry.example/cat:latest", first)
        with self.assertRaises(doctor.DoctorError):
            doctor.validate_image_identity(f"registry.example/cat@{first}", second)

    def test_read_only_mount_uses_filesystem_flags(self) -> None:
        with patch.object(doctor.os, "statvfs", return_value=SimpleNamespace(f_flag=doctor.os.ST_RDONLY)):
            self.assertTrue(doctor.is_read_only_mount(Path("/mount")))
        with patch.object(doctor.os, "statvfs", return_value=SimpleNamespace(f_flag=0)):
            self.assertFalse(doctor.is_read_only_mount(Path("/mount")))

    def test_health_mode_never_dispatches_full_runtime_probe(self) -> None:
        with (
            patch.object(doctor, "load_contract", return_value={}),
            patch.object(doctor, "check_health", return_value={"ok": True}) as health,
            patch.object(doctor, "check_runtime", side_effect=AssertionError("CUDA probe")),
            redirect_stdout(StringIO()),
        ):
            self.assertEqual(doctor.main(["--mode", "health", "--quiet"]), 0)
        health.assert_called_once_with({})

    def test_adapter_smoke_failures_are_not_hidden(self) -> None:
        expected = {"dataset_rows": 1, "reward_rows": 8}
        with patch.object(doctor, "run_adapter_smoke", return_value=expected):
            self.assertEqual(doctor.check_adapter_smoke(), expected)
        with (
            patch.object(doctor, "run_adapter_smoke", side_effect=TypeError("API changed")),
            self.assertRaisesRegex(doctor.DoctorError, "API changed"),
        ):
            doctor.check_adapter_smoke()
        source = (ROOT / "infra/server/adapter_smoke.py").read_text()
        self.assertIn("create_rl_dataset", source)
        self.assertIn("load_reward_manager", source)
        self.assertIn("DataProto.from_dict", source)
        self.assertIn("ray.is_initialized()", source)
        self.assertIn("torch.cuda.is_initialized()", source)

    def test_host_environment_requires_disjoint_gpus_and_complete_mounts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            revision = "1234567890abcdef1234567890abcdef12345678"
            model = root / revision
            data = root / "data"
            output = root / "output"
            config = root / "config"
            for directory in (model, data / "raw", data / "math500", output, config):
                directory.mkdir(parents=True, exist_ok=True)
            for path in (
                data / "raw/math500-test.jsonl",
                data / "math500/questions.jsonl",
                data / "math500/labels.jsonl",
                *(config / name for name in (
                    "workflow.toml", "training.toml", "raw.toml", "synthesis.toml", "scoring.toml"
                )),
            ):
                path.touch()
            digest = "sha256:" + "0123456789abcdef" * 4
            environment = {
                **_gpu_devices(),
                "CAT_ANCHOR_MODE": "local",
                "CAT_MODEL_REVISION": revision,
                "CAT_TRAINER_IMAGE": f"registry.example/cat@{digest}",
                "CAT_TRAINER_IMAGE_DIGEST": digest,
                "CAT_MODEL_HOST_DIR": str(model),
                "CAT_DATA_HOST_DIR": str(data),
                "CAT_OUTPUT_HOST_DIR": str(output),
                "CAT_CONFIG_HOST_DIR": str(config),
            }
            self.assertTrue(_validate_host(environment)["ready"])
            environment["CAT_ANCHOR_GPU_DEVICE"] = "7"
            with self.assertRaises(validate_server_env.EnvironmentError):
                _validate_host(environment)
            environment["CAT_ANCHOR_MODE"] = "remote"
            environment.pop("CAT_ANCHOR_GPU_DEVICE")
            self.assertIsNone(_validate_host(environment)["anchor_gpu_id"])
            environment["CAT_ANCHOR_GPU_DEVICE"] = "8"
            with self.assertRaises(validate_server_env.EnvironmentError):
                _validate_host(environment)

            environment.pop("CAT_ANCHOR_GPU_DEVICE")
            link = model / "linked-weight.safetensors"
            link.symlink_to(root / "missing-cache-blob")
            with self.assertRaisesRegex(
                validate_server_env.EnvironmentError, "materialized without symlinks"
            ):
                _validate_host(environment)
            link.unlink()

            nested_output = data / "nested-output"
            nested_output.mkdir()
            environment["CAT_OUTPUT_HOST_DIR"] = str(nested_output)
            with self.assertRaisesRegex(
                validate_server_env.EnvironmentError, "distinct, non-nested"
            ):
                _validate_host(environment)

    def test_trained_policy_host_contract_requires_export_and_disjoint_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            revision = "1234567890abcdef1234567890abcdef12345678"
            model = root / revision
            data = root / "data"
            output = root / "output"
            config = root / "config"
            export = output / validate_server_env.TRAINED_EXPORT
            for directory in (model, data / "raw", data / "math500", export, config):
                directory.mkdir(parents=True, exist_ok=True)
            for path in (
                data / "raw/math500-test.jsonl",
                data / "math500/questions.jsonl",
                data / "math500/labels.jsonl",
                *(config / name for name in (
                    "workflow.toml", "training.toml", "raw.toml", "synthesis.toml", "scoring.toml"
                )),
                export / "config.json",
                export / "model-00001-of-00001.safetensors",
            ):
                path.touch()
            digest = "sha256:" + "0123456789abcdef" * 4
            environment = {
                **_gpu_devices(),
                "CAT_ANCHOR_MODE": "local",
                "CAT_TRAINED_POLICY_GPU_DEVICE": "0",
                "CAT_MODEL_REVISION": revision,
                "CAT_TRAINER_IMAGE": f"registry.example/cat@{digest}",
                "CAT_TRAINER_IMAGE_DIGEST": digest,
                "CAT_MODEL_HOST_DIR": str(model),
                "CAT_DATA_HOST_DIR": str(data),
                "CAT_OUTPUT_HOST_DIR": str(output),
                "CAT_CONFIG_HOST_DIR": str(config),
            }
            result = _validate_host(
                environment, require_trained_policy=True
            )
            self.assertEqual(result["trained_policy_gpu_id"], "0")
            environment["CAT_TRAINED_POLICY_GPU_DEVICE"] = "8"
            with self.assertRaisesRegex(
                validate_server_env.EnvironmentError, "must be disjoint"
            ):
                _validate_host(
                    environment, require_trained_policy=True
                )

    def test_host_validator_cryptographically_verifies_the_full_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            expected = {"rows": 500, "source_sha256": "s", "labels_sha256": "l"}
            with patch.object(
                validate_server_env,
                "verify_dataset",
                return_value=expected,
            ) as verify:
                self.assertEqual(validate_server_env._verify_full_dataset(data), expected)
            verify.assert_called_once_with(data.parent, validate_server_env.DATASET_LOCK)
            with self.assertRaisesRegex(
                validate_server_env.EnvironmentError,
                "basename must be data",
            ):
                validate_server_env._verify_full_dataset(root / "dataset")

    def test_combined_readiness_requires_both_layers(self) -> None:
        passed = ready.combined_report(
            Path("workflow.toml"),
            doctor_runner=lambda: {"ok": True},
            workflow_runner=lambda: {"ready": True},
        )
        self.assertTrue(passed["ready"])
        failed = ready.combined_report(
            Path("workflow.toml"),
            doctor_runner=lambda: {"ok": True},
            workflow_runner=lambda: {"ready": False},
        )
        self.assertFalse(failed["ready"])

    def test_runtime_contract_pins_paper_environment_core(self) -> None:
        contract = doctor.load_contract()
        self.assertEqual(contract["cuda"], "12.6")
        self.assertEqual(contract["cudnn"], 90800)
        self.assertEqual(contract["runtime_uid"], 10001)
        self.assertEqual(contract["python"], {"minimum": "3.10", "maximum_exclusive": "3.11"})
        self.assertEqual(
            contract["verl_revision"],
            "8fdc4d3f202f41461f4de9f42a637228e342668b",
        )
        self.assertEqual(
            contract["packages"],
            {
                "antlr4-python3-runtime": "4.13.2",
                "latex2sympy2-extended": "1.11.0",
                "math-verify": "0.9.0",
                "mpmath": "1.3.0",
                "sympy": "1.14.0",
                "tomli": "2.2.1",
                "torch": "2.7.0",
                "verl": "0.5.0",
                "vllm": "0.9.1",
                "wandb": "0.21.1",
            },
        )
        constraints = (ROOT / "infra/server/runtime-constraints.txt").read_text()
        for name, version in contract["packages"].items():
            self.assertIn(f"{name}=={version}", constraints)
        overlay = (ROOT / "infra/server/overlay-requirements.lock").read_text()
        self.assertIn("wandb==0.21.1", overlay)
        self.assertIn("5ded9313672630c0630f5b13c598ce9aa0e932e811ebc18823fcc4d73acfb6bb", overlay)
        self.assertIn("tomli==2.2.1", overlay)
        self.assertIn("cb55c73c5f4408779d0cf3eef9f762b9c9f147a77de7b258bef0a5628adc85cc", overlay)
        self.assertIn("3703e7c4885354027fa84409d762a596a2906d1fd4deb78361876bd905a76194", overlay)
        self.assertIn("aebb77d52ce269e25028e4bea89ddb14d242ba36bcf7b636496fb5fd9728d234", overlay)

    def test_image_source_metadata_covers_the_copy_allowlist(self) -> None:
        source = "1234567890abcdef1234567890abcdef12345678"
        verl = "8fdc4d3f202f41461f4de9f42a637228e342668b"
        digest = "sha256:" + "0123456789abcdef" * 4
        value = write_image_metadata.metadata(
            source, "registry.example/base:tag", digest, verl, ROOT
        )
        inventory, tree_sha256 = write_image_metadata.source_tree(ROOT)
        self.assertEqual(value["source_inventory"], inventory)
        self.assertEqual(value["source_tree_sha256"], tree_sha256)
        with self.assertRaises(ValueError):
            write_image_metadata.metadata("replace_me", "base:tag", digest, verl, ROOT)

    def test_dockerfile_uses_an_explicit_source_allowlist(self) -> None:
        dockerfile = (ROOT / "infra/server/Dockerfile").read_text()
        self.assertNotIn("COPY . ", dockerfile)
        self.assertNotIn("COPY data", dockerfile)
        self.assertNotIn("COPY notebooks", dockerfile)
        self.assertNotIn("COPY docs", dockerfile)
        self.assertNotIn("compute_as_a_teacher.pdf", dockerfile)
        self.assertIn("COPY --chown=10001:10001 src ", dockerfile)
        self.assertIn("COPY --chown=10001:10001 configs/server ", dockerfile)
        self.assertIn("scripts/prepare_math500.py", dockerfile)
        self.assertIn("USER 10001:10001", dockerfile)
        self.assertIn("REPLACE_WITH_REVIEWED_BASE_IMAGE_AT_SHA256_DIGEST", dockerfile)
        self.assertIn("org.opencontainers.image.base.digest", dockerfile)
        self.assertIn("ln -s /mnt/data /opt/cat/repo/data", dockerfile)
        self.assertIn("ln -s /mnt/outputs /opt/cat/repo/outputs", dockerfile)
        self.assertIn("-name '*.safetensors'", dockerfile)
        self.assertIn("-name '*.gguf'", dockerfile)
        self.assertIn("--force-reinstall --no-deps --require-hashes --only-binary=:all:", dockerfile)
        self.assertIn("python -m pip check", dockerfile)
        self.assertIn("RUN python infra/server/doctor.py --mode image", dockerfile)

    def test_dockerignore_defends_against_large_or_sensitive_context(self) -> None:
        ignored = set((ROOT / ".dockerignore").read_text().splitlines())
        required = {
            "**",
            ".git",
            ".env",
            ".env.*",
            ".venv",
            "compute_as_a_teacher.pdf",
            "notebooks",
            "data",
            "models",
            "outputs",
            "wandb",
            "**/*.safetensors",
            "**/*.bin",
            "**/*.gguf",
            "**/*.pem",
            "**/*.key",
        }
        self.assertTrue(required <= ignored)
        for admitted in (
            "!src/**",
            "!scripts/train_math500.py",
            "!configs/server/**",
            "!infra/server/**",
        ):
            self.assertIn(admitted, ignored)

    def test_server_environment_template_is_complete_and_nonsecret(self) -> None:
        template = (ROOT / "infra/server/server.env.example").read_text()
        for name in (
            "CAT_TRAINER_IMAGE",
            "CAT_TRAINER_IMAGE_DIGEST",
            "CAT_MODEL_REVISION",
            "CAT_ANCHOR_MODE",
            *(f"CAT_TRAINER_GPU_DEVICE_{index}" for index in range(8)),
            "CAT_ANCHOR_GPU_DEVICE",
            "CAT_TRAINED_POLICY_GPU_DEVICE",
            "CAT_MODEL_HOST_DIR",
            "CAT_DATA_HOST_DIR",
            "CAT_OUTPUT_HOST_DIR",
            "CAT_CONFIG_HOST_DIR",
        ):
            self.assertIn(f"{name}=", template)
        self.assertNotIn("API_KEY=", template)
        self.assertIn(".env.*", (ROOT / ".gitignore").read_text())

    def test_compose_mounts_inputs_read_only_and_requires_digest(self) -> None:
        compose = (ROOT / "infra/server/compose.yaml").read_text()
        entrypoint_source = (ROOT / "infra/server/entrypoint.py").read_text()
        anchor_section, remaining = compose.split("  trained-policy:", 1)
        trained_policy_body, remaining = remaining.split("  trainer:", 1)
        trainer_body, remaining = remaining.split("  evaluator:", 1)
        evaluator_body, scorer_body = remaining.split("  scorer:", 1)
        self.assertIn("CAT_TRAINER_IMAGE:?", compose)
        self.assertIn("CAT_TRAINER_IMAGE_DIGEST:?", compose)
        for index in range(8):
            self.assertIn(f"CAT_TRAINER_GPU_DEVICE_{index}:?", compose)
        self.assertIn("CAT_ANCHOR_GPU_DEVICE:-anchor_gpu_not_configured", compose)
        self.assertIn("CAT_ANCHOR_MODE:?", compose)
        self.assertNotIn("gpus: all", compose)
        self.assertEqual(compose.count("device_ids:"), 3)
        self.assertIn("source: ${CAT_MODEL_HOST_DIR:?", compose)
        question_source = (
            "source: ${CAT_DATA_HOST_DIR:?set the prepared data directory}"
            "/math500/questions.jsonl"
        )
        labels_source = (
            "source: ${CAT_DATA_HOST_DIR:?set the prepared data directory}"
            "/math500/labels.jsonl"
        )
        self.assertEqual(compose.count(question_source), 3)
        self.assertEqual(compose.count(labels_source), 1)
        self.assertIn("source: ${CAT_CONFIG_HOST_DIR:?", compose)
        self.assertIn("CAT_WORKFLOW_PATH: /mnt/config/workflow.toml", compose)
        self.assertIn("CAT_TRAINER_IMAGE_REFERENCE: ${CAT_TRAINER_IMAGE:?", compose)
        self.assertIn("/mnt/models/${CAT_MODEL_REVISION:?", compose)
        self.assertIn("vllm.entrypoints.openai.api_server", entrypoint_source)
        self.assertIn('command: ["serve-anchor"]', compose)
        self.assertIn('command: ["serve-trained-policy"]', compose)
        self.assertEqual(compose.count("ipc: host"), 3)
        self.assertEqual(compose.count('PYTHONDONTWRITEBYTECODE: "1"'), 5)
        self.assertIn('"Qwen/Qwen3-4B"', entrypoint_source)
        self.assertIn('"cat-frozen-qwen3-4b"', entrypoint_source)
        self.assertGreaterEqual(compose.count("read_only: true"), 3)
        self.assertIn("CAT_ANCHOR_API_KEY: ${CAT_ANCHOR_API_KEY:-local-compose-no-auth}", trainer_body)
        self.assertIn("WANDB_API_KEY: ${WANDB_API_KEY:-}", trainer_body)
        self.assertNotIn("CAT_TRAINED_POLICY_API_KEY", trainer_body)
        self.assertEqual(compose.count('user: "10001:10001"'), 5)
        self.assertIn('profiles: ["trained-policy"]', compose)
        self.assertIn("CAT_TRAINED_POLICY_GPU_DEVICE", compose)
        self.assertIn(
            "${CAT_OUTPUT_HOST_DIR:?set the output directory}/exports/qwen3-4b-math500-cat-step-1000",
            compose,
        )
        self.assertIn("target: /mnt/trained-model", compose)
        self.assertNotIn("CAT_OUTPUT_HOST_DIR", anchor_section)
        self.assertIn("- /cache:exec,mode=1777", anchor_section)
        self.assertIn('"--tokenizer"', entrypoint_source)
        self.assertIn('"math500-cat-final"', entrypoint_source)
        self.assertNotIn("8002:8002", compose)

        for name, section in (("trainer", trainer_body), ("evaluator", evaluator_body)):
            with self.subTest(service=name):
                self.assertIn(question_source, section)
                self.assertIn("target: /mnt/data/math500/questions.jsonl", section)
                self.assertNotIn("labels.jsonl", section)
                self.assertNotIn("raw/math500-test.jsonl", section)
        self.assertNotIn("CAT_MODEL_HOST_DIR", evaluator_body)
        self.assertNotIn("gpus:", evaluator_body)
        self.assertNotIn("WANDB", evaluator_body)
        self.assertIn("CAT_ANCHOR_API_KEY", evaluator_body)
        self.assertIn("CAT_TRAINED_POLICY_API_KEY", evaluator_body)

        self.assertIn(question_source, scorer_body)
        self.assertIn(labels_source, scorer_body)
        self.assertNotIn("raw/math500-test.jsonl", scorer_body)
        self.assertNotIn("CAT_MODEL_HOST_DIR", scorer_body)
        self.assertNotIn("gpus:", scorer_body)
        self.assertNotIn("API_KEY", scorer_body)
        self.assertNotIn("WANDB", scorer_body)
        self.assertIn("network_mode: none", scorer_body)
        self.assertNotIn("CAT_OUTPUT_HOST_DIR", anchor_section)
        self.assertIn(
            "${CAT_OUTPUT_HOST_DIR:?set the output directory}/exports/"
            "qwen3-4b-math500-cat-step-1000",
            trained_policy_body,
        )

        guide = (ROOT / "infra/server/README.md").read_text()
        doctor_command = "run --rm trainer doctor"
        anchor_command = "phase anchor --execute"
        self.assertIn(doctor_command, guide)
        self.assertIn(anchor_command, guide.replace("\\\n     ", ""))
        self.assertLess(guide.index(doctor_command), guide.index(anchor_command))

    def test_inventory_digest_matches_preflight_encoding(self) -> None:
        packages, digest = doctor.package_inventory()
        encoded = json.dumps(packages, separators=(",", ":")).encode()
        self.assertEqual(digest, hashlib.sha256(encoded).hexdigest())


if __name__ == "__main__":
    unittest.main()
