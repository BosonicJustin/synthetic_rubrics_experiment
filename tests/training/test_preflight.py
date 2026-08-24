from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from compute_as_a_teacher.evaluation.artifacts import sha256_text  # noqa: E402
from compute_as_a_teacher.training.errors import TrainingError  # noqa: E402
from compute_as_a_teacher.training.preflight import (  # noqa: E402
    compose_verl_config,
    discover_model_identity,
    discover_runtime_identity,
    hash_model_snapshot_tree,
    inspect_model_snapshot,
    probe_anchor,
    probe_runtime,
    probe_tokenizer,
    run_preflight,
    write_preflight_receipt,
)
from compute_as_a_teacher.training.verl_adapter import VerlCommand  # noqa: E402
from tests.training.test_config_and_planning import (  # noqa: E402
    load_text,
    resolved_config_text,
    resolved_single_h100_config_text,
)


REVISION = "a" * 40


def write_safetensors(
    path: Path,
    tensor_name: str = "model.weight",
    *,
    dtype: str = "BF16",
) -> None:
    data_bytes = 1_000_000
    bytes_per_element = {"BF16": 2, "F32": 4}[dtype]
    header = json.dumps(
        {
            tensor_name: {
                "dtype": dtype,
                "shape": [data_bytes // bytes_per_element],
                "data_offsets": [0, data_bytes],
            }
        },
        separators=(",", ":"),
    ).encode("utf-8")
    path.write_bytes(len(header).to_bytes(8, "little") + header + b"\0" * data_bytes)


def model_fixture(root: Path):
    model_path = root / REVISION
    model_path.mkdir()
    template = "{% for message in messages %}{{ message.content }}{% endfor %}"
    (model_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "torch_dtype": "bfloat16",
                "max_position_embeddings": 32768,
            }
        ),
        encoding="utf-8",
    )
    (model_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": template}), encoding="utf-8"
    )
    (model_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    write_safetensors(model_path / "model.safetensors")
    tree_sha256 = hash_model_snapshot_tree(model_path)["tree_sha256"]
    text = (
        resolved_config_text()
        .replace("b" * 40, REVISION)
        .replace("c" * 64, sha256_text(template))
        .replace("d" * 64, tree_sha256)
        .replace("/models/qwen3-4b", str(model_path))
    )
    return load_text(text), model_path


def command() -> VerlCommand:
    return VerlCommand(
        argv=("/opt/verl/bin/python", "-m", "verl.trainer.main_ppo"),
        cwd="/opt/verl/source",
        environment={
            "HF_HUB_OFFLINE": "1",
            "PYTHONPATH": str(REPOSITORY_ROOT / "src"),
        },
        framework_revision="8fdc4d3f202f41461f4de9f42a637228e342668b",
        adapter_version="cat-verl-batch-reward-v2",
    )


class PreflightTests(unittest.TestCase):
    def test_runtime_identity_can_be_discovered_before_config_is_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            python = root / "python"
            base_python = root / "base-python"
            base_python.write_text("#!/bin/sh\n", encoding="utf-8")
            base_python.chmod(0o755)
            python.symlink_to(base_python)
            source = root / "verl"
            source.mkdir()
            observed = {}

            def runner(*args, **kwargs):
                observed["argv"] = args[0]
                observed.update(kwargs)
                value = {
                    "package_inventory_sha256": "f" * 64,
                    "trainer_image_digest": "sha256:" + "e" * 64,
                }
                return subprocess.CompletedProcess(
                    args[0], 0, "CAT_PREFLIGHT_JSON=" + json.dumps(value) + "\n", ""
                )

            result = discover_runtime_identity(
                python,
                source,
                REPOSITORY_ROOT,
                runner=runner,
            )
            self.assertEqual(result["package_inventory_sha256"], "f" * 64)
            self.assertEqual(observed["argv"][0], str(python))
            self.assertEqual(
                observed["env"]["PYTHONPATH"],
                str(REPOSITORY_ROOT / "src"),
            )

    def test_anchor_canary_requires_the_unanimous_expected_answer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config, _ = model_fixture(Path(temporary))

            class Client:
                endpoint = "http://127.0.0.1:8001/v1/chat/completions"

                def __init__(self, response: str, context_response: str | None = None):
                    self.responses = iter((response, context_response or response))

                def complete(self, **kwargs):
                    return next(self.responses)

            with patch(
                "compute_as_a_teacher.training.preflight."
                "OpenAIChatCompletionsClient.from_environment",
                return_value=Client(
                    r"Canary synthesis \boxed{42}",
                    r"Long canary synthesis \boxed{tail-nonce}",
                ),
            ):
                result = probe_anchor(
                    config,
                    REPOSITORY_ROOT,
                    context_canary_message=(
                        r"full-length context canary with unique \boxed{tail-nonce}"
                    ),
                    context_canary_expected_answer="tail-nonce",
                )
            self.assertEqual(result["unanimous_agreement_rewards"], [1] * 8)
            self.assertTrue(result["long_context_request_accepted"])
            self.assertTrue(result["long_context_tail_answer_preserved"])

            with patch(
                "compute_as_a_teacher.training.preflight."
                "OpenAIChatCompletionsClient.from_environment",
                return_value=Client(
                    r"Canary synthesis \boxed{42}",
                    r"Only an earlier answer survived \boxed{42}",
                ),
            ):
                with self.assertRaisesRegex(TrainingError, "tail boxed answer"):
                    probe_anchor(
                        config,
                        REPOSITORY_ROOT,
                        context_canary_message=(
                            r"full-length canary with unique \boxed{tail-nonce}"
                        ),
                        context_canary_expected_answer="tail-nonce",
                    )

            with patch(
                "compute_as_a_teacher.training.preflight."
                "OpenAIChatCompletionsClient.from_environment",
                return_value=Client(r"Valid but wrong \boxed{7}"),
            ):
                with self.assertRaisesRegex(TrainingError, "unanimous boxed answer"):
                    probe_anchor(config, REPOSITORY_ROOT)

    def test_model_snapshot_binds_revision_template_dtype_context_and_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config, model_path = model_fixture(Path(temporary))
            result = inspect_model_snapshot(config, hash_all_files=True)
            identity = discover_model_identity(model_path)
            self.assertEqual(identity["snapshot_revision"], REVISION)
            self.assertEqual(
                identity["chat_template_sha256"],
                config.policy.chat_template_sha256,
            )
            self.assertEqual(result["model_revision"], REVISION)
            self.assertEqual(result["required_policy_context_length"], 3584)
            self.assertEqual(result["required_anchor_context_length"], 15872)
            self.assertGreaterEqual(
                result["context_length"], result["required_anchor_context_length"]
            )
            self.assertEqual(
                result["all_files_tree_sha256"],
                config.runtime.model_snapshot_tree_sha256,
            )
            self.assertEqual(result["safetensors"]["tensors"], 1)
            self.assertEqual(result["safetensors"]["dtypes"], ["BF16"])

            model_config = json.loads(
                (model_path / "config.json").read_text(encoding="utf-8")
            )
            model_config["max_position_embeddings"] = 15_000
            (model_path / "config.json").write_text(
                json.dumps(model_config), encoding="utf-8"
            )
            with self.assertRaisesRegex(TrainingError, "at least 15872"):
                inspect_model_snapshot(config)
            model_config["max_position_embeddings"] = 32_768
            (model_path / "config.json").write_text(
                json.dumps(model_config), encoding="utf-8"
            )

            tokenizer = json.loads(
                (model_path / "tokenizer_config.json").read_text(encoding="utf-8")
            )
            tokenizer["chat_template"] = "changed"
            (model_path / "tokenizer_config.json").write_text(
                json.dumps(tokenizer), encoding="utf-8"
            )
            with self.assertRaisesRegex(TrainingError, "chat template"):
                inspect_model_snapshot(config)

    def test_snapshot_tree_and_safetensors_fail_closed_without_loading_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config, model_path = model_fixture(Path(temporary))
            original = hash_model_snapshot_tree(model_path)
            self.assertGreater(original["bytes"], 1_000_000)
            (model_path / "extra.txt").write_text("changed\n", encoding="utf-8")
            self.assertNotEqual(
                hash_model_snapshot_tree(model_path)["tree_sha256"],
                original["tree_sha256"],
            )
            with self.assertRaisesRegex(TrainingError, "tree SHA-256"):
                inspect_model_snapshot(config, hash_all_files=True)

        with tempfile.TemporaryDirectory() as temporary:
            config, model_path = model_fixture(Path(temporary))
            (model_path / "model.safetensors").write_bytes(b"x" * 1_000_001)
            with self.assertRaisesRegex(TrainingError, "Malformed safetensors header"):
                inspect_model_snapshot(config)

        with tempfile.TemporaryDirectory() as temporary:
            config, model_path = model_fixture(Path(temporary))
            (model_path / "model.safetensors").unlink()
            write_safetensors(model_path / "model.safetensors", dtype="F32")
            with self.assertRaisesRegex(TrainingError, "unexpected dtypes"):
                inspect_model_snapshot(config)

        with tempfile.TemporaryDirectory() as temporary:
            config, model_path = model_fixture(Path(temporary))
            (model_path / "model.safetensors").unlink()
            write_safetensors(model_path / "model-00001-of-00002.safetensors", "a")
            write_safetensors(model_path / "model-00002-of-00002.safetensors", "b")
            (model_path / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": 2_000_000},
                        "weight_map": {
                            "a": "model-00001-of-00002.safetensors",
                            "b": "model-00001-of-00002.safetensors",
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(TrainingError, "index does not match"):
                inspect_model_snapshot(config)

    def test_runtime_probe_requires_exact_topology_and_image_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config, _ = model_fixture(Path(temporary))
            result = {
                "python": "3.11.9",
                "executable": "/opt/verl/bin/python",
                "platform": "linux",
                "packages": {
                    name: ("0.5.0" if name == "verl" else "1.0")
                    for name in (
                        "torch",
                        "ray",
                        "vllm",
                        "transformers",
                        "datasets",
                        "hydra-core",
                        "omegaconf",
                        "flash-attn",
                        "verl",
                    )
                },
                "package_inventory": [],
                "package_inventory_sha256": config.runtime.package_inventory_sha256,
                "package_count": 9,
                "torch_cuda_version": "12.6",
                "cudnn_version": 90800,
                "nccl_version": [2, 26, 2],
                "cuda_available": True,
                "gpu_count": 8,
                "gpus": [
                    {
                        "index": index,
                        "name": "NVIDIA H100 80GB HBM3",
                        "capability": [9, 0],
                        "bf16_supported": True,
                        "free_memory_bytes": 78_000_000_000,
                        "total_memory_bytes": 80_000_000_000,
                    }
                    for index in range(8)
                ],
                "verl_module": "/opt/verl/source/verl/__init__.py",
                "actor_optimizer": {
                    "kind": "torch.optim.AdamW",
                    "assignment_count": 1,
                    "source": "/opt/verl/source/verl/workers/fsdp_workers.py",
                    "source_sha256": "1" * 64,
                },
                "trainer_image_digest": config.runtime.trainer_image_digest,
                "custom_modules": {
                    "dataset": {
                        "module": "compute_as_a_teacher.training.verl_dataset",
                        "name": "JsonlRLHFDataset",
                        "source": str(
                            REPOSITORY_ROOT
                            / "src/compute_as_a_teacher/training/verl_dataset.py"
                        ),
                        "torch_dataset_subclass": True,
                    },
                    "reward": {
                        "module": "compute_as_a_teacher.training.verl_reward",
                        "name": "compute_score",
                        "source": str(
                            REPOSITORY_ROOT
                            / "src/compute_as_a_teacher/training/verl_reward.py"
                        ),
                        "parameters": [
                            {
                                "name": name,
                                "kind": (
                                    "POSITIONAL_OR_KEYWORD"
                                    if index < 4
                                    else "KEYWORD_ONLY"
                                ),
                                "has_default": name
                                in {
                                    "max_answer_chars",
                                    "anchor_failure_policy",
                                    "anchor_client",
                                },
                            }
                            for index, name in enumerate(
                                (
                                    "data_sources",
                                    "solution_strs",
                                    "ground_truths",
                                    "extra_infos",
                                    "repository_root",
                                    "prompt_path",
                                    "prompt_version",
                                    "prompt_prefix",
                                    "anchor_base_url",
                                    "anchor_model",
                                    "anchor_api_key_env",
                                    "anchor_timeout_seconds",
                                    "anchor_max_concurrency",
                                    "anchor_temperature",
                                    "anchor_top_p",
                                    "anchor_top_k",
                                    "anchor_max_tokens",
                                    "base_seed",
                                    "max_answer_chars",
                                    "anchor_failure_policy",
                                    "anchor_client",
                                )
                            )
                        ],
                    },
                },
            }

            def runner(*args, **kwargs):
                return subprocess.CompletedProcess(
                    args[0], 0, "CAT_PREFLIGHT_JSON=" + json.dumps(result) + "\n", ""
                )

            self.assertEqual(
                probe_runtime(config, command(), runner=runner)["gpu_count"], 8
            )
            result["gpus"][0]["name"] = "NVIDIA A100"
            with self.assertRaisesRegex(TrainingError, "H100"):
                probe_runtime(config, command(), runner=runner)
            result["gpus"][0]["name"] = "NVIDIA H100 80GB HBM3"
            result["gpus"][0]["free_memory_bytes"] = 1
            with self.assertRaisesRegex(TrainingError, "insufficient free memory"):
                probe_runtime(config, command(), runner=runner)
            result["gpus"][0]["free_memory_bytes"] = 78_000_000_000
            result["package_inventory_sha256"] = "0" * 64
            with self.assertRaisesRegex(TrainingError, "package inventory SHA-256"):
                probe_runtime(config, command(), runner=runner)
            result["package_inventory_sha256"] = config.runtime.package_inventory_sha256
            result["custom_modules"]["reward"]["source"] = "/tmp/wrong.py"
            with self.assertRaisesRegex(TrainingError, "wrong custom reward"):
                probe_runtime(config, command(), runner=runner)
            result["custom_modules"]["reward"]["source"] = str(
                REPOSITORY_ROOT
                / "src/compute_as_a_teacher/training/verl_reward.py"
            )
            result["actor_optimizer"]["kind"] = None
            with self.assertRaisesRegex(TrainingError, "does not use AdamW"):
                probe_runtime(config, command(), runner=runner)

            single = load_text(resolved_single_h100_config_text())
            result["actor_optimizer"]["kind"] = "torch.optim.AdamW"
            result["gpu_count"] = 1
            result["gpus"] = [
                {
                    "index": 0,
                    "name": "NVIDIA H100 80GB HBM3",
                    "capability": [9, 0],
                    "bf16_supported": True,
                    "free_memory_bytes": 60_000_000_000,
                    "total_memory_bytes": 80_000_000_000,
                }
            ]
            result["trainer_image_digest"] = None
            observed = probe_runtime(single, command(), runner=runner)
            self.assertEqual(
                observed["runtime_identity"]["kind"],
                "direct_host_package_inventory_v1",
            )
            result["trainer_image_digest"] = "sha256:" + "e" * 64
            with self.assertRaisesRegex(TrainingError, "must not claim"):
                probe_runtime(single, command(), runner=runner)

    def test_hydra_and_tokenizer_probes_are_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config, _ = model_fixture(Path(temporary))

            def hydra_runner(*args, **kwargs):
                return subprocess.CompletedProcess(args[0], 0, "resolved: true\n", "")

            composed = compose_verl_config(command(), runner=hydra_runner)
            self.assertEqual(composed["bytes"], len(b"resolved: true\n"))

            canary_answer = sha256_text(
                f"{config.fingerprint}:anchor-context-tail"
            )[:32]
            tokenizer_result = {
                "rows": 500,
                "min_prompt_tokens": 10,
                "max_prompt_tokens": 300,
                "overlong_rows": 0,
                "chat_template_sha256": config.policy.chat_template_sha256,
                "tokenizer_class": "Qwen2TokenizerFast",
                "anchor_prompt_overhead_tokens": 600,
                "anchor_rollout_token_budget": 8 * 1536,
                "anchor_completion_tokens": 1536,
                "anchor_boundary_margin_tokens": 16,
                "required_anchor_context_tokens": 600 + 8 * 1536 + 1536 + 16,
                "anchor_context_canary_message": (
                    f"large anchor context canary with unique \\boxed{{{canary_answer}}}"
                ),
                "anchor_context_canary_prompt_tokens": 600 + 8 * 1536 + 16,
                "anchor_context_canary_required_tokens": (
                    600 + 8 * 1536 + 16 + 1536
                ),
            }

            def tokenizer_runner(*args, **kwargs):
                return subprocess.CompletedProcess(
                    args[0],
                    0,
                    "CAT_PREFLIGHT_JSON=" + json.dumps(tokenizer_result) + "\n",
                    "",
                )

            self.assertEqual(
                probe_tokenizer(
                    config,
                    command(),
                    Path(temporary),
                    REPOSITORY_ROOT,
                    runner=tokenizer_runner,
                )["rows"],
                500,
            )
            tokenizer_result["overlong_rows"] = 1
            with self.assertRaisesRegex(TrainingError, "overlong"):
                probe_tokenizer(
                    config,
                    command(),
                    Path(temporary),
                    REPOSITORY_ROOT,
                    runner=tokenizer_runner,
                )
            tokenizer_result["overlong_rows"] = 0
            tokenizer_result["rows"] = 8
            self.assertEqual(
                probe_tokenizer(
                    config,
                    command(),
                    Path(temporary),
                    REPOSITORY_ROOT,
                    expected_rows=8,
                    runner=tokenizer_runner,
                )["rows"],
                8,
            )
            with self.assertRaisesRegex(TrainingError, "expected 7"):
                probe_tokenizer(
                    config,
                    command(),
                    Path(temporary),
                    REPOSITORY_ROOT,
                    expected_rows=7,
                    runner=tokenizer_runner,
                )
            tokenizer_result["rows"] = 500
            tokenizer_result["anchor_prompt_overhead_tokens"] = 30_000
            tokenizer_result["required_anchor_context_tokens"] = (
                30_000 + 8 * 1536 + 1536 + 16
            )
            tokenizer_result["anchor_context_canary_prompt_tokens"] = (
                30_000 + 8 * 1536 + 16
            )
            tokenizer_result["anchor_context_canary_required_tokens"] = (
                30_000 + 8 * 1536 + 16 + 1536
            )
            with self.assertRaisesRegex(TrainingError, "Model context cannot fit"):
                probe_tokenizer(
                    config,
                    command(),
                    Path(temporary),
                    REPOSITORY_ROOT,
                    runner=tokenizer_runner,
                )

    def test_receipt_is_operationally_ready_only_after_hash_and_anchor_canary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config, _ = model_fixture(Path(temporary))
            manifest = {"plan_fingerprint": "f" * 64}
            patches = (
                patch(
                    "compute_as_a_teacher.training.preflight.verify_verl_checkout"
                ),
                patch(
                    "compute_as_a_teacher.training.preflight.inspect_model_snapshot",
                    return_value={"all_files_tree_sha256": "a" * 64},
                ),
                patch(
                    "compute_as_a_teacher.training.preflight.probe_runtime",
                    return_value={"gpu_count": 8},
                ),
                patch(
                    "compute_as_a_teacher.training.preflight.compose_verl_config",
                    return_value={"sha256": "b" * 64},
                ),
                patch(
                    "compute_as_a_teacher.training.preflight.probe_tokenizer",
                    return_value={
                        "rows": 500,
                        "anchor_context_canary_message": (
                            "large context canary with unique "
                            f"\\boxed{{{sha256_text(f'{config.fingerprint}:anchor-context-tail')[:32]}}}"
                        ),
                        "anchor_context_canary_expected_answer": sha256_text(
                            f"{config.fingerprint}:anchor-context-tail"
                        )[:32],
                        "anchor_context_canary_required_tokens": 14_000,
                        "model_context_tokens": 32_768,
                    },
                ),
                patch(
                    "compute_as_a_teacher.training.preflight.probe_anchor",
                    return_value={
                        "model": config.runtime.anchor_model,
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
                ),
            )
            with (
                patches[0],
                patches[1],
                patches[2],
                patches[3],
                patches[4] as tokenizer_probe,
                patches[5],
            ):
                partial = run_preflight(
                    config,
                    manifest,
                    command(),
                    Path(temporary),
                    REPOSITORY_ROOT,
                    hash_model=False,
                    check_anchor=False,
                )
                full = run_preflight(
                    config,
                    manifest,
                    command(),
                    Path(temporary),
                    REPOSITORY_ROOT,
                    hash_model=True,
                    check_anchor=True,
                    qualification_lineage={
                        "profile": {
                            "name": "one_step",
                            "prompt_count": 8,
                            "prompt_batch_size": 8,
                            "max_steps": 1,
                            "save_every_steps": 1,
                        },
                        "qualification_fingerprint": "1" * 64,
                        "training_data_sha256": "2" * 64,
                        "command_sha256": "3" * 64,
                    },
                )
            self.assertFalse(partial["operationally_ready_to_launch"])
            self.assertEqual(
                partial["missing_gates"], ["model content hash", "anchor canary"]
            )
            self.assertTrue(full["operationally_ready_to_launch"])
            self.assertFalse(full["scientifically_attested"])
            self.assertEqual(
                full["qualification_lineage"]["profile"]["name"],
                "one_step",
            )
            self.assertEqual(tokenizer_probe.call_args.kwargs["expected_rows"], 8)
            self.assertNotIn(
                "anchor_context_canary_message",
                full["checks"]["tokenizer"],
            )
            self.assertRegex(full["preflight_fingerprint"], r"^[0-9a-f]{64}$")
            receipt_path = write_preflight_receipt(Path(temporary), full)
            self.assertTrue(receipt_path.is_file())
            self.assertTrue(
                (
                    Path(temporary)
                    / "preflights"
                    / f"{full['preflight_fingerprint']}.json"
                ).is_file()
            )
            changed = dict(full)
            changed["operationally_ready_to_launch"] = False
            with self.assertRaisesRegex(TrainingError, "fingerprint mismatch"):
                write_preflight_receipt(Path(temporary), changed)


if __name__ == "__main__":
    unittest.main()
