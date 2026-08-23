from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from compute_as_a_teacher.evaluation.artifacts import canonical_json_bytes, sha256_bytes
from compute_as_a_teacher.evaluation.config import (
    MATH500_PROTOCOL_VERSION,
    PromptSpec,
    SynthesisEvalConfig,
)
from compute_as_a_teacher.evaluation.schemas import ModelSpec, SamplingSpec
from compute_as_a_teacher.training.config import TRAINING_PROTOCOL_VERSION
from compute_as_a_teacher.training.errors import TrainingError
from compute_as_a_teacher.training.experiment_registry import (
    load_experiment_preregistration,
    load_final_experiment_registry,
    verify_final_experiment_registry,
    verify_preregistered_training_stage,
    write_experiment_preregistration,
    write_final_experiment_registry,
)


MODULE = "compute_as_a_teacher.training.experiment_registry"


def _model(
    model_id: str,
    revision: str,
    *,
    provider: str = "huggingface",
    adapter_version: str = "transformers-generate-v1",
) -> ModelSpec:
    return ModelSpec(
        provider=provider,
        model_id=model_id,
        revision=revision,
        tokenizer_id="qwen3-tokenizer",
        tokenizer_revision="b" * 40,
        chat_template_sha256="c" * 64,
        adapter_version=adapter_version,
        dtype="bfloat16",
        quantization="none",
        seed_support="strict",
    )


def _sampling(seed: int) -> SamplingSpec:
    return SamplingSpec(
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        max_new_tokens=1536,
        num_beams=1,
        repetition_penalty=1.0,
        stop=(),
        base_seed=seed,
    )


class ExperimentRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.run_dirs = {
            name: self.root / name
            for name in (
                "initial_raw",
                "initial_synthesis",
                "training",
                "trained_raw",
                "trained_synthesis",
            )
        }
        for run_dir in self.run_dirs.values():
            run_dir.mkdir()
            (run_dir / "manifest.json").write_text(
                '{"fixture":true}\n', encoding="utf-8"
            )
        self.synthesis_config_path = self.root / "initial_synthesis.toml"
        self.synthesis_config_path.write_text("# fixture\n", encoding="utf-8")
        self.preregistration_path = self.root / "preregistration.json"
        self.registry_path = self.root / "registry.json"

        self.policy = _model("qwen3-policy", "a" * 40)
        self.initial_eval_model = _model(
            "initial-policy-endpoint",
            "a" * 40,
            provider="openai-compatible",
            adapter_version="openai-compatible-chat-v1",
        )
        self.trained_policy = _model("trained-final-export", "d" * 64)
        self.trained_anchor = _model(
            "frozen-initial-policy",
            "a" * 40,
            provider="openai-compatible",
            adapter_version="openai-compatible-chat-v1",
        )
        self.raw_prompt = PromptSpec(
            path="prompts/math500/raw.txt",
            version="raw_math500_local_v1",
            prefix="/no_think\n",
        )
        self.synthesis_prompt = PromptSpec(
            path="prompts/math500/synthesis.txt",
            version="synthesis_cot_appendix_f_literal_v1",
            prefix="/no_think\n",
        )
        self.raw_sampling = _sampling(1729)
        self.synthesis_sampling = _sampling(2718)
        self.synthesis_config = SynthesisEvalConfig(
            schema_version=2,
            kind="synthesis",
            protocol_version=MATH500_PROTOCOL_VERSION,
            run_name="initial-synthesis",
            required_rollouts=8,
            anchor_relation="same_as_raw",
            prompt=self.synthesis_prompt,
            anchor=self.initial_eval_model,
            sampling=self.synthesis_sampling,
        )

        self.questions = {
            "path": "data/math500_questions.jsonl",
            "sha256": "1" * 64,
            "bytes": 100,
            "rows": 500,
        }
        self.dataset_lock = {
            "path": "data/math500.lock.json",
            "sha256": "2" * 64,
            "bytes": 200,
        }
        self.raw_prompt_manifest = {
            "version": self.raw_prompt.version,
            "template_sha256": "4" * 64,
            "contract_sha256": "5" * 64,
        }
        self.synthesis_prompt_manifest = {
            "version": self.synthesis_prompt.version,
            "template_sha256": "6" * 64,
            "contract_sha256": "7" * 64,
        }
        self.eval_firewall = {
            "labels_loaded": False,
            "reference_answers_loaded": False,
            "reference_solutions_loaded": False,
            "locked_questions_verified": True,
        }
        self.synthesis_firewall = {
            **self.eval_firewall,
            "question_field_supplied_to_synthesis": False,
        }

        raw_config = {
            "schema_version": 1,
            "kind": "raw",
            "protocol_version": MATH500_PROTOCOL_VERSION,
            "run_name": "initial-raw",
            "questions_path": self.questions["path"],
            "dataset_lock_path": self.dataset_lock["path"],
            "rollouts_per_problem": 8,
            "prompt": self.raw_prompt.to_dict(),
            "model": self.initial_eval_model.to_dict(),
            "sampling": self.raw_sampling.to_dict(),
        }
        self.initial_raw_manifest = {
            "kind": "raw",
            "protocol_version": MATH500_PROTOCOL_VERSION,
            "plan_fingerprint": "8" * 64,
            "model": self.initial_eval_model.to_dict(),
            "config": raw_config,
            "inputs": {
                "questions": self.questions,
                "dataset_lock": self.dataset_lock,
            },
            "counts": {"problems": 500, "requests": 4000},
            "label_firewall": self.eval_firewall,
            "prompt": self.raw_prompt_manifest,
        }
        training_config = {
            "questions_path": self.questions["path"],
            "dataset_lock_path": self.dataset_lock["path"],
            "policy": self.policy.to_dict(),
            "anchor": {
                "source": "initial_policy",
                "frozen": True,
                "model": self.policy.to_dict(),
            },
            "rollouts": {
                "group_size": 8,
                "prompt": self.raw_prompt.to_dict(),
                "sampling": self.raw_sampling.to_dict(),
            },
            "synthesis": {
                "required_rollouts": 8,
                "prompt": self.synthesis_prompt.to_dict(),
                "sampling": self.synthesis_sampling.to_dict(),
            },
            "reward": {"labels_allowed": False},
            "grpo": {"max_steps": 1000},
            "checkpointing": {"selected_checkpoint": "fixed_final_step"},
            "runtime": {"anchor_model": self.trained_anchor.model_id},
        }
        self.training_manifest = {
            "protocol_version": TRAINING_PROTOCOL_VERSION,
            "plan_fingerprint": "9" * 64,
            "config_fingerprint": "a" * 64,
            "config": training_config,
            "inputs": {
                "questions": self.questions,
                "dataset_lock": self.dataset_lock,
                "raw_prompt": {
                    "path": self.raw_prompt.path,
                    "sha256": self.raw_prompt_manifest["template_sha256"],
                    "bytes": 20,
                },
                "synthesis_prompt": {
                    "path": self.synthesis_prompt.path,
                    "sha256": self.synthesis_prompt_manifest["template_sha256"],
                    "bytes": 200,
                },
                "raw_prompt_contract_sha256": self.raw_prompt_manifest[
                    "contract_sha256"
                ],
                "synthesis_prompt_contract_sha256": self.synthesis_prompt_manifest[
                    "contract_sha256"
                ],
            },
            "counts": {"problems": 500},
            "label_firewall": {
                "labels_loaded": False,
                "reference_answers_loaded": False,
                "reference_solutions_loaded": False,
                "reward_uses_synthesized_answer_only": True,
                "checkpoint_selection_uses_labels": False,
            },
        }
        self.initial_synthesis_manifest = self._synthesis_manifest(
            self.synthesis_config.to_dict(),
            self.initial_eval_model,
            self.initial_raw_manifest["plan_fingerprint"],
            "b" * 64,
        )
        trained_raw_config = copy.deepcopy(raw_config)
        trained_raw_config["run_name"] = "trained-raw"
        trained_raw_config["model"] = self.trained_policy.to_dict()
        self.trained_raw_manifest = {
            **copy.deepcopy(self.initial_raw_manifest),
            "plan_fingerprint": "c" * 64,
            "model": self.trained_policy.to_dict(),
            "config": trained_raw_config,
        }
        trained_synthesis_config = {
            **self.synthesis_config.to_dict(),
            "run_name": "trained-synthesis",
            "anchor_relation": "frozen_initial_for_trained_raw",
            "anchor": self.trained_anchor.to_dict(),
        }
        self.trained_synthesis_manifest = self._synthesis_manifest(
            trained_synthesis_config,
            self.trained_anchor,
            self.trained_raw_manifest["plan_fingerprint"],
            "e" * 64,
        )
        self.checkpoint = {
            "training_plan_fingerprint": self.training_manifest[
                "plan_fingerprint"
            ],
            "selected_by": "fixed_final_step",
            "step": 1000,
            "base_model_fingerprint": sha256_bytes(
                canonical_json_bytes(self.policy.to_dict())
            ),
            "export": {
                "path": "exports/final",
                "tree_sha256": self.trained_policy.revision,
                "files": [],
            },
        }
        self.completion = {
            "training_plan_fingerprint": self.training_manifest[
                "plan_fingerprint"
            ],
            "completed_step": 1000,
            "selection": "fixed_final_step_without_labels",
            "labels_loaded": False,
            "export_tree_sha256": self.trained_policy.revision,
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _synthesis_manifest(
        self,
        config: dict[str, object],
        anchor: ModelSpec,
        raw_fingerprint: str,
        fingerprint: str,
    ) -> dict[str, object]:
        return {
            "kind": "synthesis",
            "protocol_version": MATH500_PROTOCOL_VERSION,
            "plan_fingerprint": fingerprint,
            "model": anchor.to_dict(),
            "config": config,
            "inputs": {"raw_plan_fingerprint": raw_fingerprint},
            "counts": {"problems": 500, "requests": 500},
            "label_firewall": self.synthesis_firewall,
            "prompt": self.synthesis_prompt_manifest,
        }

    def _load_plan(self, run_dir: Path, *, expected_kind: str):
        manifest = {
            "initial_raw": self.initial_raw_manifest,
            "initial_synthesis": self.initial_synthesis_manifest,
            "trained_raw": self.trained_raw_manifest,
            "trained_synthesis": self.trained_synthesis_manifest,
        }[Path(run_dir).name]
        self.assertEqual(manifest["kind"], expected_kind)
        return copy.deepcopy(manifest), []

    @contextmanager
    def _patched_loaders(self):
        with ExitStack() as stack:
            stack.enter_context(patch(f"{MODULE}.load_plan", self._load_plan))
            stack.enter_context(
                patch(
                    f"{MODULE}.load_training_plan",
                    return_value=(copy.deepcopy(self.training_manifest), []),
                )
            )
            stack.enter_context(
                patch(
                    f"{MODULE}.load_synthesis_config",
                    return_value=self.synthesis_config,
                )
            )
            stack.enter_context(
                patch(
                    f"{MODULE}.load_registered_checkpoint",
                    return_value=(
                        copy.deepcopy(self.checkpoint),
                        copy.deepcopy(self.completion),
                    ),
                )
            )
            yield

    def _preregister(self) -> dict[str, object]:
        return write_experiment_preregistration(
            self.preregistration_path,
            self.run_dirs["initial_raw"],
            self.synthesis_config_path,
            self.run_dirs["training"],
        )

    def _finalize(self) -> dict[str, object]:
        (self.run_dirs["training"] / "checkpoint_manifest.json").write_text(
            '{"fixture":"checkpoint"}\n', encoding="utf-8"
        )
        (self.run_dirs["training"] / "completion.json").write_text(
            '{"fixture":"completion"}\n', encoding="utf-8"
        )
        return write_final_experiment_registry(
            self.registry_path,
            self.preregistration_path,
            self.run_dirs["initial_raw"],
            self.run_dirs["initial_synthesis"],
            self.run_dirs["training"],
            self.run_dirs["trained_raw"],
            self.run_dirs["trained_synthesis"],
        )

    def test_two_phase_registry_joins_pi0_training_and_piT_evaluations(self) -> None:
        with self._patched_loaders():
            preregistration = self._preregister()
            loaded_preregistration = load_experiment_preregistration(
                self.preregistration_path
            )
            registry = self._finalize()
            verified = verify_final_experiment_registry(
                self.registry_path,
                self.preregistration_path,
                self.run_dirs["initial_raw"],
                self.run_dirs["initial_synthesis"],
                self.run_dirs["training"],
                self.run_dirs["trained_raw"],
                self.run_dirs["trained_synthesis"],
            )

        self.assertEqual(preregistration, loaded_preregistration)
        self.assertEqual(registry, verified)
        self.assertFalse(registry["results_included"])
        self.assertFalse(registry["labels_loaded"])
        self.assertEqual(
            registry["lineage"]["initial_chain"]["declared_relation"],
            "pi0_rollouts_synthesized_by_pi0",
        )
        self.assertEqual(
            registry["lineage"]["trained_chain"]["declared_relation"],
            "piT_rollouts_synthesized_by_frozen_pi0",
        )
        self.assertEqual(
            registry["stages"]["fixed_final_checkpoint"]["step"], 1000
        )
        self.assertFalse(preregistration["scientifically_attested"])
        self.assertFalse(registry["scientifically_attested"])

    def test_finalization_rejects_nested_stage_directories(self) -> None:
        with self._patched_loaders():
            self._preregister()
        self.run_dirs["trained_raw"] = self.run_dirs["training"] / "trained_raw"
        self.run_dirs["trained_raw"].mkdir()
        (self.run_dirs["trained_raw"] / "manifest.json").write_text(
            '{"fixture":true}\n', encoding="utf-8"
        )
        with self._patched_loaders(), self.assertRaisesRegex(
            TrainingError, "must be disjoint"
        ):
            self._finalize()

    def test_preregistration_must_precede_result_artifacts(self) -> None:
        (self.run_dirs["initial_raw"] / "results").mkdir()
        with self._patched_loaders(), self.assertRaisesRegex(
            TrainingError, "before result artifacts exist"
        ):
            self._preregister()

    def test_preregistration_must_precede_training_logs(self) -> None:
        (self.run_dirs["training"] / "logs").mkdir()
        with self._patched_loaders(), self.assertRaisesRegex(
            TrainingError, "before result artifacts exist"
        ):
            self._preregister()

    def test_preregistration_reverifies_the_current_training_stage(self) -> None:
        with self._patched_loaders():
            preregistration = self._preregister()
            verified = verify_preregistered_training_stage(
                self.preregistration_path,
                self.run_dirs["training"],
            )
        self.assertEqual(verified, preregistration)

    def test_launch_verification_detects_baseline_source_drift(self) -> None:
        with self._patched_loaders():
            self._preregister()
        self.synthesis_config_path.write_text("# changed\n", encoding="utf-8")
        with self._patched_loaders(), self.assertRaisesRegex(
            TrainingError, "changed after registration"
        ):
            verify_preregistered_training_stage(
                self.preregistration_path,
                self.run_dirs["training"],
            )

    def test_registry_outputs_must_be_outside_bound_stage_trees(self) -> None:
        with self._patched_loaders(), self.assertRaisesRegex(
            TrainingError, "outside every bound run"
        ):
            write_experiment_preregistration(
                self.run_dirs["initial_raw"] / "requests.jsonl",
                self.run_dirs["initial_raw"],
                self.synthesis_config_path,
                self.run_dirs["training"],
                force=True,
            )

        with self._patched_loaders():
            self._preregister()
        self.registry_path = self.run_dirs["trained_raw"] / "scores.jsonl"
        with self._patched_loaders(), self.assertRaisesRegex(
            TrainingError, "outside every bound run"
        ):
            self._finalize()

    def test_final_registry_cannot_write_inside_checkpoint_export(self) -> None:
        with self._patched_loaders():
            self._preregister()
        export_dir = self.root / "external-export"
        export_dir.mkdir()
        self.checkpoint["export"]["path"] = str(export_dir)
        self.registry_path = export_dir / "registry.json"
        with self._patched_loaders(), self.assertRaisesRegex(
            TrainingError, "outside the registered checkpoint export"
        ):
            self._finalize()

    def test_preregistration_rejects_synthesis_seed_drift(self) -> None:
        self.synthesis_config = replace(
            self.synthesis_config,
            sampling=replace(self.synthesis_sampling, base_seed=999),
        )
        with self._patched_loaders(), self.assertRaisesRegex(
            TrainingError, "Synthesis sampling differs from training"
        ):
            self._preregister()

    def test_preregistration_rejects_dataset_content_drift(self) -> None:
        self.training_manifest["inputs"]["questions"] = {
            **self.questions,
            "sha256": "f" * 64,
        }
        with self._patched_loaders(), self.assertRaisesRegex(
            TrainingError, "question artifacts differ"
        ):
            self._preregister()

    def test_finalization_rejects_wrong_trained_synthesis_anchor(self) -> None:
        with self._patched_loaders():
            self._preregister()
        wrong_anchor = replace(self.trained_anchor, revision="f" * 40)
        self.trained_synthesis_manifest["model"] = wrong_anchor.to_dict()
        self.trained_synthesis_manifest["config"]["anchor"] = wrong_anchor.to_dict()
        with self._patched_loaders(), self.assertRaisesRegex(
            TrainingError, "not frozen pi0"
        ):
            self._finalize()

    def test_finalization_rejects_common_eval_seed_drift(self) -> None:
        with self._patched_loaders():
            self._preregister()
        self.trained_raw_manifest["config"]["sampling"]["base_seed"] = 999
        with self._patched_loaders(), self.assertRaisesRegex(
            TrainingError, "evaluation seeds or sampling differ"
        ):
            self._finalize()

    def test_finalization_rejects_label_based_checkpoint_selection(self) -> None:
        with self._patched_loaders():
            self._preregister()
        self.completion["labels_loaded"] = True
        with self._patched_loaders(), self.assertRaisesRegex(
            TrainingError, "label-free fixed-step checkpoint selection"
        ):
            self._finalize()

    def test_registry_fingerprint_detects_tampering(self) -> None:
        with self._patched_loaders():
            self._preregister()
            self._finalize()
        value = json.loads(self.registry_path.read_text(encoding="utf-8"))
        value["state"] = "tampered"
        self.registry_path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaises(TrainingError):
            load_final_experiment_registry(self.registry_path)

    def test_verification_detects_registered_stage_tampering(self) -> None:
        with self._patched_loaders():
            self._preregister()
            self._finalize()
        (self.run_dirs["initial_raw"] / "manifest.json").write_text(
            '{"fixture":"tampered"}\n', encoding="utf-8"
        )
        with self._patched_loaders(), self.assertRaisesRegex(
            TrainingError, "changed after registration"
        ):
            verify_final_experiment_registry(
                self.registry_path,
                self.preregistration_path,
                self.run_dirs["initial_raw"],
                self.run_dirs["initial_synthesis"],
                self.run_dirs["training"],
                self.run_dirs["trained_raw"],
                self.run_dirs["trained_synthesis"],
            )


if __name__ == "__main__":
    unittest.main()
