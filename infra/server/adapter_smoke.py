#!/usr/bin/env python3
"""CPU-only compatibility smoke for the pinned Verl adapters."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path


class _Tokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize=False):
        del messages, add_generation_prompt
        return [1, 2] if tokenize else "user: synthetic prompt\nassistant:"

    def __call__(self, text, *, return_tensors, add_special_tokens):
        del text, return_tensors, add_special_tokens
        import torch

        return {
            "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        }

    def encode(self, text, *, add_special_tokens):
        del text, add_special_tokens
        return [1, 2]

    def decode(self, token_ids, *, skip_special_tokens):
        del token_ids, skip_special_tokens
        return r"Synthetic reasoning. \boxed{2}"


class _Anchor:
    def complete(self, **kwargs):
        del kwargs
        return r"# UNIFIED RESPONSE\nTherefore, the final answer is: $\boxed{2}$."


def run(repository_root: Path) -> dict:
    import numpy as np
    import ray
    import torch
    from omegaconf import OmegaConf
    from verl import DataProto
    from verl.trainer.main_ppo import create_rl_dataset
    from verl.trainer.ppo.reward import load_reward_manager

    if ray.is_initialized() or torch.cuda.is_initialized():
        raise RuntimeError("adapter smoke must begin without Ray or CUDA initialization")

    row = {
        "prompt": [{"role": "user", "content": "What is 1+1?"}],
        "data_source": "math500",
        "reward_model": {"ground_truth": None},
        "extra_info": {"question_id": "synthetic-q0", "index": 0},
    }
    with tempfile.TemporaryDirectory(prefix="cat-verl-smoke-") as temporary:
        source = Path(temporary) / "train.jsonl"
        source.write_text(json.dumps(row) + "\n", encoding="utf-8")
        reward_module = Path(temporary) / "reward.py"
        reward_module.write_text(
            "from compute_as_a_teacher.training.verl_reward import compute_score\n",
            encoding="utf-8",
        )
        data_config = OmegaConf.create(
            {
                "prompt_key": "prompt",
                "reward_fn_key": "data_source",
                "max_prompt_length": 32,
                "filter_overlong_prompts": False,
                "return_raw_chat": True,
                "truncation": "error",
                "custom_cls": {
                    "path": "pkg://compute_as_a_teacher.training.verl_dataset",
                    "name": "JsonlRLHFDataset",
                },
            }
        )
        dataset = create_rl_dataset(
            [str(source)], data_config, _Tokenizer(), None, is_train=True
        )
        item = dataset[0]
        batch_size = 8
        data = DataProto.from_dict(
            tensors={
                "prompts": torch.tensor([[1, 2]] * batch_size, dtype=torch.long),
                "responses": torch.tensor([[3, 4, 5]] * batch_size, dtype=torch.long),
                "attention_mask": torch.ones((batch_size, 5), dtype=torch.long),
            },
            non_tensors={
                "data_source": np.array(["math500"] * batch_size, dtype=object),
                "reward_model": np.array(
                    [{"ground_truth": None} for _ in range(batch_size)], dtype=object
                ),
                "extra_info": np.array(
                    [{"question_id": "synthetic-q0"} for _ in range(batch_size)],
                    dtype=object,
                ),
            },
        )
        config = OmegaConf.create(
            {
                "data": {"reward_fn_key": "data_source"},
                "reward_model": {"reward_manager": "batch"},
                "custom_reward_function": {
                    "path": str(reward_module),
                    "name": "compute_score",
                    "reward_kwargs": {
                        "repository_root": str(repository_root),
                        "prompt_path": "prompts/math500/synthesis_cot_appendix_f_literal.txt",
                        "prompt_version": "paper_appendix_f_cot_literal_v1",
                        "prompt_prefix": "/no_think\n",
                        "anchor_base_url": "http://127.0.0.1:1/v1",
                        "anchor_model": "cat-smoke-anchor",
                        "anchor_api_key_env": "CAT_UNUSED_SMOKE_KEY",
                        "anchor_timeout_seconds": 1.0,
                        "anchor_max_concurrency": 1,
                        "anchor_temperature": 0.7,
                        "anchor_top_p": 0.8,
                        "anchor_top_k": 20,
                        "anchor_max_tokens": 64,
                        "base_seed": 2718,
                    },
                },
            }
        )
        manager = load_reward_manager(
            config, _Tokenizer(), num_examine=0, anchor_client=_Anchor()
        )
        result = manager(data, return_dict=True)
        rewards = result["reward_tensor"][:, -1]
        dataset_name = type(dataset).__name__
        dataset_rows = len(dataset)
        manager_name = type(manager).__name__

    if len(dataset) != 1 or "raw_prompt" not in item:
        raise RuntimeError("JsonlRLHFDataset smoke produced an invalid row")
    if rewards.shape != (batch_size,) or not torch.equal(rewards, torch.ones(batch_size)):
        raise RuntimeError("BatchRewardManager smoke produced invalid rewards")
    if data.batch["acc"].tolist() != [1.0] * batch_size:
        raise RuntimeError("BatchRewardManager smoke produced invalid accuracy values")
    if ray.is_initialized() or torch.cuda.is_initialized():
        raise RuntimeError("adapter smoke initialized Ray or CUDA")
    return {
        "dataset_class": dataset_name,
        "dataset_rows": dataset_rows,
        "reward_manager_class": manager_name,
        "reward_rows": batch_size,
        "network_requests": 0,
        "model_weights_loaded": False,
        "ray_initialized": False,
        "cuda_initialized": False,
    }


if __name__ == "__main__":
    print(json.dumps(run(Path(__file__).resolve().parents[2]), sort_keys=True))
