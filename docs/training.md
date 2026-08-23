# MATH-500 RL-synthesis training

This pipeline reproduces the paper's reference-free MATH-500 training loop on top
of a pinned `verl` batch-reward interface. The repository prepares and verifies the
run, but it does not download a model, install a GPU stack, start a server, or run
training automatically.

## Protocol boundary

For each update, the implementation:

1. samples eight fresh responses per question from the current policy;
2. sends those eight ordered response texts, without a separate question, to the
   frozen initial policy;
3. gives each response reward `1` when its final boxed answer exactly matches the
   synthesized final boxed answer, and `0` otherwise; and
4. lets `verl` compute GRPO group advantages and update only the current policy,
   with reward-level KL regularization against the initial policy.

The registered paper profile fixes a prompt batch of 256, group size 8, learning
rate `5e-7`, constant schedule with no warmup, KL coefficient `1e-3`, 1,000 updates,
1,536 generated tokens, AdamW, FSDP, and eight GPUs. Qwen3-4B sampling uses
temperature `0.7`, top-p `0.8`, top-k `20`, and `/no_think`.

Several implementation details are not reported by the paper. This repository
declares them as local choices: `verl` v0.5.0 at commit
[`8fdc4d3f202f41461f4de9f42a637228e342668b`](https://github.com/verl-project/verl/commit/8fdc4d3f202f41461f4de9f42a637228e342668b),
sample standard deviation (`ddof=1`) with epsilon `1e-6`, zero advantages for a
constant-reward group, PPO clip `0.2`, one PPO epoch, mini-batch size 256 sequences,
AdamW defaults recorded in the config, checkpointing every 100 steps, and selection
of the fixed step-1,000 checkpoint. The raw problem prompt, synthesis delimiters,
base seeds, exact boxed-string extractor, model registry identity, and repaired
Appendix F prompt are also versioned local choices.

The older `verl` revision is intentional. Its
[`BatchRewardManager`](https://github.com/verl-project/verl/blob/8fdc4d3f202f41461f4de9f42a637228e342668b/verl/workers/reward_manager/batch.py)
passes the complete response batch to one custom reward function, allowing the
adapter to group eight sibling rollouts before calling the synthesizer. This adapter
should not be moved to another `verl` revision without revalidating the reward and
Hydra contracts.

## No-download setup

Prepare or verify the locked question-only dataset first:

```bash
python3 scripts/prepare_math500.py
python3 scripts/prepare_math500.py --verify-only
```

Provide these assets yourself; the training CLI never fetches them:

- an immutable local Qwen3-4B snapshot, including its tokenizer and chat template;
- a Python executable in a compatible `verl`/Torch/Ray/vLLM GPU environment; and
- a checkout of `verl` at the required commit.

For example, create and pin the framework checkout outside this repository, then
install its GPU dependencies according to that revision's upstream instructions:

```bash
git clone https://github.com/verl-project/verl.git /abs/path/to/verl
git -C /abs/path/to/verl checkout 8fdc4d3f202f41461f4de9f42a637228e342668b
git -C /abs/path/to/verl rev-parse HEAD
```

Copy `configs/training/math500_cat_grpo.example.toml` and replace every
`required_*` field. Pin the full model and tokenizer commit SHAs and the SHA-256 of
the exact raw chat-template source. `policy` and `anchor.model` must be identical.
Set absolute paths for `runtime.python_executable`, `runtime.verl_source_path`, and
`runtime.model_path`. Keep `runtime.download_allowed = false`; launch also exports
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.

## Frozen anchor service

The reward worker expects an OpenAI-compatible chat-completions endpoint at
`runtime.anchor_base_url`. It sends one user message containing only the eight
rollout texts, plus the configured model name, seed, temperature, top-p, top-k, and
token limit. The endpoint must serve the untouched initial snapshot throughout the
run, never the actor checkpoint being updated.

One possible local vLLM launch is:

```bash
export CAT_ANCHOR_API_KEY='replace-with-a-local-secret'
vllm serve /abs/path/to/qwen3-4b-snapshot \
  --served-model-name cat-frozen-qwen3-4b \
  --host 127.0.0.1 --port 8001 \
  --api-key "$CAT_ANCHOR_API_KEY"
```

Keep the service name, endpoint, and API-key environment-variable name aligned with
the training config. Verify that the server honors `top_k`, `seed`, and the same
tokenizer/chat template. An invalid or unavailable anchor fails the reward batch
closed; gold answers are not a fallback.

## Prepare and launch

The example config supports a full model-free check despite its placeholders:

```bash
python3 scripts/train_math500.py prepare \
  --config configs/training/math500_cat_grpo.example.toml \
  --run-dir outputs/training/unused-dry-run \
  --dry-run
```

It should report 500 problems, 256 prompts and 2,048 policy trajectories per update,
and 256 anchor calls per update. It writes nothing and imports neither `verl` nor a
model framework.

The offline contract tests also use no model or GPU:

```bash
python3 -m unittest discover -s tests/training -v
```

With a resolved config, write and verify an immutable plan:

```bash
python3 scripts/train_math500.py prepare \
  --config configs/training/math500_cat_grpo.toml \
  --run-dir outputs/training/math500-cat

python3 scripts/train_math500.py inspect \
  --run-dir outputs/training/math500-cat
```

The plan contains the label-free `math500_train.jsonl`, exact `verl` argv and
environment, and hashes of the prompt and Python source tree. Later commands refuse
changed live code or prompts. Launch is a safe preflight by default:

```bash
python3 scripts/train_math500.py launch \
  --config configs/training/math500_cat_grpo.toml \
  --run-dir outputs/training/math500-cat
```

It prints the command without executing it. Only the explicit form starts the GPU
job, after checking the Python executable, local model directory, config/plan match,
offline policy, and exact `verl` Git revision:

```bash
python3 scripts/train_math500.py launch \
  --config configs/training/math500_cat_grpo.toml \
  --run-dir outputs/training/math500-cat \
  --execute
```

## Resume and fixed checkpoint

`checkpointing.resume_mode = "auto"` and the planned `trainer.default_local_dir`
make relaunching the same resolved config and run directory resume from the latest
valid `verl` checkpoint. Do not re-plan after checkpoints or other descendants
exist. Inspect the plan, restart the frozen anchor from the same initial snapshot,
then rerun `launch --execute`.

The protocol does not use labels for early stopping or checkpoint selection. A run
is complete only when `latest_checkpointed_iteration.txt` records step 1,000. Ask
the CLI for the exact FSDP-to-Hugging-Face export argv:

```bash
python3 scripts/train_math500.py merge-command \
  --config configs/training/math500_cat_grpo.toml \
  --run-dir outputs/training/math500-cat \
  --export-dir outputs/training/math500-cat/exports/global_step_1000
```

`merge-command` only prints the command. Run that argv in the pinned `verl`
environment, then register the export:

```bash
python3 scripts/train_math500.py register-checkpoint \
  --run-dir outputs/training/math500-cat \
  --export-dir outputs/training/math500-cat/exports/global_step_1000
```

Registration requires `config.json` and model weights, inventories and hashes both
the actor checkpoint and merged export, and records fixed-step, label-free lineage.

## Evaluate the trained export

Create raw and inference-time-synthesis configs bound to the registered export
hash:

```bash
python3 scripts/train_math500.py plan-trained-eval \
  --run-dir outputs/training/math500-cat \
  --output-dir outputs/training/math500-cat/eval-configs \
  --served-model math500-cat-final
```

Serve `exports/global_step_1000` under that exact model name with an
OpenAI-compatible server. For example:

```bash
export CAT_EVAL_API_KEY='replace-with-a-local-secret'
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 vllm serve \
  outputs/training/math500-cat/exports/global_step_1000 \
  --served-model-name math500-cat-final \
  --host 127.0.0.1 --port 8000 \
  --api-key "$CAT_EVAL_API_KEY"
```

The server is a separate process; the evaluation command does not fetch or load the
export. Use the generated raw config to plan eight trained-policy samples per
problem:

```bash
python3 scripts/evaluate_math500.py plan-raw \
  --config outputs/training/math500-cat/eval-configs/math500_trained_raw.toml \
  --run-dir outputs/evals/math500-cat-final-raw
```

Execute the plan through the resumable standard-library adapter:

```bash
python3 scripts/evaluate_math500.py run-openai \
  --run-dir outputs/evals/math500-cat-final-raw \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key-env CAT_EVAL_API_KEY \
  --batch-size 16
```

Use `--max-requests` for a small smoke test; rerunning the same command resumes from
validated per-task results. Once raw execution is complete, run synthesis and the
separate labels-only scoring boundary:

```bash
python3 scripts/evaluate_math500.py plan-synthesis \
  --config outputs/training/math500-cat/eval-configs/math500_trained_synthesis.toml \
  --raw-run-dir outputs/evals/math500-cat-final-raw \
  --run-dir outputs/evals/math500-cat-final-synthesis

python3 scripts/evaluate_math500.py run-openai \
  --run-dir outputs/evals/math500-cat-final-synthesis \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key-env CAT_EVAL_API_KEY \
  --batch-size 16

python3 scripts/evaluate_math500.py score-raw \
  --run-dir outputs/evals/math500-cat-final-raw \
  --config configs/evals/math500_scoring.toml

python3 scripts/evaluate_math500.py score-synthesis \
  --run-dir outputs/evals/math500-cat-final-synthesis \
  --raw-run-dir outputs/evals/math500-cat-final-raw \
  --config configs/evals/math500_scoring.toml
```

See [`docs/evaluation.md`](evaluation.md) for metric and artifact details. The two
runs measure the trained policy's raw performance and its optional additional gain
from inference-time self-synthesis.

## Label firewall and limitations

Training reads only `data/math500/questions.jsonl`. The generated `verl` records
contain the rendered prompt, an opaque question ID, and a `ground_truth: null`
placeholder required by the pinned batch-reward API. The reward adapter rejects any
non-null ground truth. `labels.jsonl`, answers, and reference solutions are excluded
from reward, prompt selection, early stopping, and checkpoint selection; scoring is
the first labels-only boundary.

The paper reports eight H100s for its trainer, but this implementation also requires
an external frozen-anchor endpoint. Its hardware is not included in the configured
eight-GPU trainer allocation, so a run using separate anchor hardware is a
system-resource reproduction rather than an exact accounting reproduction.
Checkpoint registration remains non-reportable because the external anchor service
and distributed runtime are not content-attested. `run-openai` is also marked
non-reportable because an HTTP endpoint cannot prove it serves the registered
checkpoint. No training or real-model evaluation has been run as part of building
this infrastructure.
