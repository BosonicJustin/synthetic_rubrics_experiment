# MATH-500 RL-synthesis training

This pipeline implements paper-aligned planning and contracts for the reference-free
MATH-500 loop on a pinned `verl` batch-reward interface. It does not download a
model, install or attest a GPU stack, start a server, or claim that training has run.
Complete every gate in the [experiment plan](experiment_plan.md) before launching
the canonical 1,000-step job.

## Protocol boundary

For each update, the implementation:

1. samples eight fresh responses per question from the current policy;
2. sends those eight ordered response texts, without a separate question, to the
   frozen initial policy;
3. gives each response reward `1` when its final boxed answer exactly matches the
   synthesized final boxed answer, and `0` otherwise; and
4. lets `verl` compute GRPO group advantages and update only the current policy,
   with reward-level KL regularization against the initial policy.

The registered profile interprets the paper's global batch size 256 as prompts and
fixes group size 8, learning rate `5e-7`, constant schedule with no warmup, KL
coefficient `1e-3`, 1,000 updates, 1,536 generated tokens, AdamW, FSDP, and eight
GPUs. Qwen3-4B sampling uses temperature `0.7`, top-p `0.8`, top-k `20`, and
`/no_think`. The actor loss is pinned to sequence-mean/token-mean aggregation,
matching Eq. 6's token mean within each response followed by the group mean.

Several implementation details are not reported by the paper. This repository
declares them as local choices: `verl` v0.5.0 at commit
[`8fdc4d3f202f41461f4de9f42a637228e342668b`](https://github.com/verl-project/verl/commit/8fdc4d3f202f41461f4de9f42a637228e342668b),
sample standard deviation (`ddof=1`) with epsilon `1e-6`, zero advantages for a
constant-reward group, PPO clip `0.2`, one PPO epoch, and a mini-batch input of 256
prompts. Pinned Verl multiplies that local choice by eight rollouts before FSDP
sharding, yielding 2,048 response trajectories globally and 256 per rank. The
sampled-token `kl` penalty uses a fixed controller; AdamW defaults are recorded in the
config, checkpointing every 100 steps, and selection of the fixed step-1,000
checkpoint. The raw problem prompt, synthesis delimiters, base seeds, registered
boxed-string extractor, model registry identity, `do_sample=true`, single-beam
generation, repetition penalty `1.0`, and no stop strings are also versioned local
choices. The pinned Verl rollout and anchor adapters rely on their defaults for the
last three fields where the paper publishes no setting. The
canonical synthesis body is the displayed Appendix F text, including its
`$ boxed{answer}$` wording; the registered one-backslash repair is excluded from
this protocol. Both training and evaluation accept only `stop` and `length` finish
reasons and use the same 50,000-character boxed-answer safety cap. Target-runtime
qualification must confirm the generation defaults remain unchanged.
Appendix I does not publish its extraction regex, so the balanced-brace,
last-`\\boxed` scanner is a disclosed local approximation. The exact Appendix F
ending may itself produce unextractable `boxed{...}` text without a slash; canonical
training fails closed, while the repaired prompt is reserved for a new sensitivity
protocol. Target preflight also verifies from the pinned Verl source that the FSDP
actor constructs AdamW.

The older `verl` revision is intentional. Its
[`BatchRewardManager`](https://github.com/verl-project/verl/blob/8fdc4d3f202f41461f4de9f42a637228e342668b/verl/workers/reward_manager/batch.py)
passes the complete response batch to one custom reward function, allowing the
adapter to group eight sibling rollouts before calling the synthesizer. This adapter
should not be moved to another `verl` revision without revalidating the reward and
Hydra contracts.

## No-download setup

Prepare or verify the locked question-only dataset first:

```bash
uv sync --extra evaluation --frozen
source .venv/bin/activate
python3 scripts/prepare_math500.py
python3 scripts/prepare_math500.py --verify-only
```

Keep this repository environment active for the planning and orchestration commands
below. The separately configured `runtime.python_executable` remains the pinned GPU
environment used by Verl itself.

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

Discover the local identities without downloading or loading tensors:

```bash
export CAT_MODEL_REVISION='REPLACE_WITH_FULL_COMMIT_SHA'
python3 scripts/train_math500.py model-identity \
  --model-path "/abs/path/to/snapshots/$CAT_MODEL_REVISION"

export CAT_TRAINER_IMAGE_DIGEST='sha256:replace-with-64-hex-image-digest'
python3 scripts/train_math500.py runtime-identity \
  --python /abs/path/to/verl-environment/bin/python \
  --verl-source /abs/path/to/verl
```

Copy `configs/training/math500_cat_grpo.example.toml` and replace every
`required_*` field. The model and tokenizer revisions must be the same full commit
SHA, and `runtime.model_path` must end in that exact SHA. Pin the raw chat-template
SHA-256, full snapshot-tree SHA-256, trainer image digest, and target package-
inventory SHA-256 reported by the identity commands. `model-identity` prints the
snapshot revision, chat-template SHA-256, and tree SHA-256. It rejects directory
symlinks inside the resolved snapshot tree; ordinary file symlinks in a Hugging Face
snapshot are followed and their contents are hashed. `policy` and `anchor.model`
must be identical.

Set absolute paths for `runtime.python_executable`, `runtime.verl_source_path`, and
`runtime.model_path`. Obtain the immutable container-image digest from the
orchestrator or container metadata and set `CAT_TRAINER_IMAGE_DIGEST` yourself;
`runtime-identity` records that caller-supplied digest and the discovered package
inventory, but does not discover an image digest. Keep
`runtime.download_allowed = false`; launch also exports `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`.

## Frozen anchor service

The reward worker expects an OpenAI-compatible chat-completions endpoint at
`runtime.anchor_base_url`. It sends one user message containing only the eight
rollout texts, plus the configured model name, seed, temperature, top-p, top-k, and
token limit. The endpoint must serve the untouched initial snapshot throughout the
run, never the actor checkpoint being updated.

One possible local vLLM launch is:

```bash
export CAT_ANCHOR_API_KEY='replace-with-a-local-secret'
export CAT_ANCHOR_GPU_ID='REPLACE_WITH_ANCHOR_ONLY_DEVICE'
CUDA_VISIBLE_DEVICES="$CAT_ANCHOR_GPU_ID" vllm serve \
  "/abs/path/to/snapshots/$CAT_MODEL_REVISION" \
  --served-model-name cat-frozen-qwen3-4b \
  --host 127.0.0.1 --port 8001 \
  --api-key "$CAT_ANCHOR_API_KEY"
```

Keep the service name, endpoint, and API-key environment-variable name aligned with
the training config. Verify that the server honors `top_k`, `seed`, and the same
tokenizer/chat template. An invalid or unavailable anchor fails the reward batch
closed; gold answers are not a fallback. The trainer must see exactly eight separate
H100s, none visible to the anchor. If there is no ninth local GPU, serve the anchor
on another host or otherwise isolated hardware and update `runtime.anchor_base_url`.

## Optional W&B tracking

W&B is disabled by default, so the generated Verl command uses only the console
logger. To enable it, set `tracking.wandb.enabled = true` in the schema-v2 training
config and provide explicit `project` and `entity` values. The configured
`sdk_version = "0.21.1"` is a local reproducibility pin, not a setting reported by
the paper or pinned by Verl. `group` and `tags` are optional fingerprinted metadata.

For online mode, name the credential environment variable in `api_key_env` and
export the secret only in the launch environment. For example, set
`api_key_env = "CAT_WANDB_API_KEY"` and then:

```bash
export CAT_WANDB_API_KEY='replace-with-a-secret'
```

The credential value is injected only into the trainer process; it is not written to
the config, command plan, manifest, preview, or preflight receipt. Enabled tracking
is online-only: W&B 0.21.1 ignores `resume` in offline mode, so accepting that mode
would silently create a new run after restart. Leave W&B disabled for a no-network
run. Preflight never calls `wandb.init` or sends a telemetry request. When tracking
is enabled, it requires a successful import of W&B 0.21.1 from the configured Python
environment, checks the pinned Verl tracking call shape, and fails readiness if the
named credential is unset or blank.

The canonical W&B ID is `cat-` plus the first 32 hexadecimal characters of the
immutable training-config fingerprint. Qualification IDs append `-q-<profile>` and
also receive distinct non-reportable names, groups, and tags. The adapter fixes W&B
resume to `allow`, so restarting the same config resumes its telemetry run while a
changed config gets a new ID. Do not supply or generate another run ID.

W&B is observability, not recovery state. The local Verl checkpoint directory,
`latest_checkpointed_iteration.txt`, checkpoint validation, and the registered fixed
step-1,000 export remain authoritative even if W&B is unavailable or missing events.
Use the first executed `one_step` qualification as the live telemetry test: verify
the expected project, entity, ID, metrics, group, and tags before proceeding. The
`resume_three_step` qualification should then demonstrate that both the local
checkpoint and the same W&B run continue after restart.

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

The registered config remains the canonical 256-prompt, eight-GPU, 1,000-step run.
Qualification commands derive constrained, non-reportable plans from it without
weakening the canonical contract.

With a resolved config, write and verify an immutable plan:

```bash
python3 scripts/train_math500.py prepare \
  --config configs/training/math500_cat_grpo.toml \
  --run-dir outputs/training/math500-cat

python3 scripts/train_math500.py inspect \
  --run-dir outputs/training/math500-cat
```

The canonical plan contains the label-free `math500_train.jsonl`, exact `verl` argv
and environment, and hashes of the prompt and Python source tree. Later commands
refuse changed live code or prompts. Training and qualification plan writers use the
same advisory locks as launch, and execution reloads each plan only after acquiring
those locks, so `--force` cannot replace a live job's artifacts.

Before executing the initial raw baseline or creating trainer results, logs,
rollout records, or checkpoints, bind its plan, the intended initial synthesis
config, and the canonical training plan into a single preregistration:

```bash
python3 scripts/train_math500.py preregister-experiment \
  --output outputs/experiment/math500-preregistration.json \
  --initial-raw-run-dir outputs/evals/math500-initial-raw \
  --initial-synthesis-config configs/evals/math500_synthesis.toml \
  --training-run-dir outputs/training/math500-cat
```

Run this only after `plan-raw` has created the initial raw plan and `prepare` has
created the canonical training plan, but before either directory contains results,
logs, rollout records, or checkpoints. It rejects drift in the dataset, `pi_0`,
prompt bytes and contracts, sampling and seeds, group size, label firewall, or fixed
step-1,000 selection. The registered initial synthesis config must use
`anchor_relation = "same_as_raw"`; the later trained synthesis plan must use the
frozen-`pi_0` relation generated by `plan-trained-eval`.

In the target environment, record the immutable trainer image and run the complete
operational preflight before any trainer/Verl job. The frozen anchor service must
already be running for its canaries:

```bash
export CAT_TRAINER_IMAGE_DIGEST='sha256:replace-with-64-hex-image-digest'
python3 scripts/train_math500.py preflight \
  --config configs/training/math500_cat_grpo.toml \
  --run-dir outputs/training/math500-cat \
  --hash-model \
  --check-anchor \
  --write
```

This rehashes the snapshot against the config pin; validates Qwen safetensors headers,
shard indexes, BF16 tensor storage/no-quantization config, tokenizer assets, and chat template;
checks separate policy and worst-case eight-rollout synthesis context budgets;
tokenizes all 500 prompts; and composes the real Hydra job. It also requires the
exact pinned package inventory and trainer image, imports the repository dataset and
reward hooks from their expected paths, checks the reward signature, inventories
eight BF16 H100s, and enforces the configured free-memory floor before sending a
short semantic anchor canary and an exact-tokenized long-context canary whose boxed
nonce appears only near the tail. This verifies request acceptance and preservation
of that unique late value, not the endpoint's hidden truncation implementation. Omit
`--check-anchor` or `--hash-model` only for diagnosis;
such a receipt is not operationally launch-ready. Files and safetensors metadata are
read for hashing and validation, but tensors are not loaded and downloads remain
disabled.

Operational readiness is not scientific endpoint attestation. An HTTP model alias
cannot prove the anchor serves the pinned `pi_0` weights or uses disjoint hardware;
the launch attestation requires those external facts. Current execution schemas
nevertheless remain scientifically non-reportable. A reportable run requires a new
versioned, attested backend and schema.

Before the canonical launch, derive, inspect, and preview the one-step smoke in its
own directory:

```bash
python3 scripts/train_math500.py prepare-qualification \
  --run-dir outputs/training/math500-cat \
  --qualification-dir outputs/training/math500-cat-one-step \
  --profile one_step

python3 scripts/train_math500.py inspect-qualification \
  --qualification-dir outputs/training/math500-cat-one-step

python3 scripts/train_math500.py launch-qualification \
  --config configs/training/math500_cat_grpo.toml \
  --qualification-dir outputs/training/math500-cat-one-step
```

Repeat with separate directories for `resume_three_step` and
`full_shape_five_step`. These plans are label-free, fingerprinted to the canonical
plan, and hard non-reportable. The launch command is a preview unless `--execute` is
added; use `--execute` only inside the provisioned GPU environment.

After all three profiles reach their terminal checkpoints, copy
`configs/training/math500_launch_attestation.example.json` outside every run
directory. Replace every placeholder and false/zero limit with reviewed evidence
and numeric ceilings derived from the full-shape qualification. The total-cost cap
must cover the declared trainer and anchor GPU-hour ceilings at their recorded rates,
plus storage and network cost. These values are approval metadata; configure the
actual scheduler/provider wall-time, spend, and storage controls separately. First
compute the exact evidence fingerprint:

```bash
python3 scripts/train_math500.py inspect-launch-evidence \
  --preregistration outputs/experiment/math500-preregistration.json \
  --training-run-dir outputs/training/math500-cat \
  --one-step-dir outputs/training/math500-cat-one-step \
  --resume-three-step-dir outputs/training/math500-cat-resume-three-step \
  --full-shape-five-step-dir outputs/training/math500-cat-full-shape-five-step
```

Paste that value into `reviewed_evidence_fingerprint`, review exactly those
artifacts, and bind the attestation, qualification receipts, logs, rollout records,
and checkpoints into one approval. Keep the approval output outside every bound
training, qualification, and initial-raw run directory, and do not reuse a bound
source-config path:

```bash
python3 scripts/train_math500.py write-launch-approval \
  --output outputs/experiment/math500-launch-approval.json \
  --preregistration outputs/experiment/math500-preregistration.json \
  --training-run-dir outputs/training/math500-cat \
  --one-step-dir outputs/training/math500-cat-one-step \
  --resume-three-step-dir outputs/training/math500-cat-resume-three-step \
  --full-shape-five-step-dir outputs/training/math500-cat-full-shape-five-step \
  --manual-attestation outputs/experiment/math500-launch-attestation.json

python3 scripts/train_math500.py inspect-launch-approval \
  --launch-approval outputs/experiment/math500-launch-approval.json \
  --preregistration outputs/experiment/math500-preregistration.json \
  --training-run-dir outputs/training/math500-cat
```

This gate verifies evidence integrity; it cannot independently prove operator
attestations about endpoint weights, hardware separation, or reviewed metrics. It
refuses evidence from a qualification directory with a live launch lock.

Canonical launch is a safe preview by default:

```bash
python3 scripts/train_math500.py launch \
  --config configs/training/math500_cat_grpo.toml \
  --run-dir outputs/training/math500-cat
```

It prints the command without executing it. Only the explicit form starts the GPU
job. The explicit form reruns every complete operational gate and refuses a dirty or
wrong `verl` checkout, a changed checkpoint/tokenizer, an incompatible topology,
failed Hydra composition, an overlong prompt, or a failed frozen-anchor canary:

```bash
python3 scripts/train_math500.py launch \
  --config configs/training/math500_cat_grpo.toml \
  --run-dir outputs/training/math500-cat \
  --preregistration outputs/experiment/math500-preregistration.json \
  --launch-approval outputs/experiment/math500-launch-approval.json \
  --execute
```

One run-directory lock spans the final preflight receipt, trainer process group, and
terminal-checkpoint verification. Cleanup terminates same-group workers before that
lock is released.

## Resume and fixed checkpoint

`checkpointing.resume_mode = "auto"` and the planned `trainer.default_local_dir`
make relaunching the same resolved config and run directory resume from the latest
valid `verl` checkpoint. Do not re-plan after checkpoints or other descendants
exist. Inspect the plan, restart the frozen anchor from the same initial snapshot,
then rerun `launch --execute` with the same preregistration and launch-approval
arguments. The approval is reverified against its current evidence on every start.

The protocol does not use labels for early stopping or checkpoint selection. A run
is complete only when `latest_checkpointed_iteration.txt` records step 1,000.
Preview the exact FSDP-to-Hugging-Face export, then use the guarded command to run
and register it as one operation:

```bash
python3 scripts/train_math500.py export-register \
  --config configs/training/math500_cat_grpo.toml \
  --run-dir outputs/training/math500-cat \
  --export-dir outputs/exports/qwen3-4b-math500-cat-step-1000
python3 scripts/train_math500.py export-register \
  --config configs/training/math500_cat_grpo.toml \
  --run-dir outputs/training/math500-cat \
  --export-dir outputs/exports/qwen3-4b-math500-cat-step-1000 \
  --execute
```

The export parent must already be a canonical private directory and the exact target
must not exist. `export-register --execute` runs the pinned merger without a shell,
uses a secret-free allowlisted environment, detects actor mutation, publishes a
fingerprinted merge receipt, and immediately registers it. The recovery-only
`register-checkpoint` command refuses exports without that verified receipt.

## Evaluate the trained export

Create a raw config bound to the registered `pi_T` export hash and a synthesis
config bound to the frozen initial-policy (`pi_0`) identity:

```bash
python3 scripts/train_math500.py plan-trained-eval \
  --run-dir outputs/training/math500-cat \
  --output-dir outputs/training/math500-cat/eval-configs \
  --served-model math500-cat-final
```

For the reproducible server path, do not launch vLLM manually. Use the guarded
`handoff`, host-only `trained-policy`, label-free `trained-eval-generation`, offline
`trained-eval-scoring`, and `finalize` phases in the
[server runbook](server.md#canonical-run-and-trained-evaluation). They revalidate
the live receipt before service startup, keep trained `pi_T` raw generation distinct
from frozen `pi_0` synthesis, and keep labels out of every GPU/model-bearing service.
The lower-level commands
below remain useful for model-free local development of evaluation artifacts. Use
the generated raw config to plan eight trained-policy samples per problem:

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
validated per-task results. The runner binds the run to a hash of the normalized
chat-completions URL without storing the URL or API key, so switching endpoints
fails resume validation. Once raw execution is complete, plan synthesis, execute it
against the frozen `pi_0` anchor, and enter the separate labels-only scoring
boundary:

```bash
python3 scripts/evaluate_math500.py plan-synthesis \
  --config outputs/training/math500-cat/eval-configs/math500_trained_synthesis.toml \
  --raw-run-dir outputs/evals/math500-cat-final-raw \
  --run-dir outputs/evals/math500-cat-final-synthesis

python3 scripts/evaluate_math500.py run-openai \
  --run-dir outputs/evals/math500-cat-final-synthesis \
  --base-url http://127.0.0.1:8001/v1 \
  --api-key-env CAT_ANCHOR_API_KEY \
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
runs measure the trained policy's raw performance and the additional gain, if any,
from frozen-`pi_0` synthesis over `pi_T` rollouts.

After the fixed checkpoint is registered and all four evaluation plans exist, join
the complete `pi_0 -> training -> pi_T` lineage and verify it against the live stage
artifacts:

```bash
python3 scripts/train_math500.py finalize-experiment \
  --output outputs/experiment/math500-registry.json \
  --preregistration outputs/experiment/math500-preregistration.json \
  --initial-raw-run-dir outputs/evals/math500-initial-raw \
  --initial-synthesis-run-dir outputs/evals/math500-initial-synthesis \
  --training-run-dir outputs/training/math500-cat \
  --trained-raw-run-dir outputs/evals/math500-cat-final-raw \
  --trained-synthesis-run-dir outputs/evals/math500-cat-final-synthesis

python3 scripts/train_math500.py verify-experiment \
  --registry outputs/experiment/math500-registry.json \
  --preregistration outputs/experiment/math500-preregistration.json \
  --initial-raw-run-dir outputs/evals/math500-initial-raw \
  --initial-synthesis-run-dir outputs/evals/math500-initial-synthesis \
  --training-run-dir outputs/training/math500-cat \
  --trained-raw-run-dir outputs/evals/math500-cat-final-raw \
  --trained-synthesis-run-dir outputs/evals/math500-cat-final-synthesis
```

The registry content-addresses plans and checkpoint lineage, not generated answers
or scores. Archive verified execution and scoring artifacts separately in the run
record. Registry outputs must remain outside all bound stage directories and the
registered checkpoint export; trained-evaluation handoff files likewise cannot be
written into either registered checkpoint tree.

## Label firewall and limitations

Training reads only `data/math500/questions.jsonl`. The generated `verl` records
contain the rendered prompt, an opaque question ID, and a `ground_truth: null`
placeholder required by the pinned batch-reward API. The reward adapter rejects any
non-null ground truth. `labels.jsonl`, answers, and reference solutions are excluded
from reward, prompt selection, early stopping, and checkpoint selection; scoring is
the first labels-only boundary.

The paper reports eight H100s but does not publish the anchor placement or whether
that count includes it. This implementation uses an external frozen-anchor endpoint,
so every run must separately record and budget its placement and contention policy;
using additional anchor hardware is an accounting and runtime deviation.
Checkpoint registration remains non-reportable because the external anchor service
and distributed runtime are not content-attested. `run-openai` is also marked
non-reportable because an HTTP endpoint cannot prove it serves the registered
checkpoint. Supplying the external evidence required for launch does not change
those hard-coded reportability fields; scientific reportability requires a new
versioned execution backend and schema. No training or real-model evaluation has
been run as part of building this planning and contract infrastructure. The staged
gates, observability record, seed policy, and token/cost ceilings are in the
[experiment plan](experiment_plan.md).
