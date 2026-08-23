# Server runbook

This is the operator sequence for the containerized MATH-500 experiment. The image
and workflow are fail-closed templates, not a prebuilt training environment. Resolve
every `required_*` value, image digest, model/tokenizer identity, package-inventory
hash, config path, mount, and provider resource limit before allocating GPUs. No
model, GPU, anchor, evaluation, training, or W&B run was performed while creating
this infrastructure.

Use [the image guide](../infra/server/README.md) to review and build the immutable
trainer image. Prepare and fully verify the locked MATH-500 files on the host before
starting any service. Containers never prepare the source snapshot: label-free
checks use `--verify-questions-only`. Put
these resolved files in `CAT_CONFIG_HOST_DIR`, mounted read-only at `/mnt/config`:

- `workflow.toml`, based on `configs/server/math500_server.example.toml`;
- `training.toml`, `raw.toml`, `synthesis.toml`, and `scoring.toml`.

The workflow uses `/mnt/config`, `/mnt/data`, and `/mnt/outputs`. Its initial raw and
synthesis stages both call the same frozen `pi_0` service at
`http://anchor:8001/v1`; the trained `pi_T` service is introduced only after the
fixed checkpoint is exported. The resolved training config must likewise use
`http://anchor:8001/v1` for `runtime.anchor_base_url` in local-anchor mode.

Materialize the model snapshot as a self-contained directory whose basename is its
full 40-character revision. Standard Hugging Face snapshot symlinks often point to
a sibling cache directory and break when only the snapshot is mounted; copy or
materialize their targets inside this directory before hashing it. Keep the model,
data, output, and config host directories pairwise distinct and non-nested, and do
not route them through aliases to the same filesystem tree.

The host validator opens and cryptographically verifies the raw snapshot, questions,
and labels. At runtime, `trainer` and `evaluator` receive only the exact questions
file. `scorer` receives the exact questions and labels files, but no raw snapshot,
GPU, model mount, endpoint credential, W&B credential, or network. Run every
generation before starting the scorer so label-derived score files cannot exist
during training or generation. The role-locked image entrypoint exposes no arbitrary
execution route, and both trainer preflight and launch reject any stale scoring or
final-registry artifact anywhere under the output mount. After scoring, start a new
output root for any new training or generation run.

## Host preparation and anchor

Set the host mounts, immutable trainer image/digest, full model revision, eight
trainer GPU IDs, and one disjoint anchor GPU ID as described in the image guide.
Use Docker Engine client 24.0.0+, Compose 2.30.0+, and Buildx 0.14.0+; the host
validator checks these versions without contacting the Docker daemon.
Resolve the five configuration files and validate the complete host contract before
starting a service. From the host checkout, create and activate the locked repository
environment first; this supplies TOML support on Python 3.10:

```bash
uv sync --extra evaluation --frozen
source .venv/bin/activate
export CAT_DATA_REPOSITORY_ROOT=/srv/compute-as-a-teacher
export CAT_DATA_HOST_DIR="$CAT_DATA_REPOSITORY_ROOT/data"
python3 scripts/prepare_math500.py --repo-root "$CAT_DATA_REPOSITORY_ROOT"
python3 scripts/prepare_math500.py \
  --repo-root "$CAT_DATA_REPOSITORY_ROOT" --verify-only
export CAT_ANCHOR_MODE=local
export CAT_TRAINER_GPU_DEVICE_0=0
export CAT_TRAINER_GPU_DEVICE_1=1
export CAT_TRAINER_GPU_DEVICE_2=2
export CAT_TRAINER_GPU_DEVICE_3=3
export CAT_TRAINER_GPU_DEVICE_4=4
export CAT_TRAINER_GPU_DEVICE_5=5
export CAT_TRAINER_GPU_DEVICE_6=6
export CAT_TRAINER_GPU_DEVICE_7=7
export CAT_ANCHOR_GPU_DEVICE=8
python3 infra/server/validate_server_env.py
```

Run the explicit container doctor before starting either anchor topology:

```bash
docker compose -f infra/server/compose.yaml run --rm trainer doctor
```

Preview and then start the local frozen anchor from the host checkout:

```bash
python3 scripts/server_math500.py \
  --workflow "$CAT_CONFIG_HOST_DIR/workflow.toml" phase anchor
python3 scripts/server_math500.py \
  --workflow "$CAT_CONFIG_HOST_DIR/workflow.toml" phase anchor --execute
```

This delegates only the exact `docker compose --profile local-anchor up -d --wait
--wait-timeout 600 anchor` argv after rerunning the host validator and requiring a
clean Git checkout with no untracked files. It returns only after the health check passes. Secrets
are inherited from the environment, never placed in the command or artifacts. Do
not run this phase inside the trainer and do not mount the Docker socket there.

For a remote isolated anchor, first resolve both `pi_0` URLs and the injected key,
set `CAT_ANCHOR_MODE=remote`, unset `CAT_ANCHOR_GPU_DEVICE`, then run the
same host validator and doctor. Provision and health-check that endpoint with its
external lifecycle before continuing. Do not execute the local Compose phase;
previewing `phase anchor` returns an explicit external-lifecycle result with no
command.

For online W&B, have the server secret manager inject the configured
`tracking.wandb.api_key_env` into the Compose client process. The default
`WANDB_API_KEY` and `CAT_ANCHOR_API_KEY` are forwarded at container creation and
never written to commands or artifacts. If a resolved config uses a different
variable name, add `-e NAME` to each corresponding `compose run`; never put its
value in TOML, an argv, the image, or a committed environment file.
When W&B is enabled, use `tracking.wandb.mode = "online"`; W&B 0.21.1 ignores
offline resume semantics and is not accepted for the resume qualification.
Keep resolved host configuration outside the Git checkout: host service phases
reject both tracked changes and untracked files before executing Compose.

Now use the label-free evaluator to verify only the mounted questions and write the
locked training and initial-raw plans:

```bash
docker compose -f infra/server/compose.yaml run --rm evaluator \
  workflow --workflow /mnt/config/workflow.toml phase prepare
docker compose -f infra/server/compose.yaml run --rm evaluator \
  workflow --workflow /mnt/config/workflow.toml phase prepare --execute
```

Now run the one-command no-download readiness gate:

```bash
docker compose -f infra/server/compose.yaml run --rm trainer ready
```

`ready` reruns the doctor and combines it with experiment readiness. It verifies the
copied source-tree receipt, locked questions and resolved configs, storage, local model
hash, pinned Verl/runtime identities, exactly eight available BF16 H100s,
tokenizer/Hydra composition, W&B readiness, and both anchor canaries. It reads but
does not load model weights and launches no model. A failed check exits with status
2.

## Registered baseline

Preregister before generating any baseline result:

```bash
docker compose -f infra/server/compose.yaml run --rm evaluator \
  workflow --workflow /mnt/config/workflow.toml phase preregister
docker compose -f infra/server/compose.yaml run --rm evaluator \
  workflow --workflow /mnt/config/workflow.toml phase preregister --execute
```

Then preview and execute baseline generation. It runs 16 raw requests first, reruns
the same append-only plan to prove resume, completes all 4,000 raw generations, and
creates and executes the 500 synthesis requests. Do not score yet: baseline scoring
is locked until the fixed final checkpoint and guarded handoff exist.

```bash
docker compose -f infra/server/compose.yaml run --rm evaluator \
  workflow --workflow /mnt/config/workflow.toml phase baseline-generation
docker compose -f infra/server/compose.yaml run --rm evaluator \
  workflow --workflow /mnt/config/workflow.toml phase baseline-generation --execute
```

## Qualifications and approval

Preview and execute `one_step` first:

```bash
docker compose -f infra/server/compose.yaml run --rm trainer \
  workflow --workflow /mnt/config/workflow.toml phase qualification --profile one_step
docker compose -f infra/server/compose.yaml run --rm trainer \
  workflow --workflow /mnt/config/workflow.toml phase qualification --profile one_step --execute
```

The resume qualification cannot be satisfied by one uninterrupted three-step job.
Preview it and prepare its plan:

```bash
docker compose -f infra/server/compose.yaml run --rm trainer \
  workflow --workflow /mnt/config/workflow.toml phase qualification \
  --profile resume_three_step
docker compose -f infra/server/compose.yaml run --rm trainer \
  workflow --workflow /mnt/config/workflow.toml phase qualification \
  --profile resume_three_step --resume-action prepare --execute
```

Run its initial leg under a host scheduler that sends SIGTERM only after the
complete step-1 checkpoint is visible and validated. The workflow replaces itself
with the existing qualification CLI so termination reaches its process-group
cleanup directly:

```bash
docker compose -f infra/server/compose.yaml run --rm trainer \
  workflow --workflow /mnt/config/workflow.toml phase qualification \
  --profile resume_three_step --resume-action initial --execute
```

After the intentional interruption, the guarded restart refuses to run unless it
finds exactly that valid step-1 checkpoint:

```bash
docker compose -f infra/server/compose.yaml run --rm trainer \
  workflow --workflow /mnt/config/workflow.toml phase qualification \
  --profile resume_three_step --resume-action restart --execute
```

Finally run `full_shape_five_step` with the same preview-then-execute pattern:

```bash
docker compose -f infra/server/compose.yaml run --rm trainer \
  workflow --workflow /mnt/config/workflow.toml phase qualification \
  --profile full_shape_five_step
docker compose -f infra/server/compose.yaml run --rm trainer \
  workflow --workflow /mnt/config/workflow.toml phase qualification \
  --profile full_shape_five_step --execute
```

Review the three non-reportable runs, W&B identity/continuity if enabled, metrics,
anchor behavior, checkpoint/resume evidence, and projected time, storage, GPU-hours,
and cost. Configure those hard ceilings in the external scheduler/provider. Fill the
manual attestation at `/mnt/outputs/registry/manual-attestation.json` with the
inspected evidence fingerprint, then preview and write the content-addressed
approval:

```bash
docker compose -f infra/server/compose.yaml run --rm trainer \
  workflow --workflow /mnt/config/workflow.toml phase approval
docker compose -f infra/server/compose.yaml run --rm trainer \
  workflow --workflow /mnt/config/workflow.toml phase approval --execute
```

## Canonical run and trained evaluation

Preview the canonical command. Execute it only after checking that the scheduler
ceilings are active; the underlying CLI revalidates preregistration, approval, full
preflight, model/runtime identity, and anchor canaries on every start or resume:

```bash
docker compose -f infra/server/compose.yaml run --rm trainer \
  workflow --workflow /mnt/config/workflow.toml phase canonical
docker compose -f infra/server/compose.yaml run --rm trainer \
  workflow --workflow /mnt/config/workflow.toml phase canonical --execute
```

After the fixed step-1,000 checkpoint, preview and execute the guarded handoff:

```bash
docker compose -f infra/server/compose.yaml run --rm trainer \
  workflow --workflow /mnt/config/workflow.toml phase handoff
docker compose -f infra/server/compose.yaml run --rm trainer \
  workflow --workflow /mnt/config/workflow.toml phase handoff --execute
```

The `prepare` phase created only the private `/mnt/outputs/exports` parent; the
canonical export target remained nonexistent. Handoff runs the exact pinned Verl
merger without a shell, records a content-addressed receipt binding the plan,
runtime, base model, actor, export, log, argv, and allowlisted environment, registers
that receipt, and writes the trained-evaluation configs. A manual merge or arbitrary
Hugging Face directory cannot be registered.

Stop training before reusing any trainer GPU. On the host, assign one evaluation GPU
that differs from a local anchor GPU, then preview and start the separate `pi_T`
service. The phase validates host topology, revalidates the live registered export
inside the trainer container, and waits for service health:

```bash
export CAT_TRAINED_POLICY_GPU_DEVICE=0
python3 scripts/server_math500.py \
  --workflow "$CAT_CONFIG_HOST_DIR/workflow.toml" phase trained-policy
python3 scripts/server_math500.py \
  --workflow "$CAT_CONFIG_HOST_DIR/workflow.toml" phase trained-policy --execute
```

The trained service exposes no host port and mounts only the exact export read-only
at `/mnt/trained-model`; the frozen `pi_0` anchor stays live at its separate URL.
Inject `CAT_TRAINED_POLICY_API_KEY` and `CAT_ANCHOR_API_KEY` into the evaluator, then
preview and execute trained generation. It performs a 16-request `pi_T` raw canary,
resumes the same raw plan to completion, and sends synthesis only to frozen `pi_0`:

```bash
docker compose -f infra/server/compose.yaml run --rm evaluator \
  workflow --workflow /mnt/config/workflow.toml phase trained-eval-generation
docker compose -f infra/server/compose.yaml run --rm evaluator \
  workflow --workflow /mnt/config/workflow.toml phase trained-eval-generation --execute
```

All training and generation is now complete. Start the offline, CPU-only scorer to
cross the labels boundary. Baseline scoring itself revalidates the terminal guarded
handoff, so it cannot publish label-derived artifacts while training is still
possible. Score baseline first, then the trained evaluation:

```bash
docker compose -f infra/server/compose.yaml run --rm scorer \
  workflow --workflow /mnt/config/workflow.toml phase baseline-scoring
docker compose -f infra/server/compose.yaml run --rm scorer \
  workflow --workflow /mnt/config/workflow.toml phase baseline-scoring --execute
docker compose -f infra/server/compose.yaml run --rm scorer \
  workflow --workflow /mnt/config/workflow.toml phase trained-eval-scoring
docker compose -f infra/server/compose.yaml run --rm scorer \
  workflow --workflow /mnt/config/workflow.toml phase trained-eval-scoring --execute
```

Finally use the same scorer to publish and independently verify the registry only
after all four scored plans and their lineage validate:

```bash
docker compose -f infra/server/compose.yaml run --rm scorer \
  workflow --workflow /mnt/config/workflow.toml phase finalize
docker compose -f infra/server/compose.yaml run --rm scorer \
  workflow --workflow /mnt/config/workflow.toml phase finalize --execute
```

These commands are an exact handoff, not evidence of execution: placeholders still
must be resolved, and no GPU, model, W&B, anchor, evaluation, or training run was
performed while building this repository.
