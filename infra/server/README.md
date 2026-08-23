# Trainer image

This directory is a fail-closed deployment template. It does not contain or fetch
model weights or experiment data. The image accepts those only through read-only
runtime mounts; outputs use a separate writable mount.

The host requires Docker Engine client 24.0.0 or newer, Docker Compose 2.30.0 or
newer, Docker Buildx 0.14.0 or newer, and a working NVIDIA Container Toolkit. The
host validator checks the three CLI versions before launch; the container doctor
then checks actual GPU exposure.

1. Resolve and review a CUDA 12.6/cuDNN 9.8 base with the package versions in
   `runtime-constraints.txt`. Record its immutable registry digest. The pinned
   upstream app Dockerfile uses Python 3.10 and Torch 2.7.0, while its adjacent
   README says Torch 2.7.1; this template follows the executable Dockerfile and
   requires the operator to verify the resolved image. A suggested starting tag is
   listed in `base-image.env.example`, but that mutable tag is not evidence.
   The derived image installs the Python 3.10 `tomli` compatibility package, W&B
   SDK, and evaluation stack from checked-in wheel hashes, without resolving
   dependencies. `pip check` and import checks fail the build on a base conflict.
2. From a clean clone, build with BuildKit:

   ```bash
   python3 infra/server/build_image.py \
     --base-image 'reviewed/image:tag@sha256:<real digest>' \
     --tag registry.example/compute-as-a-teacher:paper-v1 \
     --metadata-out outputs/image/server-image-build.json \
     --accept-reviewed-base --push
   ```

   Build metadata records the base reference/digest, repository and Verl revisions,
   image ID, registry digest, and BuildKit metadata. Use the receipt's
   `registry_digest` to set both `CAT_TRAINER_IMAGE=tag@digest` and
   `CAT_TRAINER_IMAGE_DIGEST`. The build requests provenance and SBOM attestations,
   which the registry preserves. `--load` is only for local inspection; a loaded
   tag is not an immutable server deployment unless transferred and independently
   verified. The image contains the complete Python inventory and its SHA-256.
   Read that locked value without GPUs or model mounts and place it in the resolved
   training config's `runtime.package_inventory_sha256`:

   ```bash
   docker run --rm --entrypoint python 'registry.example/image@sha256:<receipt digest>' -c \
     'import json; print(json.load(open("/opt/cat/image-metadata/python-packages.json"))["sha256"])'
   ```
3. Prepare and fully verify MATH-500 on the host before starting a service, place
   the reviewed model snapshot on the host,
   create a config directory containing the resolved training, evaluation, scoring,
   and server-workflow files, and set the required `CAT_*_HOST_*` values. Inside the
   workflow, use absolute container paths under `/mnt/config`, `/mnt/data`, and
   `/mnt/outputs`.
   Materialize a standard Hugging Face cache snapshot before mounting it: the
   `snapshots/<revision>` tree commonly links into a sibling `blobs/` directory,
   which is outside this mount. For example, copy an already-present snapshot with
   symlink dereferencing (`cp --archive --dereference SOURCE/. DESTINATION/`); this
   performs no download. `DESTINATION` must be named with the full revision, and the
   host validator recursively rejects every symlink, including broken links. Keep
   the resolved model, data, config, and output roots pairwise distinct and
   non-nested.
   Set `CAT_TRAINER_GPU_DEVICE_0` through `CAT_TRAINER_GPU_DEVICE_7` to eight
   distinct Docker/NVIDIA device IDs; reserve any separate local anchor GPU instead
   of exposing it to the trainer. Compose requests only those exact devices, and the
   trainer sees them as logical CUDA devices `0` through `7`.
   Set `CAT_MODEL_REVISION` to the model's full 40-character commit; the snapshot is
   mounted at `/mnt/models/$CAT_MODEL_REVISION`, matching runtime identity checks.
   `CAT_DATA_HOST_DIR` must be a directory named `data`; the host validator checks
   the raw snapshot, questions, and labels against the lock. Compose never mounts
   that directory wholesale: `trainer` and `evaluator` receive only
   `math500/questions.jsonl`; the offline, CPU-only `scorer` receives the exact
   questions and labels files but no raw snapshot, model, GPU, service/W&B secret,
   or network.
   Ensure UID/GID `10001:10001` can traverse and read the model, data, and config
   roots and can create files under the output root. Inputs should remain
   non-writable. The container doctor below runs as that exact UID and opens every
   required input plus each weight shard without loading weights. Treat that doctor
   as the authoritative UID/mount-access probe and do not start a service if it
   fails.
   Keep the W&B API key in the server's secret manager; this repository contains no
   key and the Compose file does not persist one.
   With `CAT_ANCHOR_MODE=local` and a ninth GPU, set its exact device ID in
   `CAT_ANCHOR_GPU_DEVICE` and use `http://anchor:8001/v1` in the resolved training
   config. Start it only through the guarded host `phase anchor` sequence in the
   [server runbook](../../docs/server.md#host-preparation-and-anchor); that path
   revalidates the host, clean checkout, device separation, and service health.
   With `CAT_ANCHOR_MODE=remote`, leave `CAT_ANCHOR_GPU_DEVICE` unset, configure the
   remote URL and injected key, and do not run the local anchor Compose command.
   The local server registers both `Qwen/Qwen3-4B` for baseline evaluation and
   `cat-frozen-qwen3-4b` for the frozen training anchor against the same snapshot.
   The local anchor publishes no host port and uses a documented non-secret bearer
   marker because it has no API authentication. Inject a real key at runtime for a
   remote authenticated anchor. The trainer never mounts the host Docker socket;
   Compose lifecycle commands are run on the host. Both trainer and local anchor
   use host IPC so the pinned CUDA/vLLM stack is not constrained by Docker's small
   default shared-memory allocation; use this template only on a dedicated,
   reviewed training host.
   Copy `server.env.example` to the ignored `.env.server`, replace every marker,
   then load it explicitly before validation: `set -a; . ./.env.server; set +a`.
   The template contains no API keys; inject W&B and remote-service credentials
   from the server secret manager into `CAT_ANCHOR_API_KEY` and `WANDB_API_KEY`.
   Do not persist rendered Compose output. If a resolved config names a different
   API-key environment variable, add `-e NAME` to every corresponding `compose run`.
4. Validate the host contract, probe the label-free trainer mounts as the runtime
   UID, start the local anchor when selected, and create the model-free plans with
   the label-free evaluator:

   ```bash
   python3 infra/server/validate_server_env.py
   docker compose -f infra/server/compose.yaml run --rm trainer doctor
   python3 scripts/server_math500.py \
     --workflow "$CAT_CONFIG_HOST_DIR/workflow.toml" phase anchor
   python3 scripts/server_math500.py \
     --workflow "$CAT_CONFIG_HOST_DIR/workflow.toml" phase anchor --execute
   docker compose -f infra/server/compose.yaml run --rm evaluator \
     workflow --workflow /mnt/config/workflow.toml phase prepare --execute
   ```

5. Run every read-only/no-download readiness gate with one command:

   ```bash
   docker compose -f infra/server/compose.yaml run --rm trainer ready
   ```

   This combines the container doctor with repository, config, questions, model identity,
   GPU runtime, tokenizer, anchor canary, W&B, storage, and training preflight checks.

After the guarded handoff registers the fixed final export, stop the training job
and reserve one explicit evaluation GPU. It may reuse a trainer GPU but must differ
from the local frozen-anchor GPU. Start the internal-only trained policy through the
host workflow phase so registration and receipt are revalidated inside a trainer
container immediately before Compose startup:

```bash
python3 scripts/server_math500.py \
  --workflow "$CAT_CONFIG_HOST_DIR/workflow.toml" phase trained-policy
python3 scripts/server_math500.py \
  --workflow "$CAT_CONFIG_HOST_DIR/workflow.toml" phase trained-policy --execute
```

The service exposes no host port. Inside the Compose network it serves
`math500-cat-final` at `http://trained-policy:8002/v1` from the exact canonical
export path. Only that exact export subtree is mounted at `/mnt/trained-model`,
read-only, and the tokenizer comes from the
pinned base-model mount instead of being copied into the export. The evaluator receives
the separate `CAT_TRAINED_POLICY_API_KEY` environment value; the local unauthenticated
service uses the non-secret default marker. Never run the trained policy concurrently
with training when it reuses a trainer GPU.

The doctor requires eight BF16 H100 GPUs, immutable image identity, exact core
package versions, the pinned Verl checkout, read-only model/questions mounts, and a
writable output mount. It inspects model filenames but never imports or loads model
weights. Run the repository's existing preflight and qualification gates after the
doctor passes; a passing doctor alone does not authorize the canonical experiment.
The image maps the repository's default `data/` and `outputs/` paths to their
runtime mounts, so unchanged relative dataset paths cannot fall back to baked data.
Run baseline and trained generation through `evaluator`; only after the guarded
terminal handoff and all generation are complete, run both scoring phases and
`finalize` through `scorer`. Service-role checks reject phase execution in the wrong
container.
