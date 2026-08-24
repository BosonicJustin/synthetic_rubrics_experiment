# One-H100 RunPod baseline pilot

This is the supported pilot path for frozen `Qwen/Qwen3-4B` raw generation followed
by frozen-model synthesis on one H100. It is generation-only: the RunPod receives
questions, never labels, and this runbook does not score results or use W&B.

The OpenAI-compatible execution backend deliberately marks every artifact
`reportable: false`. This pilot checks the endpoint, prompts, sampling payload,
append-only resume, context capacity, runtime cost, and raw-to-synthesis lineage. It
is not the registered multi-GPU RL experiment or a paper result.

## Frozen contract

- repository release: an immutable reviewed tag or 40-character commit containing
  the `run-baseline` command; resolve and archive its commit before generation
- raw config: `configs/evals/math500_raw.toml`
- synthesis config: `configs/evals/math500_synthesis.toml`
- model and tokenizer: `Qwen/Qwen3-4B`
- model and tokenizer revision: `1cfa9a7208912126459214e8b04321603b3df60c`
- tokenizer chat-template SHA-256:
  `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8`
- serving runtime: Python `3.10.20`, vLLM `0.9.1`, BF16, one H100
- API model name: `Qwen/Qwen3-4B` only
- API port: `${CAT_VLLM_PORT:-18001}`; RunPod nginx commonly occupies `8001`

The resolved configs were first committed at
`134bd901405d81526590092c9fefc931dda59882`. That is observed pilot provenance, not
the release to check out because `run-baseline` was added afterward. The config
hashes that the release must retain are:

```text
c78c97298bba126768f6a85cb2eb9200d0be99d389b31de65515739bc8723cda  configs/evals/math500_raw.toml
6137d10ac1f74c9a2f7677a70d20f95009721113e93e9322314cd1d2b97e0120  configs/evals/math500_synthesis.toml
```

Keep the repository, model, results, logs, and package caches under persistent
`/workspace`. Put Python and virtual environments on the pod-local filesystem:

```text
/opt/cat-python-install    ephemeral uv-managed Python
/opt/cat-eval-venv        ephemeral evaluator environment
/opt/cat-serve-venv       ephemeral vLLM environment
/workspace/models         persistent model snapshots
/workspace/cat-results    persistent plans, generations, logs, and receipts
/workspace/cache          persistent uv, Hugging Face, and kernel caches
```

Do not put `.venv*` in the repository on the network volume. Imports from there can
take minutes. Recreate the `/opt` environments after a pod restart; the persistent uv
cache makes subsequent installs much faster.

## Checkout and question-only input

On the pod:

```bash
export CAT_RELEASE_REF="${CAT_RELEASE_REF:?set the reviewed release tag or 40-character commit}"
export CAT_REPO=/workspace/synthetic_rubrics_experiment
export CAT_MODEL_REVISION=1cfa9a7208912126459214e8b04321603b3df60c
export CAT_MODEL_DIR=/workspace/models/$CAT_MODEL_REVISION
export CAT_VLLM_PORT="${CAT_VLLM_PORT:-18001}"
export CAT_BASE_URL=http://127.0.0.1:$CAT_VLLM_PORT/v1
export UV_CACHE_DIR=/workspace/cache/uv
export CUDA_VISIBLE_DEVICES=0

git clone git@github.com:BosonicJustin/synthetic_rubrics_experiment.git "$CAT_REPO"
cd "$CAT_REPO"
git checkout --detach "$CAT_RELEASE_REF"
export CAT_SOURCE_REVISION="$(git rev-parse HEAD)"
test "${#CAT_SOURCE_REVISION}" = 40
test -z "$(git status --porcelain)"

export CAT_RUN_ROOT=/workspace/cat-results/math500-baseline-$CAT_SOURCE_REVISION
export CAT_RAW_RUN=$CAT_RUN_ROOT/raw
export CAT_SYNTHESIS_RUN=$CAT_RUN_ROOT/synthesis
mkdir -p data/math500 "$CAT_RUN_ROOT/runtime" /workspace/cache

printf '%s\n' "$CAT_SOURCE_REVISION" > "$CAT_RUN_ROOT/runtime/source-revision.txt"
uv --version > "$CAT_RUN_ROOT/runtime/uv-version.txt"
git archive --format=tar "$CAT_SOURCE_REVISION" \
  > "$CAT_RUN_ROOT/runtime/source-tree.tar"
sha256sum "$CAT_RUN_ROOT/runtime/source-tree.tar" \
  > "$CAT_RUN_ROOT/runtime/source-tree.tar.sha256"

printf '%s  %s\n' \
  c78c97298bba126768f6a85cb2eb9200d0be99d389b31de65515739bc8723cda configs/evals/math500_raw.toml \
  6137d10ac1f74c9a2f7677a70d20f95009721113e93e9322314cd1d2b97e0120 configs/evals/math500_synthesis.toml \
  | sha256sum --check --strict
```

Prepare and verify MATH-500 on a trusted machine, then transfer only
`data/math500/questions.jsonl` to the same relative path on the pod. Do not transfer
`labels.jsonl` or `data/raw/math500-test.jsonl`, and do not run the full dataset
preparer on the GPU host. The required questions file has 500 rows, 124,398 bytes,
and SHA-256
`f0a73e8f8f397a10dc306cc8bf5fe97b8ed47c31d1b73b6c653491e2b79be7e4`.

## Fast local environments

Use the persistent uv cache but copy installed packages into `/opt`:

```bash
uv python install 3.10.20 \
  --install-dir /opt/cat-python-install \
  --no-bin

export CAT_PYTHON=/opt/cat-python-install/cpython-3.10.20-linux-x86_64-gnu/bin/python3.10
test "$($CAT_PYTHON -c 'import platform; print(platform.python_version())')" = 3.10.20

uv venv --python "$CAT_PYTHON" --link-mode copy /opt/cat-eval-venv
UV_PROJECT_ENVIRONMENT=/opt/cat-eval-venv uv sync \
  --python "$CAT_PYTHON" \
  --extra evaluation \
  --frozen \
  --no-dev \
  --link-mode copy

uv venv --python "$CAT_PYTHON" --link-mode copy /opt/cat-serve-venv
uv pip install \
  --python /opt/cat-serve-venv/bin/python \
  --link-mode copy \
  --exclude-newer '2025-06-11T00:00:00Z' \
  'vllm==0.9.1'

uv pip check --python /opt/cat-eval-venv/bin/python
uv pip check --python /opt/cat-serve-venv/bin/python
uv pip freeze --python /opt/cat-eval-venv/bin/python \
  > "$CAT_RUN_ROOT/runtime/eval-freeze.txt"
uv pip freeze --python /opt/cat-serve-venv/bin/python \
  > "$CAT_RUN_ROOT/runtime/serve-freeze.txt"

export CAT_EVAL_PYTHON=/opt/cat-eval-venv/bin/python
export CAT_SERVE_PYTHON=/opt/cat-serve-venv/bin/python
```

The first bootstrap may download Python and wheels. After `/workspace/cache/uv` is
warm, rerunning the same block restores the ephemeral environments from that cache;
`UV_OFFLINE=1` can be added on a restart only after confirming the cache is complete.

## Verify data, runtime, and snapshot

```bash
cd "$CAT_REPO"

test ! -e data/math500/labels.jsonl
test ! -e data/raw/math500-test.jsonl
test "$(sha256sum data/math500/questions.jsonl | cut -d' ' -f1)" = \
  f0a73e8f8f397a10dc306cc8bf5fe97b8ed47c31d1b73b6c653491e2b79be7e4
"$CAT_EVAL_PYTHON" scripts/prepare_math500.py --verify-questions-only

"$CAT_SERVE_PYTHON" - <<'PY' | tee "$CAT_RUN_ROOT/runtime/serve-runtime.json"
import importlib.metadata as metadata
import json
import platform
import torch

value = {
    "python": platform.python_version(),
    "vllm": metadata.version("vllm"),
    "torch": metadata.version("torch"),
    "transformers": metadata.version("transformers"),
    "torch_cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "visible_gpus": torch.cuda.device_count(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
}
assert value["python"] == "3.10.20"
assert value["vllm"] == "0.9.1"
assert value["visible_gpus"] == 1
assert value["gpu"] and "H100" in value["gpu"]
assert value["bf16"] is True
print(json.dumps(value, sort_keys=True))
PY
```

Populate `$CAT_MODEL_DIR` while network access is allowed by transferring a reviewed
snapshot or downloading the exact public revision. This command materializes files
in the persistent destination instead of serving from a mutable Hub cache alias:

```bash
export HF_HOME=/workspace/cache/huggingface
mkdir -p "$HF_HOME" "$CAT_MODEL_DIR"

"$CAT_SERVE_PYTHON" - <<'PY'
import os
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download

source = snapshot_download(
    repo_id="Qwen/Qwen3-4B",
    revision=os.environ["CAT_MODEL_REVISION"],
    cache_dir=os.environ["HF_HOME"],
)
destination = Path(os.environ["CAT_MODEL_DIR"])
if destination.exists() and any(destination.iterdir()):
    raise SystemExit(f"refusing to replace populated snapshot: {destination}")
destination.parent.mkdir(parents=True, exist_ok=True)
shutil.copytree(source, destination, symlinks=False, dirs_exist_ok=True)
PY
```

The directory name must remain the full revision, all files must be materialized
locally, and it must contain no symlinks. Then record its content identity without
loading tensors:

```bash
test "$(basename "$CAT_MODEL_DIR")" = "$CAT_MODEL_REVISION"
test -z "$(find "$CAT_MODEL_DIR" -type l -print -quit)"

"$CAT_EVAL_PYTHON" scripts/train_math500.py model-identity \
  --model-path "$CAT_MODEL_DIR" \
  | tee "$CAT_RUN_ROOT/runtime/model-identity.json"

"$CAT_EVAL_PYTHON" - "$CAT_RUN_ROOT/runtime/model-identity.json" <<'PY'
import json
import os
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
assert value["snapshot_revision"] == os.environ["CAT_MODEL_REVISION"]
assert value["chat_template_sha256"] == (
    "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
)
assert value["model_loaded"] is False
print(value["model_snapshot_tree_sha256"])
PY
```

The tree hash is a receipt for this materialized snapshot. The HTTP evaluation
schema does not content-attest that tree, which is one reason the pilot remains
non-reportable.

## Start vLLM

Keep the endpoint loopback-only. Do not expose it through a RunPod public port. Use
one served name: both checked-in baseline configs require the response model to be
exactly `Qwen/Qwen3-4B`.

First verify the installed 0.9.1 parser recognizes the `vllm` generation-config
sentinel:

```bash
"$CAT_SERVE_PYTHON" -m vllm.entrypoints.openai.api_server --help \
  > "$CAT_RUN_ROOT/runtime/vllm-help.txt"
grep -A6 -- '--generation-config' "$CAT_RUN_ROOT/runtime/vllm-help.txt"
grep -q "'vllm'" "$CAT_RUN_ROOT/runtime/vllm-help.txt"
```

After the snapshot is complete, close Hub access and launch the server. Explicit
`--generation-config vllm` prevents model `generation_config.json` defaults from
changing request fields that the evaluator does not send.

```bash
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN WANDB_API_KEY WANDB_PROJECT WANDB_ENTITY
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/workspace/cache/huggingface
export XDG_CACHE_HOME=/workspace/cache/xdg
export TRITON_CACHE_DIR=/workspace/cache/triton
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR"

if curl -fsS --max-time 2 "http://127.0.0.1:$CAT_VLLM_PORT/health"; then
  echo "refusing to replace an existing service on port $CAT_VLLM_PORT" >&2
  exit 1
fi

nohup "$CAT_SERVE_PYTHON" -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 \
  --port "$CAT_VLLM_PORT" \
  --model "$CAT_MODEL_DIR" \
  --served-model-name Qwen/Qwen3-4B \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 24 \
  --generation-config vllm \
  </dev/null > "$CAT_RUN_ROOT/runtime/vllm.log" 2>&1 &

echo "$!" > "$CAT_RUN_ROOT/runtime/vllm.pid"
```

`nohup` survives an SSH disconnect, but not a pod restart. After a restart, recreate
the `/opt` environments, verify the persistent inputs, and relaunch on the same port.

## Endpoint gate

Wait for `/health`, then verify the single model alias. Do not send an unrelated
probe completion: `run-baseline` first writes the raw plan, then its 16 planned
requests exercise and audit every sampling extension used by the repository.

```bash
for _ in $(seq 1 120); do
  curl -fsS "http://127.0.0.1:$CAT_VLLM_PORT/health" && break
  sleep 5
done
curl -fsS "http://127.0.0.1:$CAT_VLLM_PORT/health"

"$CAT_EVAL_PYTHON" - <<'PY'
import json
import os
import urllib.request

port = os.environ["CAT_VLLM_PORT"]
root = f"http://127.0.0.1:{port}"
models = json.load(urllib.request.urlopen(root + "/v1/models", timeout=10))
assert [item["id"] for item in models["data"]] == ["Qwen/Qwen3-4B"]
print(json.dumps(models, sort_keys=True))
PY
```

Do not reduce `--max-model-len` to address load pressure: worst-case synthesis must
fit eight 1,536-token rollouts plus framing and a 1,536-token completion. Reduce
`--workers` if necessary. Before the full run, also inspect `vllm.log` for downloads,
fallbacks, out-of-memory errors, or an unexpected generation configuration.

## Raw to synthesis generation

The label-free `run-baseline --pilot` workflow explicitly records that this run has
no canonical experiment preregistration. It writes the raw plan, generates and
audits 16 raw canaries, resumes to all 4,000 raw requests, creates synthesis solely
from those frozen outputs, audits 16 synthesis canaries, and completes all 500
synthesis requests. Run it in `tmux`, or redirect it under a supervisor if the SSH
session is not stable:

```bash
set -o pipefail
cd "$CAT_REPO"
"$CAT_EVAL_PYTHON" scripts/evaluate_math500.py run-baseline \
  --pilot \
  --raw-config configs/evals/math500_raw.toml \
  --synthesis-config configs/evals/math500_synthesis.toml \
  --raw-run-dir "$CAT_RAW_RUN" \
  --synthesis-run-dir "$CAT_SYNTHESIS_RUN" \
  --base-url "$CAT_BASE_URL" \
  --timeout-seconds 300 \
  --workers 24 \
  --batch-size 24 \
  --canary-results 16 \
  --synthesis-canary-results 16 \
  2>&1 | tee -a "$CAT_RUN_ROOT/baseline.log"
```

If the evaluator or pod stops, restore the environments and server, then rerun that
exact command with the same commit, configs, model, run directories, base URL, and
port. It verifies the existing contiguous result prefix and resumes pending requests;
do not add `--force`, delete result files, change the endpoint port, or create a new
run directory.

Resume only after an operational interruption. If a canary has the wrong model or
finish reason, or lacks the prompt-required final `\boxed{...}`, stop and preserve
the failure. Do not delete it, repeatedly regenerate it, or select a passing canary;
continuing would require an explicit new protocol decision.

A successful final JSON result reports `registration_mode: "pilot"`, `pilot: true`,
`preregistration_verified: false`, `4,000` raw and `500` synthesis requests, both
stages `reportable: false`, `labels_loaded: false`, and `scored: false`. Preserve the
complete `$CAT_RUN_ROOT`.

## Hard boundary and next step

- Do not upload or mount labels on this pod.
- Do not invoke `score-raw` or `score-synthesis` here.
- Do not install, configure, or log into W&B for this pilot.
- Do not inspect outputs and then tune prompts, seeds, training settings, or checkpoint
  selection if these artifacts will later be treated as preregistered evidence.
- Do not describe HTTP-backend outputs as reportable, even if every integrity check
  passes.
- Do not reuse `--pilot` artifacts as canonical baseline evidence. Canonical mode
  requires the separately prepared training run and experiment preregistration
  arguments and does not permit `--pilot`.

The dependency cutoff and recorded freezes make this pilot inspectable; they do not
replace the digest-pinned image and package inventory required for full runtime
attestation.

The full experiment requires the preregistration, runtime attestation, isolated
frozen anchor, eight trainer H100s, qualification runs, checkpoint handoff, and
separate labels-only scorer documented in the [experiment plan](experiment_plan.md),
[server runbook](server.md), and [training guide](training.md). Protocol rationale
and artifact details are in the [evaluation guide](evaluation.md).
