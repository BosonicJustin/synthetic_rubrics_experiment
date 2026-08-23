# Compute as a Teacher — reproduction

This repository is a reproducible implementation workspace for the experiments in
*Compute as a Teacher*. It covers pinned MATH-500 acquisition, raw and synthesis
evaluation, and a reproducible RL-synthesis training pipeline with fixed-checkpoint
evaluation handoff.

## Scientific boundary

The experiment is **transductive/test-time training**: the same 500 MATH-500
questions are used during training and evaluation. Reference answers and solutions
must not be used for generation, synthesis, reward, prompt selection, checkpoint
selection, or early stopping. They are reserved for final evaluation.

The dataset preparation enforces that boundary with two files:

| File | Exact record keys | Allowed use |
| --- | --- | --- |
| `data/math500/questions.jsonl` | `id`, `problem` | Model generation and training |
| `data/math500/labels.jsonl` | `id`, `answer`, `solution`, `subject`, `level` | Evaluation only |

Training code should receive the explicit questions-file path and load it with
`load_locked_questions`, which verifies its locked checksum before parsing. It should
never scan the data directory or load the raw snapshot. Processed IDs are opaque,
deterministic hashes used for bookkeeping; neither IDs nor evaluation metadata should
be inserted into model prompts.

## Recreate MATH-500 from a clean clone

Requirements: Python 3.11 or newer and network access to Hugging Face. The data
preparer has no third-party Python dependencies.

```bash
python3 scripts/prepare_math500.py
python3 scripts/prepare_math500.py --verify-only
python3 -m unittest discover -s tests -v
```

The equivalent shortcuts are `make data`, `make verify-data`, and `make test`.
If `uv` is installed, the checked-in lock also supports:

```bash
uv sync --frozen
uv run --frozen python scripts/prepare_math500.py
```

The preparation command is idempotent. It refuses a corrupt source or an existing
mismatched output. `--force` is available only for an intentional replacement, and
`--offline` rebuilds from an already verified raw snapshot without network access.
For an air-gapped mirror, use `--source-file /path/to/test.jsonl`; the supplied file
must still match the locked SHA-256.

The downloader fetches the pinned JSONL file directly. It does not import or execute
remote dataset-loading code.

## Explore the dataset

The [MATH-500 explorer notebook](notebooks/math500_explorer.ipynb) provides schema
checks, summary tables, subject/level charts, text search, random sampling, and an
opt-in reference-answer reveal. Prepare the data first, then launch it with:

```bash
uv sync --extra notebook --frozen
uv run --extra notebook --frozen jupyter lab notebooks/math500_explorer.ipynb
```

You can also use `make notebook` after syncing the optional notebook environment.
The notebook keeps reference answers hidden unless you explicitly enable them.

## Plan raw and synthesis evaluations

The evaluation layer creates deterministic, provider-neutral request queues, resumes
atomically per task, and scores only after crossing a separate labels boundary. A
small OpenAI-compatible adapter can execute a plan against an already served model;
it never downloads or loads model weights itself.

Start with the model-free full-dataset planning check:

```bash
python3 scripts/evaluate_math500.py plan-raw \
  --config configs/evals/math500_raw.example.toml \
  --run-dir outputs/evals/unused-dry-run \
  --dry-run
```

It should report 500 problems and 4,000 raw requests without writing artifacts,
loading labels, or resolving a model. The example config intentionally refuses a
runnable plan until its model revision, tokenizer revision, chat-template digest,
and adapter version are pinned.

See [the evaluation guide](docs/evaluation.md) for the raw → synthesis → scoring
workflow, paper-versus-local protocol choices, artifact schemas, and fake-backend
tests.

## Plan RL-synthesis training

The training CLI compiles a strict semantic config into a pinned `verl` v0.5.0
launch plan. It implements eight fresh current-policy rollouts, one frozen-policy
synthesis call per question, boxed-answer agreement reward, GRPO, reward-level KL,
and fixed final-checkpoint selection. This command performs a full dataset check
without resolving or loading a model:

```bash
python3 scripts/train_math500.py prepare \
  --config configs/training/math500_cat_grpo.example.toml \
  --run-dir outputs/training/unused-dry-run \
  --dry-run
```

See [the training guide](docs/training.md) for the no-download setup, frozen anchor
service, safe launch and resume flow, checkpoint export, and trained-model raw and
synthesis evaluation.

## Pinned source

- Repository: [`HuggingFaceH4/MATH-500`](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)
- Configuration/split: `default/test`
- Revision: `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`
- Source file: `test.jsonl`
- Rows: 500
- Bytes: 446,564
- SHA-256: `35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132`

The complete source and derived-artifact contract is checked into
[`configs/datasets/math500.lock.json`](configs/datasets/math500.lock.json). Generated
dataset files are deliberately ignored by Git; a clone recreates them byte-for-byte.
This avoids silently redistributing a snapshot whose Hugging Face card does not state
a license or citation. The upstream OpenAI PRM800K and original MATH repositories do
contain MIT license files, but this project does not infer a license for the Hugging
Face snapshot from that fact.

Forty-two questions contain `[asy]...[/asy]` Asymptote diagram source rather than a
rendered image. Preparation preserves that source text exactly.

## Repository layout

```text
configs/datasets/math500.lock.json  immutable dataset and artifact contract
configs/evals/                       paper-profile examples and scoring config
configs/training/                    CaT GRPO semantic training config
scripts/prepare_math500.py          clean-clone entry point
scripts/evaluate_math500.py         planning, endpoint execution, and scoring CLI
scripts/train_math500.py            training plan, launch, and checkpoint CLI
notebooks/math500_explorer.ipynb    interactive dataset explorer
src/compute_as_a_teacher/data/      validation, splitting, and safe question loader
src/compute_as_a_teacher/evaluation/ plans, provenance, grading, and paired metrics
src/compute_as_a_teacher/training/   rewards, verl bridge, plans, and checkpoints
prompts/math500/                     versioned and hash-locked prompt contracts
tests/                              offline unit tests and local-data integrity test
data/raw/                           ignored immutable upstream bytes
data/math500/questions.jsonl        ignored label-free training input
data/math500/labels.jsonl           ignored evaluation-only labels
```

## What comes next

The infrastructure is ready for a separately provisioned model and GPU environment,
but no training or real-model evaluation has been run in this repository. External
anchor and inference endpoints remain explicitly non-reportable until their runtime
and checkpoint identity can be content-attested.

## Dataset provenance

MATH-500 was introduced with Lightman et al., *Let's Verify Step by Step* (2023),
using problems from Hendrycks et al., *Measuring Mathematical Problem Solving With
the MATH Dataset* (NeurIPS 2021). The pinned snapshot links back to the
[`openai/prm800k`](https://github.com/openai/prm800k) source repository.
