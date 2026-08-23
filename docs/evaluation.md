# MATH-500 raw and synthesis evaluation

This milestone implements the complete model-independent evaluation path for the
paper’s MATH-500 experiment:

1. construct eight raw policy-rollout requests per problem;
2. treat rollout index 0 as the predeclared single-sample raw baseline while also
   reporting mean-of-eight, any-correct-at-eight, and literal plurality diagnostics;
3. construct one synthesis request per problem from the eight ordered rollout texts;
4. grade raw and synthesis outputs at a separate labels-only boundary; and
5. report paired synthesis deltas against raw rollout 0 and the raw eight-sample mean.

There is intentionally no real model adapter or model-execution command yet. No
Transformers, vLLM, provider SDK, checkpoint, or model download is part of this
repository milestone. The end-to-end path is tested with an injected fake backend,
and every fake or externally unattested result is permanently marked non-reportable.

## Paper contract and explicit local choices

The implementation follows these paper-specified choices:

- group size `G = 8`;
- the synthesizer is the same frozen initial policy used for raw rollouts;
- synthesis receives the rollout texts, without a separate original-question field;
- the Appendix F CoT/reasoning synthesis prompt is versioned and hash-locked,
  with the disclosed repair described below;
- Qwen3-4B uses `/no_think`, temperature `0.7`, top-p `0.8`, top-k `20`, and a
  1,536-token output cap; and
- evaluation extracts the final boxed expression and checks string equivalence.

The paper does not publish its raw problem prompt, rollout delimiters, random seeds,
exact string-normalization code, model revisions, tokenizer revision, chat-template
bytes, or runtime adapter. This repository therefore records the following local
choices rather than presenting them as paper text:

- `raw_math500_local_v1` is the raw prompt;
- the paper's displayed Appendix F instruction prints `$ boxed{answer}$`; the
  registered `paper_appendix_f_cot_boxfix_v1` prompt adds the missing backslash as
  `$\boxed{answer}$`. This one-character repair is not presented as a byte-for-byte
  transcription;
- synthesis rollouts are serialized as `## RESPONSE 1` through `## RESPONSE 8`;
- SHA-derived per-question seeds use a checked-in base seed;
- `Qwen/Qwen3-4B` is this repository's Hugging Face Hub mapping for the paper's
  `Qwen3-4B` model label; the paper does not publish a registry ID or revision;
- `last_boxed_string_exact_v1` strips only outer whitespace and is the primary
  reproduction grader; and
- `math_verify_v0.9.0` is a separately named equivalence diagnostic.

Both prompt files have registered SHA-256 digests. Editing their text without
declaring a new prompt version fails before a plan can be written.

## Label firewall

Generation planning loads only `data/math500/questions.jsonl`, whose exact keys are
`id` and `problem`. It verifies that the supplied typed records serialize to the
same locked bytes. Raw model messages contain only the rendered problem prompt.

Synthesis planning consumes only a verified raw `generations.jsonl`. It joins every
row to its immutable raw request and supplies only the eight ordered output texts to
the synthesis prompt. A policy rollout can naturally repeat its problem text; the
important contract is that synthesis receives no separate question or label field.

`data/math500/labels.jsonl` is first opened by the scoring command, after generation
is complete. Production scoring accepts only the exact path, digest, size, row count,
and schema declared by the dataset lock. The raw plan preserves that lock file's
digest and exact question-artifact reference; scoring requires both to match the
lock that supplied the labels, so an aligned-ID artifact from another or replaced
lock is rejected. The scorer retains only `id`, `answer`, `subject`, and `level` in
memory, immediately discards `solution`, and never serializes a reference answer or
solution into score rows. Tests use an explicit fixture-only scoring entry point;
fixture questions or labels always add a non-reportable reason and cannot produce a
reportable result.

## Setup and model-free checks

Prepare the pinned dataset and install the optional symbolic diagnostic:

```bash
python3 scripts/prepare_math500.py
uv sync --extra notebook --extra evaluation --frozen
```

Inspect the available commands:

```bash
python3 scripts/evaluate_math500.py --help
```

The checked-in raw example deliberately contains unresolved `required_*` model
fields. A dry run accepts those placeholders, verifies the question and prompt
artifacts, constructs all semantic request identities in memory, and writes nothing:

```bash
python3 scripts/evaluate_math500.py plan-raw \
  --config configs/evals/math500_raw.example.toml \
  --run-dir outputs/evals/unused-dry-run \
  --dry-run
```

Expected counts are 500 problems and 4,000 raw rollout requests. The command also
prints `model_loaded: false` and `labels_loaded: false`.

Run the no-model integration suite with:

```bash
uv run --extra notebook --extra evaluation --frozen \
  python -m unittest discover -s tests/evaluation -v
```

The suite uses two synthetic questions, 16 keyed raw outputs, and two synthesis
outputs. It tests crash-safe append-only resume, reversed batch-return order, prompt
leakage sentinels, request/result tampering, cross-lock and stale-score rejection,
wrong raw lineage, unsupported sampling fields, external-ingest reportability,
paired metrics, nested boxed expressions, and symbolic answer equivalence. It never
calls a real model or scores MATH-500 model predictions.

## Creating a runnable plan later

Copy the example configs and replace every unresolved model field with real immutable
provenance:

```bash
cp configs/evals/math500_raw.example.toml configs/evals/math500_raw.toml
cp configs/evals/math500_synthesis.example.toml configs/evals/math500_synthesis.toml
```

The raw and synthesis model blocks must match exactly, including provider adapter,
model revision, tokenizer revision, chat-template SHA-256, adapter version, dtype,
quantization, and seed-support declaration. Planning refuses floating markers such
as `main`, `latest`, `dev`, or `required_*`, and requires the chat-template digest
to be 64 lowercase hexadecimal characters.

`chat_template_sha256` means the SHA-256 of the exact raw tokenizer chat-template
source encoded as UTF-8. It is not a hash of any rendered prompt or message. Preserve
those source bytes before hashing so whitespace or template edits change the model
identity recorded in the plan.

Writing a raw plan still does not instantiate a backend or load a model:

```bash
python3 scripts/evaluate_math500.py plan-raw \
  --config configs/evals/math500_raw.toml \
  --run-dir outputs/evals/qwen3-4b-raw
```

The resulting `requests.jsonl` is the adapter-neutral work queue. Each request is
keyed by a semantic task ID and includes the fully resolved model contract, message,
sampling parameters, and deterministic seed.

Once a future real adapter is implemented, it should implement
`GenerationBackend`, declare every supported sampling field, return outputs keyed by
both `task_id` and `request_fingerprint`, validate the synthesis input against its
context window after tokenization, and call the resumable `execute_plan` API. The API
rejects positional output mix-ups, a changed backend on resume, unsupported `top_k`,
and any cached result whose request, backend, or output hash changed. An interrupted
checkpoint can recover only a validated, same-backend, contiguous append-only result
prefix; gaps or changes to checkpointed results fail closed. Before
synthesis planning or scoring, a shared verifier revalidates the complete execution
against the immutable plan, every per-task result, the result-set fingerprint, the
canonical generation artifact, the exact backend descriptor, and derived
reportability reasons; it does not trust a stored `complete` or `non_reportable`
flag on its own. It is not wired to this milestone’s CLI so there is no accidental
real-model execution path.

`reportable: true` means that these provenance and locked-input checks passed. It
does not promise byte-identical replay of stochastic sampling across runtimes or
hardware. The model contract's `seed_support` value (`strict`, `best_effort`, or
`none`) records the adapter's declared ability to honor per-request seeds and is
copied into execution provenance as `seed_reproducibility`.

## Synthesis planning and scoring

After all raw results have verified execution provenance and a canonical
`generations.jsonl`, write the synthesis plan:

```bash
python3 scripts/evaluate_math500.py plan-synthesis \
  --config configs/evals/math500_synthesis.toml \
  --raw-run-dir outputs/evals/qwen3-4b-raw \
  --run-dir outputs/evals/qwen3-4b-synthesis
```

This should produce 500 synthesis requests, each bound to the hashes and task IDs of
exactly eight raw rollouts. Changing a raw output invalidates the synthesis lineage.

Only after generation is complete should labels be loaded:

```bash
python3 scripts/evaluate_math500.py score-raw \
  --run-dir outputs/evals/qwen3-4b-raw \
  --config configs/evals/math500_scoring.toml

python3 scripts/evaluate_math500.py score-synthesis \
  --run-dir outputs/evals/qwen3-4b-synthesis \
  --raw-run-dir outputs/evals/qwen3-4b-raw \
  --config configs/evals/math500_scoring.toml
```

Synthesis scoring refuses a different raw run, changed raw generations, changed raw
scores, a different label digest, or a different scoring config. Reportability is
transitive: a non-reportable raw dependency makes the synthesis report
non-reportable even if its own execution provenance is otherwise valid.

## External response ingestion

The CLI can validate complete adapter-produced response files for development:

```bash
python3 scripts/evaluate_math500.py ingest-raw \
  --run-dir outputs/evals/qwen3-4b-raw \
  --responses /path/to/raw-responses.jsonl
```

Each line must use this exact schema:

```json
{
  "schema_version": 1,
  "task_id": "raw-...",
  "request_fingerprint": "64-lowercase-hex",
  "backend_fingerprint": "64-lowercase-hex",
  "text": "reasoning and \\boxed{answer}",
  "output_sha256": "sha256-of-the-exact-text",
  "finish_reason": "stop",
  "usage": {"prompt_tokens": 0, "completion_tokens": 0},
  "provider_metadata": {}
}
```

Coverage must exactly equal the plan. Duplicate, missing, extra, mismatched, or
mixed-backend rows fail before publication. External ingest remains non-reportable
because the file alone cannot attest that the declared model, tokenizer, template,
runtime, and seed behavior produced those bytes. A future real backend adapter is
the route to reportable execution.

## Artifact layout

```text
outputs/evals/<run>/
├── manifest.json              immutable semantic plan and label-firewall record
├── requests.jsonl             deterministic provider-neutral work queue
├── results/<task-id>.json     atomic, resumable per-task result
├── generations.jsonl          canonical plan-ordered predictions
├── execution.json             backend fingerprint, completeness, reportability
├── scores.jsonl               predictions and grades, never references
├── summary.json               metrics and paired analyses
└── scoring_manifest.json      exact generation/lock/label/raw-score lineage
```

Plans and results use canonical JSON plus SHA-256 identities. Per-task publication is
atomic. Resume skips only a valid result with the same request and backend
fingerprints. Re-planning a directory containing descendant artifacts is refused;
use a new run directory instead.
