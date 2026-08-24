# MATH-500 raw and synthesis evaluation

This milestone implements the complete model-independent evaluation path for the
paper’s MATH-500 experiment:

1. construct eight raw policy-rollout requests per problem;
2. select one existing raw rollout per problem with a fixed, uniformly distributed
   deterministic selector;
3. construct one synthesis request per problem from the eight ordered rollout texts;
4. grade only the selected raw answer and its synthesized answer at a separate
   labels-only boundary; and
5. report paired synthesis deltas against that selected raw baseline.

The repository includes a standard-library OpenAI-compatible adapter for an already
served model. It does not import Transformers or vLLM, load a checkpoint, or download
weights. Fake, external, and HTTP-endpoint results are unconditionally marked
non-reportable by the current adapters; runtime and checkpoint attestation would
require a separately versioned execution backend.

The paper's raw “single” result is one rollout but does not name an index. This
repository selects one of the eight already-generated attempts independently for
each problem using the fixed scoring seed. It never makes another raw model call for
scoring. The paper's Table 1 disagreement results average seven seeds, so this
single-seed profile is not a Table 1 reproduction.

## Paper contract and explicit local choices

The implementation follows these paper-specified choices:

- group size `G = 8`;
- the synthesizer is the frozen initial policy, π0; it matches the raw policy for
  initial-policy evaluation, while trained-policy evaluation uses πT rollouts and
  retains π0 as the synthesis anchor;
- synthesis receives the rollout texts, without a separate original-question field;
- the Appendix F CoT/reasoning synthesis prompt is versioned and hash-locked,
  preserving the displayed body exactly, including `$ boxed{answer}$`;
- Qwen3-4B uses `/no_think`, temperature `0.7`, top-p `0.8`, top-k `20`, and a
  1,536-token output cap; and
- evaluation extracts the final boxed expression and checks string equivalence.

The plan builders enforce the registered prompt, Qwen prefix and sampling profile,
token cap, and group size before writing requests. The cross-stage preregistration
also locks the local seeds and initial/final model relations.

The paper does not publish its raw problem prompt, rollout delimiters, random seeds,
exact string-normalization code, model revisions, tokenizer revision, chat-template
bytes, or runtime adapter. This repository therefore records the following local
choices rather than presenting them as paper text:

- `raw_math500_local_v1` is the raw prompt;
- the canonical `paper_appendix_f_cot_literal_v1` prompt preserves the displayed
  Appendix F instruction, including `$ boxed{answer}$`. The separately registered
  `paper_appendix_f_cot_boxfix_v1` variant adds a backslash for a future sensitivity
  protocol and is not used by this reproduction;
- synthesis rollouts are serialized as `## RESPONSE 1` through `## RESPONSE 8`;
- SHA-derived per-question seeds use a checked-in base seed;
- the raw comparison baseline uses
  `sha256_uniform_per_question_v1` with scoring seed `1729`: a domain-separated
  SHA-256 digest of the UTF-8 canonical JSON line
  `["compute_as_a_teacher/raw_baseline/sha256_uniform_per_question_v1",
  protocol_version, 1729, question_id]`, including its terminal newline, interpreted
  as a big-endian integer and reduced modulo eight;
- sampling also fixes `do_sample=true`, single-beam generation, repetition penalty
  `1.0`, and no stop strings; these values are local choices, not paper facts;
- endpoint outputs accept only `stop` and `length` finish reasons, and boxed-answer
  extraction has a 50,000-character safety cap; unsupported finish reasons are
  infrastructure failures, while extraction failures are scored incorrect;
- Appendix I says regex extraction plus string matching but does not publish the
  regex. `last_boxed_string_exact_v1` is therefore a disclosed balanced-brace,
  last-`\\boxed` approximation, not a claim of byte-exact parser reproduction;
- the literal Appendix F ending says `$ boxed{answer}$` while this implementation
  requires `\\boxed{...}` for extraction. A naturally unextractable evaluation
  output is preserved and graded incorrect; the one-backslash prompt repair is
  reserved for a separately preregistered sensitivity run;
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

The server profile enforces this boundary with mounts, not only API discipline.
`trainer` and `evaluator` mount the single locked questions file; neither can see the
raw snapshot or labels path. The separate network-disabled scorer mounts only the
locked questions and labels files, with no GPU, model mount, endpoint credential, or
W&B credential. Baseline and trained generation both finish before scoring begins,
and baseline scoring is gated on the registered terminal handoff, so label-derived
score artifacts do not exist during training or generation.
Trainer preflight and launch also reject stale label-derived artifacts in the output
mount, so the guarded workflow cannot re-enter training after scoring.

## Setup and model-free checks

Prepare the pinned dataset and install the optional symbolic diagnostic:

```bash
python3 scripts/prepare_math500.py
uv sync --extra notebook --extra evaluation --frozen
source .venv/bin/activate
```

Keep this checked-in environment active for the commands below. This matters on
Python 3.10, where TOML support comes from the locked compatibility dependency.

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
deterministic raw selection, paired metrics, nested boxed expressions, and symbolic
answer equivalence. It never
calls a real model or scores MATH-500 model predictions.

## Creating a runnable plan later

Copy the example configs and replace every unresolved model field with real immutable
provenance:

```bash
cp configs/evals/math500_raw.example.toml configs/evals/math500_raw.toml
cp configs/evals/math500_synthesis.example.toml configs/evals/math500_synthesis.toml
```

The checked-in synthesis example declares `anchor_relation = "same_as_raw"`, so its
raw and synthesis model blocks must match exactly, including provider adapter,
model revision, tokenizer revision, chat-template SHA-256, adapter version, dtype,
quantization, and seed-support declaration. A trained-checkpoint handoff instead
declares `frozen_initial_for_trained_raw`, binding raw rollout requests to πT and
synthesis requests to the training plan's frozen π0. Planning refuses floating
markers such as `main`, `latest`, `dev`, or `required_*`, and requires the
chat-template digest to be 64 lowercase hexadecimal characters.

`chat_template_sha256` means the SHA-256 of the exact raw tokenizer chat-template
source encoded as UTF-8. It is not a hash of any rendered prompt or message. Preserve
those source bytes before hashing so whitespace or template edits change the model
identity recorded in the plan.

Writing a raw plan still does not instantiate a backend or load a model:

```bash
python3 scripts/evaluate_math500.py plan-raw \
  --config configs/evals/math500_raw.toml \
  --run-dir outputs/evals/math500-initial-raw
```

For the full RL experiment, stop here and create the canonical training plan and
experiment preregistration described in the training guide before executing any raw
requests. This prevents baseline results from influencing later training choices.

The resulting `requests.jsonl` is the adapter-neutral work queue. Each request is
keyed by a semantic task ID and includes the fully resolved model contract, message,
sampling parameters, and deterministic seed. `inspect-plan --run-dir <run>`
reverifies and summarizes either kind of plan without loading labels or a model.

The OpenAI-compatible adapter maps all planned messages and sampling fields and
returns outputs keyed by both `task_id` and `request_fingerprint`. Serve the exact
planned model name, then explicitly start or resume execution with the shipped
adapter. This command requires `provider = "openai-compatible"` and
`adapter_version = "openai-compatible-chat-v1"` in the model contract:

```bash
python3 scripts/evaluate_math500.py run-openai \
  --run-dir outputs/evals/math500-initial-raw \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key-env CAT_EVAL_API_KEY \
  --batch-size 16
```

The API rejects positional output mix-ups, a changed backend on resume, unsupported
sampling fields, and any cached result whose request, backend, or output hash
changed. The backend fingerprint includes a SHA-256 identity for the normalized
chat-completions endpoint, without recording its URL or API key. An interrupted run
can recover only a validated, same-endpoint backend, contiguous
append-only result prefix; gaps or changes to checkpointed results fail closed. Before
synthesis planning or scoring, a shared verifier revalidates the complete execution
against the immutable plan, every per-task result, the result-set fingerprint, the
canonical generation artifact, the exact backend descriptor, and derived
reportability reasons; it does not trust a stored `complete` or `non_reportable`
flag on its own. Execution requires the explicit `run-openai` command and an endpoint
that the user has started separately.

The current fake, external-ingest, and OpenAI-compatible HTTP backends always emit
`reportable: false`, even when every integrity check passes. The
model contract's `seed_support` value (`strict`, `best_effort`, or `none`) is still
recorded as `seed_reproducibility`, but it is a declaration rather than proof of
runtime behavior. Reportable execution requires a new versioned backend and schema
that attest checkpoint, tokenizer, chat template, runtime, and seed handling.

## Synthesis planning and scoring

After all raw results have verified execution provenance and a canonical
`generations.jsonl`, write the synthesis plan:

```bash
python3 scripts/evaluate_math500.py plan-synthesis \
  --config configs/evals/math500_synthesis.toml \
  --raw-run-dir outputs/evals/math500-initial-raw \
  --run-dir outputs/evals/math500-initial-synthesis
```

This should produce 500 synthesis requests, each bound to the hashes and task IDs of
exactly eight raw rollouts. Changing a raw output invalidates the synthesis lineage.

Execute the synthesis requests against the same frozen initial model:

```bash
python3 scripts/evaluate_math500.py run-openai \
  --run-dir outputs/evals/math500-initial-synthesis \
  --base-url http://127.0.0.1:8001/v1 \
  --api-key-env CAT_ANCHOR_API_KEY \
  --batch-size 16
```

Only after both raw and synthesis generation are complete should labels be loaded.
The synthesis scoring command reads the existing raw generations, selects one raw
attempt per problem according to the locked scoring config, and grades the 500 raw
and synthesis pairs. It does not run inference and does not require a prior
`score-raw` command:

```bash
python3 scripts/evaluate_math500.py score-synthesis \
  --run-dir outputs/evals/math500-initial-synthesis \
  --raw-run-dir outputs/evals/math500-initial-raw \
  --config configs/evals/math500_scoring.toml
```

For a canonical run, `preregister-experiment --scoring-config ...` content-addresses
this exact config before any result exists. Finalization rejects a changed selector
method, seed, grader, label path, or scoring limit.

Synthesis scoring refuses a different raw run, changed raw generations, changed raw
lineage, a different label digest, or a different scoring config. The fixed selector
is applied to the eight source rollouts named in each synthesis request, so the
comparison is stable across reruns and cannot be chosen from the labels. Reportability
is transitive: a non-reportable raw dependency makes the synthesis report
non-reportable even if its own execution provenance is otherwise valid.

For every configured grader, `summary.json` reports `synthesis_accuracy`,
`raw_baseline_accuracy`, and `paired_delta_vs_raw_baseline`, plus their standard
errors. It also records the selector method, seed, domain, and selected-index
histogram. `paired_scores.jsonl` contains one auditable selected-raw/synthesis pair
per problem.

## External response ingestion

The CLI can validate complete adapter-produced response files for development:

```bash
python3 scripts/evaluate_math500.py ingest-raw \
  --run-dir outputs/evals/math500-initial-raw \
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
runtime, and seed behavior produced those bytes. The HTTP adapter is also
non-reportable because the endpoint cannot prove that identity. Use the identical
schema with `ingest-synthesis` for a synthesis plan.

## Artifact layout

```text
outputs/evals/<run>/
├── manifest.json              immutable semantic plan and label-firewall record
├── requests.jsonl             deterministic provider-neutral work queue
├── results/<task-id>.json     atomic, resumable per-task result
├── generations.jsonl          canonical plan-ordered predictions
├── execution.json             backend fingerprint, completeness, reportability
├── scores.jsonl               synthesis predictions and grades, never references
├── paired_scores.jsonl        selected-raw/synthesis paired grades (synthesis run)
├── summary.json               metrics and paired analyses
└── scoring_manifest.json      exact generation/lock/label/raw-generation lineage
```

`score-synthesis` writes its label-derived artifacts only in the synthesis run
directory. The raw run remains unchanged and supplies verified generation bytes and
execution provenance read-only.
The selected-baseline synthesis summary uses schema version 2, paired rows use
version 1, and the changed scoring-manifest lineage uses version 3.

Plans and results use canonical JSON plus SHA-256 identities. Per-task publication is
atomic. Resume skips only a valid result with the same request and backend
fingerprints. Re-planning a directory containing descendant artifacts is refused;
use a new run directory instead.
