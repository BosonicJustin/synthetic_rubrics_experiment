# MATH-500 experiment plan

This is the predeclared order of operations for the first real run. Freeze the
repository commit, model inventory, runtime images, configs, seeds, hypotheses, and
analysis before reading baseline scores. Labels may determine reported metrics, but
never prompts, rewards, runtime gates, checkpoint selection, or training changes.

## Fidelity boundary

The paper fixes the scientific contract but does not publish enough information for
a byte-exact prompt or runtime reproduction. Do not present a locked local choice as
a paper fact.

| Boundary | Published fact | Locked treatment in this repository |
| --- | --- | --- |
| Actors and grouping | Each `pi_t` produces `G = 8` rollouts; frozen initial `pi_0` synthesizes from the rollout texts only. | Preserve ordering and never add the question or a label to synthesis. |
| Synthesis and reward | Appendix F gives the CoT synthesis instruction; reward uses regex extraction and equality of boxed strings. | Use and hash the displayed prompt exactly, including `$ boxed{answer}$`; lock a disclosed balanced-brace extractor because the paper's regex is unpublished, and keep the one-backslash prompt repair outside the canonical protocol. |
| Qwen generation | Temperature `0.7`, top-p `0.8`, top-k `20`, `/no_think`, and 1,536 generated tokens. | Lock the otherwise unpublished model/tokenizer revision, chat-template bytes, adapter, `do_sample`, one-sequence, repetition, stop, and runtime choices. |
| GRPO run | LR `5e-7`, constant schedule, no warmup, global batch 256, reward KL `1e-3`, 1,000 steps, AdamW, FSDP, eight H100s, and per-response token averaging in Eq. 6. | Interpret the unpublished batch unit as prompts; pin sequence-mean/token-mean loss aggregation; verify Verl's FSDP actor constructs AdamW; and lock the unpublished `ddof`, epsilon, clip, PPO epochs and mini-batch, sampled-token KL estimator/controller, optimizer details, and distributed stack. |
| Raw inputs and randomness | The exact raw problem prompt, rollout delimiters, and seeds are not published. | Treat the versioned raw prompt, delimiters, and seeds as preregistered local choices and blockers to a byte-exact claim. |
| Reported raw “single” score | The paper defines it as one rollout but does not identify a deterministic sample index. | Use rollout 0 as the predeclared single sample; report mean-of-eight only as a diagnostic. |

## Hypotheses and metrics

The canonical run uses data-order seed `42`, policy-rollout base seed `1729`,
synthesis base seed `2718`, and the fixed step-1,000 checkpoint. These are separate
seed streams; the code does not claim that `42` initializes every framework RNG.

- H1: frozen initial-policy synthesis improves accuracy over initial-policy raw
  rollout 0.
- H2: the final trained policy's raw rollout-0 accuracy exceeds the initial raw
  rollout-0 accuracy.
- H3: frozen initial-policy synthesis over final-policy rollouts improves accuracy
  over final-policy raw rollout 0.

Primary metrics are raw rollout-0 accuracy, synthesis accuracy, and their paired
deltas. Also report mean-of-eight raw accuracy, empirical any-correct-at-eight,
literal plurality accuracy, extraction-failure rates, and the standard errors already
emitted by scoring. Report every predeclared comparison even when its sign is not as
hypothesized.

The emitted synthesis-versus-selection disagreement diagnostics belong to this
single-seed run. Do not present them as a reproduction of paper Table 1, which
averages seven seeds, unless the multi-seed extension below is preregistered and run.

Training gates are label-free. Record reward mean and variance, zero/one reward
fractions, rollout and anchor extraction statuses, KL, policy loss, gradient norm,
step time, token throughput, GPU memory, anchor latency/error rate, and checkpoint
time and size. Do not use MATH-500 accuracy for early stopping.

## Scale and budget ceiling

The canonical plan contains 2,048,000 policy trajectories and 256,000 anchor calls.
Its configured hard caps imply:

| Quantity | Upper bound |
| --- | ---: |
| Policy completion tokens | 3,145,728,000 |
| Anchor completion tokens | 393,216,000 |
| Total generated training tokens | 3,538,944,000 |
| Logical policy prompt-plus-completion tokens | 7,340,032,000 |
| Anchor input from rollout texts, before delimiters | about 3,145,728,000 |
| One raw-plus-synthesis evaluation | 6,912,000 generated tokens |
| Initial and final evaluation pairs | 13,824,000 generated tokens |

These are safety ceilings, not forecasts; EOS should reduce actual usage. The paper
reports eight H100s but does not publish anchor placement. This repository's
external-anchor architecture requires its hardware and contention policy to be
recorded and budgeted explicitly; do not infer that the paper's count excludes it.
After full-shape qualification, set a wall-time ceiling to 1.25 times the
extrapolated 1,000-step time, including ten measured checkpoint writes. Derive
trainer and anchor GPU-hour ceilings from that wall time and their GPU counts. The
currency ceiling is those GPU-hour ceilings
times the locked provider rates, plus storage and network charges. Record all rates
and numeric limits in the manual attestation and bind them into the launch approval.
Reserve observed checkpoint size times three, one merged export, logs, and at least
25% storage headroom. Do not start the canonical run until these ceilings are
written down and approved. They are content-addressed approval metadata, not an
in-process kill switch; enforce wall-time, spend, and storage limits in the scheduler
or provider and archive those controls with the run.

## Staged execution

Each stage has a stop gate. Smoke runs are operational checks and must use separate,
explicitly non-reportable run directories; they are not paper results.

Launch commands verify terminal checkpoint structure and preserve trainer stdout and
per-step rollout records. They do not yet turn Verl metrics into a machine-validated
qualification decision. Review each named gate and archive an explicit sign-off;
`qualification_process_finished` never means `qualified`.

### 0. Freeze the experiment

- Pin the Git commit, clean `verl` commit, trainer image digest, exact target-package
  inventory, configured CUDA/cuDNN versions, GPU inventory, full local model-tree
  hash, model/tokenizer revisions, and chat-template hash. Record the observed host
  driver and NCCL versions with the external run record; this protocol does not
  independently config-pin them beyond the immutable image and provider controls.
- Require structurally valid Qwen safetensors/index metadata, sufficient context for
  the policy and worst-case eight-rollout anchor request, repo-owned dataset/reward
  imports, and the preregistered per-GPU free-memory floor.
- Archive the host-validator JSON, provider allocation, and anchor startup/health
  log for the same content-locked `pi_0` snapshot. The reviewer records this external
  hardware evidence in the manual attestation; the current schema binds endpoint
  canaries, not the external anchor process itself.
- Freeze initial and final evaluation seeds. Preregister the initial raw plan,
  intended synthesis contract, and canonical training plan now; the final checkpoint
  and final evaluation plans are joined later by `finalize-experiment`.
- Write `preregister-experiment` after the initial raw and canonical training plans
  exist but before raw results, trainer logs, rollout logs, or checkpoints exist.
- Gate: all local identities resolve, the label firewall passes, and dependencies
  are content-locked. Freeze configured endpoint aliases and archive the external
  startup evidence while retaining the explicit scientific nonattestation of HTTP
  endpoint weights and runtime behavior.

### 1. Register the initial baseline

- Freeze resolved raw-`pi_0` and synthesis-`pi_0` configs, then write the 4,000-row
  raw plan. Synthesis will consume only each ordered group of eight rollout texts
  after raw execution completes.
- Freeze the raw plan fingerprint, expected synthesis contract, and training plan
  before any scoring.
- Verify that the preregistration records `pi_0` raw rollouts synthesized by the same
  frozen `pi_0`, with the exact registered prompt hashes and predeclared seeds.
- Gate: plans and configs contain no labels, model identities match, raw count is
  4,000, and no later training choice may depend on baseline scores.

### 2. Endpoint canary and baseline completion

- Run 16 raw requests with `run-openai --max-requests 16`, then rerun to prove
  validated resume. Probe the frozen anchor with the semantic and exact-tokenized
  long-context eight-rollout canaries, using a unique boxed nonce only near the tail.
- Verify model name, sampling-field support, seed handling, long-request acceptance
  with the boxed answer preserved at the tail,
  response schema, extraction, latency, and that synthesis receives no separate
  question or label.
- Gate: zero request failures, stable endpoint identity, successful resume, and
  valid boxed extraction. Then complete raw execution, write the 500-row synthesis
  plan, and complete synthesis. Do not score yet; the labels boundary remains closed
  until terminal training, guarded handoff, and trained-policy generation finish.

### 3. One-step training smoke

- Use a dedicated non-reportable smoke profile with a tiny label-free question set,
  one update, and checkpointing every step. Do not weaken the canonical config.
- Gate: the real pinned `verl` job composes, the custom dataset and reward hook load,
  the anchor is called once per rollout group, losses/KL/gradients are finite, reward
  is not uniformly invalid, and a checkpoint is written.

Use the derived `one_step` qualification plan. It is fingerprinted to the canonical
plan, label-free, stored separately, and hard non-reportable. Execute it only in the
locked GPU environment.

### 4. Kill-and-resume smoke

- Use the derived `resume_three_step` plan, terminate after the complete step-1
  checkpoint, and restart with `resume_mode = "auto"` through step 3. This smoke
  validates interruption and checkpoint recovery only; guarded merge/load validation
  is performed later on the fixed canonical checkpoint.
- Gate: the resumed step and data position are correct, the frozen anchor identity
  is unchanged, cached artifacts validate, and no concurrent process can write the
  same run directory.

### 5. Full-shape qualification

- Use `full_shape_five_step` for five updates—the lower bound of this 5–10-step
  gate—with the canonical 256 prompts, eight rollouts, eight-H100 trainer topology,
  and final anchor hardware. Use no labels and do not score a checkpoint.
- Gate: no OOM or anchor errors; reward/extraction, KL, loss, and gradient metrics are
  sane; p95 anchor latency does not stall workers; checkpoint/resume works; projected
  time, cost, and storage fit the approved budget with headroom.
- Complete the fail-closed manual attestation with numeric wall-time, trainer and
  anchor GPU-hour, storage, and currency ceilings. Write and reverify the canonical
  launch approval, which content-addresses all three qualification plans, preflights,
  terminal checkpoints, trainer logs, and rollout logs. The reviewer must copy the
  inspected evidence-bundle fingerprint into the attestation before signing it.

### 6. Canonical 1,000-step run

- Launch the immutable data-order-42, rollout-1729, synthesis-2718 plan; resume only
  from its own verified checkpoints and select exactly step 1,000 without
  label-based inspection.
- Require the preregistration and current launch approval on every canonical start
  or resume; rerun the full model, runtime, tokenizer, Hydra, and anchor preflight
  immediately before spawning Verl.
- Merge and register the fixed checkpoint, then evaluate `pi_T` raw rollouts and
  frozen-`pi_0` synthesis using the predeclared evaluation seeds.
- Only after all four generation runs verify complete, start the offline scorer and
  score the initial raw/synthesis pair and trained raw/synthesis pair exactly once.
- Finalize and reverify the cross-stage registry after all four evaluation plans and
  the fixed checkpoint exist. It must record `pi_T` rollouts synthesized by frozen
  `pi_0`; it does not replace archival of results or scores.
- Gate: all stage fingerprints link, environment and anchor identities remained
  fixed, the terminal export loads, and both raw and synthesis score artifacts verify.

### 7. Optional multi-seed robustness

The registered v1 profile intentionally locks its three seed streams. A multi-seed
extension therefore requires a new protocol version that explicitly varies data
order, policy rollouts, synthesis, and any framework RNGs discovered during target
runtime qualification. Predeclare every seed tuple, keep evaluation requests common
across compared checkpoints, report every run, and never select the best seed as the
headline result.

## Required run record

Archive the resolved configs, plan fingerprints, code and prompt hashes, model and
container inventories, sanitized environment manifest, hardware inventory, stdout
and structured metrics, anchor health/latency log, checkpoint inventories, merge
lineage, evaluation artifacts, failure/resume events, measured token counts,
GPU-hours, storage, and cost. Until endpoint and distributed-runtime identity are
attested by a new versioned execution backend and schema, artifacts produced by the
current fake, external, HTTP, and training adapters remain explicitly non-reportable
outside exploratory use.
