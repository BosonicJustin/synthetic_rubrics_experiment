# Generated data

Run `python3 scripts/prepare_math500.py` from the repository root. The command
creates three ignored files:

- `raw/math500-test.jsonl`: exact bytes from the immutable upstream revision;
- `math500/questions.jsonl`: label-free training/generation records;
- `math500/labels.jsonl`: evaluation-only references and metadata.

Do not point generation or RL code at this directory. Pass the exact
`data/math500/questions.jsonl` path to `load_locked_questions`, which verifies the
file against the lock before returning typed question records. The lock at
`../configs/datasets/math500.lock.json` records every expected checksum.
