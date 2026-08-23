from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from compute_as_a_teacher.data.math500 import (  # noqa: E402
    DatasetPreparationError,
    _load_questions,
    _preflight_destination,
    acquire_source,
    build_views,
    canonical_jsonl,
    load_dataset_lock,
    load_locked_questions,
    main,
    read_source_rows,
    verify_dataset,
    verify_locked_questions,
)


FIXTURE = REPOSITORY_ROOT / "tests/fixtures/math500_tiny.jsonl"
LOCK_PATH = REPOSITORY_ROOT / "configs/datasets/math500.lock.json"


class Math500UnitTests(unittest.TestCase):
    def test_checked_in_lock_uses_an_immutable_revision(self) -> None:
        lock = load_dataset_lock(LOCK_PATH)
        revision = lock["dataset"]["resolved_revision"]
        self.assertRegex(revision, re.compile(r"^[0-9a-f]{40}$"))
        self.assertIn(f"/resolve/{revision}/", lock["dataset"]["source_url"])

    def test_lock_rejects_a_floating_revision(self) -> None:
        lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        lock["dataset"]["requested_revision"] = "main"
        lock["dataset"]["resolved_revision"] = "main"
        lock["dataset"]["source_url"] = (
            "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "floating.lock.json"
            path.write_text(json.dumps(lock), encoding="utf-8")
            with self.assertRaisesRegex(DatasetPreparationError, "full 40-character"):
                load_dataset_lock(path)

    def test_views_have_exact_separate_schemas(self) -> None:
        rows = read_source_rows(FIXTURE, expected_rows=2)
        questions, labels = build_views(rows)
        self.assertEqual(set(questions[0]), {"id", "problem"})
        self.assertEqual(
            set(labels[0]), {"id", "answer", "solution", "subject", "level"}
        )
        self.assertEqual(
            [record["id"] for record in questions],
            [record["id"] for record in labels],
        )
        self.assertRegex(questions[0]["id"], re.compile(r"^math500-[0-9a-f]{16}$"))
        self.assertNotIn("algebra", questions[0]["id"])

    def test_question_loader_rejects_a_label_sentinel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "unsafe.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "id": "fixture/1",
                        "problem": "A harmless prompt",
                        "answer": "DO_NOT_LEAK_THIS_SENTINEL",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(DatasetPreparationError, "unsafe schema"):
                _load_questions(path)

    def test_question_loader_returns_only_typed_question_fields(self) -> None:
        rows = read_source_rows(FIXTURE, expected_rows=2)
        questions, _ = build_views(rows)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "questions.jsonl"
            path.write_bytes(canonical_jsonl(questions))
            loaded = _load_questions(path, expected_rows=2)
        self.assertEqual(set(dataclasses.asdict(loaded[0])), {"id", "problem"})
        self.assertFalse(hasattr(loaded[0], "answer"))
        self.assertFalse(hasattr(loaded[0], "solution"))

    def test_locked_question_loader_rejects_tampering(self) -> None:
        questions = [
            {"id": f"fixture/{index}", "problem": f"Problem {index}"}
            for index in range(500)
        ]
        payload = canonical_jsonl(questions)
        lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        lock["outputs"]["questions"]["bytes"] = len(payload)
        lock["outputs"]["questions"]["sha256"] = hashlib.sha256(payload).hexdigest()

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            questions_path = root / "questions.jsonl"
            lock_path = root / "math500.lock.json"
            questions_path.write_bytes(payload)
            lock_path.write_text(json.dumps(lock), encoding="utf-8")

            loaded = load_locked_questions(questions_path, lock_path=lock_path)
            self.assertEqual(len(loaded), 500)

            questions_path.write_bytes(payload + b"tampered\n")
            with self.assertRaisesRegex(DatasetPreparationError, "checksum/size mismatch"):
                load_locked_questions(questions_path, lock_path=lock_path)

    def test_question_only_verification_never_opens_absent_raw_or_labels(self) -> None:
        questions = [
            {"id": f"fixture/{index}", "problem": f"Problem {index}"}
            for index in range(500)
        ]
        payload = canonical_jsonl(questions)
        lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        lock["outputs"]["questions"]["bytes"] = len(payload)
        lock["outputs"]["questions"]["sha256"] = hashlib.sha256(payload).hexdigest()

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            questions_path = root / lock["outputs"]["questions"]["path"]
            lock_path = root / "math500.lock.json"
            questions_path.parent.mkdir(parents=True)
            questions_path.write_bytes(payload)
            lock_path.write_text(json.dumps(lock), encoding="utf-8")
            forbidden = {
                (root / lock["source"]["path"]).resolve(),
                (root / lock["outputs"]["labels"]["path"]).resolve(),
            }
            real_open = Path.open

            def guarded_open(path: Path, *args: object, **kwargs: object):
                if path.resolve() in forbidden:
                    raise AssertionError(f"opened forbidden artifact: {path}")
                return real_open(path, *args, **kwargs)

            output = io.StringIO()
            with mock.patch.object(Path, "open", guarded_open), redirect_stdout(output):
                status = main(
                    [
                        "--repo-root",
                        str(root),
                        "--lock-file",
                        str(lock_path),
                        "--verify-questions-only",
                    ]
                )

            self.assertEqual(status, 0)
            self.assertIn("rows=500", output.getvalue())
            self.assertFalse((root / lock["source"]["path"]).exists())
            self.assertFalse((root / lock["outputs"]["labels"]["path"]).exists())

    def test_question_only_verification_rejects_tampering(self) -> None:
        questions = [
            {"id": f"fixture/{index}", "problem": f"Problem {index}"}
            for index in range(500)
        ]
        payload = canonical_jsonl(questions)
        lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        lock["outputs"]["questions"]["bytes"] = len(payload)
        lock["outputs"]["questions"]["sha256"] = hashlib.sha256(payload).hexdigest()

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            questions_path = root / lock["outputs"]["questions"]["path"]
            lock_path = root / "math500.lock.json"
            questions_path.parent.mkdir(parents=True)
            questions_path.write_bytes(payload + b"tampered\n")
            lock_path.write_text(json.dumps(lock), encoding="utf-8")

            with self.assertRaisesRegex(
                DatasetPreparationError,
                "checksum/size mismatch",
            ):
                verify_locked_questions(root, lock_path)

    def test_explicit_bad_mirror_is_not_hidden_by_a_valid_cache(self) -> None:
        cached_payload = b"locked source bytes\n"
        lock = {
            "dataset": {"source_url": "https://invalid.example/source.jsonl"},
            "source": {
                "path": "data/raw/source.jsonl",
                "bytes": len(cached_payload),
                "sha256": hashlib.sha256(cached_payload).hexdigest(),
            },
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            cached_path = root / "data/raw/source.jsonl"
            mirror_path = root / "bad-mirror.jsonl"
            cached_path.parent.mkdir(parents=True)
            cached_path.write_bytes(cached_payload)
            mirror_path.write_bytes(b"not the locked source\n")

            with self.assertRaisesRegex(DatasetPreparationError, "does not match"):
                acquire_source(lock, root, source_file=mirror_path)

    def test_source_validation_rejects_duplicate_ids(self) -> None:
        rows = [json.loads(line) for line in FIXTURE.read_text().splitlines()]
        rows[1]["unique_id"] = rows[0]["unique_id"]
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "duplicates.jsonl"
            path.write_bytes(canonical_jsonl(rows))
            with self.assertRaisesRegex(DatasetPreparationError, "Duplicate unique_id"):
                read_source_rows(path, expected_rows=2)

    def test_source_validation_requires_exact_schema(self) -> None:
        row = json.loads(FIXTURE.read_text().splitlines()[0])
        row["unexpected"] = "field"
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "extra-key.jsonl"
            path.write_bytes(canonical_jsonl([row]))
            with self.assertRaisesRegex(DatasetPreparationError, "has keys"):
                read_source_rows(path, expected_rows=1)

    def test_canonical_serialization_is_byte_deterministic(self) -> None:
        rows = read_source_rows(FIXTURE, expected_rows=2)
        questions, labels = build_views(rows)
        first = canonical_jsonl(questions)
        second = canonical_jsonl(questions)
        self.assertEqual(first, second)
        self.assertTrue(first.endswith(b"\n"))
        self.assertNotEqual(
            hashlib.sha256(first).hexdigest(),
            hashlib.sha256(canonical_jsonl(labels)).hexdigest(),
        )

    def test_mismatched_artifact_is_not_silently_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "questions.jsonl"
            path.write_bytes(b"tampered\n")
            with self.assertRaisesRegex(DatasetPreparationError, "Refusing to overwrite"):
                _preflight_destination(path, b"expected\n", force=False)
            _preflight_destination(path, b"expected\n", force=True)

    def test_local_full_snapshot_matches_lock_when_present(self) -> None:
        raw_path = REPOSITORY_ROOT / "data/raw/math500-test.jsonl"
        if not raw_path.exists():
            self.skipTest("Run scripts/prepare_math500.py for the local integration check")
        summary = verify_dataset(REPOSITORY_ROOT, LOCK_PATH)
        self.assertEqual(summary["rows"], 500)
        locked_questions = load_locked_questions(
            REPOSITORY_ROOT / "data/math500/questions.jsonl",
            lock_path=LOCK_PATH,
        )
        self.assertEqual(len(locked_questions), 500)


if __name__ == "__main__":
    unittest.main()
