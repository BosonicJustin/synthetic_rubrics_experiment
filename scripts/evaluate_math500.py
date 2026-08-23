#!/usr/bin/env python3
"""Model-free entry point for raw and synthesis evaluation infrastructure."""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from compute_as_a_teacher.evaluation.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
