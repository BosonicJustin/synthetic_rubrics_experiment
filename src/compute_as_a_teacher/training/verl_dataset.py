"""Local JSONL dataset adapter for the pinned verl release."""

from __future__ import annotations

import json
from pathlib import Path

import datasets
from verl.utils.dataset.rl_dataset import RLHFDataset


class JsonlRLHFDataset(RLHFDataset):
    def _download(self, use_origin_parquet: bool = False) -> None:
        source = self.original_data_files if use_origin_parquet else self.data_files
        paths = [Path(item).resolve() for item in source]
        if any(not path.is_file() for path in paths):
            raise FileNotFoundError("CaT training JSONL is missing")
        self.data_files = [str(path) for path in paths]

    def _read_files_and_tokenize(self) -> None:
        rows = []
        for filename in self.data_files:
            with Path(filename).open(encoding="utf-8") as handle:
                rows.extend(json.loads(line) for line in handle if line.strip())
        if not rows:
            raise ValueError("CaT training JSONL is empty")
        self.dataframe = datasets.Dataset.from_list(rows)
        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)


__all__ = ["JsonlRLHFDataset"]
