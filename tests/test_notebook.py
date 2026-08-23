from __future__ import annotations

import json
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPOSITORY_ROOT / "notebooks/math500_explorer.ipynb"


class NotebookSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

    def test_notebook_is_valid_unsaved_output_free_json(self) -> None:
        self.assertEqual(self.notebook["nbformat"], 4)
        code_cells = [
            cell for cell in self.notebook["cells"] if cell["cell_type"] == "code"
        ]
        self.assertTrue(code_cells)
        self.assertTrue(all(cell["execution_count"] is None for cell in code_cells))
        self.assertTrue(all(cell["outputs"] == [] for cell in code_cells))

    def test_reference_reveal_defaults_to_off(self) -> None:
        source = "".join(
            line
            for cell in self.notebook["cells"]
            if cell["cell_type"] == "code"
            for line in cell["source"]
        )
        self.assertIn("REVEAL_REFERENCE = False", source)
        self.assertNotIn("REVEAL_REFERENCE = True\n", source)


if __name__ == "__main__":
    unittest.main()
