"""Focused tests for natural-India evaluation bookkeeping."""

import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, "experiments")
sys.path.insert(0, "solution/business_entity_resolution/src")
from diagnostics import candidate_diagnostic, summarize_retrieval
from eval_natural import ensure_index, split_sample
from train import calculate_macro_f05


class NaturalEvaluationTests(unittest.TestCase):
    def test_retrieval_and_oracle_use_only_ground_truth_in_candidates(self):
        record = {"name": "Acme", "core_tokens": ["acme"], "norm_addr": "", "country": "India"}
        items = [("S2-1", 5), ("S2-3", 3)]
        row = candidate_diagnostic("S1-1", record, {"S2-1", "S2-2"}, items, {"S2-1": ["exact"]})
        self.assertEqual(row["recovered_gt_count"], 1)
        self.assertEqual(row["missing_gt_count"], 1)
        self.assertEqual(row["gt_min_rank"], 1)
        self.assertEqual(row["entity_full_recall"], 0)
        summary = summarize_retrieval([row])
        self.assertEqual(summary["candidate_pair_recall"], 0.5)
        self.assertEqual(summary["entity_full_recall"], 0.0)
        oracle = {"S1-1": {"S2-1"}}
        self.assertAlmostEqual(calculate_macro_f05({"S1-1": {"S2-1", "S2-2"}}, oracle), 5 / 6)

    def test_confirmation_sample_is_disjoint_from_development(self):
        rows = [(f"S1-{i}", "Name", "", "India") for i in range(2001)]
        with patch("eval_natural.source_rows", return_value=iter(rows)), \
             patch("pathlib.Path.open") as opened, \
             patch("eval_natural.json.load", return_value=[f"S1-{i}" for i in range(2001)]), \
             patch("eval_natural.prepare_source_record", side_effect=lambda n, a, c: {"name": n}):
            opened.return_value.__enter__.return_value = opened.return_value
            development = split_sample("development")
        with patch("eval_natural.source_rows", return_value=iter(rows)), \
             patch("pathlib.Path.open") as opened, \
             patch("eval_natural.json.load", return_value=[f"S1-{i}" for i in range(2001)]), \
             patch("eval_natural.prepare_source_record", side_effect=lambda n, a, c: {"name": n}):
            opened.return_value.__enter__.return_value = opened.return_value
            confirmation = split_sample("confirmation")
        self.assertEqual(len(development), 1000)
        self.assertEqual(len(confirmation), 1000)
        self.assertFalse({sid for sid, _ in development} & {sid for sid, _ in confirmation})

    def test_source_fingerprint_failure_closes_building_index(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from unittest.mock import MagicMock

        with TemporaryDirectory() as temp_dir:
            index = MagicMock()
            index.set_source_fingerprint.side_effect = ValueError("source changed")
            with patch("eval_natural.DiskCountryBlockingIndex", return_value=index):
                with self.assertRaisesRegex(ValueError, "source changed"):
                    ensure_index(Path(temp_dir) / "index.sqlite")
            index.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
