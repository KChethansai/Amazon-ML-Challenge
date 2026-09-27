import tempfile
import unittest
from pathlib import Path

from resolve_target_ownership import resolve, select_owner


class OwnershipTests(unittest.TestCase):
    def test_probability_tie_is_deterministic(self):
        self.assertEqual(select_owner([("S1-b", 0.9), ("S1-a", 0.9)], 0.8, 0)[0], "S1-a")
        self.assertEqual(select_owner([("S1-b", 0.9), ("S1-a", 0.9)], 0.8, 0.02)[1], "margin")
        self.assertEqual(select_owner([("S1-a", 0.79)], 0.8, 0.2)[1], "probability")

    def test_full_row_coverage_and_unique_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matching, scores, output = (root / name for name in ("matching.tsv", "scores.tsv", "resolved.tsv"))
            matching.write_text("source1_entity_id\tmatched_entity_ids\n"
                                "S1-a\tS2-x,S2-y\nS1-b\tS2-x,S3-z\nS1-c\t\n")
            scores.write_text("target_id\tcandidate_s1_id\tmodel_probability\tretrieval_score\tcandidate_rank\n"
                              "S2-x\tS1-a\t0.96\t12\t1\nS2-y\tS1-a\t0.98\t11\t2\n"
                              "S2-x\tS1-b\t0.94\t10\t1\nS3-z\tS1-b\t0.93\t9\t2\n")
            stats = resolve(matching, scores, output, 0.9, 0.05)
            self.assertEqual(output.read_text().splitlines(),
                             ["source1_entity_id\tmatched_entity_ids", "S1-a\tS2-y", "S1-b\tS3-z", "S1-c\t"])
            self.assertEqual(stats["conflicted_targets_before"], 1)
            self.assertEqual(stats["rejected_margin"], 1)
            self.assertEqual(stats["cross_s1_duplicate_assignments"], 0)
            self.assertEqual(stats["s1_rows"], 3)

    def test_score_stream_must_match_accepted_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matching, scores, output = (root / name for name in ("matching.tsv", "scores.tsv", "resolved.tsv"))
            matching.write_text("source1_entity_id\tmatched_entity_ids\nS1-a\tS2-x\n")
            scores.write_text("target_id\tcandidate_s1_id\tmodel_probability\nS2-y\tS1-a\t0.99\n")
            with self.assertRaisesRegex(ValueError, "lacks model probability"):
                resolve(matching, scores, output, 0.9, 0.05)
            self.assertFalse(output.exists())

    def test_inputs_and_existing_output_are_protected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matching, scores, output = (root / name for name in ("matching.tsv", "scores.tsv", "resolved.tsv"))
            matching.write_text("source1_entity_id\tmatched_entity_ids\nS1-a\t\n")
            scores.write_text("target_id\tcandidate_s1_id\tmodel_probability\n")
            with self.assertRaisesRegex(ValueError, "must differ"):
                resolve(matching, scores, scores, 0.9, 0.05)
            output.write_text("protected")
            with self.assertRaises(FileExistsError):
                resolve(matching, scores, output, 0.9, 0.05)
            self.assertEqual(output.read_text(), "protected")


if __name__ == "__main__":
    unittest.main()
