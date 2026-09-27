"""Small focused checks for the bounded submission validator."""

import importlib.util
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location(
    "validate_submission_fast", Path(__file__).with_name("validate_submission_fast.py"))
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


class StreamingValidatorTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.matching = self.root / "matching_results.tsv"
        self.candidate = self.root / "candidate_pairs.tsv"
        (self.root / "test_source1.tsv").write_text("id\tname\nS1-1\ta\nS1-2\tb\n")
        (self.root / "test_source2.tsv").write_text("id\tname\nS2-1\ta\nS2-2\tb\n")
        (self.root / "test_source3.tsv").write_text("id\tname\nS3-1\tc\n")
        self.matching.write_text("source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1\nS1-2\t\n")
        self.candidate.write_text("source1_entity_id\tcandidate_entity_ids\nS1-1\tS2-1,S3-1\nS1-2\tS2-2\n")

    def check(self):
        return VALIDATOR.validate(str(self.matching), str(self.candidate),
                                  str(self.root), check_ids=True, progress_interval=0)[0]

    def test_valid(self):
        self.assertEqual(self.check(), [])

    def test_subset_failure(self):
        self.candidate.write_text("source1_entity_id\tcandidate_entity_ids\nS1-1\tS3-1\nS1-2\tS2-2\n")
        self.assertTrue(any("absent from candidate_pairs" in e for e in self.check()))

    def test_order_failure(self):
        self.matching.write_text("source1_entity_id\tmatched_entity_ids\nS1-2\t\nS1-1\tS2-1\n")
        self.assertTrue(any("order/coverage mismatch" in e for e in self.check()))

    def test_unknown_and_duplicate(self):
        self.candidate.write_text("source1_entity_id\tcandidate_entity_ids\nS1-1\tS2-1,S2-1,S2-404\nS1-2\tS2-2\n")
        errors = self.check()
        self.assertTrue(any("repeated ID" in e for e in errors))
        self.assertTrue(any("absent from test" in e for e in errors))

    def test_missing_row(self):
        self.matching.write_text("source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1\n")
        self.assertTrue(any("Missing matching row" in e for e in self.check()))


if __name__ == "__main__":
    unittest.main()
