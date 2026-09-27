"""Small end-to-end check for disk-backed output ordering and subset rules."""

import sys
import tempfile
import unittest
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from inference import run_inference


class DiskInferenceTests(unittest.TestCase):
    def test_order_empty_and_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "test"
            output = root / "output"
            data.mkdir()
            header = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            (data / "test_source1.tsv").write_text(
                header + "S1-1\tAcme Services\t24 MG Road 560001\tIndia\n"
                "S1-2\tNo Candidate\tNowhere\tFrance\n"
                "S1-3\tCafe Bon\t1 Rue 75001\tFrance\n", encoding="utf-8")
            (data / "test_source2.tsv").write_text(
                header + "S2-1\tAcme Services\t24 MG Road 560001\tIndia\n"
                "S2-2\tCafe Bon\t1 Rue 75001\tFrance\n", encoding="utf-8")
            (data / "test_source3.tsv").write_text(header, encoding="utf-8")
            models = Path(__file__).parents[1] / "models"
            scores_path = root / "accepted_scores.tsv"
            run_inference(str(data), str(models), str(output), str(scores_path))
            candidates = (output / "candidate_pairs.tsv").read_text(encoding="utf-8").splitlines()[1:]
            matches = (output / "matching_results.tsv").read_text(encoding="utf-8").splitlines()[1:]
            self.assertEqual([row.partition("\t")[0] for row in candidates], ["S1-1", "S1-2", "S1-3"])
            self.assertEqual([row.partition("\t")[0] for row in matches], ["S1-1", "S1-2", "S1-3"])
            self.assertEqual(candidates[1], "S1-2\t")
            self.assertEqual(matches[1], "S1-2\t")
            for candidate_row, match_row in zip(candidates, matches):
                candidate_ids = set(candidate_row.partition("\t")[2].split(",")) - {""}
                match_ids = set(match_row.partition("\t")[2].split(",")) - {""}
                self.assertTrue(match_ids <= candidate_ids)
            with scores_path.open(newline="", encoding="utf-8") as file:
                scored = list(csv.DictReader(file, delimiter="\t"))
            accepted = {(row.partition("\t")[0], target)
                        for row in matches for target in row.partition("\t")[2].split(",") if target}
            self.assertEqual({(row["candidate_s1_id"], row["target_id"]) for row in scored}, accepted)
            self.assertTrue(all(0 <= float(row["model_probability"]) <= 1 for row in scored))
            self.assertTrue(all(int(row["candidate_rank"]) >= 1 for row in scored))
            self.assertTrue(all(float(row["retrieval_score"]) >= 0 for row in scored))


if __name__ == "__main__":
    unittest.main()
