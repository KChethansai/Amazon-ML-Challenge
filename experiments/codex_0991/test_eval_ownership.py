"""Small invariants for the development-only ownership policy."""

import unittest

from eval_ownership import assign_ownership


class OwnershipTests(unittest.TestCase):
    def test_equal_scores_choose_stable_s1(self):
        scored = {"S1-b": [("S2-x", 0.9)], "S1-a": [("S2-x", 0.9)]}
        baseline = {sid: {"S2-x"} for sid in scored}
        owned, counts = assign_ownership(scored, baseline)
        self.assertEqual(owned, {"S1-b": set(), "S1-a": {"S2-x"}})
        self.assertEqual(counts["targets_with_multiple_baseline_owners"], 1)

    def test_reject_if_best_competitor_did_not_accept(self):
        scored = {"S1-a": [("S2-x", 0.95)], "S1-b": [("S2-x", 0.8)]}
        baseline = {"S1-a": set(), "S1-b": {"S2-x"}}
        owned, counts = assign_ownership(scored, baseline)
        self.assertEqual(owned, {"S1-a": set(), "S1-b": set()})
        self.assertEqual(counts["rejected_because_best_owner_did_not_accept"], 1)


if __name__ == "__main__":
    unittest.main()
