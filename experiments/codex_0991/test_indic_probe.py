import unittest

from indic_probe import learn_mapping, reverse_variants


class IndicProbeTest(unittest.TestCase):
    def test_support_and_ambiguity(self):
        pairs = [
            ("laxmi traders", "lakshmi traders"),
            ("laxmi stores", "lakshmi stores"),
            ("laxmi foods", "lakshmi foods"),
            ("shakti tools", "sakthi tools"),
            ("wrong length", "wrong"),
        ]
        self.assertEqual(learn_mapping(pairs, min_count=3, min_share=0.6), {"laxmi": "lakshmi"})

    def test_tie_is_rejected_and_reverse_is_deterministic(self):
        pairs = [("raja", "raja")]*3 + [("nava", "nav")]*3 + [("nava", "nova")]*3
        self.assertEqual(learn_mapping(pairs, min_count=3, min_share=0.5), {})
        self.assertEqual(reverse_variants({"laxmi": "lakshmi", "laksmi": "lakshmi"}),
                         {"lakshmi": ("laksmi", "laxmi")})


if __name__ == "__main__":
    unittest.main()
