import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from views import soundex, dm_primary, extract_phones, compact_alnum, transliterate_text, has_indic
from blocking import CountryBlockingIndex, assess_difficulty, _trigrams
from features2 import extra_for_pair, EXTRA_NAMES
from graph import bridge_support
from train import calculate_macro_f05


def test_soundex():
    assert soundex("Smith") == soundex("Smyth")
    assert soundex("") == ""
    assert len(soundex("abc")) == 4


def test_dm():
    assert dm_primary("Smith") == dm_primary("Smyth")
    assert dm_primary("") == ""


def test_phones():
    ph = extract_phones("Call 9876543210 or +91 80 4123 4567")
    assert "9876543210" in ph
    assert any(len(p) >= 7 for p in ph)
    assert extract_phones("") == set()
    assert extract_phones("no digits here") == set()


def test_compact():
    assert compact_alnum("Tata-Consultancy! Services") == "tataconsultancyservices"


def test_indic_translit():
    assert has_indic("శ్రీ వెంకటేశ్వర") is True
    assert has_indic("Sri Venkateshwara") is False
    tr = transliterate_text("శ్రీ")
    assert tr and all(ord(c) < 128 for c in tr)


def test_trigrams():
    assert _trigrams("abcde") == {"abc", "bcd", "cde"}
    assert _trigrams("") == set()


def test_blocking_provenance_and_adaptive():
    idx = CountryBlockingIndex("US")
    idx.add_target_record("S2-1", "tata consultancy services", ["tata", "consultancy", "services"],
                          "road street", [], set(), is_s2=1, raw_addr="123 Main St", raw_name="Tata Consultancy Services")
    idx.add_target_record("S2-2", "infosys limited", ["infosys"],
                          "park avenue", [], set(), is_s2=0, raw_addr="5 Park Ave", raw_name="Infosys")
    idx.prune_frequent_keys()
    cands, prov = idx.query_candidates("tata consultancy services", ["tata", "consultancy", "services"],
                                       "road street", [], s1_raw_addr="123 Main St",
                                       s1_raw_name="Tata Consultancy Services",
                                       max_candidates=10, return_weights=False, return_provenance=True)
    assert "S2-1" in cands
    assert "exact" in prov["S2-1"]
    c2 = idx.query_candidates("tata consultancy services", ["tata", "consultancy", "services"],
                              "road street", [], s1_raw_addr="123 Main St", s1_raw_name="Tata",
                              max_candidates=60, return_weights=True, return_provenance=False, adaptive=True)
    assert len(c2) <= 250
    assert assess_difficulty("x", [], "", 0, 0.0) == "hard"


def test_empty_blocking_query():
    idx = CountryBlockingIndex("US")
    assert idx.query_candidates("", [], "", [], return_weights=True, return_provenance=True) == ([], {})


def test_extra_dims():
    s1 = {"norm_name": "tata consultancy", "norm_addr": "road", "core_tokens": ["tata", "consultancy"],
          "name": "Tata Consultancy", "addr": "Road", "digits": set()}
    v = extra_for_pair(s1, "tata consultancy", "road", ["tata", "consultancy"], channels=("exact", "token"))
    assert len(v) == len(EXTRA_NAMES) == 22
    assert v[14] == 2.0  # agree_count


def test_bridge_support():
    ns, ad = bridge_support("tata consultancy", "main road", "tata consultancy", "main road")
    assert ns == 100.0 and ad == 100.0
    ns2, _ = bridge_support("abc", "x", "xyz completely different", "y")
    assert ns2 < 100.0


def test_metric():
    gt = {"a": {"X"}, "b": set()}
    assert calculate_macro_f05(gt, {"a": {"X"}, "b": set()}) == 1.0
    assert calculate_macro_f05(gt, {"a": {"X"}, "b": {"Y"}}) == 0.5
