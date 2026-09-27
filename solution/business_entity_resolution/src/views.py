"""Multi-view representations for names/addresses. Raw is never destroyed.

Covers: transliterated (Indic scripts -> latin), ascii, compact, initials,
acronym, phonetic tokens (double-metaphone primary + soundex), digit/phone
tokens. Deterministic; stdlib + indic_transliteration + Metaphone only.
"""
import re
import unicodedata
from functools import lru_cache

try:
    from indic_transliteration import sanscript
    _HAS_INDIC = True
except Exception:
    sanscript = None
    _HAS_INDIC = False

try:
    from metaphone import doublemetaphone
    _HAS_DM = True
except Exception:
    doublemetaphone = None
    _HAS_DM = False

WS_RE = re.compile(r"\s+")
ALNUM_RE = re.compile(r"[^a-z0-9]")
PHONE_SEQ_RE = re.compile(r"\+?\d[\d\s\-()]{6,}\d")

# Unicode block -> sanscript scheme for Indic scripts seen in India data.
_BLOCK_SCHEMES = [
    (0x0900, 0x097F, "devanagari"),
    (0x0980, 0x09FF, "bengali"),
    (0x0A00, 0x0A7F, "gurmukhi"),
    (0x0A80, 0x0AFF, "gujarati"),
    (0x0B00, 0x0B7F, "oriya"),
    (0x0B80, 0x0BFF, "tamil"),
    (0x0C00, 0x0C7F, "telugu"),
    (0x0C80, 0x0CFF, "kannada"),
    (0x0D00, 0x0D7F, "malayalam"),
]


def _scheme_for_char(ch):
    o = ord(ch)
    for lo, hi, scheme in _BLOCK_SCHEMES:
        if lo <= o <= hi:
            return scheme
    return None


def has_indic(text):
    if not text:
        return False
    return any(_scheme_for_char(c) for c in text if ord(c) > 127)


@lru_cache(maxsize=200000)
def transliterate_token(tok):
    """Transliterate one token from its native Indic block to latin (itrans)."""
    if not _HAS_INDIC or not tok:
        return ""
    # group chars by scheme; transliterate per contiguous run
    out = []
    run = ""
    run_scheme = None
    for ch in tok:
        s = _scheme_for_char(ch)
        if s is None:
            if run:
                try:
                    out.append(sanscript.transliterate(run, run_scheme, sanscript.ITRANS))
                except Exception:
                    out.append(run)
                run, run_scheme = "", None
            out.append(ch)
        else:
            if run_scheme is None:
                run_scheme = s
            if s != run_scheme:
                try:
                    out.append(sanscript.transliterate(run, run_scheme, sanscript.ITRANS))
                except Exception:
                    out.append(run)
                run, run_scheme = "", s
            run += ch
    if run:
        try:
            out.append(sanscript.transliterate(run, run_scheme, sanscript.ITRANS))
        except Exception:
            out.append(run)
    return "".join(out).lower()


def transliterate_text(text):
    """Latin view of text with Indic-script tokens transliterated."""
    if not text or not has_indic(text):
        return ""
    return " ".join(transliterate_token(t) for t in text.split())


def ascii_fold(text):
    if not text:
        return ""
    d = unicodedata.normalize("NFKD", text)
    return "".join(c for c in d if not unicodedata.combining(c)).lower()


@lru_cache(maxsize=200000)
def soundex(tok):
    tok = re.sub(r"[^a-z]", "", (tok or "").lower())
    if not tok:
        return ""
    first = tok[0].upper()
    mapping = {"b": "1", "f": "1", "p": "1", "v": "1", "c": "2", "g": "2",
               "j": "2", "k": "2", "q": "2", "s": "2", "x": "2", "z": "2",
               "d": "3", "t": "3", "l": "4", "m": "5", "n": "5", "r": "6"}
    coded = [mapping.get(c, "") for c in tok[1:]]
    # collapse repeats
    prev, digits = "", []
    for d in coded:
        if d != prev:
            if d:
                digits.append(d)
            prev = d
    return (first + "".join(digits))[:4].ljust(4, "0")


@lru_cache(maxsize=200000)
def dm_primary(tok):
    tok = re.sub(r"[^a-z]", "", (tok or "").lower())
    if not tok or not _HAS_DM:
        return ""
    try:
        p, _ = doublemetaphone(tok)
        return (p or "").lower()
    except Exception:
        return ""


def compact_alnum(text):
    return ALNUM_RE.sub("", (text or "").lower())


def extract_phones(raw):
    """Normalized phone-ish sequences (7-13 digits, optional leading +)."""
    if not raw:
        return set()
    out = set()
    for m in PHONE_SEQ_RE.findall(raw):
        d = re.sub(r"\D", "", m)
        if 7 <= len(d) <= 13:
            out.add(d)
            if len(d) > 10 and d.startswith("91") and len(d) == 12:
                out.add(d[2:])  # India country-code variant
            if len(d) > 10 and d.startswith("1") and len(d) == 11:
                out.add(d[1:])  # NANP country-code variant
    return out


def name_views(raw_name, norm_name, core_tokens):
    tr = transliterate_text(raw_name or "")
    tr_tokens = tr.split() if tr else []
    phon = [dm_primary(t) for t in core_tokens]
    return {
        "norm": norm_name,
        "translit": re.sub(r"[^a-z0-9 ]", " ", tr).strip(),
        "translit_tokens": [t for t in tr_tokens if len(t) >= 2],
        "compact": compact_alnum(norm_name),
        "initials": "".join(t[0] for t in core_tokens if t),
        "phonetic": [p for p in phon if p],
        "soundex": [soundex(t) for t in core_tokens if len(t) >= 3],
    }


def address_views(raw_addr, norm_addr):
    tr = transliterate_text(raw_addr or "")
    return {
        "norm": norm_addr,
        "translit": re.sub(r"[^a-z0-9 ]", " ", tr).strip() if tr else "",
        "compact": compact_alnum(norm_addr),
        "phones": extract_phones(raw_addr),
    }
