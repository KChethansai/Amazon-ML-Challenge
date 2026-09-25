import re
import unicodedata

PUNCT_RE = re.compile(r"[^\w\s]")
WS_RE = re.compile(r"\s+")
DIGIT_RE = re.compile(r"\b\d+[\w/-]*\b")
NUM_ONLY_RE = re.compile(r"\d+")

LEGAL_SUFFIXES = {
    "inc", "incorporated", "llc", "corp", "corporation",
    "ltd", "limited", "pvt", "private", "co", "company",
    "llp", "gmbh", "sarl", "sas", "sa", "plc", "spa", "bv"
}

ADDR_ABBREVIATIONS = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "dr": "drive",
    "blvd": "boulevard",
    "ln": "lane",
    "ct": "court",
    "hwy": "highway",
    "pk": "park",
    "pkwy": "parkway",
    "apt": "apartment",
    "ste": "suite",
    "bldg": "building",
    "fl": "floor",
    "pl": "place",
    "cir": "circle",
    "sq": "square",
    "terr": "terrace"
}

STOPWORDS = {
    "and", "the", "of", "in", "for", "to", "at", "by", "from", "&", "a", "an",
    "limited", "private", "llc", "inc", "ltd", "pvt", "corp", "co", "company"
}

def clean_unicode_to_ascii(text: str) -> str:
    """Normalize unicode characters, decompose accents (e.g. French e/e/c to e/e/c)."""
    if not text:
        return ""
    # NFKD decomposes accented characters into base char + combining mark
    decomposed = unicodedata.normalize("NFKD", text)
    # Strip combining marks (accents)
    ascii_text = "".join(c for c in decomposed if not unicodedata.combining(c))
    return ascii_text

def normalize_name(name: str) -> str:
    """Clean business name for matching: strips handles, domains, legal suffixes."""
    if not name:
        return ""
    s = clean_unicode_to_ascii(name).lower()
    
    # Strip web prefixes/handles: @handle, www., .com, .in, .org, .net
    s = re.sub(r"^[@*#\s]+", "", s)
    s = re.sub(r"^www\.", "", s)
    s = re.sub(r"\.(com|in|org|net|co|io|fr|us)\b", "", s)
    
    # Replace punctuation with whitespace
    s = PUNCT_RE.sub(" ", s)
    s = WS_RE.sub(" ", s).strip()
    return s

def extract_name_tokens(clean_name: str):
    """Extract significant name tokens, separating core tokens from legal suffixes."""
    tokens = clean_name.split()
    core_tokens = []
    suffix_tokens = []
    for t in tokens:
        if t in LEGAL_SUFFIXES:
            suffix_tokens.append(t)
        elif t not in STOPWORDS and len(t) >= 2:
            core_tokens.append(t)
    return core_tokens, suffix_tokens

def normalize_address(addr: str) -> str:
    """Clean and standardize address tokens and abbreviations."""
    if not addr:
        return ""
    s = clean_unicode_to_ascii(addr).lower()
    s = PUNCT_RE.sub(" ", s)
    tokens = s.split()
    expanded = [ADDR_ABBREVIATIONS.get(t, t) for t in tokens]
    return " ".join(expanded)

def extract_address_digits(addr: str):
    """Extract distinct numeric/alphanumeric tokens representing street/PIN numbers."""
    if not addr:
        return set()
    nums = NUM_ONLY_RE.findall(addr)
    return set(nums)

COMMON_ADDR_STOPWORDS = {
    "road", "street", "avenue", "drive", "lane", "boulevard", "floor", "suite", 
    "apartment", "unit", "ste", "apt", "bldg", "building", "fl", "pl", "circle",
    "square", "near", "opp", "opposite", "behind", "beside", "at", "post", "po", "dist",
    "door", "no", "plot", "block", "phase", "sector", "stage"
}

def extract_address_blocking_keys(addr: str, norm_addr: str):
    """Extract compound (street_num, postal) and (street_num, locality) keys for high-precision blocking."""
    if not addr:
        return []
    nums = NUM_ONLY_RE.findall(addr)
    tokens = [t for t in norm_addr.split() if len(t) >= 4 and t not in COMMON_ADDR_STOPWORDS]
    keys = []
    
    postals = [n for n in nums if len(n) in (5, 6)]
    street_nums = [n for n in nums if 1 <= len(n) <= 5]
    
    # 1. Compound (street_num, postal_code)
    for snum in street_nums[:2]:
        for p in postals[:2]:
            keys.append(("ST_POST", snum, p))
            
    # 2. Compound (street_num, locality) - first 2 tokens and last 2 tokens (city/state)
    locality_candidates = tokens[:2] + tokens[-2:] if len(tokens) > 3 else tokens
    for snum in street_nums[:2]:
        for tok in set(locality_candidates):
            keys.append(("ST_LOC", snum, tok))
            
    # 3. Individual postal code for sparse queries
    for p in postals[:1]:
        keys.append(("POSTAL", p))
        
    return keys
