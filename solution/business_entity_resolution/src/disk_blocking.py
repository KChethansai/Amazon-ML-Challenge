"""SQLite-backed CountryBlockingIndex with bounded Python state."""
import json
import math
import os
import sqlite3
import hashlib
import re
import fcntl
from collections.abc import Mapping

try:
    from packed_postings import PackedPostings
except ImportError:
    from solution.business_entity_resolution.src.packed_postings import PackedPostings

try:
    from views import transliterate_text, dm_primary, extract_phones
except ImportError:
    from solution.business_entity_resolution.src.views import transliterate_text, dm_primary, extract_phones

try:
    from blocking import (CountryBlockingIndex, MAX_POSTING_LIST_LEN,
                          MAX_NGRAM_POSTING_LEN, MAX_ADDR_TOKEN_POSTING_LEN,
                          MAX_PHON_POSTING_LEN, MAX_TRLIT_POSTING_LEN,
                          _trigrams, extract_acronyms, extract_postal_prefixes,
                          assess_difficulty, ADAPTIVE_BUDGETS)
except ImportError:
    from solution.business_entity_resolution.src.blocking import (
        CountryBlockingIndex, MAX_POSTING_LIST_LEN, MAX_NGRAM_POSTING_LEN,
        MAX_ADDR_TOKEN_POSTING_LEN, MAX_PHON_POSTING_LEN, MAX_TRLIT_POSTING_LEN,
                          _trigrams, extract_acronyms, extract_postal_prefixes,
                          assess_difficulty, ADAPTIVE_BUDGETS)

try:
    from normalization import COMMON_ADDR_STOPWORDS, NUM_ONLY_RE
except ImportError:
    from solution.business_entity_resolution.src.normalization import COMMON_ADDR_STOPWORDS, NUM_ONLY_RE

_CHANNELS = {
    'exact_name_idx': 'exact', 'compound_addr_idx': 'addr',
    'token_idx': 'token', 'prefix_idx': 'prefix', 'ngram_idx': 'ngram',
    'acronym_idx': 'acronym', 'tok_post_idx': 'tokpost',
    'addr_token_idx': 'addr_tok', 'tr_name_idx': 'tr_name',
    'tr_token_idx': 'tr_tok', 'phon_idx': 'phon', 'phone_idx': 'phone',
}
_LIMITS = {
    'token': 40000, 'prefix': MAX_POSTING_LIST_LEN,
    'acronym': MAX_POSTING_LIST_LEN, 'tokpost': MAX_POSTING_LIST_LEN,
    'tr_tok': MAX_TRLIT_POSTING_LEN, 'tr_name': MAX_TRLIT_POSTING_LEN,
    'phon': MAX_PHON_POSTING_LEN, 'addr': MAX_POSTING_LIST_LEN,
    'ngram': MAX_NGRAM_POSTING_LEN, 'addr_tok': 10000,
}


def _key(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))



_CHANNEL_BITS = {name: 1 << i for i, name in enumerate(
    ('exact','addr','token','prefix','acronym','tokpost','addr_tok','ngram',
     'tr_name','tr_tok','phon','phone'))}


class _SpillScores:
    """Exact Counter semantics with a capped Python cache and SQLite spill."""
    LIMIT = 50000

    def __init__(self, db):
        self.db = db
        self.cache = {}
        self.next_order = 0
        self.total = 0
        self.spilled = False
        db.execute('CREATE TEMP TABLE IF NOT EXISTS query_scores('
                   'rid INTEGER PRIMARY KEY, score REAL, first_seen INTEGER, mask INTEGER)')
        db.execute('DELETE FROM query_scores')

    def _entry(self, eid):
        row = self.cache.get(eid)
        if row is not None:
            return row
        if self.spilled:
            old = self.db.execute('SELECT score,first_seen,mask FROM query_scores WHERE rid=?',
                                  (eid,)).fetchone()
            if old is not None:
                row = list(old)
                self.cache[eid] = row
                return row
        row = [0.0, self.next_order, 0]
        self.next_order += 1
        self.total += 1
        self.cache[eid] = row
        return row

    def __getitem__(self, eid):
        return self._entry(eid)[0]

    def __setitem__(self, eid, score):
        self._entry(eid)[0] = score
        if len(self.cache) >= self.LIMIT:
            self._flush()

    def add_channel(self, eid, channel):
        self._entry(eid)[2] |= _CHANNEL_BITS[channel]

    def channels(self, eid):
        row = self.cache.get(eid)
        if row is None and self.spilled:
            row = self.db.execute('SELECT score,first_seen,mask FROM query_scores WHERE rid=?',
                                  (eid,)).fetchone()
        mask = row[2] if row else 0
        return {channel for channel, bit in _CHANNEL_BITS.items() if mask & bit}

    def _flush(self):
        if not self.cache:
            return
        self.db.executemany(
            'INSERT INTO query_scores(rid,score,first_seen,mask) VALUES (?,?,?,?) '
            'ON CONFLICT(rid) DO UPDATE SET score=excluded.score,mask=excluded.mask',
            ((eid, row[0], row[1], row[2]) for eid, row in self.cache.items()))
        self.cache.clear()
        self.spilled = True

    def __len__(self):
        return self.total

    def __bool__(self):
        return self.total > 0

    def values(self):
        if self.spilled:
            self._flush()
            return (row[0] for row in self.db.execute('SELECT score FROM query_scores'))
        return (row[0] for row in self.cache.values())

    def most_common(self, n):
        if self.spilled:
            self._flush()
            return list(self.db.execute(
                'SELECT rid,score FROM query_scores ORDER BY score DESC,first_seen ASC LIMIT ?', (n,)))
        return [(eid, row[0]) for eid, row in
                sorted(self.cache.items(), key=lambda item: (-item[1][0], item[1][1]))[:n]]


class _ChannelSet:
    def __init__(self, scores, eid):
        self.scores, self.eid = scores, eid

    def add(self, channel):
        self.scores.add_channel(self.eid, channel)


class _Provenance:
    def __init__(self, scores):
        self.scores = scores

    def __getitem__(self, eid):
        return _ChannelSet(self.scores, eid)

    def get(self, eid, default=None):
        return self.scores.channels(eid)

class _Posting:
    def __init__(self, owner, channel, key):
        self.owner, self.channel, self.key = owner, channel, key

    def append(self, eid):
        self.owner._pending.append((self.channel, self.key, self.owner._rid))

    def __iter__(self):
        if self.owner.packed is not None:
            return iter(self.owner.packed.lookup(self.channel, self.key))
        return (row[0] for row in self.owner.db.execute(
            'SELECT rid FROM postings WHERE channel=? AND key=? ORDER BY seq',
            (self.channel, self.key)))


class _PostingMap(Mapping):
    def __init__(self, owner, channel):
        self.owner, self.channel = owner, channel

    def __getitem__(self, key):
        k = _key(key)
        if not self.owner._building and not self.__contains__(key):
            raise KeyError(key)
        return _Posting(self.owner, self.channel, k)

    def __contains__(self, key):
        return self.owner.db.execute(
            'SELECT 1 FROM key_counts WHERE channel=? AND key=?',
            (self.channel, _key(key))).fetchone() is not None

    def __iter__(self):
        return (json.loads(row[0]) for row in self.owner.db.execute(
            'SELECT key FROM key_counts WHERE channel=?', (self.channel,)))

    def __len__(self):
        return self.owner.db.execute(
            'SELECT count(*) FROM key_counts WHERE channel=?', (self.channel,)).fetchone()[0]


class _IdfMap(Mapping):
    def __init__(self, owner, channel):
        self.owner, self.channel = owner, channel

    def __getitem__(self, key):
        count = self.owner.db.execute(
            'SELECT n FROM key_counts WHERE channel=? AND key=?',
            (self.channel, _key(key))).fetchone()
        if count is None:
            raise KeyError(key)
        value = math.log((self.owner._n + 1.0) / (count[0] + 1.0))
        if self.channel == 'ngram':
            return value + 1.0
        return min(8.0, max(1.5, value)) if self.channel == 'addr' else min(8.0, value)

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0


class _Records(Mapping):
    def __init__(self, owner):
        self.owner = owner

    def __setitem__(self, eid, value):
        row = self.owner.db.execute(
            'INSERT INTO records(eid, payload) VALUES (?, ?)',
            (eid, json.dumps(value, ensure_ascii=False, separators=(',', ':'))))
        self.owner._rid = row.lastrowid
        self.owner._n += 1

    def __getitem__(self, eid):
        column = 'rid' if isinstance(eid, int) else 'eid'
        row = self.owner.db.execute(f'SELECT payload FROM records WHERE {column}=?', (eid,)).fetchone()
        if row is None:
            raise KeyError(eid)
        data = json.loads(row[0])
        return (data[0], data[1], data[2], data[3], data[4], data[5])

    def get_many(self, eids):
        """Fetch multiple records by rid/eid in chunks of 990."""
        if not eids:
            return {}
        result = {}
        first = next(iter(eids))
        column = 'rid' if isinstance(first, int) else 'eid'
        eids_list = list(eids)
        chunk_size = 990
        for i in range(0, len(eids_list), chunk_size):
            chunk = eids_list[i:i + chunk_size]
            placeholders = ','.join('?' for _ in chunk)
            cursor = self.owner.db.execute(
                f'SELECT {column}, payload FROM records WHERE {column} IN ({placeholders})', chunk)
            for k, payload in cursor:
                data = json.loads(payload)
                result[k] = (data[0], data[1], data[2], data[3], data[4], data[5])
        return result

    def __iter__(self):
        return (row[0] for row in self.owner.db.execute('SELECT eid FROM records'))

    def __len__(self):
        return self.owner._n


class DiskCountryBlockingIndex(CountryBlockingIndex):
    def __init__(self, country, db_path, create=False):
        super().__init__(country)
        self.path = os.fspath(db_path)
        self._building = create
        self._pending = []
        self._rid = 0
        self._failed = False
        self.packed = None
        if create:
            self._lock = open(self.path + '.lock', 'a+b')
            try:
                fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                self._lock.close()
                raise RuntimeError('Index already has an active builder') from exc
            self._build_path = self.path + '.building'
            existing = os.path.exists(self._build_path)
            self.db = sqlite3.connect(self._build_path)
            self.db.execute('PRAGMA journal_mode=DELETE')
            self.db.execute('PRAGMA synchronous=NORMAL')
            self.db.execute('PRAGMA temp_store=FILE')
            self.db.execute('PRAGMA cache_size=-16384')
            if not existing:
                self.db.executescript('CREATE TABLE records(rid INTEGER PRIMARY KEY, eid TEXT UNIQUE, payload TEXT);'
                                      'CREATE TABLE postings(seq INTEGER PRIMARY KEY, channel TEXT, key TEXT, rid INTEGER);'
                                      'CREATE TABLE key_counts(channel TEXT, key TEXT, n INTEGER, '
                                      'PRIMARY KEY(channel,key)) WITHOUT ROWID;'
                                      'CREATE TABLE meta(k TEXT PRIMARY KEY, v TEXT);')
            meta = dict(self.db.execute('SELECT k,v FROM meta'))
            if meta.get('country', country) != country:
                raise ValueError('Build country does not match')
            self.db.execute('INSERT OR IGNORE INTO meta VALUES (?,?)', ('country', country))
            if existing and meta.get('ready') == '1':
                actual = self.db.execute('SELECT count(*) FROM records').fetchone()[0]
                check = self.db.execute('PRAGMA quick_check').fetchone()[0]
                if actual != int(meta.get('records', '-1')) or check != 'ok':
                    raise ValueError('Ready build failed record-count or integrity check')
                self.db.close()
                os.replace(self._build_path, self.path)
                fcntl.flock(self._lock, fcntl.LOCK_UN)
                self._lock.close()
                self.__init__(country, self.path, create=False)
                return
            if existing:
                # Legacy builds could commit in the middle of a target. Rebuild
                # the last record, including all its postings, on every resume.
                last = self.db.execute('SELECT max(rid) FROM records').fetchone()[0]
                if last is not None and int(meta.get('checkpoint_records', '-1')) != last:
                    self.db.execute('DELETE FROM postings WHERE rid=?', (last,))
                    self.db.execute('DELETE FROM records WHERE rid=?', (last,))
            self.db.commit()
            self._n = self.db.execute('SELECT count(*) FROM records').fetchone()[0]
            self._resume = self.db.execute('SELECT eid,payload FROM records ORDER BY rid')
            self._resume_left = self._n
        else:
            self.db = sqlite3.connect('file:' + os.path.abspath(self.path) + '?mode=ro', uri=True)
            self.db.execute('PRAGMA cache_size=-16384')
            self.db.execute('PRAGMA temp_store=FILE')
            meta = dict(self.db.execute('SELECT k,v FROM meta'))
            if meta.get('ready') != '1' or meta.get('country') != country:
                raise ValueError('Index is incomplete or country does not match')
            self._n = int(meta['records'])
            try:
                self.packed = PackedPostings.open_if_ready(self.path)
            except Exception:
                self.db.close()
                raise
        self.records = _Records(self)
        for attr, channel in _CHANNELS.items():
            setattr(self, attr, _PostingMap(self, channel))
        self.compound_idf = _IdfMap(self, 'addr')
        self.ngram_idf = _IdfMap(self, 'ngram')
        self.addr_token_idf = _IdfMap(self, 'addr_tok')
        self._has_indic_phon = self.db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='indic_phon_postings'").fetchone() is not None

    def set_source_fingerprint(self, paths):
        """Reject a resumed build if its source files changed."""
        h = hashlib.sha256()
        for path in paths:
            path = os.path.abspath(os.fspath(path))
            st = os.stat(path)
            h.update(json.dumps((path, st.st_size, st.st_mtime_ns)).encode())
            with open(path, 'rb') as source:
                h.update(source.read(65536))
                if st.st_size > 65536:
                    source.seek(max(0, st.st_size - 65536))
                    h.update(source.read(65536))
        digest = h.hexdigest()
        previous = self.db.execute('SELECT v FROM meta WHERE k=?', ('source_fingerprint',)).fetchone()
        if previous and previous[0] != digest:
            raise ValueError('Index source fingerprint differs from current files')
        if self._building and not previous:
            self.db.execute('INSERT INTO meta VALUES (?,?)', ('source_fingerprint', digest))
            self.db.commit()

    def add_target_record(self, eid, norm_name, core_tokens, norm_addr, addr_keys,
                          digits, is_s2, raw_addr='', raw_name=''):
        if self._failed:
            raise RuntimeError('Cannot append after a failed target record')
        if self._resume_left:
            expected = (norm_name, norm_addr, is_s2, list(core_tokens),
                        raw_name if raw_name and any(ord(c)>127 for c in raw_name) else '',
                        raw_addr if raw_addr and any(ord(c)>127 for c in raw_addr) else '')
            prior = self._resume.fetchone()
            if prior is None or prior[0] != eid or json.loads(prior[1]) != list(expected):
                self._failed = True
                raise ValueError(f'Resumed source diverges at target {eid}')
            self._resume_left -= 1
            return
        # Keep records and postings in one outer transaction. Releasing a
        # savepoint must never commit a record without its postings.
        if not self.db.in_transaction:
            self.db.execute('BEGIN')
        self.db.execute('SAVEPOINT target_record')
        before_pending, before_n, before_rid = len(self._pending), self._n, self._rid
        try:
            super().add_target_record(eid, norm_name, core_tokens, norm_addr,
                                      addr_keys, digits, is_s2, raw_addr, raw_name)
            self.db.execute('RELEASE SAVEPOINT target_record')
        except Exception:
            self.db.execute('ROLLBACK TO SAVEPOINT target_record')
            self.db.execute('RELEASE SAVEPOINT target_record')
            del self._pending[before_pending:]
            self._n, self._rid = before_n, before_rid
            self._failed = True
            raise
        if len(self._pending) >= 20000:
            self._flush()

    def _flush(self):
        if self._failed:
            raise RuntimeError('Cannot commit after a failed target record')
        if self.db.in_transaction or self._pending:
            try:
                if self._pending:
                    self.db.executemany('INSERT INTO postings(channel,key,rid) VALUES (?,?,?)', self._pending)
                self.db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)', ('checkpoint_records', str(self._n)))
                self.db.commit()
            except Exception:
                self.db.rollback()
                self._pending.clear()
                self._n = self.db.execute('SELECT count(*) FROM records').fetchone()[0]
                self._failed = True
                raise
            self._pending.clear()
            self.db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)', ('checkpoint_records', str(self._n)))
            self.db.commit()

    def prune_frequent_keys(self):
        if self._failed:
            raise RuntimeError('Cannot publish a failed build')
        if self._resume_left:
            raise ValueError('Source stream ended before resume checkpoint')
        self._flush()
        self.db.execute('CREATE INDEX IF NOT EXISTS posting_lookup ON postings(channel,key,seq)')
        self.db.execute('DELETE FROM key_counts')
        self.db.execute('INSERT INTO key_counts SELECT channel,key,count(*) FROM postings GROUP BY channel,key')
        for channel, limit in _LIMITS.items():
            self.db.execute('DELETE FROM key_counts WHERE channel=? AND n>?', (channel, limit))
        self.db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)', ('country', self.country))
        self.db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)', ('records', str(self._n)))
        self.db.commit()
        integrity = self.db.execute('PRAGMA quick_check').fetchone()[0]
        records = self.db.execute('SELECT count(*) FROM records').fetchone()[0]
        if integrity != 'ok' or records != self._n:
            raise ValueError(f'Index integrity failure: {integrity}')
        self.db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)', ('ready', '1'))
        self.db.commit()
        self.db.close()
        os.replace(self._build_path, self.path)
        fcntl.flock(self._lock, fcntl.LOCK_UN)
        self._lock.close()
        self.db = sqlite3.connect('file:' + os.path.abspath(self.path) + '?mode=ro', uri=True)
        self.db.execute('PRAGMA cache_size=-16384')
        self.db.execute('PRAGMA temp_store=FILE')
        self._building = False
        try:
            self.packed = PackedPostings.open_if_ready(self.path)
        except Exception:
            self.db.close()
            raise

    def close(self):
        if self._building:
            if self._failed:
                self.db.rollback()
            else:
                self._flush()
        self.db.close()
        if self.packed is not None:
            self.packed.close()
            self.packed = None
        if getattr(self, '_lock', None) and not self._lock.closed:
            fcntl.flock(self._lock, fcntl.LOCK_UN)
            self._lock.close()

    def _eid(self, rid):
        row = self.db.execute('SELECT eid FROM records WHERE rid=?', (rid,)).fetchone()
        if row is None:
            raise KeyError(rid)
        return row[0]

    def _eids(self, rids):
        if not rids:
            return {}
        result = {}
        rids_list = list(rids)
        chunk_size = 990
        for i in range(0, len(rids_list), chunk_size):
            chunk = rids_list[i:i + chunk_size]
            placeholders = ','.join('?' for _ in chunk)
            for rid, eid in self.db.execute(f'SELECT rid, eid FROM records WHERE rid IN ({placeholders})', chunk):
                result[rid] = eid
        return result


    # Query logic mirrors CountryBlockingIndex; only the accumulator differs.
    def query_candidates(self, s1_norm_name: str, s1_core_tokens: list,
                         s1_norm_addr: str, s1_addr_keys: list,
                         s1_raw_addr: str = "", s1_raw_name: str = "",
                         max_candidates: int = 60,
                         return_weights: bool = False,
                         return_provenance: bool = False,
                         rerank: bool = True,
                         depth: int = None,
                         adaptive: bool = False):
        """Union PixelDust retrieval with trigram-jaccard rerank.

        Provenance channels: exact, addr, token, prefix, acronym, tokpost,
        addr_tok, ngram, tr_name, tr_tok, phon, phone.
        """
        candidate_scores = _SpillScores(self.db)
        prov = _Provenance(candidate_scores) if return_provenance else None
        depth = depth or max(self.rerank_depth, 350)
        cand_name_hits = {}
        cand_addr_hits = {}

        # 1. Exact core name match (+15) with legal suffix variants
        exact_queries = {s1_norm_name}
        if " limited" in s1_norm_name or " private" in s1_norm_name:
            exact_queries.add(s1_norm_name.replace(" limited", " ltd").replace(" private", " pvt"))
            exact_queries.add(s1_norm_name.replace(" limited", "").replace(" ltd", "").strip())
        if " ltd" in s1_norm_name or " pvt" in s1_norm_name:
            exact_queries.add(s1_norm_name.replace(" ltd", " limited").replace(" pvt", " private"))
            exact_queries.add(s1_norm_name.replace(" ltd", "").replace(" limited", "").strip())
        if " llp" in s1_norm_name:
            exact_queries.add(s1_norm_name.replace(" llp", " l l p"))
        if " l l p" in s1_norm_name:
            exact_queries.add(s1_norm_name.replace(" l l p", " llp"))
        for eq in exact_queries:
            if eq in self.exact_name_idx:
                for target_id in self.exact_name_idx[eq]:
                    self._add(candidate_scores, prov, target_id, 15.0, "exact")
                    cand_name_hits[target_id] = cand_name_hits.get(target_id, 0) + 1

        # 2. Compound address keys with selectivity weight
        query_addr_keys = set(s1_addr_keys)
        addr_toks = [t for t in s1_norm_addr.split() if len(t) >= 4 and t not in COMMON_ADDR_STOPWORDS]
        nums = NUM_ONLY_RE.findall(s1_raw_addr or s1_norm_addr)
        street_nums = [n for n in nums if 1 <= len(n) <= 5]
        for snum in street_nums[:3]:
            for tok in addr_toks[:6]:
                query_addr_keys.add(("ST_LOC", snum, tok))
        for k in query_addr_keys:
            if k in self.compound_addr_idx:
                w = self.compound_idf.get(k, 4.0)
                for target_id in self.compound_addr_idx[k]:
                    self._add(candidate_scores, prov, target_id, w, "addr")
                    cand_addr_hits[target_id] = cand_addr_hits.get(target_id, 0) + 1

        # 3. Significant core name tokens (+3)
        cand_token_hits = {}
        for t in set(s1_core_tokens):
            if t in self.token_idx:
                for target_id in self.token_idx[t]:
                    self._add(candidate_scores, prov, target_id, 3.0, "token")
                    cand_token_hits[target_id] = cand_token_hits.get(target_id, 0) + 1
                    cand_name_hits[target_id] = cand_name_hits.get(target_id, 0) + 1
        for target_id, count in cand_token_hits.items():
            if count >= 2:
                self._add(candidate_scores, prov, target_id, 25.0, "token")

        # 3b. Concatenation / handle / abbreviation lookup (eindustries, apicalbros, dvventures)
        words_sources = []
        if len(s1_core_tokens) >= 2:
            words_sources.append(s1_core_tokens)
        all_words = s1_norm_name.split()
        if len(all_words) >= 2 and all_words != s1_core_tokens:
            words_sources.append(all_words)

        seen_concats = set()
        for wlist in words_sources:
            concat_2 = "".join(wlist[:2])
            if len(concat_2) >= 5:
                seen_concats.add(concat_2)
                seen_concats.add(concat_2.replace("brothers", "bros"))
            if len(wlist) >= 3:
                concat_3 = "".join(wlist[:3])
                if len(concat_3) >= 6:
                    seen_concats.add(concat_3)
                    seen_concats.add(concat_3.replace("brothers", "bros").replace("private", "pvt"))
        for ctok in seen_concats:
            if ctok in self.token_idx:
                for target_id in self.token_idx[ctok]:
                    self._add(candidate_scores, prov, target_id, 15.0, "token")
                    cand_name_hits[target_id] = cand_name_hits.get(target_id, 0) + 1

        # 4. 4-character prefix (+1)
        if len(s1_norm_name) >= 4:
            pfx = s1_norm_name[:4]
            if pfx in self.prefix_idx:
                for target_id in self.prefix_idx[pfx]:
                    self._add(candidate_scores, prov, target_id, 1.0, "prefix")
                    cand_name_hits[target_id] = cand_name_hits.get(target_id, 0) + 1

        # 5. Acronym exact blocking key (+8)
        for acr in extract_acronyms(s1_norm_name, s1_core_tokens):
            if acr in self.acronym_idx:
                for target_id in self.acronym_idx[acr]:
                    self._add(candidate_scores, prov, target_id, 8.0, "acronym")
                    cand_name_hits[target_id] = cand_name_hits.get(target_id, 0) + 1

        # 6. Compound (first_significant_token, 2_digit_postal_prefix) (+7)
        if s1_core_tokens:
            first_tok = s1_core_tokens[0]
            for pfx in extract_postal_prefixes(s1_raw_addr, s1_norm_addr):
                k = (first_tok, pfx)
                if k in self.tok_post_idx:
                    for target_id in self.tok_post_idx[k]:
                        self._add(candidate_scores, prov, target_id, 7.0, "tokpost")
                        cand_name_hits[target_id] = cand_name_hits.get(target_id, 0) + 1
                        cand_addr_hits[target_id] = cand_addr_hits.get(target_id, 0) + 1
        if self.address_tokens:
            tokens = [token for token in set(s1_norm_addr.split()) if token in self.addr_token_idx]
            tokens.sort(key=lambda token: self.addr_token_idf[token], reverse=True)
            for token in tokens[:6]:
                weight = self.addr_token_idf[token] * 2.0
                for target_id in self.addr_token_idx[token]:
                    self._add(candidate_scores, prov, target_id, weight, "addr_tok")

        # 7. Character 3-gram channel — always on, top rare trigrams
        if len(s1_norm_name) >= 3:
            q_ngrams = [s1_norm_name[i:i+3] for i in range(len(s1_norm_name)-2)
                        if s1_norm_name[i:i+3] in self.ngram_idx]
            if q_ngrams:
                q_ngrams.sort(key=lambda ng: self.ngram_idf.get(ng, 0.0), reverse=True)
                top_ngrams = q_ngrams[:8]
                for ng in top_ngrams:
                    w = self.ngram_idf.get(ng, 1.0)
                    for target_id in self.ngram_idx[ng]:
                        self._add(candidate_scores, prov, target_id, w, "ngram")

        # 8. Transliterated query views (recover native<->latin mismatches)
        tr_q = transliterate_text(s1_raw_name or "")
        if tr_q:
            tr_q = re.sub(r"[^a-z0-9 ]", " ", tr_q).strip()
            if tr_q in self.exact_name_idx:
                for target_id in self.exact_name_idx[tr_q]:
                    self._add(candidate_scores, prov, target_id, 12.0, "tr_name")
                    cand_name_hits[target_id] = cand_name_hits.get(target_id, 0) + 1
            if tr_q in self.tr_name_idx:
                for target_id in self.tr_name_idx[tr_q]:
                    self._add(candidate_scores, prov, target_id, 12.0, "tr_name")
                    cand_name_hits[target_id] = cand_name_hits.get(target_id, 0) + 1
            for t in set(tr_q.split()):
                if t in self.token_idx and len(t) >= 3:
                    for target_id in self.token_idx[t]:
                        self._add(candidate_scores, prov, target_id, 3.0, "tr_tok")
                        cand_name_hits[target_id] = cand_name_hits.get(target_id, 0) + 1
                if t in self.tr_token_idx:
                    for target_id in self.tr_token_idx[t]:
                        self._add(candidate_scores, prov, target_id, 4.0, "tr_tok")
                        cand_name_hits[target_id] = cand_name_hits.get(target_id, 0) + 1

        # 8b. Schwa / transliteration token variant (+3)
        for t in s1_core_tokens:
            if len(t) >= 3:
                variants = set()
                if not t.endswith('a'):
                    variants.add(t + 'a')
                elif len(t) >= 4:
                    variants.add(t[:-1])
                for vt in variants:
                    if vt in self.tr_token_idx:
                        for target_id in self.tr_token_idx[vt]:
                            self._add(candidate_scores, prov, target_id, 3.0, "tr_tok")
                            cand_name_hits[target_id] = cand_name_hits.get(target_id, 0) + 1
                    elif vt in self.token_idx:
                        for target_id in self.token_idx[vt]:
                            self._add(candidate_scores, prov, target_id, 3.0, "token")
                            cand_name_hits[target_id] = cand_name_hits.get(target_id, 0) + 1

        # 9. Phonetic query tokens (+2, +6 for indic transliterated phonetics)
        cand_phon_hits = {}
        for t in set(s1_core_tokens):
            if len(t) >= 3:
                p = dm_primary(t)
                if p:
                    if p in self.phon_idx:
                        for target_id in self.phon_idx[p]:
                            self._add(candidate_scores, prov, target_id, 2.0, "phon")
                            cand_name_hits[target_id] = cand_name_hits.get(target_id, 0) + 1
                    if getattr(self, '_has_indic_phon', False):
                        pfx = p[:4] if len(p) >= 4 else p
                        cursor = self.db.execute('SELECT rid FROM indic_phon_postings WHERE key IN (?, ?)', (p, pfx))
                        seen_rid = set()
                        for row in cursor:
                            rid = row[0]
                            if rid not in seen_rid:
                                seen_rid.add(rid)
                                self._add(candidate_scores, prov, rid, 6.0, "phon")
                                cand_phon_hits[rid] = cand_phon_hits.get(rid, 0) + 1
                                cand_name_hits[rid] = cand_name_hits.get(rid, 0) + 1
        for target_id, count in cand_phon_hits.items():
            if count >= 2:
                self._add(candidate_scores, prov, target_id, 20.0, "phon")

        # 10. Phone exact (+9)
        for ph in extract_phones(s1_raw_addr or ""):
            if ph in self.phone_idx:
                for target_id in self.phone_idx[ph]:
                    self._add(candidate_scores, prov, target_id, 9.0, "phone")

        # Cross-field dual evidence agreement bonus (+20.0)
        cross_hits = set(cand_name_hits) & set(cand_addr_hits)
        for target_id in cross_hits:
            self._add(candidate_scores, prov, target_id, 20.0, "addr")

        if not candidate_scores:
            return ([], {}) if return_provenance else []

        if adaptive:
            has_exact = s1_norm_name in self.exact_name_idx
            best_w = max(candidate_scores.values())
            tier = assess_difficulty(s1_norm_name, s1_core_tokens, s1_norm_addr,
                                     len(candidate_scores), best_w)
            if has_exact and tier == "hard":
                tier = "medium"
            max_candidates = ADAPTIVE_BUDGETS[tier]

        # Depth + trigram-jaccard rerank (validated EXP_009 pattern)
        if rerank and len(candidate_scores) > max_candidates:
            top = candidate_scores.most_common(depth)
            top_tids = {tid for tid, _ in top}
            # Guaranteed depth retention for multi-token name / phonetic / cross-field matches
            must_include = [tid for tid, count in cand_token_hits.items() if count >= 2 and tid not in top_tids]
            must_include.extend([tid for tid, count in cand_phon_hits.items() if count >= 2 and tid not in top_tids])
            must_include.extend([tid for tid in cross_hits if tid not in top_tids])
            if must_include:
                for tid in set(must_include):
                    top.append((tid, candidate_scores[tid]))
            tids = [tid for tid, _ in top]
            rec_map = self.records.get_many(tids)
            q_tri = _trigrams(s1_norm_name)
            tr_q = transliterate_text(s1_raw_name or "")
            tr_q_tri = _trigrams(tr_q) if tr_q else None
            s1_phons = set(dm_primary(t)[:4] for t in s1_core_tokens if len(t) >= 3 and dm_primary(t))

            rescored = []
            for tid, w in top:
                rec = rec_map.get(tid)
                sim = 0.0
                if rec is not None:
                    c_tri = _trigrams(rec[0]) if rec[0] else None
                    if q_tri and c_tri:
                        sim = max(sim, len(q_tri & c_tri) / (len(q_tri | c_tri) or 1))
                    if tr_q_tri and c_tri:
                        sim = max(sim, len(tr_q_tri & c_tri) / (len(tr_q_tri | c_tri) or 1))
                    if len(rec) > 4 and rec[4]:
                        tr_name = transliterate_text(rec[4])
                        if tr_name:
                            tr_c_tri = _trigrams(tr_name)
                            if q_tri and tr_c_tri:
                                sim = max(sim, len(q_tri & tr_c_tri) / (len(q_tri | tr_c_tri) or 1))
                            if tr_q_tri and tr_c_tri:
                                sim = max(sim, len(tr_q_tri & tr_c_tri) / (len(tr_q_tri | tr_c_tri) or 1))
                            if s1_phons:
                                tgt_phons = set(dm_primary(t)[:4] for t in tr_name.split() if len(t) >= 3 and dm_primary(t))
                                if s1_phons & tgt_phons:
                                    phon_sim = len(s1_phons & tgt_phons) / len(s1_phons)
                                    sim = max(sim, phon_sim)
                rescored.append((tid, w + self.trigram_rerank_boost * sim))
            rescored.sort(key=lambda x: -x[1])
            top_candidates = rescored[:max_candidates]
        else:
            top_candidates = candidate_scores.most_common(max_candidates)

        rids = [rid for rid, _ in top_candidates]
        eid_map = self._eids(rids)
        selected = [(eid_map[rid], score, rid) for rid, score in top_candidates if rid in eid_map]
        if return_provenance:
            prov_out = {eid: sorted(prov.get(rid, ())) for eid, _, rid in selected}
            if return_weights:
                return [(eid, score) for eid, score, _ in selected], prov_out
            return [eid for eid, _, _ in selected], prov_out
        if return_weights:
            return [(eid, score) for eid, score, _ in selected]
        return [eid for eid, _, _ in selected]
