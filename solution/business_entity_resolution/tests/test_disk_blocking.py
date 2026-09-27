import sys
import sqlite3
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))
from blocking import CountryBlockingIndex
from disk_blocking import DiskCountryBlockingIndex, _SpillScores
from packed_postings import build_packed_postings
from features import compute_candidate_features_for_s1
from features2 import extra_batch_for_s1


def test_disk_index_parity(tmp_path):
    memory = CountryBlockingIndex('India')
    disk = DiskCountryBlockingIndex('India', tmp_path / 'index.sqlite', create=True)
    rows = [
        ('a', 'tata consultancy services', ['tata', 'consultancy', 'services'],
         '12 main street 560001', [('main', '12')], {12}, 1, '12 Main Street 560001', ''),
        ('b', 'tata consulting', ['tata', 'consulting'],
         '13 main street 560001', [('main', '13')], {13}, 0, '13 Main Street 560001', ''),
        ('c', 'wipro technologies', ['wipro', 'technologies'],
         '12 main street 560001', [('main', '12')], {12}, 1, '12 Main Street 560001', ''),
        ('d', 'acme labs', ['acme', 'labs'], 'elsewhere', [], set(), 0, '', ''),
    ]
    for index in (memory, disk):
        for row in rows:
            index.add_target_record(*row)
        index.prune_frequent_keys()
    disk.close()

    disk = DiskCountryBlockingIndex('India', tmp_path / 'index.sqlite')
    args = ('tata consultancy services', ['tata', 'consultancy', 'services'],
            '12 main street 560001', [('main', '12')])
    opts = dict(s1_raw_addr='12 Main Street 560001', max_candidates=3,
                return_weights=True, return_provenance=True, adaptive=True)
    left, lp = memory.query_candidates(*args, **opts)
    right, rp = disk.query_candidates(*args, **opts)
    assert [x[0] for x in left] == [x[0] for x in right]
    assert [x[1] for x in left] == [x[1] for x in right]
    assert lp == rp
    s1 = {'norm_name': args[0], 'norm_addr': args[2], 'core_tokens': args[1], 'addr_tokens': set(args[2].split()), 'digits': {'12', '560001'}}
    assert compute_candidate_features_for_s1(s1.copy(), left, memory.records) == compute_candidate_features_for_s1(s1.copy(), right, disk.records)
    l_extra = extra_batch_for_s1(s1.copy(), left, memory.records, prov_map=lp)
    r_extra = extra_batch_for_s1(s1.copy(), right, disk.records, prov_map=rp)
    assert l_extra[0] == r_extra[0]
    assert (l_extra[1] == r_extra[1]).all() if hasattr(l_extra[1], 'shape') else l_extra[1] == r_extra[1]
    assert len(l_extra[1][0]) == 22
    assert disk.query_candidates('', [], '', [], return_weights=True, return_provenance=True) == ([], {})
    disk.close()


def test_resume_and_spill(tmp_path):
    rows = [(f'e{i}', f'acme shop {i}', ['acme', 'shop'],
             f'{i} main street', [('main', str(i))], {i}, 1,
             f'{i} main street', '') for i in range(12)]
    path = tmp_path / 'index.sqlite'
    interrupted = DiskCountryBlockingIndex('India', path, create=True)
    for row in rows[:6]:
        interrupted.add_target_record(*row)
    interrupted.close()
    resumed = DiskCountryBlockingIndex('India', path, create=True)
    assert resumed._resume_left == 6
    for row in rows:
        resumed.add_target_record(*row)
    resumed.prune_frequent_keys()
    resumed.close()
    memory = CountryBlockingIndex('India')
    for row in rows:
        memory.add_target_record(*row)
    memory.prune_frequent_keys()
    disk = DiskCountryBlockingIndex('India', path)
    assert isinstance(next(iter(disk.token_idx['acme'])), int)
    assert disk.records[next(iter(disk.token_idx['acme']))][0].startswith('acme')
    prior_limit = _SpillScores.LIMIT
    _SpillScores.LIMIT = 2
    try:
        args = ('acme shop', ['acme', 'shop'], '', [])
        opts = dict(return_weights=True, return_provenance=True, max_candidates=10)
        assert disk.query_candidates(*args, **opts) == memory.query_candidates(*args, **opts)
    finally:
        _SpillScores.LIMIT = prior_limit
        disk.close()


def test_resume_rejects_changed_source(tmp_path):
    path = tmp_path / 'index.sqlite'
    row = ('e0', 'acme', ['acme'], '', [], set(), 1, '', '')
    idx = DiskCountryBlockingIndex('India', path, create=True)
    idx.add_target_record(*row)
    idx.close()
    resumed = DiskCountryBlockingIndex('India', path, create=True)
    changed = ('e0', 'other', ['other'], '', [], set(), 1, '', '')
    try:
        try:
            resumed.add_target_record(*changed)
        except ValueError as exc:
            assert 'diverges' in str(exc)
        else:
            assert False, 'changed source was accepted'
    finally:
        resumed.close()


def test_failed_target_never_commits_partial_postings(tmp_path):
    path = tmp_path / 'index.sqlite'
    idx = DiskCountryBlockingIndex('India', path, create=True)
    good = ('good', 'acme', ['acme'], '', [], set(), 1, '', '')
    idx.add_target_record(*good)

    def broken_keys():
        yield ('street', '1')
        raise ValueError('bad source')

    bad = ('bad', 'other', ['other'], '', broken_keys(), set(), 1, '', '')
    try:
        idx.add_target_record(*bad)
    except ValueError:
        pass
    else:
        assert False, 'broken target was accepted'
    idx.close()
    resumed = DiskCountryBlockingIndex('India', path, create=True)
    # Previous uncommitted batch rolls back; the source must replay from row 1.
    assert resumed._resume_left == 0
    resumed.add_target_record(*good)
    resumed.prune_frequent_keys()
    assert len(resumed.records) == 1
    resumed.close()


def test_packed_postings_resume_and_parity(tmp_path):
    path = tmp_path / 'index.sqlite'
    rows = [(f'e{i}', f'acme shop {i}', ['acme', 'shop'],
             f'{i} main street', [('main', str(i))], {i}, 1,
             f'{i} main street', '') for i in range(20)]
    builder = DiskCountryBlockingIndex('India', path, create=True)
    memory = CountryBlockingIndex('India')
    for row in rows:
        builder.add_target_record(*row)
        memory.add_target_record(*row)
    builder.prune_frequent_keys()
    builder.close()
    memory.prune_frequent_keys()
    assert build_packed_postings(path, batch_keys=2, stop_after_keys=3) is None
    assert not (tmp_path / 'index.sqlite.postings.manifest.json').exists()
    packed = build_packed_postings(path, batch_keys=2)
    assert packed.bytes > 0
    packed.close()
    disk = DiskCountryBlockingIndex('India', path)
    assert disk.packed is not None
    prior_limit = _SpillScores.LIMIT
    _SpillScores.LIMIT = 2
    try:
        args = ('acme shop', ['acme', 'shop'], '', [])
        opts = dict(return_weights=True, return_provenance=True, max_candidates=15)
        assert disk.query_candidates(*args, **opts) == memory.query_candidates(*args, **opts)
    finally:
        _SpillScores.LIMIT = prior_limit
        disk.close()
    directory = tmp_path / 'index.sqlite.postings.dir.sqlite'
    with sqlite3.connect(directory) as db:
        db.execute("UPDATE meta SET v='wrong-directory' WHERE k='identity'")
    try:
        DiskCountryBlockingIndex('India', path)
    except ValueError as exc:
        assert 'metadata' in str(exc)
    else:
        assert False, 'mismatched packed directory was accepted'


if __name__ == '__main__':
    from tempfile import TemporaryDirectory
    for test in (test_disk_index_parity, test_resume_and_spill,
                 test_resume_rejects_changed_source, test_failed_target_never_commits_partial_postings,
                 test_packed_postings_resume_and_parity):
        with TemporaryDirectory() as directory:
            test(Path(directory))
        print(f'PASS {test.__name__}')
