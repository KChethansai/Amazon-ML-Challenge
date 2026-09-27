"""Resumable, read-only mmap postings derived from a ready SQLite index."""

import array
import fcntl
import hashlib
import json
import mmap
import os
import sqlite3
import sys
import uuid
from pathlib import Path


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class PackedPostings:
    """Integer posting lists stored contiguously; directory remains disk-backed."""

    def __init__(self, prefix, source_path):
        self.file = self.mm = self.directory = None
        prefix = Path(prefix)
        manifest_path = Path(str(prefix) + '.manifest.json')
        manifest = json.loads(manifest_path.read_text())
        source = Path(source_path).stat()
        if (manifest['format'] != 1 or manifest['source_size'] != source.st_size or
                manifest['source_mtime_ns'] != source.st_mtime_ns or
                sys.byteorder != 'little'):
            raise ValueError('Packed postings do not match the source index or platform')
        data_path = str(prefix) + '.rid32'
        dir_path = str(prefix) + '.dir.sqlite'
        try:
            self.file = open(data_path, 'rb')
            if os.fstat(self.file.fileno()).st_size != manifest['packed_bytes']:
                raise ValueError('Packed postings size differs from manifest')
            self.directory = sqlite3.connect('file:' + str(Path(dir_path).resolve()) + '?mode=ro', uri=True)
            self.directory.execute('PRAGMA cache_size=-8192')
            self.directory.execute('PRAGMA temp_store=FILE')
            saved = dict(self.directory.execute('SELECT k,v FROM meta'))
            expected_source = json.dumps((source.st_size, source.st_mtime_ns,
                                          str(manifest['records'])))
            if (saved.get('source') != expected_source or
                    saved.get('identity') != manifest['directory_identity'] or
                    int(saved.get('keys', '-1')) != manifest['keys'] or
                    int(saved.get('end', '-1')) * 4 != manifest['packed_bytes']):
                raise ValueError('Packed directory metadata differs from manifest')
            count, min_offset, min_n, max_end = self.directory.execute(
                'SELECT count(*),min(offset),min(n),max(offset+n) FROM postings_dir').fetchone()
            if (count != manifest['keys'] or
                    (count and (min_offset < 0 or min_n <= 0 or max_end > manifest['packed_bytes'] // 4))):
                raise ValueError('Packed directory offsets or row count are invalid')
            if self.directory.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
                raise ValueError('Packed directory integrity check failed')
            if (_sha256(dir_path) != manifest['directory_sha256'] or
                    _sha256(data_path) != manifest['packed_sha256']):
                raise ValueError('Packed files differ from manifest checksums')
            self.mm = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ) if manifest['packed_bytes'] else None
            self.bytes = manifest['packed_bytes']
        except Exception:
            self.close()
            raise

    @classmethod
    def open_if_ready(cls, source_path):
        prefix = str(source_path) + '.postings'
        if not Path(prefix + '.manifest.json').is_file():
            return None
        return cls(prefix, source_path)

    def lookup(self, channel, key):
        row = self.directory.execute(
            'SELECT offset,n FROM postings_dir WHERE channel=? AND key=?',
            (channel, key)).fetchone()
        if row is None:
            return ()
        offset, count = row
        return memoryview(self.mm)[offset * 4:(offset + count) * 4].cast('I')

    def close(self):
        if getattr(self, 'directory', None):
            self.directory.close()
            self.directory = None
        if getattr(self, 'mm', None):
            self.mm.close()
            self.mm = None
        if getattr(self, 'file', None):
            self.file.close()
            self.file = None


def build_packed_postings(source_path, *, batch_keys=1000, stop_after_keys=None):
    """Stream retained SQLite postings into a compact index; safe to resume.

    stop_after_keys is a test hook. A manifest is published only after all keys.
    """
    lock_path = str(source_path) + '.postings.lock'
    with open(lock_path, 'a+b') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Packed postings already have an active builder') from exc
        return _build_unlocked(source_path, batch_keys, stop_after_keys)


def _build_unlocked(source_path, batch_keys, stop_after_keys):
    if sys.byteorder != 'little':
        raise ValueError('Packed postings require a little-endian host')
    source_path = Path(source_path)
    prefix = str(source_path) + '.postings'
    manifest_path = Path(prefix + '.manifest.json')
    if manifest_path.is_file():
        return PackedPostings(prefix, source_path)
    source_stat = source_path.stat()
    source = sqlite3.connect('file:' + str(source_path.resolve()) + '?mode=ro', uri=True)
    source.execute('PRAGMA cache_size=-8192')
    meta = dict(source.execute('SELECT k,v FROM meta'))
    if meta.get('ready') != '1':
        source.close()
        raise ValueError('Source SQLite index is not ready')
    part_path = Path(prefix + '.rid32.building')
    dir_part_path = Path(prefix + '.dir.sqlite.building')
    if not part_path.exists() and Path(prefix + '.rid32').exists():
        os.replace(prefix + '.rid32', part_path)
    if not dir_part_path.exists() and Path(prefix + '.dir.sqlite').exists():
        os.replace(prefix + '.dir.sqlite', dir_part_path)
    directory = sqlite3.connect(dir_part_path)
    directory.execute('PRAGMA journal_mode=DELETE')
    directory.execute('PRAGMA synchronous=FULL')
    directory.execute('PRAGMA cache_size=-8192')
    directory.execute('PRAGMA temp_store=FILE')
    directory.execute('CREATE TABLE IF NOT EXISTS postings_dir('
                      'channel TEXT,key TEXT,offset INTEGER,n INTEGER,'
                      'PRIMARY KEY(channel,key)) WITHOUT ROWID')
    directory.execute('CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY,v TEXT)')
    saved = dict(directory.execute('SELECT k,v FROM meta'))
    fingerprint = json.dumps((source_stat.st_size, source_stat.st_mtime_ns, meta['records']))
    if saved.get('source') not in (None, fingerprint):
        directory.close(); source.close()
        raise ValueError('Source changed during packed-postings build')
    if not saved:
        directory.execute('INSERT INTO meta VALUES (?,?)', ('source', fingerprint))
        directory.execute('INSERT INTO meta VALUES (?,?)', ('identity', uuid.uuid4().hex))
        directory.execute('INSERT INTO meta VALUES (?,?)', ('last_channel', ''))
        directory.execute('INSERT INTO meta VALUES (?,?)', ('last_key', ''))
        directory.execute('INSERT INTO meta VALUES (?,?)', ('end', '0'))
        directory.execute('INSERT INTO meta VALUES (?,?)', ('keys', '0'))
        directory.commit()
        saved = dict(directory.execute('SELECT k,v FROM meta'))
    end = int(saved['end'])
    identity = saved['identity']
    if part_path.exists():
        part_file = open(part_path, 'r+b')
    else:
        part_file = open(part_path, 'w+b')
    if os.fstat(part_file.fileno()).st_size < end * 4:
        part_file.close(); directory.close(); source.close()
        raise ValueError('Packed data is shorter than the committed checkpoint')
    part_file.truncate(end * 4)
    part_file.seek(end * 4)
    last_channel, last_key = saved['last_channel'], saved['last_key']
    keys_done = int(saved['keys'])
    pending = []
    processed = 0
    try:
        keys = source.execute(
            'SELECT channel,key,n FROM key_counts WHERE (channel,key)>(?,?) '
            'ORDER BY channel,key', (last_channel, last_key))
        for channel, key, expected in keys:
            start = end
            buffer = array.array('I')
            for (rid,) in source.execute(
                    'SELECT rid FROM postings WHERE channel=? AND key=? ORDER BY seq',
                    (channel, key)):
                if rid < 0 or rid >= 2**32:
                    raise ValueError('RID exceeds uint32 range')
                buffer.append(rid)
                if len(buffer) >= 16384:
                    buffer.tofile(part_file)
                    end += len(buffer)
                    buffer = array.array('I')
            if buffer:
                buffer.tofile(part_file)
                end += len(buffer)
            if end - start != expected:
                raise ValueError(f'Posting count mismatch for {channel}:{key}')
            pending.append((channel, key, start, expected))
            processed += 1
            if len(pending) >= batch_keys:
                part_file.flush(); os.fsync(part_file.fileno())
                directory.executemany('INSERT INTO postings_dir VALUES (?,?,?,?)', pending)
                keys_done += len(pending)
                directory.executemany('INSERT OR REPLACE INTO meta VALUES (?,?)',
                                      [('last_channel', channel), ('last_key', key),
                                       ('end', str(end)), ('keys', str(keys_done))])
                directory.commit()
                pending.clear()
            if stop_after_keys is not None and processed >= stop_after_keys:
                break
        if pending:
            part_file.flush(); os.fsync(part_file.fileno())
            directory.executemany('INSERT INTO postings_dir VALUES (?,?,?,?)', pending)
            keys_done += len(pending)
            channel, key = pending[-1][:2]
            directory.executemany('INSERT OR REPLACE INTO meta VALUES (?,?)',
                                  [('last_channel', channel), ('last_key', key),
                                   ('end', str(end)), ('keys', str(keys_done))])
            directory.commit()
        complete = stop_after_keys is None or processed < stop_after_keys
        if not complete:
            return None
        expected_keys = source.execute('SELECT count(*) FROM key_counts').fetchone()[0]
        actual_keys = directory.execute('SELECT count(*) FROM postings_dir').fetchone()[0]
        check = directory.execute('PRAGMA quick_check').fetchone()[0]
        if actual_keys != expected_keys or actual_keys != keys_done or check != 'ok':
            raise ValueError('Packed directory validation failed')
        part_file.flush(); os.fsync(part_file.fileno())
    finally:
        part_file.close(); directory.close(); source.close()
    os.replace(part_path, prefix + '.rid32')
    os.replace(dir_part_path, prefix + '.dir.sqlite')
    manifest = {'format': 1, 'source_size': source_stat.st_size,
                'source_mtime_ns': source_stat.st_mtime_ns,
                'records': int(meta['records']), 'keys': keys_done,
                'packed_bytes': end * 4, 'directory_identity': identity,
                'directory_sha256': _sha256(prefix + '.dir.sqlite'),
                'packed_sha256': _sha256(prefix + '.rid32')}
    temp_manifest = prefix + '.manifest.json.building'
    with open(temp_manifest, 'w') as file:
        json.dump(manifest, file, sort_keys=True)
        file.flush(); os.fsync(file.fileno())
    os.replace(temp_manifest, manifest_path)
    return PackedPostings(prefix, source_path)
