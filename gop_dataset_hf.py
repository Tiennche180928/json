#!/usr/bin/env python3
"""Gộp Duy/Long/Phúc/Tiến từ HF thành một dataset local, chỉ đọc nguồn.

Python >=3.10; dependencies: huggingface_hub, packaging.
Chạy --package humanqueen để xem trước; bỏ tùy chọn đó để gộp cả bốn nguồn.
Tự tiếp tục cùng commit nguồn khi chạy lại cùng --output.
Không chạy, cài đặt hoặc import mã nguồn mẫu; không ghi lên Hugging Face.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from email.parser import Parser
from email import policy
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tarfile
import time
import zipfile

# Tải từng tệp bằng HTTP để hạn chế RAM trên VM nhỏ; dùng lại đăng nhập HF có sẵn.
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

from packaging.utils import canonicalize_name, canonicalize_version, parse_sdist_filename, parse_wheel_filename
from packaging.version import Version, InvalidVersion

RULE_VERSION = '2.0.0'
REPO = 'lbms-dataset/package-pypi'
BASE = 'Dataset-all-package'
SOURCES = {'Dataset-Duy': 'DataDog', 'Dataset-Long': 'H2GLM',
           'Dataset-Phuc': 'Hercule', 'Dataset-Tien': 'MalOSS'}
LAYOUTS = {'Dataset-Duy': ('Dataset2/dataset_multi_version', 'Dataset2/dataset_single_version'),
           'Dataset-Long': ('multi-version', 'single-version'),
           'Dataset-Phuc': ('',), 'Dataset-Tien': ('multi_version', 'single_version')}
SRC_ZIPS = {'src.zip', 'source.zip', 'sourcecode.zip', 'source_code.zip'}
SRC_DIRS = {'sourcecode', 'source_code', 'source', 'src'}
META_NAMES = {'metadata.json', 'metadata_recovered.json', 'pypi-release.json', 'pypi_release.json'}
MAX_TEXT = 32 * 1024**2
MAX_MEMBER = 256 * 1024**2
MAX_UNPACKED = 2 * 1024**3
MAX_MEMBERS = 200000
MIN_FREE = 1024**3


def log(s):
    print(s, flush=True)


def safe(s):
    return (isinstance(s, str) and bool(s) and not s.startswith('/') and '\\' not in s
            and not any(ord(c) < 32 for c in s)
            and all(p not in ('', '.', '..') for p in s.split('/')))


def sha_file(p, algorithm='sha256', git=False):
    h = hashlib.new(algorithm)
    if git:
        h.update(f'blob {p.stat().st_size}\0'.encode())
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    tmp.replace(p)


def strict_json(data):
    def pairs(items):
        d = {}
        for k, v in items:
            if k in d:
                raise ValueError('Khóa JSON lặp: '+k)
            d[k] = v
        return d
    def bad(s):
        raise ValueError('JSON không hợp lệ: '+s)
    return json.loads(data.decode('utf-8-sig'), object_pairs_hook=pairs, parse_constant=bad)


def nk(s):
    return str(canonicalize_name(s, validate=True))


def vk(s):
    Version(s)
    return canonicalize_version(s)


def missing(v):
    return v is None or v == [] or v == {} or (isinstance(v, str) and v.strip().lower() in ('', 'unknown', 'none', 'null', 'n/a'))


def iso(s):
    if not isinstance(s, str):
        return None
    try:
        d = datetime.fromisoformat(s.strip().replace(' UTC', '+00:00').replace('Z', '+00:00'))
        return d.astimezone(timezone.utc) if d.tzinfo else None
    except ValueError:
        return None


def is_archive(s):
    return s.lower().endswith(('.tar.gz', '.tar.bz2', '.tar.xz', '.tgz', '.tar', '.whl', '.zip'))


def locate(path):
    """Có hỗ trợ một lớp wrapper như setdotwork/vtk-osmesa/<version>."""
    if not safe(path):
        return None
    for source, layouts in LAYOUTS.items():
        prefix = f'{BASE}/{source}/'
        if not path.startswith(prefix):
            continue
        rest = path[len(prefix):]
        for layout in layouts:
            lead = layout+'/' if layout else ''
            if not rest.startswith(lead):
                continue
            parts = rest[len(lead):].split('/')
            for offset in (0, 1):
                if len(parts) < offset+3:
                    continue
                name, ver = parts[offset:offset+2]
                try:
                    normalized, version = nk(name), vk(ver)
                except (ValueError, InvalidVersion, TypeError):
                    continue
                rel = '/'.join(parts[offset+2:])
                return {'source': source, 'name': normalized, 'version_key': version,
                        'folder_version': ver, 'relative_path': rel,
                        'directory': path[:-(len(rel)+1)]}
    return None


def hub_entry(entry):
    lfs = getattr(entry, 'lfs', None)
    digest = lfs.get('sha256') if isinstance(lfs, dict) else getattr(lfs, 'sha256', None)
    return {'path': entry.path, 'size': entry.size, 'sha256': digest,
            'blob_id': entry.blob_id}


def inventory(api, control):
    target = control/'inventory.json'
    if target.exists():
        data = json.loads(target.read_text(encoding='utf-8'))
        if data.get('rule_version') != RULE_VERSION or data.get('repo') != REPO:
            raise ValueError('Lượt cũ dùng quy tắc/kho khác; hãy chọn --output mới.')
        return data
    revision = api.repo_info(REPO, repo_type='dataset').sha
    files = []
    for source in SOURCES:
        log('Đọc danh mục '+source+' ...')
        for entry in api.list_repo_tree(REPO, repo_type='dataset', revision=revision,
                                        path_in_repo=f'{BASE}/{source}', recursive=True):
            if hasattr(entry, 'blob_id'):
                files.append(hub_entry(entry))
                if len(files) % 5000 == 0:
                    log(f'  Đã đọc {len(files):,} tệp.')
    records = {}
    ignored = []
    for file in files:
        where = locate(file['path'])
        if where is None:
            ignored.append(file)
            continue
        directory = where['directory']
        record = records.setdefault(directory, {k: v for k, v in where.items() if k != 'relative_path'})
        record.setdefault('files', []).append({**file, 'relative_path': where['relative_path']})
    data = {'repo': REPO, 'revision': revision, 'rule_version': RULE_VERSION,
            'records': list(records.values()), 'outside_versions': ignored}
    atomic_json(target, data)
    return data


class Downloads:
    def __init__(self, root, revision):
        self.root, self.revision = root, revision
        self.by_content = {}

    def get(self, file):
        from huggingface_hub import hf_hub_download
        key = (file.get('sha256') or file.get('blob_id'), file['size'])
        if key[0] and key in self.by_content:
            return self.by_content[key]
        if not safe(file['path']):
            raise ValueError('Đường dẫn nguồn không hợp lệ')
        if shutil.disk_usage(self.root).free < file['size'] + MIN_FREE:
            raise OSError('Thiếu dung lượng trống; cần chừa ít nhất 1 GiB khi tải.')
        for attempt in range(4):
            try:
                path = Path(hf_hub_download(REPO, file['path'], repo_type='dataset',
                    revision=self.revision, local_dir=str(self.root), etag_timeout=20))
                if path.is_symlink() or path.stat().st_size != file['size']:
                    raise ValueError('Kích thước/liên kết tệp sai: '+file['path'])
                if file.get('sha256'):
                    ok = sha_file(path) == file['sha256']
                elif file.get('blob_id'):
                    ok = sha_file(path, 'sha1', git=True) == file['blob_id']
                else:
                    raise ValueError('Nguồn không có hash đối chiếu: '+file['path'])
                if not ok:
                    raise ValueError('Hash tải về không khớp commit nguồn: '+file['path'])
                if key[0]:
                    self.by_content[key] = path
                return path
            except Exception as exc:
                status = getattr(getattr(exc, 'response', None), 'status_code', None)
                retry = status in (408, 429, 500, 502, 503, 504) or isinstance(exc, (OSError, TimeoutError))
                if not retry or attempt == 3:
                    raise
                delay = min(30, 5 * 2**attempt)
                log(f'  Tải lỗi; thử lại sau {delay}s: {type(exc).__name__}')
                time.sleep(delay)

    def text(self, file):
        if file['size'] > MAX_TEXT:
            raise ValueError('Tệp văn bản vượt giới hạn 32 MiB: '+file['path'])
        return self.get(file).read_bytes()


@contextmanager
def open_members(path):
    """Chỉ stream các thành viên thường, không extract ra hệ thống tệp."""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            entries = [e for e in archive.infolist() if not e.is_dir()]
            for e in entries:
                mode = e.external_attr >> 16
                if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG)):
                    raise ValueError('ZIP có liên kết/tệp đặc biệt')
            yield [(e.filename, e.file_size, e) for e in entries], archive.open
    else:
        with tarfile.open(path, 'r:*') as archive:
            entries = [e for e in archive.getmembers() if not e.isdir()]
            if any(not e.isfile() for e in entries):
                raise ValueError('TAR có liên kết/tệp đặc biệt; cần kiểm tra riêng')
            yield [(e.name, e.size, e) for e in entries], archive.extractfile


def core_info(data):
    msg = Parser(policy=policy.compat32).parsestr(data.decode('utf-8', errors='replace'))
    d = {key: [str(v) for v in msg.get_all(key)] for key in msg.keys()}
    body = msg.get_payload()
    if isinstance(body, str) and body.strip():
        d['Description'] = [body.strip()]
    return d


def core_identity(d, name, ver):
    try:
        return nk(d['Name'][0]) == name and vk(d['Version'][0]) == ver
    except (KeyError, IndexError, ValueError, TypeError):
        return False


def member_name(s):
    while s.startswith('./'):
        s = s[2:]
    if not safe(s):
        raise ValueError('Đường dẫn trong archive không hợp lệ: '+repr(s))
    return s


def inspect_archive(path, name, ver, output_zip=None):
    """Hash nội dung từng thành viên + đọc core metadata, RAM giới hạn theo block."""
    tree, cores, total = {}, [], 0
    with open_members(path) as (members, opener):
        if len(members) > MAX_MEMBERS:
            raise ValueError('Archive có quá nhiều thành viên')
        names = [member_name(e[0]) for e in members]
        if len(set(names)) != len(names):
            raise ValueError('Archive có tên thành viên lặp')
        strip = bool(names) and all('/' in s for s in names) and len({s.split('/')[0] for s in names}) == 1
        z = zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) if output_zip else None
        try:
            for (raw, size, item), full in zip(members, names):
                total += size
                if size > MAX_MEMBER or total > MAX_UNPACKED:
                    raise ValueError('Archive vượt giới hạn đọc tĩnh')
                rel = full.split('/', 1)[1] if strip else full
                h, read_size = hashlib.sha256(), 0
                small = bytearray() if Path(rel).name in ('PKG-INFO', 'METADATA') and size <= 8*1024**2 else None
                stream = opener(item)
                if stream is None:
                    raise ValueError('Không đọc được thành viên archive')
                writer = None
                if z:
                    zi = zipfile.ZipInfo(full, (1980, 1, 1, 0, 0, 0))
                    zi.compress_type, zi.create_system = zipfile.ZIP_DEFLATED, 3
                    zi.external_attr = (stat.S_IFREG | 0o644) << 16
                    writer = z.open(zi, 'w', force_zip64=True)
                try:
                    with stream:
                        for block in iter(lambda: stream.read(1024**2), b''):
                            read_size += len(block)
                            if read_size > size:
                                raise ValueError('Kích thước thành viên không khớp')
                            h.update(block)
                            if small is not None:
                                small.extend(block)
                            if writer:
                                writer.write(block)
                finally:
                    if writer:
                        writer.close()
                if read_size != size:
                    raise ValueError('Thành viên archive bị cắt ngắn')
                tree[rel] = h.hexdigest()
                if small is not None:
                    cores.append((rel, core_info(bytes(small))))
        finally:
            if z:
                z.close()
    # Core ở root trước, rồi *.dist-info / *.egg-info; không suy luận từ setup.py.
    cores.sort(key=lambda c: (c[0].count('/'), c[0]))
    primary = cores[0][1] if cores else {}
    identity = core_identity(primary, name, ver) if primary else None
    code_count = sum(p.endswith(('.py', '.pyi', '.pyx', '.c', '.cpp', '.h', '.rs', '.js', '.ts')) for p in tree)
    if not tree or not code_count:
        raise ValueError('Chưa thấy mã nguồn trong archive')
    return {'tree': tree, 'core_metadata': primary, 'identity_matches': identity,
            'code_files': code_count, 'member_count': len(tree), 'unpacked_size': total}


def object_identities(meta):
    for label, obj in [('root', meta)] + [(k, meta[k]) for k in ('info', 'pkg_info', 'package_metadata_raw') if isinstance(meta.get(k), dict)]:
        for key in ('name', 'Name', 'normalized_name', 'version', 'Version'):
            value = obj.get(key)
            if not missing(value):
                yield label+'.'+key, ('name' if key.lower() != 'version' else 'version'), value
    for key in ('raw_core_metadata', 'core_metadata'):
        if isinstance(meta.get(key), dict):
            for field in ('Name', 'Version'):
                value = meta[key].get(field)
                if isinstance(value, list):
                    value = value[0] if value else None
                if not missing(value):
                    yield key+'.'+field, field.lower(), value


def valid_metadata(meta, name, ver):
    seen = set()
    for _, kind, value in object_identities(meta):
        seen.add(kind)
        try:
            if (nk(value) if kind == 'name' else vk(value)) != (name if kind == 'name' else ver):
                return False
        except (TypeError, ValueError):
            return False
    return seen == {'name', 'version'}


def meta_values(meta):
    """Map các schema hiện có về các trường dùng được; giữ core metadata đầy đủ riêng."""
    out = {}
    for obj in (meta.get('info'), meta.get('pkg_info'), meta.get('package_metadata_raw'), meta):
        if not isinstance(obj, dict):
            continue
        for key in ('summary', 'description', 'description_content_type', 'author', 'author_email',
                    'maintainer', 'maintainer_email', 'license', 'license_expression', 'home_page',
                    'project_urls', 'requires_python', 'requires_dist', 'classifiers', 'keywords',
                    'provides_extra', 'platform', 'download_url', 'license_files', 'dynamic',
                    'docs_url', 'bugtrack_url'):
            if not missing(obj.get(key)) and missing(out.get(key)):
                out[key] = obj[key]
    mapping = {'Summary': 'summary', 'Description': 'description', 'Description-Content-Type': 'description_content_type',
               'Author': 'author', 'Author-email': 'author_email', 'Maintainer': 'maintainer',
               'Maintainer-email': 'maintainer_email', 'License': 'license', 'License-Expression': 'license_expression',
               'Home-page': 'home_page', 'Requires-Python': 'requires_python', 'Requires-Dist': 'requires_dist',
               'Classifier': 'classifiers', 'Keywords': 'keywords', 'Provides-Extra': 'provides_extra'}
    for core in (meta.get('core_metadata'), meta.get('raw_core_metadata')):
        if not isinstance(core, dict):
            continue
        for raw, key in mapping.items():
            value = core.get(raw)
            if isinstance(value, list) and key not in ('requires_dist', 'classifiers', 'provides_extra'):
                value = value[0] if value else None
            if not missing(value) and missing(out.get(key)):
                out[key] = value
    return out


def hash_value(obj):
    if not isinstance(obj, dict):
        return None
    digests = obj.get('digests')
    value = obj.get('sha256') or (digests.get('sha256') if isinstance(digests, dict) else None)
    return value.lower() if isinstance(value, str) and re.fullmatch('[a-fA-F0-9]{64}', value) else None


def claimed_hashes(meta):
    found = set()
    for value in (meta.get('original_sha256'), hash_value(meta)):
        if isinstance(value, str) and re.fullmatch('[a-fA-F0-9]{64}', value):
            found.add(value.lower())
    for key in ('distributions', 'files', 'urls'):
        if isinstance(meta.get(key), list):
            found.update(h for obj in meta[key] if (h := hash_value(obj)))
    return found


def time_claims(meta, file, original_hash):
    result = []
    hashes = claimed_hashes(meta)
    def add(value, field, hash_link=None, candidate=False, evidence=None):
        d = iso(value)
        if d:
            result.append({'value': d.isoformat().replace('+00:00', 'Z'), 'field': field,
                'source_path': file['path'], 'hash_matches_original': bool(original_hash and hash_link == original_hash),
                'candidate': candidate, 'evidence': evidence})
    # Chỉ coi timestamp root liên kết nếu bản ghi chỉ tham chiếu đúng một artifact.
    root_hash = next(iter(hashes)) if len(hashes) == 1 else None
    add(meta.get('upload_time'), 'upload_time', root_hash)
    add(meta.get('upload_time_candidate'), 'upload_time_candidate', root_hash, True)
    for key in ('distributions', 'files', 'urls'):
        if not isinstance(meta.get(key), list):
            continue
        for obj in meta[key]:
            if not isinstance(obj, dict):
                continue
            h = hash_value(obj)
            if original_hash and h and h != original_hash:
                continue
            add(obj.get('upload_time_iso_8601') or obj.get('upload_time'), key+'.upload_time', h,
                evidence={k: obj[k] for k in ('source', 'evidence_file', 'csv_sha256', 'csv_row', 'local_artifact_match') if k in obj})
    supplement = meta.get('supplement', {}).get('upload_time', {}) if isinstance(meta.get('supplement'), dict) else {}
    if isinstance(supplement, dict):
        add(supplement.get('value'), 'supplement.upload_time.value', root_hash)
    return result


def choose_time(claims, notes):
    candidates = [c for c in claims if c['hash_matches_original'] and not c['candidate']]
    if not candidates:
        notes.append('upload_time chưa có bằng chứng liên kết đúng SHA256 phát hành gốc')
        return None, None
    dates = [(iso(c['value']), c) for c in candidates]
    # Khác phần thập phân với bản bị cắt tới giây được coi là khác độ chính xác.
    seconds = {d.replace(microsecond=0) for d, _ in dates}
    precise = {d for d, _ in dates if d.microsecond}
    if len(seconds) != 1 or len(precise) > 1:
        notes.append('upload_time mâu thuẫn giữa các bằng chứng cùng SHA256; chưa tự chọn')
        return None, None
    chosen = next((c for d, c in dates if d.microsecond), dates[0][1])
    return chosen['value'], {'method': 'matching_original_sha256_in_source_metadata',
        'source_path': chosen['source_path'], 'field': chosen['field'],
        'independently_requeried': False, 'source_evidence': chosen.get('evidence')}


def clean_text(data):
    try:
        return data.decode('utf-8-sig').replace('\r\n', '\n').replace('\r', '\n')
    except UnicodeDecodeError:
        return data.decode('latin-1').replace('\r\n', '\n').replace('\r', '\n')


def lineset(text):
    return {re.sub(r'\s+', ' ', s.strip()).lower() for s in text.splitlines() if s.strip()}


def text_score(text, role, ver, previous):
    lines = lineset(text)
    non_template = [s for s in lines if not any(t in s for t in ('chưa có', 'not available', 'not found', 'no readme', 'placeholder'))]
    scope = 0
    if role == 'version_changes':
        if previous:
            scope = 2 if previous in text and ver in text else 0
            if any(s in text.lower() for s in ('first version', 'phiên bản đầu tiên', 'baseline')):
                scope = -1
        elif any(s in text.lower() for s in ('first version', 'phiên bản đầu tiên', 'baseline')):
            scope = 2
    commits = len(set(re.findall(r'\b[a-f0-9]{40}\b', text))) if role == 'history' else 0
    # Bao phủ dòng/thông tin thực; không có ưu tiên cố định theo tên người.
    return (scope, commits, bool(text.strip()), len(non_template), len(lines))


def pick_text(candidates, role, ver, previous, notes):
    if not candidates:
        return None, None
    unique = {}
    for path, data in candidates:
        text = clean_text(data)
        key = '\n'.join(sorted(lineset(text)))
        unique.setdefault(key, (path, text))
    scored = sorted(unique.values(), key=lambda x: (text_score(x[1], role, ver, previous), x[0]), reverse=True)
    chosen = scored[0]
    if len(scored) > 1:
        a = lineset(chosen[1])
        for _, text in scored[1:]:
            b = lineset(text)
            if a and b and len(a & b) / min(len(a), len(b)) < .8:
                notes.append(f'{role}: các bản khác đáng kể; đã chọn theo phạm vi và độ bao phủ, cần đọc lại nếu dùng phân tích')
                break
    return chosen[1].encode('utf-8'), chosen[0]


def scan_document(path, role, ver, previous):
    """Báo cáo litellm có thể >130 MiB: đọc từng đoạn, không nạp cả tệp vào RAM."""
    lines, commits, has_ver, has_prev, baseline = set(), set(), False, False, False
    meaningful = 0
    with path.open('rb') as stream:
        while True:
            raw = stream.readline(1024**2)
            if not raw:
                break
            text = clean_text(raw)
            norm = re.sub(r'\s+', ' ', text.strip()).lower()
            if not norm:
                continue
            has_ver |= ver in text
            has_prev |= bool(previous and previous in text)
            baseline |= any(s in norm for s in ('first version', 'phiên bản đầu tiên', 'baseline'))
            if len(lines) < 200000:
                digest = hashlib.sha256(norm.encode()).digest()[:16]
                if digest not in lines:
                    lines.add(digest)
                    meaningful += not any(s in norm for s in ('chưa có', 'not available', 'not found', 'placeholder'))
            if role == 'history':
                commits.update(re.findall(r'\b[a-f0-9]{40}\b', text))
    scope = 0
    if role == 'version_changes':
        scope = (2 if has_prev and has_ver else -1 if baseline else 0) if previous else (2 if baseline else 0)
    return (scope, len(commits), bool(lines), meaningful, len(lines)), lines


def pick_document(candidates, role, ver, previous, notes):
    unique = {}
    for origin, path in candidates:
        digest = sha_file(path)
        unique.setdefault(digest, (origin, path))
    if not unique:
        return None, None
    scored = []
    for origin, path in unique.values():
        score, lines = scan_document(path, role, ver, previous)
        scored.append((score, origin, path, lines))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best = scored[0]
    for _, _, _, other in scored[1:]:
        if best[3] and other and len(best[3] & other)/min(len(best[3]), len(other)) < .8:
            notes.append(f'{role}: các bản khác đáng kể; đã chọn theo phạm vi và độ bao phủ, cần đọc lại nếu dùng phân tích')
            break
    return best[2], best[1]


def pack_expanded(record, dl, target):
    files = [f for f in record['files'] if f['relative_path'].split('/')[0] in SRC_DIRS and '/' in f['relative_path']]
    if not files:
        return False
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as out:
        for file in files:
            rel = file['relative_path'].split('/', 1)[1]
            if not safe(rel):
                raise ValueError('Đường dẫn mã nguồn không hợp lệ')
            out.write(dl.get(file), rel)
    return True


def select_source(records, metadata, dl, work, name, ver, notes):
    candidates, excluded = [], []
    allfiles = [(r, f) for r in records for f in r['files']]
    originals = [(r, f) for r, f in allfiles if '/' not in f['relative_path']
                 and is_archive(f['relative_path']) and f['relative_path'].lower() not in SRC_ZIPS]
    zips = [(r, f) for r, f in allfiles if f['relative_path'].lower() in SRC_ZIPS]
    expected = Counter(h for _, m in metadata for h in claimed_hashes(m))
    checked = {}
    for is_original, pairs in ((True, originals), (False, zips)):
        for record, file in pairs:
            try:
                path = dl.get(file)
                digest = sha_file(path)
                if digest not in checked:
                    checked[digest] = inspect_archive(path, name, ver)
                analysis = checked[digest]
                if analysis['identity_matches'] is False:
                    raise ValueError('Core metadata trong mã nguồn khác package/version thư mục')
                if analysis['identity_matches'] is None:
                    # Không chạy setup.py; chỉ cho filename hợp lệ + metadata đúng làm hỗ trợ.
                    if not metadata:
                        raise ValueError('Không đủ định danh tĩnh cho mã nguồn')
                    if is_original:
                        fname = Path(file['path']).name
                        try:
                            parsed = parse_wheel_filename(fname) if fname.endswith('.whl') else parse_sdist_filename(fname)
                            filename_ok = nk(str(parsed[0])) == name and vk(str(parsed[1])) == ver
                        except ValueError:
                            filename_ok = False
                        if not filename_ok:
                            raise ValueError('Không xác minh được định danh archive qua PKG-INFO/filename')
                    notes.append('Một mã nguồn thiếu PKG-INFO/METADATA; định danh chỉ được hỗ trợ bởi metadata ngoài')
                candidates.append({'record': record, 'file': file, 'path': path, 'sha256': digest,
                    'original': is_original, **analysis})
            except (ValueError, tarfile.TarError, zipfile.BadZipFile, RuntimeError) as exc:
                excluded.append({'path': file['path'], 'reason': str(exc)})
        # Nếu có archive gốc kiểm tra được, tạo src.zip trực tiếp từ nó: khỏi tải ZIP chuyển đổi thừa.
        if any(c['original'] for c in candidates):
            break
    if not candidates:
        for idx, record in enumerate(records):
            path = work/f'expanded-{idx}.zip'
            if pack_expanded(record, dl, path):
                try:
                    analysis = inspect_archive(path, name, ver)
                    if analysis['identity_matches'] is not True:
                        raise ValueError('Mã nguồn dạng thư mục chưa có core metadata đúng định danh')
                    candidates.append({'record': record, 'file': None, 'path': path, 'sha256': sha_file(path),
                        'original': False, **analysis})
                except (ValueError, zipfile.BadZipFile) as exc:
                    excluded.append({'path': record['directory']+'/sourcecode', 'reason': str(exc)})
    if not candidates:
        raise ValueError('Không có mã nguồn đọc được và đúng định danh; '+json.dumps(excluded, ensure_ascii=False)[:1000])
    support = Counter(c['sha256'] for c in candidates)
    candidates.sort(key=lambda c: (c['identity_matches'] is True, c['original'],
        expected[c['sha256']], support[c['sha256']], c['code_files'], c['member_count'], c['sha256']), reverse=True)
    chosen = candidates[0]
    fingerprints = {tuple(sorted(c['tree'].items())) for c in candidates}
    if len(fingerprints) > 1:
        notes.append('Mã nguồn cùng định danh có nội dung khác nhau: bản chính được chọn theo định danh, artifact gốc, SHA256 và độ bao phủ; cần kiểm tra')
    return chosen, excluded


def merge_version(records, dl, work, name, version_key, display_version, previous, revision):
    notes, rejected, metadata, files = [], [], [], []
    for r in records:
        files.extend((r, f) for f in r['files'])
    for r, f in files:
        if f['relative_path'].lower() not in META_NAMES:
            continue
        try:
            m = strict_json(dl.text(f))
            if not isinstance(m, dict) or not valid_metadata(m, name, version_key):
                raise ValueError('Metadata không khớp package/version hoặc thiếu định danh')
            metadata.append((f, m))
        except (ValueError, UnicodeError) as exc:
            rejected.append({'path': f['path'], 'reason': str(exc)})
    chosen, bad_sources = select_source(records, metadata, dl, work, name, version_key, notes)
    rejected.extend(bad_sources)
    output = work/'ready'
    output.mkdir()
    core = chosen['core_metadata']
    # Chuẩn hóa ZIP từ đúng source được chọn; không trộn code của hai artifact khác nhau.
    inspect_archive(chosen['path'], name, version_key, output/'src.zip')
    original_hash, original_name = None, None
    if chosen['original']:
        original_name = Path(chosen['file']['path']).name
        if not safe(original_name) or original_name in ('metadata.json', 'src.zip'):
            raise ValueError('Tên artifact phát hành không hợp lệ')
        original_hash = chosen['sha256']
        if shutil.disk_usage(output).free < chosen['path'].stat().st_size + MIN_FREE:
            raise OSError('Thiếu dung lượng khi lưu artifact gốc')
        shutil.copyfile(chosen['path'], output/original_name)
    else:
        notes.append('Chưa có tệp phát hành gốc; src.zip được lấy từ bản mã nguồn có sẵn')
    values = meta_values({'core_metadata': core})
    field_sources = {key: 'selected_source_core_metadata' for key in values}
    for f, m in sorted(metadata, key=lambda fm: (original_hash in claimed_hashes(fm[1]) if original_hash else False,
                                               len(meta_values(fm[1])), fm[0]['path']), reverse=True):
        for key, value in meta_values(m).items():
            if missing(values.get(key)):
                values[key], field_sources[key] = value, f['path']
            elif values[key] != value and key not in ('description', 'classifiers', 'project_urls'):
                notes.append('Metadata khác giá trị ở trường '+key+'; ưu tiên core metadata trong artifact đã chọn')
    claims = [c for f, m in metadata for c in time_claims(m, f, original_hash)]
    upload_time, time_source = choose_time(claims, notes)
    readmes, changes, histories = [], [], []
    for r, f in files:
        rel, lower = f['relative_path'], f['relative_path'].lower()
        if '/' not in rel and re.fullmatch(r'readme(?:\.[a-z0-9_-]+)?', lower):
            readmes.append((f['path'], dl.text(f)))
        elif '/' not in rel and ('version_change' in lower or 'change_version' in lower):
            changes.append((f['path'], dl.get(f)))
        elif '/' not in rel and (lower in ('history-commit.md', 'history_commit.md', 'commits.json', 'commit_history.json') or 'history-commit' in lower):
            path = dl.get(f)
            if lower.endswith('.json'):
                # Một lịch sử chuẩn Markdown, giữ đầy đủ nội dung JSON ở trong.
                converted = work/('history-'+hashlib.sha256(f['path'].encode()).hexdigest()+'.md')
                with converted.open('wb') as out:
                    if path.stat().st_size:
                        out.write(b'# Commit history\n\n```json\n')
                        with path.open('rb') as stream:
                            shutil.copyfileobj(stream, out, 1024**2)
                        out.write(b'\n```\n')
                path = converted
            histories.append((f['path'], path))
    # Thu lịch sử theo commit/<sha>/info.json + code.diff vào một tệp duy nhất.
    commit_groups = defaultdict(list)
    for r, f in files:
        parts = f['relative_path'].split('/')
        if len(parts) >= 3 and parts[0] == 'commit':
            commit_groups[(r['directory'], parts[1])].append(f)
    per_record = {}
    for (directory, commit), entries in sorted(commit_groups.items()):
        history_path = per_record.setdefault(directory, work/('commits-'+hashlib.sha256(directory.encode()).hexdigest()+'.md'))
        with history_path.open('ab') as out:
            out.write(('## '+commit+'\n\n').encode())
            for f in sorted(entries, key=lambda f: f['relative_path']):
                path = dl.get(f)
                out.write(('### '+Path(f['path']).name+'\n\n').encode())
                # Thụt 4 dấu cách để giữ nguyên diff/JSON, kể cả khi chứa backticks.
                with path.open('rb') as stream:
                    while True:
                        raw = stream.readline(1024**2)
                        if not raw:
                            break
                        out.write(b'    '+raw)
                        if not raw.endswith(b'\n'):
                            out.write(b'\n')
                out.write(b'\n')
    histories.extend((d+'/commit/', p) for d, p in per_record.items())
    # Description thật từ PKG-INFO là fallback README, không viết mô tả giả.
    if isinstance(values.get('description'), str) and values['description'].strip():
        readmes.append(('selected_metadata.description', values['description'].encode('utf-8')))
    choices = {}
    for role, canonical, candidates in [('readme', 'README.md', readmes),
                                        ('version_changes', 'version_changes.md', changes),
                                        ('history', 'history-commit.md', histories)]:
        if role == 'readme':
            data, origin = pick_text(candidates, role, display_version, previous, notes)
        else:
            doc, origin = pick_document(candidates, role, display_version, previous, notes)
            data = None
            if doc is not None:
                shutil.copyfile(doc, output/canonical)
                choices[canonical] = origin
                continue
        if data is not None:
            (output/canonical).write_bytes(data)
            choices[canonical] = origin
        elif role == 'history':
            # Theo yêu cầu: luôn có history-commit.md, kể cả chưa được bổ sung.
            (output/canonical).write_bytes(b'')
            choices[canonical] = 'empty_placeholder_requested'
        else:
            notes.append('Chưa có '+canonical)
    if not (output/'history-commit.md').stat().st_size:
        notes.append('history-commit.md đang rỗng, vẫn giữ theo yêu cầu')
    labels = []
    for file, meta in metadata:
        if 'label' in meta:
            labels.append({'source_path': file['path'], 'reported_value': meta['label']})
    distribution = {'filename': original_name, 'sha256': original_hash,
                    'size': chosen['path'].stat().st_size} if original_name else None
    if distribution:
        for _, meta in metadata:
            for field in ('distributions', 'files', 'urls'):
                if not isinstance(meta.get(field), list):
                    continue
                for obj in meta[field]:
                    if hash_value(obj) == original_hash:
                        for k in ('url', 'packagetype', 'python_version', 'requires_python', 'yanked', 'yanked_reason'):
                            if k in obj and k not in distribution:
                                distribution[k] = obj[k]
    outmeta = {'name': name, 'normalized_name': name, 'version': display_version, 'ecosystem': 'PyPI', **values,
        'upload_time': upload_time, 'upload_time_evidence': time_source,
        'filename': original_name, 'original_sha256': original_hash,
        'original_distribution': distribution,
        'source_labels': labels,
        'src': {'filename': 'src.zip', 'sha256': sha_file(output/'src.zip'),
                'member_count': chosen['member_count'], 'identity_checked_in_core_metadata': chosen['identity_matches'] is True},
        'core_metadata': core,
        'collection': {'repo': REPO, 'revision': revision,
            'sources': [{'dataset': r['source'], 'source': SOURCES[r['source']], 'path': r['directory']} for r in records]},
        'merge': {'rule_version': RULE_VERSION, 'selected_source': chosen['file']['path'] if chosen['file'] else chosen['record']['directory']+'/sourcecode',
                  'field_sources': field_sources, 'selected_files': choices,
                  'history_present': (output/'history-commit.md').stat().st_size > 0,
                  'notes': sorted(set(notes))}}
    atomic_json(output/'metadata.json', outmeta)
    retained = {p.name: {'size': p.stat().st_size, 'sha256': sha_file(p)} for p in output.iterdir()}
    # Chỉ đúng các vai trò đã chốt, không để nguon/ hay Dataset-* lồng vào version.
    expected_names = {'metadata.json', 'src.zip', 'README.md', 'version_changes.md', 'history-commit.md'}
    if original_name:
        expected_names.add(original_name)
    if set(retained) - expected_names or any(p.is_dir() for p in output.iterdir()):
        raise ValueError('Đầu ra có tệp/thư mục ngoài cấu trúc đã chốt')
    return {'name': name, 'version': display_version, 'files': retained,
        'sources': sorted({r['source'] for r in records}), 'notes': sorted(set(notes)), 'excluded': rejected,
        'input_file_occurrences': len(files), 'selected_files': choices,
        'upload_time': upload_time, 'original_sha256': original_hash,
        'input_files_not_copied_separately': [f['path'] for _, f in files]}, output


def record_ok(path, outcome):
    if not path.is_dir() or path.is_symlink():
        return False
    if {p.name for p in path.iterdir()} != set(outcome['files']):
        return False
    return all(not (path/name).is_symlink() and (path/name).is_file()
               and (path/name).stat().st_size == info['size']
               and sha_file(path/name) == info['sha256'] for name, info in outcome['files'].items())


def report(control, output, inv, results, failures):
    atomic_json(control/'KET_QUA_GOP.json', {'rule_version': RULE_VERSION, 'repo': REPO,
        'revision': inv['revision'], 'output': str(output), 'versions': list(results.values()),
        'failures': failures, 'outside_versions': inv['outside_versions']})
    packages = {r['name'] for r in results.values()}
    with (control/'TONG_KET_GOP.md').open('w', encoding='utf-8') as out:
        out.write('# Kết quả gộp dataset\n\n')
        out.write(f'- Kho nguồn: {REPO}\n- Commit nguồn: `{inv["revision"]}`\n')
        out.write(f'- Đã gộp: **{len(packages)} package, {len(results)} version**.\n')
        out.write(f'- Version chưa gộp do lỗi: **{len(failures)}**.\n')
        out.write(f'- Tệp ngoài thư mục version: **{len(inv["outside_versions"])}**, danh mục ở KET_QUA_GOP.json.\n')
        out.write('- Bốn dataset gốc giữ nguyên. Mỗi version đầu ra có một tệp cho mỗi vai trò.\n')
        out.write('- History commit rỗng vẫn giữ. Tệp văn bản khác nhau được chọn tự động theo phạm vi và độ bao phủ; đây không phải chứng nhận nội dung đúng hoàn toàn.\n\n')
        out.write('| Package/version | Ghi chú cần biết |\n|---|---|\n')
        for r in results.values():
            # Không nhắc lại lỗi chỉ thiếu/rỗng history để báo cáo gọn.
            notes = [n for n in r['notes'] if not n.startswith('history-commit.md đang rỗng')]
            if notes or r['excluded']:
                text = '; '.join(notes + [str(len(r['excluded']))+' tệp nguồn không dùng vì sai/lỗi'] if r['excluded'] else notes)
                out.write('| '+r['name']+'/'+r['version']+' | '+text.replace('|', '\\|').replace('\n', ' ')+' |\n')
        for key, error in failures.items():
            out.write('| '+key+' | CHƯA GỘP: '+error.replace('|', '\\|').replace('\n', ' ')+' |\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path.home()/'datasets/Dataset-Tong-hop')
    parser.add_argument('--package', help='Chỉ gộp thử một package, ví dụ humanqueen')
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    control = output.parent/(output.name+'_quan_ly')
    if output == Path('/') or output == Path.home() or output.name in SOURCES:
        raise SystemExit('Hãy chọn một thư mục đầu ra riêng.')
    control.mkdir(parents=True, exist_ok=True)
    with (control/'running.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Đã có một tiến trình đang gộp vào thư mục này.')
        return run(args, output, control)


def run(args, output, control):
    from huggingface_hub import HfApi
    api = HfApi()
    log('Nguồn: Duy + Long + Phúc + Tiến. Bỏ qua Đạt. Chỉ tạo dữ liệu local.')
    inv = inventory(api, control)
    groups = defaultdict(list)
    for record in inv['records']:
        groups[(record['name'], record['version_key'])].append(record)
    versions = defaultdict(list)
    for (name, ver), records in groups.items():
        # Cùng version PEP440 (1.0 và 1.0.0) chỉ giữ một tên thư mục ổn định.
        display = sorted({r['folder_version'] for r in records}, key=lambda v: (len(v), v))[0]
        versions[name].append((Version(ver), ver, display))
    for name in versions:
        versions[name].sort()
    selected_names = [nk(args.package)] if args.package else sorted(versions)
    if any(name not in versions for name in selected_names):
        raise SystemExit('Không thấy package trong bốn nguồn đã chọn.')
    state_dir = control/'done'
    state_dir.mkdir(exist_ok=True)
    results, failures = {}, {}
    earlier_report = control/'KET_QUA_GOP.json'
    if earlier_report.exists():
        failures.update(json.loads(earlier_report.read_text(encoding='utf-8')).get('failures', {}))
    for p in state_dir.glob('*.json'):
        result = json.loads(p.read_text(encoding='utf-8'))
        results[result['name']+'/'+result['version']] = result
    if output.exists() and not results and any(output.iterdir()) and not (control/'pending.json').exists():
        raise SystemExit('Thư mục đầu ra đã có dữ liệu không thuộc lượt gộp này; chọn --output mới.')
    output.mkdir(exist_ok=True)
    total = sum(len(versions[name]) for name in selected_names)
    log(f'Commit cố định: {inv["revision"]}\nCần xử lý {len(selected_names)} package / {total} version.')
    index = skipped = 0
    interrupted = False
    try:
        for name in selected_names:
            kind = 'multi_version' if len(versions[name]) >= 2 else 'single_version'
            for j, (_, version_key, display) in enumerate(versions[name]):
                index += 1
                key = name+'/'+display
                target = output/kind/name/display
                token = hashlib.sha256(key.encode()).hexdigest()
                log(f'[{index}/{total}] {kind}/{key}')
                if key in results:
                    if record_ok(target, results[key]):
                        failures.pop(key, None)
                        skipped += 1
                        log('  Đã có bản khớp hash; bỏ qua.')
                        continue
                    raise ValueError('Bản đã gộp bị sửa/thừa tệp: '+str(target)+'; chọn --output mới nếu muốn chạy lại.')
                if target.exists():
                    # Có thể ngắt sau rename nhưng trước khi ghi done: khôi phục từ pending.
                    pending = control/'pending.json'
                    p = json.loads(pending.read_text(encoding='utf-8')) if pending.exists() else {}
                    if p.get('key') == key and record_ok(target, p.get('result', {})):
                        results[key] = p['result']
                        failures.pop(key, None)
                        atomic_json(state_dir/(token+'.json'), p['result'])
                        continue
                    raise ValueError('Không ghi đè dữ liệu có sẵn: '+str(target))
                work = control/'working'
                # working là thư mục tạm riêng của tool, không phải dataset gốc.
                if work.exists():
                    shutil.rmtree(work)
                work.mkdir()
                dl = Downloads(work, inv['revision'])
                previous = versions[name][j-1][2] if j else None
                try:
                    result, ready = merge_version(groups[(name, version_key)], dl, work,
                        name, version_key, display, previous, inv['revision'])
                    atomic_json(control/'pending.json', {'key': key, 'result': result})
                    target.parent.mkdir(parents=True, exist_ok=True)
                    ready.rename(target)
                    results[key] = result
                    failures.pop(key, None)
                    atomic_json(state_dir/(token+'.json'), result)
                    log(f'  XONG: {len(result["files"])} tệp chính; '+('có upload_time' if result['upload_time'] else 'upload_time cần kiểm tra'))
                except OSError:
                    raise
                except Exception as exc:
                    failures[key] = f'{type(exc).__name__}: {exc}'
                    log('  CHƯA GỘP: '+failures[key][:400])
                finally:
                    # Xóa bản tải tạm của version đã xử lý, không giữ cả bốn bản tải.
                    if work.exists():
                        shutil.rmtree(work)
                if index % 20 == 0:
                    report(control, output, inv, results, failures)
    except KeyboardInterrupt:
        interrupted = True
        log('\nĐã dừng. Các version đã gộp được giữ; chạy lại cùng lệnh để tiếp tục.')
    except Exception:
        report(control, output, inv, results, failures)
        raise
    report(control, output, inv, results, failures)
    if args.package:
        demo_zip = control/f'Demo_{nk(args.package)}.zip'
        with zipfile.ZipFile(demo_zip, 'w', zipfile.ZIP_DEFLATED) as archive:
            for p in output.rglob('*'):
                if p.is_file() and nk(args.package) in p.relative_to(output).parts[1:2]:
                    archive.write(p, Path(output.name)/p.relative_to(output))
        log('ZIP xem thử: '+str(demo_zip))
    log(f'\nĐã gộp {len({r["name"] for r in results.values()})} package / {len(results)} version; bỏ qua {skipped} version đã xong.')
    log('Dataset: '+str(output)+'\nTổng kết: '+str(control/'TONG_KET_GOP.md'))
    if failures:
        log(f'Còn {len(failures)} version chưa gộp. Xem báo cáo, không coi là đã hoàn tất.')
    return 130 if interrupted else 2 if failures else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        log('DỪNG: '+str(exc))
        log('Dữ liệu nguồn không thay đổi. Chạy lại cùng lệnh sẽ kiểm tra và tiếp tục các version còn lại.')
        sys.exit(1)
