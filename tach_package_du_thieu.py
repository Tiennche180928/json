#!/usr/bin/env python3
"""Tách nguyên package theo dữ liệu đã gộp, không tải lại và không nhân đôi.

Cho phép thiếu upload_time, README, version change, history commit và tệp phát
hành gốc khi vẫn có src.zip đã được kiểm tra cùng metadata đúng định danh.
Chạy --apply để di chuyển; không có --apply thì chỉ thống kê. --undo hoàn tác.
"""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import zipfile

from packaging.utils import canonicalize_name, canonicalize_version
from packaging.version import Version

RULE = 'source_metadata_required_time_readme_optional_v1'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    tmp.replace(path)


def strict(data):
    def pairs(items):
        d = {}
        for k, v in items:
            if k in d:
                raise ValueError('Khóa JSON lặp: '+k)
            d[k] = v
        return d
    def bad(value):
        raise ValueError('JSON không hợp lệ: '+value)
    return json.loads(data, object_pairs_hook=pairs, parse_constant=bad)


def name_key(s):
    return str(canonicalize_name(s, validate=True))


def version_key(s):
    Version(s)
    return canonicalize_version(s)


def version_check(directory, package):
    errors, later = [], []
    meta_path, src = directory/'metadata.json', directory/'src.zip'
    if directory.is_symlink():
        return ['Thư mục version là liên kết'], []
    try:
        if meta_path.is_symlink() or not meta_path.is_file():
            raise ValueError('Không có metadata.json là tệp thường')
        if meta_path.stat().st_size > 32*1024**2:
            raise ValueError('metadata.json lớn bất thường')
        meta = strict(meta_path.read_text(encoding='utf-8-sig'))
        if not isinstance(meta, dict):
            raise ValueError('metadata.json không phải object')
        if name_key(meta.get('name')) != name_key(package):
            errors.append('Tên package trong metadata không khớp thư mục')
        if version_key(meta.get('version')) != version_key(directory.name):
            errors.append('Version trong metadata không khớp thư mục')
        source_info = meta.get('src', {})
        if not isinstance(source_info, dict) or source_info.get('identity_checked_in_core_metadata') is not True:
            errors.append('Mã nguồn chưa được xác minh định danh bằng core metadata')
        notes = meta.get('merge', {}).get('notes', [])
        for note in notes:
            if isinstance(note, str) and note.startswith('Mã nguồn cùng định danh có nội dung khác nhau'):
                errors.append('Mã nguồn cùng package/version khác nhau giữa các nguồn')
        if not meta.get('upload_time'):
            later.append('upload_time')
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        errors.append('Metadata cần kiểm tra: '+str(exc))
    if src.is_symlink() or not src.is_file() or src.stat().st_size == 0:
        errors.append('Không có src.zip là tệp thường, có nội dung')
    else:
        try:
            if not zipfile.is_zipfile(src):
                errors.append('src.zip không có cấu trúc ZIP hợp lệ')
        except OSError as exc:
            errors.append('Không đọc được src.zip: '+str(exc))
    readme = directory/'README.md'
    if readme.is_symlink() or not readme.is_file() or readme.stat().st_size == 0:
        later.append('README')
    # Dùng kết quả kiểm tra mã nguồn của bước gộp, không đọc/hash lại hàng GB.
    return sorted(set(errors)), later


def failed_versions(control):
    path = control/'KET_QUA_GOP.json'
    if not path.exists():
        return {}
    value = strict(path.read_text(encoding='utf-8-sig'))
    result = {}
    for key, reason in value.get('failures', {}).items():
        package = key.split('/', 1)[0]
        result.setdefault(package, []).append({'version': key, 'reason': reason})
    return result


def scan(source, output, failures):
    moves, seen = [], set()
    for group in ('multi_version', 'single_version'):
        root = source/group
        if not root.exists():
            continue
        if root.is_symlink() or not root.is_dir():
            raise ValueError('Nhóm nguồn không phải thư mục thường: '+str(root))
        for pkg in sorted(root.iterdir()):
            if pkg.name.startswith('.'):
                continue
            if pkg.is_symlink() or not pkg.is_dir():
                raise ValueError('Cần xem riêng đường dẫn ngoài cấu trúc package: '+str(pkg))
            name = name_key(pkg.name)
            if name in seen:
                raise ValueError('Package xuất hiện ở cả hai nhóm: '+name)
            seen.add(name)
            entries = list(pkg.iterdir())
            versions = [p for p in entries if p.is_dir() or p.is_symlink()]
            issues, later = [], []
            if not versions:
                issues.append({'version': '', 'errors': ['Không có thư mục version']})
            for version in sorted(versions):
                errors, missing = version_check(version, pkg.name)
                if errors:
                    issues.append({'version': version.name, 'errors': errors})
                if missing:
                    later.append({'version': version.name, 'fields': missing})
            if name in failures:
                issues.append({'version': '', 'errors': ['Có version chưa gộp được; xem báo cáo gộp'],
                               'failed_versions': failures[name]})
            status = 'can_kiem_tra' if issues else 'du_dieu_kien'
            # Đếm lại số thư mục version thực có sau gộp.
            outgroup = 'multi_version' if len(versions) >= 2 else 'single_version'
            target = output/status/outgroup/pkg.name
            moves.append({'package': pkg.name, 'status': status, 'group': outgroup,
                'from': str(pkg), 'to': str(target), 'version_count': len(versions),
                'issues': issues, 'bo_sung_sau': later, 'moved': False})
    return moves


def summary(state, control):
    counts = Counter(m['status'] for m in state['moves'])
    versions = Counter()
    for row in state['moves']:
        versions[row['status']] += row['version_count']
    lines = ['# Phân nhóm package sau gộp', '',
        '- Thiếu upload time hoặc README vẫn xếp nhóm đủ điều kiện.',
        '- History commit, version change và tệp phát hành gốc còn thiếu không dùng để loại khi mã nguồn và metadata đã được kiểm tra.',
        '- Một version có vấn đề thì chuyển cả package, giữ nguyên các version cùng nhau.',
        '- Đây là phân nhóm theo kết quả bước gộp và kiểm tra nhanh, chưa đánh giá độc/lành.', '',
        '| Nhóm | Package | Version hiện có |', '|---|---:|---:|']
    for key in ('du_dieu_kien', 'can_kiem_tra'):
        lines.append(f'| {key} | {counts[key]} | {versions[key]} |')
    lines.extend(['', '## Package cần kiểm tra', '', '| Package | Vấn đề |', '|---|---|'])
    for row in state['moves']:
        if row['issues']:
            detail = '; '.join((i['version']+': ' if i['version'] else '')+', '.join(i['errors']) for i in row['issues'])
            lines.append('| '+row['package']+' | '+detail.replace('|', '\\|').replace('\n', ' ')+' |')
    lines.extend(['', '## Upload time và README bổ sung sau', '', '| Package/version | Cần bổ sung |', '|---|---|'])
    for row in state['moves']:
        for missing in row['bo_sung_sau']:
            lines.append('| '+row['package']+'/'+missing['version']+' | '+', '.join(missing['fields'])+' |')
    (control/'KET_QUA_PHAN_NHOM.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f"ĐỦ: {counts['du_dieu_kien']} package / {versions['du_dieu_kien']} version.", flush=True)
    print(f"CẦN KIỂM TRA: {counts['can_kiem_tra']} package / {versions['can_kiem_tra']} version.", flush=True)
    print('Báo cáo: '+str(control/'KET_QUA_PHAN_NHOM.md'), flush=True)


def execute(state, plan_path, undo=False):
    rows = list(reversed(state['moves'])) if undo else state['moves']
    for index, row in enumerate(rows, 1):
        origin, target = Path(row['from']), Path(row['to'])
        if undo:
            origin, target = target, origin
        if origin.exists() and target.exists():
            raise ValueError('Hai vị trí cùng có dữ liệu; không ghi đè: '+str(target))
        if not origin.exists() and target.is_dir():
            row['moved'] = not undo
            continue
        if not origin.is_dir() or origin.is_symlink():
            raise ValueError('Không thấy package ở vị trí dự kiến: '+str(origin))
        target.parent.mkdir(parents=True, exist_ok=True)
        if origin.stat().st_dev != target.parent.stat().st_dev:
            raise ValueError('Hãy chọn đầu ra cùng phân vùng để di chuyển nhanh, không nhân đôi dữ liệu.')
        origin.rename(target)
        row['moved'] = not undo
        if index % 100 == 0:
            print(f'[{index}/{len(rows)}] '+row['package'], flush=True)
    state['undone'] = undo
    save(plan_path, state)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path.home()/'datasets/Dataset-Tong-hop')
    parser.add_argument('--output', type=Path, default=Path.home()/'datasets/Dataset-Tong-hop-phan-loai')
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--apply', action='store_true')
    actions.add_argument('--undo', action='store_true')
    args = parser.parse_args(argv)
    source, output = args.source.expanduser().resolve(), args.output.expanduser().resolve()
    if source == output or source in output.parents or output in source.parents:
        raise SystemExit('Nguồn và đích phải là hai thư mục riêng, không lồng nhau.')
    merge_control = source.parent/(source.name+'_quan_ly')
    merge_control.mkdir(parents=True, exist_ok=True)
    control = output.parent/(output.name+'_quan_ly')
    control.mkdir(parents=True, exist_ok=True)
    plan_path = control/'KE_HOACH_PHAN_NHOM.json'
    with (merge_control/'running.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Công cụ gộp còn đang chạy. Đợi gộp kết thúc rồi mới phân nhóm.')
        if plan_path.exists():
            state = strict(plan_path.read_text(encoding='utf-8'))
            if (state.get('source'), state.get('output'), state.get('rule')) != (str(source), str(output), RULE):
                raise ValueError('Kế hoạch có nguồn, đích hoặc quy tắc khác.')
            # Nếu chưa di chuyển hoặc đã hoàn tác xong, đọc lại để nhận metadata vừa sửa.
            if not args.undo and all(Path(r['from']).is_dir() and not Path(r['to']).exists() for r in state['moves']):
                state['moves'] = scan(source, output, failed_versions(merge_control))
                state['undone'] = False
                save(plan_path, state)
        else:
            if args.undo:
                raise ValueError('Chưa có lần phân nhóm để hoàn tác.')
            if output.exists() and any(output.iterdir()):
                raise ValueError('Đích đã có dữ liệu; chọn --output khác, không ghi đè.')
            if not source.is_dir():
                raise ValueError('Không thấy thư mục nguồn: '+str(source))
            print('Đang đọc metadata và kết quả kiểm tra đã có; không tải lại mã nguồn...', flush=True)
            moves = scan(source, output, failed_versions(merge_control))
            if not moves:
                raise ValueError('Không tìm thấy package trong multi_version/single_version.')
            state = {'rule': RULE, 'source': str(source), 'output': str(output), 'moves': moves, 'undone': False}
            save(plan_path, state)
        summary(state, control)
        if args.apply or args.undo:
            output.mkdir(parents=True, exist_ok=True)
            execute(state, plan_path, args.undo)
            print('ĐÃ HOÀN TÁC.' if args.undo else 'ĐÃ TÁCH XONG: '+str(output), flush=True)
            if not args.undo:
                print('Giữ các tệp quản lý. Muốn chạy lại công cụ gộp vào đường dẫn cũ thì hoàn tác bằng --undo trước.', flush=True)
        else:
            print('Mới thống kê, chưa di chuyển. Thêm --apply để tách các package.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nĐã dừng. Chạy lại cùng lệnh để tiếp tục; có thể dùng --undo để hoàn tác.')
    except Exception as exc:
        raise SystemExit('DỪNG: '+str(exc))
