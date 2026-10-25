#!/usr/bin/env python3
"""Khảo sát bốn nguồn trên Hugging Face. Chỉ đọc kho, chưa gộp/xóa/commit.

Python >= 3.10; pip install -U huggingface_hub packaging.
Chạy lại cùng --output sẽ tiếp tục lần khảo sát bị ngắt, tại cùng commit nguồn.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
import time
import zipfile

from packaging.utils import canonicalize_name, canonicalize_version, parse_sdist_filename, parse_wheel_filename
from packaging.version import Version, InvalidVersion

TOOL_VERSION = "1.0.0"
REPO = "lbms-dataset/package-pypi"
BASE = "Dataset-all-package"
# Các lớp thư mục đã quan sát trực tiếp trong kho, không suy đoán theo tên người.
LAYOUTS = {
    "Dataset-Duy": ("Dataset2/dataset_multi_version", "Dataset2/dataset_single_version"),
    "Dataset-Long": ("multi-version", "single-version"),
    "Dataset-Phuc": ("",),
    "Dataset-Tien": ("multi_version", "single_version"),
}
SOURCE_NAMES = {"Dataset-Duy": "DataDog", "Dataset-Long": "H2GLM", "Dataset-Phuc": "Hercule", "Dataset-Tien": "MalOSS"}
MAX_JSON = 16 * 1024 * 1024
MAX_HASH_DOWNLOAD = 32 * 1024 * 1024
ARCHIVES = (".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".tar", ".whl", ".zip")
SOURCE_ZIPS = {"src.zip", "source.zip", "sourcecode.zip", "source_code.zip"}
ANALYSIS_FILES = {"version_changes.md", "version_change.md", "package_conclusion.md", "package_conclution.md", "history-commit.md", "commits.json"}


def log(message):
    print(message, flush=True)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Khóa JSON lặp: {key}")
            result[key] = value
        return result
    def bad(value):
        raise ValueError(f"Giá trị JSON không hợp lệ: {value}")
    obj = json.loads(data.decode("utf-8-sig"), object_pairs_hook=pairs, parse_constant=bad)
    if not isinstance(obj, dict):
        raise ValueError("JSON phải là một object")
    return obj


def safe_path(path):
    return (isinstance(path, str) and path and not path.startswith("/")
            and "\\" not in path and not any(ord(c) < 32 for c in path)
            and all(p not in ("", ".", "..") for p in path.split("/")))


def sha(value):
    return value.lower() if isinstance(value, str) and re.fullmatch(r"[a-fA-F0-9]{64}", value) else None


def version_key(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Version không phải chuỗi hợp lệ")
    Version(value)
    return canonicalize_version(value)


def name_key(value):
    if not isinstance(value, str):
        raise ValueError("Tên package không phải chuỗi")
    return str(canonicalize_name(value, validate=True))


def empty(value):
    return value is None or (isinstance(value, str) and value.strip().lower() in ("", "unknown", "null", "none", "n/a"))


def timestamp(value):
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith(" UTC"):
        text = text[:-4] + "+00:00"
    try:
        instant = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if instant.tzinfo is None or instant.utcoffset() is None:
            return None
        return instant.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    except ValueError:
        return None


def entry_info(entry):
    lfs = getattr(entry, "lfs", None)
    digest = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
    return {"path": entry.path, "size": entry.size, "blob_id": entry.blob_id,
            "sha256": sha(digest), "sha256_method": "HUB_LFS_XET" if sha(digest) else None}


def locate(path):
    """Trả về (nguồn, lớp, package, version, đường dẫn trong version)."""
    if not safe_path(path):
        return None, "DUONG_DAN_KHONG_HOP_LE"
    for source, layouts in LAYOUTS.items():
        prefix = f"{BASE}/{source}/"
        if not path.startswith(prefix):
            continue
        rel = path[len(prefix):]
        if rel in ("README.md", ".gitkeep"):
            return None, "MO_TA_NGUON"
        for layout in layouts:
            lead = layout + "/" if layout else ""
            if not rel.startswith(lead):
                continue
            parts = rel[len(lead):].split("/")
            if len(parts) < 3:
                return None, "TEP_NGOAI_VERSION"
            pkg, ver = parts[:2]
            try:
                name_key(pkg)
                version_key(ver)
            except (ValueError, InvalidVersion):
                return None, "CHUA_NHAN_DIEN_PACKAGE_VERSION"
            return (source, layout, pkg, ver, "/".join(parts[2:])), None
        return None, "LOP_THU_MUC_CHUA_HO_TRO"
    return None, "NGOAI_BON_NGUON"


def group_inventory(files):
    buckets, unassigned = {}, []
    for file in files:
        found, why = locate(file["path"])
        if found is None:
            unassigned.append({**file, "reason": why})
            continue
        source, layout, pkg, ver, rel = found
        directory = file["path"][:-(len(rel) + 1)]
        record = buckets.setdefault(directory, {"source": source, "source_family": SOURCE_NAMES[source],
            "layout": layout, "directory": directory, "folder_name": pkg, "folder_version": ver,
            "name": name_key(pkg), "version_key": version_key(ver), "files": []})
        record["files"].append({**file, "relative_path": rel})
    return [buckets[k] for k in sorted(buckets)], unassigned


def identity_check(meta, record, issues):
    identities = []
    for label, obj in [("root", meta)] + [(k, meta[k]) for k in ("info", "pkg_info", "package_metadata_raw") if k in meta]:
        if not isinstance(obj, dict):
            issues.append({"code": "SAI_KIEU_METADATA", "detail": label})
            continue
        for key in ("name", "Name", "normalized_name"):
            if not empty(obj.get(key)):
                identities.append((label + "." + key, "name", obj[key]))
        for key in ("version", "Version"):
            if not empty(obj.get(key)):
                identities.append((label + "." + key, "version", obj[key]))
    for field, kind, value in identities:
        try:
            actual = name_key(value) if kind == "name" else version_key(value)
            expected = record["name"] if kind == "name" else record["version_key"]
            if actual != expected:
                issues.append({"code": "SAI_DINH_DANH", "detail": f"{field}={value!r}; thư mục={record['folder_name']}/{record['folder_version']}"})
        except (ValueError, InvalidVersion):
            issues.append({"code": "DINH_DANH_KHONG_HOP_LE", "detail": field})
    if not any(k == "name" for _, k, _ in identities) or not any(k == "version" for _, k, _ in identities):
        issues.append({"code": "THIEU_DINH_DANH_METADATA", "detail": "Chưa đủ cả name và version trong JSON"})


def is_original(file, record):
    rel = file["relative_path"]
    lower = rel.lower()
    if not lower.endswith(ARCHIVES) or lower.rsplit("/", 1)[-1] in SOURCE_ZIPS:
        return False
    # Archive trong mã nguồn có thể chỉ là tài nguyên; chỉ nhận archive con
    # khi tên phát hành khớp package/version, không nhận mọi ZIP lồng nhau.
    if "/" not in rel:
        return True
    if rel.split("/", 1)[0].lower() not in ("sourcecode", "source_code", "source", "src", "original", "artifacts"):
        return False
    return archive_identity(file, record) is True


def archive_identity(file, record):
    filename = PurePosixPath(file["relative_path"]).name
    try:
        if filename.lower().endswith(".whl"):
            n, v, *_ = parse_wheel_filename(filename)
        elif filename.lower().endswith((".tar.gz", ".zip")):
            n, v = parse_sdist_filename(filename)
        else:
            return None
        return name_key(str(n)) == record["name"] and version_key(str(v)) == record["version_key"]
    except ValueError:
        return None


def distribution_claims(meta):
    claims = []
    for key in ("files", "distributions", "urls"):
        items = meta.get(key, [])
        if isinstance(items, list):
            claims += [(f"{key}[{i}]", x) for i, x in enumerate(items) if isinstance(x, dict)]
    enrich = meta.get("pypi_enrichment")
    if isinstance(enrich, dict):
        claims.append(("pypi_enrichment", enrich))
    claims.append(("root", meta))
    return claims


def analyze(record, meta, read_file):
    r = {**record, "issues": [], "notes": [], "metadata_identity_checked": False,
         "original_files": [], "upload_times": [], "metadata_error": None}
    issues = r["issues"]
    if meta is None:
        issues.append({"code": "THIEU_METADATA", "detail": "Không có metadata.json ở ngay trong version"})
        meta = {}
    else:
        identity_check(meta, r, issues)
        r["metadata_identity_checked"] = not any(x["code"] in ("SAI_DINH_DANH", "DINH_DANH_KHONG_HOP_LE", "THIEU_DINH_DANH_METADATA", "SAI_KIEU_METADATA") for x in issues)
    for key in ("files", "distributions", "urls"):
        if key in meta and (not isinstance(meta[key], list) or any(not isinstance(x, dict) for x in meta[key])):
            issues.append({"code": "SAI_KIEU_METADATA", "detail": key + " phải là danh sách object"})
    sources = [f for f in r["files"] if f["relative_path"].lower() in SOURCE_ZIPS and f["size"] > 0]
    expanded = [f for f in r["files"] if f["relative_path"].split("/", 1)[0].lower() in ("sourcecode", "source_code", "source", "src") and "/" in f["relative_path"] and f["size"] > 0]
    r["has_src_zip"] = bool(sources)
    r["expanded_source_files"] = len(expanded)
    if not sources and not expanded:
        issues.append({"code": "CHUA_THAY_SRC", "detail": "Chưa thấy src.zip hoặc tệp trong thư mục mã nguồn"})
    elif not sources:
        r["notes"].append("MA_NGUON_DANG_THU_MUC: cần đóng ZIP khi lập bản tổng hợp")
    for f in r["files"]:
        if not is_original(f, r):
            continue
        f = dict(f)
        if not f["sha256"]:
            if f["size"] <= MAX_HASH_DOWNLOAD:
                data = read_file(f, MAX_HASH_DOWNLOAD)
                f["sha256"] = hashlib.sha256(data).hexdigest()
                f["sha256_method"] = "TINH_TU_BYTES_GOC"
            else:
                r["notes"].append("CHUA_CO_SHA256_GOC: " + f["relative_path"])
        filename_identity = archive_identity(f, r)
        f["filename_identity_matches"] = filename_identity
        if filename_identity is False:
            issues.append({"code": "TEN_ARCHIVE_SAI_DINH_DANH", "detail": f["relative_path"]})
        elif filename_identity is None:
            r["notes"].append("TEN_ARCHIVE_CHUA_XAC_MINH: " + f["relative_path"])
        r["original_files"].append(f)
    if not r["original_files"]:
        r["notes"].append("KHONG_CO_ARCHIVE_GOC: chưa thể khử trùng bằng SHA256 phát hành gốc")
    original_by_hash = {f["sha256"]: f for f in r["original_files"] if f["sha256"]}
    by_name = defaultdict(list)
    for f in r["original_files"]:
        by_name[PurePosixPath(f["relative_path"]).name].append(f)
    for where, claim in distribution_claims(meta):
        digests = claim.get("digests")
        claimed_hash = sha(claim.get("original_sha256")) or sha(claim.get("sha256")) or (sha(digests.get("sha256")) if isinstance(digests, dict) else None)
        filename = claim.get("original_filename") or claim.get("filename")
        if isinstance(filename, str):
            filename = filename.replace("\\", "/").rsplit("/", 1)[-1]
        elif filename is not None:
            issues.append({"code": "SAI_KIEU_METADATA", "detail": where + ".filename phải là chuỗi"})
            filename = None
        match = original_by_hash.get(claimed_hash) if claimed_hash else None
        candidates = by_name.get(filename, []) if filename else []
        generic_is_src = (where == "root" and not sha(claim.get("original_sha256"))
                          and claimed_hash and any(f["sha256"] == claimed_hash for f in sources))
        if generic_is_src:
            r["notes"].append("SHA256_CHUNG_THUOC_SRC_ZIP: không dùng làm SHA256 archive gốc")
            match = None
        if claimed_hash and candidates and not generic_is_src and all(f["sha256"] and f["sha256"] != claimed_hash for f in candidates):
            issues.append({"code": "SHA256_GOC_MAU_THUAN", "detail": f"{where}: {filename}"})
        if where == "root" and sha(claim.get("original_sha256")) and not filename and len(r["original_files"]) == 1:
            actual = r["original_files"][0]["sha256"]
            if actual and actual != claimed_hash:
                issues.append({"code": "SHA256_GOC_MAU_THUAN", "detail": "root.original_sha256"})
        raw_time = claim.get("upload_time_iso_8601") or claim.get("upload_time")
        if not empty(raw_time):
            value = timestamp(raw_time)
            kind = "HASH_LIEN_KET_METADATA" if match else "CHUA_LIEN_KET_SHA256"
            status = str(claim.get("upload_time_status", "")).lower()
            if any(s in status for s in ("candidate", "name_version_only", "filename_only", "unverified")) or claim.get("local_artifact_match") in ("name_version_only", "name_version_filename_candidate"):
                kind = "UNG_VIEN"
            r["upload_times"].append({"field": where + ".upload_time", "raw": raw_time,
                "value_utc": value, "sha256": claimed_hash, "kind": kind if value else "THOI_GIAN_KHONG_HOP_LE"})
        if not empty(claim.get("upload_time_candidate")):
            r["upload_times"].append({"field": where + ".upload_time_candidate", "raw": claim["upload_time_candidate"],
                "value_utc": timestamp(claim["upload_time_candidate"]), "sha256": claimed_hash, "kind": "UNG_VIEN"})
    supplement = meta.get("supplement")
    if isinstance(supplement, dict):
        t = supplement.get("upload_time")
        if isinstance(t, dict) and not empty(t.get("value")):
            links = {sha(x.get("sha256")) for x in meta.get("files", []) if isinstance(x, dict)} if isinstance(meta.get("files"), list) else set()
            links.discard(None)
            bound = links.intersection(original_by_hash)
            # Một thời gian ở cấp version chỉ gắn cho artifact khi duy nhất một
            # artifact được nhận diện; không áp nó cho nhiều wheel/sdist khác nhau.
            linked_hash = next(iter(bound)) if len(bound) == 1 and len(links) == 1 else None
            value = timestamp(t["value"])
            r["upload_times"].append({"field": "supplement.upload_time", "raw": t["value"],
                "value_utc": value, "sha256": linked_hash, "declared_source": t.get("source"),
                "kind": ("HASH_LIEN_KET_METADATA" if linked_hash else "CHUA_LIEN_KET_SHA256") if value else "THOI_GIAN_KHONG_HOP_LE"})
    expected_src = sha(meta.get("src_zip_sha256"))
    if expected_src and len(sources) == 1 and sources[0]["sha256"] and expected_src != sources[0]["sha256"]:
        issues.append({"code": "SHA256_SRC_MAU_THUAN", "detail": sources[0]["relative_path"]})
    r["has_hash_bound_time"] = any(t["kind"] == "HASH_LIEN_KET_METADATA" for t in r["upload_times"])
    if not r["has_hash_bound_time"]:
        r["notes"].append("UPLOAD_TIME_CHUA_LIEN_KET_SHA256_GOC")
    r["status"] = "CAN_KIEM_TRA" if issues else "CHO_BO_SUNG_DOI_CHIEU" if (not r["has_hash_bound_time"] or not r["original_files"]) else "KHOP_KHAO_SAT"
    return r


def aggregate(records):
    groups, hash_index = defaultdict(list), defaultdict(list)
    for r in records:
        groups[(r["name"], r["version_key"])].append(r)
        for f in r.get("original_files", []):
            if f["sha256"]:
                hash_index[f["sha256"]].append({"name": r["name"], "version_key": r["version_key"], "source": r["source"], "directory": r["directory"], "path": f["path"]})
    conflicts, duplicate_hashes = [], []
    for digest, refs in sorted(hash_index.items()):
        identities = {(x["name"], x["version_key"]) for x in refs}
        if len(identities) > 1:
            conflicts.append({"code": "SHA256_TRUNG_NHIEU_DINH_DANH", "sha256": digest, "references": refs})
        if len(refs) > 1:
            duplicate_hashes.append({"sha256": digest, "references": refs})
    result = []
    for (name, ver), members in sorted(groups.items()):
        times = defaultdict(list)
        for r in members:
            for t in r.get("upload_times", []):
                if t["kind"] == "HASH_LIEN_KET_METADATA":
                    times[t["sha256"]].append({**t, "directory": r["directory"]})
        time_conflicts = []
        for digest, claims in times.items():
            if len({t["value_utc"] for t in claims}) > 1:
                time_conflicts.append({"code": "THOI_GIAN_KHAC_NHAU_CAN_DOI_CHIEU", "sha256": digest, "claims": claims})
        displayed = sorted({r["folder_version"] for r in members}, key=lambda v: (Version(v), v))[0]
        hashes = sorted({f["sha256"] for r in members for f in r.get("original_files", []) if f["sha256"]})
        result.append({"name": name, "version": displayed, "version_key": ver,
            "sources": sorted({r["source"] for r in members}), "directories": [r["directory"] for r in members],
            "original_sha256": hashes, "time_conflicts": time_conflicts,
            "source_statuses": {r["directory"]: r["status"] for r in members},
            "needs_review": any(r["status"] == "CAN_KIEM_TRA" for r in members) or bool(time_conflicts)
                or any(any(ref["directory"] == r["directory"] for c in conflicts for ref in c["references"]) for r in members)})
    versions_per_package = Counter(g["name"] for g in result)
    for g in result:
        g["destination_group"] = "multi_version" if versions_per_package[g["name"]] >= 2 else "single_version"
    return result, duplicate_hashes, conflicts


class Reader:
    def __init__(self, repo, revision, cache):
        self.repo, self.revision, self.cache = repo, revision, Path(cache)

    def __call__(self, file, maximum):
        from huggingface_hub import hf_hub_download
        if not safe_path(file["path"]) or file["size"] > maximum:
            raise ValueError("Tệp vượt giới hạn hoặc đường dẫn không hợp lệ")
        path = hf_hub_download(repo_id=self.repo, repo_type="dataset", revision=self.revision,
            filename=file["path"], cache_dir=str(self.cache), etag_timeout=20)
        with open(path, "rb") as stream:
            data = stream.read(maximum + 1)
        if len(data) != file["size"] or len(data) > maximum:
            raise ValueError("Kích thước tệp tải về không khớp danh mục")
        if file.get("sha256"):
            if hashlib.sha256(data).hexdigest() != file["sha256"]:
                raise ValueError("SHA256 bytes tải về không khớp Hub")
        elif file.get("blob_id"):
            actual = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            if actual != file["blob_id"]:
                raise ValueError("Git blob tải về không khớp commit đã chốt")
        return data


def short_error(exc):
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    # Không đưa token, header hoặc URL ký sẵn vào log/báo cáo.
    return f"{type(exc).__name__}" + (f"; HTTP {status}" if status else "")


def inspect_record(record, reader):
    metadata = next((f for f in record["files"] if f["relative_path"] == "metadata.json"), None)
    if metadata is None:
        return analyze(record, None, reader)
    try:
        data = reader(metadata, MAX_JSON)
        meta = strict_json(data)
    except (ValueError, UnicodeError) as exc:
        r = analyze({**record, "files": []}, None, reader)
        explanation = str(exc)[:300]
        r.update(files=record["files"], status="CAN_KIEM_TRA", metadata_error=explanation)
        r["issues"] = [{"code": "METADATA_JSON_LOI", "detail": explanation}]
        return r
    return analyze(record, meta, reader)


def with_retry(fn):
    for attempt in range(5):
        try:
            return fn()
        except (ValueError, UnicodeError):
            raise
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            network_exception = isinstance(exc, (OSError, ConnectionError, TimeoutError)) or type(exc).__module__.split('.')[0] in ('httpx', 'httpx2', 'requests', 'urllib3')
            transient = status in (408, 429, 500, 502, 503, 504) or (status is None and network_exception)
            if not transient or attempt == 4:
                raise
            seconds = min(60, 5 * 2**attempt)
            response = getattr(exc, "response", None)
            if status == 429 and response is not None:
                try:
                    seconds = min(300, max(seconds, int(response.headers.get("Retry-After", seconds))))
                except (TypeError, ValueError):
                    pass
            log(f"Lỗi mạng/giới hạn truy cập ({short_error(exc)}); thử lại sau {seconds}s.")
            # In nhịp chờ để không trông như chương trình đứng im.
            while seconds > 0:
                time.sleep(min(30, seconds))
                seconds -= min(30, seconds)
                if seconds:
                    log(f"  Vẫn đang chờ; còn {seconds}s.")


def export_reports(run, state, files, records, unassigned, complete, expected):
    groups, duplicates, conflicts = aggregate(records)
    per_source = []
    for source in LAYOUTS:
        source_files = [f for f in files if f["path"].startswith(f"{BASE}/{source}/")]
        subset = [r for r in records if r["source"] == source]
        per_source.append({"source": source, "family": SOURCE_NAMES[source], "files": len(source_files),
            "bytes": sum(f["size"] for f in source_files), "version_directories": sum(1 for r in expected if r["source"] == source),
            "packages_by_folder": len({r["name"] for r in expected if r["source"] == source}),
            "metadata_checked": len(subset), "statuses": dict(Counter(r["status"] for r in subset))})
    package_versions = Counter(g["name"] for g in groups)
    summary = {"complete": complete, "source_revision": state["source_revision"], "repo": state["repo"],
        "sources": per_source, "input_version_directories": len(expected), "checked_version_directories": len(records),
        "candidate_packages": len(package_versions), "candidate_versions": len(groups),
        "single_version_packages": sum(n == 1 for n in package_versions.values()),
        "multi_version_packages": sum(n >= 2 for n in package_versions.values()),
        "unique_original_sha256": len({f["sha256"] for r in records for f in r.get("original_files", []) if f["sha256"]}),
        "duplicate_original_occurrences": sum(len(x["references"]) - 1 for x in duplicates),
        "cross_identity_hash_conflicts": len(conflicts), "unassigned_files": len(unassigned)}
    atomic_json(run / "KET_QUA_KHAO_SAT.json", {"tool_version": TOOL_VERSION, "summary": summary, "records": records,
        "duplicate_original_hashes": duplicates, "cross_identity_conflicts": conflicts, "unassigned_files": unassigned})
    atomic_json(run / "UNG_VIEN_GOP.json", {"complete": complete, "repo": state["repo"],
        "source_revision": state["source_revision"], "kind": "INVENTORY_ONLY_NOT_EXECUTABLE_MERGE_PLAN", "groups": groups})
    heading = "HOÀN TẤT KHẢO SÁT" if complete else "CHƯA KHẢO SÁT XONG — số bên dưới chỉ là phần đã đọc"
    lines = ["# Khảo sát trước khi gộp bốn nguồn", "", f"**{heading}**", "",
        f"- Kho: `{state['repo']}`", f"- Commit nguồn: `{state['source_revision']}`", "- Chưa gộp, xóa, ghi đè hoặc commit tệp lên Hugging Face.",
        "- Không kiểm tra README/commit/version change; không đánh giá độc/lành.",
        "- Chỉ kiểm tra danh mục, JSON, định danh, SHA và liên kết thời gian. Chưa mở archive hoặc xác minh nội dung code.",
        "- SHA256_GOC lấy từ nội dung lưu trên Hub; Git blob SHA1 không được dùng thay SHA256.",
        "- HASH_LIEN_KET_METADATA nghĩa là thời gian do metadata khai báo có hash khớp archive; chưa phải truy vấn độc lập lại PyPI/BigQuery.", "",
        "| Nguồn | Package theo thư mục | Thư mục version | Tệp | Metadata đã kiểm tra |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {s['source']} ({s['family']}) | {s['packages_by_folder']} | {s['version_directories']} | {s['files']} | {s['metadata_checked']} |" for s in per_source]
    lines += ["", "## Số liệu ứng viên sau nhóm tên/version", "", f"- Package: **{summary['candidate_packages']}**.",
        f"- Version: **{summary['candidate_versions']}**.", f"- Package 1 version: **{summary['single_version_packages']}**.",
        f"- Package từ 2 version: **{summary['multi_version_packages']}**.",
        f"- SHA256 archive gốc khác nhau: **{summary['unique_original_sha256']}**.",
        f"- Lượt archive trùng SHA256: **{summary['duplicate_original_occurrences']}**.",
        "", "Các số trên bao gồm ứng viên cần kiểm tra; chưa phải số dataset đã đạt hoặc đã được gộp.",
        "Tên nguồn H2GLM/Hercule/DataDog/MalOSS là phân công thư mục; xuất xứ cụ thể từng mẫu vẫn cần giữ theo metadata.",
        "", "## Cách đọc kết quả", "", "- `KHOP_KHAO_SAT`: các trường đã kiểm tra không mâu thuẫn; chưa thẩm định nội dung archive.",
        "- `CHO_BO_SUNG_DOI_CHIEU`: chưa thấy archive gốc hoặc thời gian chưa gắn được với SHA256 gốc.",
        "- `CAN_KIEM_TRA`: JSON/định danh/hash hoặc cấu trúc có vấn đề.",
        "- `UNG_VIEN_GOP.json` lưu các version cùng tên để chuẩn bị bước gộp; không chứa lệnh ghi lên kho.",
        "", "Gửi lại `GUI_LAI_KET_QUA.zip` để đối chiếu trước khi tạo công cụ gộp thực tế."]
    (run / "TONG_QUAN_KHAO_SAT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    detail = ["# Những mục cần kiểm tra trước khi gộp", "", f"Commit: `{state['source_revision']}`", "",
        "| Thư mục version | Trạng thái | Cần kiểm tra / bổ sung |", "|---|---|---|"]
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")
    for r in records:
        issues = [x["code"] + ": " + x["detail"] for x in r["issues"]] + r["notes"]
        if issues:
            detail.append(f"| {cell(r['directory'])} | {r['status']} | {cell('; '.join(issues))} |")
    for g in groups:
        for conflict in g["time_conflicts"]:
            detail.append(f"| {g['name']}/{g['version']} | CAN_KIEM_TRA | {conflict['code']}; xem JSON để giữ đầy đủ các thời gian |")
    for conflict in conflicts:
        detail.append(f"| {conflict['sha256']} | CAN_KIEM_TRA | SHA256 gốc được xếp vào nhiều định danh; không tự gộp |")
    detail += ["", "## Tệp chưa xếp được vào version", "", "| Tệp | Lý do |", "|---|---|"]
    detail += [f"| {cell(f['path'])} | {f['reason']} |" for f in unassigned]
    (run / "CAN_KIEM_TRA.md").write_text("\n".join(detail) + "\n", encoding="utf-8")
    with zipfile.ZipFile(run / "GUI_LAI_KET_QUA.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for filename in ("TONG_QUAN_KHAO_SAT.md", "CAN_KIEM_TRA.md", "KET_QUA_KHAO_SAT.json", "UNG_VIEN_GOP.json"):
            archive.write(run / filename, arcname=filename)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", type=Path, default=Path.home() / "datasets/Khao_sat_gop_HF")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--new", action="store_true", help="Khảo sát mới tại commit hiện tại, không tiếp tục lần dở")
    args = parser.parse_args(argv)
    from huggingface_hub import HfApi
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    latest = output / "LATEST.txt"
    run, state = None, None
    if latest.exists() and not args.new:
        label = latest.read_text(encoding="utf-8").strip()
        if re.fullmatch(r"[0-9]{8}T[0-9]{6}_[0-9]{6}Z", label):
            candidate = output / label
            state_path = candidate / "state.json"
            if state_path.is_file():
                prior = strict_json(state_path.read_bytes())
                if prior.get("repo") == args.repo and prior.get("requested_revision") == args.revision and prior.get("status") != "complete":
                    run, state = candidate, prior
    api = HfApi()
    if state is None:
        info = with_retry(lambda: api.repo_info(args.repo, repo_type="dataset", revision=args.revision))
        revision = info.sha
        if not re.fullmatch(r"[a-f0-9]{40}", revision):
            raise RuntimeError("Chưa chốt được commit nguồn")
        run = output / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        run.mkdir()
        state = {"tool_version": TOOL_VERSION, "repo": args.repo, "requested_revision": args.revision,
            "source_revision": revision, "status": "reading_inventory"}
        atomic_json(run / "state.json", state)
        latest.write_text(run.name + "\n", encoding="utf-8")
    elif state.get("tool_version") != TOOL_VERSION:
        raise RuntimeError("Lần dở dùng phiên bản tool khác. Chạy --new để khảo sát mới.")
    log(f"Kết quả: {run}\nCommit nguồn: {state['source_revision']}\nCHỈ ĐỌC: chưa gộp/xóa/commit lên Hugging Face.")
    inventory_path = run / "inventory.json"
    if inventory_path.exists():
        files = strict_json(inventory_path.read_bytes())["files"]
    else:
        files = []
        for source in LAYOUTS:
            log(f"Đọc toàn bộ danh mục {source} (có phân trang tự động)...")
            def list_source(source=source):
                from huggingface_hub import RepoFile
                return [entry_info(e) for e in api.list_repo_tree(args.repo, repo_type="dataset", revision=state["source_revision"],
                    path_in_repo=f"{BASE}/{source}", recursive=True, expand=False) if isinstance(e, RepoFile)]
            entries = with_retry(list_source)
            if not entries:
                raise RuntimeError(f"{source} trống: chưa thể coi khảo sát đủ cả bốn nguồn")
            files.extend(entries)
            log(f"  {len(entries)} tệp.")
        atomic_json(inventory_path, {"files": files})
    expected, unassigned = group_inventory(files)
    if not expected or any(not any(r["source"] == source for r in expected) for source in LAYOUTS):
        atomic_json(run / "LOI_CAU_TRUC.json", {"unassigned_files": unassigned})
        raise RuntimeError("Có nguồn chưa nhận diện được version. Xem LOI_CAU_TRUC.json; chưa gộp gì.")
    records_dir = run / "records"
    records_dir.mkdir(exist_ok=True)
    reader = Reader(args.repo, state["source_revision"], output / "cache_hf")
    state["status"] = "reading_metadata"
    atomic_json(run / "state.json", state)
    def one(r):
        key = hashlib.sha256(r["directory"].encode()).hexdigest()
        cached = records_dir / (key + ".json")
        if cached.exists():
            return strict_json(cached.read_bytes())
        try:
            result = with_retry(lambda: inspect_record(r, reader))
            atomic_json(cached, result)
            return result
        except Exception as exc:
            return {**r, "status": "CAN_KIEM_TRA", "issues": [{"code": "LOI_DOC_TEP", "detail": short_error(exc)}],
                "notes": [], "original_files": [], "upload_times": [], "retry_needed": True}
    records = []
    interrupted = False
    pool = ThreadPoolExecutor(max_workers=args.workers)
    try:
        # executor.map nạp công việc, chỉ có tối đa --workers lượt tải đồng thời.
        for index, result in enumerate(pool.map(one, expected), 1):
            records.append(result)
            log(f"[{index}/{len(expected)}] {result['status']} | {result['directory']}")
            with (run / "tien_do.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"directory": result["directory"], "status": result["status"]}, ensure_ascii=False) + "\n")
    except KeyboardInterrupt:
        interrupted = True
        log("Đang dừng; giữ kết quả đã đọc. Chạy lại cùng lệnh để tiếp tục.")
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    complete = not interrupted and len(records) == len(expected) and not any(r.get("retry_needed") for r in records)
    summary = export_reports(run, state, files, records, unassigned, complete, expected)
    state["status"] = "complete" if complete else "partial"
    atomic_json(run / "state.json", state)
    log("\n" + ("HOÀN TẤT KHẢO SÁT" if complete else "CHƯA ĐỌC XONG — chạy lại cùng lệnh"))
    log(f"Đã đọc {len(records)}/{len(expected)} thư mục version; {summary['candidate_packages']} package ứng viên; {summary['candidate_versions']} version ứng viên.")
    log(f"Gửi lại tệp: {run / 'GUI_LAI_KET_QUA.zip'}")
    log("Chưa có thay đổi nào trên kho Hugging Face.")
    return 0 if complete else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("Đã dừng. Chạy lại cùng lệnh để tiếp tục; chưa ghi lên Hugging Face.")
        raise SystemExit(130)
    except Exception as error:
        log("DỪNG: " + short_error(error))
        if isinstance(error, (ValueError, RuntimeError)):
            log(str(error))
        log("Chưa gộp/xóa/commit lên Hugging Face.")
        raise SystemExit(1)
