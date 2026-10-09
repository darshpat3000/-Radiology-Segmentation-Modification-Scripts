#!/usr/bin/env python3
"""
CLI_AssetDiscovery.py — fast asset discovery only (no header parsing, no pairing)

Writes: AssetsIndex.yaml (and optional JSON)

DISCOVERY POLICY
- DICOM series discovery is content-based (DICOM-like files), returning leaf series directories.
- NIfTI/seg discovery is extension-based (.nii/.nii.gz/.nrrd/.mha/.mhd/.npz).
- Parent folder-name filters are DISABLED by default. Enable explicitly with --use-parent-foldernames.

Typical usage
-------------
python CLI_AssetDiscovery.py scan --dicom-root "D:/Exports" --seg-root "D:/SegmentationsRoot" -v
python CLI_AssetDiscovery.py scan --dicom-root "D:/Exports" --nifti-root "D:/NiftiExports" --seg-root "D:/SegmentationsRoot"
"""

import argparse
import hashlib
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml
import json


# =========================
# Config & logging
# =========================

@dataclass
class DiscoverConfig:
    folder_sample_limit: int = 200
    dicom_magic_offset: int = 128
    dicom_magic: bytes = b"DICM"
    max_workers: int = 4  # reserved (discovery is mostly IO-bound but fast)

CONFIG = DiscoverConfig()
logger = logging.getLogger(__name__)

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )

def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")

def posix_rel(p: Path, start: Path) -> str:
    try:
        return Path(os.path.relpath(str(p), str(start))).as_posix()
    except Exception:
        return p.as_posix()

def relpath_safe(p: Path, start: Path) -> str:
    rp = posix_rel(p, start)
    return "." if rp in ("", ".") else rp

def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def file_stat_dict(p: Path) -> Dict:
    try:
        st = p.stat()
        return {
            "size_bytes": int(getattr(st, "st_size", 0) or 0),
            "mtime": datetime.fromtimestamp(getattr(st, "st_mtime", 0) or 0).astimezone().isoformat(timespec="seconds"),
        }
    except Exception:
        return {"size_bytes": 0, "mtime": ""}

def write_yaml(path: Path, doc: Dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True)
    return path

def write_json(path: Path, doc: Dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    return path


# =========================
# DICOM directory detection
# =========================

UNITED_IMAGING_PATTERNS = (
    "IM", "IMG", "I", "CT", "PET"
)

def is_probably_dicom_name(name: str) -> bool:
    n = name.strip().lower()
    if n.endswith(".dcm") or n.endswith(".dcm.gz"):
        return True
    # common vendor patterns
    base = Path(name).stem
    if base.isdigit():
        return True
    for pref in UNITED_IMAGING_PATTERNS:
        if base.upper().startswith(pref) and base[len(pref):].isdigit():
            return True
    return False

def has_dicom_magic(fp: Path) -> bool:
    try:
        with open(fp, "rb") as f:
            data = f.read(CONFIG.dicom_magic_offset + 4)
        if len(data) >= CONFIG.dicom_magic_offset + 4:
            return data[CONFIG.dicom_magic_offset:CONFIG.dicom_magic_offset + 4] == CONFIG.dicom_magic
    except Exception:
        pass
    return False

def folder_has_dicom(folder: Path, sample_limit: int) -> bool:
    tried = 0
    try:
        for f in folder.iterdir():
            if not f.is_file():
                continue
            # quick heuristics
            if is_probably_dicom_name(f.name):
                return True
            if has_dicom_magic(f):
                return True
            tried += 1
            if tried >= sample_limit:
                break
    except Exception:
        pass
    return False

def discover_dicom_leaf_series_dirs(dicom_root: Path, follow_symlinks: bool) -> List[Path]:
    """
    Returns leaf dirs containing DICOM-like content.
    """
    has_dicom: Set[Path] = set()
    for dirpath, _, _ in os.walk(dicom_root, followlinks=follow_symlinks):
        base = Path(dirpath)
        if folder_has_dicom(base, CONFIG.folder_sample_limit):
            try:
                has_dicom.add(base.resolve())
            except Exception:
                has_dicom.add(base)

    if not has_dicom:
        return []

    sorted_dirs = sorted(has_dicom, key=lambda p: len(p.parts))
    leaves: List[Path] = []
    for i, d in enumerate(sorted_dirs):
        dp = str(d) + os.sep
        is_leaf = True
        for j in range(i + 1, len(sorted_dirs)):
            c = sorted_dirs[j]
            if str(c).startswith(dp):
                is_leaf = False
                break
        if is_leaf:
            leaves.append(d)

    return leaves


# =========================
# File discovery (NIfTI / seg)
# =========================

NIFTI_SUFFIXES = (".nii.gz", ".nii")
SEG_SUFFIXES = (".nii.gz", ".nii", ".nrrd", ".mha", ".mhd", ".npz")

def is_nifti_file(p: Path) -> bool:
    n = p.name.lower()
    return any(n.endswith(s) for s in NIFTI_SUFFIXES)

def is_seg_file(p: Path) -> bool:
    n = p.name.lower()
    return any(n.endswith(s) for s in SEG_SUFFIXES)

def discover_files(root: Path, follow_symlinks: bool, predicate) -> List[Path]:
    out: List[Path] = []
    for dirpath, _, files in os.walk(root, followlinks=follow_symlinks):
        base = Path(dirpath)
        for fn in files:
            p = base / fn
            try:
                if p.is_file() and predicate(p):
                    out.append(p)
            except Exception:
                continue
    return out


# =========================
# Optional parent folder-name filter (explicit opt-in)
# =========================

def match_name_rules(text: str, includes: List[str], excludes: List[str], mode: str) -> bool:
    """
    Return True if text passes include/exclude filters.
    mode: substring | regex
    """
    import re
    t = text or ""
    if mode == "regex":
        for ex in excludes:
            if re.search(ex, t, flags=re.IGNORECASE):
                return False
        if includes:
            return any(re.search(inc, t, flags=re.IGNORECASE) for inc in includes)
        return True
    else:
        tl = t.lower()
        for ex in excludes:
            if ex.lower() in tl:
                return False
        if includes:
            return any(inc.lower() in tl for inc in includes)
        return True


# =========================
# Build AssetsIndex
# =========================

def make_asset_id(kind: str, root_label: str, path_rel: str) -> str:
    return f"{kind.upper()}:{sha1_text(f'{kind}|{root_label}|{path_rel}')[:16]}"

def scan_assets(
    dicom_root: Optional[Path],
    nifti_roots: List[Path],
    seg_roots: List[Path],
    out_dir: Path,
    follow_symlinks: bool,
    seg_follow_symlinks: bool,
    use_parent_foldernames: bool,
    include_parent_name: List[str],
    exclude_parent_name: List[str],
    parent_name_mode: str,
) -> Dict:
    assets: List[Dict] = []
    roots_meta: List[Dict] = []

    # DICOM series dirs
    if dicom_root:
        dicom_root = dicom_root.resolve()
        roots_meta.append({"label": "dicom", "kind": "dicom_root", "path": str(dicom_root)})
        series_dirs = discover_dicom_leaf_series_dirs(dicom_root, follow_symlinks=follow_symlinks)
        logger.info(f"[dicom] leaf series dirs: {len(series_dirs)}")

        for sd in series_dirs:
            sd_rel = relpath_safe(sd, dicom_root)
            parent_rel = relpath_safe(sd.parent, dicom_root)
            parent_name = sd.parent.name

            if use_parent_foldernames:
                if not match_name_rules(parent_name, include_parent_name, exclude_parent_name, parent_name_mode):
                    continue
            elif include_parent_name or exclude_parent_name:
                logger.warning("[filters] include/exclude parent-name provided but --use-parent-foldernames not set; ignoring filters.")

            asset = {
                "asset_id": make_asset_id("dicom_series_dir", "dicom", sd_rel),
                "kind": "dicom_series_dir",
                "root_label": "dicom",
                "root_path": str(dicom_root),
                "path_rel": sd_rel,
                "parent_rel": parent_rel,
                "parent_name": parent_name,  # informational only; pairing scripts should not use it by default
            }
            # series dir stats (fast)
            try:
                asset["dir_stats"] = {
                    "file_count_shallow": sum(1 for x in sd.iterdir() if x.is_file()),
                }
            except Exception:
                asset["dir_stats"] = {"file_count_shallow": None}
            assets.append(asset)

    # NIfTI image files (CT/PET exports etc.) — discovery only, no modality guess here
    for i, nr in enumerate(nifti_roots, 1):
        nr = nr.resolve()
        label = f"nifti{i}"
        roots_meta.append({"label": label, "kind": "nifti_root", "path": str(nr)})
        files = discover_files(nr, follow_symlinks=follow_symlinks, predicate=is_nifti_file)
        logger.info(f"[nifti:{label}] files: {len(files)}")
        for fp in files:
            rel = relpath_safe(fp, nr)
            parent_rel = relpath_safe(fp.parent, nr)
            assets.append({
                "asset_id": make_asset_id("nifti_image", label, rel),
                "kind": "nifti_image",
                "root_label": label,
                "root_path": str(nr),
                "path_rel": rel,
                "parent_rel": parent_rel,
                "file_stats": file_stat_dict(fp),
            })

    # segmentation-like files
    for i, sr in enumerate(seg_roots, 1):
        sr = sr.resolve()
        label = f"seg{i}"
        roots_meta.append({"label": label, "kind": "seg_root", "path": str(sr)})
        files = discover_files(sr, follow_symlinks=seg_follow_symlinks, predicate=is_seg_file)
        logger.info(f"[seg:{label}] files: {len(files)}")
        for fp in files:
            rel = relpath_safe(fp, sr)
            parent_rel = relpath_safe(fp.parent, sr)
            assets.append({
                "asset_id": make_asset_id("segmentation_file", label, rel),
                "kind": "segmentation_file",
                "root_label": label,
                "root_path": str(sr),
                "path_rel": rel,
                "parent_rel": parent_rel,
                "file_stats": file_stat_dict(fp),
            })

    # stats
    counts_by_kind: Dict[str, int] = {}
    for a in assets:
        counts_by_kind[a["kind"]] = counts_by_kind.get(a["kind"], 0) + 1

    parent_count = len({(a.get("root_label",""), a.get("parent_rel","")) for a in assets})

    doc = {
        "meta": {
            "tool": "CLI_AssetDiscovery.py",
            "generated_at": now_iso(),
            "out_dir": str(out_dir),
            "roots": roots_meta,
            "filters": {
                "use_parent_foldernames": bool(use_parent_foldernames),
                "include_parent_name": list(include_parent_name),
                "exclude_parent_name": list(exclude_parent_name),
                "parent_name_mode": parent_name_mode,
            }
        },
        "stats": {
            "total_assets": int(len(assets)),
            "unique_parent_buckets": int(parent_count),
            "by_kind": counts_by_kind,
        },
        "assets": sorted(assets, key=lambda r: (r.get("kind",""), r.get("root_label",""), r.get("parent_rel",""), r.get("path_rel",""))),
    }
    return doc


# =========================
# CLI
# =========================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Asset discovery (DICOM series dirs + NIfTI + segmentations).")
    sub = p.add_subparsers(dest="mode", required=True)

    s = sub.add_parser("scan", help="Scan roots and write AssetsIndex.yaml")
    s.add_argument("--dicom-root", type=Path, default=None, help="Root of DICOM exports")
    s.add_argument("--nifti-root", type=Path, action="append", default=[], help="Root of NIfTI images (repeatable)")
    s.add_argument("--seg-root", type=Path, action="append", default=[], help="Root of segmentation files (repeatable)")
    s.add_argument("--out-dir", type=Path, default=None, help='Output dir (default: "<dicom-root parent>/Directory Plan" else CWD)')
    s.add_argument("--follow-symlinks", action="store_true", help="Follow symlinks while scanning dicom/nifti")
    s.add_argument("--seg-follow-symlinks", action="store_true", help="Follow symlinks while scanning seg roots")

    # explicit opt-in filters
    s.add_argument("--use-parent-foldernames", action="store_true",
                   help="Enable filtering by parent folder names (disabled by default).")
    s.add_argument("--include-parent-name", action="append", default=[],
                   help="Keep only parents whose folder-name matches this rule (repeatable). Requires --use-parent-foldernames.")
    s.add_argument("--exclude-parent-name", action="append", default=[],
                   help="Exclude parents whose folder-name matches this rule (repeatable). Requires --use-parent-foldernames.")
    s.add_argument("--parent-name-mode", choices=("substring","regex"), default="substring",
                   help="How include/exclude rules are matched (default: substring).")

    s.add_argument("--formats", type=str, default="yaml", help='Comma-separated: yaml,json (default: yaml)')
    s.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    return p

def main():
    args = build_parser().parse_args()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    dicom_root = args.dicom_root
    nifti_roots = list(args.nifti_root or [])
    seg_roots = list(args.seg_root or [])

    if dicom_root is None and not nifti_roots and not seg_roots:
        logger.error("Provide at least one root: --dicom-root and/or --nifti-root and/or --seg-root")
        sys.exit(2)

    if args.out_dir is not None:
        out_dir = args.out_dir.resolve()
    else:
        if dicom_root is not None:
            out_dir = (dicom_root.resolve().parent / "Directory Plan").resolve()
        else:
            out_dir = (Path.cwd() / "Directory Plan").resolve()

    fmts = {x.strip().lower() for x in str(args.formats).split(",") if x.strip()}
    if not (fmts & {"yaml","json"}):
        logger.error('Invalid --formats. Use "yaml", "json", or both.')
        sys.exit(2)

    doc = scan_assets(
        dicom_root=dicom_root,
        nifti_roots=nifti_roots,
        seg_roots=seg_roots,
        out_dir=out_dir,
        follow_symlinks=bool(args.follow_symlinks),
        seg_follow_symlinks=bool(args.seg_follow_symlinks),
        use_parent_foldernames=bool(args.use_parent_foldernames),
        include_parent_name=list(args.include_parent_name or []),
        exclude_parent_name=list(args.exclude_parent_name or []),
        parent_name_mode=str(args.parent_name_mode),
    )

    written = []
    if "yaml" in fmts:
        written.append(write_yaml(out_dir / "AssetsIndex.yaml", doc))
    if "json" in fmts:
        written.append(write_json(out_dir / "AssetsIndex.json", doc))

    logger.info(f"[done] wrote {len(written)} file(s)")
    for p in written:
        print(f"  - {p}")

if __name__ == "__main__":
    main()