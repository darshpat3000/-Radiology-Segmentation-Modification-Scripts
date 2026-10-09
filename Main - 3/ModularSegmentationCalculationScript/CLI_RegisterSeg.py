#!/usr/bin/env python3
"""
CLI_RegisterSeg.py — Register external segmentations with existing PET/CT pairs.

Adds a segmentation record to PairIndex.yaml so that slicer-metrics
can compute HU/SUV/Volume for it. Works with any NIfTI segmentation — not just
TotalSegmentator output.

Usage examples:

  # Register a specific nifti file as a multilabel seg for a pair
  python3 CLI_RegisterSeg.py \\
    --pairs "/mnt/f/HNC/Directory Plan/PairIndex.yaml" \\
    --parent "HN001-72/study" \\
    --task "my_custom_seg" \\
    --seg-path "/mnt/f/HNC/CustomSegs/HN001-72_tumor.nii.gz"

  # Register a folder of per-organ niftis
  python3 CLI_RegisterSeg.py \\
    --pairs "/mnt/f/HNC/Directory Plan/PairIndex.yaml" \\
    --parent "HN001-72/study" \\
    --task "my_organs" \\
    --seg-path "/mnt/f/HNC/CustomSegs/HN001-72_organs/"

  # Auto-detect: look for seg files in the same folder tree as the CT
  python3 CLI_RegisterSeg.py \\
    --pairs "/mnt/f/HNC/Directory Plan/PairIndex.yaml" \\
    --headers "/mnt/f/HNC/Directory Plan/HeaderIndex.yaml" \\
    --task "external_seg" \\
    --auto-detect --seg-pattern "*.nii.gz"

  # List available pairs
  python3 CLI_RegisterSeg.py \\
    --pairs "/mnt/f/HNC/Directory Plan/PairIndex.yaml" \\
    --list-pairs

  # Dry run — show what would be written without modifying PairIndex
  python3 CLI_RegisterSeg.py \\
    --pairs "/mnt/f/HNC/Directory Plan/PairIndex.yaml" \\
    --parent "HN001-72/study" \\
    --task "my_seg" \\
    --seg-path "/path/to/seg.nii.gz" \\
    --dry-run
"""

import argparse
import os
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

import yaml


# ---------------------------------------------------------------------------
# Path helpers (centralized in platform_paths.py)
# ---------------------------------------------------------------------------

from platform_paths import (
    coerce_any_path as coerce_path,
    _is_windows_abs_path, _is_wsl_mnt_path,
)


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def load_yaml(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def save_yaml(doc: dict, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.dump(doc, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

def list_nifti_files(folder: Path) -> List[Path]:
    if not folder.is_dir():
        return []
    files = list(folder.glob("*.nii.gz")) + list(folder.glob("*.nii"))
    return sorted([f for f in files if f.is_file()], key=lambda p: p.name.lower())

def is_nifti(p: Path) -> bool:
    n = p.name.lower()
    return n.endswith(".nii.gz") or n.endswith(".nii")


# ---------------------------------------------------------------------------
# PairIndex helpers
# ---------------------------------------------------------------------------

def get_all_pairs(pair_doc: dict) -> List[Dict[str, Any]]:
    """Extract all pairs from selected_pairs and all_selected_pairs."""
    pairs = []
    for parent_rel, sel in (pair_doc.get("selected_pairs") or {}).items():
        if not isinstance(sel, dict):
            continue
        pairs.append({
            "parent_rel": parent_rel,
            "pet_asset_id": sel.get("pet_asset_id", ""),
            "ct_asset_id": sel.get("ct_asset_id", ""),
            "source": "selected_pairs",
            "entry": sel,
        })
    for parent_rel, entries in (pair_doc.get("all_selected_pairs") or {}).items():
        if not isinstance(entries, list):
            continue
        for i, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            pairs.append({
                "parent_rel": parent_rel,
                "pet_asset_id": entry.get("pet_asset_id", ""),
                "ct_asset_id": entry.get("ct_asset_id", ""),
                "source": f"all_selected_pairs[{i}]",
                "entry": entry,
            })
    return pairs


def find_pairs_for_parent(pair_doc: dict, parent_rel: str) -> List[Dict[str, Any]]:
    return [p for p in get_all_pairs(pair_doc) if p["parent_rel"] == parent_rel]


def make_relative(seg_path: Path, seg_root: Path) -> str:
    try:
        return str(seg_path.resolve().relative_to(seg_root.resolve())).replace("\\", "/")
    except ValueError:
        return str(seg_path.resolve()).replace("\\", "/")


# ---------------------------------------------------------------------------
# Seg detection and record building
# ---------------------------------------------------------------------------

def detect_seg_style(seg_path: Path) -> Dict[str, Any]:
    """Detect ml file vs folder of per-organ files."""
    seg_path = seg_path.resolve()

    if seg_path.is_file() and is_nifti(seg_path):
        return {"style": "ml", "ml_file": seg_path, "organ_files": []}

    if seg_path.is_dir():
        niftis = list_nifti_files(seg_path)
        if not niftis:
            return {"style": None, "ml_file": None, "organ_files": []}
        if len(niftis) == 1:
            return {"style": "ml", "ml_file": niftis[0], "organ_files": []}
        return {
            "style": "nonml",
            "ml_file": None,
            "organ_files": [f.name for f in niftis],
        }

    return {"style": None, "ml_file": None, "organ_files": []}


def build_seg_record(seg_path: Path, seg_root: Optional[Path]) -> Dict[str, Any]:
    """Build a segmentation record for PairIndex."""
    det = detect_seg_style(seg_path)
    if det["style"] is None:
        raise ValueError(f"No nifti files found at {seg_path}")

    rec: Dict[str, Any] = {
        "status": "done",
        "source": "external",
        "registered_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    if det["style"] == "ml":
        rec["seg_type"] = "combined"
        rel = make_relative(det["ml_file"], seg_root) if seg_root else str(det["ml_file"])
        rec["combined_path"] = {"seg_path_rel": rel}
    else:
        rec["seg_type"] = "individual"
        rel = make_relative(seg_path.resolve(), seg_root) if seg_root else str(seg_path.resolve())
        rec["organ_dir"] = {
            "dir_path_rel": rel,
            "organ_files": det["organ_files"],
        }

    return rec


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_seg(pair_doc: dict, parent_rel: str, task: str,
                 seg_record: Dict[str, Any], pair_index: int = 0,
                 overwrite: bool = False) -> bool:
    """
    Write seg record into the pair's segmentations dict.
    Tries all_selected_pairs first, falls back to selected_pairs.
    Returns True if written.
    """
    # Try all_selected_pairs
    all_sel = pair_doc.get("all_selected_pairs") or {}
    if parent_rel in all_sel and isinstance(all_sel[parent_rel], list):
        entries = all_sel[parent_rel]
        if pair_index < len(entries) and isinstance(entries[pair_index], dict):
            entry = entries[pair_index]
            if not isinstance(entry.get("segmentations"), dict):
                entry["segmentations"] = {}
            if task in entry["segmentations"] and not overwrite:
                return False
            entry["segmentations"][task] = seg_record
            return True

    # Fall back to selected_pairs
    sel = pair_doc.get("selected_pairs") or {}
    if parent_rel in sel and isinstance(sel[parent_rel], dict):
        entry = sel[parent_rel]
        if not isinstance(entry.get("segmentations"), dict):
            entry["segmentations"] = {}
        if task in entry["segmentations"] and not overwrite:
            return False
        entry["segmentations"][task] = seg_record
        return True

    raise ValueError(f"Parent '{parent_rel}' not found in PairIndex")


# ---------------------------------------------------------------------------
# Auto-detect
# ---------------------------------------------------------------------------

def auto_detect_segs(pair_doc: dict, headers_doc: dict,
                     seg_pattern: str) -> List[Dict[str, Any]]:
    """Find seg files matching pattern near each pair's CT folder."""
    series_map = {}
    for r in headers_doc.get("dicom_series", []) or []:
        aid = (r.get("asset_id") or "").strip()
        root_path = coerce_path(str(r.get("root_path") or ""))
        series_rel = str(r.get("series_rel") or "")
        if aid and root_path and series_rel:
            series_map[aid] = Path(root_path) / series_rel

    results = []
    for p in get_all_pairs(pair_doc):
        ct_dir = series_map.get(p["ct_asset_id"])
        if not ct_dir or not ct_dir.exists():
            continue

        found = []
        for sd in [ct_dir, ct_dir.parent]:
            if not sd.is_dir():
                continue
            for f in sd.glob(seg_pattern):
                if f.is_file() and is_nifti(f):
                    found.append(f)
            for sub in sd.iterdir():
                if sub.is_dir():
                    for f in sub.glob(seg_pattern):
                        if f.is_file() and is_nifti(f):
                            found.append(f)

        if found:
            # Deduplicate
            seen = set()
            unique = []
            for f in found:
                fp = str(f.resolve())
                if fp not in seen:
                    seen.add(fp)
                    unique.append(f)
            results.append({
                "parent_rel": p["parent_rel"],
                "ct_dir": str(ct_dir),
                "found_files": unique,
            })

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Register external segmentations with PET/CT pairs in PairIndex.yaml",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--pairs", required=True, type=Path, help="PairIndex.yaml")
    p.add_argument("--headers", type=Path, default=None, help="HeaderIndex.yaml (for --auto-detect)")
    p.add_argument("--seg-root", type=str, default=None,
                   help="Seg root dir for relative paths. Auto-inferred from PairIndex if not set.")

    p.add_argument("--list-pairs", action="store_true", help="List all pairs and exit")

    p.add_argument("--parent", type=str, default=None, help="Parent rel path (e.g. HN001-72/study)")
    p.add_argument("--task", type=str, default=None, help="Task name for the segmentation")
    p.add_argument("--seg-path", type=str, default=None, help="Path to seg file or folder")
    p.add_argument("--pair-index", type=int, default=0, help="Pair index within parent (default: 0)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing task record")

    p.add_argument("--auto-detect", action="store_true", help="Find segs near CT folders")
    p.add_argument("--seg-pattern", type=str, default="*.nii.gz", help="Glob pattern (default: *.nii.gz)")

    p.add_argument("--dry-run", action="store_true", help="Show changes without saving")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    pairs_path = args.pairs.resolve()
    pair_doc = load_yaml(pairs_path)

    seg_root = None
    if args.seg_root:
        seg_root = Path(coerce_path(args.seg_root)).resolve()
    else:
        derived = pair_doc.get("derived") or {}
        out_root = str((derived.get("last_totalseg_run") or {}).get("out_root") or "").strip()
        if out_root:
            seg_root = Path(coerce_path(out_root)).resolve()

    if seg_root and args.verbose:
        print(f"[seg-root] {seg_root}")

    # ---- List ----
    if args.list_pairs:
        pairs = get_all_pairs(pair_doc)
        if not pairs:
            print("No pairs found.")
            return
        for p in sorted(pairs, key=lambda x: x["parent_rel"]):
            segs = p["entry"].get("segmentations") or {}
            tasks = sorted(segs.keys()) if isinstance(segs, dict) else []
            print(f"  {p['parent_rel']}")
            print(f"    CT:   {p['ct_asset_id']}")
            print(f"    PET:  {p['pet_asset_id']}")
            print(f"    segs: {', '.join(tasks) if tasks else '(none)'}")
        print(f"\nTotal: {len(pairs)} pairs")
        return

    # ---- Auto-detect ----
    if args.auto_detect:
        if not args.headers:
            sys.exit("ERROR: --auto-detect requires --headers")
        if not args.task:
            sys.exit("ERROR: --auto-detect requires --task")

        headers_doc = load_yaml(args.headers.resolve())
        results = auto_detect_segs(pair_doc, headers_doc, args.seg_pattern)

        if not results:
            print("No segmentation files found near any CT series.")
            return

        registered = 0
        for r in results:
            parent = r["parent_rel"]
            found = r["found_files"]
            print(f"\n  {parent}: {len(found)} file(s)")
            for f in found:
                print(f"    {f}")

            seg_p = found[0] if len(found) == 1 else found[0].parent
            try:
                rec = build_seg_record(seg_p, seg_root)
            except ValueError as e:
                print(f"    SKIP: {e}")
                continue

            if args.dry_run:
                print(f"    DRY RUN: would register '{args.task}' ({rec['seg_type']})")
                continue

            try:
                if register_seg(pair_doc, parent, args.task, rec, overwrite=args.overwrite):
                    print(f"    REGISTERED '{args.task}' ({rec['seg_type']})")
                    registered += 1
                else:
                    print(f"    SKIP: '{args.task}' exists (use --overwrite)")
            except ValueError as e:
                print(f"    ERROR: {e}")

        if registered > 0:
            save_yaml(pair_doc, pairs_path)
            print(f"\n[saved] {pairs_path} ({registered} registrations)")
        return

    # ---- Single registration ----
    if not args.parent:
        sys.exit("ERROR: --parent required (or use --list-pairs / --auto-detect)")
    if not args.task:
        sys.exit("ERROR: --task required")
    if not args.seg_path:
        sys.exit("ERROR: --seg-path required")

    seg_path = Path(coerce_path(args.seg_path)).resolve()
    if not seg_path.exists():
        sys.exit(f"ERROR: path does not exist: {seg_path}")

    matching = find_pairs_for_parent(pair_doc, args.parent)
    if not matching:
        parents = sorted(set(x["parent_rel"] for x in get_all_pairs(pair_doc)))
        print(f"ERROR: no pairs for '{args.parent}'. Available:", file=sys.stderr)
        for p in parents:
            print(f"  {p}", file=sys.stderr)
        sys.exit(1)

    if args.pair_index >= len(matching):
        sys.exit(f"ERROR: pair-index {args.pair_index} out of range ({len(matching)} pairs)")

    try:
        rec = build_seg_record(seg_path, seg_root)
    except ValueError as e:
        sys.exit(f"ERROR: {e}")

    det = detect_seg_style(seg_path)
    print(f"Path:    {seg_path}")
    print(f"Style:   {det['style']}")
    if det["style"] == "ml":
        print(f"File:    {det['ml_file'].name}")
    else:
        print(f"Organs:  {len(det['organ_files'])} files")
    print(f"Task:    {args.task}")
    print(f"Parent:  {args.parent}")

    if args.dry_run:
        print(f"\n[dry-run] record: {rec}")
        return

    try:
        written = register_seg(pair_doc, args.parent, args.task, rec,
                               pair_index=args.pair_index, overwrite=args.overwrite)
    except ValueError as e:
        sys.exit(f"ERROR: {e}")

    if written:
        save_yaml(pair_doc, pairs_path)
        print(f"\n[saved] {pairs_path}")
    else:
        print(f"\nSKIP: '{args.task}' already exists (use --overwrite)")


if __name__ == "__main__":
    main()
