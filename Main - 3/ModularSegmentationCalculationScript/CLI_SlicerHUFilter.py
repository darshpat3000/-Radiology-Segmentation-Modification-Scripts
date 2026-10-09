#!/usr/bin/env python3
"""
CLI_SlicerHUFilter.py — HU-filter segmentations using 3D Slicer.

Reads PairIndex + HeaderIndex, generates a config for slicer_hu_filter_worker.py,
runs Slicer headless, then registers filtered outputs back in PairIndex.

Usage:
  python3 CLI_SlicerHUFilter.py \
    --pairs PairIndex.yaml --headers HeaderIndex.yaml \
    --source-task "EAT" --output-task "EAT_fat" \
    --hu "[-200,0]" --seg-root "/" \
    --slicer "C:\\path\\to\\Slicer.exe"
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

import yaml


# ---------------------------------------------------------------------------
# Path helpers (centralized in platform_paths.py)
# ---------------------------------------------------------------------------

from platform_paths import (
    to_wsl, to_win, to_slicer_path, coerce_any_path,
    slicer_is_windows_target, set_slicer_target,
    _is_windows_abs, _is_wsl_mnt, win_to_wsl, wsl_to_win,
)
import platform_paths as _pp


def load_yaml(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def save_yaml(doc, p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.dump(doc, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


# ---------------------------------------------------------------------------
# HU condition parsing
# ---------------------------------------------------------------------------

def parse_hu_condition(s: str) -> Dict[str, Any]:
    """Parse HU condition string into dict for config."""
    s = s.strip()

    # Range: [-200,0] or [-200, 0]
    m = re.match(r"^\[?\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]?$", s)
    if m:
        return {"type": "range", "min": float(m.group(1)), "max": float(m.group(2))}

    # Comparison: >=130, <=50, >0, <-500
    m = re.match(r"^(>=|<=|>|<)\s*(-?\d+\.?\d*)$", s)
    if m:
        op = m.group(1)
        val = float(m.group(2))
        op_map = {">=": "gte", "<=": "lte", ">": "gt", "<": "lt"}
        return {"type": op_map[op], "value": val}

    raise ValueError(f"Cannot parse HU condition: '{s}'. Examples: [-200,0], >=130, <-500")


# ---------------------------------------------------------------------------
# PairIndex reading (minimal)
# ---------------------------------------------------------------------------

def build_series_by_asset(headers_doc):
    out = {}
    for r in headers_doc.get("dicom_series", []) or []:
        aid = (r.get("asset_id") or "").strip()
        if aid:
            out[aid] = r
    return out

def series_abs_dir(series_row, dicom_root=None):
    """Resolve a CT series directory.

    If dicom_root is provided, the stored (possibly stale/absolute) root_path is
    IGNORED and the directory is resolved as  dicom_root / series_rel  — the same
    relative-root pattern used for segmentations via --seg-root. This is what
    makes indexes portable across machines/mount points.

    If dicom_root is None, falls back to the stored root_path (legacy behavior).
    """
    if not series_row:
        return None
    rel = str(series_row.get("series_rel") or "").strip()
    if not rel:
        return None
    if dicom_root:
        base = coerce_any_path(str(dicom_root).strip())
        if base:
            return str((Path(base) / Path(rel)).resolve())
    root = str(series_row.get("root_path") or "").strip()
    root = coerce_any_path(root)
    if not root:
        return None
    return str((Path(root) / Path(rel)).resolve())

def sanitize_rel(rel):
    rel = (rel or "").strip().replace("\\", "/")
    while rel.startswith("./"):
        rel = rel[2:]
    return rel.lstrip("/")

def iter_pairs(pair_doc, parents_filter=None):
    seen = set()
    for src_key in ["all_selected_pairs", "selected_pairs"]:
        src = pair_doc.get(src_key) or {}
        for parent_rel, val in src.items():
            if parents_filter and parent_rel not in parents_filter:
                continue
            entries = val if isinstance(val, list) else [val] if isinstance(val, dict) else []
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                pet = (entry.get("pet_asset_id") or "").strip()
                ct = (entry.get("ct_asset_id") or "").strip()
                segs = entry.get("segmentations") or {}
                if ct and segs:
                    dedup_key = (parent_rel, pet, ct)
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)
                    yield parent_rel, pet, ct, segs, src_key, entry


def resolve_seg_path(task_rec, seg_root_wsl):
    seg_type = str(task_rec.get("seg_type") or "").strip().lower()
    if seg_type == "combined":
        cp = task_rec.get("combined_path") or {}
        rel = str(cp.get("seg_path_rel") or "").strip()
        if rel:
            wsl_rel = to_wsl(rel)
            if wsl_rel.startswith("/"):
                return wsl_rel
            if seg_root_wsl:
                return str((Path(seg_root_wsl) / rel).resolve())
    return None


# ---------------------------------------------------------------------------
# Register filtered output in PairIndex
# ---------------------------------------------------------------------------

def register_filtered(pair_doc, results, output_task):
    """Register filtered outputs back into PairIndex segmentations."""
    for res in results:
        if res["status"] != "ok":
            continue
        parent_rel = res["parent_rel"]
        out_path = res["out_path"]

        for src_key in ["selected_pairs", "all_selected_pairs"]:
            src = pair_doc.get(src_key) or {}
            val = src.get(parent_rel)
            if val is None:
                continue

            entries = val if isinstance(val, list) else [val] if isinstance(val, dict) else []
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                if "segmentations" not in entry or not isinstance(entry.get("segmentations"), dict):
                    entry["segmentations"] = {}
                entry["segmentations"][output_task] = {
                    "status": "done",
                    "source": "slicer_hu_filter",
                    "registered_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                    "seg_type": "combined",
                    "combined_path": {"seg_path_rel": out_path.replace("\\", "/")},
                    "voxels_before": res.get("voxels_before", 0),
                    "voxels_after": res.get("voxels_after", 0),
                }
                break  # registered for first matching entry
            break  # found the right src_key


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="HU-filter segmentations using 3D Slicer.")
    p.add_argument("--pairs", required=True, type=str, help="PairIndex.yaml")
    p.add_argument("--headers", required=True, type=str, help="HeaderIndex.yaml")
    p.add_argument("--source-task", required=True, help="Source segmentation task to filter")
    p.add_argument("--output-task", required=True, help="Task name to register filtered output as")
    p.add_argument("--hu", action="append", required=True,
                   help="HU condition (repeatable): [-200,0], >=130, etc.")
    p.add_argument("--seg-root", type=str, default=None)
    p.add_argument("--dicom-root", type=str, default=None,
                   help="Root for CT DICOM series. When set, the stored root_path in "
                        "HeaderIndex is IGNORED and dirs resolve as dicom-root/series_rel "
                        "(makes indexes portable across machines/mount points).")
    p.add_argument("--parents", default="", help="Comma list of parent_rel to process")
    p.add_argument("--out-subdir", default="HUFiltered", help="(deprecated; output now always sibling in CT seg dir)")
    p.add_argument("--slicer", type=str,
                   default=os.environ.get("SLICER_PATH",
                            r"C:\Users\kirim\AppData\Local\slicer.org\Slicer 5.8.1\Slicer.exe"),
                   help="Path to the Slicer executable. Windows: ...\\Slicer.exe ; "
                        "Linux: /opt/Slicer-5.8.1-linux-amd64/Slicer . "
                        "Can also be set via the SLICER_PATH environment variable.")
    p.add_argument("--worker", type=str, default=None,
                   help="Path to slicer_hu_filter_worker.py")
    p.add_argument("--config-only", action="store_true",
                   help="Write config.json only, don't call Slicer")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    slicer_is_win = set_slicer_target(args.slicer)
    print(f"[slicer target] {'Windows' if slicer_is_win else 'Linux'}: {args.slicer}")

    # Parse HU conditions
    hu_conditions = [parse_hu_condition(h) for h in args.hu]
    print(f"[HU conditions] {hu_conditions}")

    # Load indexes
    pairs_wsl = to_wsl(args.pairs)
    headers_wsl = to_wsl(args.headers)
    pair_doc = load_yaml(pairs_wsl)
    headers_doc = load_yaml(headers_wsl)
    by_asset = build_series_by_asset(headers_doc)

    # Seg root
    seg_root_wsl = None
    if args.seg_root:
        seg_root_wsl = to_wsl(args.seg_root)
    else:
        derived = pair_doc.get("derived") or {}
        out_root = str((derived.get("last_totalseg_run") or {}).get("out_root") or "")
        if out_root:
            seg_root_wsl = to_wsl(out_root)

    parents_filter = set(t.strip() for t in args.parents.split(",") if t.strip()) if args.parents.strip() else None

    # Build jobs
    jobs = []
    for parent_rel, pet_aid, ct_aid, segs, src_key, entry in iter_pairs(pair_doc, parents_filter):
        task_rec = segs.get(args.source_task)
        if not task_rec:
            continue

        # Skip failed segmentations
        seg_status = str(task_rec.get("status") or "").strip().lower()
        if seg_status == "failed":
            if args.verbose:
                print(f"  SKIP {parent_rel}: {args.source_task} status=failed")
            continue

        # Check if already registered
        if not args.overwrite and args.output_task in segs:
            if args.verbose:
                print(f"  SKIP {parent_rel}: {args.output_task} already registered")
            continue

        ct_row = by_asset.get(ct_aid)
        ct_dir_wsl = series_abs_dir(ct_row, dicom_root=args.dicom_root) if ct_row else None
        if not ct_dir_wsl:
            print(f"  SKIP {parent_rel}: CT dir not resolved")
            continue

        seg_path_wsl = resolve_seg_path(task_rec, seg_root_wsl)
        if not seg_path_wsl or not os.path.isfile(seg_path_wsl):
            print(f"  SKIP {parent_rel}: seg not found ({seg_path_wsl})")
            continue

        # Output path: a sibling folder in the CT seg dir, matching how
        # EAT_region / total / trunk_cavities are laid out:
        #   .../<CT seg dir>/<output-task>/<output-task>.nii.gz
        seg_dir = os.path.dirname(seg_path_wsl)
        out_filename = f"{args.output_task}.nii.gz"
        src_task_folder = os.path.basename(seg_dir)
        if src_task_folder == args.source_task:
            ct_seg_dir = os.path.dirname(seg_dir)
        else:
            ct_seg_dir = seg_dir
        out_dir = os.path.join(ct_seg_dir, args.output_task)
        out_path_wsl = os.path.join(out_dir, out_filename)

        job = {
            "parent_rel": parent_rel,
            "source_task": args.source_task,
            "output_task": args.output_task,
            "ct_dicom_dir": to_slicer_path(ct_dir_wsl),
            "seg_path": to_slicer_path(seg_path_wsl),
            "out_path": to_slicer_path(out_path_wsl),
            "hu_conditions": hu_conditions,
        }
        jobs.append(job)

        if args.verbose:
            print(f"  JOB: {parent_rel}")
            print(f"    CT:  {job['ct_dicom_dir']}")
            print(f"    Seg: {job['seg_path']}")
            print(f"    Out: {job['out_path']}")

    if not jobs:
        print("ERROR: No jobs to run.")
        sys.exit(1)

    print(f"\n{len(jobs)} job(s)")

    if args.dry_run:
        print("[dry-run] No changes.")
        return

    # Write config
    config_dir = os.path.dirname(pairs_wsl)
    config_path_wsl = os.path.join(config_dir, "slicer_hu_filter_config.json")
    config_path_win = to_slicer_path(config_path_wsl)

    with open(config_path_wsl, "w") as f:
        json.dump({"jobs": jobs}, f, indent=2)
    print(f"Config: {config_path_wsl}")

    # Resolve worker
    worker_path = args.worker
    if not worker_path:
        worker_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "slicer_hu_filter_worker.py")

    if args.config_only:
        if _pp.SLICER_IS_WINDOWS:
            print(f"\n[config-only] Run from PowerShell:")
            print(f'  & "{args.slicer}" --no-splash --no-main-window --python-script "{to_win(worker_path)}" "{config_path_win}"')
        else:
            print(f"\n[config-only] Run manually:")
            print(f'  "{args.slicer}" --no-splash --no-main-window --python-script "{worker_path}" "{config_path_wsl}"')
        return

    if not os.path.isfile(worker_path):
        print(f"ERROR: worker not found: {worker_path}")
        sys.exit(1)

    worker_for_slicer = to_slicer_path(worker_path)
    slicer_exe = to_wsl(args.slicer) if _pp.SLICER_IS_WINDOWS else args.slicer
    if not os.path.isfile(slicer_exe):
        print(f"ERROR: Slicer executable not found: {slicer_exe}")
        sys.exit(1)

    cmd = [
        slicer_exe,
        "--no-splash",
        "--no-main-window",
        "--python-script",
        worker_for_slicer,
        config_path_win,
    ]

    pretty_cmd = " ".join(f'"{c}"' if (" " in str(c) or "\t" in str(c)) else str(c) for c in cmd)
    print(f"\nCalling Slicer:\n  {pretty_cmd}\n")
    ret = subprocess.call(cmd)

    if ret != 0:
        print(f"Slicer exited with code {ret}")
        sys.exit(ret)

    # Read results and register in PairIndex
    results_path = config_path_wsl.replace(".json", "_results.json")
    if os.path.isfile(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)
        register_filtered(pair_doc, results, args.output_task)
        save_yaml(pair_doc, pairs_wsl)
        ok = sum(1 for r in results if r["status"] == "ok")
        print(f"\n[done] Registered {ok} filtered seg(s) as '{args.output_task}' in PairIndex")
    else:
        print(f"\nWARN: results file not found: {results_path}")
        print("  Slicer may have failed. Check output manually.")


if __name__ == "__main__":
    main()
