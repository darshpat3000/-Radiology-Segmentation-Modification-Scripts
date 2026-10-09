#!/usr/bin/env python3
"""
CLI_SlicerMetrics.py — Generate Slicer config from PairIndex and run Slicer headless.

Reads PairIndex.yaml + HeaderIndex.yaml, builds a config.json for
slicer_metrics_worker.py, then calls Slicer.exe to compute metrics.

Usage (from WSL):

  python3 CLI_SlicerMetrics.py \
    --pairs "/mnt/c/Users/kirim/Documents/Flu90/Directory Plan/PairIndex.yaml" \
    --headers "/mnt/c/Users/kirim/Documents/Flu90/Directory Plan/HeaderIndex.yaml" \
    --tasks "EAT" \
    --seg-root "/" \
    --out-csv "/mnt/c/Users/kirim/Documents/Flu90/Metrics/slicer_metrics.csv" \
    --slicer "C:\\Users\\kirim\\AppData\\Local\\slicer.org\\Slicer 5.8.1\\Slicer.exe"

  # Config only (don't call Slicer, just write config.json):
  python3 CLI_SlicerMetrics.py ... --config-only

Then from PowerShell if needed:
  & "C:\\Users\\kirim\\AppData\\Local\\slicer.org\\Slicer 5.8.1\\Slicer.exe" `
    --no-splash --no-main-window --python-script slicer_metrics_worker.py config.json
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml

try:
    import totalseg_tasks as tst
except Exception:
    tst = None


# ---------------------------------------------------------------------------
# Path helpers (centralized in platform_paths.py)
# ---------------------------------------------------------------------------

from platform_paths import (
    to_wsl, to_win, to_slicer_path,
    slicer_is_windows_target, set_slicer_target,
    _is_windows_abs, _is_wsl_mnt, win_to_wsl, wsl_to_win,
)
import platform_paths as _pp


def load_yaml(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# PairIndex / HeaderIndex reading (minimal, reused from pipeline)
# ---------------------------------------------------------------------------

def build_series_by_asset(headers_doc):
    out = {}
    for r in headers_doc.get("dicom_series", []) or []:
        aid = (r.get("asset_id") or "").strip()
        if aid:
            out[aid] = r
    return out

def series_abs_dir(series_row, dicom_root=None):
    """Resolve a series directory. If dicom_root is given, the stored (possibly
    stale/absolute) root_path is IGNORED and the dir is resolved as
    dicom_root / series_rel — makes indexes portable across machines."""
    if not series_row:
        return None
    rel = str(series_row.get("series_rel") or "").strip()
    if not rel:
        return None
    if dicom_root:
        base = str(dicom_root).strip()
        if _is_windows_abs(base):
            base = win_to_wsl(base)
        if base:
            return str((Path(base) / Path(rel)).resolve())
    root = str(series_row.get("root_path") or "").strip()
    if _is_windows_abs(root):
        root = win_to_wsl(root)
    if not root:
        return None
    return str((Path(root) / Path(rel)).resolve())

def iter_pairs(pair_doc, tasks_filter):
    """Yield (parent_rel, pet_aid, ct_aid, seg_tasks) for pairs with matching tasks. Deduplicates."""
    seen = set()
    for src_key in ["all_selected_pairs", "selected_pairs"]:
        src = pair_doc.get(src_key) or {}
        for parent_rel, val in src.items():
            entries = val if isinstance(val, list) else [val] if isinstance(val, dict) else []
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                pet = (entry.get("pet_asset_id") or "").strip()
                ct = (entry.get("ct_asset_id") or "").strip()
                segs = entry.get("segmentations") or {}
                matching = {t: segs[t] for t in tasks_filter if t in segs} if tasks_filter else segs
                if pet and ct and matching:
                    dedup_key = (parent_rel, pet, ct)
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)
                    yield parent_rel, pet, ct, matching


def resolve_seg_path(task_rec, seg_root_wsl):
    """Resolve seg file path from task record."""
    seg_type = str(task_rec.get("seg_type") or "").strip().lower()

    if seg_type == "combined":
        cp = task_rec.get("combined_path") or {}
        rel = str(cp.get("seg_path_rel") or "").strip()
        if rel:
            wsl_rel = to_wsl(rel)
            if wsl_rel.startswith("/"):
                return wsl_rel, "combined"
            if seg_root_wsl:
                return str((Path(seg_root_wsl) / rel).resolve()), "combined"
    if seg_type == "individual":
        od = task_rec.get("organ_dir") or {}
        rel = str(od.get("dir_path_rel") or "").strip()
        if rel:
            wsl_rel = to_wsl(rel)
            if wsl_rel.startswith("/"):
                return wsl_rel, "individual"
            if seg_root_wsl:
                return str((Path(seg_root_wsl) / rel).resolve()), "individual"

    return None, None


# ---------------------------------------------------------------------------
# SUV factor (reuse pipeline logic)
# ---------------------------------------------------------------------------

def _parse_tm_seconds(t_raw):
    if not t_raw:
        return None
    s = str(t_raw).strip().split(".")[0]
    s = re.sub(r"[^0-9]", "", s)
    if len(s) < 2:
        return None
    try:
        hh, mm, ss = int(s[0:2]), int(s[2:4]) if len(s) >= 4 else 0, int(s[4:6]) if len(s) >= 6 else 0
        if not (0 <= hh <= 23 and 0 <= mm <= 59 and 0 <= ss <= 59):
            return None
        return hh * 3600 + mm * 60 + ss
    except Exception:
        return None

def _parse_dt_seconds(dt_raw):
    if not dt_raw:
        return None
    s = str(dt_raw).strip().split(".")[0]
    s = re.sub(r"[^0-9]", "", s)
    if len(s) < 10:
        return None
    try:
        hh, mm, ss = int(s[8:10]), int(s[10:12]) if len(s) >= 12 else 0, int(s[12:14]) if len(s) >= 14 else 0
        if not (0 <= hh <= 23 and 0 <= mm <= 59 and 0 <= ss <= 59):
            return None
        return hh * 3600 + mm * 60 + ss
    except Exception:
        return None

def compute_suv_factor(pet_dir_wsl):
    """Compute SUV factor from first PET DICOM file."""
    import pydicom
    import SimpleITK as sitk
    import numpy as np

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(pet_dir_wsl)
    if not series_ids:
        return None
    files = reader.GetGDCMSeriesFileNames(pet_dir_wsl, series_ids[0])
    if not files:
        return None

    dcm = pydicom.dcmread(files[0], stop_before_pixels=True, force=True)
    wt = getattr(dcm, "PatientWeight", None)
    seq = getattr(dcm, "RadiopharmaceuticalInformationSequence", None)
    if not wt or not seq:
        return None

    item = seq[0]
    dose = getattr(item, "RadionuclideTotalDose", None)
    hl = getattr(item, "RadionuclideHalfLife", None)

    # Injection time
    inj_s = None
    start_dt = getattr(item, "RadiopharmaceuticalStartDateTime", None)
    if start_dt:
        inj_s = _parse_dt_seconds(str(start_dt))
    if inj_s is None:
        start_tm = getattr(item, "RadiopharmaceuticalStartTime", None)
        if start_tm:
            inj_s = _parse_tm_seconds(str(start_tm))

    acq_s = _parse_tm_seconds(str(getattr(dcm, "AcquisitionTime", "") or ""))

    if None in (dose, hl, inj_s, acq_s):
        return None

    try:
        wt_kg = float(wt)
        dose_bq = float(dose)
        hl_s = float(hl)
        if wt_kg <= 0 or dose_bq <= 0 or hl_s <= 0:
            return None
        dt = abs(acq_s - inj_s)
        dt = min(dt, 86400 - dt)
        decay = float(np.exp(-np.log(2) * (dt / hl_s)))
        return float((wt_kg * 1000.0) / (dose_bq * decay))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Generate Slicer config from PairIndex and optionally run Slicer.")
    p.add_argument("--pairs", required=True, type=str, help="PairIndex.yaml")
    p.add_argument("--headers", required=True, type=str, help="HeaderIndex.yaml")
    p.add_argument("--tasks", required=True, type=str, help="Comma-separated task names")
    p.add_argument("--seg-root", type=str, default=None, help="Seg root for relative paths")
    p.add_argument("--dicom-root", type=str, default=None,
                   help="Root for CT/PET DICOM series. When set, the stored root_path in "
                        "HeaderIndex is IGNORED and dirs resolve as dicom-root/series_rel "
                        "(portable across machines/mount points).")
    p.add_argument("--out-csv", type=str, required=True, help="Output CSV path")
    p.add_argument("--parents", type=str, default="", help="Filter by parent_rel (comma list)")
    p.add_argument("--slicer", type=str,
                   default=os.environ.get("SLICER_PATH",
                            r"C:\Users\kirim\AppData\Local\slicer.org\Slicer 5.8.1\Slicer.exe"),
                   help="Path to the Slicer executable. Windows: ...\\Slicer.exe ; "
                        "Linux: /opt/Slicer-5.8.1-linux-amd64/Slicer . "
                        "Can also be set via the SLICER_PATH environment variable.")
    p.add_argument("--worker", type=str, default=None,
                   help="Path to slicer_metrics_worker.py (default: same dir as this script)")
    p.add_argument("--config-only", action="store_true",
                   help="Write config.json only, don't call Slicer")
    p.add_argument("--include-raw-pet", action="store_true",
                   help="Include raw PET value columns alongside SUV columns")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    # Decide path format for everything we hand to Slicer.
    slicer_is_win = set_slicer_target(args.slicer)
    print(f"[slicer target] {'Windows' if slicer_is_win else 'Linux'}: {args.slicer}")

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

    # Parse filters
    tasks_filter = [t.strip() for t in args.tasks.split(",") if t.strip()]
    parents_filter = set(t.strip() for t in args.parents.split(",") if t.strip()) if args.parents.strip() else None

    # Look up label maps for each task
    label_maps = {}
    for task_name in tasks_filter:
        lm = None
        if tst:
            lm = tst.try_get_label_map(task_name)
        if lm:
            # Convert int keys to str for JSON
            label_maps[task_name] = {str(k): v for k, v in lm.items()}
            print(f"  [label_map] {task_name}: {len(lm)} labels")
        else:
            print(f"  [label_map] {task_name}: not found (segments will be named Segment_N)")

    # Build jobs
    jobs = []
    for parent_rel, pet_aid, ct_aid, seg_tasks in iter_pairs(pair_doc, tasks_filter):
        if parents_filter and parent_rel not in parents_filter:
            continue

        ct_row = by_asset.get(ct_aid)
        pet_row = by_asset.get(pet_aid)
        ct_dir_wsl = series_abs_dir(ct_row, dicom_root=args.dicom_root) if ct_row else None
        pet_dir_wsl = series_abs_dir(pet_row, dicom_root=args.dicom_root) if pet_row else None

        if not ct_dir_wsl:
            print(f"  SKIP {parent_rel}: CT dir not resolved")
            continue

        # SUV factor
        suv_factor = None
        if pet_dir_wsl and os.path.isdir(pet_dir_wsl):
            try:
                suv_factor = compute_suv_factor(pet_dir_wsl)
            except Exception as e:
                print(f"  WARN {parent_rel}: SUV factor failed: {e}")

        for task, task_rec in seg_tasks.items():
            # Skip failed segmentations
            seg_status = str(task_rec.get("status") or "").strip().lower()
            if seg_status == "failed":
                print(f"  SKIP {parent_rel}/{task}: seg status=failed")
                continue

            seg_path_wsl, seg_type = resolve_seg_path(task_rec, seg_root_wsl)

            if seg_type == "combined":
                if not seg_path_wsl or not os.path.isfile(seg_path_wsl):
                    print(f"  SKIP {parent_rel}/{task}: seg not found ({seg_path_wsl})")
                    continue
            elif seg_type == "individual":
                if not seg_path_wsl or not os.path.isdir(seg_path_wsl):
                    print(f"  SKIP {parent_rel}/{task}: seg not found ({seg_path_wsl})")
                    continue
            else:
                print(f"  SKIP {parent_rel}/{task}: Unknown seg type ({seg_type})")
                continue

            job = {
                "parent_rel": parent_rel,
                "task": task,
                "ct_dicom_dir": to_slicer_path(ct_dir_wsl),
                "pet_dicom_dir": to_slicer_path(pet_dir_wsl) if pet_dir_wsl else "",
                "seg_path": to_slicer_path(seg_path_wsl),
                "suv_factor": suv_factor,
            }
            jobs.append(job)

            if args.verbose:
                print(f"  JOB: {parent_rel}/{task}")
                print(f"    CT:  {job['ct_dicom_dir']}")
                print(f"    PET: {job['pet_dicom_dir']}")
                print(f"    Seg: {job['seg_path']}")
                print(f"    SUV: {suv_factor}")

    if not jobs:
        print("ERROR: No jobs to run.")
        sys.exit(1)

    print(f"\n{len(jobs)} job(s) to process")

    # Write config
    out_csv_win = to_slicer_path(args.out_csv)
    config = {
        "jobs": jobs,
        "out_csv": out_csv_win,
        "include_raw_pet": bool(args.include_raw_pet),
        "label_maps": label_maps,
    }

    config_dir = os.path.dirname(to_wsl(args.out_csv)) or "."
    os.makedirs(config_dir, exist_ok=True)
    config_path_wsl = os.path.join(config_dir, "slicer_config.json")
    config_path_win = to_slicer_path(config_path_wsl)

    with open(config_path_wsl, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config: {config_path_wsl}")

    if args.config_only:
        worker_guess = args.worker or os.path.join(os.path.dirname(os.path.abspath(__file__)), "slicer_metrics_worker.py")
        if _pp.SLICER_IS_WINDOWS:
            print(f"\n[config-only] Run manually from PowerShell:")
            print(f'  & "{args.slicer}" --no-splash --no-main-window --python-script "{to_win(worker_guess)}" "{config_path_win}"')
        else:
            print(f"\n[config-only] Run manually:")
            print(f'  "{args.slicer}" --no-splash --no-main-window --python-script "{worker_guess}" "{config_path_wsl}"')
        return

    # Resolve worker script
    worker_path = args.worker
    if not worker_path:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        worker_path = os.path.join(script_dir, "slicer_metrics_worker.py")
    if not os.path.isfile(worker_path):
        print(f"ERROR: worker script not found: {worker_path}")
        sys.exit(1)

    worker_for_slicer = to_slicer_path(worker_path)

    # Locate the Slicer executable. For a Windows .exe we may have been given a
    # Windows path; convert to its WSL mount so subprocess can launch it.
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

    # Check output
    out_csv_wsl = to_wsl(args.out_csv)
    if os.path.isfile(out_csv_wsl):
        print(f"\n[done] {out_csv_wsl}")
    else:
        print(f"\nWARN: expected output not found: {out_csv_wsl}")


if __name__ == "__main__":
    main()
