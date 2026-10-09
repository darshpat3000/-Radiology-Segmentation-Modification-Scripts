#!/usr/bin/env python3
"""
CLI_TotalSegmentatorFromPairIndex.py — Run TotalSegmentator using modular PairIndex + HeaderIndex,
and WRITE segmentation outputs back INTO PairIndex.yaml.

Inputs (modular):
- PairIndex.yaml   (from CLI_PairFromHeaders.py)
- HeaderIndex.yaml (from CLI_HeaderExtract.py)

Outputs:
- Updates PairIndex.yaml in-place (creates a .bak backup by default)
- Optional: write a job log yaml with --write-run-log (OFF by default)

Where outputs are recorded in PairIndex (new schema):
1) selected_pairs[parent_rel].segmentations[task] = { source_ct_asset_id, seg_type, ... }
2) all_selected_pairs[parent_rel][i].segmentations[task] = { ... } (when pair matches)
3) derived.segmentations_by_pair_id[pair_id][task] = { ... } (always, includes multi-pair mode)

Segmentation record schema (per task):
  "<task_name>":
    source_ct_asset_id: str            # CT this seg was generated from (always set)
    seg_type: "combined" | "individual"
    combined_path:                     # present when seg_type == "combined"
      seg_root_label: str
      seg_path_rel: str
    organ_dir:                         # present when seg_type == "individual"
      seg_root_label: str
      dir_path_rel: str
      organ_files: [str, ...]
    ts_version: str
    created_at: str                    # ISO timestamp
    status: str                        # done/skipped/indexed_only/failed
    task_info: dict                    # from totalseg_tasks if available
    error: str                         # empty unless failed

KEY FIXES (path + structure robustness):
- Input CT dir is resolved from HeaderIndex *row* (root_path + series_rel), not by assuming one global dicom_root layout.
- Output dir is planned as:
    * If PairIndex already has a recorded path for (pair_id, task), we REUSE IT.
    * Otherwise fall back to <out_root>/<ct_series_rel>/<task> (legacy behavior).
- Works whether you run in WSL or Windows Python and even if YAML contains mixed path styles.

IMPORTANT FIX (ML outputs):
- When --ml is used, TotalSegmentator expects -o to be a FILE path, not a directory.
  We therefore force ML output into: <out_dir>/<task>.nii.gz
- Also normalizes legacy outputs where TS wrote: <out_dir>.nii(.gz) next to empty <out_dir>/ folder.

Examples:
  Plan:
    python CLI_TotalSegmentatorFromPairIndex.py plan --pairs PairIndex.yaml --headers HeaderIndex.yaml --tasks "total,lung_vessels"

  Run selected pairs (default reasons auto_selected,manual_override):
    python CLI_TotalSegmentatorFromPairIndex.py run --pairs PairIndex.yaml --headers HeaderIndex.yaml --tasks "total" --skip-existing

  Multilabel output:
    python CLI_TotalSegmentatorFromPairIndex.py run --pairs PairIndex.yaml --headers HeaderIndex.yaml --tasks "total" --extra-cli "--ml"

  Multi-pair mode (all candidate pairs per parent; output nested per pair hash):
    python CLI_TotalSegmentatorFromPairIndex.py run --pairs PairIndex.yaml --headers HeaderIndex.yaml --use-all-candidate-pairs --tasks "total" --skip-existing
"""

import argparse
import copy
import gc
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time

import yaml
import psutil
import torch

# Optional shared module
try:
    import totalseg_tasks as tst  # expects totalseg_tasks.py importable
except Exception:
    tst = None


# ---------------------------
# Basic helpers
# ---------------------------

def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")

def load_yaml(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def write_yaml(p: Path, doc: dict) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True)
    return p

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def short_hash(s: str) -> str:
    import hashlib
    return hashlib.sha1((s or "").encode("utf-8", errors="ignore")).hexdigest()[:10]


# ---------------------------
# TotalSegmentator version detection
# ---------------------------

_TS_VERSION_CACHE: Optional[str] = None

def detect_ts_version() -> str:
    """Try to detect TotalSegmentator version from the installed package."""
    global _TS_VERSION_CACHE
    if _TS_VERSION_CACHE is not None:
        return _TS_VERSION_CACHE

    # Method 1: importlib
    try:
        from importlib.metadata import version as pkg_version
        v = pkg_version("TotalSegmentator")
        if v:
            _TS_VERSION_CACHE = v.strip()
            return _TS_VERSION_CACHE
    except Exception:
        pass

    # Method 2: CLI --version
    try:
        proc = subprocess.run(
            ["TotalSegmentator", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        out = (proc.stdout or "").strip() + " " + (proc.stderr or "").strip()
        m = re.search(r"(\d+\.\d+[\.\d]*)", out)
        if m:
            _TS_VERSION_CACHE = m.group(1)
            return _TS_VERSION_CACHE
    except Exception:
        pass

    _TS_VERSION_CACHE = "unknown"
    return _TS_VERSION_CACHE


# ---------------------------
# WSL/Windows path coercion (centralized in platform_paths.py)
# ---------------------------

from platform_paths import (
    coerce_path, is_wsl,
    _is_windows_abs_path, _is_posix_abs_path,
    win_to_wsl_path, wsl_to_win_path,
)

def sanitize_rel(rel: str) -> str:
    rel = (rel or "").strip().replace("\\", "/")
    if _is_posix_abs_path(rel) or _is_windows_abs_path(rel):
        return rel
    while rel.startswith("./") or rel.startswith(".\\"):
        rel = rel[2:]
    rel = rel.lstrip("/")
    return rel


# ---------------------------
# TotalSegmentator output detection
# ---------------------------

NIFTI_EXTS = (".nii", ".nii.gz")

def _is_nifti(p: Path) -> bool:
    n = p.name.lower()
    return any(n.endswith(x) for x in NIFTI_EXTS)

def extra_cli_has_ml(extra_cli: List[str]) -> bool:
    return any(str(x).strip() == "--ml" for x in (extra_cli or []))

def normalize_ml_sibling_into_outdir(out_dir: Path) -> None:
    try:
        if not out_dir.exists() or not out_dir.is_dir():
            return
        for p in out_dir.iterdir():
            if p.is_file() and _is_nifti(p):
                return
    except Exception:
        return

    sib_gz = Path(str(out_dir) + ".nii.gz")
    sib_ni = Path(str(out_dir) + ".nii")
    src = sib_gz if sib_gz.exists() else (sib_ni if sib_ni.exists() else None)
    if src is None:
        return

    dst = out_dir / src.name
    if dst.exists():
        return
    try:
        shutil.move(str(src), str(dst))
        print(f"  .. normalized legacy ml output: {src} -> {dst}")
    except Exception:
        return

def looks_completed(out_dir: Path) -> bool:
    try:
        if out_dir.exists() and out_dir.is_file():
            return _is_nifti(out_dir)
        if out_dir.exists() and out_dir.is_dir():
            for p in out_dir.iterdir():
                if p.is_file() and _is_nifti(p):
                    return True
            sib_gz = Path(str(out_dir) + ".nii.gz")
            sib_ni = Path(str(out_dir) + ".nii")
            if sib_gz.exists() or sib_ni.exists():
                return True
    except Exception:
        return False
    return False

def detect_output_format(out_dir: Path) -> str:
    """Returns 'combined', 'individual', or 'empty'."""
    if tst is not None:
        try:
            fmt = str(tst.detect_output_format(out_dir))
            # Map legacy format names to new schema
            if fmt in ("multilabel", "mixed"):
                return "combined"
            if fmt == "per_structure":
                return "individual"
            return fmt
        except Exception:
            pass
    if not out_dir.exists() or not out_dir.is_dir():
        return "empty"
    files = [p for p in out_dir.iterdir() if p.is_file() and _is_nifti(p)]
    if not files:
        return "empty"
    if len(files) == 1:
        return "combined"
    if len(files) >= 10:
        sizes = sorted([p.stat().st_size for p in files], reverse=True)
        if len(sizes) >= 2 and sizes[1] > 0 and sizes[0] >= 4 * sizes[1]:
            return "combined"  # one big file + some small extras
        return "individual"
    sizes = sorted([p.stat().st_size for p in files], reverse=True)
    if len(sizes) >= 2 and sizes[1] > 0 and sizes[0] >= 4 * sizes[1]:
        return "combined"
    return "individual"

def list_nifti_files(out_dir: Path) -> List[str]:
    try:
        files = [p.name for p in out_dir.iterdir() if p.is_file() and _is_nifti(p)]
        files.sort()
        return files
    except Exception:
        return []

def pick_multilabel_file(out_dir: Path) -> Optional[str]:
    try:
        files = [p for p in out_dir.iterdir() if p.is_file() and _is_nifti(p)]
        if not files:
            return None
        files_sorted = sorted(files, key=lambda p: p.stat().st_size, reverse=True)
        return files_sorted[0].name if files_sorted else None
    except Exception:
        return None

def task_info(task: str) -> Dict[str, Any]:
    if tst is None:
        return {"name": task, "modality": "unknown", "availability": "unknown", "notes": ""}
    try:
        ti = tst.get_task(task)
        if not ti:
            return {"name": task, "modality": "unknown", "availability": "unknown", "notes": ""}
        return {"name": ti.name, "modality": ti.modality, "availability": ti.availability, "notes": ti.notes}
    except Exception:
        return {"name": task, "modality": "unknown", "availability": "unknown", "notes": ""}

def maybe_label_map(task: str, include: bool) -> Tuple[bool, Optional[Dict[int, str]]]:
    if not include or tst is None or not task:
        return False, None
    try:
        lm = tst.try_get_label_map(task)
        if lm:
            return True, lm
    except Exception:
        pass
    return False, None


# ---------------------------
# Pair/series resolution
# ---------------------------

def choose_dicom_root_from_headers(headers: dict, override_root: Optional[Path]) -> Path:
    if override_root:
        return override_root.resolve()
    roots: Dict[str, int] = {}
    for r in headers.get("dicom_series", []) or []:
        rp = str(r.get("root_path", "") or "").strip()
        if rp:
            roots[rp] = roots.get(rp, 0) + 1
    if roots:
        best = sorted(roots.items(), key=lambda kv: kv[1], reverse=True)[0][0]
        return coerce_path(best).resolve()
    return Path.cwd().resolve()

def default_out_dir_for(dicom_root: Path) -> Path:
    return (dicom_root.parent / "SegmentationsRoot").resolve()

def build_series_map(headers: dict) -> Dict[str, dict]:
    by_a: Dict[str, dict] = {}
    for r in headers.get("dicom_series", []) or []:
        aid = (r.get("asset_id") or "").strip()
        if aid:
            by_a[aid] = r
    return by_a

def iter_selected_pairs(pairs_doc: dict, include_reasons: List[str]) -> List[Dict[str, Any]]:
    out = []
    sp = pairs_doc.get("selected_pairs", {}) or {}
    for parent_rel, v in sp.items():
        if not isinstance(v, dict):
            continue
        reason = v.get("reason", "")
        if include_reasons and reason not in include_reasons:
            continue
        pet_a = (v.get("pet_asset_id") or "").strip()
        ct_a  = (v.get("ct_asset_id") or "").strip()
        if not (pet_a and ct_a):
            continue
        out.append({
            "parent_rel": parent_rel if parent_rel not in ("", None) else ".",
            "pet_asset_id": pet_a,
            "ct_asset_id": ct_a,
            "reason": reason,
            "pairing_method": v.get("pairing_method", ""),
        })
    return out

def iter_all_candidate_pairs(pairs_doc: dict) -> List[Dict[str, Any]]:
    out = []
    for parent in pairs_doc.get("parents", []) or []:
        if not isinstance(parent, dict):
            continue
        pr = parent.get("parent_rel", ".") or "."
        for cand in (parent.get("candidate_pairs", []) or []):
            try:
                pet_a = ((cand.get("pet") or {}).get("asset_id") or "").strip()
                ct_a  = ((cand.get("ct") or {}).get("asset_id") or "").strip()
                if pet_a and ct_a:
                    out.append({
                        "parent_rel": pr,
                        "pet_asset_id": pet_a,
                        "ct_asset_id": ct_a,
                        "reason": "candidate_pair",
                        "pairing_method": cand.get("pairing_method", ""),
                    })
            except Exception:
                continue
    seen = set()
    dedup = []
    for x in out:
        k = (x["parent_rel"], x["pet_asset_id"], x["ct_asset_id"])
        if k in seen:
            continue
        seen.add(k)
        dedup.append(x)
    return dedup

def pair_id_for(parent_rel: str, pet_aid: str, ct_aid: str) -> str:
    pr = parent_rel if parent_rel else "."
    return f"PAIR:{pr}:{pet_aid}:{ct_aid}"


def existing_seg_rec(pair_doc: dict, parent_rel: str, pet_aid: str, ct_aid: str, task: str) -> Optional[dict]:
    """
    Look up existing segmentation record for a (pair, task) combination.
    Search order:
      1) derived.segmentations_by_pair_id[pair_id][task]
      2) selected_pairs[parent_rel].segmentations[task]
      3) all_selected_pairs[parent_rel][i].segmentations[task] (matching pair identity)
    """
    pid = pair_id_for(parent_rel, pet_aid, ct_aid)

    # 1) derived index
    derived = ((pair_doc.get("derived") or {}).get("segmentations_by_pair_id") or {})
    if isinstance(derived, dict):
        tm = derived.get(pid)
        if isinstance(tm, dict):
            rec = tm.get(task)
            if isinstance(rec, dict):
                return rec

    # 2) selected_pairs
    sp = (pair_doc.get("selected_pairs") or {}).get(parent_rel)
    if isinstance(sp, dict):
        segs = sp.get("segmentations")
        if isinstance(segs, dict):
            rec = segs.get(task)
            if isinstance(rec, dict) and sp.get("pet_asset_id") == pet_aid and sp.get("ct_asset_id") == ct_aid:
                return rec

    # 3) all_selected_pairs
    all_sp = (pair_doc.get("all_selected_pairs") or {}).get(parent_rel)
    if isinstance(all_sp, list):
        for entry in all_sp:
            if not isinstance(entry, dict):
                continue
            if entry.get("pet_asset_id") == pet_aid and entry.get("ct_asset_id") == ct_aid:
                segs = entry.get("segmentations")
                if isinstance(segs, dict):
                    rec = segs.get(task)
                    if isinstance(rec, dict):
                        return rec

    return None


def resolve_out_dir_from_rec(out_root: Path, rec: dict) -> Optional[Path]:
    """
    Resolve output directory from an existing segmentation record.
    Handles both new schema (combined_path/organ_dir) and legacy (out_dir_rel/out_dir).
    """
    if not isinstance(rec, dict):
        return None

    seg_type = rec.get("seg_type", "")

    # New schema: combined_path
    if seg_type == "combined":
        cp = rec.get("combined_path")
        if isinstance(cp, dict):
            rel = str(cp.get("seg_path_rel") or "").strip()
            if rel:
                p = (out_root / Path(sanitize_rel(rel))).resolve()
                # combined_path points to a file — return its parent dir
                if _is_nifti(p):
                    return p.parent
                return p

    # New schema: organ_dir
    if seg_type == "individual":
        od = rec.get("organ_dir")
        if isinstance(od, dict):
            rel = str(od.get("dir_path_rel") or "").strip()
            if rel:
                return (out_root / Path(sanitize_rel(rel))).resolve()

    # Legacy fallback: out_dir_rel
    rel = str(rec.get("out_dir_rel") or "").strip()
    if rel:
        p = (out_root / Path(sanitize_rel(rel))).resolve()
        if _is_nifti(p):
            return p.parent
        return p

    # Legacy fallback: out_dir absolute
    od = str(rec.get("out_dir") or "").strip()
    if od:
        p = coerce_path(od).resolve()
        if _is_nifti(p):
            return p.parent
        return p

    return None


def resolve_ct_input_dir(ct_row: dict) -> Optional[Path]:
    if not isinstance(ct_row, dict):
        return None
    root_path = str(ct_row.get("root_path") or "").strip()
    series_rel = str(ct_row.get("series_rel") or "").strip()
    if not series_rel:
        return None
    series_rel_s = sanitize_rel(series_rel)
    if _is_posix_abs_path(series_rel_s) or _is_windows_abs_path(series_rel_s):
        return coerce_path(series_rel_s).resolve()
    if not root_path:
        return None
    root_p = coerce_path(root_path).resolve()
    return (root_p / Path(series_rel_s)).resolve()


# ---------------------------
# Segmentation record builder (new schema)
# ---------------------------

SEG_ROOT_LABEL = "seg_out"  # label for the output root in segmentation records

def build_seg_record(
    task: str,
    ct_asset_id: str,
    out_dir: Path,
    out_root: Path,
    status: str,
    task_info_dict: dict,
    error: str = "",
    include_label_map: bool = False,
) -> dict:
    """
    Build a segmentation record conforming to the schema defined in CLI_PairFromHeaders.py.
    """
    out_dir = out_dir.resolve()
    out_root = out_root.resolve()

    try:
        dir_rel = str(out_dir.relative_to(out_root)).replace("\\", "/")
    except Exception:
        dir_rel = str(out_dir).replace("\\", "/")

    seg_type = detect_output_format(out_dir)
    if seg_type == "empty":
        # Failed or not yet run — still record as combined placeholder
        seg_type = "combined"

    ts_version = detect_ts_version()

    rec: Dict[str, Any] = {
        "source_ct_asset_id": ct_asset_id,
        "seg_type": seg_type,
        "ts_version": ts_version,
        "created_at": now_iso(),
        "status": status,
        "task_info": task_info_dict,
    }

    if error:
        rec["error"] = error

    if seg_type == "combined":
        ml_file = pick_multilabel_file(out_dir)
        if ml_file:
            try:
                file_rel = str((out_dir / ml_file).relative_to(out_root)).replace("\\", "/")
            except Exception:
                file_rel = f"{dir_rel}/{ml_file}"
            rec["combined_path"] = {
                "seg_root_label": SEG_ROOT_LABEL,
                "seg_path_rel": file_rel,
            }
        else:
            # No file yet (planned/failed) — record the dir so we can find it later
            rec["combined_path"] = {
                "seg_root_label": SEG_ROOT_LABEL,
                "seg_path_rel": dir_rel,
            }
    elif seg_type == "individual":
        organ_files = list_nifti_files(out_dir)
        rec["organ_dir"] = {
            "seg_root_label": SEG_ROOT_LABEL,
            "dir_path_rel": dir_rel,
            "organ_files": organ_files,
        }

    if include_label_map:
        ok, lm = maybe_label_map(task, include=True)
        rec["label_map_available"] = ok
        if ok:
            rec["label_map"] = lm

    return rec


# ---------------------------
# Job planning
# ---------------------------

def plan_jobs(
    pair_doc: dict,
    by_asset: Dict[str, dict],
    pair_rows: List[Dict[str, Any]],
    out_root: Path,
    tasks: List[str],
    multi_pair: bool,
    include_tracers: List[str]
) -> List[dict]:
    jobs: List[dict] = []
    out_root = out_root.resolve()

    for pr in pair_rows:
        parent_rel = pr["parent_rel"]
        pet_aid = pr["pet_asset_id"]
        ct_aid  = pr["ct_asset_id"]

        ct_row = by_asset.get(ct_aid)
        series_rel = str(ct_row.get("series_rel") or "").lower()

        if include_tracers:
            if not any(t in series_rel for t in include_tracers):
                continue

        if not ct_row:
            jobs.append({
                "status": "invalid",
                "error": "ct_asset_not_found_in_headers",
                "parent_rel": parent_rel,
                "pet_asset_id": pet_aid,
                "ct_asset_id": ct_aid,
                "task": None,
            })
            continue

        in_dir = resolve_ct_input_dir(ct_row)
        if in_dir is None:
            jobs.append({
                "status": "invalid",
                "error": "ct_series_rel_or_root_missing",
                "parent_rel": parent_rel,
                "pet_asset_id": pet_aid,
                "ct_asset_id": ct_aid,
                "task": None,
            })
            continue

        ct_series_rel_for_out = sanitize_rel(str(ct_row.get("series_rel") or "").strip()) or "CT_UNKNOWN"

        pid = pair_id_for(parent_rel, pet_aid, ct_aid)
        pair_short = short_hash(pid)

        for task in tasks:
            rec_existing = existing_seg_rec(pair_doc, parent_rel, pet_aid, ct_aid, task)
            reused_out_dir = resolve_out_dir_from_rec(out_root, rec_existing) if rec_existing else None

            if reused_out_dir is not None:
                out_dir = reused_out_dir
            else:
                base_out = (out_root / Path(ct_series_rel_for_out) / task)
                out_dir = (base_out / f"pair_{pair_short}") if multi_pair else base_out

            # normalize: out_dir must be directory
            if _is_nifti(out_dir):
                out_dir = out_dir.parent

            try:
                out_dir_rel = str(out_dir.resolve().relative_to(out_root.resolve())).replace("\\", "/")
            except Exception:
                out_dir_rel = str(out_dir).replace("\\", "/")

            jobs.append({
                "status": "planned",
                "pair_id": pid,
                "pair_short": pair_short,
                "parent_rel": parent_rel,
                "pet_asset_id": pet_aid,
                "ct_asset_id": ct_aid,
                "ct_series_rel": ct_series_rel_for_out,
                "input_dir": str(in_dir),
                "out_dir": str(out_dir.resolve()),
                "out_dir_rel": out_dir_rel,
                "task": task,
                "task_info": task_info(task),
                "reused_out_dir": bool(reused_out_dir is not None),
            })

    jobs.sort(key=lambda j: (j.get("parent_rel","."), j.get("ct_series_rel",""), j.get("task") or "", j.get("out_dir_rel","")))
    return jobs

def print_plan(jobs: List[dict]):
    print("\n=== Planned TotalSegmentator Jobs (modular → updates PairIndex) ===")
    for j in jobs:
        if j.get("status") == "invalid":
            print(f"- [INVALID] parent={j.get('parent_rel')} pet={j.get('pet_asset_id')} ct={j.get('ct_asset_id')}")
            print(f"    error: {j.get('error')}")
            continue
        tag = "REUSE" if j.get("reused_out_dir") else "NEW"
        print(f"- [{j['task']}] ({tag}) parent={j['parent_rel']} ct={j['ct_series_rel']}")
        print(f"    in : {j['input_dir']}")
        print(f"    out: {j['out_dir']}")
    print(f"Total jobs: {len(jobs)}\n")

def estimate_batch_size(
        min_batch: int = 1,
        max_batch: int = 50,
        memory_per_job_gb: float = 4.0,
        safety_factor: float = 0.6,
) -> int:
    try:
        available_gb = psutil.virtual_memory().available / (1024 ** 3)
        usable_gb = available_gb * safety_factor
        batch = int(usable_gb // memory_per_job_gb)
        batch = max(batch, min(batch, max_batch))
        return batch
    except Exception:
        print(f"[batch] could not detect RAM, default to batch size = 1.")
        return min_batch


# ---------------------------
# Execution
# ---------------------------

def totalsegmentator_on_path() -> bool:
    return shutil.which("TotalSegmentator") is not None

def run_ts_cli(input_dir: Path, out_dir: Path, task: str, extra_cli: List[str]) -> int:
    out_target: Path = out_dir
    if extra_cli_has_ml(extra_cli):
        ensure_dir(out_dir)
        out_target = out_dir / f"{task}.nii.gz"

    cmd = ["TotalSegmentator", "-i", str(input_dir), "-o", str(out_target), "-ta", task]
    if extra_cli:
        cmd += extra_cli

    print("[cli] $", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, check=False)
        return int(proc.returncode)
    except FileNotFoundError:
        print('ERROR: "TotalSegmentator" not found on PATH.', file=sys.stderr)
        return 127

def run_jobs(
    jobs: List[dict],
    skip_existing: bool,
    extra_cli: List[str],
    include_label_map: bool,
    write_index_only: bool,
    out_root: Path,
    max_jobs: int = 1,
    sleep_between: int = 45,
    start_index: int = 0,
) -> Tuple[int, List[dict], int]:
    if not totalsegmentator_on_path() and not write_index_only:
        print('ERROR: "TotalSegmentator" not found on PATH. Activate env or add to PATH.', file=sys.stderr)
        sys.exit(1)

    failures = 0
    ran = 0
    last_index = start_index

    for idx, j in enumerate(jobs[start_index:], start_index + 1):
        last_index = idx
        if j.get("status") == "invalid":
            failures += 1
            continue

        in_dir = coerce_path(str(j["input_dir"])).resolve()
        out_dir = coerce_path(str(j["out_dir"])).resolve()
        task = j["task"]

        print(f"[{idx}/{len(jobs)}] {task} :: {in_dir} -> {out_dir}")

        if not in_dir.exists():
            j["status"] = "failed"
            j["error"] = "input_not_found"
            j["seg_record"] = build_seg_record(
                task=task, ct_asset_id=j.get("ct_asset_id", ""),
                out_dir=out_dir, out_root=out_root,
                status="failed", task_info_dict=j.get("task_info", {}),
                error="input_not_found", include_label_map=include_label_map,
            )
            failures += 1
            print(f"  !! input not found: {in_dir}", file=sys.stderr)
            continue

        ensure_dir(out_dir)
        normalize_ml_sibling_into_outdir(out_dir)

        if write_index_only:
            j["status"] = "indexed_only"
            j["seg_record"] = build_seg_record(
                task=task, ct_asset_id=j.get("ct_asset_id", ""),
                out_dir=out_dir, out_root=out_root,
                status="indexed_only", task_info_dict=j.get("task_info", {}),
                include_label_map=include_label_map,
            )
            print("  .. indexed_only")
            continue

        if skip_existing and looks_completed(out_dir):
            j["status"] = "skipped"
            j["note"] = "skip_existing_has_nifti_or_ml_sibling"
            normalize_ml_sibling_into_outdir(out_dir)
            j["seg_record"] = build_seg_record(
                task=task, ct_asset_id=j.get("ct_asset_id", ""),
                out_dir=out_dir, out_root=out_root,
                status="skipped", task_info_dict=j.get("task_info", {}),
                include_label_map=include_label_map,
            )
            print("  .. skip (already exists)")
            continue

        rc = run_ts_cli(in_dir, out_dir, task, extra_cli)
        gc.collect()

        # More robust cache emptying using torch
        try:
          if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception:
          pass
        ran += 1

        if sleep_between > 0:
            for remaining in range(sleep_between, -1, -1):
                mins, secs = divmod(remaining, 60)
                print(f"\rDelay: {mins:02d}:{secs:02d}", end="", flush=True)
                time.sleep(1)

        if rc != 0:
            j["status"] = "failed"
            j["error"] = f"ts_failed_rc_{rc}"
            j["seg_record"] = build_seg_record(
                task=task, ct_asset_id=j.get("ct_asset_id", ""),
                out_dir=out_dir, out_root=out_root,
                status="failed", task_info_dict=j.get("task_info", {}),
                error=f"ts_failed_rc_{rc}", include_label_map=include_label_map,
            )
            failures += 1
            print(f"  !! TotalSegmentator failed (rc={rc})", file=sys.stderr)
            continue

        j["status"] = "done"
        normalize_ml_sibling_into_outdir(out_dir)
        j["seg_record"] = build_seg_record(
            task=task, ct_asset_id=j.get("ct_asset_id", ""),
            out_dir=out_dir, out_root=out_root,
            status="done", task_info_dict=j.get("task_info", {}),
            include_label_map=include_label_map,
        )
        print("  .. done")

        if max_jobs and ran >= max_jobs:
            print(f"[batch] reached max_jobs={max_jobs}, re-running.")
            break

    return (0 if failures == 0 else 1), jobs, last_index


# ---------------------------
# PairIndex update
# ---------------------------

def ensure_pairindex_structures(pair_doc: dict):
    pair_doc.setdefault("derived", {})
    pair_doc["derived"].setdefault("segmentations_by_pair_id", {})

def update_pairindex_with_jobs(pair_doc: dict, jobs: List[dict], out_root: Path):
    """
    Writes segmentation records (new schema) into:
    - selected_pairs[parent_rel].segmentations[task]       (when pair matches)
    - all_selected_pairs[parent_rel][i].segmentations[task] (when pair matches)
    - derived.segmentations_by_pair_id[pair_id][task]       (always)
    """
    ensure_pairindex_structures(pair_doc)

    seg_by_pid: dict = pair_doc["derived"]["segmentations_by_pair_id"]
    selected_pairs: dict = pair_doc.get("selected_pairs", {}) or {}
    all_selected_pairs: dict = pair_doc.get("all_selected_pairs", {}) or {}

    for j in jobs:
        if j.get("status") in ("invalid",):
            continue

        pair_id = j.get("pair_id", "")
        parent_rel = j.get("parent_rel", ".")
        pet_aid = j.get("pet_asset_id", "")
        ct_aid  = j.get("ct_asset_id", "")
        task = j.get("task", "")

        rec = j.get("seg_record")
        if not isinstance(rec, dict):
            # Fallback: build minimal record
            out_dir = coerce_path(str(j.get("out_dir", ""))).resolve()
            rec = build_seg_record(
                task=task, ct_asset_id=ct_aid,
                out_dir=out_dir, out_root=out_root,
                status=j.get("status", "unknown"),
                task_info_dict=j.get("task_info", {}),
                error=j.get("error", ""),
            )

        # 1) Always store under derived pair_id map
        if pair_id:
            seg_by_pid.setdefault(pair_id, {})
            seg_by_pid[pair_id][task] = rec

        # 2) Attach to selected_pairs entry if identity matches
        sp = selected_pairs.get(parent_rel)
        if isinstance(sp, dict):
            if sp.get("pet_asset_id", "") == pet_aid and sp.get("ct_asset_id", "") == ct_aid:
                sp.setdefault("segmentations", {})
                sp["segmentations"][task] = rec

        # 3) Attach to matching all_selected_pairs entry
        all_sp_list = all_selected_pairs.get(parent_rel)
        if isinstance(all_sp_list, list):
            for entry in all_sp_list:
                if not isinstance(entry, dict):
                    continue
                if entry.get("pet_asset_id", "") == pet_aid and entry.get("ct_asset_id", "") == ct_aid:
                    entry.setdefault("segmentations", {})
                    entry["segmentations"][task] = rec
                    break  # only one match per parent per pair identity

def backup_file(path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = path.with_suffix(path.suffix + f".bak_{ts}")
    bak.write_bytes(path.read_bytes())
    return bak


# ---------------------------
# CLI
# ---------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run TotalSegmentator from modular PairIndex + HeaderIndex and update PairIndex with outputs.")
    sub = p.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--pairs", required=True, type=Path, help="Path to PairIndex.yaml")
    common.add_argument("--headers", required=True, type=Path, help="Path to HeaderIndex.yaml")
    common.add_argument("--dicom-root", type=str, default=None,
                        help="Override DICOM root (used ONLY to infer default out-dir if --out-dir not provided).")
    common.add_argument("--out-dir", type=str, default=None,
                        help='Output root (default: "<dicom_root parent>/SegmentationsRoot").')
    common.add_argument("--tasks", type=str, default="total", help="Comma list of tasks, e.g. 'total,lung_vessels'")
    common.add_argument("--skip-existing", action="store_true",
                        help="Skip jobs whose output dir already contains a NIfTI (or legacy ml sibling exists).")
    common.add_argument("--extra-cli", type=str, default="",
                        help='Extra args for TotalSegmentator (space-separated), e.g. "-nr 1 -ns 1"')
    common.add_argument("--ml", action="store_true",
                        help="Use multilabel output mode (single combined NIfTI file per task).")

    common.add_argument("--include-reasons", type=str, default="auto_selected,manual_override",
                        help='Comma list of selected_pairs[*].reason values to include')
    common.add_argument("--use-all-candidate-pairs", action="store_true",
                        help="Use PairIndex.parents[].candidate_pairs (multi-pair). Output nested per pair hash.")

    common.add_argument("--include-label-map", action="store_true",
                        help="Include label maps when available (can be large). Default OFF.")
    common.add_argument("--write-index-only", action="store_true",
                        help="Do not run TS; just index existing outputs and update PairIndex.")
    common.add_argument("--no-backup", action="store_true",
                        help="Do not create a PairIndex backup before updating (default: create backup).")
    common.add_argument("--write-run-log", action="store_true",
                        help="Also write SegmentationRunsIndex.yaml under out_root (OFF by default).")
    common.add_argument("--max-jobs", type=int, default=0,
                        help="Maximum number of jobs to run (default: all available CPUs).")
    common.add_argument("--sleep-between", type=int, default=45,
                        help="Seconds to sleep between jobs (default: 45 seconds).")
    common.add_argument("--include-tracers", type=str, default="",
                        help="Only include indicated subfolders/tracers (comma separated). If empty, include all tracers.")

    sub.add_parser("plan", parents=[common], help="Print planned jobs (no execution)")
    sub.add_parser("run",  parents=[common], help="Execute jobs and update PairIndex")
    return p


# ---------------------------
# Main
# ---------------------------

def main():
    args = build_argparser().parse_args()

    pairs_path = coerce_path(str(args.pairs)).resolve()
    headers_path = coerce_path(str(args.headers)).resolve()

    if not pairs_path.exists():
        print(f"ERROR: PairIndex not found: {pairs_path}", file=sys.stderr)
        sys.exit(2)
    if not headers_path.exists():
        print(f"ERROR: HeaderIndex not found: {headers_path}", file=sys.stderr)
        sys.exit(2)

    pair_doc = load_yaml(pairs_path)
    headers_doc = load_yaml(headers_path)

    by_asset = build_series_map(headers_doc)

    dicom_root = choose_dicom_root_from_headers(headers_doc, coerce_path(args.dicom_root) if args.dicom_root else None)
    if not dicom_root.exists():
        print(f"WARNING: inferred dicom_root does not exist: {dicom_root}", file=sys.stderr)

    out_root = coerce_path(args.out_dir).resolve() if args.out_dir else default_out_dir_for(dicom_root).resolve()
    ensure_dir(out_root)

    tasks = [t.strip() for t in (args.tasks or "").split(",") if t.strip()]
    if not tasks:
        print("ERROR: --tasks is empty", file=sys.stderr)
        sys.exit(2)

    include_reasons = [x.strip() for x in (args.include_reasons or "").split(",") if x.strip()]

    if args.use_all_candidate_pairs:
        pair_rows = iter_all_candidate_pairs(pair_doc)
        multi_pair = True
    else:
        pair_rows = iter_selected_pairs(pair_doc, include_reasons=include_reasons)
        multi_pair = False

    if not pair_rows:
        hint = "candidate_pairs" if args.use_all_candidate_pairs else "selected_pairs"
        print(f"Nothing to do: no usable pairs found in PairIndex ({hint}).", file=sys.stderr)
        sys.exit(1)

    include_tracers = [t.strip().lower() for t in (args.include_tracers or "").split(",") if t.strip()]

    jobs = plan_jobs(
        pair_doc=pair_doc,
        by_asset=by_asset,
        pair_rows=pair_rows,
        out_root=out_root,
        tasks=tasks,
        multi_pair=multi_pair,
        include_tracers=include_tracers,
    )

    if args.mode == "plan":
        print_plan(jobs)
        sys.exit(0)

    extra_cli = args.extra_cli.split() if (args.extra_cli or "").strip() else []
    if args.ml and "--ml" not in extra_cli:
        extra_cli.append("--ml")

    if not args.max_jobs:
        max_jobs = estimate_batch_size()
    else:
        max_jobs = args.max_jobs

    start_index = 0
    while start_index < len(jobs):

        rc, jobs_done, start_index = run_jobs(
            jobs=jobs,
            skip_existing=bool(args.skip_existing),
            extra_cli=extra_cli,
            include_label_map=bool(args.include_label_map),
            write_index_only=bool(args.write_index_only),
            out_root=out_root,
            max_jobs=max_jobs,
            start_index=start_index,
        )

        remaining = len(jobs) - start_index
        if remaining > 0:
            print(f"[batch] {remaining}/{len(jobs)} jobs left")

    if not args.no_backup:
        bak = backup_file(pairs_path)
        print(f"[pairindex] backup: {bak}")

    update_pairindex_with_jobs(pair_doc, jobs_done, out_root=out_root)

    pair_doc.setdefault("derived", {})
    pair_doc["derived"].setdefault("last_totalseg_run", {})
    pair_doc["derived"]["last_totalseg_run"] = {
        "generated_at": now_iso(),
        "tool": "CLI_TotalSegmentatorFromPairIndex.py",
        "out_root": str(out_root.resolve()),
        "dicom_root": str(dicom_root.resolve()) if dicom_root else "",
        "tasks": tasks,
        "extra_cli": extra_cli,
        "skip_existing": bool(args.skip_existing),
        "multi_pair": bool(multi_pair),
        "include_label_map": bool(args.include_label_map),
        "write_index_only": bool(args.write_index_only),
        "totalseg_tasks_available": bool(tst is not None),
        "ts_version": detect_ts_version(),
    }

    write_yaml(pairs_path, pair_doc)
    print(f"[pairindex] updated: {pairs_path.resolve()}")

    if args.write_run_log:
        runs_doc = {
            "meta": {
                "tool": "CLI_TotalSegmentatorFromPairIndex.py",
                "generated_at": now_iso(),
                "pairs_file": str(pairs_path.resolve()),
                "headers_file": str(headers_path.resolve()),
                "out_root": str(out_root.resolve()),
                "dicom_root": str(dicom_root.resolve()) if dicom_root else "",
                "ts_version": detect_ts_version(),
            },
            "stats": {
                "total_jobs": int(len(jobs_done)),
                "done": int(sum(1 for j in jobs_done if j.get("status") == "done")),
                "skipped": int(sum(1 for j in jobs_done if j.get("status") == "skipped")),
                "failed": int(sum(1 for j in jobs_done if j.get("status") in ("failed", "invalid"))),
                "indexed_only": int(sum(1 for j in jobs_done if j.get("status") == "indexed_only")),
                "reused_out_dir": int(sum(1 for j in jobs_done if j.get("reused_out_dir") and j.get("status") != "invalid")),
            },
            "jobs": jobs_done,
        }
        log_path = out_root / "SegmentationRunsIndex.yaml"
        write_yaml(log_path, runs_doc)
        print(f"[runlog] wrote: {log_path.resolve()}")

    sys.exit(rc)


if __name__ == "__main__":
    main()
