#!/usr/bin/env python3
"""
totalseg_tasks.py — TotalSegmentator task registry + output format utilities.

Design goals
------------
- Centralize task metadata so future scripts don't embed task lists repeatedly.
- Handle TotalSegmentator output variations:
    - --ml output (single multilabel file) → "combined"
    - per-structure files (one file per class) → "individual"
- Optionally fetch label maps from installed TotalSegmentator (if available),
  so you do not need to hardcode class maps.

Task list source
----------------
This registry is based on the tasks enumerated in the TotalSegmentator README (CT/MR subtasks list).
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# =========================
# Task registry
# =========================

@dataclass(frozen=True)
class TaskInfo:
    name: str
    modality: str              # "CT" | "MR" | "CT/MR" | "unknown"
    availability: str          # "open" | "license" | "cc-by-nc" | "unknown"
    notes: str = ""


# Tasks (names only + basic metadata). You can extend this dict in one place.
TASKS: Dict[str, TaskInfo] = {
    # Open
    "total": TaskInfo("total", "CT", "open", "Default task (multi-structure)"),
    "total_mr": TaskInfo("total_mr", "MR", "open", "Default MR task"),
    "lung_vessels": TaskInfo("lung_vessels", "CT", "open", ""),
    "body": TaskInfo("body", "CT", "open", ""),
    "body_mr": TaskInfo("body_mr", "MR", "open", ""),
    "vertebrae_mr": TaskInfo("vertebrae_mr", "MR", "open", ""),
    "cerebral_bleed": TaskInfo("cerebral_bleed", "CT", "open", "Often marked with * in README; robustness depends on dataset"),
    "hip_implant": TaskInfo("hip_implant", "CT", "open", ""),
    "pleural_pericard_effusion": TaskInfo("pleural_pericard_effusion", "CT", "open", ""),
    "head_glands_cavities": TaskInfo("head_glands_cavities", "CT", "open", ""),
    "head_muscles": TaskInfo("head_muscles", "CT", "open", ""),
    "headneck_bones_vessels": TaskInfo("headneck_bones_vessels", "CT", "open", ""),
    "headneck_muscles": TaskInfo("headneck_muscles", "CT", "open", ""),
    "lung_nodules": TaskInfo("lung_nodules", "CT", "open", ""),
    "kidney_cysts": TaskInfo("kidney_cysts", "CT", "open", ""),
    "breasts": TaskInfo("breasts", "CT", "open", ""),
    "liver_segments": TaskInfo("liver_segments", "CT", "open", ""),
    "liver_segments_mr": TaskInfo("liver_segments_mr", "MR", "open", ""),

    # License-required according to README section (free non-commercial licenses exist)
    "heartchambers_highres": TaskInfo("heartchambers_highres", "CT", "license", ""),
    "appendicular_bones": TaskInfo("appendicular_bones", "CT", "license", ""),
    "appendicular_bones_mr": TaskInfo("appendicular_bones_mr", "MR", "license", ""),
    "tissue_types": TaskInfo("tissue_types", "CT", "license", ""),
    "tissue_types_mr": TaskInfo("tissue_types_mr", "MR", "license", ""),
    "tissue_4_types": TaskInfo("tissue_4_types", "CT", "license", ""),
    "brain_structures": TaskInfo("brain_structures", "CT", "license", ""),
    "vertebrae_body": TaskInfo("vertebrae_body", "CT", "license", ""),
    "face": TaskInfo("face", "CT", "license", "For anonymization"),
    "face_mr": TaskInfo("face_mr", "MR", "license", "For anonymization"),
    "thigh_shoulder_muscles": TaskInfo("thigh_shoulder_muscles", "CT", "license", ""),
    "thigh_shoulder_muscles_mr": TaskInfo("thigh_shoulder_muscles_mr", "MR", "license", ""),
    "coronary_arteries": TaskInfo("coronary_arteries", "CT", "license", ""),

    # Special restrictive license (as per README note)
    "brain_aneurysm": TaskInfo("brain_aneurysm", "MR", "cc-by-nc", "TOF MRI only; CC BY-NC 4.0"),
}


def list_tasks() -> List[str]:
    return sorted(TASKS.keys())

def get_task(name: str) -> Optional[TaskInfo]:
    if not name:
        return None
    return TASKS.get(name.strip())


# =========================
# Output format detection
# =========================

NIFTI_EXTS = (".nii", ".nii.gz")

def _is_nifti(p: Path) -> bool:
    n = p.name.lower()
    return any(n.endswith(x) for x in NIFTI_EXTS)

def detect_output_format(output_dir: Path) -> str:
    """
    Determine TotalSegmentator output style by inspecting files in output_dir.

    Returns:
      - "empty"       : no nifti files found
      - "combined"    : single multilabel nifti (--ml output or single primary file)
      - "individual"  : many nifti files (one per structure, typical non-ml output)

    Notes:
      - We do not rely on filenames (e.g., "total.nii.gz") to decide unless necessary.
      - This is intentionally tolerant across different pipelines.
    """
    output_dir = Path(output_dir)
    if not output_dir.exists() or not output_dir.is_dir():
        return "empty"

    files = [p for p in output_dir.iterdir() if p.is_file() and _is_nifti(p)]
    if not files:
        return "empty"

    n = len(files)

    if n == 1:
        return "combined"

    if n >= 10:
        # If there is a single dominant file (much larger than the rest),
        # this is still "combined" — the big file is the multilabel output
        # and the small files are auxiliaries or leftovers.
        sizes = sorted([p.stat().st_size for p in files], reverse=True)
        med = sizes[len(sizes) // 2]
        if med > 0 and sizes[0] >= 4 * med:
            return "combined"
        return "individual"

    # n in 2..9: check size skew
    sizes = sorted([p.stat().st_size for p in files], reverse=True)
    if sizes[1] > 0 and sizes[0] >= 4 * sizes[1]:
        return "combined"

    # Small number of similarly-sized files — treat as individual
    return "individual"


# =========================
# Label map access (from installed TotalSegmentator)
# =========================

def try_get_label_map(task_name: str) -> Optional[Dict[int, str]]:
    """
    If TotalSegmentator is installed, attempt to return {label_id: label_name} for the task.

    This avoids hardcoding class maps in your scripts.

    Returns None if unavailable.
    """
    if not task_name:
        return None
    tn = task_name.strip()
    try:
        from totalsegmentator.map_to_binary import class_map  # type: ignore
    except Exception:
        return None

    try:
        m = class_map.get(tn)
        if not isinstance(m, dict):
            return None
        out: Dict[int, str] = {}
        for k, v in m.items():
            try:
                out[int(k)] = str(v)
            except Exception:
                continue
        return out or None
    except Exception:
        return None


# =========================
# Convenience: infer task (best-effort; optional)
# =========================

def infer_task_from_folder_name(name: str) -> Optional[str]:
    """
    OPTIONAL helper for pipelines that name output folders by task.
    Use only if you explicitly choose to.
    """
    if not name:
        return None
    n = name.strip().lower()
    for t in TASKS:
        if t.lower() == n:
            return t
    return None