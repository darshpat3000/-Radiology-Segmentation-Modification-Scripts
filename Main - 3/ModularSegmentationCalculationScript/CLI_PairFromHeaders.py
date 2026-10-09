#!/usr/bin/env python3
"""
CLI_PairFromHeaders.py — pair PET↔CT from HeaderIndex (pure logic, no DICOM reading)

Writes:
- PairIndex.yaml
- SelectionTemplate.yaml (optional)

DEFAULTS
- Pairing uses StudyInstanceUID first, then PatientID+time-window (optionally prefers FoR match).
- Folder-name text matching is NOT used.
- Manual overrides supported via Overrides.yaml.

Overrides.yaml format (minimal)
------------------------------
selections:
  "<parent_rel>":
    pet_asset_id: "DICOM_SERIES:<StudyUID>:<SeriesUID>"
    ct_asset_id:  "DICOM_SERIES:<StudyUID>:<SeriesUID>"

You can also use series_uid keys:
  "<parent_rel>":
    pet_series_uid: "<SeriesInstanceUID>"
    ct_series_uid: "<SeriesInstanceUID>"

Segmentation schema (appended by downstream modules)
----------------------------------------------------
Each pair (in selected_pairs and all_selected_pairs) supports a 'segmentations'
dict keyed by task name. Each entry records:
  segmentations:
    "<task_name>":           # e.g. "total", "cardiac", "vertebrae"
      source_ct_asset_id: str         # CT this seg was generated from (always set)
      seg_type: "combined" | "individual"
      combined_path:                  # if seg_type == "combined"
        seg_root_label: str
        seg_path_rel: str
      organ_dir:                      # if seg_type == "individual"
        seg_root_label: str
        dir_path_rel: str
        organ_files: [str, ...]       # filenames inside dir
      ts_version: str                 # e.g. "2.0.0"
      created_at: str                 # ISO timestamp

Typical usage
-------------
python CLI_PairFromHeaders.py pair --headers "HeaderIndex.yaml" --time-window 15 --out-dir .
python CLI_PairFromHeaders.py pair --headers HeaderIndex.yaml --overrides Overrides.yaml --write-selection-template
python CLI_PairFromHeaders.py pair --headers HeaderIndex.yaml --all-pairs
"""

import argparse
import logging
import os
import sys
import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml
import json

logger = logging.getLogger(__name__)

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )

def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")

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

def load_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def parse_dt(dt_str: str) -> Optional[datetime]:
    if not dt_str:
        return None
    try:
        return datetime.fromisoformat(dt_str)
    except Exception:
        return None

def dt_sort_key(dt: Optional[datetime]) -> float:
    if dt is None:
        return float("inf")
    try:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.timestamp()
    except Exception:
        return float("inf")

def _safe_minutes_delta(a: Optional[datetime], b: Optional[datetime]) -> float:
    """Like minutes_delta but returns float('inf') instead of None for safe comparisons."""
    d = minutes_delta(a, b)
    return d if d is not None else float("inf")

def minutes_delta(a: Optional[datetime], b: Optional[datetime]) -> Optional[float]:
    if not a or not b:
        return None
    try:
        if a.tzinfo and b.tzinfo:
            a = a.astimezone(timezone.utc)
            b = b.astimezone(timezone.utc)
        else:
            a = a.replace(tzinfo=None)
            b = b.replace(tzinfo=None)
        return abs((a - b).total_seconds()) / 60.0
    except Exception:
        return None


# =========================
# Policy filters
# =========================

def filter_pairs_by_policy(pairs: List[Dict], prefer_non_gated: bool, use_nac_only: bool, allow_unknown_ac: bool) -> List[Dict]:
    if not pairs:
        return []
    out = list(pairs)

    if prefer_non_gated:
        ng = [p for p in out if not (p["pet"]["flags"].get("gated") or p["ct"]["flags"].get("gated"))]
        if ng:
            out = ng

    ac_true = [p for p in out if p["pet"]["flags"].get("ac") is True]
    ac_unknown = [p for p in out if p["pet"]["flags"].get("ac") is None]
    ac_false = [p for p in out if p["pet"]["flags"].get("ac") is False]

    if use_nac_only:
        return ac_false
    if ac_true:
        return ac_true
    if allow_unknown_ac and ac_unknown:
        return ac_unknown
    return []


def explain_policy(pairs: List[Dict], prefer_non_gated: bool, use_nac_only: bool, allow_unknown_ac: bool) -> Dict:
    total = len(pairs)
    gated = sum(1 for p in pairs if p["pet"]["flags"].get("gated") or p["ct"]["flags"].get("gated"))
    non_gated = total - gated
    ac_true = sum(1 for p in pairs if p["pet"]["flags"].get("ac") is True)
    ac_unknown = sum(1 for p in pairs if p["pet"]["flags"].get("ac") is None)
    ac_false = sum(1 for p in pairs if p["pet"]["flags"].get("ac") is False)
    kept = len(filter_pairs_by_policy(pairs, prefer_non_gated, use_nac_only, allow_unknown_ac))

    reasons = []
    if total == 0:
        reasons.append("no_pairs")
    elif kept == 0:
        if ac_true == 0 and ac_unknown > 0 and not allow_unknown_ac:
            reasons.append("policy_excluded_unknown_ac")
        if ac_false == total and not use_nac_only:
            reasons.append("policy_excluded_nac(use_nac_only=False)")
        if non_gated == 0 and prefer_non_gated:
            reasons.append("only_gated_available(prefer_non_gated=True)")

    return {
        "counts": {
            "total_pairs": total,
            "non_gated": non_gated,
            "gated": gated,
            "pet_ac_true": ac_true,
            "pet_ac_unknown": ac_unknown,
            "pet_ac_false_nac": ac_false,
            "kept_after_policy": kept,
        },
        "reasons": reasons,
    }


# =========================
# CT quality scoring (lower = better)
# =========================

def ct_quality_score(ct: Dict, prefer_wb: bool = True, prefer_non_contrast: bool = True, prefer_non_gated: bool = True) -> Tuple[int, int, int]:
    """
    Score a CT series for quality/preference. Lower tuple = better.
    Returns (contrast_rank, gated_rank, wb_rank).

    contrast_rank: 0=non-contrast, 1=unknown, 2=contrast
    gated_rank:    0=non-gated, 1=gated
    wb_rank:       0=whole_body, 1=other
    """
    flags = ct.get("flags", {}) or {}
    classification = ct.get("classification", {}) or {}

    # Contrast: prefer non-contrast
    contrast = flags.get("contrast")
    if prefer_non_contrast:
        contrast_rank = 0 if contrast is False else (2 if contrast is True else 1)
    else:
        contrast_rank = 0  # don't care

    # Gated: prefer non-gated
    gated = flags.get("gated", False)
    if prefer_non_gated:
        gated_rank = 1 if gated else 0
    else:
        gated_rank = 0

    # Whole body: prefer whole body
    wb = classification.get("is_whole_body", False)
    if prefer_wb:
        wb_rank = 0 if wb else 1
    else:
        wb_rank = 0

    return (contrast_rank, gated_rank, wb_rank)


# =========================
# Pairing logic
# =========================

def series_stub(r: Dict) -> Dict:
    return {
        "asset_id": r.get("asset_id",""),
        "series_rel": r.get("series_rel",""),
        "series_dt": r.get("series_dt",""),
        "study_uid": r.get("study_uid",""),
        "series_uid": r.get("series_uid",""),
        "for_uid": r.get("for_uid",""),
        "patient_id": r.get("patient_id",""),
        "series_desc": r.get("series_desc",""),
        "protocol_name": r.get("protocol_name",""),
        "flags": r.get("flags", {}),
        "classification": r.get("classification", {}),
        "pet": r.get("pet", {}) if r.get("modality") == "PET" else {},
    }

def compare_headers(pet: Dict, ct: Dict) -> Dict:
    pet_dt = parse_dt(pet.get("series_dt",""))
    ct_dt = parse_dt(ct.get("series_dt",""))
    dt_min = minutes_delta(ct_dt, pet_dt)
    return {
        "patient_id_match": bool(pet.get("patient_id") and pet.get("patient_id") == ct.get("patient_id")),
        "study_uid_match": bool(pet.get("study_uid") and pet.get("study_uid") == ct.get("study_uid")),
        "for_uid_match": bool(pet.get("for_uid") and pet.get("for_uid") == ct.get("for_uid")),
        "acquisition_time_delta_min": dt_min,
    }

def pair_pet_ct_within_parent(pets: List[Dict], cts: List[Dict], time_window_min: int, allow_ct_reuse: bool,
                              prefer_wb: bool = True, prefer_non_contrast: bool = True, prefer_non_gated_ct: bool = True) -> List[Dict]:
    """
    Returns candidate pairs (not policy-filtered, not selected).
    Pairing methods:
      (1) StudyInstanceUID exact
      (2) PatientID + time_window, score improves if FoR matches
    CT selection within each method prefers: non-contrast, non-gated, whole-body.
    """
    def pet_pref_key(p):
        ac = p.get("flags",{}).get("ac")
        gated = p.get("flags",{}).get("gated")
        wb = (p.get("classification",{}) or {}).get("is_whole_body", False)
        dt = dt_sort_key(parse_dt(p.get("series_dt","")))
        ac_rank = 0 if ac is True else 1 if ac is None else 2
        return (ac_rank, 1 if gated else 0, 0 if wb else 1, dt)

    def ct_sort_key(ct, pet_dt=None):
        """Combined key: CT quality first, then time proximity."""
        quality = ct_quality_score(ct, prefer_wb=prefer_wb, prefer_non_contrast=prefer_non_contrast, prefer_non_gated=prefer_non_gated_ct)
        time = _safe_minutes_delta(parse_dt(ct.get("series_dt","")), pet_dt) if pet_dt else float("inf")
        return (quality, time)

    pets_sorted = sorted(pets, key=pet_pref_key)
    cts_sorted = sorted(cts, key=lambda c: dt_sort_key(parse_dt(c.get("series_dt",""))))

    used_ct: Set[str] = set()
    pairs: List[Dict] = []

    # (1) Study UID
    for pet in pets_sorted:
        suid = pet.get("study_uid","")
        if not suid:
            continue
        pet_dt = parse_dt(pet.get("series_dt",""))
        candidates = [ct for ct in cts_sorted if ct.get("study_uid","") == suid and (allow_ct_reuse or ct.get("asset_id","") not in used_ct)]
        if not candidates:
            continue
        best = min(candidates, key=lambda ct: ct_sort_key(ct, pet_dt))
        if not allow_ct_reuse:
            used_ct.add(best.get("asset_id",""))
        pairs.append({
            "pairing_method": "StudyInstanceUID",
            "pet": series_stub(pet),
            "ct": series_stub(best),
            "header_comparison": compare_headers(pet, best),
            "why_paired": "Same StudyInstanceUID; CT selected by quality + time.",
        })

    # (2) Time ± FoR (PatientID gated)
    for pet in pets_sorted:
        pet_id = pet.get("asset_id","")
        if any(p["pet"]["asset_id"] == pet_id for p in pairs):
            continue

        pet_dt = parse_dt(pet.get("series_dt",""))
        if pet_dt is None:
            continue

        cand_scored: List[Tuple] = []
        for ct in cts_sorted:
            if (not allow_ct_reuse) and (ct.get("asset_id","") in used_ct):
                continue
            if pet.get("patient_id") and ct.get("patient_id") and pet.get("patient_id") != ct.get("patient_id"):
                continue
            ct_dt = parse_dt(ct.get("series_dt",""))
            dm = minutes_delta(ct_dt, pet_dt)
            if dm is None or dm > float(time_window_min):
                continue
            quality = ct_quality_score(ct, prefer_wb=prefer_wb, prefer_non_contrast=prefer_non_contrast, prefer_non_gated=prefer_non_gated_ct)
            for_bonus = -10.0 if (pet.get("for_uid") and ct.get("for_uid") and pet.get("for_uid") == ct.get("for_uid")) else 0.0
            # Sort key: quality tuple first, then time + FoR bonus
            cand_scored.append((quality, float(dm) + for_bonus, ct))

        if not cand_scored:
            continue

        cand_scored.sort(key=lambda x: (x[0], x[1]))
        best = cand_scored[0][2]
        if not allow_ct_reuse:
            used_ct.add(best.get("asset_id",""))
        cmp_ = compare_headers(pet, best)
        method = "Time±FoR" if cmp_.get("for_uid_match") else "TimeOnly"
        pairs.append({
            "pairing_method": method,
            "pet": series_stub(pet),
            "ct": series_stub(best),
            "header_comparison": cmp_,
            "why_paired": f"Same PatientID (when available) and ≤{time_window_min} min; CT selected by quality. " + ("FoR matched." if method == "Time±FoR" else "FoR unequal/unavailable."),
        })

    return pairs


def pair_all_timepoints(pets: List[Dict], cts: List[Dict], time_window_min: int,
                        prefer_wb: bool = True, prefer_non_contrast: bool = True, prefer_non_gated_ct: bool = True) -> List[Dict]:
    """
    Generate ALL valid PET↔CT pairs within a parent (for multi-timepoint folders).
    Each PET gets its best-matching CT independently. CTs can be reused across PETs.
    CT selection prefers: non-contrast, non-gated, whole-body.
    Pairs are returned sorted by PET acquisition time.
    """
    pairs: List[Dict] = []

    for pet in pets:
        pet_dt = parse_dt(pet.get("series_dt", ""))
        best_ct = None
        best_key = None
        best_method = ""

        suid = pet.get("study_uid", "")

        for ct in cts:
            ct_dt = parse_dt(ct.get("series_dt", ""))
            quality = ct_quality_score(ct, prefer_wb=prefer_wb, prefer_non_contrast=prefer_non_contrast, prefer_non_gated=prefer_non_gated_ct)

            # Method 1: StudyUID match — strong preference
            if suid and ct.get("study_uid", "") == suid:
                dm = _safe_minutes_delta(ct_dt, pet_dt)
                for_bonus = -10.0 if (pet.get("for_uid") and ct.get("for_uid") and pet["for_uid"] == ct["for_uid"]) else 0.0
                # StudyUID match gets massive priority, then quality, then time
                key = (0, quality, dm + for_bonus)
                if best_key is None or key < best_key:
                    best_key = key
                    best_ct = ct
                    best_method = "StudyInstanceUID"
                continue

            # Method 2: Time window (± FoR)
            if pet.get("patient_id") and ct.get("patient_id") and pet["patient_id"] != ct["patient_id"]:
                continue
            dm = minutes_delta(ct_dt, pet_dt)
            if dm is None or dm > float(time_window_min):
                continue
            for_bonus = -10.0 if (pet.get("for_uid") and ct.get("for_uid") and pet["for_uid"] == ct["for_uid"]) else 0.0
            method = "Time±FoR" if for_bonus < 0 else "TimeOnly"
            key = (1, quality, float(dm) + for_bonus)
            if best_key is None or key < best_key:
                best_key = key
                best_ct = ct
                best_method = method

        if best_ct is None:
            continue

        cmp = compare_headers(pet, best_ct)
        pairs.append({
            "pairing_method": best_method,
            "pet": series_stub(pet),
            "ct": series_stub(best_ct),
            "header_comparison": cmp,
            "why_paired": f"{best_method} match (all-pairs mode); CT selected by quality.",
        })

    # Sort by PET acquisition time
    pairs.sort(key=lambda p: dt_sort_key(parse_dt(p["pet"].get("series_dt", ""))))
    return pairs


# =========================
# Pair entry helpers
# =========================

def _make_pair_entry(pair: Dict, reason: str) -> Dict:
    """
    Build a standardized pair entry with a segmentations slot.
    Downstream modules append segmentation records into the 'segmentations' dict.

    Segmentation schema (written by later modules, not this tool):
      segmentations:
        "<task_name>":                   # e.g. "total", "cardiac", "vertebrae"
          source_ct_asset_id: str        # always set — the CT this seg was generated from
          seg_type: "combined" | "individual"
          combined_path:                 # present if seg_type == "combined"
            seg_root_label: str
            seg_path_rel: str
          organ_dir:                     # present if seg_type == "individual"
            seg_root_label: str
            dir_path_rel: str
            organ_files: [str, ...]      # filenames inside dir
          ts_version: str
          created_at: str
    """
    entry = {
        "reason": reason,
        "pet_asset_id": pair["pet"]["asset_id"],
        "pet_series_rel": pair["pet"].get("series_rel", ""),
        "ct_asset_id": pair["ct"]["asset_id"],
        "ct_series_rel": pair["ct"].get("series_rel", ""),
        "pairing_method": pair["pairing_method"],
        "time_delta_min": pair["header_comparison"].get("acquisition_time_delta_min"),
        "segmentations": {},
    }
    return entry


def _make_pair_entry_from_fields(reason: str, pet_asset_id: str, pet_series_rel: str,
                                  ct_asset_id: str, ct_series_rel: str,
                                  pairing_method: str = "", time_delta_min=None) -> Dict:
    """Build a pair entry from explicit fields (for overrides/error cases)."""
    return {
        "reason": reason,
        "pet_asset_id": pet_asset_id,
        "pet_series_rel": pet_series_rel,
        "ct_asset_id": ct_asset_id,
        "ct_series_rel": ct_series_rel,
        "pairing_method": pairing_method,
        "time_delta_min": time_delta_min,
        "segmentations": {},
    }


def _make_all_pairs_entry(pair: Dict) -> Dict:
    """Build an entry for all_selected_pairs list."""
    return {
        "pet_asset_id": pair["pet"]["asset_id"],
        "pet_series_rel": pair["pet"].get("series_rel", ""),
        "pet_series_dt": pair["pet"].get("series_dt", ""),
        "ct_asset_id": pair["ct"]["asset_id"],
        "ct_series_rel": pair["ct"].get("series_rel", ""),
        "pairing_method": pair["pairing_method"],
        "time_delta_min": pair["header_comparison"].get("acquisition_time_delta_min"),
        "pet_flags": pair["pet"].get("flags", {}),
        "segmentations": {},
    }


# =========================
# Manual overrides
# =========================

def load_overrides(path: Optional[Path]) -> Dict:
    if not path:
        return {}
    try:
        data = load_yaml(path)
        return {
            "selections": dict(data.get("selections", {}) or {}),
        }
    except Exception:
        return {}

def resolve_series_by_id(series: List[Dict]) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    by_asset = {}
    by_series_uid = {}
    for r in series:
        aid = r.get("asset_id","")
        suid = r.get("series_uid","")
        if aid:
            by_asset[aid] = r
        if suid:
            by_series_uid[suid] = r
    return by_asset, by_series_uid


# =========================
# Build SelectionTemplate (for needs_selection)
# =========================

def build_selection_template(needs: List[Dict], by_parent: Dict[str, Dict]) -> Dict:
    tmpl = {
        "meta": {"generated_at": now_iso()},
        "how_to": "For each parent_rel, set exactly one PET and one CT entry to use:true, then save as Overrides.yaml under selections (asset ids preferred).",
        "candidates": {},
    }
    for it in needs:
        pr = it["parent_rel"]
        d = by_parent.get(pr, {})
        pets = d.get("pets", [])
        cts = d.get("cts", [])
        tmpl["candidates"][pr] = {
            "pet_series": [{**series_stub(p), "use": False} for p in pets],
            "ct_series": [{**series_stub(c), "use": False} for c in cts],
        }
    return tmpl


# =========================
# Preserve segmentations when re-running pair
# =========================

def _preserve_segmentations_in_entry(new_entry: Dict, old_entry: Dict) -> None:
    """
    Copy segmentations from old_entry to new_entry if the pair identity
    (pet_asset_id + ct_asset_id) matches.
    """
    if not isinstance(new_entry, dict) or not isinstance(old_entry, dict):
        return
    if (new_entry.get("pet_asset_id") != old_entry.get("pet_asset_id") or
            new_entry.get("ct_asset_id") != old_entry.get("ct_asset_id")):
        return
    old_seg = old_entry.get("segmentations")
    if isinstance(old_seg, dict) and old_seg:
        new_entry.setdefault("segmentations", {})
        for k, v in old_seg.items():
            if k not in new_entry["segmentations"]:
                new_entry["segmentations"][k] = copy.deepcopy(v)


def preserve_existing_segmentations(new_doc: dict, old_doc: dict) -> None:
    """
    Preserve existing segmentation records when re-running pairing.

    Preserves:
      - old_doc.derived (entire dict) into new_doc.derived (without overwriting new keys)
      - selected_pairs[parent].segmentations ONLY if pet_asset_id and ct_asset_id match
      - all_selected_pairs[parent][i].segmentations with same identity matching

    This prevents losing TS-runner outputs when PairFromHeaders is re-run.
    """
    if not isinstance(old_doc, dict) or not isinstance(new_doc, dict):
        return

    # Preserve derived block (without overwriting keys that new_doc already has)
    old_derived = old_doc.get("derived")
    if isinstance(old_derived, dict) and old_derived:
        new_doc.setdefault("derived", {})
        for k, v in old_derived.items():
            if k not in new_doc["derived"]:
                new_doc["derived"][k] = copy.deepcopy(v)

    # Preserve segmentations in selected_pairs
    old_sp = old_doc.get("selected_pairs") or {}
    new_sp = new_doc.get("selected_pairs") or {}
    if isinstance(old_sp, dict) and isinstance(new_sp, dict):
        for pr, new_sel in new_sp.items():
            old_sel = old_sp.get(pr)
            if isinstance(old_sel, dict):
                _preserve_segmentations_in_entry(new_sel, old_sel)

    # Preserve segmentations in all_selected_pairs
    old_all = old_doc.get("all_selected_pairs") or {}
    new_all = new_doc.get("all_selected_pairs") or {}
    if isinstance(old_all, dict) and isinstance(new_all, dict):
        for pr, new_list in new_all.items():
            old_list = old_all.get(pr)
            if not isinstance(new_list, list) or not isinstance(old_list, list):
                continue
            # Build lookup from old entries by (pet_asset_id, ct_asset_id)
            old_by_key = {}
            for entry in old_list:
                if isinstance(entry, dict):
                    key = (entry.get("pet_asset_id", ""), entry.get("ct_asset_id", ""))
                    old_by_key[key] = entry
            for new_entry in new_list:
                if not isinstance(new_entry, dict):
                    continue
                key = (new_entry.get("pet_asset_id", ""), new_entry.get("ct_asset_id", ""))
                old_entry = old_by_key.get(key)
                if old_entry:
                    _preserve_segmentations_in_entry(new_entry, old_entry)


# =========================
# Main pairing
# =========================

def pair_from_headers(
    header_doc: Dict,
    time_window_min: int,
    allow_ct_reuse: bool,
    prefer_non_gated: bool,
    use_nac_only: bool,
    allow_unknown_ac: bool,
    overrides: Dict,
    generate_all_pairs: bool = False,
    prefer_wb: bool = True,
    prefer_non_contrast: bool = True,
    prefer_non_gated_ct: bool = True,
    exclude_contrast_ct: bool = False,
) -> Tuple[Dict, Optional[Dict]]:
    series = list(header_doc.get("dicom_series", []) or [])
    # Only PET/CT; exclude scouts from candidate pools
    series = [r for r in series if r.get("modality") in ("PET","CT")]
    by_asset, by_series_uid = resolve_series_by_id(series)

    parents = sorted({r.get("parent_rel",".") for r in series})
    by_parent: Dict[str, Dict] = {}
    needs_selection: List[Dict] = []
    selected_pairs: Dict[str, Dict] = {}
    all_pairs_by_parent: Dict[str, List[Dict]] = {}

    # apply manual overrides first
    override_selections = (overrides.get("selections", {}) or {}) if overrides else {}

    # pre-group
    for pr in parents:
        rows = [r for r in series if r.get("parent_rel",".") == pr]
        pets = [r for r in rows if r.get("modality") == "PET" and not (r.get("flags",{}).get("scout"))]
        cts  = [r for r in rows if r.get("modality") == "CT"  and not (r.get("flags",{}).get("scout"))]
        if exclude_contrast_ct:
            cts = [c for c in cts if c.get("flags",{}).get("contrast") is not True]
        by_parent[pr] = {"pets": pets, "cts": cts}

    # manual selections
    handled: Set[str] = set()
    for pr, sel in override_selections.items():
        if pr not in by_parent:
            selected_pairs[pr] = _make_pair_entry_from_fields(
                reason="override_parent_not_found",
                pet_asset_id="", pet_series_rel="",
                ct_asset_id="", ct_series_rel="",
            )
            handled.add(pr)
            continue

        pet_aid = sel.get("pet_asset_id","") or ""
        ct_aid  = sel.get("ct_asset_id","") or ""
        pet_suid = sel.get("pet_series_uid","") or ""
        ct_suid  = sel.get("ct_series_uid","") or ""

        pet = by_asset.get(pet_aid) if pet_aid else by_series_uid.get(pet_suid)
        ct  = by_asset.get(ct_aid)  if ct_aid  else by_series_uid.get(ct_suid)

        if pet and ct:
            selected_pairs[pr] = _make_pair_entry_from_fields(
                reason="manual_override",
                pet_asset_id=pet.get("asset_id",""),
                pet_series_rel=pet.get("series_rel",""),
                ct_asset_id=ct.get("asset_id",""),
                ct_series_rel=ct.get("series_rel",""),
                pairing_method="manual",
            )
        else:
            selected_pairs[pr] = _make_pair_entry_from_fields(
                reason="override_invalid_ids",
                pet_asset_id=pet_aid or pet_suid,
                pet_series_rel="",
                ct_asset_id=ct_aid or ct_suid,
                ct_series_rel="",
            )
        handled.add(pr)

    # algorithmic pairing for the rest
    per_parent_details: List[Dict] = []
    for pr in parents:
        if pr in handled:
            continue

        pets = by_parent[pr]["pets"]
        cts  = by_parent[pr]["cts"]

        candidates = pair_pet_ct_within_parent(pets, cts, time_window_min=time_window_min, allow_ct_reuse=allow_ct_reuse,
                                                prefer_wb=prefer_wb, prefer_non_contrast=prefer_non_contrast, prefer_non_gated_ct=prefer_non_gated_ct)

        # policy-filter
        filtered = filter_pairs_by_policy(candidates, prefer_non_gated, use_nac_only, allow_unknown_ac)
        pol = explain_policy(candidates, prefer_non_gated, use_nac_only, allow_unknown_ac)

        # choose best among filtered
        def score(pair):
            method_rank = {"StudyInstanceUID": 0, "Time±FoR": 1, "TimeOnly": 2}.get(pair["pairing_method"], 9)
            ct_q = ct_quality_score(pair["ct"], prefer_wb=prefer_wb, prefer_non_contrast=prefer_non_contrast, prefer_non_gated=prefer_non_gated_ct)
            dt = pair["header_comparison"].get("acquisition_time_delta_min")
            dt = float(dt) if dt is not None else float("inf")
            for_match = 0 if pair["header_comparison"].get("for_uid_match") else 1
            return (method_rank, ct_q, for_match, dt)

        selected = None
        if filtered:
            selected = sorted(filtered, key=score)[0]
            selected_pairs[pr] = _make_pair_entry(selected, reason="auto_selected")
        else:
            needs_selection.append({
                "parent_rel": pr,
                "reason": "no_candidates_after_policy" if candidates else "no_pairs_found",
                "policy": pol,
                "pet_count": len(pets),
                "ct_count": len(cts),
                "candidate_pair_count": len(candidates),
            })

        # --- all-pairs mode (multi-timepoint) ---
        if generate_all_pairs:
            all_tp = pair_all_timepoints(pets, cts, time_window_min=time_window_min,
                                        prefer_wb=prefer_wb, prefer_non_contrast=prefer_non_contrast, prefer_non_gated_ct=prefer_non_gated_ct)
            # policy-filter each pair individually
            all_filtered = filter_pairs_by_policy(all_tp, prefer_non_gated, use_nac_only, allow_unknown_ac)
            if all_filtered:
                all_pairs_by_parent[pr] = [_make_all_pairs_entry(p) for p in all_filtered]

        # record parent detail
        per_parent_details.append({
            "parent_rel": pr,
            "pets": [series_stub(p) for p in pets],
            "cts": [series_stub(c) for c in cts],
            "candidate_pairs": candidates,
            "policy_summary": pol,
            "selected": selected if selected else None,
        })

    # stats
    total_parents = len(parents)
    paired_parents = sum(1 for pr in parents if pr in selected_pairs and selected_pairs[pr].get("reason") in ("auto_selected","manual_override"))
    needs_count = len(needs_selection)
    unpaired_parents = total_parents - paired_parents

    doc = {
        "meta": {
            "tool": "CLI_PairFromHeaders.py",
            "generated_at": now_iso(),
            "source_headers_generated_at": header_doc.get("meta", {}).get("generated_at",""),
            "policy": {
                "time_window_min": int(time_window_min),
                "allow_ct_reuse": bool(allow_ct_reuse),
                "prefer_non_gated": bool(prefer_non_gated),
                "use_nac_only": bool(use_nac_only),
                "allow_unknown_ac": bool(allow_unknown_ac),
                "generate_all_pairs": bool(generate_all_pairs),
                "prefer_wb": bool(prefer_wb),
                "prefer_non_contrast": bool(prefer_non_contrast),
                "prefer_non_gated_ct": bool(prefer_non_gated_ct),
                "exclude_contrast_ct": bool(exclude_contrast_ct),
            },
        },
        "stats": {
            "total_parents": int(total_parents),
            "paired_parents": int(paired_parents),
            "needs_selection": int(needs_count),
            "unpaired_parents": int(unpaired_parents),
            "total_series_pet": int(sum(1 for r in series if r.get("modality") == "PET")),
            "total_series_ct": int(sum(1 for r in series if r.get("modality") == "CT")),
        },
        "selected_pairs": selected_pairs,
        "needs_selection": needs_selection,
        "parents": per_parent_details,
    }

    if generate_all_pairs:
        doc["all_selected_pairs"] = all_pairs_by_parent

    selection_template = build_selection_template(needs_selection, by_parent) if needs_selection else None
    return doc, selection_template


# =========================
# CLI
# =========================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pair PET↔CT from HeaderIndex (no filesystem reads).")
    sub = p.add_subparsers(dest="mode", required=True)

    pa = sub.add_parser("pair", help="Pair PET↔CT and write PairIndex.yaml")
    pa.add_argument("--headers", type=Path, required=True, help="HeaderIndex.yaml")
    pa.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: alongside HeaderIndex)")
    pa.add_argument("--time-window", type=int, default=15, help="Pairing window in minutes (time-only step)")
    pa.add_argument("--allow-ct-reuse", action="store_true", help="Allow same CT to be paired to multiple PETs within a parent")
    pa.add_argument("--allow-gated-selection", dest="prefer_non_gated", action="store_false",
                    help="Allow gated pairs even when non-gated alternatives exist")
    pa.set_defaults(prefer_non_gated=True)
    pa.add_argument("--use-nac-only", action="store_true", help="Keep only NAC PET pairs")
    pa.add_argument("--allow-unknown-ac", action="store_true", help="Allow unknown AC if AC not available")
    pa.add_argument("--all-pairs", action="store_true",
                    help="Generate all valid PET↔CT pairs per parent (multi-timepoint support). "
                         "Results written to 'all_selected_pairs' in PairIndex.")
    pa.add_argument("--no-prefer-wb", dest="prefer_wb", action="store_false",
                    help="Do NOT prefer whole-body CT over limited FOV CT (default: prefer WB).")
    pa.set_defaults(prefer_wb=True)
    pa.add_argument("--no-prefer-non-contrast", dest="prefer_non_contrast", action="store_false",
                    help="Do NOT prefer non-contrast CT (default: prefer non-contrast).")
    pa.set_defaults(prefer_non_contrast=True)
    pa.add_argument("--no-prefer-non-gated-ct", dest="prefer_non_gated_ct", action="store_false",
                    help="Do NOT prefer non-gated CT (default: prefer non-gated).")
    pa.set_defaults(prefer_non_gated_ct=True)
    pa.add_argument("--exclude-contrast-ct", action="store_true",
                    help="Completely exclude contrast-enhanced CTs from pairing. "
                         "Parents with only contrast CTs will go to needs_selection.")
    pa.add_argument("--overrides", type=Path, default=None, help="Overrides.yaml with manual selections")
    pa.add_argument("--write-selection-template", action="store_true", help="Also write SelectionTemplate.yaml if needed")
    pa.add_argument("--formats", type=str, default="yaml", help='Comma-separated: yaml,json (default: yaml)')
    pa.add_argument("--verbose","-v", action="store_true", help="Verbose logging")
    return p

def main():
    args = build_parser().parse_args()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    headers_path = args.headers.resolve()
    header_doc = load_yaml(headers_path)
    out_dir = args.out_dir.resolve() if args.out_dir else headers_path.parent.resolve()

    overrides = load_overrides(args.overrides.resolve()) if args.overrides else {}

    pair_doc, tmpl = pair_from_headers(
        header_doc=header_doc,
        time_window_min=int(args.time_window),
        allow_ct_reuse=bool(args.allow_ct_reuse),
        prefer_non_gated=bool(args.prefer_non_gated),
        use_nac_only=bool(args.use_nac_only),
        allow_unknown_ac=bool(args.allow_unknown_ac),
        overrides=overrides,
        generate_all_pairs=bool(args.all_pairs),
        prefer_wb=bool(args.prefer_wb),
        prefer_non_contrast=bool(args.prefer_non_contrast),
        prefer_non_gated_ct=bool(args.prefer_non_gated_ct),
        exclude_contrast_ct=bool(args.exclude_contrast_ct),
    )

    # ---- Preserve prior segmentations in PairIndex.yaml if it exists ----
    existing_pairindex = out_dir / "PairIndex.yaml"
    if existing_pairindex.exists():
        try:
            old_doc = load_yaml(existing_pairindex)
            preserve_existing_segmentations(pair_doc, old_doc)
            logger.info("[pair] preserved existing segmentations from prior PairIndex.yaml")
        except Exception as e:
            logger.warning(f"[pair] could not preserve existing segmentations: {e}")

    fmts = {x.strip().lower() for x in str(args.formats).split(",") if x.strip()}
    if not (fmts & {"yaml","json"}):
        logger.error('Invalid --formats. Use "yaml", "json", or both.')
        sys.exit(2)

    written = []
    if "yaml" in fmts:
        written.append(write_yaml(out_dir / "PairIndex.yaml", pair_doc))
        if args.write_selection_template and tmpl is not None:
            written.append(write_yaml(out_dir / "SelectionTemplate.yaml", tmpl))
    if "json" in fmts:
        written.append(write_json(out_dir / "PairIndex.json", pair_doc))
        if args.write_selection_template and tmpl is not None:
            written.append(write_json(out_dir / "SelectionTemplate.json", tmpl))

    logger.info(f"[done] wrote {len(written)} file(s)")
    for p in written:
        print(f"  - {p}")

if __name__ == "__main__":
    main()
