#!/usr/bin/env python3
"""
CLI_HeaderExtract.py — header extraction from AssetsIndex (DICOM series + optional NIfTI headers)

Writes: HeaderIndex.yaml (+ optional JSON)

DEFAULTS
- Pairing-relevant fields come from headers (DICOM tags), not folder names.
- Folder-name signals are OFF by default. Enable explicitly with --use-folder-name-signals.

Optional nibabel
- If nibabel is installed, NIfTI headers (shape/zooms/affine) are extracted.
- If not installed, NIfTI assets are still carried through with only file stats.

Typical usage
-------------
python CLI_HeaderExtract.py extract --assets "Directory Plan/AssetsIndex.yaml" -v
python CLI_HeaderExtract.py extract --assets AssetsIndex.yaml --use-folder-name-signals
"""

import argparse
import gzip
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml
import json
import pydicom
from pydicom.errors import InvalidDicomError


# =========================
# Logging / helpers
# =========================

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

def relpath_safe(p: Path, start: Path) -> str:
    try:
        rp = os.path.relpath(str(p), str(start))
        rp = str(Path(rp).as_posix())
        return "." if rp in ("", ".") else rp
    except Exception:
        return "."

def safe_str(x) -> str:
    try:
        return str(x or "").strip()
    except Exception:
        return ""


# =========================
# DICOM read/cache
# =========================

@dataclass
class ExtractConfig:
    dicom_preamble_size: int = 132
    dicom_magic: bytes = b"DICM"
    cache_size: int = 2000
    max_series_files_probe: int = 48
    max_named_sweep_files: int = 64
    max_subdirs_probe: int = 12
    sample_limit: int = 200
    whole_body_slice_threshold: int = 150

CONFIG = ExtractConfig()

_CT_SOPS = {"1.2.840.10008.5.1.4.1.1.2", "1.2.840.10008.5.1.4.1.1.2.1"}
_PET_SOPS = {"1.2.840.10008.5.1.4.1.1.128", "1.2.840.10008.5.1.4.1.1.130"}

UNITED_IMAGING_PATTERNS = [
    re.compile(r"^IM\d+$", re.I), re.compile(r"^\d+$"),
    re.compile(r"^I\d+$", re.I), re.compile(r"^IMG\d+", re.I),
    re.compile(r"^CT\d+", re.I), re.compile(r"^PET\d+", re.I),
]

def has_dicom_preamble(fp: Path) -> bool:
    try:
        with open(fp, "rb") as f:
            head = f.read(CONFIG.dicom_preamble_size)
        if len(head) >= CONFIG.dicom_preamble_size:
            return head[128:132] == CONFIG.dicom_magic
    except Exception:
        pass
    return False

@lru_cache(maxsize=CONFIG.cache_size)
def cached_dicom_read(file_path: str) -> Optional[pydicom.dataset.FileDataset]:
    try:
        if file_path.casefold().endswith(".gz"):
            with gzip.open(file_path, "rb") as g:
                return pydicom.dcmread(g, stop_before_pixels=True, force=True)
        return pydicom.dcmread(file_path, stop_before_pixels=True, force=True)
    except (InvalidDicomError, FileNotFoundError, PermissionError, UnicodeDecodeError):
        return None
    except Exception:
        return None

def looks_like_dicom(fp: Path) -> bool:
    if not fp.is_file():
        return False
    n = fp.name.casefold()
    if n.endswith(".dcm") or n.endswith(".dcm.gz"):
        return True
    if any(p.match(fp.name) for p in UNITED_IMAGING_PATTERNS):
        if has_dicom_preamble(fp):
            return True
    return cached_dicom_read(str(fp)) is not None

def _is_image_instance(ds) -> bool:
    try:
        mod = safe_str(getattr(ds, "Modality", "")).upper()
        if mod in {"CT","PT","PET"}:
            return True
        sop = safe_str(getattr(ds, "SOPClassUID", ""))
        ms = safe_str(getattr(getattr(ds, "file_meta", None), "MediaStorageSOPClassUID", ""))
        return sop in _CT_SOPS or sop in _PET_SOPS or ms in _CT_SOPS or ms in _PET_SOPS
    except Exception:
        return False

def _pick_files_for_probe(dirpath: Path, limit: int) -> List[Path]:
    files = []
    try:
        for f in dirpath.iterdir():
            if f.is_file():
                files.append(f)
    except Exception:
        return []

    def score(p: Path):
        n = p.name.casefold()
        s = 0
        if n.endswith(".dcm") or n.endswith(".dcm.gz"):
            s -= 3
        if any(pat.match(p.name) for pat in UNITED_IMAGING_PATTERNS):
            s -= 2
        if has_dicom_preamble(p):
            s -= 4
        return (s, n)

    files.sort(key=score)
    return files[:limit]

def read_any_dicom(folder: Path) -> pydicom.dataset.FileDataset:
    for p in _pick_files_for_probe(folder, CONFIG.max_series_files_probe):
        ds = cached_dicom_read(str(p))
        if ds is None:
            continue
        if _is_image_instance(ds):
            return ds

    swept = 0
    try:
        for p in folder.iterdir():
            if not p.is_file():
                continue
            n = p.name.casefold()
            if not (n.endswith(".dcm") or n.endswith(".dcm.gz")):
                continue
            ds = cached_dicom_read(str(p))
            if ds and _is_image_instance(ds):
                return ds
            swept += 1
            if swept >= CONFIG.max_named_sweep_files:
                break
    except Exception:
        pass

    subdirs = []
    try:
        for c in folder.iterdir():
            if c.is_dir():
                subdirs.append(c)
    except Exception:
        subdirs = []
    subdirs.sort()

    for sd in subdirs[:CONFIG.max_subdirs_probe]:
        for p in _pick_files_for_probe(sd, max(8, CONFIG.max_series_files_probe // 3)):
            ds = cached_dicom_read(str(p))
            if ds and _is_image_instance(ds):
                return ds

    raise FileNotFoundError(f"No readable CT/PT/PET DICOM found in: {folder}")


# =========================
# Core header parsing
# =========================

def modality_of(ds) -> str:
    try:
        m = safe_str(getattr(ds, "Modality", "")).upper()
        if m in ("CT","PT","PET"):
            return "CT" if m == "CT" else "PET"
        sop = safe_str(getattr(ds, "SOPClassUID", ""))
        if sop in _CT_SOPS:
            return "CT"
        if sop in _PET_SOPS:
            return "PET"
    except Exception:
        pass
    return "Other"

def parse_dicom_tz_offset(ds) -> Optional[timezone]:
    try:
        off = safe_str(getattr(ds, "TimezoneOffsetFromUTC", ""))
        if not off or len(off) < 3:
            return None
        sign = 1 if off.startswith("+") else -1
        hh = int(off[1:3])
        mm = int(off[3:5]) if len(off) >= 5 else 0
        if hh > 14 or (hh == 14 and mm > 0):
            return None
        return timezone(sign * timedelta(hours=hh, minutes=mm))
    except Exception:
        return None

def parse_acquisition_datetime(ds) -> Tuple[Optional[datetime], str]:
    tz = parse_dicom_tz_offset(ds)

    def parse_dt(dt_raw: str) -> Optional[datetime]:
        if not dt_raw:
            return None
        try:
            main, dot, rest = dt_raw.strip().partition(".")
            frac = ""
            off = None
            if dot:
                m = re.match(r"^(\d{1,6})([+\-]\d{2,4})?$", rest)
                if m:
                    frac = m.group(1) or ""
                    off = m.group(2)
            else:
                m = re.match(r"^(.+?)([+\-]\d{2,4})$", main)
                if m:
                    main = m.group(1)
                    off = m.group(2)

            if len(main) < 8:
                return None
            base = datetime.strptime(main[:14].ljust(14, "0"), "%Y%m%d%H%M%S")
            if frac:
                base = base.replace(microsecond=int((frac + "000000")[:6]))

            if tz:
                return base.replace(tzinfo=tz)
            if off:
                s = 1 if off.startswith("+") else -1
                hh = int(off[1:3])
                mm = int(off[3:5]) if len(off) >= 5 else 0
                return base.replace(tzinfo=timezone(s * timedelta(hours=hh, minutes=mm)))
            return base
        except Exception:
            return None

    for tag in ("AcquisitionDateTime", "ContentDateTime"):
        dt = parse_dt(safe_str(getattr(ds, tag, "")))
        if dt:
            return dt, tag

    try:
        d = safe_str(getattr(ds, "AcquisitionDate", "")) or safe_str(getattr(ds, "SeriesDate", ""))
        t = safe_str(getattr(ds, "AcquisitionTime", "")) or safe_str(getattr(ds, "SeriesTime", ""))
        if d and len(d) >= 8:
            y, mn, dy = int(d[0:4]), int(d[4:6]), int(d[6:8])
            hh = int(t[0:2]) if len(t) >= 2 else 0
            mm = int(t[2:4]) if len(t) >= 4 else 0
            ss = int(t[4:6]) if len(t) >= 6 else 0
            base = datetime(y, mn, dy, hh, mm, ss)
            if tz:
                base = base.replace(tzinfo=tz)
            return base, "Date+Time"
    except Exception:
        pass

    return None, ""

# gating detection (header-only by default)
UNGATED_PATTERNS = [
    re.compile(r"(?i)\bnon[-\s]?gated\b"),
    re.compile(r"(?i)\bungated\b"),
    re.compile(r"(?i)\bnot\s+gated\b"),
    re.compile(r"(?i)\bno\s+gating\b"),
    re.compile(r"(?i)\bwithout\s+gating\b"),
    re.compile(r"(?i)\bfree[-\s]?breath(ing)?\b"),
    re.compile(r"(?i)\bnon[-\s]?(ecg|cardiac)\s+gated\b"),
    re.compile(r"(?i)\bnon[-\s]?resp(iratory)?\s+gated\b"),
]
_RE_WORD = r"(?i)(^|[^A-Za-z0-9])({})([^A-Za-z0-9]|$)"
PET_GATED_PATTERNS = [
    re.compile(_RE_WORD.format("gated")),
    re.compile(r"(?i)\b(cardiac|resp(iratory)?)\b.{0,8}\bgated\b"),
    re.compile(r"(?i)\b4d\b.*\bpet\b"),
]
CT_GATED_PATTERNS = [
    re.compile(r"(?i)\b(ecg|cardiac)\b.{0,8}\bgated\b"),
    re.compile(_RE_WORD.format("cine")),
    re.compile(_RE_WORD.format("gated")),
]

def _split_tokens(val) -> List[str]:
    if val is None:
        return []
    s = " ".join(str(x) for x in val) if isinstance(val, (list, tuple)) else str(val)
    return [t for t in re.split(r"[^A-Za-z0-9]+", s) if t]

def detect_gated_flag(modality: str, series_desc: str, protocol_name: str, image_type_val, folder_name: str, use_folder: bool) -> bool:
    # 1) header text negations
    t = (" " + " ".join([series_desc or "", protocol_name or "", " ".join(_split_tokens(image_type_val))]) + " ").upper()
    for rx in UNGATED_PATTERNS:
        if rx.search(t):
            return False

    # 2) optional folder-name signals (explicit opt-in)
    if use_folder and folder_name:
        fn = (" " + folder_name + " ").upper()
        for rx in UNGATED_PATTERNS:
            if rx.search(fn):
                return False
        if re.search(r"(?i)(^|[ _-])gated($|[ _-])", folder_name) or folder_name.lower().startswith("gated"):
            return True

    # 3) positive patterns
    patterns = PET_GATED_PATTERNS if modality == "PET" else CT_GATED_PATTERNS
    return any(rx.search(t) for rx in patterns)


# =========================
# Contrast detection (CT)
# =========================

# Positive contrast indicators
CONTRAST_POSITIVE_PATTERNS = [
    re.compile(r"(?i)\bC\+\b"),
    re.compile(r"(?i)\bCE\b"),
    re.compile(r"(?i)\bCECT\b"),
    re.compile(r"(?i)\bwith\s+contrast\b"),
    re.compile(r"(?i)\bpost[-\s]?contrast\b"),
    re.compile(r"(?i)\bcontrast[-\s]?enhanced\b"),
    re.compile(r"(?i)\benhanced\b"),
    re.compile(r"(?i)\biodine\b"),
    re.compile(r"(?i)\barterial\b"),
    re.compile(r"(?i)\bvenous\b"),
    re.compile(r"(?i)\bportal\b"),
    re.compile(r"(?i)\bdelayed\b.*\bphase\b"),
    re.compile(r"(?i)\bpost[-\s]?gad\b"),
    re.compile(r"(?i)\bcontrast\b"),
    re.compile(r"(?i)\bIV\s+contrast\b"),
    re.compile(r"(?i)\bdynamic\b"),
]

# Negative contrast indicators (non-contrast)
CONTRAST_NEGATIVE_PATTERNS = [
    re.compile(r"(?i)\bC-\b"),
    re.compile(r"(?i)\bNC\b"),
    re.compile(r"(?i)\bNCCT\b"),
    re.compile(r"(?i)\bnon[-\s]?contrast\b"),
    re.compile(r"(?i)\bwithout\s+contrast\b"),
    re.compile(r"(?i)\bpre[-\s]?contrast\b"),
    re.compile(r"(?i)\bnative\b"),
    re.compile(r"(?i)\bnon[-\s]?enhanced\b"),
    re.compile(r"(?i)\bunenhanced\b"),
    re.compile(r"(?i)\bno\s+contrast\b"),
    re.compile(r"(?i)\bw/?o\s+contrast\b"),
    re.compile(r"(?i)\bplain\b"),
    re.compile(r"(?i)\bLDCT\b"),            # low-dose CT (typically non-contrast)
    re.compile(r"(?i)\blow[-\s]?dose\b"),
    re.compile(r"(?i)\bAC\s+CT\b"),          # attenuation correction CT (non-contrast)
]

def detect_contrast_flag(ds, series_desc: str, protocol_name: str, image_type_val) -> Tuple[Optional[bool], str]:
    """
    Detect whether a CT series used IV contrast.

    Returns (is_contrast, reason):
      - (True, reason)  — contrast detected
      - (False, reason) — non-contrast detected
      - (None, reason)  — cannot determine

    Detection priority:
      1) ContrastBolusAgent DICOM tag (most reliable)
      2) Text patterns in series_desc / protocol_name / ImageType (negatives checked first)
    """
    # 1) Check ContrastBolusAgent tag
    cba = safe_str(getattr(ds, "ContrastBolusAgent", ""))
    bad_vals = {"", "NONE", "NO", "N/A", "NA", "NULL", "-", "0"}
    if cba and cba.strip().upper() not in bad_vals:
        return True, f"ContrastBolusAgent={cba}"

    # Also check ContrastBolusAgentSequence
    cba_seq = getattr(ds, "ContrastBolusAgentSequence", None)
    if cba_seq and len(cba_seq) > 0:
        return True, "ContrastBolusAgentSequence_present"

    # 2) Text analysis on series_desc, protocol, image_type
    image_type_tokens = " ".join(_split_tokens(image_type_val))
    hay = f" {series_desc or ''} {protocol_name or ''} {image_type_tokens} "

    # Check negatives first (non-contrast)
    for rx in CONTRAST_NEGATIVE_PATTERNS:
        if rx.search(hay):
            return False, f"text_negative:{rx.pattern}"

    # Check positives
    for rx in CONTRAST_POSITIVE_PATTERNS:
        if rx.search(hay):
            return True, f"text_positive:{rx.pattern}"

    # Cannot determine
    return None, "undetermined"


# PET AC/NAC
AC_TEXT_TOKENS = [
    "AC","CTAC","MRAC","ATTENUATION CORRECTED","ATTENUATION-CORRECTED",
    "WITH ATTENUATION CORRECTION","AC-ONLY","AC ONLY","AC RECON","AC SERIES",
]
NAC_TEXT_TOKENS = [
    "NAC","NO AC","NON-AC","NONAC","NO-ATTN","NO ATTN","NON-ATTN","NON ATTN",
    "UNATTENUATED","UN-ATTENUATED","UNCORR","UN-CORR","UNCORRECTED","RAW (UNCORR)","RAW(UNCORR)",
]

def _token_set(val) -> Set[str]:
    return {t.upper() for t in _split_tokens(val)}

def _has_token_bound(s: str, token: str) -> bool:
    return re.search(rf"(?i)(^|[^A-Za-z0-9]){re.escape(token)}($|[^A-Za-z0-9])", s) is not None

def detect_pet_ac_flag(ds, series_desc: str, protocol_name: str, image_type_val) -> Tuple[Optional[bool], Dict]:
    ev = {"decisions": [], "text": {}, "corrected_image_tokens": [], "attenuation_tags": []}
    ci_tokens = _token_set(getattr(ds, "CorrectedImage", None))
    ev["corrected_image_tokens"] = sorted(ci_tokens)

    nac_ci = {"NAC","NOATTN","NONATTN","NO-ATTN","UNCORR","UNCORRECTED","UNATTENUATED","NONAC"}
    if nac_ci & ci_tokens:
        ev["decisions"].append("CorrectedImage indicates NAC")
        return False, ev
    if "ATTN" in ci_tokens:
        ev["decisions"].append("CorrectedImage indicates AC")
        return True, ev

    image_type_tokens = _token_set(image_type_val)
    image_type_str = " ".join(sorted(image_type_tokens)) if image_type_tokens else ""
    hay = f" {series_desc or ''} {protocol_name or ''} {image_type_str} ".upper()
    ev["text"] = {
        "series_desc": series_desc,
        "protocol_name": protocol_name,
        "image_type": image_type_str,
        "matched_ac_tokens": [t for t in AC_TEXT_TOKENS if _has_token_bound(hay, t)],
        "matched_nac_tokens": [t for t in NAC_TEXT_TOKENS if _has_token_bound(hay, t)],
    }
    if ev["text"]["matched_nac_tokens"]:
        ev["decisions"].append("Text tokens => NAC")
        return False, ev
    if ev["text"]["matched_ac_tokens"] or ({"AC","CTAC","MRAC"} & image_type_tokens):
        ev["decisions"].append("Text/ImageType tokens => AC")
        return True, ev

    bad_vals = {"","NONE","NO","N/A","NA","NULL","-"}
    acm = safe_str(getattr(ds, "AttenuationCorrectionMethod", ""))
    ev["attenuation_tags"].append(("AttenuationCorrectionMethod", acm))
    if acm and acm.strip().upper() not in bad_vals:
        ev["decisions"].append("AttenuationCorrectionMethod present & valid => AC")
        return True, ev
    ev["decisions"].append("No/invalid AttenuationCorrectionMethod => NAC")
    return False, ev

# PET tracer/uptake (best-effort)
def _parse_dicom_time_to_hms(t_raw: str) -> Optional[Tuple[int,int,int]]:
    if not t_raw:
        return None
    s = str(t_raw).strip()
    if not s:
        return None
    s = s.split(".")[0]
    s = re.sub(r"[^0-9]", "", s)
    if len(s) < 2:
        return None
    try:
        hh = int(s[0:2])
        mm = int(s[2:4]) if len(s) >= 4 else 0
        ss = int(s[4:6]) if len(s) >= 6 else 0
        if not (0 <= hh <= 23 and 0 <= mm <= 59 and 0 <= ss <= 59):
            return None
        return hh, mm, ss
    except Exception:
        return None

def extract_radiopharm(ds) -> Tuple[str, Optional[datetime], str]:
    name = ""
    inj_dt: Optional[datetime] = None
    src = ""
    try:
        seq = getattr(ds, "RadiopharmaceuticalInformationSequence", None)
        if seq:
            for item in seq:
                if not name:
                    name = safe_str(getattr(item, "Radiopharmaceutical", ""))
                    if name:
                        src = "RadiopharmaceuticalInformationSequence.Radiopharmaceutical"
                dt_raw = safe_str(getattr(item, "RadiopharmaceuticalStartDateTime", ""))
                if dt_raw:
                    try:
                        main = dt_raw.split(".")[0]
                        base = datetime.strptime(main[:14].ljust(14, "0"), "%Y%m%d%H%M%S")
                        tz = parse_dicom_tz_offset(ds)
                        if tz:
                            base = base.replace(tzinfo=tz)
                        inj_dt = base
                        src = "RadiopharmaceuticalStartDateTime"
                        break
                    except Exception:
                        pass
                tm_raw = safe_str(getattr(item, "RadiopharmaceuticalStartTime", ""))
                hms = _parse_dicom_time_to_hms(tm_raw)
                if hms:
                    inj_dt = datetime(1900,1,1,hms[0],hms[1],hms[2])
                    src = "RadiopharmaceuticalStartTime"
    except Exception:
        pass
    return name, inj_dt, src

def normalize_tracer(name: str, series_desc: str, protocol_name: str) -> Tuple[str, Dict]:
    hay = f" {name} {series_desc} {protocol_name} ".upper()
    ev = {"raw": name, "hits": []}
    def hit(rx, label):
        if re.search(rx, hay, flags=re.IGNORECASE):
            ev["hits"].append(label)
            return True
        return False
    if hit(r"\bFDG\b", "FDG"):
        return "FDG", ev
    if hit(r"\bNAF\b", "NaF"):
        return "NaF", ev
    if hit(r"\bFLT\b", "FLT"):
        return "FLT", ev
    if hit(r"\bPSMA\b", "PSMA"):
        return "PSMA", ev
    if hit(r"\bCHOLINE\b", "Choline"):
        return "Choline", ev
    if hit(r"\bDOTATATE\b|\bDOTA\b", "DOTATATE"):
        return "DOTATATE", ev
    return (name.strip() or ""), ev

# region heuristic (lightweight)
WB_HINTS_LONG = {"whole body","vertex to toes","head to toe","head-to-toe","skull to toes"}
SBMT_HINTS = {"skull base to mid-thigh","skull-base to mid-thigh","base of skull to mid thigh","sbmt"}

def region_from_text(series_desc: str, body_part: str) -> Optional[str]:
    t = (series_desc + " " + body_part).strip().lower()
    if any(w in t for w in WB_HINTS_LONG) or re.search(r"(?i)\bwb\b", t):
        return "whole_body"
    if any(k in t for k in SBMT_HINTS):
        return "chest"
    if any(k in t for k in ("chest","thorax","lung")):
        return "chest"
    if "head and neck" in t or "h&n" in t or ("head" in t and "neck" in t):
        return "head_neck"
    if "extremit" in t:
        return "extremities"
    return None

def estimate_slice_count(series_dir: Path) -> Optional[int]:
    total = 0
    try:
        for root, _, files in os.walk(series_dir):
            for name in files:
                p = Path(root) / name
                if not p.is_file():
                    continue
                if looks_like_dicom(p):
                    total += 1
                    if total >= CONFIG.sample_limit:
                        return total
            if total >= CONFIG.sample_limit:
                return total
    except Exception:
        return None
    return total if total > 0 else None

def classify_series(series_desc: str, body_part: str, slice_est: Optional[int]) -> Dict:
    ev = []
    if slice_est is not None:
        ev.append(f"slice_count_est={slice_est}")
        wb = slice_est >= CONFIG.whole_body_slice_threshold
        return {"region": "whole_body" if wb else "other", "source": "slice_only", "evidence": ev, "is_whole_body": wb}
    r = region_from_text(series_desc, body_part)
    if r:
        ev.append("text_region")
        return {"region": r, "source": "text_fallback", "evidence": ev, "is_whole_body": (r == "whole_body")}
    return {"region": "other", "source": "unknown", "evidence": ["no slice/text"], "is_whole_body": False}


# =========================
# NIfTI header extraction (optional)
# =========================

def try_extract_nifti_header(abs_path: Path) -> Tuple[Dict, List[str]]:
    warns: List[str] = []
    try:
        import nibabel as nib
    except Exception:
        return {}, ["nibabel_not_installed"]

    try:
        img = nib.load(str(abs_path))
        hdr = img.header
        zooms = tuple(float(x) for x in hdr.get_zooms()[:3])
        shape = tuple(int(x) for x in img.shape[:3])
        affine = img.affine
        aff = [[float(affine[r, c]) for c in range(4)] for r in range(4)]
        return {"shape": shape, "zooms": zooms, "affine": aff}, warns
    except Exception as e:
        return {}, [f"nifti_read_error:{str(e)[:200]}"]


# =========================
# Main extraction
# =========================

def build_dicom_series_record(asset: Dict, use_folder_name_signals: bool) -> Tuple[Optional[Dict], Optional[str]]:
    dicom_root = Path(asset["root_path"])
    series_dir = (dicom_root / Path(asset["path_rel"])).resolve()
    try:
        ds = read_any_dicom(series_dir)
    except Exception as e:
        return None, f"read_error:{str(e)[:200]}"

    modality = modality_of(ds)
    series_desc = safe_str(getattr(ds, "SeriesDescription", ""))
    body_part = safe_str(getattr(ds, "BodyPartExamined", ""))
    protocol = safe_str(getattr(ds, "ProtocolName", ""))
    patient_id = safe_str(getattr(ds, "PatientID", ""))
    study_uid = safe_str(getattr(ds, "StudyInstanceUID", ""))
    series_uid = safe_str(getattr(ds, "SeriesInstanceUID", ""))
    for_uid = safe_str(getattr(ds, "FrameOfReferenceUID", ""))
    manufacturer = safe_str(getattr(ds, "Manufacturer", ""))
    station_name = safe_str(getattr(ds, "StationName", ""))
    software_versions = safe_str(getattr(ds, "SoftwareVersions", ""))
    image_type_val = getattr(ds, "ImageType", None)

    series_dt, series_dt_source = parse_acquisition_datetime(ds)
    series_dt_str = series_dt.isoformat() if series_dt else ""

    slice_est = estimate_slice_count(series_dir) if modality in ("CT","PET") else None
    classification = classify_series(series_desc, body_part, slice_est)

    gated = detect_gated_flag(
        modality=modality,
        series_desc=series_desc,
        protocol_name=protocol,
        image_type_val=image_type_val,
        folder_name=series_dir.name,
        use_folder=use_folder_name_signals,
    )

    scout = any(k in (series_desc or "").lower() for k in ("scout","localizer","topogram","surview","pilot"))

    # --- Contrast detection (CT only) ---
    contrast_flag, contrast_reason = (None, "")
    if modality == "CT":
        contrast_flag, contrast_reason = detect_contrast_flag(ds, series_desc, protocol, image_type_val)

    ac_flag, ac_ev = (None, {})
    tracer, tracer_ev = ("", {})
    inj_dt_str, inj_src, uptake_min = ("", "", None)

    if modality == "PET":
        ac_flag, ac_ev = detect_pet_ac_flag(ds, series_desc, protocol, image_type_val)

        rname, inj_dt, inj_src0 = extract_radiopharm(ds)
        tracer, tracer_ev = normalize_tracer(rname, series_desc, protocol)

        if inj_dt and inj_dt.year == 1900 and series_dt:
            try:
                inj_dt = series_dt.replace(hour=inj_dt.hour, minute=inj_dt.minute, second=inj_dt.second, microsecond=0)
            except Exception:
                pass

        inj_dt_str = inj_dt.isoformat() if inj_dt else ""
        inj_src = inj_src0 or ("RadiopharmaceuticalInformationSequence" if rname else "")

        if inj_dt and series_dt:
            try:
                uptake_min = (series_dt - inj_dt).total_seconds() / 60.0
            except Exception:
                uptake_min = None

    stable_id = ""
    if study_uid and series_uid:
        stable_id = f"DICOM_SERIES:{study_uid}:{series_uid}"
    else:
        stable_id = f"DICOM_SERIES_FALLBACK:{asset['asset_id']}"

    rec = {
        "asset_id": stable_id,
        "source_asset_id": asset["asset_id"],
        "root_label": asset.get("root_label","dicom"),
        "root_path": asset["root_path"],
        "parent_rel": asset.get("parent_rel","."),
        "series_rel": asset.get("path_rel",""),
        "modality": modality,
        "series_desc": series_desc,
        "body_part": body_part,
        "protocol_name": protocol,
        "manufacturer": manufacturer,
        "station_name": station_name,
        "software_versions": software_versions,
        "patient_id": patient_id,
        "study_uid": study_uid,
        "series_uid": series_uid,
        "for_uid": for_uid,
        "series_dt": series_dt_str,
        "series_dt_source": series_dt_source,
        "slice_count_est": slice_est,
        "flags": {
            "gated": bool(gated),
            "scout": bool(scout),
            "contrast": contrast_flag,
            "contrast_reason": contrast_reason,
            "ac": ac_flag,
            "ac_reason": ac_ev if modality == "PET" else {},
        },
        "classification": classification,
        "pet": {
            "tracer": tracer,
            "tracer_evidence": tracer_ev,
            "injection_dt": inj_dt_str,
            "injection_dt_source": inj_src,
            "uptake_min": uptake_min,
        } if modality == "PET" else {},
    }
    return rec, None


def extract_headers(assets_doc: Dict, use_folder_name_signals: bool) -> Dict:
    assets = list(assets_doc.get("assets", []) or [])
    total_assets = len(assets)

    dicom_assets = [a for a in assets if a.get("kind") == "dicom_series_dir"]
    nifti_assets = [a for a in assets if a.get("kind") == "nifti_image"]
    seg_assets = [a for a in assets if a.get("kind") == "segmentation_file"]

    series_records: List[Dict] = []
    file_records: List[Dict] = []
    failures: List[Dict] = []

    for a in dicom_assets:
        rec, err = build_dicom_series_record(a, use_folder_name_signals=use_folder_name_signals)
        if rec is not None:
            series_records.append(rec)
        else:
            failures.append({"source_asset_id": a.get("asset_id",""), "kind": "dicom_series_dir", "reason": err or "unknown"})

    for a in nifti_assets + seg_assets:
        root = Path(a["root_path"]).resolve()
        abs_path = (root / Path(a["path_rel"])).resolve()
        header, warns = try_extract_nifti_header(abs_path) if abs_path.name.lower().endswith((".nii",".nii.gz")) else ({}, [])
        file_records.append({
            "asset_id": f"FILE:{a['asset_id']}",
            "source_asset_id": a["asset_id"],
            "kind": a["kind"],
            "root_label": a.get("root_label",""),
            "root_path": a["root_path"],
            "parent_rel": a.get("parent_rel","."),
            "path_rel": a.get("path_rel",""),
            "file_stats": a.get("file_stats", {}),
            "nifti_header": header,
            "warnings": warns,
        })

    by_modality: Dict[str, int] = {}
    for r in series_records:
        by_modality[r.get("modality","Other")] = by_modality.get(r.get("modality","Other"), 0) + 1

    pet_ct = {"PET": by_modality.get("PET",0), "CT": by_modality.get("CT",0)}
    parents_total = len({r.get("parent_rel",".") for r in series_records})

    doc = {
        "meta": {
            "tool": "CLI_HeaderExtract.py",
            "generated_at": now_iso(),
            "source_assets_generated_at": assets_doc.get("meta", {}).get("generated_at",""),
            "options": {
                "use_folder_name_signals": bool(use_folder_name_signals),
            },
        },
        "stats": {
            "assets_total": int(total_assets),
            "assets_dicom_series_dirs": int(len(dicom_assets)),
            "assets_nifti_images": int(len(nifti_assets)),
            "assets_seg_files": int(len(seg_assets)),
            "dicom_series_extracted": int(len(series_records)),
            "dicom_series_failed": int(len([f for f in failures if f.get("kind")=="dicom_series_dir"])),
            "dicom_series_parents": int(parents_total),
            "dicom_by_modality": by_modality,
            "dicom_pet_ct": pet_ct,
            "file_records_written": int(len(file_records)),
        },
        "dicom_series": sorted(series_records, key=lambda r: (r.get("parent_rel","."), r.get("modality",""), r.get("series_dt",""), r.get("series_rel",""))),
        "files": sorted(file_records, key=lambda r: (r.get("kind",""), r.get("root_label",""), r.get("parent_rel",""), r.get("path_rel",""))),
        "failures": failures,
    }
    return doc


# =========================
# CLI
# =========================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Header extraction from AssetsIndex.")
    sub = p.add_subparsers(dest="mode", required=True)

    e = sub.add_parser("extract", help="Read AssetsIndex and write HeaderIndex")
    e.add_argument("--assets", type=Path, required=True, help="Path to AssetsIndex.yaml")
    e.add_argument("--out-dir", type=Path, default=None, help="Output dir (default: alongside AssetsIndex)")
    e.add_argument("--use-folder-name-signals", action="store_true",
                  help="EXPLICIT opt-in: allow folder-name signals in gated detection. Default OFF.")
    e.add_argument("--formats", type=str, default="yaml", help='Comma-separated: yaml,json (default: yaml)')
    e.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    return p

def main():
    args = build_parser().parse_args()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    assets_path = args.assets.resolve()
    assets_doc = load_yaml(assets_path)
    out_dir = args.out_dir.resolve() if args.out_dir else assets_path.parent.resolve()

    fmts = {x.strip().lower() for x in str(args.formats).split(",") if x.strip()}
    if not (fmts & {"yaml","json"}):
        logger.error('Invalid --formats. Use "yaml", "json", or both.')
        sys.exit(2)

    doc = extract_headers(assets_doc, use_folder_name_signals=bool(args.use_folder_name_signals))

    written = []
    if "yaml" in fmts:
        written.append(write_yaml(out_dir / "HeaderIndex.yaml", doc))
    if "json" in fmts:
        written.append(write_json(out_dir / "HeaderIndex.json", doc))

    logger.info(f"[done] wrote {len(written)} file(s)")
    for p in written:
        print(f"  - {p}")

if __name__ == "__main__":
    main()
