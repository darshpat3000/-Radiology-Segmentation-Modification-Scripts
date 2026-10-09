#!/usr/bin/env python3
"""
platform_paths.py — Single source of truth for cross-platform path handling.

This module centralizes ALL Windows <-> WSL <-> native-Linux path logic that
used to be copy-pasted across CLI_SlicerMetrics.py, CLI_SlicerHUFilter.py,
CLI_RegisterSeg.py and CLI_TotalSegmentatorFromPairIndex.py.

To change how paths are converted anywhere in the pipeline, edit THIS file only.

------------------------------------------------------------------------------
Three environments are supported:
  1. Native Windows         (os.name == "nt")        -> Windows paths (F:\\...)
  2. WSL Ubuntu             (Linux, /mnt/<drive>/...) -> can talk to either
                                                          Windows Slicer or
                                                          Linux Slicer
  3. Native Ubuntu / Linux  (no /mnt drive letters)   -> Linux paths only

The two facts that matter:
  - Where THIS script runs (Windows vs Linux): os.name / is_wsl()
  - What the chosen Slicer expects (Windows .exe vs Linux binary):
    set via set_slicer_target() / slicer_is_windows_target()
------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import re
from pathlib import Path

__all__ = [
    # detection
    "is_wsl",
    "is_windows_abs", "is_wsl_mnt", "is_posix_abs",
    "_is_windows_abs", "_is_wsl_mnt",                  # legacy aliases (Slicer scripts)
    "_is_windows_abs_path", "_is_wsl_mnt_path", "_is_posix_abs_path",  # legacy (RegisterSeg/TotalSeg)
    # conversion
    "win_to_wsl", "wsl_to_win", "to_wsl", "to_win",
    "win_to_wsl_path", "wsl_to_win_path",             # legacy aliases (TotalSeg)
    "coerce_path", "coerce_any_path",
    # slicer targeting
    "SLICER_IS_WINDOWS", "set_slicer_target",
    "slicer_is_windows_target", "to_slicer_path",
]


# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------

def is_wsl() -> bool:
    """True if running inside WSL (Linux kernel reporting Microsoft/WSL)."""
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        with open("/proc/version", "r", encoding="utf-8", errors="ignore") as f:
            v = f.read().lower()
        return "microsoft" in v or "wsl" in v
    except Exception:
        return False


def is_windows_abs(s: str) -> bool:
    """True for a Windows absolute path like 'F:\\...' or 'C:/...'."""
    return bool(re.match(r"^[A-Za-z]:[\\/]", (s or "").strip()))


def is_wsl_mnt(s: str) -> bool:
    """True for a WSL mount path like '/mnt/f/...'."""
    return bool(re.match(r"^/mnt/[a-zA-Z]/", (s or "").strip()))


def is_posix_abs(s: str) -> bool:
    """True for any POSIX absolute path (starts with '/')."""
    return (s or "").strip().startswith("/")


# ---------------------------------------------------------------------------
# Core conversions (robust: also repair double-prefixed malformed paths)
# ---------------------------------------------------------------------------

def win_to_wsl(s: str) -> str:
    """Windows path -> WSL mount path. Pass through if not a Windows path.

    Also repairs malformed 'F:\\mnt\\f\\...' which can appear when a path is
    accidentally converted twice.
    """
    s = (s or "").strip()
    if not s:
        return s
    m_bad = re.match(r"^[A-Za-z]:[\\/]+mnt[\\/]+([a-zA-Z])[\\/]+(.*)$", s)
    if m_bad:
        drive = m_bad.group(1).lower()
        rest = m_bad.group(2).replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    if not is_windows_abs(s):
        return s
    drive = s[0].lower()
    rest = s[2:].lstrip("\\/").replace("\\", "/")
    return f"/mnt/{drive}/{rest}"


def wsl_to_win(s: str) -> str:
    """WSL mount path -> Windows path. Pass through if not a WSL mount path.

    Also repairs malformed '/mnt/f/mnt/f/...' double prefixes.
    """
    s = (s or "").strip()
    if not s:
        return s
    m_double = re.match(r"^/mnt/[a-zA-Z]/mnt/([a-zA-Z])/(.*)$", s)
    if m_double:
        d = m_double.group(1).upper()
        rest = m_double.group(2).replace("/", "\\")
        return f"{d}:\\{rest}"
    m = re.match(r"^/mnt/([a-zA-Z])/(.*)$", s)
    if not m:
        return s
    drive = m.group(1).upper()
    rest = m.group(2).replace("/", "\\")
    return f"{drive}:\\{rest}"


def to_wsl(s: str) -> str:
    """Ensure a path is in WSL/Linux format (convert only if it's Windows)."""
    if is_windows_abs(s):
        return win_to_wsl(s)
    return s


def to_win(s: str) -> str:
    """Ensure a path is in Windows format (convert only if it's a WSL mount)."""
    if is_wsl_mnt(s):
        return wsl_to_win(s)
    return s


def coerce_path(p: str) -> Path:
    """Normalize any path string to the format THIS machine's OS understands,
    returned as a pathlib.Path.

    - On native Windows: produce a Windows path (repairing 'F:\\mnt\\f' and
      converting bare POSIX-absolute strings via drive-letter assumption).
    - On Linux/WSL: produce a Linux path (converting Windows paths to /mnt/...).
    """
    s = (p or "").strip()
    if not s:
        return Path("")
    if os.name == "nt":
        m_bad = re.match(r"^[A-Za-z]:[\\/]+mnt[\\/]+([a-zA-Z])[\\/]+(.*)$", s)
        if m_bad:
            drive = m_bad.group(1).upper()
            rest = m_bad.group(2).replace("/", "\\")
            s = f"{drive}:\\{rest}"
        elif is_wsl_mnt(s):
            s = wsl_to_win(s)
    else:
        if is_windows_abs(s):
            s = win_to_wsl(s)
    return Path(s).expanduser()


def coerce_any_path(s: str) -> str:
    """Like coerce_path but returns a string and leaves relative paths alone.
    Matches the old CLI_SlicerHUFilter.coerce_any_path behavior.
    """
    s = (s or "").strip()
    if not s:
        return s
    if os.name == "nt":
        return wsl_to_win(s) if is_wsl_mnt(s) else s
    return win_to_wsl(s) if is_windows_abs(s) else s


# ---------------------------------------------------------------------------
# Slicer targeting
# ---------------------------------------------------------------------------
# A pipeline run can talk to either a Windows Slicer.exe (the WSL->Windows
# workflow) or a native Linux Slicer binary. Paths handed to Slicer must match
# whatever Slicer expects, independent of where this script runs.

SLICER_IS_WINDOWS = False


def slicer_is_windows_target(slicer_path: str) -> bool:
    """Decide whether the chosen Slicer expects Windows-style paths."""
    s = (slicer_path or "").strip()
    return s.lower().endswith(".exe") or is_windows_abs(s)


def set_slicer_target(slicer_path: str) -> bool:
    """Set the module-level SLICER_IS_WINDOWS flag from a Slicer path.
    Returns the resolved boolean. Call once after parsing --slicer.
    """
    global SLICER_IS_WINDOWS
    SLICER_IS_WINDOWS = slicer_is_windows_target(slicer_path)
    return SLICER_IS_WINDOWS


def to_slicer_path(s: str) -> str:
    """Convert a path into whatever format the chosen Slicer expects.

    Windows Slicer -> Windows path (F:\\...)
    Linux Slicer   -> Linux path, unchanged
    """
    if not s:
        return s
    if SLICER_IS_WINDOWS:
        return to_win(s)
    return to_wsl(s)


# ---------------------------------------------------------------------------
# Legacy name aliases so existing call sites keep working unchanged.
# ---------------------------------------------------------------------------

# Slicer-script naming
_is_windows_abs = is_windows_abs
_is_wsl_mnt = is_wsl_mnt

# RegisterSeg / TotalSeg naming
_is_windows_abs_path = is_windows_abs
_is_wsl_mnt_path = is_wsl_mnt
_is_posix_abs_path = is_posix_abs
win_to_wsl_path = win_to_wsl
wsl_to_win_path = wsl_to_win
