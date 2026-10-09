#!/usr/bin/env python3
"""
CLI_PipelineOrchestrator.py

Orchestrates the modular pipeline:

  1) CLI_AssetDiscovery.py    scan   -> AssetsIndex.yaml
  2) CLI_HeaderExtract.py     extract-> HeaderIndex.yaml
  3) CLI_PairFromHeaders.py   pair   -> PairIndex.yaml (+ SelectionTemplate.yaml optional)
  4) CLI_TotalSegmentatorFromPairIndex.py  run/plan -> updates PairIndex with segmentations
  5) CLI_SlicerMetrics.py          slicer  -> HU/SUV/Volume CSVs (via 3D Slicer)

Key design:
- Three distinct phases:
    A) pair       (repeatable until you're happy)
    B) segment    (runs TotalSegmentator AFTER pairing is settled)
    C) metrics    (computes HU/SUV/Volume from paired DICOM + segmentations)
- 'all' command chains pair -> segment -> metrics

This script does NOT change the core logic of your modular tools — it only calls them,
and prints stage summaries.

Assumptions:
- The component scripts live in the same folder as this orchestrator OR you pass --tools-dir.
- Output "Directory Plan" folder is usually: <dicom_root parent>/Directory Plan
- Segmentations output root is usually: <dicom_root parent>/SegmentationsRoot
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import yaml


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def eprint(*a):
    print(*a, file=sys.stderr)

def load_yaml(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def run_cmd(cmd: List[str], dry_run: bool = False) -> int:
    print("[run] $", " ".join(f'"{c}"' if " " in c else c for c in cmd))
    if dry_run:
        return 0
    env = {**os.environ, "NNUNET_N_PROC_DA": "0", "OMP_NUM_THREADS": "1"}

    proc = subprocess.Popen(cmd,env=env, start_new_session=True)
    try:
        proc.wait()
        return int(proc.returncode)

    # Force kill the entire process group when interrupted
    # Without this, the GPU memory is not reclaimed on Linux,
    # Linux does not reclaim memory like WSL does
    except KeyboardInterrupt:
        print("\n[run] Interrupted, killing TotalSegmentator process group...",
              file=sys.stderr)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            proc.kill()
        proc.wait()
        raise

def resolve_tools_dir(tools_dir: Optional[str]) -> Path:
    if tools_dir:
        return Path(tools_dir).expanduser().resolve()
    return Path(__file__).resolve().parent

def default_dir_plan(dicom_root: Path) -> Path:
    return (dicom_root.parent / "Directory Plan").resolve()

def default_seg_root(dicom_root: Path) -> Path:
    return (dicom_root.parent / "SegmentationsRoot").resolve()

def default_metrics_dir(dicom_root: Path) -> Path:
    return (dicom_root.parent / "Metrics").resolve()

def py() -> str:
    return sys.executable

def tool_path(tools_dir: Path, name: str) -> Path:
    return (tools_dir / name).resolve()

def assert_exists(p: Path, label: str):
    if not p.exists():
        raise SystemExit(f"{label} not found: {p}")

def write_overrides_stub(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    stub = {
        "selections": {},
        "note": "Add manual selections here. Example:\n"
                "selections:\n"
                "  HN001/20250101:\n"
                "    pet_asset_id: DICOM_SERIES:<StudyUID>:<SeriesUID>\n"
                "    ct_asset_id:  DICOM_SERIES:<StudyUID>:<SeriesUID>\n"
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(stub, f, sort_keys=False, allow_unicode=True)

def summarize_assets(assets_path: Path) -> str:
    try:
        doc = load_yaml(assets_path)
        st = doc.get("stats", {}) or {}
        total = st.get("total_assets", st.get("total", ""))
        by_kind = st.get("by_kind", {}) or {}
        return f"AssetsIndex: total_assets={total} by_kind={by_kind}"
    except Exception as e:
        return f"AssetsIndex: (failed to summarize) {e}"

def summarize_headers(headers_path: Path) -> str:
    try:
        doc = load_yaml(headers_path)
        st = doc.get("stats", {}) or {}
        dicom = st.get("dicom_series_extracted", "")
        by_mod = st.get("dicom_by_modality", {}) or {}
        failures = len(doc.get("failures", []) or [])
        return f"HeaderIndex: dicom_series_extracted={dicom} dicom_by_modality={by_mod} failures={failures}"
    except Exception as e:
        return f"HeaderIndex: (failed to summarize) {e}"

def summarize_pairs(pair_path: Path) -> Tuple[str, int]:
    """
    Returns (summary_str, needs_selection_count)
    """
    try:
        doc = load_yaml(pair_path)
        st = doc.get("stats", {}) or {}
        total = st.get("total_parents", "")
        paired = st.get("paired_parents", st.get("selected_single", ""))
        needs = st.get("needs_selection", "")
        if isinstance(needs, int):
            needs_n = needs
        else:
            needs_n = len(doc.get("needs_selection", []) or [])

        # Check for all_selected_pairs
        all_sp = doc.get("all_selected_pairs", {}) or {}
        all_pairs_count = sum(len(v) for v in all_sp.values() if isinstance(v, list))
        all_str = f" all_pairs_entries={all_pairs_count}" if all_pairs_count else ""

        return (f"PairIndex: total_parents={total} paired={paired} "
                f"needs_selection={needs} (count={needs_n}){all_str}"), int(needs_n)
    except Exception as e:
        return f"PairIndex: (failed to summarize) {e}", 0

def summarize_seg_update(pair_path: Path) -> str:
    """
    After running TS runner that updates PairIndex, summarize how many segmentation entries exist.
    """
    try:
        doc = load_yaml(pair_path)
        derived = (doc.get("derived", {}) or {}).get("segmentations_by_pair_id", {}) or {}
        n_pairs = len(derived) if isinstance(derived, dict) else 0
        task_set = set()
        recs = 0
        if isinstance(derived, dict):
            for _, tmap in derived.items():
                if isinstance(tmap, dict):
                    for t in tmap.keys():
                        task_set.add(t)
                        recs += 1

        # Also count segmentations in selected_pairs and all_selected_pairs
        sp_segs = 0
        for _, sel in (doc.get("selected_pairs", {}) or {}).items():
            if isinstance(sel, dict) and isinstance(sel.get("segmentations"), dict):
                sp_segs += len(sel["segmentations"])

        all_sp_segs = 0
        for _, entries in (doc.get("all_selected_pairs", {}) or {}).items():
            if isinstance(entries, list):
                for entry in entries:
                    if isinstance(entry, dict) and isinstance(entry.get("segmentations"), dict):
                        all_sp_segs += len(entry["segmentations"])

        return (f"Segmentations: derived_pair_ids={n_pairs} derived_task_records={recs} "
                f"distinct_tasks={sorted(task_set)} "
                f"selected_pairs_seg_entries={sp_segs} all_selected_pairs_seg_entries={all_sp_segs}")
    except Exception as e:
        return f"Segmentations recorded in PairIndex: (failed to summarize) {e}"


# -----------------------------------------------------------------------------
# Stage runners
# -----------------------------------------------------------------------------

def run_discovery(tools_dir: Path,
                  dicom_root: Path,
                  out_dir: Path,
                  seg_roots: List[Path],
                  nifti_roots: List[Path],
                  use_parent_foldernames: bool,
                  include_parent_name: List[str],
                  exclude_parent_name: List[str],
                  parent_name_mode: str,
                  verbose: bool,
                  dry_run: bool) -> Path:
    script = tool_path(tools_dir, "CLI_AssetDiscovery.py")
    assert_exists(script, "CLI_AssetDiscovery.py")
    ensure_dir(out_dir)

    cmd = [py(), str(script), "scan", "--dicom-root", str(dicom_root), "--out-dir", str(out_dir)]
    for r in seg_roots:
        cmd += ["--seg-root", str(r)]
    for r in nifti_roots:
        cmd += ["--nifti-root", str(r)]

    if use_parent_foldernames:
        cmd += ["--use-parent-foldernames"]
        for x in include_parent_name:
            cmd += ["--include-parent-name", x]
        for x in exclude_parent_name:
            cmd += ["--exclude-parent-name", x]
        if parent_name_mode:
            cmd += ["--parent-name-mode", parent_name_mode]

    if verbose:
        cmd += ["-v"]

    rc = run_cmd(cmd, dry_run=dry_run)
    if rc != 0:
        raise SystemExit(f"Asset discovery failed (rc={rc})")

    return out_dir / "AssetsIndex.yaml"

def run_header_extract(tools_dir: Path,
                       assets_path: Path,
                       out_dir: Path,
                       use_folder_name_signals: bool,
                       formats: str,
                       verbose: bool,
                       dry_run: bool) -> Path:
    script = tool_path(tools_dir, "CLI_HeaderExtract.py")
    assert_exists(script, "CLI_HeaderExtract.py")
    ensure_dir(out_dir)

    cmd = [py(), str(script), "extract", "--assets", str(assets_path), "--out-dir", str(out_dir), "--formats", formats]
    if use_folder_name_signals:
        cmd += ["--use-folder-name-signals"]
    if verbose:
        cmd += ["-v"]

    rc = run_cmd(cmd, dry_run=dry_run)
    if rc != 0:
        raise SystemExit(f"Header extraction failed (rc={rc})")

    return out_dir / "HeaderIndex.yaml"

def run_pair_from_headers(tools_dir: Path,
                          headers_path: Path,
                          out_dir: Path,
                          time_window: int,
                          allow_ct_reuse: bool,
                          allow_gated_selection: bool,
                          use_nac_only: bool,
                          allow_unknown_ac: bool,
                          all_pairs: bool,
                          prefer_wb: bool,
                          prefer_non_contrast: bool,
                          prefer_non_gated_ct: bool,
                          exclude_contrast_ct: bool,
                          overrides_path: Optional[Path],
                          write_selection_template: bool,
                          verbose: bool,
                          dry_run: bool) -> Path:
    script = tool_path(tools_dir, "CLI_PairFromHeaders.py")
    assert_exists(script, "CLI_PairFromHeaders.py")
    ensure_dir(out_dir)

    cmd = [
        py(), str(script), "pair",
        "--headers", str(headers_path),
        "--out-dir", str(out_dir),
        "--time-window", str(int(time_window)),
    ]
    if allow_ct_reuse:
        cmd += ["--allow-ct-reuse"]
    if allow_gated_selection:
        cmd += ["--allow-gated-selection"]
    if use_nac_only:
        cmd += ["--use-nac-only"]
    if allow_unknown_ac:
        cmd += ["--allow-unknown-ac"]
    if all_pairs:
        cmd += ["--all-pairs"]
    if not prefer_wb:
        cmd += ["--no-prefer-wb"]
    if not prefer_non_contrast:
        cmd += ["--no-prefer-non-contrast"]
    if not prefer_non_gated_ct:
        cmd += ["--no-prefer-non-gated-ct"]
    if exclude_contrast_ct:
        cmd += ["--exclude-contrast-ct"]
    if overrides_path:
        cmd += ["--overrides", str(overrides_path)]
    if write_selection_template:
        cmd += ["--write-selection-template"]
    if verbose:
        cmd += ["-v"]

    rc = run_cmd(cmd, dry_run=dry_run)
    if rc != 0:
        raise SystemExit(f"Pairing failed (rc={rc})")

    return out_dir / "PairIndex.yaml"

def run_segmentation(tools_dir: Path,
                     pair_path: Path,
                     headers_path: Path,
                     dicom_root: Optional[Path],
                     seg_out_root: Path,
                     tasks: str,
                     extra_cli: str,
                     ml: bool,
                     skip_existing: bool,
                     include_reasons: str,
                     use_all_candidate_pairs: bool,
                     include_label_map: bool,
                     write_index_only: bool,
                     no_backup: bool,
                     write_run_log: bool,
                     verbose: bool,
                     allowed_tracers:str,
                     dry_run: bool) -> None:
    script = tool_path(tools_dir, "CLI_TotalSegmentatorFromPairIndex.py")
    assert_exists(script, "CLI_TotalSegmentatorFromPairIndex.py")
    ensure_dir(seg_out_root)

    cmd = [
        py(), str(script), "run",
        "--pairs", str(pair_path),
        "--headers", str(headers_path),
        "--tasks", tasks,
        "--out-dir", str(seg_out_root),
        "--include-reasons", include_reasons,
    ]
    if dicom_root:
        cmd += ["--dicom-root", str(dicom_root)]
    if skip_existing:
        cmd += ["--skip-existing"]
    if ml:
        cmd += ["--ml"]
    if extra_cli.strip():
        cmd += [f"--extra-cli={extra_cli}"]
    if use_all_candidate_pairs:
        cmd += ["--use-all-candidate-pairs"]
    if include_label_map:
        cmd += ["--include-label-map"]
    if write_index_only:
        cmd += ["--write-index-only"]
    if no_backup:
        cmd += ["--no-backup"]
    if write_run_log:
        cmd += ["--write-run-log"]
    if allowed_tracers.strip():
        cmd += ["--include-tracers", allowed_tracers]

    rc = run_cmd(cmd, dry_run=dry_run)
    if rc != 0:
        raise SystemExit(f"Segmentation step failed (rc={rc})")

def run_slicer_hu_filter(tools_dir: Path,
                        pair_path: Path,
                        headers_path: Path,
                        seg_root: Optional[str],
                        source_task: str,
                        output_task: str,
                        hu_conditions: List[str],
                        parents: str,
                        out_subdir: str,
                        slicer_exe: str,
                        config_only: bool,
                        overwrite: bool,
                        verbose: bool,
                        dry_run: bool) -> None:
    script = tool_path(tools_dir, "CLI_SlicerHUFilter.py")
    assert_exists(script, "CLI_SlicerHUFilter.py")

    cmd = [
        py(), str(script),
        "--pairs", str(pair_path),
        "--headers", str(headers_path),
        "--source-task", source_task,
        "--output-task", output_task,
        "--slicer", slicer_exe,
        "--out-subdir", out_subdir,
    ]
    for hu in hu_conditions:
        cmd += ["--hu", hu]
    if seg_root:
        cmd += ["--seg-root", seg_root]
    if parents.strip():
        cmd += ["--parents", parents]
    if config_only:
        cmd += ["--config-only"]
    if overwrite:
        cmd += ["--overwrite"]
    if verbose:
        cmd += ["-v"]

    rc = run_cmd(cmd, dry_run=dry_run)
    if rc != 0:
        raise SystemExit(f"Slicer HU filter failed (rc={rc})")


def run_slicer_metrics(tools_dir: Path,
                      pair_path: Path,
                      headers_path: Path,
                      seg_root: Optional[str],
                      out_csv: Path,
                      tasks: str,
                      parents: str,
                      slicer_exe: str,
                      config_only: bool,
                      include_raw_pet: bool,
                      verbose: bool,
                      dry_run: bool) -> None:
    script = tool_path(tools_dir, "CLI_SlicerMetrics.py")
    assert_exists(script, "CLI_SlicerMetrics.py")
    ensure_dir(out_csv.parent)

    cmd = [
        py(), str(script),
        "--pairs", str(pair_path),
        "--headers", str(headers_path),
        "--tasks", tasks,
        "--out-csv", str(out_csv),
        "--slicer", slicer_exe,
    ]
    if seg_root:
        cmd += ["--seg-root", seg_root]
    if parents.strip():
        cmd += ["--parents", parents]
    if config_only:
        cmd += ["--config-only"]
    if include_raw_pet:
        cmd += ["--include-raw-pet"]
    if verbose:
        cmd += ["-v"]

    rc = run_cmd(cmd, dry_run=dry_run)
    if rc != 0:
        raise SystemExit(f"Slicer metrics failed (rc={rc})")


# -----------------------------------------------------------------------------
# Orchestrator commands
# -----------------------------------------------------------------------------

def _split_csv(s: str) -> List[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]

def _infer_dicom_root_from_headers(headers_path: Path) -> Optional[Path]:
    """Infer dicom_root from HeaderIndex if not explicitly provided."""
    try:
        hdr = load_yaml(headers_path)
        roots = {}
        for r in hdr.get("dicom_series", []) or []:
            rp = str(r.get("root_path") or "").strip()
            if rp:
                roots[rp] = roots.get(rp, 0) + 1
        if roots:
            return Path(sorted(roots.items(), key=lambda kv: kv[1], reverse=True)[0][0]).expanduser().resolve()
    except Exception:
        pass
    return None

def _infer_seg_root_from_pair(pair_path: Path) -> Optional[str]:
    """Infer seg_root from PairIndex derived.last_totalseg_run.out_root."""
    try:
        doc = load_yaml(pair_path)
        derived = doc.get("derived") or {}
        last_run = derived.get("last_totalseg_run") or {}
        out_root = str(last_run.get("out_root") or "").strip()
        if out_root:
            return out_root
    except Exception:
        pass
    return None


def cmd_pair(args) -> int:
    tools_dir = resolve_tools_dir(args.tools_dir)
    dicom_root = Path(args.dicom_root).expanduser().resolve()
    if not dicom_root.exists():
        raise SystemExit(f"DICOM root not found: {dicom_root}")

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else default_dir_plan(dicom_root)
    ensure_dir(out_dir)

    overrides_path = Path(args.overrides).expanduser().resolve() if args.overrides else (out_dir / "Overrides.yaml")
    if args.init_overrides:
        write_overrides_stub(overrides_path)
        print(f"[overrides] ensured: {overrides_path}")

    assets_path = out_dir / "AssetsIndex.yaml"
    headers_path = out_dir / "HeaderIndex.yaml"

    # Discovery
    if not args.skip_discovery:
        seg_roots = [Path(x).expanduser().resolve() for x in (args.seg_root or [])]
        nifti_roots = [Path(x).expanduser().resolve() for x in (args.nifti_root or [])]
        assets_path = run_discovery(
            tools_dir=tools_dir,
            dicom_root=dicom_root,
            out_dir=out_dir,
            seg_roots=seg_roots,
            nifti_roots=nifti_roots,
            use_parent_foldernames=bool(args.use_parent_foldernames),
            include_parent_name=_split_csv(args.include_parent_name) if args.include_parent_name else [],
            exclude_parent_name=_split_csv(args.exclude_parent_name) if args.exclude_parent_name else [],
            parent_name_mode=str(args.parent_name_mode or "substring"),
            verbose=bool(args.verbose),
            dry_run=bool(args.dry_run),
        )
        print("[summary]", summarize_assets(assets_path))
    else:
        print("[skip] discovery (using existing AssetsIndex.yaml)")
        assert_exists(assets_path, "AssetsIndex.yaml")

    # Header extract
    if not args.skip_header:
        headers_path = run_header_extract(
            tools_dir=tools_dir,
            assets_path=assets_path,
            out_dir=out_dir,
            use_folder_name_signals=bool(args.use_folder_name_signals),
            formats="yaml",
            verbose=bool(args.verbose),
            dry_run=bool(args.dry_run),
        )
        print("[summary]", summarize_headers(headers_path))
    else:
        print("[skip] header extraction (using existing HeaderIndex.yaml)")
        assert_exists(headers_path, "HeaderIndex.yaml")
        print("[summary]", summarize_headers(headers_path))

    # Pair
    if not args.skip_pair:
        pair_path = run_pair_from_headers(
            tools_dir=tools_dir,
            headers_path=headers_path,
            out_dir=out_dir,
            time_window=int(args.time_window),
            allow_ct_reuse=bool(args.allow_ct_reuse),
            allow_gated_selection=bool(args.allow_gated_selection),
            use_nac_only=bool(args.use_nac_only),
            allow_unknown_ac=bool(args.allow_unknown_ac),
            all_pairs=bool(args.all_pairs),
            prefer_wb=bool(args.prefer_wb),
            prefer_non_contrast=bool(args.prefer_non_contrast),
            prefer_non_gated_ct=bool(args.prefer_non_gated_ct),
            exclude_contrast_ct=bool(args.exclude_contrast_ct),
            overrides_path=(overrides_path if args.use_overrides else None),
            write_selection_template=bool(args.write_selection_template),
            verbose=bool(args.verbose),
            dry_run=bool(args.dry_run),
        )
        s, needs_n = summarize_pairs(pair_path)
        print("[summary]", s)
        if needs_n > 0:
            print("[note] needs_selection > 0. Review SelectionTemplate.yaml and/or edit Overrides.yaml, then re-run 'pair'.")
    else:
        print("[skip] pairing (not run)")

    print(f"[done] outputs in: {out_dir}")
    return 0

def cmd_segment(args) -> int:
    tools_dir = resolve_tools_dir(args.tools_dir)

    pair_path = Path(args.pairs).expanduser().resolve() if args.pairs else None
    headers_path = Path(args.headers).expanduser().resolve() if args.headers else None

    if not pair_path or not headers_path:
        if not args.dicom_root:
            raise SystemExit("Provide --pairs/--headers OR provide --dicom-root so defaults can be inferred.")
        dicom_root = Path(args.dicom_root).expanduser().resolve()
        out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else default_dir_plan(dicom_root)
        pair_path = pair_path or (out_dir / "PairIndex.yaml")
        headers_path = headers_path or (out_dir / "HeaderIndex.yaml")

    assert_exists(pair_path, "PairIndex.yaml")
    assert_exists(headers_path, "HeaderIndex.yaml")

    # Safety: by default refuse to segment if needs_selection > 0
    s, needs_n = summarize_pairs(pair_path)
    print("[summary]", s)
    if needs_n > 0 and not args.allow_needs_selection:
        raise SystemExit("Refusing to segment because PairIndex.needs_selection > 0. "
                         "Fix pairing (Overrides/SelectionTemplate) or pass --allow-needs-selection.")

    # Determine dicom_root and seg_out_root defaults
    dicom_root = Path(args.dicom_root).expanduser().resolve() if args.dicom_root else None
    if dicom_root is not None and not dicom_root.exists():
        raise SystemExit(f"DICOM root not found: {dicom_root}")

    if dicom_root is None:
        dicom_root = _infer_dicom_root_from_headers(headers_path)

    seg_out_root = (Path(args.seg_out_root).expanduser().resolve() if args.seg_out_root
                    else (default_seg_root(dicom_root) if dicom_root
                          else (Path.cwd().resolve() / "SegmentationsRoot")))
    ensure_dir(seg_out_root)

    run_segmentation(
        tools_dir=tools_dir,
        pair_path=pair_path,
        headers_path=headers_path,
        dicom_root=dicom_root,
        seg_out_root=seg_out_root,
        tasks=str(args.tasks),
        extra_cli=str(args.extra_cli or ""),
        ml=bool(args.ml),
        skip_existing=bool(args.skip_existing),
        include_reasons=str(args.include_reasons),
        use_all_candidate_pairs=bool(args.use_all_candidate_pairs),
        include_label_map=bool(args.include_label_map),
        write_index_only=bool(args.write_index_only),
        no_backup=bool(args.no_backup),
        write_run_log=bool(args.write_run_log),
        verbose=bool(args.verbose),
        dry_run=bool(args.dry_run),
        allowed_tracers=str(args.allowed_tracers or ""),
    )

    print("[summary]", summarize_seg_update(pair_path))
    print(f"[done] segmentations root: {seg_out_root}")
    return 0

def cmd_all(args) -> int:
    rc = cmd_pair(args)
    if rc != 0:
        return rc

    # Build a minimal args-like object for segment
    class S: pass
    s = S()
    s.tools_dir = args.tools_dir
    s.pairs = None
    s.headers = None
    s.dicom_root = args.dicom_root
    s.out_dir = args.out_dir
    s.seg_out_root = args.seg_out_root
    s.tasks = args.tasks
    s.extra_cli = args.extra_cli
    s.ml = args.ml
    s.skip_existing = args.skip_existing
    s.include_reasons = args.include_reasons
    s.use_all_candidate_pairs = args.use_all_candidate_pairs
    s.include_label_map = args.include_label_map
    s.write_index_only = args.write_index_only
    s.no_backup = args.no_backup
    s.write_run_log = args.write_run_log
    s.verbose = args.verbose
    s.dry_run = args.dry_run
    s.allow_needs_selection = args.allow_needs_selection
    s.allowed_tracers = args.allowed_tracers

    rc = cmd_segment(s)
    if rc != 0:
        return rc

    return 0


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _add_common(x: argparse.ArgumentParser):
    x.add_argument("--tools-dir", type=str, default=None,
                   help="Folder containing the modular scripts (defaults to this script's folder).")
    x.add_argument("--out-dir", type=str, default=None,
                   help='Directory Plan folder (default: "<dicom_root parent>/Directory Plan").')
    x.add_argument("--verbose", "-v", action="store_true", help="Verbose (passes -v to subtools where supported).")
    x.add_argument("--dry-run", action="store_true", help="Print commands only; do not execute.")

def _add_pairing_args(x: argparse.ArgumentParser):
    x.add_argument("--dicom-root", type=str, required=True, help="Root directory containing exported DICOM studies")
    x.add_argument("--seg-root", action="append", default=[], help="Optional seg root(s) for AssetsIndex (repeatable).")
    x.add_argument("--nifti-root", action="append", default=[], help="Optional nifti root(s) for AssetsIndex (repeatable).")

    x.add_argument("--use-parent-foldernames", action="store_true",
                   help="EXPLICIT opt-in: enable parent folder name filters in AssetDiscovery.")
    x.add_argument("--include-parent-name", type=str, default="",
                   help="Comma list tokens for include parent-name filtering (only with --use-parent-foldernames).")
    x.add_argument("--exclude-parent-name", type=str, default="",
                   help="Comma list tokens for exclude parent-name filtering (only with --use-parent-foldernames).")
    x.add_argument("--parent-name-mode", choices=["substring", "regex"], default="substring",
                   help="Interpretation of include/exclude tokens (only with --use-parent-foldernames).")

    x.add_argument("--use-folder-name-signals", action="store_true",
                   help="EXPLICIT opt-in: allow folder-name gated hints during header extract (default OFF).")

    x.add_argument("--time-window", type=int, default=15, help="PET↔CT time window (minutes).")
    x.add_argument("--allow-ct-reuse", action="store_true", help="Allow reusing CT for multiple PET in a parent.")
    x.add_argument("--allow-gated-selection", action="store_true", help="Allow gated selection (default prefers non-gated).")
    x.add_argument("--use-nac-only", action="store_true", help="Keep only NAC PET pairs.")
    x.add_argument("--allow-unknown-ac", action="store_true", help="Allow PET with unknown AC if no AC present.")
    x.add_argument("--all-pairs", action="store_true",
                   help="Generate all valid PET↔CT pairs per parent (multi-timepoint). Writes to all_selected_pairs.")
    x.add_argument("--no-prefer-wb", dest="prefer_wb", action="store_false",
                   help="Do NOT prefer whole-body CT over limited FOV CT (default: prefer WB).")
    x.set_defaults(prefer_wb=True)
    x.add_argument("--no-prefer-non-contrast", dest="prefer_non_contrast", action="store_false",
                   help="Do NOT prefer non-contrast CT (default: prefer non-contrast).")
    x.set_defaults(prefer_non_contrast=True)
    x.add_argument("--no-prefer-non-gated-ct", dest="prefer_non_gated_ct", action="store_false",
                   help="Do NOT prefer non-gated CT (default: prefer non-gated).")
    x.set_defaults(prefer_non_gated_ct=True)
    x.add_argument("--exclude-contrast-ct", action="store_true",
                   help="Completely exclude contrast-enhanced CTs from pairing.")
    x.add_argument("--write-selection-template", action="store_true", help="Write SelectionTemplate.yaml when needed.")
    x.add_argument("--init-overrides", action="store_true", help="Create empty Overrides.yaml in out-dir if missing.")
    x.add_argument("--overrides", type=str, default=None, help="Overrides.yaml path (default <out-dir>/Overrides.yaml).")
    x.add_argument("--use-overrides", action="store_true", help="Actually pass --overrides into PairFromHeaders.")
    x.add_argument("--skip-discovery", action="store_true", help="Skip discovery; reuse existing AssetsIndex.yaml.")
    x.add_argument("--skip-header", action="store_true", help="Skip header extract; reuse existing HeaderIndex.yaml.")
    x.add_argument("--skip-pair", action="store_true", help="Skip pairing.")

def _add_segment_args(x: argparse.ArgumentParser):
    x.add_argument("--seg-out-root", type=str, default=None,
                   help='Segmentations output root (default: "<dicom_root parent>/SegmentationsRoot").')
    x.add_argument("--tasks", type=str, default="total", help='Comma list of TS tasks (e.g. "total,lung_vessels").')
    x.add_argument("--extra-cli", type=str, default="", help='Extra TotalSegmentator args (e.g. "-nr 1 -ns 1").')
    x.add_argument("--ml", action="store_true",
                   help="Use multilabel output mode (single combined NIfTI per task).")
    x.add_argument("--skip-existing", action="store_true", help="Skip existing non-empty output folders.")
    x.add_argument("--include-reasons", type=str, default="auto_selected,manual_override",
                   help="Which selected_pairs[*].reason values to include (selected mode).")
    x.add_argument("--use-all-candidate-pairs", action="store_true", help="Use candidate pairs (multi-pair).")
    x.add_argument("--include-label-map", action="store_true", help="Include label maps when available.")
    x.add_argument("--write-index-only", action="store_true", help="Index existing outputs only; do not run TS.")
    x.add_argument("--no-backup", action="store_true", help="Do not backup PairIndex before updating.")
    x.add_argument("--write-run-log", action="store_true", help="Write SegmentationRunsIndex.yaml.")
    x.add_argument("--allow-needs-selection", action="store_true",
                   help="Allow segmentation even if PairIndex.needs_selection > 0 (default refuses).")

def _add_slicer_hu_filter_args(x: argparse.ArgumentParser):
    x.add_argument("--source-task", required=True, help="Source segmentation task to filter")
    x.add_argument("--output-task", required=True, help="Task name to register filtered output as")
    x.add_argument("--hu", action="append", required=True,
                   help="HU condition (repeatable): [-200,0], >=130, etc.")
    x.add_argument("--slicer-seg-root", type=str, default=None, help="Seg root for relative paths")
    x.add_argument("--parents", type=str, default="", help="Comma list of parent_rel to include")
    x.add_argument("--out-subdir", default="HUFiltered", help="Subfolder for outputs")
    x.add_argument("--slicer-exe", type=str,
                   default=os.environ.get("SLICER_PATH",
                            r"C:\Users\kirim\AppData\Local\slicer.org\Slicer 5.8.1\Slicer.exe"),
                   help="Path to Slicer executable (Windows .exe or Linux binary; or set SLICER_PATH).")
    x.add_argument("--config-only", action="store_true", help="Write config only, don't call Slicer")
    x.add_argument("--overwrite", action="store_true", help="Overwrite existing output-task registration")

def _add_slicer_metrics_args(x: argparse.ArgumentParser):
    x.add_argument("--slicer-tasks", type=str, required=True,
                   help='Comma list of task names to compute metrics for.')
    x.add_argument("--slicer-seg-root", type=str, default=None,
                   help="Seg root for relative paths (auto from PairIndex if not provided).")
    x.add_argument("--slicer-out-csv", type=str, default=None,
                   help='Output CSV path (default: "<parent>/Metrics/slicer_metrics.csv").')
    x.add_argument("--slicer-exe", type=str,
                   default=os.environ.get("SLICER_PATH",
                            r"C:\Users\kirim\AppData\Local\slicer.org\Slicer 5.8.1\Slicer.exe"),
                   help="Path to Slicer executable (Windows .exe or Linux binary; or set SLICER_PATH).")
    x.add_argument("--parents", type=str, default="",
                   help="Comma list of parent_rel to include.")
    x.add_argument("--config-only", action="store_true",
                   help="Write config.json only, don't call Slicer.")
    x.add_argument("--include-raw-pet", action="store_true",
                   help="Include raw PET value columns alongside SUV columns.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Orchestrate discovery -> header extract -> pairing -> segmentation -> slicer metrics (modular pipeline).")
    sub = p.add_subparsers(dest="cmd", required=True)

    # PAIR
    pr = sub.add_parser("pair", help="Run discovery + header extraction + pairing (repeatable).")
    _add_common(pr)
    _add_pairing_args(pr)

    # SEGMENT
    sg = sub.add_parser("segment", help="Run segmentation using existing PairIndex/HeaderIndex (does NOT re-pair).")
    _add_common(sg)
    sg.add_argument("--pairs", type=str, default=None, help="PairIndex.yaml (optional if --dicom-root given).")
    sg.add_argument("--headers", type=str, default=None, help="HeaderIndex.yaml (optional if --dicom-root given).")
    sg.add_argument("--dicom-root", type=str, default=None, help="Optional DICOM root for defaults inference.")
    sg.add_argument("--allowed-tracers", type=str, default="", help="Comma list of allowed tracers.")
    _add_segment_args(sg)

    # ALL
    al = sub.add_parser("all", help="Convenience: run pair -> segment.")
    _add_common(al)
    _add_pairing_args(al)
    _add_segment_args(al)
    al.add_argument("--allowed-tracers", type=str, default="", help="Comma list of allowed tracers.")

    # SLICER-HU-FILTER
    hf = sub.add_parser("slicer-hu-filter", help="HU-filter segmentations using 3D Slicer.")
    _add_common(hf)
    hf.add_argument("--pairs", type=str, default=None, help="PairIndex.yaml")
    hf.add_argument("--headers", type=str, default=None, help="HeaderIndex.yaml")
    hf.add_argument("--dicom-root", type=str, default=None, help="Optional DICOM root for defaults inference")
    _add_slicer_hu_filter_args(hf)

    # SLICER-METRICS
    sm = sub.add_parser("slicer-metrics", help="Compute HU/SUV/Volume using 3D Slicer (handles all coordinate transforms).")
    _add_common(sm)
    sm.add_argument("--pairs", type=str, default=None, help="PairIndex.yaml (optional if --dicom-root given).")
    sm.add_argument("--headers", type=str, default=None, help="HeaderIndex.yaml (optional if --dicom-root given).")
    sm.add_argument("--dicom-root", type=str, default=None, help="Optional DICOM root for defaults inference.")
    _add_slicer_metrics_args(sm)

    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.cmd == "pair":
        return cmd_pair(args)
    if args.cmd == "segment":
        return cmd_segment(args)
    if args.cmd == "slicer-hu-filter":
        return cmd_slicer_hu_filter(args)
    if args.cmd == "slicer-metrics":
        return cmd_slicer_metrics(args)
    if args.cmd == "all":
        return cmd_all(args)
    raise SystemExit("Unknown command")


def cmd_slicer_hu_filter(args) -> int:
    tools_dir = resolve_tools_dir(args.tools_dir)

    pair_path = Path(args.pairs).expanduser().resolve() if args.pairs else None
    headers_path = Path(args.headers).expanduser().resolve() if args.headers else None

    if not pair_path or not headers_path:
        if not args.dicom_root:
            raise SystemExit("Provide --pairs/--headers OR --dicom-root.")
        dicom_root = Path(args.dicom_root).expanduser().resolve()
        out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else default_dir_plan(dicom_root)
        pair_path = pair_path or (out_dir / "PairIndex.yaml")
        headers_path = headers_path or (out_dir / "HeaderIndex.yaml")

    assert_exists(pair_path, "PairIndex.yaml")
    assert_exists(headers_path, "HeaderIndex.yaml")

    seg_root = args.slicer_seg_root if args.slicer_seg_root else _infer_seg_root_from_pair(pair_path)
    if not seg_root:
        dicom_root = Path(args.dicom_root).expanduser().resolve() if args.dicom_root else _infer_dicom_root_from_headers(headers_path)
        if dicom_root:
            seg_root = str(default_seg_root(dicom_root))

    run_slicer_hu_filter(
        tools_dir=tools_dir,
        pair_path=pair_path,
        headers_path=headers_path,
        seg_root=seg_root,
        source_task=args.source_task,
        output_task=args.output_task,
        hu_conditions=args.hu,
        parents=str(args.parents or ""),
        out_subdir=str(args.out_subdir or "HUFiltered"),
        slicer_exe=str(args.slicer_exe),
        config_only=bool(args.config_only),
        overwrite=bool(args.overwrite),
        verbose=bool(args.verbose),
        dry_run=bool(args.dry_run),
    )
    return 0


def cmd_slicer_metrics(args) -> int:
    tools_dir = resolve_tools_dir(args.tools_dir)

    pair_path = Path(args.pairs).expanduser().resolve() if args.pairs else None
    headers_path = Path(args.headers).expanduser().resolve() if args.headers else None

    if not pair_path or not headers_path:
        if not args.dicom_root:
            raise SystemExit("Provide --pairs/--headers OR provide --dicom-root so defaults can be inferred.")
        dicom_root = Path(args.dicom_root).expanduser().resolve()
        out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else default_dir_plan(dicom_root)
        pair_path = pair_path or (out_dir / "PairIndex.yaml")
        headers_path = headers_path or (out_dir / "HeaderIndex.yaml")

    assert_exists(pair_path, "PairIndex.yaml")
    assert_exists(headers_path, "HeaderIndex.yaml")

    # Resolve seg root
    seg_root = args.slicer_seg_root if args.slicer_seg_root else _infer_seg_root_from_pair(pair_path)
    if not seg_root:
        dicom_root = Path(args.dicom_root).expanduser().resolve() if args.dicom_root else _infer_dicom_root_from_headers(headers_path)
        if dicom_root:
            seg_root = str(default_seg_root(dicom_root))
    if seg_root:
        print(f"[seg-root] {seg_root}")

    # Resolve output CSV
    if args.slicer_out_csv:
        out_csv = Path(args.slicer_out_csv).expanduser().resolve()
    elif args.dicom_root:
        out_csv = default_metrics_dir(Path(args.dicom_root).expanduser().resolve()) / "slicer_metrics.csv"
    elif args.out_dir:
        out_csv = Path(args.out_dir).expanduser().resolve() / "Metrics" / "slicer_metrics.csv"
    else:
        out_csv = pair_path.parent / "Metrics" / "slicer_metrics.csv"

    run_slicer_metrics(
        tools_dir=tools_dir,
        pair_path=pair_path,
        headers_path=headers_path,
        seg_root=seg_root,
        out_csv=out_csv,
        tasks=str(args.slicer_tasks),
        parents=str(args.parents or ""),
        slicer_exe=str(args.slicer_exe),
        config_only=bool(args.config_only),
        include_raw_pet=bool(args.include_raw_pet),
        verbose=bool(args.verbose),
        dry_run=bool(args.dry_run),
    )

    print(f"[done] slicer metrics output: {out_csv}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
