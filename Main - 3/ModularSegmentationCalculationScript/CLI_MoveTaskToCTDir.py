#!/usr/bin/env python3
"""
CLI_MoveTaskToCTDir.py — Relocate an existing single-file task up to the CT
segmentation directory (sibling of total/ and trunk_cavities/) and re-register it
in PairIndex as a COMBINED single file.

Does NOT recompute anything. It moves the one .nii.gz the task currently points
at into  <CT-seg-dir>/<output-task>/<output-task>.nii.gz , updates the PairIndex
registration to seg_type: combined at the new path, and (optionally) removes the
now-empty old folder.

The CT segmentation directory is inferred as the directory that CONTAINS the
source modules (total/, trunk_cavities/) — i.e. the parent of whichever module
folder the task's file currently lives under. By default we anchor on a
reference module that is known to live directly in the CT seg dir.

Usage:

  python3 CLI_MoveTaskToCTDir.py \
    --pairs   "/Volumes/T7/Alavi Lab/CAMONAOG/Directory Plan/PairIndex.yaml" \
    --seg-root "/Volumes/T7/Alavi Lab/CAMONAOG/SegmentationsRoot" \
    --task "EAT_region" \
    --anchor-module "trunk_cavities" \
    --rename \
    --delete-old \
    -v

  # dry-run first to see the planned moves without touching anything
  python3 CLI_MoveTaskToCTDir.py ... --dry-run -v
"""

import argparse
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import yaml

from platform_paths import to_wsl


def load_yaml(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def save_yaml(doc, p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.dump(doc, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


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
                key = (parent_rel, pet, ct)
                if key in seen:
                    continue
                seen.add(key)
                yield parent_rel, entry


def resolve_seg_path(task_rec, seg_root_wsl):
    """Return (abs_path_wsl, seg_type). For individual returns the dir path."""
    seg_type = str(task_rec.get("seg_type") or "").strip().lower()
    if seg_type == "combined":
        rel = str((task_rec.get("combined_path") or {}).get("seg_path_rel") or "").strip()
        if rel:
            w = to_wsl(rel)
            return (w if w.startswith("/") else str((Path(seg_root_wsl) / rel).resolve())), "combined"
    if seg_type == "individual":
        rel = str((task_rec.get("organ_dir") or {}).get("dir_path_rel") or "").strip()
        if rel:
            w = to_wsl(rel)
            return (w if w.startswith("/") else str((Path(seg_root_wsl) / rel).resolve())), "individual"
    return None, None


def ct_seg_dir_from_anchor(entry, anchor_module, seg_root_wsl):
    """The CT seg dir = the directory that CONTAINS the anchor module.
    For an individual anchor (folder) at .../CTDIR/trunk_cavities  -> .../CTDIR
    For a combined anchor file  at .../CTDIR/total/total.nii.gz     -> .../CTDIR
    For a combined anchor file  at .../CTDIR/total.nii.gz           -> .../CTDIR
    """
    rec = (entry.get("segmentations") or {}).get(anchor_module)
    if not rec:
        return None
    path, seg_type = resolve_seg_path(rec, seg_root_wsl)
    if not path:
        return None
    p = Path(path)
    if seg_type == "individual":
        # path is the module folder itself -> parent is the CT seg dir
        return p.parent
    # combined file: if it sits in a subfolder named after the module, go up twice
    if p.parent.name == anchor_module:
        return p.parent.parent
    return p.parent


def single_file_in_task(task_rec, seg_root_wsl):
    """Return the single .nii.gz path this task points at, or None if not
    resolvable as exactly one file."""
    path, seg_type = resolve_seg_path(task_rec, seg_root_wsl)
    if not path:
        return None
    p = Path(path)
    if seg_type == "combined":
        return p if p.is_file() else None
    # individual: expect exactly one nii in the folder
    if p.is_dir():
        files = sorted([f for f in p.glob("*.nii*")])
        if len(files) == 1:
            return files[0]
        return None
    return None


def rel_to_segroot(path, seg_root_wsl):
    if seg_root_wsl:
        try:
            return str(Path(path).relative_to(Path(seg_root_wsl)))
        except Exception:
            pass
    return str(path)


def main():
    ap = argparse.ArgumentParser(description="Move a single-file task up to the CT seg dir and re-register as combined.")
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--seg-root", required=True)
    ap.add_argument("--task", required=True, help="Task to move (e.g. EAT_region)")
    ap.add_argument("--anchor-module", default="trunk_cavities",
                    help="A module that lives directly in the CT seg dir, used to locate it (default: trunk_cavities)")
    ap.add_argument("--rename", action="store_true",
                    help="Rename the moved file to <task>.nii.gz (recommended). Otherwise keep its filename.")
    ap.add_argument("--delete-old", action="store_true",
                    help="Remove the now-empty old task folder after moving.")
    ap.add_argument("--parents", default="")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite if a file already exists at the destination.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    pairs_wsl = to_wsl(args.pairs)
    seg_root_wsl = to_wsl(args.seg_root)
    pair_doc = load_yaml(pairs_wsl)
    parents_filter = set(t.strip() for t in args.parents.split(",") if t.strip()) if args.parents.strip() else None

    moved = 0
    skipped = 0

    for parent_rel, entry in iter_pairs(pair_doc, parents_filter):
        segs = entry.get("segmentations") or {}
        rec = segs.get(args.task)
        if not rec:
            continue

        src_file = single_file_in_task(rec, seg_root_wsl)
        if not src_file or not src_file.is_file():
            print(f"  SKIP {parent_rel}: could not resolve a single file for '{args.task}'")
            skipped += 1
            continue

        ctdir = ct_seg_dir_from_anchor(entry, args.anchor_module, seg_root_wsl)
        if not ctdir:
            print(f"  SKIP {parent_rel}: could not locate CT seg dir via anchor '{args.anchor_module}'")
            skipped += 1
            continue

        dest_dir = Path(ctdir) / args.task
        dest_name = f"{args.task}.nii.gz" if args.rename else src_file.name
        dest_file = dest_dir / dest_name

        print(f"\n[{parent_rel}]")
        print(f"  from: {src_file}")
        print(f"  to:   {dest_file}")

        if dest_file.exists() and not args.overwrite:
            print(f"  SKIP: destination exists (use --overwrite)")
            skipped += 1
            continue

        if args.dry_run:
            continue

        dest_dir.mkdir(parents=True, exist_ok=True)
        # move the file
        shutil.move(str(src_file), str(dest_file))

        # re-register as combined at the new location
        rel = rel_to_segroot(str(dest_file), seg_root_wsl).replace("\\", "/")
        new_rec = {
            "status": "done",
            "source": rec.get("source", "subtraction"),
            "registered_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "seg_type": "combined",
            "combined_path": {"seg_path_rel": rel},
            "moved_from": rec.get("organ_dir", {}).get("dir_path_rel")
                          or rec.get("combined_path", {}).get("seg_path_rel"),
        }
        # carry provenance if present
        for k in ("base", "subtracted", "union"):
            if k in rec:
                new_rec[k] = rec[k]
        segs[args.task] = new_rec

        # optionally delete the old now-empty folder
        if args.delete_old:
            old_parent = src_file.parent
            try:
                # remove the file's old folder if empty (or the whole task subtree)
                # src was inside .../trunk_cavities/Subtracted/<task>/file
                # remove <task> folder, and Subtracted if it becomes empty
                if old_parent.is_dir() and not any(old_parent.iterdir()):
                    old_parent.rmdir()
                    gp = old_parent.parent
                    if gp.name.lower() == "subtracted" and not any(gp.iterdir()):
                        gp.rmdir()
                    if args.verbose:
                        print(f"  removed old: {old_parent}")
            except Exception as e:
                print(f"  WARN: could not remove old folder: {e}")

        moved += 1
        print(f"  moved + re-registered '{args.task}' as combined")

    if not args.dry_run and moved:
        save_yaml(pair_doc, pairs_wsl)
        print(f"\nUpdated PairIndex: {pairs_wsl}  ({moved} moved, {skipped} skipped)")
    else:
        print(f"\nDone. moved={moved}, skipped={skipped}" + (" (dry-run)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
