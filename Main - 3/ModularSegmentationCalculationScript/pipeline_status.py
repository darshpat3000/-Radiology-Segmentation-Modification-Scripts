#!/usr/bin/env python3
"""
pipeline_status.py — Diagnostic summary of PairIndex + HeaderIndex.

Shows tallies for: parents, pairs, unpaired, segmentations, tasks,
failed, empty, and missing items.

Usage:
  python3 pipeline_status.py --dir "/mnt/f/HNC/Directory Plan"
  python3 pipeline_status.py --dir "/mnt/f/HNC/Directory Plan" --task "head_glands_cavities" --check-files
"""

import argparse
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml


def load_yaml(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def coerce_path(s):
    import re
    s = (s or "").strip()
    if not s:
        return s
    if os.name != "nt" and re.match(r"^[A-Za-z]:[\\/]", s):
        drive = s[0].lower()
        rest = s[2:].lstrip("\\/").replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return s


def series_abs_dir(series_row):
    if not series_row:
        return None
    root = coerce_path(str(series_row.get("root_path") or ""))
    rel = str(series_row.get("series_rel") or "").strip()
    if not root or not rel:
        return None
    return str((Path(root) / Path(rel)).resolve())


def resolve_seg_path(task_rec, seg_root):
    seg_type = str(task_rec.get("seg_type") or "").strip().lower()
    if seg_type == "combined":
        cp = task_rec.get("combined_path") or {}
        rel = str(cp.get("seg_path_rel") or "").strip()
        if rel:
            p = coerce_path(rel)
            if p.startswith("/"):
                return p
            if seg_root:
                return str((Path(seg_root) / rel).resolve())
    out_dir = str(task_rec.get("out_dir") or "").strip()
    if out_dir:
        return coerce_path(out_dir)
    return None


def collect_pairs(pair_doc):
    pairs = []
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
                dedup = (parent_rel, pet, ct)
                is_dup = dedup in seen
                seen.add(dedup)
                segs = entry.get("segmentations") or {}
                pairs.append({
                    "parent_rel": parent_rel,
                    "pet_asset_id": pet,
                    "ct_asset_id": ct,
                    "source": src_key,
                    "reason": entry.get("reason", ""),
                    "seg_tasks": segs if isinstance(segs, dict) else {},
                    "is_duplicate": is_dup,
                })
    return pairs


def collect_all_parents_from_headers(headers_doc):
    """Get all unique parent_rel values from HeaderIndex."""
    parents = set()
    for r in headers_doc.get("dicom_series", []) or []:
        rel = str(r.get("series_rel") or "").strip().replace("\\", "/")
        if not rel:
            continue
        parts = rel.split("/")
        if len(parts) >= 2:
            parent = "/".join(parts[:-1])
        else:
            parent = "."
        parents.add(parent)
    return parents


def collect_needs_selection(pair_doc):
    ns = pair_doc.get("needs_selection")
    if isinstance(ns, dict):
        return list(ns.keys())
    if isinstance(ns, list):
        return [str(x) for x in ns]
    return []


def collect_unpaired(pair_doc):
    unpaired = []
    for key in ("unpaired", "unmatched", "no_match", "skipped_parents"):
        val = pair_doc.get(key)
        if isinstance(val, dict):
            unpaired.extend(val.keys())
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, str):
                    unpaired.append(item)
                elif isinstance(item, dict):
                    pr = item.get("parent_rel") or item.get("parent") or ""
                    if pr:
                        unpaired.append(str(pr))
    return unpaired


def main():
    p = argparse.ArgumentParser(description="Diagnostic summary of PairIndex contents.")
    p.add_argument("--dir", type=str, default=None, help="Directory Plan folder")
    p.add_argument("--pairs", type=str, default=None, help="PairIndex.yaml")
    p.add_argument("--headers", type=str, default=None, help="HeaderIndex.yaml")
    p.add_argument("--task", type=str, default=None, help="Focus on a specific task")
    p.add_argument("--check-files", action="store_true", help="Check if CT/PET/seg files exist on disk")
    p.add_argument("--show-missing", action="store_true", help="List parents missing the --task segmentation")
    p.add_argument("--show-duplicates", action="store_true", help="List duplicate pairs")
    args = p.parse_args()

    if args.dir:
        d = Path(args.dir).expanduser().resolve()
        pairs_path = d / "PairIndex.yaml"
        headers_path = d / "HeaderIndex.yaml"
    elif args.pairs:
        pairs_path = Path(args.pairs).expanduser().resolve()
        headers_path = Path(args.headers).expanduser().resolve() if args.headers else None
    else:
        p.error("Provide --dir or --pairs")
        return

    pair_doc = load_yaml(pairs_path)

    by_asset = {}
    headers_doc = {}
    if headers_path and headers_path.exists():
        headers_doc = load_yaml(headers_path)
        for r in headers_doc.get("dicom_series", []) or []:
            aid = (r.get("asset_id") or "").strip()
            if aid:
                by_asset[aid] = r

    seg_root = None
    derived = pair_doc.get("derived") or {}
    out_root = str((derived.get("last_totalseg_run") or {}).get("out_root") or "")
    if out_root:
        seg_root = coerce_path(out_root)

    pairs = collect_pairs(pair_doc)

    # =========================================================================
    print("=" * 70)
    print("PIPELINE STATUS")
    print("=" * 70)

    # --- Sources ---
    sp_count = len(pair_doc.get("selected_pairs") or {})
    asp_entries = pair_doc.get("all_selected_pairs") or {}
    asp_count = sum(len(v) if isinstance(v, list) else 1 for v in asp_entries.values())
    print(f"\nPairIndex sources:")
    print(f"  selected_pairs:     {sp_count} parents")
    print(f"  all_selected_pairs: {len(asp_entries)} parents, {asp_count} entries")

    # --- Needs selection / unpaired ---
    needs_sel_list = collect_needs_selection(pair_doc)
    unpaired_list = collect_unpaired(pair_doc)
    needs_sel_count = pair_doc.get("needs_selection")
    if isinstance(needs_sel_count, int):
        print(f"  needs_selection:    {needs_sel_count} (count only)")
    elif needs_sel_list:
        print(f"  needs_selection:    {len(needs_sel_list)} parents")
    if unpaired_list:
        print(f"  unpaired/skipped:   {len(unpaired_list)} parents")

    # --- All parents from HeaderIndex ---
    not_in_pairs = []
    if headers_doc:
        all_header_parents = collect_all_parents_from_headers(headers_doc)
        paired_parents = set(p["parent_rel"] for p in pairs)
        not_in_pairs = sorted(all_header_parents - paired_parents)
        print(f"\n  Total parents in HeaderIndex: {len(all_header_parents)}")
        print(f"  Parents in PairIndex:         {len(paired_parents)}")
        print(f"  Parents NOT paired:           {len(not_in_pairs)}")
        if not_in_pairs:
            for pr in not_in_pairs[:30]:
                print(f"    {pr}")
            if len(not_in_pairs) > 30:
                print(f"    ... and {len(not_in_pairs) - 30} more")

    # --- Unique pairs ---
    unique = [p for p in pairs if not p["is_duplicate"]]
    dups = [p for p in pairs if p["is_duplicate"]]
    print(f"\nPairs (after dedup):")
    print(f"  Total unique pairs: {len(unique)}")
    print(f"  Duplicates:         {len(dups)}")

    parents_list = sorted(set(p["parent_rel"] for p in unique))
    print(f"  Unique parents:     {len(parents_list)}")

    parent_counts = Counter(p["parent_rel"] for p in unique)
    multi = {k: v for k, v in parent_counts.items() if v > 1}
    single = {k: v for k, v in parent_counts.items() if v == 1}
    print(f"  Single-pair parents: {len(single)}")
    print(f"  Multi-pair parents:  {len(multi)}")
    if multi:
        top = sorted(multi.items(), key=lambda x: -x[1])[:10]
        for pr, cnt in top:
            print(f"    {pr}: {cnt} pairs")

    # --- Reasons ---
    reasons = Counter(p["reason"] for p in unique if p["reason"])
    if reasons:
        print(f"\nPair reasons:")
        for r, c in reasons.most_common():
            print(f"  {r}: {c}")

    # --- Segmentation tasks ---
    task_counts = Counter()
    for p in unique:
        tasks = list(p["seg_tasks"].keys())
        for t in tasks:
            task_counts[t] += 1
        if not tasks:
            task_counts["(no segmentations)"] += 1

    print(f"\nSegmentation tasks:")
    for t, c in task_counts.most_common():
        print(f"  {t}: {c} pairs")

    # =========================================================================
    # TASK DETAIL
    # =========================================================================
    focus_task = args.task
    if focus_task:
        print(f"\n{'=' * 70}")
        print(f"TASK DETAIL: {focus_task}")
        print(f"{'=' * 70}")

        has_task = [p for p in unique if focus_task in p["seg_tasks"]]
        no_task = [p for p in unique if focus_task not in p["seg_tasks"]]
        print(f"  Pairs WITH {focus_task}: {len(has_task)}")
        print(f"  Pairs WITHOUT {focus_task}: {len(no_task)}")

        # Status breakdown
        statuses = Counter()
        all_issues = []

        for p in has_task:
            rec = p["seg_tasks"][focus_task]
            st = str(rec.get("status") or "unknown").strip()
            statuses[st] += 1
            if st in ("failed", "empty", "error"):
                ct_row = by_asset.get(p["ct_asset_id"])
                ct_dir = series_abs_dir(ct_row) if ct_row else None
                seg_path = resolve_seg_path(rec, seg_root)
                seg_exists = os.path.isfile(seg_path) if seg_path else False
                seg_size = os.path.getsize(seg_path) if seg_exists else 0
                all_issues.append({
                    "parent_rel": p["parent_rel"],
                    "category": f"seg_{st}",
                    "ct_asset_id": p["ct_asset_id"],
                    "pet_asset_id": p["pet_asset_id"],
                    "ct_dir": ct_dir,
                    "ct_exists": os.path.isdir(ct_dir) if ct_dir else False,
                    "seg_path": seg_path,
                    "seg_exists": seg_exists,
                    "seg_size_bytes": seg_size,
                    "error": str(rec.get("error") or rec.get("message") or ""),
                })

        # Add non-segmented pairs
        for p in no_task:
            ct_row = by_asset.get(p["ct_asset_id"])
            ct_dir = series_abs_dir(ct_row) if ct_row else None
            all_issues.append({
                "parent_rel": p["parent_rel"],
                "category": "not_segmented",
                "ct_asset_id": p["ct_asset_id"],
                "pet_asset_id": p["pet_asset_id"],
                "ct_dir": ct_dir,
                "ct_exists": os.path.isdir(ct_dir) if ct_dir else False,
                "seg_path": None,
                "seg_exists": False,
                "seg_size_bytes": 0,
                "error": "",
            })

        if statuses:
            print(f"\n  Status breakdown:")
            for s, c in statuses.most_common():
                print(f"    {s}: {c}")
            print(f"    (not segmented): {len(no_task)}")

        # Check "done" segs for empty-on-disk
        if args.check_files:
            print(f"\n  Checking 'done' seg files on disk...")
            found = 0
            for p in has_task:
                rec = p["seg_tasks"][focus_task]
                st = str(rec.get("status") or "").strip().lower()
                if st != "done":
                    continue
                sp = resolve_seg_path(rec, seg_root)
                if sp and os.path.exists(sp):
                    sz = os.path.getsize(sp)
                    if sz < 500:
                        all_issues.append({
                            "parent_rel": p["parent_rel"],
                            "category": "done_but_empty",
                            "ct_asset_id": p["ct_asset_id"],
                            "pet_asset_id": p["pet_asset_id"],
                            "ct_dir": series_abs_dir(by_asset.get(p["ct_asset_id"])),
                            "ct_exists": True,
                            "seg_path": sp,
                            "seg_exists": True,
                            "seg_size_bytes": sz,
                            "error": "file exists but < 500 bytes",
                        })
                    else:
                        found += 1
                elif sp:
                    all_issues.append({
                        "parent_rel": p["parent_rel"],
                        "category": "done_but_missing",
                        "ct_asset_id": p["ct_asset_id"],
                        "pet_asset_id": p["pet_asset_id"],
                        "ct_dir": series_abs_dir(by_asset.get(p["ct_asset_id"])),
                        "ct_exists": True,
                        "seg_path": sp,
                        "seg_exists": False,
                        "seg_size_bytes": 0,
                        "error": "marked done but file missing",
                    })
            print(f"    Done + valid on disk: {found}")

        # Category tally
        cat_counts = Counter(x["category"] for x in all_issues)
        print(f"\n  Issue tally:")
        for cat, cnt in cat_counts.most_common():
            print(f"    {cat}: {cnt}")
        print(f"    TOTAL issues: {len(all_issues)}")

        # Print details to console (abbreviated)
        if all_issues:
            print(f"\n  All issues ({len(all_issues)}):")
            for fd in sorted(all_issues, key=lambda x: (x["category"], x["parent_rel"])):
                ct_short = fd['ct_asset_id'].split(':')[-1] if ':' in fd['ct_asset_id'] else fd['ct_asset_id']
                line = f"    [{fd['category']}] {fd['parent_rel']}  CT={ct_short}"
                if fd['seg_exists']:
                    line += f"  seg_size={fd['seg_size_bytes']}"
                if fd['error']:
                    line += f"  err={fd['error'][:80]}"
                print(line)

        # ---- Write comprehensive file ----
        report_out = pairs_path.parent / f"issues_{focus_task}.txt"
        with open(report_out, "w") as f:
            f.write(f"Pipeline issues for task: {focus_task}\n")
            f.write(f"Generated from: {pairs_path}\n")
            f.write(f"{'=' * 60}\n\n")

            f.write(f"SUMMARY\n")
            f.write(f"  Total unique pairs:       {len(unique)}\n")
            f.write(f"  Pairs with {focus_task}:  {len(has_task)}\n")
            f.write(f"  Pairs without:            {len(no_task)}\n")
            f.write(f"  Parents NOT paired at all: {len(not_in_pairs)}\n")
            if statuses:
                f.write(f"\n  Seg status breakdown:\n")
                for s, c in statuses.most_common():
                    f.write(f"    {s}: {c}\n")
                f.write(f"    (not segmented): {len(no_task)}\n")
            cat_counts2 = Counter(x["category"] for x in all_issues)
            f.write(f"\n  Issue categories:\n")
            for cat, cnt in cat_counts2.most_common():
                f.write(f"    {cat}: {cnt}\n")
            f.write(f"    TOTAL: {len(all_issues)}\n")

            # Unpaired parents
            if not_in_pairs:
                f.write(f"\n{'=' * 60}\n")
                f.write(f"UNPAIRED PARENTS ({len(not_in_pairs)})\n")
                f.write(f"These parents exist in HeaderIndex but have no PET/CT pair.\n")
                f.write(f"{'=' * 60}\n\n")
                for pr in not_in_pairs:
                    # Show what series exist for this parent
                    series_for_parent = []
                    for r in headers_doc.get("dicom_series", []) or []:
                        rel = str(r.get("series_rel") or "").replace("\\", "/")
                        modality = str(r.get("modality") or r.get("Modality") or "?")
                        if rel.startswith(pr + "/") or (pr == "." and "/" not in rel):
                            series_name = rel.split("/")[-1] if "/" in rel else rel
                            series_for_parent.append(f"{modality}:{series_name}")
                    f.write(f"  {pr}\n")
                    if series_for_parent:
                        f.write(f"    series: {', '.join(series_for_parent)}\n")
                    f.write(f"\n")

            # All issues detail by category
            f.write(f"\n{'=' * 60}\n")
            f.write(f"ALL ISSUES ({len(all_issues)})\n")
            f.write(f"{'=' * 60}\n")

            for cat_name in ["not_segmented", "seg_failed", "seg_empty", "seg_error",
                             "done_but_empty", "done_but_missing"]:
                items = [x for x in all_issues if x["category"] == cat_name]
                if not items:
                    continue
                f.write(f"\n--- {cat_name} ({len(items)}) ---\n\n")
                for fd in sorted(items, key=lambda x: x["parent_rel"]):
                    ct_short = fd['ct_asset_id'].split(':')[-1] if ':' in fd['ct_asset_id'] else fd['ct_asset_id']
                    f.write(f"{fd['parent_rel']}\n")
                    f.write(f"  ct_asset:   {fd['ct_asset_id']}\n")
                    f.write(f"  ct_dir:     {fd['ct_dir']}\n")
                    f.write(f"  ct_exists:  {fd['ct_exists']}\n")
                    if fd['seg_path']:
                        f.write(f"  seg_path:   {fd['seg_path']}\n")
                        f.write(f"  seg_exists: {fd['seg_exists']}\n")
                        f.write(f"  seg_size:   {fd['seg_size_bytes']} bytes\n")
                    else:
                        f.write(f"  seg_path:   (none)\n")
                    if fd['error']:
                        f.write(f"  error:      {fd['error']}\n")
                    f.write(f"\n")

        print(f"\n  Full report saved to: {report_out}")

        # Seg type breakdown
        seg_types = Counter()
        for p in has_task:
            rec = p["seg_tasks"][focus_task]
            st = str(rec.get("seg_type") or "unknown").strip()
            seg_types[st] += 1
        if seg_types:
            print(f"\n  Seg type breakdown:")
            for s, c in seg_types.most_common():
                print(f"    {s}: {c}")

    # =========================================================================
    # FILE CHECKS (CT/PET)
    # =========================================================================
    if args.check_files and by_asset:
        print(f"\n{'=' * 70}")
        print(f"FILE CHECKS")
        print(f"{'=' * 70}")

        ct_found = 0
        ct_missing = []
        pet_found = 0
        pet_missing = []
        pet_none = 0

        for p in unique:
            ct_row = by_asset.get(p["ct_asset_id"])
            ct_dir = series_abs_dir(ct_row) if ct_row else None
            if ct_dir and os.path.isdir(ct_dir):
                ct_found += 1
            else:
                ct_missing.append((p["parent_rel"], p["ct_asset_id"], ct_dir))

            if not p["pet_asset_id"]:
                pet_none += 1
                continue
            pet_row = by_asset.get(p["pet_asset_id"])
            pet_dir = series_abs_dir(pet_row) if pet_row else None
            if pet_dir and os.path.isdir(pet_dir):
                pet_found += 1
            else:
                pet_missing.append((p["parent_rel"], p["pet_asset_id"], pet_dir))

        print(f"\n  CT directories:")
        print(f"    Found: {ct_found}")
        print(f"    Missing: {len(ct_missing)}")
        for pr, aid, d in ct_missing[:10]:
            print(f"      {pr}: {aid} -> {d}")

        print(f"\n  PET directories:")
        print(f"    Found: {pet_found}")
        print(f"    Missing: {len(pet_missing)}")
        print(f"    No PET paired: {pet_none}")
        for pr, aid, d in pet_missing[:10]:
            print(f"      {pr}: {aid} -> {d}")

    # =========================================================================
    # DUPLICATES
    # =========================================================================
    if args.show_duplicates and dups:
        print(f"\n{'=' * 70}")
        print(f"DUPLICATES ({len(dups)})")
        print(f"{'=' * 70}")
        for d in dups[:30]:
            print(f"  {d['parent_rel']} [{d['source']}]")

    print()


if __name__ == "__main__":
    main()
