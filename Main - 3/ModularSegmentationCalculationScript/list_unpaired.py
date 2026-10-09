#!/usr/bin/env python3
"""
list_unpaired.py — Show unpaired parents and their DICOM series folder names.

Reads PairIndex.yaml and HeaderIndex.yaml, prints each parent that landed
in needs_selection with the series folders inside it.

Usage:
  python3 list_unpaired.py --dir "/mnt/f/HNC/Directory Plan"
  python3 list_unpaired.py --pairs PairIndex.yaml --headers HeaderIndex.yaml
"""

import argparse
import sys
from pathlib import Path

import yaml


def load_yaml(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    p = argparse.ArgumentParser(description="List unpaired parents with their DICOM series folders.")
    p.add_argument("--dir", type=str, default=None, help="Directory Plan folder (contains PairIndex.yaml + HeaderIndex.yaml)")
    p.add_argument("--pairs", type=str, default=None, help="PairIndex.yaml path")
    p.add_argument("--headers", type=str, default=None, help="HeaderIndex.yaml path")
    args = p.parse_args()

    if args.dir:
        d = Path(args.dir).expanduser().resolve()
        pairs_path = d / "PairIndex.yaml"
        headers_path = d / "HeaderIndex.yaml"
    elif args.pairs and args.headers:
        pairs_path = Path(args.pairs).expanduser().resolve()
        headers_path = Path(args.headers).expanduser().resolve()
    else:
        print("Provide --dir or both --pairs and --headers", file=sys.stderr)
        sys.exit(1)

    pair_doc = load_yaml(pairs_path)
    header_doc = load_yaml(headers_path)

    # Get unpaired parents
    needs = pair_doc.get("needs_selection", []) or []
    unpaired_parents = set()
    for item in needs:
        pr = item.get("parent_rel", "")
        if pr:
            unpaired_parents.add(pr)

    if not unpaired_parents:
        print("No unpaired parents found. All parents are paired.")
        return

    # Group series by parent_rel
    series_by_parent = {}
    for r in header_doc.get("dicom_series", []) or []:
        pr = r.get("parent_rel", ".")
        if pr not in unpaired_parents:
            continue
        if pr not in series_by_parent:
            series_by_parent[pr] = []
        series_by_parent[pr].append(r)

    print(f"{'='*70}")
    print(f"UNPAIRED PARENTS: {len(unpaired_parents)}")
    print(f"{'='*70}\n")

    for pr in sorted(unpaired_parents):
        # Find reason
        reason = ""
        for item in needs:
            if item.get("parent_rel") == pr:
                reason = item.get("reason", "")
                break

        print(f"Parent: {pr}")
        print(f"  Reason: {reason}")

        rows = series_by_parent.get(pr, [])
        if not rows:
            print("  (no series found in HeaderIndex)")
        else:
            for r in rows:
                mod = r.get("modality", "?")
                rel = r.get("series_rel", "")
                desc = r.get("series_desc", "")
                flags = r.get("flags", {}) or {}
                contrast = flags.get("contrast")
                gated = flags.get("gated", False)
                ac = flags.get("ac")
                wb = (r.get("classification", {}) or {}).get("is_whole_body", False)

                tags = []
                if contrast is True:
                    tags.append("CONTRAST")
                elif contrast is False:
                    tags.append("non-contrast")
                if gated:
                    tags.append("GATED")
                if mod == "PET":
                    if ac is True:
                        tags.append("AC")
                    elif ac is False:
                        tags.append("NAC")
                if wb:
                    tags.append("WB")
                if flags.get("scout"):
                    tags.append("SCOUT")

                tag_str = f" [{', '.join(tags)}]" if tags else ""
                print(f"  {mod:4s} | {rel}")
                print(f"         desc: {desc}{tag_str}")
        print()

    print(f"{'='*70}")
    print(f"Total unpaired: {len(unpaired_parents)}")


if __name__ == "__main__":
    main()