#!/usr/bin/env python3
"""
CLI_SubtractFromPairIndex.py — Subtract structures from base structures, within
the PairIndex framework.

Both --base and --subtract accept one or more `module/structure` refs (separator
may be `/` or `:`). Structures are gathered from any registered task in the same
pair, whether that task is a combined multilabel file or a folder of
per-structure NIfTIs.

Behavior
--------
- Base = the structures you keep. Each is carried through as its OWN ROI.
- Subtract = all subtractor structures are UNIONed into one removal mask, which
  is zeroed out of EVERY base structure.
- Output (default): a FOLDER of per-structure NIfTIs, one per surviving base
  structure, named `<module>__<structure>.nii.gz`, registered as an INDIVIDUAL
  task. This is the layout slicer_metrics_worker.py turns into named ROI columns
  (`<module>__<structure>_HU_mean`, ...), so `total/pancreas` and
  `trunk_cavities/pancreas` never collide.
- Output (--union): all surviving base voxels merged into ONE binary mask, saved
  as a single file, registered as a COMBINED task.

A `module/structure` ref:
  - module    = a task registered in PairIndex for the pair (combined or folder)
  - structure = label NUMBER (e.g. 51), label NAME (e.g. pancreas, via the task's
                label map), or a filename stem. 'ALL' or '*' = every structure.

Usage:

  python3 CLI_SubtractFromPairIndex.py \
    --pairs ".../PairIndex.yaml" --seg-root "/" \
    --base "total/pancreas" "total/spleen" \
    --subtract "total/aorta" "trunk_cavities/51" \
    --output-task "abd_minus_vessels"

  python3 CLI_SubtractFromPairIndex.py ... \
    --base "total/pancreas" --subtract "total/aorta" \
    --output-task "pancreas_clean" --union
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
import numpy as np
import SimpleITK as sitk

from platform_paths import to_wsl, to_win

try:
    import totalseg_tasks as tst
except Exception:
    tst = None


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
                dedup_key = (parent_rel, pet, ct)
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                yield parent_rel, entry


def resolve_seg_path(task_rec, seg_root_wsl):
    seg_type = str(task_rec.get("seg_type") or "").strip().lower()
    if seg_type == "combined":
        rel = str((task_rec.get("combined_path") or {}).get("seg_path_rel") or "").strip()
        if rel:
            w = to_wsl(rel)
            if w.startswith("/"):
                return w, "combined"
            if seg_root_wsl:
                return str((Path(seg_root_wsl) / rel).resolve()), "combined"
    if seg_type == "individual":
        rel = str((task_rec.get("organ_dir") or {}).get("dir_path_rel") or "").strip()
        if rel:
            w = to_wsl(rel)
            if w.startswith("/"):
                return w, "individual"
            if seg_root_wsl:
                return str((Path(seg_root_wsl) / rel).resolve()), "individual"
    return None, None


def module_label_map(task_name, task_rec):
    lm = {}
    if tst:
        try:
            got = tst.try_get_label_map(task_name)
            if got:
                lm = {int(k): str(v) for k, v in got.items()}
        except Exception:
            pass
    for k, v in (task_rec.get("label_map") or {}).items():
        try:
            lm[int(k)] = str(v)
        except Exception:
            continue
    return lm


def parse_ref(spec):
    """'module/structure' or 'module:structure' -> (module, structure).
    'module' alone -> (module, '') meaning ALL."""
    s = (spec or "").strip()
    sep = None
    for cand in ("/", ":"):
        if cand in s:
            sep = cand
            break
    if sep is None:
        return s, ""
    mod, struct = s.split(sep, 1)
    return mod.strip(), struct.strip()


def read_image(path_wsl):
    return sitk.ReadImage(path_wsl)

def binarize(img, label=None):
    arr = sitk.GetArrayFromImage(img)
    b = (arr != 0) if label is None else (arr == label)
    out = sitk.GetImageFromArray(b.astype(np.uint8))
    out.CopyInformation(img)
    return out

def resample_mask_to_ref(mask, ref, label=None):
    # Same-CT tasks share a grid -> direct math, no resample, no drift.
    if (mask.GetSize() == ref.GetSize()
            and mask.GetSpacing() == ref.GetSpacing()
            and mask.GetOrigin() == ref.GetOrigin()
            and mask.GetDirection() == ref.GetDirection()):
        return mask
    if label:
        print(f"    WARN: grid mismatch for '{label}' — NN-resampling onto base grid (boundary drift possible)")
    else:
        print("    WARN: grid mismatch — NN-resampling a mask onto base grid (boundary drift possible)")
    r = sitk.ResampleImageFilter()
    r.SetReferenceImage(ref)
    r.SetInterpolator(sitk.sitkNearestNeighbor)
    r.SetDefaultPixelValue(0)
    return r.Execute(mask)

def empty_like(ref):
    z = sitk.GetImageFromArray(np.zeros(sitk.GetArrayFromImage(ref).shape, dtype=np.uint8))
    z.CopyInformation(ref)
    return z


def list_module_structures(task_name, task_rec, seg_path_wsl, seg_type):
    out = []
    if seg_type == "combined":
        lm = module_label_map(task_name, task_rec)
        arr = sitk.GetArrayFromImage(read_image(seg_path_wsl))
        present = sorted(int(x) for x in np.unique(arr) if x != 0)
        for lab in present:
            out.append((lm.get(lab, f"label_{lab}"), "label", lab))
    elif seg_type == "individual":
        for f in sorted(Path(seg_path_wsl).glob("*.nii*")):
            stem = f.name
            for ext in (".nii.gz", ".nii"):
                if stem.endswith(ext):
                    stem = stem[:-len(ext)]
                    break
            out.append((stem, "file", str(f.resolve())))
    return out


def match_structure(structures, query):
    q = (query or "").strip()
    if q == "" or q.upper() in ("ALL", "*"):
        return list(structures)
    as_int = None
    try:
        as_int = int(q)
    except Exception:
        as_int = None
    matches = []
    for (name, kind, resolver) in structures:
        if as_int is not None and kind == "label" and resolver == as_int:
            matches.append((name, kind, resolver))
        elif name.lower() == q.lower():
            matches.append((name, kind, resolver))
    if matches:
        return matches
    return [t for t in structures if q.lower() in t[0].lower()]


def find_task(entry, module_name):
    return (entry.get("segmentations") or {}).get(module_name)


def resolve_refs(entry, refs, seg_root_wsl, verbose=False):
    resolved = []
    for (module_name, struct_query) in refs:
        rec = find_task(entry, module_name)
        if not rec:
            print(f"    WARN: module '{module_name}' not in this pair; skipping")
            continue
        if str(rec.get("status") or "").lower() == "failed":
            print(f"    WARN: module '{module_name}' status=failed; skipping")
            continue
        mpath, mtype = resolve_seg_path(rec, seg_root_wsl)
        if not mpath:
            print(f"    WARN: cannot resolve path for module '{module_name}'; skipping")
            continue
        structures = list_module_structures(module_name, rec, mpath, mtype)
        chosen = match_structure(structures, struct_query)
        if not chosen:
            print(f"    WARN: no structure matched '{module_name}/{struct_query}'; skipping")
            continue
        for (name, kind, resolver) in chosen:
            if kind == "label":
                b = binarize(read_image(mpath), label=resolver)
            else:
                b = binarize(read_image(resolver), label=None)
            resolved.append((module_name, name, b))
            if verbose:
                print(f"    + {module_name}__{name}")
    return resolved


def sanitize(s):
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(s))

def roi_name(module, structure):
    return f"{sanitize(module)}__{sanitize(structure)}"


def output_dir_for(entry, seg_root_wsl, base_first_module, output_task):
    rec = find_task(entry, base_first_module)
    base_path, base_type = resolve_seg_path(rec, seg_root_wsl) if rec else (None, None)
    if base_path:
        # Write into the CT SEG DIR (the folder that holds total/, trunk_cavities/),
        # so output lands as a sibling of the source modules.
        #  - combined file .../CTdir/total.nii.gz            -> CTdir
        #  - combined file .../CTdir/trunk_cavities/tc.nii.gz -> CTdir (folder is module-named)
        #  - individual dir .../CTdir/trunk_cavities/         -> CTdir
        p = Path(base_path)
        if base_type == "individual":
            ct_seg_dir = p.parent  # p is the module folder itself
        else:
            # combined file: if it sits in a subfolder named after the module, go up twice
            ct_seg_dir = p.parent.parent if p.parent.name == base_first_module else p.parent
        parent = ct_seg_dir
    elif seg_root_wsl:
        parent = Path(seg_root_wsl)
    else:
        parent = Path(".")
    d = parent / output_task
    d.mkdir(parents=True, exist_ok=True)
    return d


def register_individual(entry, output_task, dir_rel, base_refs, sub_refs, overwrite):
    segs = entry.setdefault("segmentations", {})
    if output_task in segs and not overwrite:
        return False
    segs[output_task] = {
        "status": "done",
        "source": "subtraction",
        "registered_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "seg_type": "individual",
        "organ_dir": {"dir_path_rel": dir_rel},
        "base": [f"{m}/{s or 'ALL'}" for (m, s) in base_refs],
        "subtracted": [f"{m}/{s or 'ALL'}" for (m, s) in sub_refs],
    }
    return True


def register_combined(entry, output_task, file_rel, base_refs, sub_refs, overwrite):
    segs = entry.setdefault("segmentations", {})
    if output_task in segs and not overwrite:
        return False
    segs[output_task] = {
        "status": "done",
        "source": "subtraction",
        "registered_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "seg_type": "combined",
        "combined_path": {"seg_path_rel": file_rel},
        "base": [f"{m}/{s or 'ALL'}" for (m, s) in base_refs],
        "subtracted": [f"{m}/{s or 'ALL'}" for (m, s) in sub_refs],
        "union": True,
    }
    return True


def rel_to_segroot(path_wsl, seg_root_wsl):
    if seg_root_wsl:
        try:
            return str(Path(path_wsl).relative_to(Path(seg_root_wsl)))
        except Exception:
            pass
    return str(path_wsl)


def main():
    p = argparse.ArgumentParser(description="Subtract structures from base structures within PairIndex.")
    p.add_argument("--pairs", required=True)
    p.add_argument("--headers", required=False, help="(accepted for parity; unused)")
    p.add_argument("--base", nargs="+", required=True,
                   help="module/structure refs to KEEP (each becomes its own ROI).")
    p.add_argument("--subtract", nargs="*", default=[],
                   help="module/structure refs to REMOVE (unioned, removed from every base structure).")
    p.add_argument("--output-task", required=True)
    p.add_argument("--seg-root", type=str, default=None)
    p.add_argument("--parents", type=str, default="")
    p.add_argument("--union", action="store_true",
                   help="Collapse surviving base structures into one binary mask (single combined file).")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    pairs_wsl = to_wsl(args.pairs)
    pair_doc = load_yaml(pairs_wsl)

    seg_root_wsl = to_wsl(args.seg_root) if args.seg_root else None
    if not seg_root_wsl:
        out_root = str(((pair_doc.get("derived") or {}).get("last_totalseg_run") or {}).get("out_root") or "")
        if out_root:
            seg_root_wsl = to_wsl(out_root)

    parents_filter = set(t.strip() for t in args.parents.split(",") if t.strip()) if args.parents.strip() else None

    base_refs = [parse_ref(s) for s in args.base]
    sub_refs = [parse_ref(s) for s in args.subtract]

    changed = 0
    processed = 0

    for parent_rel, entry in iter_pairs(pair_doc, parents_filter):
        if not any(find_task(entry, m) for (m, _) in base_refs):
            continue

        print(f"\n[{parent_rel}] base={args.base} subtract={args.subtract or '(none)'}")

        base_list = resolve_refs(entry, base_refs, seg_root_wsl, verbose=args.verbose)
        if not base_list:
            print("  SKIP: no base structures resolved")
            continue

        ref = base_list[0][2]

        removal = None
        if sub_refs:
            sub_list = resolve_refs(entry, sub_refs, seg_root_wsl, verbose=args.verbose)
            for (_, _, b) in sub_list:
                b_on_ref = resample_mask_to_ref(b, ref)
                removal = b_on_ref if removal is None else sitk.Or(removal, b_on_ref)
        if removal is None:
            removal = empty_like(ref)
        removal_arr = sitk.GetArrayFromImage(removal) > 0

        results = []
        for (module, structure, b) in base_list:
            b_on_ref = resample_mask_to_ref(b, ref)
            arr = sitk.GetArrayFromImage(b_on_ref).astype(bool)
            before = int(arr.sum())
            kept = arr & ~removal_arr
            out_img = sitk.GetImageFromArray(kept.astype(np.uint8))
            out_img.CopyInformation(ref)
            results.append((module, structure, out_img, int(kept.sum()), before - int(kept.sum())))
            print(f"    {roi_name(module, structure)}: kept {int(kept.sum())}, removed {before - int(kept.sum())}")

        processed += 1
        if args.dry_run:
            continue

        out_dir = output_dir_for(entry, seg_root_wsl, base_refs[0][0], args.output_task)

        if args.union:
            merged = None
            for (_, _, out_img, _, _) in results:
                a = sitk.GetArrayFromImage(out_img) > 0
                merged = a if merged is None else (merged | a)
            m_img = sitk.GetImageFromArray(merged.astype(np.uint8))
            m_img.CopyInformation(ref)
            out_file = out_dir / f"{args.output_task}.nii.gz"
            sitk.WriteImage(m_img, str(out_file))
            rel = rel_to_segroot(str(out_file), seg_root_wsl)
            if register_combined(entry, args.output_task, rel.replace("\\", "/"),
                                 base_refs, sub_refs, args.overwrite):
                changed += 1
                print(f"    registered COMBINED '{args.output_task}' -> {out_file}")
            else:
                print(f"    task exists (use --overwrite); wrote {out_file}")
        else:
            written = 0
            for (module, structure, out_img, kept, _) in results:
                fname = f"{roi_name(module, structure)}.nii.gz"
                sitk.WriteImage(out_img, str(out_dir / fname))
                written += 1
            rel = rel_to_segroot(str(out_dir), seg_root_wsl)
            if register_individual(entry, args.output_task, rel.replace("\\", "/"),
                                   base_refs, sub_refs, args.overwrite):
                changed += 1
                print(f"    registered INDIVIDUAL '{args.output_task}' ({written} files) -> {out_dir}")
            else:
                print(f"    task exists (use --overwrite); wrote {written} files to {out_dir}")

    if not args.dry_run and changed:
        save_yaml(pair_doc, pairs_wsl)
        print(f"\nUpdated PairIndex: {pairs_wsl}  ({changed} task(s) registered)")
    else:
        print(f"\nDone. processed={processed}, registered={changed}"
              + (" (dry-run)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
