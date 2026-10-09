#!/usr/bin/env python3
"""
CLI_MorphFromPairIndex.py — Dilate (expand) or erode (contract) segmentation
structures by a millimetre margin, within the PairIndex framework.

Accepts one or more `module/structure` refs (separator `/` or `:`). Distances are
true millimetres via SignedMaurerDistanceMap with insideIsPositive=False
(interior distances NEGATIVE). Full-3D isotropic morphology over-erodes in the
slice (z) direction on thick slices, so IN-PLANE (axial) is the DEFAULT, with an
optional separate --z-mm margin; --mode 3d gives full isotropic.

Output (default): a FOLDER of per-structure NIfTIs, one per morphed structure,
named `<module>__<structure>.nii.gz`, registered as an INDIVIDUAL task — the
layout slicer_metrics_worker.py turns into named ROI columns. Overlaps between
grown structures are preserved because each is its own file.

Output (--union): all morphed structures merged into ONE binary mask (single
combined file).

A `module/structure` ref:
  - module    = a task registered in PairIndex (combined file or folder)
  - structure = label NUMBER, label NAME (via label map), or file stem. 'ALL'/'*'
                = every structure in the module.

Usage:

  # erode pancreas inward 2 mm in-plane (strip partial-volume rim)
  python3 CLI_MorphFromPairIndex.py \
    --pairs ".../PairIndex.yaml" --seg-root "/" \
    --op erode --mm 2 \
    --structures "total/pancreas" \
    --output-task "pancreas_ero2mm"

  # dilate vessels 3 mm before using them as a subtractor
  python3 CLI_MorphFromPairIndex.py ... \
    --op dilate --mm 3 \
    --structures "total/aorta" "total/brachiocephalic_trunk" \
    --output-task "vessels_dil3mm"

  # in-plane 2 mm, only 1 mm in z
  python3 CLI_MorphFromPairIndex.py ... --op erode --mm 2 --z-mm 1 \
    --structures "total/pancreas" --output-task "pancreas_ero"
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


# ---------------------------------------------------------------------------
# Morphology — mm-accurate; in-plane default (carried from prior erosion work)
# ---------------------------------------------------------------------------

def distance_signed_mm(binary):
    return sitk.SignedMaurerDistanceMap(binary, insideIsPositive=False,
                                        squaredDistance=False, useImageSpacing=True)

def erode_mm_3d(mask, mm):
    if mm <= 0:
        return binarize(mask)
    d = distance_signed_mm(binarize(mask))
    out = sitk.BinaryThreshold(d, lowerThreshold=-1e12, upperThreshold=-float(mm),
                               insideValue=1, outsideValue=0)
    return sitk.Cast(out, sitk.sitkUInt8)

def dilate_mm_3d(mask, mm):
    if mm <= 0:
        return binarize(mask)
    d = distance_signed_mm(binarize(mask))
    out = sitk.BinaryThreshold(d, lowerThreshold=-1e12, upperThreshold=float(mm),
                               insideValue=1, outsideValue=0)
    return sitk.Cast(out, sitk.sitkUInt8)

def _inplane_mm(mask, mm, grow):
    sp = mask.GetSpacing()  # (x, y, z)
    arr = sitk.GetArrayFromImage(binarize(mask))  # [z, y, x]
    out = np.zeros_like(arr)
    for z in range(arr.shape[0]):
        sl = arr[z]
        if sl.sum() == 0 and not grow:
            continue
        if mm <= 0:
            out[z] = sl
            continue
        sl_img = sitk.GetImageFromArray(sl.astype(np.uint8))
        sl_img.SetSpacing((sp[0], sp[1]))
        d = sitk.SignedMaurerDistanceMap(sl_img, insideIsPositive=False,
                                         squaredDistance=False, useImageSpacing=True)
        da = sitk.GetArrayFromImage(d)
        thr = float(mm) if grow else -float(mm)
        out[z] = (da <= thr).astype(np.uint8)
    return out

def _apply_z_mm(arr, z_mm, spacing_z, grow):
    if z_mm <= 0:
        return arr
    k = int(round(z_mm / float(spacing_z)))
    if k <= 0:
        return arr
    from scipy.ndimage import binary_erosion, binary_dilation
    struct = np.zeros((3, 1, 1), dtype=bool)
    struct[:, 0, 0] = True
    op = binary_dilation if grow else binary_erosion
    return op(arr.astype(bool), structure=struct, iterations=k).astype(np.uint8)

def morph_inplane(mask, mm, z_mm, grow):
    arr = _inplane_mm(mask, mm, grow)
    if z_mm > 0:
        arr = _apply_z_mm(arr, z_mm, mask.GetSpacing()[2], grow)
    out = sitk.GetImageFromArray(arr.astype(np.uint8))
    out.CopyInformation(mask)
    return out

def apply_op(mask_bin, op, mm, z_mm, mode):
    grow = (op == "dilate")
    if mode == "3d":
        if z_mm and z_mm != mm:
            return morph_inplane(mask_bin, mm, z_mm, grow)
        return dilate_mm_3d(mask_bin, mm) if grow else erode_mm_3d(mask_bin, mm)
    return morph_inplane(mask_bin, mm, z_mm, grow)


def sanitize(s):
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(s))

def roi_name(module, structure):
    return f"{sanitize(module)}__{sanitize(structure)}"


def output_dir_for(entry, seg_root_wsl, first_module, output_task):
    rec = find_task(entry, first_module)
    base_path, base_type = resolve_seg_path(rec, seg_root_wsl) if rec else (None, None)
    if base_path:
        p = Path(base_path)
        if base_type == "individual":
            ct_seg_dir = p.parent
        else:
            ct_seg_dir = p.parent.parent if p.parent.name == first_module else p.parent
        parent = ct_seg_dir
    elif seg_root_wsl:
        parent = Path(seg_root_wsl)
    else:
        parent = Path(".")
    d = parent / output_task
    d.mkdir(parents=True, exist_ok=True)
    return d


def rel_to_segroot(path_wsl, seg_root_wsl):
    if seg_root_wsl:
        try:
            return str(Path(path_wsl).relative_to(Path(seg_root_wsl)))
        except Exception:
            pass
    return str(path_wsl)


def main():
    p = argparse.ArgumentParser(description="Dilate/erode structures (mm) within PairIndex.")
    p.add_argument("--pairs", required=True)
    p.add_argument("--headers", required=False, help="(accepted for parity; unused)")
    p.add_argument("--op", required=True, choices=["dilate", "erode"])
    p.add_argument("--mm", type=float, default=0.0, help="In-plane (or 3D) margin, mm")
    p.add_argument("--z-mm", type=float, default=0.0, help="Separate z margin, mm")
    p.add_argument("--mode", choices=["inplane", "3d"], default="inplane",
                   help="inplane (default) or full 3D isotropic")
    p.add_argument("--structures", nargs="+", required=True,
                   help="module/structure refs to morph (each becomes its own ROI).")
    p.add_argument("--output-task", required=True)
    p.add_argument("--seg-root", type=str, default=None)
    p.add_argument("--parents", type=str, default="")
    p.add_argument("--union", action="store_true",
                   help="Merge morphed structures into one binary mask (single combined file).")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    if args.mm <= 0 and args.z_mm <= 0:
        print("ERROR: give a positive --mm and/or --z-mm.")
        sys.exit(1)

    pairs_wsl = to_wsl(args.pairs)
    pair_doc = load_yaml(pairs_wsl)

    seg_root_wsl = to_wsl(args.seg_root) if args.seg_root else None
    if not seg_root_wsl:
        out_root = str(((pair_doc.get("derived") or {}).get("last_totalseg_run") or {}).get("out_root") or "")
        if out_root:
            seg_root_wsl = to_wsl(out_root)

    parents_filter = set(t.strip() for t in args.parents.split(",") if t.strip()) if args.parents.strip() else None
    refs = [parse_ref(s) for s in args.structures]
    tag = ("dil" if args.op == "dilate" else "ero")

    changed = 0
    processed = 0

    for parent_rel, entry in iter_pairs(pair_doc, parents_filter):
        if not any(find_task(entry, m) for (m, _) in refs):
            continue

        print(f"\n[{parent_rel}] op={args.op} mm={args.mm} z_mm={args.z_mm} mode={args.mode} structures={args.structures}")

        struct_list = resolve_refs(entry, refs, seg_root_wsl, verbose=args.verbose)
        if not struct_list:
            print("  SKIP: no structures resolved")
            continue

        results = []
        for (module, structure, b) in struct_list:
            res = apply_op(b, args.op, args.mm, args.z_mm, args.mode)
            before = int(sitk.GetArrayFromImage(b).sum())
            after = int(sitk.GetArrayFromImage(res).sum())
            results.append((module, structure, sitk.Cast(res, sitk.sitkUInt8)))
            print(f"    {roi_name(module, structure)}: {before} -> {after} voxels")

        processed += 1
        if args.dry_run:
            continue

        out_dir = output_dir_for(entry, seg_root_wsl, refs[0][0], args.output_task)

        if args.union:
            merged = None
            ref = results[0][2]
            for (_, _, img) in results:
                a = sitk.GetArrayFromImage(img) > 0
                merged = a if merged is None else (merged | a)
            m_img = sitk.GetImageFromArray(merged.astype(np.uint8))
            m_img.CopyInformation(ref)
            out_file = out_dir / f"{args.output_task}.nii.gz"
            sitk.WriteImage(m_img, str(out_file))
            rel = rel_to_segroot(str(out_file), seg_root_wsl).replace("\\", "/")
            segs = entry.setdefault("segmentations", {})
            if args.output_task in segs and not args.overwrite:
                print(f"    task exists (use --overwrite); wrote {out_file}")
            else:
                segs[args.output_task] = {
                    "status": "done", "source": "morphology",
                    "registered_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                    "seg_type": "combined",
                    "combined_path": {"seg_path_rel": rel},
                    "op": args.op, "margin_mm": args.mm, "z_margin_mm": args.z_mm,
                    "mode": args.mode, "union": True,
                    "structures": [f"{m}/{s or 'ALL'}" for (m, s) in refs],
                }
                changed += 1
                print(f"    registered COMBINED '{args.output_task}' -> {out_file}")
        else:
            written = 0
            for (module, structure, img) in results:
                fname = f"{roi_name(module, structure)}.nii.gz"
                sitk.WriteImage(img, str(out_dir / fname))
                written += 1
            rel = rel_to_segroot(str(out_dir), seg_root_wsl).replace("\\", "/")
            segs = entry.setdefault("segmentations", {})
            if args.output_task in segs and not args.overwrite:
                print(f"    task exists (use --overwrite); wrote {written} files to {out_dir}")
            else:
                segs[args.output_task] = {
                    "status": "done", "source": "morphology",
                    "registered_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                    "seg_type": "individual",
                    "organ_dir": {"dir_path_rel": rel},
                    "op": args.op, "margin_mm": args.mm, "z_margin_mm": args.z_mm,
                    "mode": args.mode,
                    "structures": [f"{m}/{s or 'ALL'}" for (m, s) in refs],
                }
                changed += 1
                print(f"    registered INDIVIDUAL '{args.output_task}' ({written} files) -> {out_dir}")

    if not args.dry_run and changed:
        save_yaml(pair_doc, pairs_wsl)
        print(f"\nUpdated PairIndex: {pairs_wsl}  ({changed} task(s) registered)")
    else:
        print(f"\nDone. processed={processed}, registered={changed}"
              + (" (dry-run)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
