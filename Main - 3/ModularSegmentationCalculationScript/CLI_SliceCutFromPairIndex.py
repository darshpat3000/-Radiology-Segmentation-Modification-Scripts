#!/usr/bin/env python3
"""
CLI_SliceCutFromPairIndex.py — Remove axial slices of base structures relative to
an anatomical level defined by a reference structure, within the PairIndex
framework.

The cut LEVEL is taken from a reference structure's extent along the
superior-inferior (world Z) axis:
  --ref-point top     -> the reference's most-superior slice
  --ref-point bottom  -> the reference's most-inferior slice
  --ref-point center  -> the reference's mid slice

What gets removed from each base structure:
  --remove above      -> drop everything superior to the level
  --remove below      -> drop everything inferior to the level
  --between R1 R2      -> keep only the band between two references' levels
                         (remove everything outside it)

--inclusive / --exclusive controls whether the level slice itself is kept.

Directionality is resolved in WORLD space using the image direction cosines, so
"above" always means toward the head regardless of array storage order.

Refs use the same `module/structure` convention as the morph/subtract tools
(separator `/` or `:`; structure = label #, label name, or file stem; ALL/*).

Output: a FOLDER of per-structure NIfTIs named `<module>__<structure>.nii.gz`,
registered as an INDIVIDUAL task (or one merged binary file with --union).

Grid handling: all tasks derived from the same CT share its grid, so when grids
match this does direct array math (no resampling, no boundary drift). If a
reference's grid genuinely differs it is NN-resampled onto the base grid with a
warning.

Usage:

  # keep only the aorta at/below the top of L1
  python3 CLI_SliceCutFromPairIndex.py \
    --pairs ".../PairIndex.yaml" --seg-root "/" \
    --base "total/aorta" \
    --ref "total/vertebrae_L1" --ref-point top --remove above --inclusive \
    --output-task "aorta_below_L1"

  # keep aorta between the top of L1 and the bottom of L4
  python3 CLI_SliceCutFromPairIndex.py ... \
    --base "total/aorta" \
    --between "total/vertebrae_L1" "total/vertebrae_L4" \
    --output-task "aorta_L1_L4"
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


# --------------------------------------------------------------------------
# YAML / PairIndex (shared conventions)
# --------------------------------------------------------------------------

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
                dedup = (parent_rel, pet, ct)
                if dedup in seen:
                    continue
                seen.add(dedup)
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
    m, st = s.split(sep, 1)
    return m.strip(), st.strip()


# --------------------------------------------------------------------------
# Image helpers
# --------------------------------------------------------------------------

def read_image(path_wsl):
    return sitk.ReadImage(path_wsl)

def binarize(img, label=None):
    arr = sitk.GetArrayFromImage(img)
    b = (arr != 0) if label is None else (arr == label)
    out = sitk.GetImageFromArray(b.astype(np.uint8))
    out.CopyInformation(img)
    return out

def grids_match(a, b):
    return (a.GetSize() == b.GetSize()
            and a.GetSpacing() == b.GetSpacing()
            and a.GetOrigin() == b.GetOrigin()
            and a.GetDirection() == b.GetDirection())

def resample_mask_to_ref(mask, ref):
    if grids_match(mask, ref):
        return mask
    r = sitk.ResampleImageFilter()
    r.SetReferenceImage(ref)
    r.SetInterpolator(sitk.sitkNearestNeighbor)
    r.SetDefaultPixelValue(0)
    return r.Execute(mask)


def z_axis_points_superior(img):
    """Return True if increasing array-z (SimpleITK index k) moves toward the
    head (superior). Uses the direction cosine for the k axis' world-Z component.

    SimpleITK GetDirection() is a 9-tuple (row-major) mapping index (i,j,k) to
    world (x,y,z). The world-Z contribution of the k index is element [8]
    (the (z, k) entry). Positive means +k -> +worldZ (superior)."""
    d = img.GetDirection()
    # world Z from index k is d[6]*? ... build 3x3
    # d = [xx, xy, xz, yx, yy, yz, zx, zy, zz] mapping (i,j,k)->(x,y,z)
    # world_z = zx*i + zy*j + zz*k  => coefficient of k is d[8]
    return d[8] >= 0


def slice_world_z_of_index(img, k):
    """World Z coordinate of voxel index (0,0,k)."""
    return img.TransformIndexToPhysicalPoint((0, 0, int(k)))[2]


def ref_level_index(ref_bin_on_base, base_img, point):
    """Given a reference binary already on the base grid, return the base
    array-z index (k) corresponding to the requested anatomical point
    (top/bottom/center) in WORLD superior-inferior terms."""
    arr = sitk.GetArrayFromImage(ref_bin_on_base)  # [k, j, i] (z,y,x)
    ks = np.where(arr.any(axis=(1, 2)))[0]
    if ks.size == 0:
        return None
    kmin, kmax = int(ks.min()), int(ks.max())
    sup = z_axis_points_superior(base_img)
    # top = most superior; bottom = most inferior
    if point == "center":
        return int(round((kmin + kmax) / 2.0))
    if point == "top":
        return kmax if sup else kmin
    if point == "bottom":
        return kmin if sup else kmax
    return None


def build_slice_keep_mask(base_img, level_k, remove, inclusive):
    """Return a boolean [k,j,i] mask of which slices to KEEP.
    remove in {'above','below'}; 'above' = superior side; direction-aware."""
    nz = base_img.GetSize()[2]
    sup = z_axis_points_superior(base_img)
    keep = np.zeros(nz, dtype=bool)

    # Determine, per array index k, whether it is superior to level.
    # If +k is superior, indices > level_k are 'above'; else indices < level_k.
    for k in range(nz):
        if sup:
            is_above = k > level_k
            is_below = k < level_k
        else:
            is_above = k < level_k
            is_below = k > level_k
        at_level = (k == level_k)

        if remove == "above":
            if is_above:
                keep[k] = False
            elif is_below:
                keep[k] = True
            else:  # at level
                keep[k] = inclusive
        else:  # remove below
            if is_below:
                keep[k] = False
            elif is_above:
                keep[k] = True
            else:
                keep[k] = inclusive
    return keep  # 1-D over k


def build_between_keep_mask(base_img, k1, k2, inclusive):
    """Keep slices between the two levels (inclusive/exclusive at the ends)."""
    nz = base_img.GetSize()[2]
    lo, hi = sorted([k1, k2])
    keep = np.zeros(nz, dtype=bool)
    for k in range(nz):
        if lo < k < hi:
            keep[k] = True
        elif k == lo or k == hi:
            keep[k] = inclusive
    return keep


# --------------------------------------------------------------------------
# Structure resolution (module/structure)
# --------------------------------------------------------------------------

def find_task(entry, module_name):
    return (entry.get("segmentations") or {}).get(module_name)

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

def resolve_one_ref(entry, ref, seg_root_wsl):
    """Resolve a single (module, structure) ref to (name, binary_img) union of
    all matched structures on that module's own grid. Returns (label, img)."""
    module_name, struct_query = ref
    rec = find_task(entry, module_name)
    if not rec:
        return None
    mpath, mtype = resolve_seg_path(rec, seg_root_wsl)
    if not mpath:
        return None
    structs = list_module_structures(module_name, rec, mpath, mtype)
    chosen = match_structure(structs, struct_query)
    if not chosen:
        return None
    union = None
    for (name, kind, resolver) in chosen:
        b = binarize(read_image(mpath), label=resolver) if kind == "label" else binarize(read_image(resolver))
        union = b if union is None else sitk.Or(union, b)
    return (f"{module_name}__{struct_query or 'ALL'}", union)

def resolve_base_refs(entry, refs, seg_root_wsl, verbose=False):
    """Each base ref -> list of (module, structure_name, binary_img) individually."""
    out = []
    for (module_name, struct_query) in refs:
        rec = find_task(entry, module_name)
        if not rec:
            print(f"    WARN: module '{module_name}' not in pair; skipping")
            continue
        mpath, mtype = resolve_seg_path(rec, seg_root_wsl)
        if not mpath:
            print(f"    WARN: cannot resolve '{module_name}'; skipping")
            continue
        structs = list_module_structures(module_name, rec, mpath, mtype)
        chosen = match_structure(structs, struct_query)
        if not chosen:
            print(f"    WARN: no match '{module_name}/{struct_query}'; skipping")
            continue
        for (name, kind, resolver) in chosen:
            b = binarize(read_image(mpath), label=resolver) if kind == "label" else binarize(read_image(resolver))
            out.append((module_name, name, b))
            if verbose:
                print(f"    base + {module_name}__{name}")
    return out


# --------------------------------------------------------------------------
# Output naming / registration
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Cut base ROI slices above/below/between a reference ROI level.")
    p.add_argument("--pairs", required=True)
    p.add_argument("--headers", required=False, help="(accepted for parity; unused)")
    p.add_argument("--base", nargs="+", required=True, help="module/structure refs to cut (each its own ROI).")

    p.add_argument("--ref", help="reference module/structure defining the cut level (with --remove).")
    p.add_argument("--ref-point", choices=["top", "bottom", "center"], default="top",
                   help="which slice of the reference sets the level.")
    p.add_argument("--remove", choices=["above", "below"], help="which side of the level to drop.")

    p.add_argument("--between", nargs=2, metavar=("REF1", "REF2"),
                   help="keep only the band between two references' levels.")
    p.add_argument("--between-points", nargs=2, metavar=("P1", "P2"), default=["top", "bottom"],
                   help="ref-points for --between (default: top bottom).")

    p.add_argument("--inclusive", action="store_true", help="keep the level slice(s) (default).")
    p.add_argument("--exclusive", action="store_true", help="remove the level slice(s).")

    p.add_argument("--output-task", required=True)
    p.add_argument("--seg-root", type=str, default=None)
    p.add_argument("--parents", type=str, default="")
    p.add_argument("--union", action="store_true", help="merge results into one binary file.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    # validate mode
    if args.between:
        if args.remove or args.ref:
            print("ERROR: use EITHER --between OR (--ref + --remove), not both.")
            sys.exit(1)
    else:
        if not (args.ref and args.remove):
            print("ERROR: need --ref and --remove (or use --between REF1 REF2).")
            sys.exit(1)
    inclusive = True
    if args.exclusive:
        inclusive = False
    if args.inclusive:
        inclusive = True

    pairs_wsl = to_wsl(args.pairs)
    pair_doc = load_yaml(pairs_wsl)
    seg_root_wsl = to_wsl(args.seg_root) if args.seg_root else None
    if not seg_root_wsl:
        out_root = str(((pair_doc.get("derived") or {}).get("last_totalseg_run") or {}).get("out_root") or "")
        if out_root:
            seg_root_wsl = to_wsl(out_root)

    parents_filter = set(t.strip() for t in args.parents.split(",") if t.strip()) if args.parents.strip() else None
    base_refs = [parse_ref(s) for s in args.base]

    changed = 0
    processed = 0

    for parent_rel, entry in iter_pairs(pair_doc, parents_filter):
        if not any(find_task(entry, m) for (m, _) in base_refs):
            continue

        print(f"\n[{parent_rel}] base={args.base} "
              + (f"between={args.between} pts={args.between_points}" if args.between
                 else f"ref={args.ref} point={args.ref_point} remove={args.remove}")
              + f" inclusive={inclusive}")

        base_list = resolve_base_refs(entry, base_refs, seg_root_wsl, verbose=args.verbose)
        if not base_list:
            print("  SKIP: no base structures resolved")
            continue
        ref_img_grid = base_list[0][2]  # base grid reference

        # Resolve the reference level(s) on the base grid
        if args.between:
            r1 = resolve_one_ref(entry, parse_ref(args.between[0]), seg_root_wsl)
            r2 = resolve_one_ref(entry, parse_ref(args.between[1]), seg_root_wsl)
            if not r1 or not r2:
                print("  SKIP: could not resolve one of the --between references")
                continue
            r1_on = r1[1] if grids_match(r1[1], ref_img_grid) else _warn_resample(r1[1], ref_img_grid, r1[0])
            r2_on = r2[1] if grids_match(r2[1], ref_img_grid) else _warn_resample(r2[1], ref_img_grid, r2[0])
            k1 = ref_level_index(r1_on, ref_img_grid, args.between_points[0])
            k2 = ref_level_index(r2_on, ref_img_grid, args.between_points[1])
            if k1 is None or k2 is None:
                print("  SKIP: reference structure(s) empty")
                continue
            keep_z = build_between_keep_mask(ref_img_grid, k1, k2, inclusive)
            print(f"    band between k={k1} and k={k2}: keeping {int(keep_z.sum())}/{keep_z.size} slices")
        else:
            rr = resolve_one_ref(entry, parse_ref(args.ref), seg_root_wsl)
            if not rr:
                print("  SKIP: could not resolve --ref")
                continue
            rr_on = rr[1] if grids_match(rr[1], ref_img_grid) else _warn_resample(rr[1], ref_img_grid, rr[0])
            level_k = ref_level_index(rr_on, ref_img_grid, args.ref_point)
            if level_k is None:
                print("  SKIP: reference structure empty")
                continue
            keep_z = build_slice_keep_mask(ref_img_grid, level_k, args.remove, inclusive)
            wz = slice_world_z_of_index(ref_img_grid, level_k)
            print(f"    level at k={level_k} (worldZ={wz:.1f}mm), remove {args.remove}: "
                  f"keeping {int(keep_z.sum())}/{keep_z.size} slices")

        # Apply the slice keep-mask to each base structure
        results = []
        for (module, structure, b) in base_list:
            b_on = b if grids_match(b, ref_img_grid) else _warn_resample(b, ref_img_grid, roi_name(module, structure))
            arr = sitk.GetArrayFromImage(b_on)  # [k,j,i]
            out = arr.copy()
            out[~keep_z, :, :] = 0
            before = int((arr != 0).sum())
            after = int((out != 0).sum())
            out_img = sitk.GetImageFromArray(out.astype(np.uint8))
            out_img.CopyInformation(ref_img_grid)
            results.append((module, structure, out_img))
            print(f"    {roi_name(module, structure)}: {before} -> {after} voxels")

        processed += 1
        if args.dry_run:
            continue

        out_dir = output_dir_for(entry, seg_root_wsl, base_refs[0][0], args.output_task)
        segs = entry.setdefault("segmentations", {})

        if args.union:
            merged = None
            for (_, _, img) in results:
                a = sitk.GetArrayFromImage(img) > 0
                merged = a if merged is None else (merged | a)
            m_img = sitk.GetImageFromArray(merged.astype(np.uint8))
            m_img.CopyInformation(ref_img_grid)
            out_file = out_dir / f"{args.output_task}.nii.gz"
            sitk.WriteImage(m_img, str(out_file))
            rel = rel_to_segroot(str(out_file), seg_root_wsl).replace("\\", "/")
            if args.output_task in segs and not args.overwrite:
                print(f"    task exists (use --overwrite); wrote {out_file}")
            else:
                segs[args.output_task] = _reg(rel, "combined", args, base_refs, inclusive)
                changed += 1
                print(f"    registered COMBINED '{args.output_task}' -> {out_file}")
        else:
            written = 0
            for (module, structure, img) in results:
                sitk.WriteImage(img, str(out_dir / f"{roi_name(module, structure)}.nii.gz"))
                written += 1
            rel = rel_to_segroot(str(out_dir), seg_root_wsl).replace("\\", "/")
            if args.output_task in segs and not args.overwrite:
                print(f"    task exists (use --overwrite); wrote {written} files to {out_dir}")
            else:
                segs[args.output_task] = _reg(rel, "individual", args, base_refs, inclusive)
                changed += 1
                print(f"    registered INDIVIDUAL '{args.output_task}' ({written} files) -> {out_dir}")

    if not args.dry_run and changed:
        save_yaml(pair_doc, pairs_wsl)
        print(f"\nUpdated PairIndex: {pairs_wsl}  ({changed} task(s) registered)")
    else:
        print(f"\nDone. processed={processed}, registered={changed}"
              + (" (dry-run)" if args.dry_run else ""))


def _warn_resample(mask, ref, label):
    print(f"    WARN: grid mismatch for '{label}' — NN-resampling onto base grid (boundary drift possible)")
    r = sitk.ResampleImageFilter()
    r.SetReferenceImage(ref)
    r.SetInterpolator(sitk.sitkNearestNeighbor)
    r.SetDefaultPixelValue(0)
    return r.Execute(mask)


def _reg(rel, seg_type, args, base_refs, inclusive):
    d = {
        "status": "done",
        "source": "slice_cut",
        "registered_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "seg_type": seg_type,
        "base": [f"{m}/{s or 'ALL'}" for (m, s) in base_refs],
        "inclusive": inclusive,
    }
    if seg_type == "combined":
        d["combined_path"] = {"seg_path_rel": rel}
        d["union"] = True
    else:
        d["organ_dir"] = {"dir_path_rel": rel}
    if args.between:
        d["mode"] = "between"
        d["between"] = list(args.between)
        d["between_points"] = list(args.between_points)
    else:
        d["mode"] = "remove"
        d["ref"] = args.ref
        d["ref_point"] = args.ref_point
        d["remove"] = args.remove
    return d


if __name__ == "__main__":
    main()
