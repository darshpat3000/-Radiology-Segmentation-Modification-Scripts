#!/usr/bin/env python3
"""
CLI_TMJ_FromPairIndex.py — Build temporomandibular joint (TMJ) ROIs from
craniofacial_structures (mandible + skull), within the PairIndex framework.

Recipe (per the degenerative-disease design):
  1. Take the mandible, keep only its SUPERIOR portion (the condyle) — the top
     --condyle-mm millimetres in world Z.
  2. DILATE the condyle by --dilate-mm (bridges the joint space, reaches the
     fossa surface).
  3. INTERSECT the dilated condyle with the skull -> the fossa / temporal-bone
     piece adjacent to the condyle (since there is no separate temporal label).
  4. UNION (dilated condyle) with (adjacent skull piece) -> a solid ROI covering
     the condyle, the joint space, and the fossa.
  5. SPLIT into LEFT and RIGHT by x-position (two well-separated condyles).

Outputs (default): per-side files in a sibling folder in the CT seg dir:
    <CT seg dir>/<output-task>/<output-task>__left.nii.gz
    <CT seg dir>/<output-task>/<output-task>__right.nii.gz
registered as an INDIVIDUAL task — so slicer_metrics_worker names the ROIs
"<output-task>__left" / "__right" and you get separate L/R SUV columns.

Left/right naming follows RADIOLOGICAL convention by default: patient-left is on
the image's higher-x side for LPS/RAS — we resolve it from the image direction so
it is anatomically correct rather than array-order dependent. Use
--no-anatomical-lr to fall back to raw array x split.

Usage:
  python3 CLI_TMJ_FromPairIndex.py \
    --pairs   ".../PairIndex.yaml" \
    --seg-root "/Volumes/T7/.../SegmentationsRoot" \
    --mandible "craniofacial_structures/mandible" \
    --skull    "craniofacial_structures/skull" \
    --condyle-mm 25 --dilate-mm 8 \
    --output-task "TMJ" \
    --parents "1/Fdg180" -v
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml
import numpy as np
import SimpleITK as sitk

from platform_paths import to_wsl

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
                k = (parent_rel, pet, ct)
                if k in seen:
                    continue
                seen.add(k)
                yield parent_rel, entry


def resolve_seg_path(task_rec, seg_root_wsl):
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
    for c in ("/", ":"):
        if c in s:
            sep = c
            break
    if sep is None:
        return s, ""
    m, st = s.split(sep, 1)
    return m.strip(), st.strip()


def find_task(entry, module_name):
    return (entry.get("segmentations") or {}).get(module_name)

def read_image(p):
    return sitk.ReadImage(p)

def binarize(img, label=None):
    arr = sitk.GetArrayFromImage(img)
    b = (arr != 0) if label is None else (arr == label)
    o = sitk.GetImageFromArray(b.astype(np.uint8)); o.CopyInformation(img)
    return o


def list_module_structures(task_name, task_rec, path, seg_type):
    out = []
    if seg_type == "combined":
        lm = module_label_map(task_name, task_rec)
        arr = sitk.GetArrayFromImage(read_image(path))
        for lab in sorted(int(x) for x in np.unique(arr) if x != 0):
            out.append((lm.get(lab, f"label_{lab}"), "label", lab))
    else:
        for f in sorted(Path(path).glob("*.nii*")):
            stem = f.name
            for ext in (".nii.gz", ".nii"):
                if stem.endswith(ext):
                    stem = stem[:-len(ext)]; break
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
    m = []
    for (name, kind, res) in structures:
        if as_int is not None and kind == "label" and res == as_int:
            m.append((name, kind, res))
        elif name.lower() == q.lower():
            m.append((name, kind, res))
    if m:
        return m
    return [t for t in structures if q.lower() in t[0].lower()]

def resolve_ref_binary(entry, ref, seg_root_wsl):
    """Resolve one module/structure ref to a single binary image (union of
    matches) on its own grid, plus the module's label map for reference."""
    module_name, query = ref
    rec = find_task(entry, module_name)
    if not rec:
        return None
    path, seg_type = resolve_seg_path(rec, seg_root_wsl)
    if not path:
        return None
    structs = list_module_structures(module_name, rec, path, seg_type)
    chosen = match_structure(structs, query)
    if not chosen:
        return None
    union = None
    for (name, kind, res) in chosen:
        b = binarize(read_image(path), label=res) if kind == "label" else binarize(read_image(res))
        union = b if union is None else sitk.Or(union, b)
    return union


# -------- geometry helpers --------

def grids_match(a, b):
    return (a.GetSize()==b.GetSize() and a.GetSpacing()==b.GetSpacing()
            and a.GetOrigin()==b.GetOrigin() and a.GetDirection()==b.GetDirection())

def resample_to(mask, ref):
    if grids_match(mask, ref):
        return mask
    r = sitk.ResampleImageFilter(); r.SetReferenceImage(ref)
    r.SetInterpolator(sitk.sitkNearestNeighbor); r.SetDefaultPixelValue(0)
    return r.Execute(mask)

def z_points_superior(img):
    return img.GetDirection()[8] >= 0

def keep_superior_mm(mask, mm):
    """Keep only the top `mm` (world-superior) of a binary mask."""
    arr = sitk.GetArrayFromImage(mask)  # [k,j,i]
    ks = np.where(arr.any(axis=(1,2)))[0]
    if ks.size == 0:
        return mask
    sp_z = mask.GetSpacing()[2]
    n_slices = max(1, int(round(mm / sp_z)))
    sup = z_points_superior(img=mask)
    kmin, kmax = int(ks.min()), int(ks.max())
    keep = np.zeros(arr.shape[0], dtype=bool)
    if sup:
        # superior = high k; keep top n_slices down from kmax
        lo = max(kmin, kmax - n_slices + 1)
        keep[lo:kmax+1] = True
    else:
        # superior = low k; keep bottom n_slices up from kmin
        hi = min(kmax, kmin + n_slices - 1)
        keep[kmin:hi+1] = True
    out = arr.copy()
    out[~keep, :, :] = 0
    o = sitk.GetImageFromArray(out.astype(np.uint8)); o.CopyInformation(mask)
    return o

def dilate_mm(mask, mm):
    if mm <= 0:
        return mask
    d = sitk.SignedMaurerDistanceMap(mask, insideIsPositive=False,
                                     squaredDistance=False, useImageSpacing=True)
    out = sitk.BinaryThreshold(d, -1e12, float(mm), 1, 0)
    return sitk.Cast(out, sitk.sitkUInt8)

def split_left_right(mask, anatomical=True):
    """Split a binary mask into (left_img, right_img) by x.
    anatomical=True: patient-left vs patient-right using image direction so it is
    anatomically correct. Returns two sitk images on the same grid."""
    arr = sitk.GetArrayFromImage(mask)  # [k,j,i] -> i is x index
    nx = arr.shape[2]
    # world-x direction sign for +i is direction element [0]
    xdir = mask.GetDirection()[0]
    # centroid split at the mask's x center of mass (robust to off-center heads)
    xs = np.where(arr.any(axis=(0,1)))[0]
    if xs.size == 0:
        z = np.zeros_like(arr)
        a = sitk.GetImageFromArray(z); a.CopyInformation(mask)
        b = sitk.GetImageFromArray(z.copy()); b.CopyInformation(mask)
        return a, b
    xsplit = int(round((xs.min() + xs.max()) / 2.0))
    low = arr.copy();  low[:, :, xsplit:] = 0     # lower i
    high = arr.copy(); high[:, :, :xsplit] = 0    # higher i
    # Determine which side (low-i / high-i) is patient-RIGHT.
    # In LPS, +x = patient-left. If +i maps to +x (xdir>0), higher i = patient-left.
    if anatomical and xdir != 0:
        if xdir > 0:
            left_arr, right_arr = high, low      # +i is patient-left
        else:
            left_arr, right_arr = low, high
    else:
        # raw array split: call higher-i "right" arbitrarily
        left_arr, right_arr = low, high
    li = sitk.GetImageFromArray(left_arr.astype(np.uint8));  li.CopyInformation(mask)
    ri = sitk.GetImageFromArray(right_arr.astype(np.uint8)); ri.CopyInformation(mask)
    return li, ri


def sanitize(s):
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(s))

def ct_seg_dir(entry, seg_root_wsl, anchor_module):
    rec = find_task(entry, anchor_module)
    if not rec:
        return None
    path, seg_type = resolve_seg_path(rec, seg_root_wsl)
    if not path:
        return None
    p = Path(path)
    if seg_type == "individual":
        return p.parent
    return p.parent.parent if p.parent.name == anchor_module else p.parent

def rel_to_segroot(path, seg_root_wsl):
    try:
        return str(Path(path).relative_to(Path(seg_root_wsl)))
    except Exception:
        return str(path)


def run_standalone(args):
    """Build TMJ ROIs directly from mandible/skull files, no PairIndex."""
    mand_path = to_wsl(args.mandible_file)
    skull_path = to_wsl(args.skull_file)
    mand = binarize(read_image(mand_path))
    skull = binarize(read_image(skull_path))

    n_mand = int(sitk.GetArrayFromImage(mand).sum())
    print(f"mandible voxels: {n_mand}")
    if n_mand == 0:
        print("ERROR: mandible empty (head out of FOV?)"); return

    skull = resample_to(skull, mand)
    condyle = keep_superior_mm(mand, args.condyle_mm)
    condyle_d = dilate_mm(condyle, args.dilate_mm)
    adj_skull = sitk.And(condyle_d, skull)
    tmj = sitk.Or(condyle_d, adj_skull)

    n_cond = int(sitk.GetArrayFromImage(condyle).sum())
    n_tmj = int(sitk.GetArrayFromImage(tmj).sum())
    n_adj = int(sitk.GetArrayFromImage(adj_skull).sum())
    print(f"condyle {n_cond} vox -> dilated+skull TMJ {n_tmj} vox (adjacent skull {n_adj})")

    out_dir = Path(to_wsl(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.no_split:
        fn = out_dir / f"{sanitize(args.output_task)}.nii.gz"
        sitk.WriteImage(tmj, str(fn))
        print(f"wrote {fn}")
    else:
        left, right = split_left_right(tmj, anatomical=not args.no_anatomical_lr)
        nl = int(sitk.GetArrayFromImage(left).sum()); nr = int(sitk.GetArrayFromImage(right).sum())
        print(f"left {nl} vox | right {nr} vox")
        lf = out_dir / f"{sanitize(args.output_task)}__left.nii.gz"
        rf = out_dir / f"{sanitize(args.output_task)}__right.nii.gz"
        sitk.WriteImage(left, str(lf)); sitk.WriteImage(right, str(rf))
        print(f"wrote {lf}\nwrote {rf}")
    print("\nOpen these over the CT in Slicer to check placement, then tune --condyle-mm / --dilate-mm.")


def main():
    ap = argparse.ArgumentParser(description="Build L/R TMJ ROIs from mandible + skull.")
    ap.add_argument("--pairs")
    ap.add_argument("--seg-root")
    ap.add_argument("--mandible", default="craniofacial_structures/mandible")
    ap.add_argument("--skull", default="craniofacial_structures/skull")
    # standalone mode
    ap.add_argument("--standalone", action="store_true",
                    help="Bypass PairIndex; take mandible/skull files directly.")
    ap.add_argument("--mandible-file", help="(standalone) path to mandible nii.gz")
    ap.add_argument("--skull-file", help="(standalone) path to skull nii.gz")
    ap.add_argument("--out-dir", help="(standalone) output folder for the TMJ ROIs")

    ap.add_argument("--condyle-mm", type=float, default=25.0,
                    help="Superior portion of mandible to keep as the condyle (mm).")
    ap.add_argument("--dilate-mm", type=float, default=8.0,
                    help="Dilate the condyle to bridge the joint and reach the fossa (mm).")
    ap.add_argument("--output-task", default="TMJ")
    ap.add_argument("--no-split", action="store_true", help="Single bilateral ROI (no L/R split).")
    ap.add_argument("--no-anatomical-lr", action="store_true",
                    help="Split by raw array x instead of anatomical patient L/R.")
    ap.add_argument("--parents", default="")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if args.standalone:
        if not (args.mandible_file and args.skull_file and args.out_dir):
            print("ERROR: --standalone needs --mandible-file, --skull-file, --out-dir")
            sys.exit(1)
        run_standalone(args)
        return

    if not (args.pairs and args.seg_root):
        print("ERROR: need --pairs and --seg-root (or use --standalone)")
        sys.exit(1)

    pairs_wsl = to_wsl(args.pairs)
    seg_root_wsl = to_wsl(args.seg_root)
    pair_doc = load_yaml(pairs_wsl)
    parents_filter = set(t.strip() for t in args.parents.split(",") if t.strip()) if args.parents.strip() else None

    mand_ref = parse_ref(args.mandible)
    skull_ref = parse_ref(args.skull)

    changed = 0
    processed = 0

    for parent_rel, entry in iter_pairs(pair_doc, parents_filter):
        if not find_task(entry, mand_ref[0]):
            continue

        mand = resolve_ref_binary(entry, mand_ref, seg_root_wsl)
        skull = resolve_ref_binary(entry, skull_ref, seg_root_wsl)
        if mand is None or skull is None:
            print(f"  SKIP {parent_rel}: mandible or skull not resolved")
            continue
        # sanity: non-empty and in-FOV
        if int(sitk.GetArrayFromImage(mand).sum()) == 0:
            print(f"  SKIP {parent_rel}: mandible empty (head likely out of FOV)")
            continue

        print(f"\n[{parent_rel}]")
        skull = resample_to(skull, mand)

        # 1. condyle = superior portion of mandible
        condyle = keep_superior_mm(mand, args.condyle_mm)
        # 2. dilate condyle
        condyle_d = dilate_mm(condyle, args.dilate_mm)
        # 3. adjacent skull piece = dilated condyle ∩ skull
        adj_skull = sitk.And(condyle_d, skull)
        # 4. union -> solid TMJ ROI
        tmj = sitk.Or(condyle_d, adj_skull)

        n_cond = int(sitk.GetArrayFromImage(condyle).sum())
        n_tmj = int(sitk.GetArrayFromImage(tmj).sum())
        n_adj = int(sitk.GetArrayFromImage(adj_skull).sum())
        print(f"    condyle {n_cond} vox -> dilated+skull TMJ {n_tmj} vox (adjacent skull {n_adj})")

        if args.dry_run:
            processed += 1
            continue

        cdir = ct_seg_dir(entry, seg_root_wsl, mand_ref[0])
        if cdir is None:
            print(f"  SKIP {parent_rel}: could not locate CT seg dir")
            continue
        out_dir = Path(cdir) / args.output_task
        out_dir.mkdir(parents=True, exist_ok=True)

        segs = entry.setdefault("segmentations", {})
        if args.output_task in segs and not args.overwrite:
            print(f"    task exists (use --overwrite)")
            continue

        written = []
        if args.no_split:
            fn = out_dir / f"{sanitize(args.output_task)}.nii.gz"
            sitk.WriteImage(tmj, str(fn)); written.append(fn)
        else:
            left, right = split_left_right(tmj, anatomical=not args.no_anatomical_lr)
            nl = int(sitk.GetArrayFromImage(left).sum()); nr = int(sitk.GetArrayFromImage(right).sum())
            print(f"    left {nl} vox | right {nr} vox")
            lf = out_dir / f"{sanitize(args.output_task)}__left.nii.gz"
            rf = out_dir / f"{sanitize(args.output_task)}__right.nii.gz"
            sitk.WriteImage(left, str(lf)); sitk.WriteImage(right, str(rf))
            written += [lf, rf]

        rel = rel_to_segroot(str(out_dir), seg_root_wsl).replace("\\", "/")
        segs[args.output_task] = {
            "status": "done", "source": "tmj_builder",
            "registered_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "seg_type": "individual",
            "organ_dir": {"dir_path_rel": rel},
            "mandible": args.mandible, "skull": args.skull,
            "condyle_mm": args.condyle_mm, "dilate_mm": args.dilate_mm,
            "split_lr": (not args.no_split),
        }
        changed += 1
        processed += 1
        print(f"    registered '{args.output_task}' ({len(written)} files) -> {out_dir}")

    if not args.dry_run and changed:
        save_yaml(pair_doc, pairs_wsl)
        print(f"\nUpdated PairIndex: {pairs_wsl}  ({changed} task(s))")
    else:
        print(f"\nDone. processed={processed}, registered={changed}" + (" (dry-run)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
