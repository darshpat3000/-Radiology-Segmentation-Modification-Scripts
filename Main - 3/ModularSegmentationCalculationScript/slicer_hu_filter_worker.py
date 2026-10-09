"""
slicer_hu_filter_worker.py — Run INSIDE 3D Slicer to apply HU filtering.

Slicer handles all DICOM↔NIfTI coordinate alignment internally.

Called by CLI_SlicerHUFilter.py:
  Slicer.exe --no-splash --no-main-window --python-script slicer_hu_filter_worker.py config.json
  (Linux) /opt/Slicer-5.8.1-linux-amd64/Slicer --no-splash --no-main-window --python-script slicer_hu_filter_worker.py config.json

config.json format:
{
  "jobs": [
    {
      "parent_rel": ".",
      "source_task": "EAT",
      "output_task": "EAT_fat",
      "ct_dicom_dir": "/path/to/CT  (or C:\\path\\to\\CT on Windows-Slicer)",
      "seg_path": "C:\\path\\to\\seg.nii.gz",
      "out_path": "C:\\path\\to\\output.nii.gz",
      "hu_conditions": [{"type": "range", "min": -200, "max": 0}]
    }
  ]
}

hu_conditions types:
  {"type": "range", "min": -200, "max": 0}     → keep -200 ≤ HU ≤ 0
  {"type": "gte", "value": 130}                  → keep HU ≥ 130
  {"type": "lte", "value": 50}                   → keep HU ≤ 50
  {"type": "gt", "value": 0}                     → keep HU > 0
  {"type": "lt", "value": -500}                  → keep HU < -500
"""

import json
import sys
import os
import traceback

config_path = None
for arg in sys.argv:
    if arg.endswith(".json") and os.path.isfile(arg):
        config_path = arg
        break

if not config_path:
    print("ERROR: pass config.json as argument")
    sys.exit(1)

with open(config_path, "r") as f:
    config = json.load(f)

jobs = config.get("jobs", [])
print(f"[slicer_hu_filter] {len(jobs)} jobs")

import slicer
from DICOMLib import DICOMUtils
import numpy as np


def load_dicom_volume(dicom_dir, db_suffix=""):
    db_path = os.path.join(slicer.app.temporaryPath, f"HUFilterDB{db_suffix}")
    DICOMUtils.openTemporaryDatabase(db_path)
    DICOMUtils.importDicom(dicom_dir)

    db = slicer.dicomDatabase
    all_series = []
    for patient in db.patients():
        for study in db.studiesForPatient(patient):
            for series_uid in db.seriesForStudy(study):
                all_series.append(series_uid)

    if not all_series:
        raise RuntimeError(f"No series found in {dicom_dir}")

    loaded = DICOMUtils.loadSeriesByUID([all_series[0]])
    if not loaded:
        raise RuntimeError(f"Failed to load series from {dicom_dir}")

    node = slicer.mrmlScene.GetNodeByID(loaded[0])
    if node is None:
        raise RuntimeError(f"Loaded node is None for {dicom_dir}")
    return node


def apply_hu_conditions(ct_array, seg_array, conditions):
    """Apply HU conditions to seg array. Returns filtered seg array."""
    mask = np.ones(seg_array.shape, dtype=bool)

    for cond in conditions:
        t = cond["type"]
        if t == "range":
            mask &= (ct_array >= cond["min"]) & (ct_array <= cond["max"])
        elif t == "gte":
            mask &= (ct_array >= cond["value"])
        elif t == "lte":
            mask &= (ct_array <= cond["value"])
        elif t == "gt":
            mask &= (ct_array > cond["value"])
        elif t == "lt":
            mask &= (ct_array < cond["value"])

    filtered = seg_array.copy()
    filtered[~mask] = 0
    return filtered


results = []

for i, job in enumerate(jobs):
    parent_rel = job.get("parent_rel", "")
    source_task = job.get("source_task", "")
    output_task = job.get("output_task", "")
    ct_dir = job.get("ct_dicom_dir", "")
    seg_path = job.get("seg_path", "")
    out_path = job.get("out_path", "")
    conditions = job.get("hu_conditions", [])

    print(f"\n[{i+1}/{len(jobs)}] {parent_rel} / {source_task} → {output_task}")

    result = {"parent_rel": parent_rel, "status": "ok", "error": "",
              "out_path": out_path, "voxels_before": 0, "voxels_after": 0}

    try:
        slicer.mrmlScene.Clear(False)

        # Load CT
        print(f"  CT: {ct_dir}")
        ct_node = load_dicom_volume(ct_dir, db_suffix=f"_ct_{i}")

        # Load seg as labelmap
        print(f"  Seg: {seg_path}")
        seg_node = slicer.util.loadLabelVolume(seg_path)
        if seg_node is None:
            raise RuntimeError(f"Failed to load seg: {seg_path}")

        # Resample seg to CT space (Slicer handles coordinate alignment)
        print("  Resampling seg to CT space...")
        resample_params = {
            "inputVolume": seg_node.GetID(),
            "referenceVolume": ct_node.GetID(),
            "outputVolume": seg_node.GetID(),
            "interpolationMode": "NearestNeighbor",
        }
        slicer.cli.runSync(slicer.modules.brainsresample, None, resample_params)

        # Get arrays
        ct_array = slicer.util.arrayFromVolume(ct_node)
        seg_array = slicer.util.arrayFromVolume(seg_node)

        voxels_before = int(np.count_nonzero(seg_array > 0))
        result["voxels_before"] = voxels_before
        print(f"  Voxels before: {voxels_before}")

        # Apply HU conditions
        print(f"  Applying {len(conditions)} HU condition(s)...")
        filtered = apply_hu_conditions(ct_array, seg_array, conditions)

        voxels_after = int(np.count_nonzero(filtered > 0))
        result["voxels_after"] = voxels_after
        pct = round(100.0 * voxels_after / voxels_before, 1) if voxels_before > 0 else 0
        print(f"  Voxels after: {voxels_after} ({pct}% retained)")

        # Write filtered seg.
        # exFAT/FAT volumes break Slicer's write-then-rename save (the hidden
        # macOS "._" AppleDouble file makes rename() return -1). So we save to a
        # temp file on the internal (APFS) drive, then COPY it to the real
        # destination — copy works on exFAT, only atomic rename fails.
        slicer.util.updateVolumeFromArray(seg_node, filtered)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        import tempfile, shutil
        tmp_dir = tempfile.mkdtemp(prefix="hufilt_")
        tmp_out = os.path.join(tmp_dir, os.path.basename(out_path))
        save_ok = slicer.util.saveNode(seg_node, tmp_out)

        if not save_ok or not os.path.isfile(tmp_out):
            raise RuntimeError(f"saveNode failed to write temp file {tmp_out}")

        # copy temp -> final destination (works on exFAT)
        shutil.copyfile(tmp_out, out_path)
        # clean up any stray AppleDouble sibling that may already exist
        ad = os.path.join(os.path.dirname(out_path), "._" + os.path.basename(out_path))
        try:
            if os.path.exists(ad):
                os.remove(ad)
        except Exception:
            pass
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

        if not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
            raise RuntimeError(f"final file missing/empty after copy: {out_path}")
        print(f"  Wrote: {out_path} ({os.path.getsize(out_path)} bytes)")

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)[:500]
        print(f"  ERROR: {e}")
        traceback.print_exc()

    results.append(result)

# Write summary
summary_path = config_path.replace(".json", "_results.json")
with open(summary_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n[done] {len(results)} jobs, summary: {summary_path}")

ok = sum(1 for r in results if r["status"] == "ok")
err = len(results) - ok
print(f"  {ok} ok, {err} errors")

sys.exit(0)
