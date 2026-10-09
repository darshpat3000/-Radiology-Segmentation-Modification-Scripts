"""
slicer_metrics_worker.py — Run INSIDE 3D Slicer to compute HU/SUV/Volume.

Slicer handles all DICOM coordinate transforms, SUV conversion, and
seg↔volume alignment internally. Numbers match what you see in the GUI.

Called by CLI_SlicerMetrics.py or manually:
  Slicer.exe --no-splash --no-main-window --python-script slicer_metrics_worker.py config.json
  (Linux) /opt/Slicer-5.8.1-linux-amd64/Slicer --no-splash --no-main-window --python-script slicer_metrics_worker.py config.json

config.json format:
{
  "jobs": [
    {
      "parent_rel": ".",
      "task": "EAT",
      "ct_dicom_dir": "/path/to/CT  (or C:\\path\\to\\CT on Windows-Slicer)",
      "pet_dicom_dir": "C:\\path\\to\\PET",
      "seg_path": "C:\\path\\to\\seg.nii.gz",
      "suv_factor": 0.00021
    }
  ],
  "out_csv": "C:\\path\\to\\output.csv"
}
"""

import json
import sys
import os
import csv
import traceback

# ---------------------------------------------------------------------------
# Find config
# ---------------------------------------------------------------------------

config_path = None
for arg in sys.argv:
    if arg.endswith(".json") and os.path.isfile(arg):
        config_path = arg
        break

if not config_path:
    print("ERROR: pass config.json as argument")
    print("Usage: Slicer.exe --no-splash --no-main-window --python-script slicer_metrics_worker.py config.json")
    sys.exit(1)

with open(config_path, "r") as f:
    config = json.load(f)

jobs = config.get("jobs", [])
out_csv = config.get("out_csv", "slicer_metrics.csv")
include_raw_pet = config.get("include_raw_pet", False)
label_maps = config.get("label_maps", {})  # {task_name: {"1": "organ_name", ...}}

print(f"[slicer_metrics] {len(jobs)} jobs, output: {out_csv}, raw_pet: {include_raw_pet}")
if label_maps:
    print(f"  Label maps: {list(label_maps.keys())}")

# ---------------------------------------------------------------------------
# Slicer imports
# ---------------------------------------------------------------------------

import slicer
from DICOMLib import DICOMUtils
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_dicom_volume(dicom_dir, db_suffix=""):
    """Import a DICOM directory and load the first series as a volume node."""
    db_path = os.path.join(slicer.app.temporaryPath, f"MetricsDB{db_suffix}")
    DICOMUtils.openTemporaryDatabase(db_path)
    DICOMUtils.importDicom(dicom_dir)

    db = slicer.dicomDatabase
    patients = db.patients()
    if not patients:
        raise RuntimeError(f"No patients found after importing {dicom_dir}")

    # Collect all series
    all_series = []
    for patient in patients:
        for study in db.studiesForPatient(patient):
            for series_uid in db.seriesForStudy(study):
                all_series.append(series_uid)

    if not all_series:
        raise RuntimeError(f"No series found in {dicom_dir}")

    # Load first series
    loaded = DICOMUtils.loadSeriesByUID([all_series[0]])
    if not loaded:
        raise RuntimeError(f"Failed to load series {all_series[0]} from {dicom_dir}")

    node = slicer.mrmlScene.GetNodeByID(loaded[0])
    if node is None:
        raise RuntimeError(f"Loaded node is None for {dicom_dir}")

    return node


def load_seg_as_segmentation(seg_path, ref_volume_node):
    """Load a NIfTI labelmap and convert to segmentation node."""
    # Try loading as segmentation directly first
    try:
        seg_node = slicer.util.loadSegmentation(seg_path)
        if seg_node and seg_node.GetSegmentation().GetNumberOfSegments() > 0:
            return seg_node
    except Exception:
        pass

    # Fallback: load as labelmap, convert to segmentation
    labelmap_node = slicer.util.loadLabelVolume(seg_path)
    if labelmap_node is None:
        raise RuntimeError(f"Failed to load seg: {seg_path}")

    seg_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
        labelmap_node, seg_node
    )
    slicer.mrmlScene.RemoveNode(labelmap_node)

    if seg_node.GetSegmentation().GetNumberOfSegments() == 0:
        raise RuntimeError(f"No segments found in {seg_path}")

    return seg_node


def compute_segment_stats(seg_node, volume_node):
    """Run SegmentStatistics and return dict of {segment_name: {stat: value}}."""
    import SegmentStatistics

    stat_logic = SegmentStatistics.SegmentStatisticsLogic()
    stat_logic.getParameterNode().SetParameter("Segmentation", seg_node.GetID())
    stat_logic.getParameterNode().SetParameter("ScalarVolume", volume_node.GetID())
    stat_logic.getParameterNode().SetParameter("LabelmapSegmentStatisticsPlugin.enabled", "True")
    stat_logic.getParameterNode().SetParameter("ScalarVolumeSegmentStatisticsPlugin.enabled", "True")
    stat_logic.computeStatistics()

    stats = stat_logic.getStatistics()
    results = {}

    segmentation = seg_node.GetSegmentation()
    for seg_idx in range(segmentation.GetNumberOfSegments()):
        segment_id = segmentation.GetNthSegmentID(seg_idx)
        segment = segmentation.GetSegment(segment_id)
        seg_name = segment.GetName()

        seg_stats = {}
        for key in stats.keys():
            # Stats are keyed as tuples: (segment_id, stat_name)
            if isinstance(key, tuple) and len(key) == 2 and key[0] == segment_id:
                seg_stats[key[1]] = stats[key]
            # Also check string keys: "segment_id.stat_name"
            elif isinstance(key, str) and key.startswith(segment_id + "."):
                stat_name = key[len(segment_id) + 1:]
                seg_stats[stat_name] = stats[key]

        results[seg_name] = seg_stats

    return results


# ---------------------------------------------------------------------------
# Process jobs
# ---------------------------------------------------------------------------

all_rows = []
all_roi_names = []

for i, job in enumerate(jobs):
    parent_rel = job.get("parent_rel", "")
    task = job.get("task", "")
    ct_dir = job.get("ct_dicom_dir", "")
    pet_dir = job.get("pet_dicom_dir", "")
    seg_path = job.get("seg_path", "")
    suv_factor = job.get("suv_factor")

    print(f"\n[{i+1}/{len(jobs)}] {parent_rel} / {task}")

    row = {
        "parent_rel": parent_rel,
        "task": task,
        "status": "ok",
        "error": "",
    }

    try:
        # Clear scene between jobs
        slicer.mrmlScene.Clear(False)

        # ---- Load CT ----
        print(f"  Loading CT: {ct_dir}")
        ct_node = load_dicom_volume(ct_dir, db_suffix="_ct")
        row["ct_ok"] = "yes"

        # ---- Load PET ----
        pet_node = None
        if pet_dir and os.path.isdir(pet_dir):
            try:
                print(f"  Loading PET: {pet_dir}")
                pet_node = load_dicom_volume(pet_dir, db_suffix="_pet")
                row["pet_ok"] = "yes"
            except Exception as e:
                row["pet_ok"] = f"failed: {str(e)[:200]}"
                print(f"  PET load failed: {e}")
        else:
            row["pet_ok"] = "missing"

        # ---- Load Seg ----
        print(f"  Loading seg: {seg_path}")
        if os.path.isdir(seg_path):
            seg_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
            nii_files = sorted([os.path.join(seg_path, f) for f in os.listdir(seg_path)
                                if f.endswith(".nii") or f.endswith(".nii.gz")])
            if not nii_files:
                raise RuntimeError(f"No NIfTI files found in {seg_path}")
            for nii_file in nii_files:
                organ_name = os.path.basename(nii_file).replace(".nii.gz", "").replace(".nii", "")
                labelmap_node = slicer.util.loadLabelVolume(nii_file)

                if labelmap_node is None:
                    print(f"Failed to load {nii_file}")
                    continue

                n_before = seg_node.GetSegmentation().GetNumberOfSegments()
                slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelmap_node, seg_node)

                n_after = seg_node.GetSegmentation().GetNumberOfSegments()
                for seg_idx in range(n_before, n_after):
                    seg_id = seg_node.GetSegmentation().GetNthSegmentID(seg_idx)
                    seg_node.GetSegmentation().GetSegment(seg_id).SetName(organ_name)
                slicer.mrmlScene.RemoveNode(labelmap_node)
        else:
            seg_node = load_seg_as_segmentation(seg_path, ct_node)

        # Link seg reference geometry to CT so SegmentStatistics can compute overlap
        seg_node.SetReferenceImageGeometryParameterFromVolumeNode(ct_node)

        # Rename segments using label map if available
        task_lm = label_maps.get(task, {})
        if task_lm:
            segmentation = seg_node.GetSegmentation()
            for seg_idx in range(segmentation.GetNumberOfSegments()):
                segment_id = segmentation.GetNthSegmentID(seg_idx)
                # Segment_N → label index N
                idx_str = segment_id.replace("Segment_", "")
                if idx_str in task_lm:
                    segmentation.GetSegment(segment_id).SetName(task_lm[idx_str])

        n_segments = seg_node.GetSegmentation().GetNumberOfSegments()
        row["seg_ok"] = "yes"
        row["n_segments"] = n_segments
        seg_names = [seg_node.GetSegmentation().GetSegment(
            seg_node.GetSegmentation().GetNthSegmentID(i)).GetName()
            for i in range(n_segments)]
        print(f"  {n_segments} segments: {seg_names[:5]}{'...' if n_segments > 5 else ''}")

        # ---- HU stats (CT) ----
        print("  Computing HU stats...")
        ct_stats = compute_segment_stats(seg_node, ct_node)

        # ---- SUV stats (PET) ----
        pet_stats = {}
        if pet_node is not None:
            print("  Computing PET stats...")
            pet_stats = compute_segment_stats(seg_node, pet_node)

        # ---- Build ROI columns ----
        for seg_name, cs in ct_stats.items():
            if seg_name not in all_roi_names:
                all_roi_names.append(seg_name)

            vox_count = cs.get("LabelmapSegmentStatisticsPlugin.voxel_count", "")
            vol_mm3 = cs.get("LabelmapSegmentStatisticsPlugin.volume_mm3", "")
            vol_ml = float(vol_mm3) / 1000.0 if vol_mm3 != "" else ""

            row[f"{seg_name}_HU_mean"] = cs.get("ScalarVolumeSegmentStatisticsPlugin.mean", "")
            row[f"{seg_name}_HU_median"] = cs.get("ScalarVolumeSegmentStatisticsPlugin.median", "")
            row[f"{seg_name}_HU_min"] = cs.get("ScalarVolumeSegmentStatisticsPlugin.min", "")
            row[f"{seg_name}_HU_max"] = cs.get("ScalarVolumeSegmentStatisticsPlugin.max", "")
            row[f"{seg_name}_VOL_mL"] = vol_ml
            row[f"{seg_name}_CT_vox"] = vox_count

            # PET / SUV
            ps = pet_stats.get(seg_name, {})
            pet_mean = ps.get("ScalarVolumeSegmentStatisticsPlugin.mean", "")
            pet_median = ps.get("ScalarVolumeSegmentStatisticsPlugin.median", "")
            pet_min = ps.get("ScalarVolumeSegmentStatisticsPlugin.min", "")
            pet_max = ps.get("ScalarVolumeSegmentStatisticsPlugin.max", "")

            # SUV columns — always present
            if pet_node is None:
                row[f"{seg_name}_SUV_mean"] = "no PET"
                row[f"{seg_name}_SUV_median"] = "no PET"
                row[f"{seg_name}_SUV_min"] = "no PET"
                row[f"{seg_name}_SUV_max"] = "no PET"
            elif not suv_factor:
                row[f"{seg_name}_SUV_mean"] = "no suv factor"
                row[f"{seg_name}_SUV_median"] = "no suv factor"
                row[f"{seg_name}_SUV_min"] = "no suv factor"
                row[f"{seg_name}_SUV_max"] = "no suv factor"
            elif pet_mean != "":
                row[f"{seg_name}_SUV_mean"] = float(pet_mean) * float(suv_factor)
                row[f"{seg_name}_SUV_median"] = float(pet_median) * float(suv_factor)
                row[f"{seg_name}_SUV_min"] = float(pet_min) * float(suv_factor)
                row[f"{seg_name}_SUV_max"] = float(pet_max) * float(suv_factor)
            else:
                row[f"{seg_name}_SUV_mean"] = "PET stats failed"
                row[f"{seg_name}_SUV_median"] = "PET stats failed"
                row[f"{seg_name}_SUV_min"] = "PET stats failed"
                row[f"{seg_name}_SUV_max"] = "PET stats failed"

            # Raw PET columns — only if requested
            if include_raw_pet and pet_mean != "":
                row[f"{seg_name}_PET_mean"] = pet_mean
                row[f"{seg_name}_PET_median"] = pet_median
                row[f"{seg_name}_PET_min"] = pet_min
                row[f"{seg_name}_PET_max"] = pet_max

        print(f"  Done: {len(ct_stats)} ROIs")

    except Exception as e:
        row["status"] = "error"
        row["error"] = str(e)[:500]
        print(f"  ERROR: {e}")
        traceback.print_exc()

    all_rows.append(row)


# ---------------------------------------------------------------------------
# Write CSV
# ---------------------------------------------------------------------------

# Build column order
fixed_cols = ["parent_rel", "task", "status", "error", "ct_ok", "pet_ok", "seg_ok", "n_segments"]
roi_cols = []
for roi in all_roi_names:
    roi_cols.extend([
        f"{roi}_HU_mean", f"{roi}_HU_median", f"{roi}_HU_min", f"{roi}_HU_max",
        f"{roi}_SUV_mean", f"{roi}_SUV_median", f"{roi}_SUV_min", f"{roi}_SUV_max",
    ])
    if include_raw_pet:
        roi_cols.extend([
            f"{roi}_PET_mean", f"{roi}_PET_median", f"{roi}_PET_min", f"{roi}_PET_max",
        ])
    roi_cols.extend([
        f"{roi}_VOL_mL", f"{roi}_CT_vox",
    ])

all_cols = fixed_cols + roi_cols

os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=all_cols, extrasaction="ignore")
    w.writeheader()
    for row in all_rows:
        w.writerow(row)

print(f"\n[done] Wrote {out_csv} ({len(all_rows)} rows, {len(all_roi_names)} ROIs)")
sys.exit(0)
