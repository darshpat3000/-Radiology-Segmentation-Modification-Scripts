# PET/CT Segmentation & Metrics Pipeline

This script automates the full workflow for quantitative PET/CT analysis on large DICOM datasets:

- **Discovers** all DICOM series under a root directory without manual file organisation
- **Pairs** each PET series to its corresponding CT using DICOM metadata (StudyInstanceUID, acquisition time, FrameOfReference)
- **Segments** paired CTs using [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) to produce organ masks
- **Extracts** per-organ HU (Hounsfield Unit), SUV (Standardised Uptake Value), and volume measurements using 3D Slicer

## Basic Functionality
### 1. Asset Discovery
`CLI_AssetDiscovery.py` uses `os.walk` to walk through the DICOM root directory and single out leaf series, which are the deepest folders that contain DICOM files.
DICOM files are verified by checking filenames and the DICM signature at byte offset 128, which is in every DICOM file. The leaf-detection functionality ensures that
if a folder contains subfolders that contain DICOM series, only the deepest subfolders are catalogued, avoiding double-counting.

Segmentation files (`.nii.gz`, `.nrrd`, `.mha`) and NIfTI exports are discovered in parallel from configurable roots.

Outputs a flat catalog called `AssetsIndex.yaml`, which contains every asset with root-relative paths and file statistics.

### 2. Header Extraction
`CLI_HeaderExtract.py` reads one representative DICOM file from each series directory and extracts all the metadata needed for pairing. For each series, the following
metadata may be extracted:

**Modality** ~ from `Modality` tag or SOPClassUID fallback, identifies what type of imaging was used to acquire the series (CT, PET, etc.)\
**Acquisition datetime** ~ parsed from `AcquisitionDateTime`, `AcquisitionDate`d+`AcquisitionTime`, or `SeriesDate`+`SeriesTime`, with full timezone handling\
**Gating status** ~ pattern matching on `SeriesDescription`, `ProtocolName`, and `ImageType`, checks for cardiac or respiratory gating\
**Contrast status (CT)** ~ checks `ContrastBolusAgent` tag first, then text patterns in series description\
**AC/NAC status (PET)** ~ checks `CorrectedImage` DICOM tag first, then text tokens\
**Tracer (PET)** ~ normalised from `RadiopharmaceuticalInformationSequence`, recognizing FDG, NaF, PSMA, DOTATATE, FLT, Choline\
**Uptake time (PET)** ~ computed here as a floating-point number of minutes between injection and acquisition:

$$t_{{uptake}} = t_{{acquisition}} - t_{{injection}}$$

**Whole-body classification** ~ estimated from slice count (≥150 slices = whole-body) or series description text

Outputs `HeaderIndex.yaml`, which contains a list of dicom series with metadata ready for pairing.

### 3. PET/CT Pairing
`CLI_PairFromHeaders.py` matches each PET series to its best CT using a two-method cascade, with configurable preference scoring. Method 1 pairs PET/CT with `StudyInstanceUID`, 
which matches when a PET and a CT scan are acquired in the same imaging session. This is given top priority, but it only works with around ~90% of clinical datasets. Method 2 matches
using the time difference between CT and PET acquisition datetimes.

$$|t_{CT} - t_{PET}| \leq \Delta t_{window}$$

Default $\Delta t_{window} = 15$ minutes. FrameOfReferenceUID match (happens if PET/CT are spatially registered to the same coordinate system by the scanner) gives a soft bonus of −10 minutes 
to the effective time distance, preferring spatially co-registered pairs without hard-excluding others.

When multiple CTs qualify as candidates for a given PET, the code picks the best one using a 3-tuple score:

$$
score =
(
\mathrm{contrast\_rank},
\mathrm{gated\_rank},
\mathrm{wb\_rank}
)
$$

| Dimension | 0 (preferred) | 1 | 2 |
|---|---|---|---|
| `contrast_rank` | Non-contrast | Unknown | Contrast-enhanced |
| `gated_rank` | Non-gated | Gated | — |
| `wb_rank` | Whole-body FOV | Limited FOV | — |

After candidate pairs are found, a policy filter is applied:

1. If `prefer_non_gated=True` (default), drop gated pairs when non-gated alternatives exist
2. Prefer AC-corrected PET over NAC; NAC over unknown; optionally allow unknown with `--allow-unknown-ac`
3. `--use-nac-only` inverts step 2 to keep only NAC pairs (e.g. for NAC-specific analyses)
4. `--exclude-contrast-ct` hard-excludes contrast CTs before pairing

After policy filtering, if exactly one pair remains it's selected automatically. If multiple pairs survive, the one with the lowest combined score (best method + best CT quality + smallest time gap) is chosen.
If no pairs survive after all filtering, the patient folder goes into the `needs_selection` list.

Outputs `PairIndex.yaml` and a `needs_selection` list for cases requiring manual review.

### 4. Segmentation
`CLI_TotalSegmentatorFromPairIndex.py` reads pairs from `PairIndex.yaml` and series path from `HeaderIndex.yaml`, then runs [TotalSegmentator](https://github.com/wasserth/TotalSegmentator)
on each paired CT. Segmentation results are written back into `PairIndex.yaml`.

The output directory resembles something like `<seg_out_root>/<ct_series_rel>/<task_name>`. If a segmentation record already exists in `PairIndex.yaml` for a given pair and task, 
the code reuses the previously recorded output directory rather than computing a new one.

TotalSegmentator is called as a subprocess in the following manner:

```bash
```TotalSegmentator -i <ct_dicom_dir> -o <output_dir> -ta <task>```
````

**Supported output modes:**
- `individual` ~ one NIfTI file per organ (default TotalSegmentator behavior)
- `combined` ~ single multilabel NIfTI (use `--ml` flag)

**Resumable:** `--skip-existing` checks for completed NIfTI outputs before running, so interrupted jobs can be resumed without reprocessing.

**Segmentation records** written into PairIndex follow this schema:

```yaml
segmentations:
  total:
    source_ct_asset_id: "DICOM_SERIES:..."
    seg_type: "individual"         # or "combined"
    organ_dir:
      seg_root_label: "seg_out"
      dir_path_rel: "PatientA/CT_20240101/total"
      organ_files: ["liver.nii.gz", "spleen.nii.gz", ...]
    ts_version: "2.0.0"
    created_at: "2025-01-01T12:00:00"
    status: "done"
```

### 5. Metric
This stage is split across two Python scripts (`CLI_SlicerMetrics.py` and `CLI_SlicerHUFilter.py`) and two worker scripts that run inside Slicer 
(`slicer_metrics_worker.py` and `slicer_hu_filter_worker.py`). 3D Slicer has its own embedded Python environment, allowing for easy resampling into
the proper voxel spaces.

## Key Formulas

### Decay Correction

$$D_{\text{effective}} = D_{\text{injected}} \cdot e^{-\lambda \cdot \Delta t}, \quad \lambda = \frac{\ln 2}{T_{1/2}}$$

### SUV Body Weight

$$\text{SUV}_{\text{bw}} = \frac{C_{\text{tissue}} \cdot W_{\text{patient}}\,[\text{g}]}{D_{\text{effective}}\,[\text{Bq}]}$$

### Time Window Pairing

$$\text{paired} \iff |\,t_{\text{CT}} - t_{\text{PET}}\,| \leq \Delta t_{\text{window}}$$

With FrameOfReference bonus:

$$d_{\text{effective}} = |\,t_{\text{CT}} - t_{\text{PET}}\,| - 10 \cdot \mathbb{1}[\text{FoR}_{\text{CT}} = \text{FoR}_{\text{PET}}]$$

### Whole-Body Classification

$$\text{is_whole_body} = \begin{cases} \text{True} & \text{if } N_{\text{slices}} \geq 150 \\ \text{text match} & \text{if slice count unavailable} \end{cases}$$

## Installation

### Requirements

| Component | Version | Notes |
|---|---|---|
| Python | ≥ 3.9 | System or conda |
| TotalSegmentator | ≥ 2.0 | Separate install |
| 3D Slicer | ≥ 5.6 | Windows `.exe` called from WSL/Linux |
| CUDA | ≥ 11.8 | Optional but strongly recommended |
| RAM | ≥ 16 GB | 32 GB recommended for large datasets |
| GPU VRAM | ≥ 8 GB | For TotalSegmentator fast mode |

### Python Dependencies

```bash
pip install pydicom pyyaml  numpy  torch  psutil  SimpleITK  nibabel  totalsegmentator
```
## Usage

All commands run through the orchestrator. Replace paths with your own.

### Run the Full Pipeline

```bash
python CLI_PipelineOrchestrator.py all \
  --dicom-root "/data/DICOM" \
  --tasks "total" \
  --slicer-tasks "total" \
  --slicer-exe "/mnt/c/Users/you/AppData/Local/slicer.org/Slicer 5.8.1/Slicer.exe" \
  --slicer-out-csv "/data/Metrics/results.csv" \
  --skip-existing
```

### Step-by-Step

**1. Pair only (run repeatedly until satisfied):**
```bash
python CLI_PipelineOrchestrator.py pair \
  --dicom-root "/data/DICOM" \
  --write-selection-template \
  --init-overrides
```

**2. Review `SelectionTemplate.yaml`, edit `Overrides.yaml`, re-run pair:**
```bash
python CLI_PipelineOrchestrator.py pair \
  --dicom-root "/data/DICOM" \
  --use-overrides \
  --skip-discovery \
  --skip-header
```

**3. Segment (after pairing is settled):**
```bash
python CLI_PipelineOrchestrator.py segment \
  --dicom-root "/data/DICOM" \
  --tasks "total,lung_vessels" \
  --skip-existing \
  --ml
```

**4. Compute metrics:**
```bash
python CLI_PipelineOrchestrator.py slicer-metrics \
  --dicom-root "/data/DICOM" \
  --slicer-tasks "total" \
  --slicer-exe "C:\path\to\Slicer.exe" \
  --slicer-out-csv "/data/Metrics/results.csv"
```

**5. HU-filter a segmentation (e.g. extract epicardial fat):**
```bash
python CLI_PipelineOrchestrator.py slicer-hu-filter \
  --dicom-root "/data/DICOM" \
  --source-task "total" \
  --output-task "EAT_fat" \
  --hu "[-200,-30]" \
  --slicer-exe "C:\path\to\Slicer.exe"
```

### Common Options

| Flag | Description |
|---|---|
| `--dry-run` | Print commands without executing |
| `--skip-existing` | Skip completed segmentations |
| `--ml` | Multilabel NIfTI output (single file per task) |
| `--all-pairs` | Generate all PET↔CT pairs per patient (multi-timepoint) |
| `--allow-ct-reuse` | Allow same CT to pair with multiple PETs |
| `--use-nac-only` | Keep only non-attenuation-corrected PET |
| `--exclude-contrast-ct` | Exclude contrast-enhanced CTs |
| `--time-window N` | PET↔CT time window in minutes (default: 15) |
| `-v` | Verbose output |

---


## Configuration & Overrides

When automatic pairing cannot resolve ambiguous cases (multiple PETs per patient, missing timestamps), the pipeline writes a `SelectionTemplate.yaml`. Copy the relevant entries into `Overrides.yaml`:

```yaml
selections:
  "PatientA/20240115":
    pet_asset_id: "DICOM_SERIES:1.2.840...StudyUID:1.2.840...SeriesUID"
    ct_asset_id:  "DICOM_SERIES:1.2.840...StudyUID:1.2.840...SeriesUID"

  "PatientB/20240120":
    # Alternative: use SeriesInstanceUID directly
    pet_series_uid: "1.2.840.10008..."
    ct_series_uid:  "1.2.840.10008..."
```

Then re-run pairing with `--use-overrides`.

## Citations
https://pmc.ncbi.nlm.nih.gov/articles/PMC10546353/ ~ Wasserthal J, Breit HC, Meyer MT, Pradella M, Hinck D, Sauter AW, Heye T, Boll DT, Cyriac J, Yang S, Bach M, Segeroth M. TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images. Radiol Artif Intell. 2023 Jul 5;5(5):e230024. doi: 10.1148/ryai.230024. PMID: 37795137; PMCID: PMC10546353.









