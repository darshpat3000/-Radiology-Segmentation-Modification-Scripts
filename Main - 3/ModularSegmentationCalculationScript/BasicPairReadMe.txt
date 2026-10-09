# Discovery → Header Extraction → Pairing Pipeline (PET/CT + Segmentations)

This README describes the **interfaces** (inputs/outputs, file schemas, and core behaviors) for the three reusable scripts:

- `CLI_AssetDiscovery.py` → writes **AssetsIndex**
- `CLI_HeaderExtract.py` → reads AssetsIndex, writes **HeaderIndex**
- `CLI_PairFromHeaders.py` → reads HeaderIndex (+ optional overrides), writes **PairIndex**
- `totalseg_tasks.py` → importable module with TotalSegmentator task registry + output-format utilities

The goal is that you can upload **this README** in future chats and request new scripts or modifications **without re-uploading the code**, because the data contracts and behaviors are spelled out here.

---

## System Diagram

```mermaid
flowchart TD
  A[Raw data on disk] --> B[CLI_AssetDiscovery.py<br/>fast discovery]
  B --> C[AssetsIndex.yaml]

  C --> D[CLI_HeaderExtract.py<br/>header parsing]
  D --> E[HeaderIndex.yaml]

  E --> F[CLI_PairFromHeaders.py<br/>pair PET↔CT]
  F --> G[PairIndex.yaml]
  F --> H[SelectionTemplate.yaml<br/>(optional)]
  I[Overrides.yaml<br/>(optional)] --> F

  J[Segmentation pipeline / metrics scripts] --> K[Use PairIndex + indices]
  L[totalseg_tasks.py] --> J
```

---

## Design Principles (Important)

### 1) No “loose folder-name matching” by default
- Pairing and header extraction **do not** use folder names for logic, except:
  - **Asset discovery filtering** can use parent folder name **only if explicitly enabled** with `--use-parent-foldernames`.
  - **Gating detection folder signals** can be enabled **only if explicitly requested** with `--use-folder-name-signals` in header extraction.

### 2) Pairing is “pure logic”
- `CLI_PairFromHeaders.py` reads **HeaderIndex.yaml only** (plus optional overrides).
- It does **not** read DICOM files or scan the filesystem.

### 3) Explicit metrics at each stage
Each index file includes a `stats` block summarizing:
- total assets/series
- CT vs PET counts
- paired vs unpaired parents
- needs-selection counts and reasons

### 4) Join keys are stable and explicit
- Grouping key is **`parent_rel`** (relative parent bucket).
- DICOM stable identifier (preferred) is:
  `asset_id = "DICOM_SERIES:<StudyInstanceUID>:<SeriesInstanceUID>"`

---

## Quick Start

### Step 1 — Discover assets
```bash
python CLI_AssetDiscovery.py scan \
  --dicom-root "D:/Exports" \
  --seg-root "D:/SegmentationsRoot" \
  -v
```

Outputs (default location):
`<dicom-root parent>/Directory Plan/AssetsIndex.yaml`

### Step 2 — Extract DICOM headers (and NIfTI headers if nibabel is installed)
```bash
python CLI_HeaderExtract.py extract \
  --assets "D:/Exports/Directory Plan/AssetsIndex.yaml" \
  -v
```

Outputs:
`HeaderIndex.yaml` (plus optional JSON)

### Step 3 — Pair PET↔CT using headers (+ optional overrides)
```bash
python CLI_PairFromHeaders.py pair \
  --headers "D:/Exports/Directory Plan/HeaderIndex.yaml" \
  --time-window 15 \
  --write-selection-template \
  -v
```

Outputs:
- `PairIndex.yaml`
- `SelectionTemplate.yaml` (only if needed + flag enabled)

---

## Script Interfaces

## A) `CLI_AssetDiscovery.py`

### Purpose
Fast scan of:
- **DICOM series directories** (leaf dirs containing DICOM-like files)
- **NIfTI image files** (`.nii`, `.nii.gz`) from one or more `--nifti-root`
- **Segmentation files** (`.nii`, `.nii.gz`, `.nrrd`, `.mha`, `.mhd`, `.npz`) from one or more `--seg-root`

### Inputs
- `--dicom-root` (optional)
- `--nifti-root` (repeatable, optional)
- `--seg-root` (repeatable, optional)

### Optional parent folder-name filtering (explicit opt-in)
Disabled by default.

Enable with:
- `--use-parent-foldernames`
- then set:
  - `--include-parent-name ...` (repeatable)
  - `--exclude-parent-name ...` (repeatable)
  - `--parent-name-mode substring|regex`

### Outputs
- `AssetsIndex.yaml` (and optional `AssetsIndex.json`)

### AssetsIndex Schema (high level)
```yaml
meta:
  generated_at: ...
  roots: [{label, kind, path}, ...]
  filters:
    use_parent_foldernames: false
    include_parent_name: []
    exclude_parent_name: []
stats:
  total_assets: N
  by_kind:
    dicom_series_dir: ...
    nifti_image: ...
    segmentation_file: ...
assets:
  - asset_id: "DICOM_SERIES_DIR:..."
    kind: dicom_series_dir
    root_label: dicom
    root_path: "D:/Exports"
    path_rel: "HN001/20250101/PET_SERIES_A"
    parent_rel: "HN001/20250101"
    parent_name: "20250101"   # informational only
    dir_stats: {file_count_shallow: 123}
  - asset_id: "NIFTI_IMAGE:..."
    kind: nifti_image
    root_label: nifti1
    root_path: "D:/Nifti"
    path_rel: "HN001/CT.nii.gz"
    parent_rel: "HN001"
    file_stats: {size_bytes: ..., mtime: ...}
  - asset_id: "SEGMENTATION_FILE:..."
    kind: segmentation_file
    root_label: seg1
    root_path: "D:/SegmentationsRoot"
    path_rel: "HN001/vessels.nii.gz"
    parent_rel: "HN001"
    file_stats: {size_bytes: ..., mtime: ...}
```

---

## B) `CLI_HeaderExtract.py`

### Purpose
Convert AssetsIndex → HeaderIndex by reading:
- A representative DICOM file per DICOM series directory (header-only, no pixels)
- NIfTI header info (if `nibabel` installed) for NIfTI-like files

### Inputs
- `--assets AssetsIndex.yaml`
- Optional: `--use-folder-name-signals`
  Enables **folder-name gating hints** (still uses header negations first). Default OFF.

### Outputs
- `HeaderIndex.yaml` (and optional JSON)

### HeaderIndex Schema (high level)
```yaml
meta:
  generated_at: ...
  options:
    use_folder_name_signals: false
stats:
  assets_total: ...
  dicom_series_extracted: ...
  dicom_by_modality: {PET: x, CT: y, Other: z}
dicom_series:
  - asset_id: "DICOM_SERIES:<StudyUID>:<SeriesUID>"  # stable id
    source_asset_id: "<original discovery asset_id>"
    parent_rel: "HN001/20250101"
    series_rel: "HN001/20250101/PET_SERIES_A"
    modality: "PET" | "CT" | "Other"
    series_dt: "2015-10-03T12:34:56-05:00"
    series_dt_source: "AcquisitionDateTime|ContentDateTime|Date+Time|"
    patient_id: ...
    study_uid: ...
    series_uid: ...
    for_uid: ...
    series_desc: ...
    protocol_name: ...
    manufacturer: ...
    station_name: ...
    software_versions: ...
    slice_count_est: <int|None>
    flags:
      gated: true|false
      scout: true|false
      ac: true|false|null           # PET only
      ac_reason: { ... }            # PET only
    classification:
      region: whole_body|chest|head_neck|extremities|other
      is_whole_body: true|false
    pet:                            # PET only
      tracer: "FDG"|"NaF"|...
      injection_dt: "<iso|empty>"
      uptake_min: <float|None>
files:
  - asset_id: "FILE:<discovery_asset_id>"
    kind: nifti_image|segmentation_file
    path_rel: ...
    nifti_header: {shape, zooms, affine}   # only when nifti and nibabel available
    warnings: [...]
failures:
  - source_asset_id: ...
    kind: dicom_series_dir
    reason: "read_error:..."
```

---

## C) `CLI_PairFromHeaders.py`

### Purpose
Pair PET↔CT **within each `parent_rel`** using header fields only.

### Inputs
- `--headers HeaderIndex.yaml`
- `--time-window <minutes>` (default 15)
- Policy flags:
  - `--allow-ct-reuse`
  - `--allow-gated-selection` (default: prefer non-gated)
  - `--use-nac-only`
  - `--allow-unknown-ac`
- Optional: `--overrides Overrides.yaml`
- Optional: `--write-selection-template` (writes template if needs_selection exists)

### Pairing methods (in order)
1. **StudyInstanceUID match**: PET.study_uid == CT.study_uid
2. **Time-window match**: within `--time-window` minutes, requiring matching PatientID when available; **prefers FoR match** (score boost)

### Outputs
- `PairIndex.yaml` (and optional JSON)
- `SelectionTemplate.yaml` (optional)

### PairIndex Schema (high level)
```yaml
meta:
  generated_at: ...
  policy:
    time_window_min: 15
    prefer_non_gated: true
    use_nac_only: false
    allow_unknown_ac: false
stats:
  total_parents: ...
  paired_parents: ...
  needs_selection: ...
  total_series_pet: ...
  total_series_ct: ...
selected_pairs:
  "<parent_rel>":
    reason: auto_selected|manual_override|override_invalid_ids|override_parent_not_found
    pet_asset_id: "DICOM_SERIES:..."
    ct_asset_id:  "DICOM_SERIES:..."
    pairing_method: StudyInstanceUID|Time±FoR|TimeOnly|manual
needs_selection:
  - parent_rel: "HN001/20250101"
    reason: no_pairs_found | no_candidates_after_policy
    pet_count: ...
    ct_count: ...
    candidate_pair_count: ...
    policy: {counts:{...}, reasons:[...]}
parents:
  - parent_rel: ...
    pets: [series_stub...]
    cts:  [series_stub...]
    candidate_pairs: [{pairing_method, pet, ct, header_comparison, why_paired}, ...]
    policy_summary: {...}
    selected: <one pair or null>
```

### Overrides.yaml (manual selection)
Use stable asset IDs whenever possible:

```yaml
selections:
  "HN001/20250101":
    pet_asset_id: "DICOM_SERIES:<StudyUID>:<SeriesUID>"
    ct_asset_id:  "DICOM_SERIES:<StudyUID>:<SeriesUID>"
```

Alternative (series UID only):
```yaml
selections:
  "HN001/20250101":
    pet_series_uid: "<SeriesInstanceUID>"
    ct_series_uid:  "<SeriesInstanceUID>"
```

### SelectionTemplate.yaml
If pairing can’t decide, this file lists candidates per parent with `use: false`. You copy the chosen `asset_id`s into Overrides.yaml.

---

## D) `totalseg_tasks.py` (module)

### Purpose
A shared registry and utilities so future segmentation/metric scripts do **not** embed TotalSegmentator task lists or output format heuristics.

### Functions you can rely on
- `list_tasks() -> List[str]`
- `get_task(name) -> TaskInfo|None`
- `detect_output_format(output_dir) -> "multilabel"|"per_structure"|"mixed"|"empty"`
- `try_get_label_map(task) -> Dict[int,str]|None`
  Returns class map **from installed TotalSegmentator** if available.
- `infer_task_from_folder_name(name) -> Optional[str]`
  **Optional convenience** only; do not use unless explicitly desired.

---

## “Contracts” for Future Scripts

If you ask ChatGPT for new scripts later, you can reference these contracts:

### Contract 1 — Never rescan raw trees unless necessary
Downstream scripts (segmentation, metrics, QA) should read:
- `PairIndex.yaml` for the “truth” PET/CT mapping
- `HeaderIndex.yaml` for metadata needed in reports
- `AssetsIndex.yaml` for physical locations if needed
…and write their own **DerivedIndex** for outputs.

### Contract 2 — Join keys
- Primary grouping: `parent_rel`
- DICOM series identity: `dicom_series.asset_id` (DICOM_SERIES:StudyUID:SeriesUID)

### Contract 3 — Folder names are non-authoritative
Folder name usage must be explicitly enabled via CLI flags.

---

## Common Requests You Can Make Later (Examples)

When you upload this README, you can ask:
- “Add a **DerivedIndex writer** format for segmentation outputs keyed by `pair_id`.”
- “Add a **PET tracer filter** to pairing, but only using header tracer fields (no folder-name tokens).”
- “Add NIfTI-based PET/CT pairing when DICOM isn’t available, using affine/shape checks.”
- “Add a report that summarizes `needs_selection` reasons across parents.”
- “Add a ‘re-run only changed series’ mode in HeaderExtract.”

---

## Versioning Notes

This README describes the initial split into 3 scripts + 1 module:
- Discovery = `AssetsIndex`
- Header extraction = `HeaderIndex`
- Pairing = `PairIndex` (+ overrides + template)
- TotalSegmentator registry utilities = `totalseg_tasks.py`

If you modify any schema field names in the future, update this README first; it is the “contract” document.
