# fatSeg

A desktop tool for segmenting fat in MRI. It reads axial Dixon fat-image (`_F`) DICOM series and has two modes:

- **Abdomen:** separates **subcutaneous fat (SAT)** from **visceral fat (VAT)**.
- **Thigh:** separates **SAT**, **intermuscular fat (IMAT)** and **muscle**, using a classical method, a U-Net, or both combined.

Every result can be corrected by hand (polygon and brush tools) and saved as NIfTI (`.nii.gz`).

## Installation

**Full step-by-step instructions, including troubleshooting, are in [INSTALL.md](INSTALL.md).** In short, on Windows with 64-bit Python 3.12:

```bat
python -m venv .seg
.seg\Scripts\python.exe -m pip install -r requirements.txt
```

Then copy `unet_thighfat_finetuned.keras` (obtained separately; only needed for thigh **AI Seg**) into the project folder and start the app with `runFat.bat`.

## Running

```bash
runFat.bat
```

or, equivalently, from the project root:

```bash
.seg\Scripts\python.exe mriFat\readSpace_threeButton.py
```

Start it from the project root: the mode icons (`sat.png`, `thigh.png`) are loaded from the current folder.

## Using the app

The image button at the top left switches modes. The **belly** icon is abdomen mode, the **leg** icon is thigh mode. The app opens in abdomen mode.

### Common to both modes

| Button | Function |
|---|---|
| **Load short axis slices** | Select the folder with the axial DICOMs. **Abdomen:** loads any series whose name ends in `_F`. **Thigh:** loads only the series named exactly `6pt_DIXON_VIBE_F`. |
| **Load long axis slices** | Optional coronal localizer. Shows the position of the current axial slice and the crop range. |
| **Load seg file (nii.gz)** | Load a previously saved segmentation. |
| **Flip Seg X/Y** | Transpose the segmentation if it was saved in the other orientation. |
| **Show Image** | Opens the editor: polygon or brush in green/red/blue/clear. "Over …" limits edits to pixels that already carry that label. Scroll the mouse wheel to change brush size. |
| **Save Segmentation** | Save as `.nii.gz`. |

- **Slider:** the three-handle slider sets the crop range (blue/red handles) and the current slice (green handle).
- **Area plot:** shows the area of each label on every slice. Its axis says "Mass (g)", but the values are **areas in cm²**, and the "Total" values add up per-slice areas without slice thickness.

### Abdomen mode: SAT / VAT

1. Load the short-axis slices (a series ending in `_F`).
2. Click **SAT/AVT Seg**.
3. Check the result with **Show Image** and correct it where needed.
4. Save.

The **Segmentation Threshold** box only drives a first pass that measures how bright fat is on each slice. Leave it at the default of 100. The threshold actually applied is printed to the console, e.g. `Abdomen: fat signal 344-411, fat threshold 138-164`.

### Thigh mode: SAT / IMAT / muscle

| Button | Method |
|---|---|
| **Draw Line** | Draw a line on the long-axis image to jump to the nearest axial slice. |
| **Thigh Seg** | Classical method (`mriFat/seg.py`). Output: **1 SAT, 2 IMAT, 3 muscle**. |
| **AI Seg** | U-Net, 256×256 input. Output: a single foreground mask (label 1). |
| **Combined** | Uses the AI Seg mask as the outline and fills in IMAT and muscle from the classical method. Run AI Seg first. |
| **Swap XY** | Rotates/flips each slice of the segmentation to correct its orientation. |

## Label conventions

| Label | Colour | Abdomen | Thigh |
|---|---|---|---|
| 0 | none | background, non-fat, vertebral marrow | background |
| 1 | green | SAT | SAT (AI Seg: foreground) |
| 2 | red | VAT | IMAT |
| 3 | blue | none | muscle |

The same numbers mean different things in each mode, so keep track of which mode a saved file came from.

Saved files contain an **identity affine**: they don't carry patient geometry. Array layout is (rows, cols, slices), with slices in slice-location order, matching the DICOMs they were made from.

## How the abdominal segmentation works

Implemented in `mriFat/abd_seg.py`, separately from the thigh code. On each axial slice:

1. **Fat threshold:** set to 40% of the local fat signal (`FAT_FRACTION`). The fat signal is the median SAT intensity of the slice, smoothed over neighbouring slices. This makes volumes comparable between scans with different intensity scales and voxel sizes.
2. **Body outline:** the subcutaneous fat ring, closed over small gaps and filled. **Arms are removed**, whether they lie close to the torso or touch it.
3. **SAT/VAT boundary:** 360 rays are cast from the body centre. On each ray, SAT is the run of fat from the skin inward up to the muscle wall. Gaps of up to 2 mm are tolerated (`GAP_TOL_MM`).
4. **Correction of faulty rays:** rays that disagree with their neighbours are replaced by the local median. These come from fat bridging into VAT, or from vessels and the navel crease inside the SAT. A wider fault along the anterior midline (linea alba) is bridged by interpolation.
5. **Labelling:** fat outside the boundary is SAT, fat inside is VAT. Fat inside the vertebral body (marrow) is left unlabelled.

The tunable settings are at the top of `abd_seg.py`.

## Other scripts

| Script | Purpose |
|---|---|
| `compare_overlap.py` | Compares SAT/VAT volumes between the `AbdoCompL3` and `T12S1…DIXON VIBE` scans of each case over their common z range (scanner coordinates, partial slices counted fractionally). Reads the `seg.nii.gz` saved in each series folder. `python compare_overlap.py <root> [--csv out.csv]` |
| `finetune.py` | Fine-tunes the thigh U-Net on `thighFat/<case>/img.nii.gz` + `seg.nii.gz`. Writes `unet_thighfat_finetuned.keras`. |
| `calBlock.py` | Prints label volumes for a folder of thigh segmentations (hard-coded path and voxel size). |
| `test_display.py` | Debugging script for AI Seg orientation (hard-coded paths). |
| `mriFat/sat_seg.py` | Former abdominal method (plain threshold). No longer used. |

**Expected layout for `compare_overlap.py`:**
```
<root>/<case>/AbdoCompL3/          DICOMs + seg.nii.gz
<root>/<case>/T12S1DIXONVIBE/      DICOMs + seg.nii.gz
```

## Thigh U-Net

- **Architecture:** U-Net, input `(256, 256, 1)`, output `(256, 256, 4)` softmax.
- **Classes:** 0 background, 1 SAT, 2 IMAT, 3 muscle. `np.argmax` over the last axis gives the label map.
- **Training data:** trained on 2D slices of 3D NIfTI volumes, then fine-tuned with `finetune.py`.
- **Preprocessing:** each slice is transposed to NIfTI orientation, resized to 256×256 and scaled to [0, 1] by its maximum.
- **Resources:** inference on CPU needs roughly 2–4 GB of RAM.
- **In the GUI:** AI Seg merges the three foreground classes into one mask, and **Combined** adds IMAT and muscle from the classical method.

## Known limitations

- **Orientation:** vertebra detection in abdomen mode assumes the standard axial view, with the front of the body at the top of the image.
- **Arms and open rings:** if the SAT ring is open and an arm also touches the torso on the same slice, that arm isn't removed.
- **Thin VAT in lean patients:** VAT in lean patients is mostly thin strands, which are sensitive to resolution. Across sequences, VAT differed by 1.3–6.6% on the five test cases (SAT by 0.2–5.6%).
- **Thigh Seg:** the classical method can fail on slices that contain no thigh.

## Data protection

The images, segmentations and derived meshes come from patients, and folder names carry patient IDs. `.gitignore` excludes DICOM, NIfTI, STL and NumPy files, the known data folders and the model weights. Run `git status` before every commit to make sure no patient data is included.
