# fatSeg

A desktop tool for segmenting fat in MRI. It reads axial DICOM series (Dixon fat image `_F` in both modes, and T1 TSE `t1_tse_tra` in thigh mode) and has two modes:

- **Abdomen:** separates **subcutaneous fat (SAT)** from **visceral fat (VAT)**.
- **Thigh:** separates **SAT**, **intermuscular fat (IMAT)** and **muscle**, using a classical method, a U-Net, or both combined.

Every result can be corrected by hand (polygon and brush tools) and saved as NIfTI (`.nii.gz`).

Recent changes are listed in [CHANGELOG.md](CHANGELOG.md).

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
| **Load short axis slices** | Select the folder with the axial DICOMs. **Abdomen:** loads any series whose name ends in `_F`. **Thigh:** loads the series named exactly `6pt_DIXON_VIBE_F`; if the folder has none, any series whose name contains `t1_tse_tra` (case-insensitive). Loading a `t1_tse_tra` folder in abdomen mode offers to switch to thigh mode. |
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

The **VAT Fat Fraction** slider (0.20–0.60 in steps of 0.01, default 0.40) sets which pixels inside the SAT boundary count as VAT: a pixel is VAT when its signal is at least this share of the slice's fat signal (the median brightness of its SAT). It does not change SAT or the SAT/VAT boundary, which always use 0.4. Lower values count more partly-fat pixels as VAT (VAT is mostly thin strands: about 9% more VAT per 0.05 step, more in lean patients). **Release the slider (or use the arrow keys) to apply a new value**: VAT is relabelled within a few seconds, reusing the SAT/VAT boundary of the last **SAT/AVT Seg**, and the image and VAT plot update. If the segmentation was edited by hand or loaded since then, you are asked before it is replaced. The slider is hidden in thigh mode. The thresholds actually applied are printed to the console, e.g. `Abdomen: fat signal 344-411; SAT fat fraction 0.4 (threshold 138-164), VAT fat fraction 0.4 (threshold 138-164)`.

### Thigh mode: SAT / IMAT / muscle

| Button | Method |
|---|---|
| **Draw Line** | Draw a line on the long-axis image to jump to the nearest axial slice. |
| **Thigh Seg** | Classical method. Output: **1 SAT, 2 IMAT, 3 muscle**. Dixon fat (`_F`) slices use `mriFat/seg.py`; `t1_tse_tra` slices use `mriFat/t1_seg.py` (see below). |
| **AI Seg** | U-Net, 256×256 input. Output: a single foreground mask (label 1). Dixon fat (`_F`) only. |
| **Combined** | Uses the AI Seg mask as the outline and fills in IMAT and muscle from the classical method. Run AI Seg first. Dixon fat (`_F`) only. |
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

Implemented in `mriFat/abd_seg.py`, separately from the thigh code. On each axial slice (step 4 also uses the neighbouring slices):

1. **Fat threshold:** 40% of the local fat signal (`FAT_FRACTION`) for SAT and the SAT/VAT boundary; for VAT, the **VAT Fat Fraction** set in the GUI (default 0.4) times the local fat signal. The fat signal is the median SAT intensity of the slice, smoothed over neighbouring slices. This makes volumes comparable between scans with different intensity scales and voxel sizes.
2. **Body outline:** the subcutaneous fat ring, closed over small gaps and filled. **Arms are removed**, whether they lie close to the torso or touch it.
3. **SAT/VAT boundary:** 360 rays are cast from the body centre. On each ray, SAT is the run of fat from the skin inward up to the muscle wall. Gaps of up to 2 mm are tolerated (`GAP_TOL_MM`).
4. **Correction of faulty rays:**
   - Rays that disagree with their neighbours on the same slice are replaced by the local median (fat bridging into VAT, or vessels and the navel crease inside the SAT).
   - A sharp dip in SAT thickness that returns within 20 rays (a ray stopped too early) is bridged by interpolation all around the body. A sharp bump that returns is bridged only at the front (±45°, e.g. along the linea alba), because thick fat pads on the flanks and pelvis are real.
   - **Across slices:** each ray is compared with the same ray on the 3 slices above and below. If its SAT is much thinner (below 75% of their median, minus 2 mm), it takes their median. This repairs wide wedges of VAT cutting into the SAT on one or two slices; real anatomy changes gradually along the body (`XSLICE_*` settings).
5. **Labelling:** fat outside the boundary is SAT, fat inside is VAT. Fat inside the vertebral body (marrow) is left unlabelled.

The tunable settings are at the top of `abd_seg.py`.

**Boundary method switch (`BOUNDARY` in `abd_seg.py`):** the SAT/VAT boundary can be found in three ways.

| Setting | How the boundary is found |
|---|---|
| `'rays'` | Steps 3–4 above. |
| `'path'` | The best closed path around the body, found by dynamic programming on the image unwrapped around the body centre. Each candidate edge scores fat on its outer side minus a dark band on its inner side (a thick muscle wall scores high, a thin fascia line low), minus the darkness crossed from the skin, minus changes in thickness between neighbouring angles. Because it is one continuous boundary, a fascia line cannot stop it locally. |
| `'combined'` (default) | The ray result, except where a ray is more than 8 mm thinner than the path (the ray stopped early); there the path is used. |

Against the hand-corrected `rawAbd` files (both `03260016NHCTHO` scans and `05250002NHCNSA` T12S1):

| Setting | Dice SAT / VAT | SAT labelled VAT | VAT labelled SAT |
|---|---|---|---|
| `'rays'` | 0.987 / 0.979 | 424 cm³ | 51 cm³ |
| `'path'` | 0.988 / 0.977 | 179 cm³ | 223 cm³ |
| `'combined'` | **0.992 / 0.984** | 103 cm³ | 110 cm³ |

`'combined'` halves the mislabelled fat and is the default. It was tuned and checked on only 3 hand-corrected scans; on the 5 unedited ones it changes the volumes by at most 0.7%. With it, `AbdoCompL3` and `T12S1…DIXON VIBE` agree over their common region within 0.9% for SAT and 2.1% for VAT on average (five cases).

## How the thigh segmentation works

**Thigh Seg** uses a separate method for each sequence. Which one runs depends on the series that was loaded.

### Dixon fat (`_F`): `mriFat/seg.py` + `mriFat/dixon_local_imat.py`

On each axial slice (`seg.segment`):

1. **SAT:** fat above a fixed threshold (100 on the raw pixel values); the largest fat region is the subcutaneous ring.
2. **Femur:** the bright marrow inside the ring that does not touch it. If no femur is found, the slice is still segmented, without bone.
3. **IMAT:** the remaining fat inside the ring (threshold 80). Bright specks outside the body are ignored.
4. **Inner edge of the SAT ring:** fat within 2 pixels of the ring that is connected to it is SAT. (The ring clean-up trims about a pixel off its inner edge, and IMAT is not searched that close to the ring, so this pure fat used to be counted as muscle.)
5. **Muscle:** everything else inside the body outline.

Over the whole stack (`seg.segment_stack`), the other leg is kept out near the hip, where it lies against the thigh and would otherwise count as SAT:

- A clean thigh outline is nearly convex (solidity ≥ 0.98). A slice where the other leg has merged in is not.
- Clean slices are kept as they are. On merged slices, the thigh may grow at most 2 pixels beyond the neighbouring slice's thigh, and it is also cut along the thin dark skin line where the two legs touch (Sato ridge filter).
- Slices are processed outward from the clean slice nearest the middle of the stack.

Finally, **`dixon_local_imat.py`** checks each IMAT pixel against its neighbours (the same check as step 5 of the T1 method below, at a level of 25%) and turns IMAT that is not clearly brighter than the surrounding muscle into muscle. It is used by Thigh Seg and Combined. The 25% level was set against the scanner's fat-fraction map (`6pt_DIXON_VIBE_FF`): the IMAT volume then equals the volume of pixels that are at least 50% fat (1.03×, 0.94–1.11× per case on `rawThigh`), and 97–98% of the IMAT pixels are at least 30% fat (without the check, about half were less than 30% fat). `ENABLED = False` in that file turns it off; deleting the file and the three lines marked `dixon_local_imat` in `readSpace_threeButton.py` removes it.

Checked on `thighFat` (11 hand-corrected cases, before the IMAT check and the inner-edge step): SAT Dice 0.998 (worst case 0.991), IMAT 0.993, muscle 0.998. Those labels were made by correcting this method's own output, so they cannot judge the two later steps; the fat-fraction map can.

### T1 TSE (`t1_tse_tra`): `mriFat/t1_seg.py`

Implemented separately from the Dixon fat method, so changes to it cannot affect `_F` results. On T1 the fat/muscle contrast is lower than on the Dixon fat image and coil sensitivity makes one side of the thigh much brighter, so fixed thresholds fail. Instead:

1. **Resample** each slice to the Dixon pixel size (0.8 mm), where the `seg.py` settings were tuned.
2. **Denoise** with non-local means at 3× the estimated noise level, which calms speckle in the muscle that would otherwise count as IMAT.
3. **Fat fraction image:** each slice is divided by its local fat signal (nearby fat, smoothed over 20 mm), so fat is ~1 and muscle ~0.35 on both sides of the thigh.
4. **Segment** with the same steps as `seg.py` (including the hip-end handling), using thresholds of 0.40 of the fat signal for both SAT and IMAT.
5. **IMAT check by local contrast:** some muscle groups (e.g. hamstrings, adductors) are uniformly brighter on T1 without containing fat. Each IMAT pixel is compared with the median of its neighbours within 10 mm (the local muscle level): it stays IMAT when it lies at least 40% of the way from that level to the local fat level, in a speck of at least 5 pixels, otherwise it becomes muscle. Only IMAT (red) is re-checked; muscle (blue) and SAT (green) are not changed by this step.

The 40% level was tuned so the T1 IMAT matches the Dixon IMAT of the same scans (see below); lower levels keep specks of grainy muscle as IMAT. The settings are at the top of `t1_seg.py` (`DENOISE_STRENGTH = 0` turns denoising off, `LOCAL_RADIUS_MM = None` turns step 5 off).

### Agreement between the two sequences

The Dixon method is the reference: its IMAT is checked against the scanner's fat-fraction map, which T1 cannot provide. The T1 level (40%) was tuned so the T1 IMAT matches it. On `rawThigh` (8 cases scanned with both sequences, same slice positions), T1 compared with Dixon:

| | Mean difference (T1 − Dixon) | Per case | Dice |
|---|---|---|---|
| SAT | −3.7% | −5.2% to −2.0% | 0.955 |
| IMAT | +3.4% | −12.7% to +18.5% | 0.504 |
| Muscle | +2.3% | +0.5% to +4.9% | 0.948 |

T1 SAT is consistently a little lower because fewer partly-fat voxels at the skin and fascia count as fat. IMAT overlap is only moderate (Dice ~0.5): IMAT streaks are one or two pixels wide, and T1 contrast cannot tell partly-fat pixels from muscle as well as the Dixon fat image (against the fat-fraction map, a third of the T1 IMAT pixels are less than 30% fat, vs 2–3% for Dixon). The volumes agree much better than the pixel positions.

**The two levels are linked:** the T1 level is tuned to the Dixon one. If `CONTRAST` in `dixon_local_imat.py` (or anything else in the Dixon method) changes, `LOCAL_CONTRAST` in `t1_seg.py` has to be re-tuned to keep the sequences in agreement.

## Other scripts

| Script | Purpose |
|---|---|
| `compare_overlap.py` | Compares SAT/VAT volumes between the `AbdoCompL3` and `T12S1…DIXON VIBE` scans of each case over the region both cover (scanner coordinates). Each scan is cut to the slab the other covers, along that scan's own slice direction, so scans tilted against each other are compared over the same region; voxels partly inside count by the fraction inside. Reads the `seg.nii.gz` saved in each series folder. `python compare_overlap.py <root> [--csv out.csv]` |
| `make_abd_stl.py` | Segments every abdominal scan with `abd_seg.py` (current `BOUNDARY`) and writes SAT and VAT surfaces as binary STL in patient coordinates (mm), for ParaView: `<out>/<case>/{L3,T12}_{SAT,VAT}.stl`, plus `_overlap` versions cut to the slab the other scan covers (along its own slice direction, so tilted scans are handled) and `segmentation_volumes.csv`. `python make_abd_stl.py <root> [--out abd_stl] [--smooth 0.7]`. Volumes should be taken from the CSV's voxel column: smoothing makes the meshes of thin VAT strands 1–7% smaller. |
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
- **Thin VAT in lean patients:** VAT in lean patients is mostly thin strands, which are sensitive to resolution. Between `AbdoCompL3` and `T12S1…DIXON VIBE` over their common region, VAT differs by 0.2–6.5% on the five test cases (SAT by 0.3–1.7%); the largest VAT difference is the leanest patient (`09260021NHCSSB`).
- **SAT counted as VAT behind fascia lines:** in thick SAT on the flanks and back, a thin dark fascia line can stop a SAT ray, so the deep SAT layer behind it is labelled VAT (e.g. `03260016NHCTHO`). With `BOUNDARY = 'combined'` (default) about 100 cm³ remains over the 3 hand-corrected scans (424 cm³ with `'rays'`); correct the rest with the brush tool. Rules tested to remove it (letting rays pass grey pixels on the flanks, a wider gap tolerance, largest fat piece as SAT, VAT next to SAT turned into SAT) moved real VAT into SAT and were not adopted.
- **Thigh Seg near the hip:** where the skin line between the legs is too faint to cut along, a piece of the other leg can still be counted as SAT on the top few slices (seen in 1 of 11 `thighFat` cases).
- **Thigh Seg tuning:** the Dixon hip-end settings were tuned on the same 11 `thighFat` cases they were checked on, and those labels were made by correcting this method's own output. The T1 settings were checked against the Dixon method, not against hand-corrected T1 labels.
- **T1 TSE:** AI Seg and Combined are not available (the U-Net was trained on Dixon fat images only).

## Data protection

The images, segmentations and derived meshes come from patients, and folder names carry patient IDs. `.gitignore` excludes DICOM, NIfTI, STL and NumPy files, the known data folders and the model weights. Run `git status` before every commit to make sure no patient data is included.
