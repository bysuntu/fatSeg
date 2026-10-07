# Changelog

## 2026-10-07: Thigh segmentation: hip-end fix, T1 TSE support, IMAT check

All changes are in thigh mode. Abdomen mode (`abd_seg.py`) is unchanged.

### Dixon fat (`_F`) thigh segmentation (`mriFat/seg.py`)

- **The other leg is no longer counted as SAT near the hip.** On the top slices, the other leg or the perineum lies against the thigh and used to be counted as SAT. The new `segment_stack()` segments the whole stack together: a slice whose thigh outline is no longer nearly convex (solidity < 0.98) is limited to the neighbouring slice's thigh plus 2 pixels, and is cut along the dark skin line between the legs (Sato ridge filter). Clean slices are unchanged.
  - On `thighFat` (11 hand-corrected cases): SAT Dice 0.976 → 0.998, worst case 0.928 → 0.991. Voxels the annotators had to erase: 587k → 31k.
- **No more `-1` labels.** Bright specks outside the body were counted as IMAT, which made the muscle label negative there. They are now ignored.
- **No more crash on slices without a femur.** Previously one such slice stopped the whole Thigh Seg run; the slice is now segmented without bone.
- **New optional parameters:** `segment()` and `segment_stack()` take `sat_thr` / `imat_thr` (defaults 100 / 80, the previous fixed values), and `segment()` can also return the body outline (`return_outer=True`). Called with the defaults, the Dixon output is identical to before these parameters existed (checked bit for bit on all 11 `thighFat` cases).

### Dixon IMAT check (new: `mriFat/dixon_local_imat.py`)

- IMAT pixels are compared with the median of the surrounding muscle within 10 mm. IMAT that is not at least 8% of the way from the local muscle level to the local fat level becomes muscle. Muscle and SAT are never changed.
- **On by default** (`ENABLED = True`), in Thigh Seg and Combined for Dixon slices. It lowers the Dixon IMAT by 20–32% (muscle goes up by the same amount; SAT is unchanged).
- The 8% level was chosen so the Dixon IMAT matches the T1 TSE IMAT of the same scans (see below).
- To turn it off: `ENABLED = False`. To remove it: delete the file and the three lines marked `dixon_local_imat` in `readSpace_threeButton.py`.

### T1 TSE (`t1_tse_tra`) thigh segmentation (new: `mriFat/t1_seg.py`)

A separate method, so changes to it cannot affect Dixon results. On T1 the fat/muscle contrast is lower and coil sensitivity makes one side of the thigh much brighter, so the Dixon thresholds do not work. Steps:

1. Resample to 0.8 mm (the Dixon pixel size the `seg.py` settings were tuned at).
2. Denoise with non-local means at 3× the estimated noise. (BM3D gave about the same result at ~15× the run time.)
3. Divide by the local fat signal (20 mm), so fat is ~1 and muscle ~0.35 everywhere.
4. Segment with `seg.segment_stack()` at 0.40 of the fat signal for SAT and IMAT (including the hip-end handling).
5. Check each IMAT pixel against its neighbours (10 mm, 15%, specks of at least 5 pixels). Some muscle groups (hamstrings, adductors) are brighter on T1 without containing fat; this step keeps them from being counted as IMAT.

The 15% level was chosen by eye: lower levels keep specks of grainy muscle as IMAT.

### Agreement between Dixon and T1 TSE

On `rawThigh` (8 cases scanned with both sequences, same slice positions; the 8% level was tuned on 7 of them, 01260017NHCSLX was not used for tuning), T1 compared with Dixon (with the IMAT check):

| | Mean difference (T1 − Dixon) | Per case | Dice |
|---|---|---|---|
| SAT | −4.0% | −5.6% to −2.2% | 0.952 |
| IMAT | +2.5% | −11.9% to +14.3% | 0.496 |
| Muscle | +2.5% | +0.2% to +5.8% | 0.918 |

The Dixon check level (8%) is tuned to the T1 level (15%). If one changes, the other has to be re-tuned.

### GUI (`mriFat/readSpace_threeButton.py`)

- **Loading in thigh mode:** loads `6pt_DIXON_VIBE_F`, or, if the folder has none, any series whose name contains `t1_tse_tra` (case-insensitive). The app remembers which sequence was loaded.
- **Loading in abdomen mode:** still `_F` only. A folder with only `t1_tse_tra` offers to switch to thigh mode instead of failing.
- **Thigh Seg** uses `seg.py` + `dixon_local_imat.py` for Dixon slices and `t1_seg.py` for T1 slices.
- **AI Seg and Combined** refuse T1 slices with a message, because the U-Net was trained on Dixon fat images only. Combined now also uses the stack-level Dixon segmentation and the IMAT check.
- **JPEG-compressed DICOMs** (e.g. JPEG Lossless) can now be read: `pylibjpeg` and `pylibjpeg-libjpeg` were added to `requirements.txt` and installed in the bundled Python.
- **DICOM reading:** the series name is checked before pixels are decoded (faster when a folder holds many series), and when no matching series is found the error lists the series that are in the folder.

### Known issues

- **Hip end:** in 1 of the 11 `thighFat` cases, the skin line between the legs is too faint on the top few slices, and part of the other leg is still counted as SAT.
- **Validation:** the Dixon hip-end settings were tuned and checked on the same 11 `thighFat` cases, whose labels are corrected output of this same method. The T1 settings were checked against the Dixon method, not against hand-corrected T1 labels. IMAT overlap between the sequences is moderate (Dice ~0.5).
