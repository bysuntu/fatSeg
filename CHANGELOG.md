# Changelog

## 2026-10-09: STL export for the abdominal scans; `compare_overlap.py` handles tilted scans

- **New `make_abd_stl.py`.** Segments every abdominal scan with `abd_seg.py` (current `BOUNDARY`) and writes SAT and VAT surfaces as binary STL in patient coordinates (mm), so both scans of a case open aligned in ParaView: `<out>/<case>/{L3,T12}_{SAT,VAT}.stl`, `_overlap` versions cut to the region the other scan covers, and `segmentation_volumes.csv`. SAT meshes match the voxel volume within 0.3%; the light smoothing makes VAT meshes 1–7% smaller (thin strands), so take volumes from the CSV's voxel column. `abd_stl/` is in `.gitignore` (folder names are patient IDs).
- **`compare_overlap.py` handles scans tilted against each other.** It used to cut both scans between the same flat z planes. The `AbdoCompL3` slices of `03260016NHCTHO` are tilted by 3.9° against its `T12S1` scan, so near the flanks the two scans were compared over different regions. Each scan is now cut to the slab the other covers, along that scan's own slice direction, with voxels partly inside counted by the fraction inside. For two axial scans the result is unchanged. For `03260016NHCTHO` (saved segmentations) the difference goes from SAT +5.6% / VAT −3.4% to SAT −0.5% / VAT +1.8%. The output now also lists the tilt; the CSV columns `z_from` / `z_to` are renamed `slab_from` / `slab_to` (limits along the L3 slice direction) and `tilt_deg` is added.
- **Corrections to earlier numbers:** the `03260016NHCTHO` L3-vs-T12 differences of about ±5–8% in the 2026-10-08 entry were mostly this tilt, not segmentation or breathing.

## 2026-10-09: Abdominal segmentation: optional boundary path (`BOUNDARY` switch)

`mriFat/abd_seg.py` has a new setting, `BOUNDARY`, for how the SAT/VAT boundary is found. The default is now `'combined'`.

- **`'path'`:** the best closed boundary around the body, found by dynamic programming on the image unwrapped around the body centre. Each candidate edge scores fat on its outer side minus a dark band on its inner side, minus the darkness crossed from the skin, minus changes in thickness between neighbouring angles (`PATH_*` settings). A thin fascia line inside thick SAT cannot stop it, because the boundary must be continuous all around. It sits about 1 pixel shallower than the rays, so it is moved 2 pixels deeper (`PATH_OFFSET_PX`).
- **`'combined'`:** the ray result, except where a ray is more than 8 mm thinner than the path (`COMBINE_TOL_MM`); there the path is used. The rays keep their pixel-accurate edge, and the path takes over where a ray stopped early.

Against the hand-corrected segmentations (both `03260016NHCTHO` scans and `05250002NHCNSA` T12S1):

| Setting | Dice SAT / VAT | SAT labelled VAT | VAT labelled SAT | Changed on the 5 unedited scans |
|---|---|---|---|---|
| `'rays'` | 0.987 / 0.979 | 424 cm³ | 51 cm³ | 11 cm³ |
| `'path'` | 0.988 / 0.977 | 179 cm³ | 223 cm³ | 235 cm³ |
| `'combined'` | 0.992 / 0.984 | 103 cm³ | 110 cm³ | 60 cm³ |

`AbdoCompL3` vs `T12S1` agreement (mean SAT / VAT difference): `'rays'` 1.7% / 3.6%, `'path'` 1.4% / 3.2%, `'combined'` 1.4% / 3.3%.

`'combined'` is the default. It was tuned and checked on only 3 hand-corrected scans; on the 5 unedited scans it changes the volumes by at most 0.7%.

## 2026-10-08: Abdominal segmentation: SAT/VAT boundary fixes

All changes are in `mriFat/abd_seg.py` (abdomen mode). Thigh mode is unchanged.

The SAT/VAT boundary rays sometimes stop inside the SAT, so part of it is counted as VAT. This shows as a wedge of VAT cutting into the SAT ring, or a deep layer of SAT behind a fascia line counted as VAT. In `03260016NHCTHO`, about 300 cm³ moved between SAT and VAT between its two scans although the total fat agreed.

- **Dips are bridged all around the body.** A sharp drop in SAT thickness that returns within 20 rays (a ray stopped too early) is now interpolated on the flanks and back too, not only at the front. Sharp bumps are still only bridged at the front, because thick fat pads on the flanks and pelvis are real. The bridging code also no longer indexes past the end of the ray array.
- **Across-slice check.** Each ray's SAT thickness is compared with the same ray on the 3 slices above and below; a ray much thinner than there (below 75% of their median, minus 2 mm) takes their median. This repairs wide VAT wedges cutting through the SAT ring on one or two slices, including at the first and last slices of a scan (`XSLICE_WINDOW`, `XSLICE_THIN`, `XSLICE_ABS_MM`).
- **Code structure:** `segment_abdomen_stack()` now measures the rays on all slices, checks them across slices, then builds the labels. `segment_abdomen()` (one slice) and `fat_reference()` work as before.

**Against the hand-corrected segmentations** (the `seg.nii.gz` files in `rawAbd`, 10 scans; 5 of them were corrected by hand, the other 5 are unedited output of the previous version):

| | Dice SAT | Dice VAT | SAT labelled VAT | VAT labelled SAT | Mean volume error SAT / VAT |
|---|---|---|---|---|---|
| Before | 0.9892 | 0.9653 | 847 cm³ | 30 cm³ | 1.84% / 5.01% |
| Now | 0.9902 | 0.9664 | 615 cm³ | 63 cm³ | 1.54% / 4.67% |

On the 5 unedited scans the result changes by at most 2 cm³ per scan.

**Agreement between the two scans** (`AbdoCompL3` vs `T12S1…DIXON VIBE` over their common range): mean difference SAT 1.9% → 1.7%, VAT 3.8% → 3.6%; `03260016NHCTHO` ±7.9% → ±6.7%.

**Tested and not adopted:**
- **Letting rays pass grey pixels on the flanks and back,** stopping only at dark (muscle) pixels. It fixed the SAT behind fascia lines, but pushed 200–500 cm³ of real VAT into SAT over the 10 corrected scans, and had the worst Dice.
- **A wider gap tolerance on the flanks,** which crossed the thin muscle between the ribs.
- **SAT = the largest fat piece, with everything inside it VAT.** VAT and SAT are usually connected through gaps in the muscle wall.
- **Smoothing the image before finding the boundary.**
- **Moving SAT pieces separate from the ring to VAT.** The large ones are real SAT split off by a VAT wedge.
- **Moving small VAT pieces next to SAT, or VAT within a few mm of SAT, to SAT.** This also moved real VAT next to the pelvic and back muscles.
- **A partial-volume VAT volume.**

**Known issue:** SAT behind a fascia line on the flanks and back can still be labelled VAT (about 615 cm³ over the 10 corrected scans). Correct it with the brush tool.

## 2026-10-07: Thigh segmentation: hip-end fix, T1 TSE support, IMAT check

All changes are in thigh mode. Abdomen mode (`abd_seg.py`) is unchanged.

### Dixon fat (`_F`) thigh segmentation (`mriFat/seg.py`)

- **The other leg is no longer counted as SAT near the hip.** On the top slices, the other leg or the perineum lies against the thigh and used to be counted as SAT. The new `segment_stack()` segments the whole stack together: a slice whose thigh outline is no longer nearly convex (solidity < 0.98) is limited to the neighbouring slice's thigh plus 2 pixels, and is cut along the dark skin line between the legs (Sato ridge filter). Clean slices are unchanged.
  - On `thighFat` (11 hand-corrected cases): SAT Dice 0.976 → 0.998, worst case 0.928 → 0.991. Voxels the annotators had to erase: 587k → 31k.
- **No more `-1` labels.** Bright specks outside the body were counted as IMAT, which made the muscle label negative there. They are now ignored.
- **No more crash on slices without a femur.** Previously one such slice stopped the whole Thigh Seg run; the slice is now segmented without bone.
- **Pure fat at the inner edge of the SAT ring is no longer counted as muscle.** The ring clean-up trims about a pixel off the inner edge, and IMAT is not searched within 2 pixels of the ring, so that fat ended up as muscle. Fat in that band that is connected to the ring is now SAT. Found with the scanner's fat-fraction map: pixels that are at least 50% fat but labelled muscle went from 60 to 16 cm³ per case on `rawThigh`; SAT +4.7%, muscle −5.0%, IMAT +1.7%.
- **New optional parameters:** `segment()` and `segment_stack()` take `sat_thr` / `imat_thr` (defaults 100 / 80, the previous fixed values), and `segment()` can also return the body outline (`return_outer=True`). Called with the defaults, the output is identical to before these parameters existed (checked bit for bit on all 11 `thighFat` cases, before the inner-edge change).

### Dixon IMAT check (new: `mriFat/dixon_local_imat.py`)

- IMAT pixels are compared with the median of the surrounding muscle within 10 mm. IMAT that is not at least 25% of the way from the local muscle level to the local fat level becomes muscle. Muscle and SAT are never changed.
- **On by default** (`ENABLED = True`), in Thigh Seg and Combined for Dixon slices. It lowers the Dixon IMAT by about 50–75% (muscle goes up by the same volume; SAT is unchanged).
- The 25% level was set against the scanner's fat-fraction map (`6pt_DIXON_VIBE_FF`) on `rawThigh` (8 cases). Without the check, about half of the Dixon IMAT pixels were less than 30% fat. With it, 97–98% are at least 30% fat (median 85–100% fat), the IMAT volume equals the volume of pixels that are at least 50% fat (1.03×, 0.94–1.11× per case), and the overlap with those pixels is Dice 0.73–0.84.
- To turn it off: `ENABLED = False`. To remove it: delete the file and the three lines marked `dixon_local_imat` in `readSpace_threeButton.py`.

### T1 TSE (`t1_tse_tra`) thigh segmentation (new: `mriFat/t1_seg.py`)

A separate method, so changes to it cannot affect Dixon results. On T1 the fat/muscle contrast is lower and coil sensitivity makes one side of the thigh much brighter, so the Dixon thresholds do not work. Steps:

1. Resample to 0.8 mm (the Dixon pixel size the `seg.py` settings were tuned at).
2. Denoise with non-local means at 3× the estimated noise. (BM3D gave about the same result at ~15× the run time.)
3. Divide by the local fat signal (20 mm), so fat is ~1 and muscle ~0.35 everywhere.
4. Segment with `seg.segment_stack()` at 0.40 of the fat signal for SAT and IMAT (including the hip-end handling).
5. Check each IMAT pixel against its neighbours (10 mm, 40%, specks of at least 5 pixels). Some muscle groups (hamstrings, adductors) are brighter on T1 without containing fat; this step keeps them, and specks of grainy muscle, from being counted as IMAT.

The 40% level was tuned so the T1 IMAT matches the Dixon IMAT of the same scans (see below).

### Agreement between Dixon and T1 TSE

The Dixon method is the reference, because its IMAT can be checked against the fat-fraction map. On `rawThigh` (8 cases scanned with both sequences, same slice positions), T1 compared with Dixon:

| | Mean difference (T1 − Dixon) | Per case | Dice |
|---|---|---|---|
| SAT | −3.7% | −5.2% to −2.0% | 0.955 |
| IMAT | +3.4% | −12.7% to +18.5% | 0.504 |
| Muscle | +2.3% | +0.5% to +4.9% | 0.948 |

IMAT agrees in volume but only moderately pixel by pixel: T1 contrast cannot tell partly-fat pixels from muscle as well as the Dixon fat image (a third of the T1 IMAT pixels are less than 30% fat, vs 2–3% for Dixon). The T1 level (40%) is tuned to the Dixon method; if the Dixon method changes, it has to be re-tuned.

### GUI (`mriFat/readSpace_threeButton.py`)

- **Loading in thigh mode:** loads `6pt_DIXON_VIBE_F`, or, if the folder has none, any series whose name contains `t1_tse_tra` (case-insensitive). The app remembers which sequence was loaded.
- **Loading in abdomen mode:** still `_F` only. A folder with only `t1_tse_tra` offers to switch to thigh mode instead of failing.
- **Thigh Seg** uses `seg.py` + `dixon_local_imat.py` for Dixon slices and `t1_seg.py` for T1 slices.
- **AI Seg and Combined** refuse T1 slices with a message, because the U-Net was trained on Dixon fat images only. Combined now also uses the stack-level Dixon segmentation and the IMAT check.
- **JPEG-compressed DICOMs** (e.g. JPEG Lossless) can now be read: `pylibjpeg` and `pylibjpeg-libjpeg` were added to `requirements.txt` and installed in the bundled Python.
- **DICOM reading:** the series name is checked before pixels are decoded (faster when a folder holds many series), and when no matching series is found the error lists the series that are in the folder.

### Known issues

- **Hip end:** in 1 of the 11 `thighFat` cases, the skin line between the legs is too faint on the top few slices, and part of the other leg is still counted as SAT.
- **Validation:** the Dixon hip-end settings were tuned and checked on the same 11 `thighFat` cases, whose labels are corrected output of this same method. The Dixon IMAT check and the inner-edge change were checked against the scanner's fat-fraction map; the T1 settings were checked against the Dixon method. There are no hand-corrected T1 labels. IMAT overlap between the sequences is moderate (Dice ~0.5).
