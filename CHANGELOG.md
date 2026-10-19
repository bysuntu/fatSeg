# Changelog

## 2026-10-19: Thigh SAT / IMAT boundary: alternatives tested, current rule kept

The current rule (fat a 10 mm ball rolling in from outside can reach is SAT) was compared with alternatives on `rawThigh` (8 cases, Dixon and T1, T1 mapped onto the Dixon grid with the DICOM geometry) and on T1 images at full resolution, where the fascia lata is visible. No alternative was better; labels unchanged.

| Rule | IMAT Dixon / T1 (ml) | T1 vs Dixon IMAT mean \|diff\| | IMAT Dice | Seen on T1 |
|---|---|---|---|---|
| 10 mm ball (current) | 1041 / 978 | 7.8% | 0.543 | fat beneath the fascia in the grooves between the hamstrings is SAT |
| Convex hull of the muscle envelope: fat inside IMAT | 1807 / 1749 | 4.3% | 0.639 | bridges wide concavities: subcutaneous fat (fibrous SAT texture, great saphenous vein) becomes IMAT |
| 20 / 30 / 50 mm ball | 1301 / 1472 / 1705 (Dixon) | 7.0 / 6.5 / 8.4% | 0.59 | right at the back (fascia bridges the hamstring grooves), wrong on the inner side (fat around the saphenous vein) |
| Non-fat specks ≤ 10–150 mm² ignored by the ball (10 mm) | 1002–977 / 941–930 | 7.6–8.2% | 0.54 | small effect; the specks that matter sit at pocket mouths, partly fragments of the fascia and its vessels |
| 30 mm ball + specks ≤ 20 mm² ignored | 1215 / 1130 | 6.9% | 0.58 | most of the 30 mm ball's extra IMAT was caused by vessels in SAT; the rest is the groove fat above (both kinds) |
| All non-fat pieces except the main muscle mass ignored | 842 / 798 | 8.1% | 0.50 | separate muscles (sartorius, gracilis, single hamstring bellies) no longer stop the ball: fat inside the fascia becomes SAT |

- **Agreement does not choose:** T1 vs Dixon agreement stays within noise for every rule, and rules that make IMAT larger raise Dice partly by adding regions both sequences label alike by construction (with the hull, the number of pixels only one sequence calls IMAT went up).
- **The current IMAT is a definition:** "fat not reachable by a 10 mm ball from outside", somewhat smaller than a fascia-based IMAT (Chaudry et al. 2020, MAGMA) in the posterior grooves. Matching the anatomy would need the fascia lata itself; it is visible as a continuous line on these T1 TSE images (not on Dixon).
- **`seg.BALL_HOLE_MM2`** (new, default 0 = off): non-fat specks of at most this area, separate from the main muscle mass, do not stop the ball. Kept as an option for further tests.

## 2026-10-15: Undo for manual edits; faster brush and AI Seg

- **Undo:** the Show Image window has an **Undo (Ctrl+Z)** button (with the number of steps). Each brush stroke or polygon is one step, up to 30; a stroke that changed nothing is not kept. Undo restores the slice and shows it. Steps belong to the segmentation they were made on: after a new Seg run, a loaded file or a flip they are dropped.
- **Brush:** distances are computed only in the box around each stroke segment (it was the whole slice), and the editor redraws when Tk is idle instead of on every mouse move (`draw_idle`), so painting keeps up with the mouse. Painted pixels are the same as before (300 random strokes, including at the image edges).
- **AI Seg:** the slices are predicted 8 at a time instead of one by one (`pipeline.predict_ai`): ~2.4 s instead of ~11 s per 40-slice `rawThigh` scan (model already loaded), masks identical on 3 scans (120 slices).

## 2026-10-15: Batch segmentation, regression check, and fixes from a code review

- **New `batch_seg.py`:** segments every case in a folder without the GUI, with the GUI's default settings, and writes labels (`.nii.gz`), per-slice areas and a volume table. See README, Other scripts.
- **New `check_results.py`:** segments `rawThigh` (Dixon and T1) and `rawAbd` and compares labels and volumes with a stored local reference, so the effect of a code change on the results is seen before it is kept.
- **`mriFat/pipeline.py`:** the GUI's Thigh Seg and Combined, `batch_seg.py` and `check_results.py` run the same segmentation code. **`mriFat/dicom_io.py`:** DICOM loading moved out of the GUI file (`readSpace_threeButton.parseDicomFolder` still works).
- **Series loading:** a folder can hold several acquisitions with the same series name (`rawAbd/03260016NHCTHO/6pt_DIXON_VIBE`: #157 and #166, different matrix sizes; loading it used to fail). Only one acquisition is loaded; the GUI asks which (latest preselected), scripts use the latest or (`compare_overlap.py`, `make_abd_stl.py`) stop with the series listed. Slices are sorted along the slice normal instead of by `SliceLocation` (same order on all 33 series of `rawThigh` and `rawAbd`).
- **Saved segmentations:** saved as shown, laid out like the images (rows, cols, slices) whichever method made them (AI Seg / Combined used to be saved transposed), with an affine from the DICOM geometry (it was the identity, or the affine of a previously loaded file). Files saved before are read as before.
- **Orientation of AI Seg / Combined:** the segmentation's layout is now recorded; on square images (e.g. 320 × 320) AI Seg and Combined overlays were shown and painted transposed. Combined needs an AI segmentation (it used to accept a Thigh Seg one and transpose it wrongly). Swap XY is a plain transpose (it stretched labels on non-square slices).
- **Area plot:** y axis in cm² (was labelled "Mass (g)"); the title total is a volume in cm³ (slice areas × slice spacing; it was a sum of areas).
- **Long-axis line:** the clicked point used the row spacing for x and the column spacing for y.
- **Thigh, `seg.py`:** `_fix_bone` also repairs a slice where no marrow was found (it was skipped); `refine_fat_ring` no longer returns a tuple when the ring has no contour (the slice came out empty). Both labels unchanged on all 16 `rawThigh` stacks.

## 2026-10-13: Short- and long-axis views swapped

- **GUI: landscape window.** The main window used to open at the size of its contents, about square. It now opens at width : height = 1.6 (`WINDOW_ASPECT` in `readSpace_threeButton.py`), 88% of the screen height (at most 95% of the screen width), centred at the top; it can still be resized. E.g. 1689 × 1056 on a 1920 × 1200 screen, 1520 × 950 on 1920 × 1080.
- **GUI:** the axial (short-axis) slice view is now the large view in the middle of the window, with the three-button slice / crop-range slider below it, where the long-axis view and its slider were. The long-axis view and its slider moved to the top right, above the area plot. Nothing else changed.
- **Abdomen, not adopted:** a SAT/VAT vote over neighbouring slices (as for the thigh) was tested on all 10 `rawAbd` scans and made the result slightly worse against the hand-corrected files (VAT labelled SAT 110 → 118 cm³ over 3 slices, 196 cm³ over 5); the abdominal contents move between slices, and the across-slice ray check already uses the neighbours.

## 2026-10-13: Thigh: SAT vs IMAT by a 1 cm rolling ball, slice vote and border rule (both sequences)

- **Problem:** `seg.py` keeps fat in SAT if it is at least ~7 pixels wide all the way from the ring, so deep fat pockets between muscles were SAT or IMAT depending on the width of their narrowest point. That depends on image sharpness: on Dixon the thin fascia across a pocket's neck is faint and the pocket stayed SAT, on T1 it was cut off as IMAT (6–22 ml per `rawThigh` scan), and the decision could flip from slice to slice.
- **Change (`seg.separate_sat_imat`, the last step of `t1_seg.relabel_imat`, so Dixon and T1, Thigh Seg, Combined and the IMAT Contrast slider):** (1) fat that a ball of `BALL_DIAMETER_MM` = 10 mm rolling in from outside the body can reach is SAT, other fat IMAT (the ball may lie in fat and outside the body, not in muscle or bone); (2) `BALL_SLICE_VOTE`: each fat pixel takes the SAT / IMAT label most of its slice and the two neighbouring slices give it; (3) `IMAT_SAT_BORDER` = 0.5: IMAT pieces whose border is more than half SAT become SAT. Muscle and bone are not changed. The IMAT states carry the pixel area (`pixel_mm2`). `BALL_DIAMETER_MM = 0`, `IMAT_SAT_BORDER = 0` turn it off: labels then identical to before (all 8 `rawThigh` T1 scans). Same labels with 4 worker processes as with 1.
- **IMAT definition changes: ~30% more IMAT** (fat between muscles beneath the surface). `rawThigh` Dixon IMAT 801 → 1041 ml, T1 768 → 978 ml.
- **Result, T1 vs Dixon** (off → on): IMAT mean |difference| per scan 9.0% → 7.8%, spread 9.9% → 8.8%, mean −2.0% → −4.0%, Dice 0.513 → 0.546; SAT Dice 0.967 → 0.971; fat one sequence calls SAT and the other IMAT 77 → 98 ml. Dixon vs the fat-fraction map: Dice vs ≥50% fat 0.777 → 0.782, IMAT pixels under 30% fat 2.3% → 5.6% (partly fat pixels at muscle surfaces); T1 IMAT median fat on the Dixon map 62% → 68%.
- **Compared (not adopted):** a 3D sphere instead of the slice vote (IMAT Dice 0.522, more swaps); the 2D ball alone (0.533); removing small IMAT specks that do not continue into the neighbouring slices (IMAT vs muscle; Dice 0.521), which was briefly added and is removed again.

## 2026-10-13: Switching mode clears the views

- **GUI:** switching between abdomen and thigh mode used to keep the loaded series, the segmentation and all three views, so e.g. abdominal SAT/VAT labels stayed on screen (and in the plot) in thigh mode. Switching now clears the short- and long-axis series, the segmentation and the stored SAT/VAT and IMAT states, drawn lines, the flip setting, the long-axis view, slice view and plot, and closes the Show Image window (`clear_views()`). If a segmentation would be lost, it asks first ("No" keeps the mode and everything as it was). Loading a `t1_tse_tra` folder in abdomen mode, which switches to thigh mode itself, clears without asking.

## 2026-10-12: Thigh: "SAT Threshold" slider

- **Advanced:** the SAT Threshold slider, and a new **IMAT Radius (mm, T1)** slider (4–30 mm, default 10 mm; shown for `t1_tse_tra` only), are folded under an **Advanced** tick box below IMAT Contrast. Releasing either runs Thigh Seg again. `t1_seg.segment_stack()` takes `radius_mm` (default `LOCAL_RADIUS_MM`; labels unchanged at the default, checked on `01260017NHCSLX`). On that scan, T1 IMAT is 65.8 ml at 5 mm, 71.4 ml at 10 mm, ~72 ml from 15 to 30 mm.
- **SAT Radius (mm, T1, 0 = off)** under Advanced (T1 only, 0–30 mm, default 0 = off): local-contrast check for SAT (`t1_seg.check_sat`, `SAT_RADIUS_MM`, `SAT_LOCAL_CONTRAST` = 0.1; `segment_stack(..., sat_radius_mm=)`). A SAT pixel stays SAT if it lies at least 0.1 of the way from the local muscle level (median of muscle + IMAT within the radius) to the local fat level; failing pixels become background at the skin, muscle elsewhere. Off by default because on `rawThigh` it only removes partly fat edge pixels and moves T1 SAT away from Dixon (T1 − Dixon SAT −2.9% → −3.7% at 10 mm / 0.1, −4.6% at 0.2, −5.7% at 0.3; Dice 0.967 → 0.965; IMAT unchanged). Labels unchanged when off.

- **GUI:** thigh mode has a **SAT Threshold** slider below IMAT Contrast. Units per sequence, reset to the sequence's default on loading: Dixon fat (`_F`) raw signal 50–200 (default 100, `seg.segment_stack` `sat_thr`), `t1_tse_tra` fat fraction 0.30–0.60 (default 0.40, `t1_seg.SAT_FRACTION`). Releasing it runs Thigh Seg again with the new value (asks first if the segmentation was edited); Thigh Seg and Combined use it. Hidden in abdomen mode (`THIGH_SAT` in `readSpace_threeButton.py` holds the ranges).
- **`t1_seg.segment_stack()`** takes `sat_thr` (default `SAT_FRACTION`). At the defaults the labels are unchanged (checked on `01260017NHCSLX` T1).
- **Example, `01260017NHCSLX`** (SAT / IMAT ml; Dixon 1125 / 64 at 100): Dixon 70 → 1166 / 63, 130 → 1098 / 65; T1 0.35 → 1120 / 56, 0.40 → 1070 / 71, 0.45 → 1042 / 80, 0.30 → 1239 / 12 (SAT spreads into bright muscle; below about 0.35 check the image).

## 2026-10-12: Abdomen: "SAT Fat Fraction" slider

- **GUI:** below the VAT Fat Fraction slider, abdomen mode has a **SAT Fat Fraction** slider (0.20–0.60 in steps of 0.01, default 0.40): a pixel outside the SAT/VAT boundary is SAT when its signal is at least this share of the fat signal. It changes SAT only; the body outline and the boundary always use 0.4 (`FAT_FRACTION`), and VAT keeps its own slider. Releasing either slider relabels SAT and VAT together from the last SAT/AVT Seg (a few seconds; same confirmation as before if the segmentation was edited). SAT/AVT Seg uses both values. Hidden in thigh mode with the VAT slider.
- **`abd_seg.py`:** `segment_abdomen_stack()` and `relabel_vat()` take `sat_fraction` (default `FAT_FRACTION`). Below `FAT_FRACTION`, SAT may take fat up to `SAT_EDGE_PX` = 2 pixels outside the body outline: the partly fat pixels at the skin, which the outline at 0.4 leaves out, so lowering the value has an effect (without this, 0.25 added only 0.3%). The console line shows the SAT fraction used.
- **Results at the defaults are unchanged** (labels identical on the abdominal scan of `01260017NHCSLX`). On that scan: 0.25 → SAT +2.3% (91% of the added pixels at the skin, none taken from VAT), 0.30 → +1.5%, 0.50 → −2.5%, 0.60 → −5.2%; VAT unchanged.

## 2026-10-12: Thigh T1: SAT ring found on a slightly blurred image; SAT core check

- **Problem:** `seg.py` keeps a fat branch in SAT only if it is at least ~7 pixels wide all the way from the SAT ring; narrower branches, and thick fat behind a narrow neck, become IMAT. T1 is sharper than Dixon and shows the thin dark fascia where a deep fat pocket (e.g. between the hamstrings) joins the ring, so T1 cut such pockets off as IMAT where Dixon kept them as SAT: 6–22 ml of pure fat per `rawThigh` scan. T1 SAT also ran 3.7% below Dixon.
- **Change (`t1_seg.py`, Dixon unchanged):** the SAT ring is found on a copy blurred by `SAT_BLUR_PX` = 0.5 pixel (0.8 mm grid), which gives T1 about Dixon's sharpness for this decision; IMAT and muscle use the sharp image. What the blur adds to the SAT must be above `SAT_BLUR_GUARD` = 0.6 of the fat signal in the sharp image: some T1 muscle groups pass the 0.40 SAT level, and without the guard the blur let a whole muscle group join the SAT on ~10 slices of `01260017NHCSLX` (SAT +9%). `SAT_BLUR_PX = 0` turns it off. `seg.segment` / `segment_stack` take `sat_blur` and `sat_guard` (default 0 = off).
- **Result of the blur on `rawThigh`, T1 vs Dixon** (before → after): SAT −3.7% → −2.5%; IMAT mean |difference| per scan 10.9% → 9.0% (spread 11.4% → 9.8%; range −12.2…+22.0% → −15.4…+10.8%); muscle +2.3% → +1.5%; Dice SAT 0.964 → 0.965, IMAT 0.511 → 0.513, muscle 0.950 → 0.951. The two scans where T1 IMAT was lowest get 3–5 points lower (−12.2% → −15.4%, −10.3% → −15.2%) (`07260002NHCTET`, `07260009NHCMA9`); the high ones improve most (`01260017NHCSLX` +22.0% → +10.8%). Same labels with several worker processes as with one.
- **Tested and not adopted:** stronger blur (0.75–1.5 px) or no guard: SAT spreads into bright muscle; stricter width test (erosion 4–5), a smooth muscle envelope and a notch (convexity-defect) rule for SAT vs IMAT: none reduced the SAT/IMAT disagreement between the sequences.
- **SAT core check (`t1_seg.SAT_CORE` = 0.6; `seg.segment(..., sat_core=)`, default 0 = off, Dixon unchanged):** on the first 3 slices of `01260017NHCSLX` T1, with or without the blur, a uniformly bright muscle group (0.44–0.73 of the fat signal, median 0.53; real SAT 0.75–1.28, muscle ≤ 0.39) passed the 0.40 SAT level and was labelled SAT (~5,900 pixels a slice). SAT now needs a core above 0.6 connected to the ring; pixels between 0.40 and 0.6 count only within 2 pixels of it. Those slices drop to ~430 wrong pixels, like the rest of the stack; the other 7 scans hardly change. The removed region becomes muscle; fat strands inside it stay muscle too (the IMAT search ran before the check).
- **Final result on `rawThigh`, T1 vs Dixon** (blur + guard + core; before → after): SAT −3.7% → −2.9% (`01260017NHCSLX` −3.8% → −4.8%, in line with the others); IMAT as above; muscle +2.3% → +1.8%; Dice SAT 0.964 → 0.967, IMAT 0.511 → 0.513, muscle 0.950 → 0.952.

## 2026-10-12: Thigh: femur marrow no longer labelled IMAT

- **Bug:** on some slices the femur marrow was labelled IMAT and a small round fat spot nearby was taken as the bone. `seg.extract_bone_in_roi` picked the roundest bright spot (area / edge²), but dark specks inside the marrow counted as edge and lowered its score, and fat spots just over the 50-pixel minimum scored as high. On `rawThigh` this hit 19 Dixon slices (5 of 8 scans; 9 in `07260009NHCMA9`) and 6 T1 slices (4 scans). Each slice put about 130–250 mm² of marrow into IMAT.
- **Fix (`mriFat/seg.py`, so Dixon and T1 both):** candidates are scored with their holes filled; the marrow is the largest candidate scoring at least `BONE_SCORE_RATIO` = 0.35 of the roundest; the marrow is returned with its holes filled (the dark specks inside it used to count as muscle). The new `_fix_bone`, called by `segment_stack`, follows the marrow from the slice that agrees best with its neighbours outward, and where an IMAT spot covers the previous slice's marrow better than the slice's own bone, the two swap.
- **Result on `rawThigh`:** no slice with the marrow as IMAT in either sequence (was 25); no correctly found bone changed by more than 15%. Dixon IMAT −10 ml in total (811 → 801 ml; −1.4 ml on average per scan, up to −6 ml in `07260009NHCMA9`), T1 IMAT −2 ml; SAT unchanged, muscle within 3 ml. Dixon vs the scanner fat-fraction map is unchanged (IMAT Dice vs ≥50% fat 0.779 → 0.777, the marrow it loses was fat), and T1 vs Dixon IMAT Dice 0.508 → 0.511.

## 2026-10-11: Thigh: "IMAT Contrast" slider

- **GUI:** thigh mode has an **IMAT Contrast** slider (0.10–0.90 in steps of 0.01), the thigh counterpart of the abdomen's VAT Fat Fraction. It sets the level of the IMAT check: an IMAT pixel stays IMAT when it lies at least this far from the local muscle level towards the local fat level, otherwise it becomes muscle. The default is set when a series is loaded: 0.25 for Dixon fat (`_F`, `dixon_local_imat.CONTRAST`), 0.40 for `t1_tse_tra` (`t1_seg.LOCAL_CONTRAST`). Releasing the slider relabels IMAT / muscle in about a second from the last Thigh Seg; SAT, bone and the outline never change. It asks first if the segmentation was edited by hand or loaded since. Thigh Seg and Combined use the current value. Hidden in abdomen mode.
- **`t1_seg.py`:** the IMAT check is split into `imat_contrast()` (contrast map) and `apply_contrast()` (threshold); `refine_imat()` does both as before. `segment_stack()` takes `contrast_thr` and `return_state`, and the new `relabel_imat(state, contrast_thr)` redoes the check with another level. **`dixon_local_imat.apply()`** takes the same two arguments.
- **Results at the defaults are unchanged:** the labels of all 10 `rawThigh` scans (Dixon and T1) are identical to the previous code. On 01260012NHCCML, Dixon IMAT goes from 72,550 pixels at 0.15 to 28,007 at 0.60 (52,894 at the default), with SAT the same at every value.
- **Limit:** only the first pass's IMAT candidates are re-checked, so below about 0.15 the first-pass thresholds (`seg.py` `imat_thr = 80` for Dixon, `t1_seg.IMAT_FRACTION` for T1), not the slider, limit the IMAT. Testing every muscle pixel instead would add only 1–4 cm³ (Dixon) at 0.10 on `rawThigh`.

## 2026-10-11: Parallel segmentation, progress popup, GUI fixes

- **Per-slice work runs on several CPU cores** (new `mriFat/parallel.py`). In `abd_seg.py` (fat signal, SAT/VAT boundary, labels), `seg.py` (the per-slice part of `segment_stack`), `t1_seg.py` (resampling, denoising, IMAT check) and `dixon_local_imat.py`. Steps that compare neighbouring slices (across-slice ray check, hip-end handling) still run in order. **Results are identical** to running slice by slice (checked on two abdominal scans, VAT relabelling, and Dixon and T1 thigh). On a 4-core laptop: abdomen (72 slices) ~65 s → ~28 s, T1 thigh 9 s → 4 s, Dixon thigh 5 s → 3 s. The 7 worker processes start with the app and use ~50 MB each; they use one math-library thread each (otherwise ~500 MB each, at the same speed). `WORKERS = 1` in `parallel.py` turns it off. Scripts that segment must keep their work under `if __name__ == '__main__':`.
- **Progress popup:** SAT/AVT Seg, Thigh Seg, AI Seg, Combined and the VAT Fat Fraction relabelling run in the background, with a popup showing the elapsed time. The window stays responsive; the buttons are disabled until the run finishes.
- **TensorFlow is loaded on the first AI Seg click,** not at startup: the window opens in ~3 s instead of 10–20 s. The U-Net is loaded once and reused on later AI Seg clicks.
- **Loading another series** clears the previous segmentation. It used to stay: with fewer slices the area plot failed ("list index out of range"), with more it showed the old curves. A seg file whose slice count doesn't match the images is reported in the plot title.
- **Area plot:** the y-axis now also fits the blue curve (thigh muscle was cut off at the top).
- **Polygon tool:** applying a polygon is now immediate (was a pixel-by-pixel loop, seconds on 512×512 slices); same result.
- **Polygon and Brush buttons:**
  - **Icons:** the buttons show their icons again. They were looked up one folder above the folder the app was started from, so with `runFat.bat` they showed "Polygon" / "Brush" text. The icons (also `sat.png`, `thigh.png`) are now found next to `mriFat\` whatever the start folder.
  - **Polygon button:** clicking it starts a new polygon in the current colour; before, nothing happened until a colour button was clicked as well.
  - **Flipped segmentations:** after **Flip Seg X/Y**, polygons were applied transposed (and failed on non-square images), and brush strokes landed at mirrored positions on square images. Both tools now edit the pixels under the cursor, as the popup shows them.
  - **Brush:** paints the whole stroke between mouse positions as a round brush (it used to paint separate squares, leaving gaps when the mouse moved fast). It reaches the image edge, and the drawn stroke is as wide as the painted area. It is also instant instead of a per-pixel loop.
- **Show Image** without loaded slices shows a warning instead of failing (typo `showwarningcerr`).
- **`requirements.txt`: PyWavelets added.** The T1 thigh method's noise estimate (`skimage.restoration.estimate_sigma`) needs it; without it **Thigh Seg on T1 TSE failed** in a fresh installation. Run `pip install -r requirements.txt` again to get it.
- **INSTALL.md step 6 also checks `pylibjpeg`, `libjpeg` and `pywt`.** They were in `requirements.txt` but not in the check. An environment built before `pylibjpeg` was added cannot load JPEG Lossless DICOMs: e.g. the `6pt_DIXON_VIBE_F` series of `01260017NHCSLX` failed with "No valid DICOM files found" although the series is there.

## 2026-10-09: Abdomen: "VAT Fat Fraction" slider replaces "Segmentation Threshold"

- **GUI:** in abdomen mode the **Segmentation Threshold** box (a raw value, default 100) is replaced by a **VAT Fat Fraction** slider (0.20–0.60 in steps of 0.01, default 0.40): a pixel inside the SAT boundary counts as VAT when its signal is at least this share of the slice's fat signal. It changes VAT only; SAT and the SAT/VAT boundary always use 0.4 (`FAT_FRACTION`). The old box only fed the first pass that measures the fat signal and had little effect. The slider is shown in abdomen mode only and hidden in thigh mode, where nothing uses it. Releasing the slider (or an arrow key) relabels VAT with the new value straight away (about 3 s instead of a full SAT/AVT Seg, same result), reusing the last SAT/VAT boundary; it asks first if the segmentation was edited by hand or loaded since.
- **`mriFat/abd_seg.py`:** `segment_abdomen_stack()` takes a `vat_fraction` argument (default `FAT_FRACTION` = 0.4), used only for the VAT pixels. The raw first-pass threshold is now the module setting `FIRST_PASS_THRESHOLD` = 100 and the default of `threshold`. `segment_abdomen_stack(..., return_state=True)` also returns the fat signal and boundary, and the new `relabel_vat()` recomputes the labels from them with another VAT fat fraction. With the defaults the results are unchanged (checked bit for bit). The console line shows the SAT and VAT fat fractions and thresholds.

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
