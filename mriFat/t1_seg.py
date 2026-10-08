"""
Thigh SAT / IMAT / muscle segmentation for axial T1 TSE slices (series "t1_tse_tra").

Kept separate from the Dixon fat (_F) method so changes here cannot affect it. The
segmentation steps are the ones in seg.py (seg.segment_stack, called with its own
thresholds); only the preparation of the image differs.

Why the Dixon fat thresholds do not work on T1:
  On the Dixon fat image the darkest fat is ~3x brighter than the brightest muscle.
  On T1 the contrast is lower (muscle ~1/3 of fat) and coil sensitivity makes one side
  of the thigh much brighter than the other, so the dim side of the SAT ring overlaps
  the bright side of the muscle. No single intensity threshold separates them.

Idea:
  1. Resample each slice to the Dixon pixel size (0.8 mm): the pixel-based settings in
     seg.py (smallest region, ring refinement, slice-to-slice shift, ridge filter) were
     tuned at that size.
  2. Denoise with non-local means at DENOISE_STRENGTH x the estimated noise level. This
     mainly calms speckle in the muscle that would otherwise count as IMAT (BM3D did
     about as well, at ~15x the run time).
  3. Fat fraction image: divide each slice by its local fat signal, the mean of nearby
     confident fat (above the slice's Otsu threshold) smoothed over FAT_SMOOTH_MM.
     Fat becomes ~1 everywhere, muscle ~0.35, whatever the coil sensitivity.
  4. seg.segment_stack on that image with SAT_FRACTION / IMAT_FRACTION as thresholds
     (fat ring, femur, IMAT, muscle, and the hip-end handling of the other leg).
  5. IMAT pixels are checked against their neighbours. Some muscle groups (e.g.
     hamstrings / adductors) are uniformly brighter on T1 without containing fat, which a
     fat-only reference cannot tell from IMAT. Local muscle level = median of the muscle +
     IMAT pixels within LOCAL_RADIUS_MM; an IMAT pixel stays IMAT when it lies at least
     LOCAL_CONTRAST of the way from that level to the local fat level, in a speck of at
     least MIN_IMAT_PIXELS pixels, otherwise it becomes muscle. Only IMAT is re-checked:
     muscle never becomes IMAT here, and SAT is never touched.
  6. Labels are resized back to the original grid.

LOCAL_CONTRAST was tuned on rawThigh (8 cases with both sequences, same slice positions)
against the Dixon method of the same scans (seg.py + dixon_local_imat.py, whose IMAT matches
the pixels that are at least 50% fat on the scanner's fat-fraction map). T1 compared with
Dixon: SAT -3.7%, IMAT +3.4% (-12.7% to +18.5% per case), muscle +2.3%; Dice SAT 0.96,
IMAT 0.50, muscle 0.95. T1 cannot tell partly-fat pixels from muscle as well as Dixon, so
its IMAT matches in volume rather than pixel by pixel.

Labels: 0 background / bone, 1 SAT, 2 IMAT, 3 muscle.
"""
import numpy as np
import cv2
from skimage.filters import threshold_otsu
from skimage.restoration import denoise_nl_means, estimate_sigma
from scipy import ndimage as ndi

import seg

SAT_FRACTION = 0.40    # fat fraction threshold for the SAT ring
IMAT_FRACTION = 0.40   # first-pass IMAT threshold inside the thigh (refined in step 5)
FAT_SMOOTH_MM = 20.0   # scale of the local fat signal (coil sensitivity varies slowly)
TARGET_SPACING_MM = 0.8  # Dixon pixel size the seg.py settings were tuned at
DENOISE_STRENGTH = 3.0   # non-local means strength, in units of the estimated noise; 0 = off
LOCAL_RADIUS_MM = 10.0   # neighbourhood for the local muscle level; None = skip step 5
LOCAL_CONTRAST = 0.40    # IMAT stays IMAT if this far from local muscle (0) towards local fat (1);
                         # lower values keep specks of grainy muscle as IMAT
MIN_IMAT_PIXELS = 5      # smaller IMAT specks (at 0.8 mm pixels) count as muscle


def denoise(image):
    """Non-local means at DENOISE_STRENGTH x the noise level estimated from the slice."""
    image = np.asarray(image, np.float32)
    scale = np.percentile(image, 99)
    if DENOISE_STRENGTH <= 0 or scale <= 0:
        return image
    x = image / scale
    sigma = DENOISE_STRENGTH * float(estimate_sigma(x))
    if sigma <= 0:
        return image
    x = denoise_nl_means(x, h=0.8 * sigma, sigma=sigma, patch_size=5, patch_distance=6, fast_mode=True)
    return (x * scale).astype(np.float32)


def local_fat_level(image, spacing_mm):
    """Local fat signal of a slice: mean of nearby confident fat, or None for a blank slice."""
    image = np.asarray(image, np.float32)
    if image.max() <= image.min():
        return None
    fat = (image > threshold_otsu(image)).astype(np.float32)
    sigma = FAT_SMOOTH_MM / spacing_mm
    local = cv2.GaussianBlur(image * fat, (0, 0), sigma) / np.maximum(cv2.GaussianBlur(fat, (0, 0), sigma), 1e-3)
    # far from any fat (background) the estimate is meaningless; keep it from blowing up
    return np.maximum(local, 0.2 * np.median(image[fat > 0]))


def fat_fraction(image, spacing_mm):
    """Slice divided by its local fat signal: fat ~1, muscle ~0.35."""
    image = np.asarray(image, np.float32)
    local = local_fat_level(image, spacing_mm)
    return np.zeros_like(image) if local is None else image / local


def refine_imat(image, fat_level, labels, radius_px, contrast_thr=None, min_pixels=None):
    """Check IMAT pixels against their neighbours; those not clearly brighter become muscle.

    contrast_thr / min_pixels default to LOCAL_CONTRAST / MIN_IMAT_PIXELS.
    """
    contrast_thr = LOCAL_CONTRAST if contrast_thr is None else contrast_thr
    min_pixels = MIN_IMAT_PIXELS if min_pixels is None else min_pixels
    inside = (labels == 2) | (labels == 3)
    if fat_level is None or not inside.any():
        return labels
    # Pixels outside the thigh (SAT, bone, background) would pull the median towards fat or
    # zero; replace them by the smoothed value of the nearby inside pixels first
    m = inside.astype(np.float32)
    fill = cv2.GaussianBlur(image * m, (0, 0), radius_px) / np.maximum(cv2.GaussianBlur(m, (0, 0), radius_px), 1e-3)
    values = np.where(inside, image, fill)
    # median over a disc-sized square window (cv2 needs 8-bit for large windows)
    scale = np.percentile(values[inside], 99.5) * 1.2
    q = np.clip(values / scale * 255, 0, 255).astype(np.uint8)
    muscle_level = cv2.medianBlur(q, 2 * int(radius_px) + 1).astype(np.float32) / 255 * scale

    contrast = (image - muscle_level) / np.maximum(fat_level - muscle_level, 1e-3)
    # Only IMAT (label 2) is re-checked: it stays IMAT where it is brighter than its
    # neighbours, otherwise it becomes muscle. Muscle (3) and SAT (1) are never changed.
    red = labels == 2
    imat = red & (contrast > contrast_thr)
    pieces, n = ndi.label(imat)
    if n:
        sizes = np.bincount(pieces.ravel())
        imat &= sizes[pieces] >= min_pixels

    out = labels.copy()
    out[red & ~imat] = 3
    return out


def segment_stack(slices, pixel_spacing=None):
    """Segment a stack of axial T1 thigh slices: 1 SAT, 2 IMAT, 3 muscle.

    slices: sequence of 2D arrays. pixel_spacing: (row, col) spacing in mm of the
    slices (DICOM PixelSpacing); without it the slices are used at their own size.
    Returns an int16 array (n_slices, H, W) on the original grid.
    """
    h, w = np.shape(slices[0])
    if pixel_spacing is None:
        row_mm = col_mm = TARGET_SPACING_MM
    else:
        row_mm, col_mm = float(pixel_spacing[0]), float(pixel_spacing[1])
    small_h = max(1, int(round(h * row_mm / TARGET_SPACING_MM)))
    small_w = max(1, int(round(w * col_mm / TARGET_SPACING_MM)))

    denoised, fat_levels, prepared = [], [], []
    for im in slices:
        im = np.asarray(im, np.float32)
        if (small_h, small_w) != (h, w):
            im = cv2.resize(im, (small_w, small_h), interpolation=cv2.INTER_AREA)
        im = denoise(im)
        level = local_fat_level(im, TARGET_SPACING_MM)
        denoised.append(im)
        fat_levels.append(level)
        prepared.append(np.zeros_like(im) if level is None else im / level)

    labels = seg.segment_stack(prepared, sat_thr=SAT_FRACTION, imat_thr=IMAT_FRACTION)
    if LOCAL_RADIUS_MM:
        radius_px = LOCAL_RADIUS_MM / TARGET_SPACING_MM
        labels = np.array([refine_imat(im, level, lab, radius_px)
                           for im, level, lab in zip(denoised, fat_levels, labels)], dtype=np.int16)
    if (small_h, small_w) == (h, w):
        return labels
    return np.array([cv2.resize(l.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                     for l in labels], dtype=np.int16)
