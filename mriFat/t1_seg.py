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
     (fat ring, femur, IMAT, muscle, and the hip-end handling of the other leg). The SAT
     ring is found on a copy blurred by SAT_BLUR_PX (0 = off). seg.py keeps in SAT only fat
     branches at least ~7 pixels wide all the way from the ring; T1 is sharper than Dixon
     and shows the thin dark fascia where a deep fat pocket joins the ring, so T1 cut such
     pockets off as IMAT where Dixon kept them as SAT, and T1 SAT ran 3.7% below Dixon. The
     blur gives T1 about Dixon's sharpness for this step only; IMAT and muscle use the sharp
     image. What the blur adds to SAT must be above SAT_BLUR_GUARD in the sharp image: some
     T1 muscle groups pass the 0.40 level, and the blur let them join the SAT in one scan.
     Likewise SAT needs a core above SAT_CORE connected to the ring; between SAT_FRACTION
     and SAT_CORE a pixel is SAT only within 2 pixels of the core, so a broad, uniformly
     bright muscle group without fat in it is not taken as SAT.
  5. IMAT pixels are checked against their neighbours. Some muscle groups (e.g.
     hamstrings / adductors) are uniformly brighter on T1 without containing fat, which a
     fat-only reference cannot tell from IMAT. Local muscle level = median of the muscle +
     IMAT pixels within LOCAL_RADIUS_MM; an IMAT pixel stays IMAT when it lies at least
     LOCAL_CONTRAST of the way from that level to the local fat level, in a speck of at
     least MIN_IMAT_PIXELS pixels, otherwise it becomes muscle. Only IMAT is re-checked:
     muscle never becomes IMAT here, and SAT is never touched. The GUI's IMAT Contrast
     slider changes LOCAL_CONTRAST for one stack (relabel_imat, no new segmentation).
  6. Labels are resized back to the original grid.

LOCAL_CONTRAST was tuned on rawThigh (8 cases with both sequences, same slice positions)
against the Dixon method of the same scans (seg.py + dixon_local_imat.py, whose IMAT matches
the pixels that are at least 50% fat on the scanner's fat-fraction map). T1 compared with
Dixon (SAT blur 0.5, guard 0.6, core 0.6): SAT -2.9%, IMAT -2.0% (-15.4% to +10.8% per
case, mean |difference| 9.0%), muscle +1.8%; Dice SAT 0.967, IMAT 0.51, muscle 0.95.
Without blur and core check: SAT -3.7%, IMAT +4.3% (-12.2% to +22.0%, mean |difference| 10.9%), muscle +2.3%. T1 cannot tell
partly-fat pixels from muscle as well as Dixon, so its IMAT matches in volume rather than
pixel by pixel.

Labels: 0 background / bone, 1 SAT, 2 IMAT, 3 muscle.
"""
import numpy as np
import cv2
from skimage.filters import threshold_otsu
from skimage.restoration import denoise_nl_means, estimate_sigma
from scipy import ndimage as ndi

import seg
from parallel import map_slices

SAT_FRACTION = 0.40    # fat fraction threshold for the SAT ring
IMAT_FRACTION = 0.40   # first-pass IMAT threshold inside the thigh (refined in step 5)
FAT_SMOOTH_MM = 20.0   # scale of the local fat signal (coil sensitivity varies slowly)
TARGET_SPACING_MM = 0.8  # Dixon pixel size the seg.py settings were tuned at
DENOISE_STRENGTH = 3.0   # non-local means strength, in units of the estimated noise; 0 = off
LOCAL_RADIUS_MM = 10.0   # neighbourhood for the local muscle level; None = skip step 5
LOCAL_CONTRAST = 0.40    # IMAT stays IMAT if this far from local muscle (0) towards local fat (1);
                         # lower values keep specks of grainy muscle as IMAT
MIN_IMAT_PIXELS = 5      # smaller IMAT specks (at 0.8 mm pixels) count as muscle
SAT_BLUR_PX = 0.5        # Gaussian sigma (0.8 mm pixels) of the image the SAT ring is found on; 0 = off
SAT_BLUR_GUARD = 0.6     # SAT the blur adds must be at least this fat fraction in the sharp image
SAT_CORE = 0.6           # SAT needs a core above this connected to the ring; between SAT_FRACTION
                         # and this only within 2 pixels of the core; 0 = off
SAT_RADIUS_MM = 0        # SAT local-contrast check: neighbourhood (mm) of the local muscle level; 0 = off
SAT_LOCAL_CONTRAST = 0.1 # with it on, SAT stays SAT if this far from local muscle (0) towards local fat (1)


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
    return apply_contrast(labels, imat_contrast(image, fat_level, labels, radius_px), contrast_thr, min_pixels)


def imat_contrast(image, fat_level, labels, radius_px):
    """Per pixel: how far from the local muscle level (0) towards the local fat level (1).

    None for a slice without fat or without muscle (refine_imat leaves those unchanged).
    """
    inside = (labels == 2) | (labels == 3)
    if fat_level is None or not inside.any():
        return None
    # Pixels outside the thigh (SAT, bone, background) would pull the median towards fat or
    # zero; replace them by the smoothed value of the nearby inside pixels first
    m = inside.astype(np.float32)
    fill = cv2.GaussianBlur(image * m, (0, 0), radius_px) / np.maximum(cv2.GaussianBlur(m, (0, 0), radius_px), 1e-3)
    values = np.where(inside, image, fill)
    # median over a disc-sized square window (cv2 needs 8-bit for large windows)
    scale = np.percentile(values[inside], 99.5) * 1.2
    q = np.clip(values / scale * 255, 0, 255).astype(np.uint8)
    muscle_level = cv2.medianBlur(q, 2 * int(radius_px) + 1).astype(np.float32) / 255 * scale

    return ((image - muscle_level) / np.maximum(fat_level - muscle_level, 1e-3)).astype(np.float32)


def apply_contrast(labels, contrast, contrast_thr=None, min_pixels=None):
    """IMAT (label 2) stays IMAT where contrast > contrast_thr, in specks of at least min_pixels;
    the rest of it becomes muscle. Muscle (3) and SAT (1) are never changed.

    contrast_thr / min_pixels default to LOCAL_CONTRAST / MIN_IMAT_PIXELS.
    """
    contrast_thr = LOCAL_CONTRAST if contrast_thr is None else contrast_thr
    min_pixels = MIN_IMAT_PIXELS if min_pixels is None else min_pixels
    if contrast is None:
        return labels
    red = labels == 2
    imat = red & (contrast > contrast_thr)
    pieces, n = ndi.label(imat)
    if n:
        sizes = np.bincount(pieces.ravel())
        imat &= sizes[pieces] >= min_pixels

    out = labels.copy()
    out[red & ~imat] = 3
    return out


def check_sat(image, fat_level, labels, radius_px, contrast_thr=None):
    """SAT local-contrast check (off by default, SAT_RADIUS_MM = 0).

    A SAT pixel stays SAT when it lies at least contrast_thr (default SAT_LOCAL_CONTRAST) of
    the way from the local muscle level (median of muscle + IMAT within radius_px, as in the
    IMAT check) to the local fat level. Failing pixels become background where they touch the
    outside of the thigh (skin edge), otherwise muscle. On rawThigh it only removes partly fat
    edge pixels, so T1 SAT moves away from Dixon (-2.9% -> -3.7% at 10 mm / 0.1).
    """
    contrast_thr = SAT_LOCAL_CONTRAST if contrast_thr is None else contrast_thr
    contrast = imat_contrast(image, fat_level, labels, radius_px)
    if contrast is None:
        return labels
    low = (labels == 1) & (contrast < contrast_thr)
    if not low.any():
        return labels
    out = np.asarray(labels).copy()
    outside = ~ndi.binary_fill_holes(labels > 0)
    pieces, _ = ndi.label(low)
    at_skin = np.unique(pieces[cv2.dilate(outside.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0])
    skin = np.isin(pieces, at_skin[at_skin > 0])
    out[low & skin] = 0
    out[low & ~skin] = 3
    return out


def _prepare(im, size):
    """Steps 1-3 for one slice: resample to `size` (h, w), denoise, divide by the local fat signal.

    Returns (denoised slice, local fat level or None, fat fraction image).
    """
    im = np.asarray(im, np.float32)
    if size != im.shape:
        im = cv2.resize(im, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    im = denoise(im)
    level = local_fat_level(im, TARGET_SPACING_MM)
    return im, level, (np.zeros_like(im) if level is None else im / level)


def relabel_imat(state, contrast_thr):
    """Labels (n_slices, H, W) from a stored IMAT state with another contrast setting.

    Only IMAT / muscle change; SAT, bone and the thigh outline come from the stored labels.
    state: from segment_stack(..., return_state=True) or dixon_local_imat.apply(..., return_state=True).
    """
    labels = np.array([apply_contrast(l, c, contrast_thr, state['min_pixels'])
                       for l, c in zip(state['labels'], state['contrast'])], dtype=np.int16)
    size = state.get('size')
    if size is None or labels.shape[1:] == size:
        return labels
    return np.array([cv2.resize(l.astype(np.uint8), (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
                     for l in labels], dtype=np.int16)


def segment_stack(slices, pixel_spacing=None, contrast_thr=None, return_state=False, sat_thr=None, radius_mm=None,
                  sat_radius_mm=None):
    """Segment a stack of axial T1 thigh slices: 1 SAT, 2 IMAT, 3 muscle.

    slices: sequence of 2D arrays. pixel_spacing: (row, col) spacing in mm of the
    slices (DICOM PixelSpacing); without it the slices are used at their own size.
    contrast_thr: IMAT contrast setting of step 5 (default LOCAL_CONTRAST).
    sat_thr: SAT fat fraction of step 4 (default SAT_FRACTION; the GUI's thigh SAT Threshold).
    radius_mm: neighbourhood of step 5 (default LOCAL_RADIUS_MM; the GUI's IMAT Radius).
    sat_radius_mm: neighbourhood of the SAT local-contrast check (default SAT_RADIUS_MM,
    0 = off; the GUI's SAT Radius), see check_sat().
    Returns an int16 array (n_slices, H, W) on the original grid; with return_state=True
    also the state for relabel_imat() (the GUI's IMAT Contrast slider).
    """
    h, w = np.shape(slices[0])
    if pixel_spacing is None:
        row_mm = col_mm = TARGET_SPACING_MM
    else:
        row_mm, col_mm = float(pixel_spacing[0]), float(pixel_spacing[1])
    small_h = max(1, int(round(h * row_mm / TARGET_SPACING_MM)))
    small_w = max(1, int(round(w * col_mm / TARGET_SPACING_MM)))

    n = len(slices)
    denoised, fat_levels, prepared = zip(*map_slices(_prepare, slices, [(small_h, small_w)] * n))

    labels = seg.segment_stack(prepared, sat_thr=SAT_FRACTION if sat_thr is None else sat_thr, imat_thr=IMAT_FRACTION,
                               sat_blur=SAT_BLUR_PX, sat_guard=SAT_BLUR_GUARD, sat_core=SAT_CORE)
    sat_radius_mm = SAT_RADIUS_MM if sat_radius_mm is None else sat_radius_mm
    if sat_radius_mm:
        labels = map_slices(check_sat, denoised, fat_levels, labels, [sat_radius_mm / TARGET_SPACING_MM] * n)
    radius_mm = LOCAL_RADIUS_MM if radius_mm is None else radius_mm
    if radius_mm:
        radius_px = radius_mm / TARGET_SPACING_MM
        contrast = map_slices(imat_contrast, denoised, fat_levels, labels, [radius_px] * n)
    else:
        contrast = [None] * n
    state = dict(labels=np.asarray(labels, np.int16), contrast=contrast, min_pixels=MIN_IMAT_PIXELS, size=(h, w))
    labels = relabel_imat(state, contrast_thr)
    return (labels, state) if return_state else labels
