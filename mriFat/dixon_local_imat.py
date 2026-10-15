"""
Optional helper for the Dixon fat (_F) thigh segmentation: check IMAT pixels against their
neighbours (the step t1_seg.py uses for T1 TSE). IMAT that is not at least CONTRAST of the
way from the local muscle level to the local fat level becomes muscle; muscle and SAT are
never changed. Used by Thigh Seg and Combined for Dixon slices.

CONTRAST was set against the scanner's fat-fraction map (6pt_DIXON_VIBE_FF) on rawThigh
(8 cases): at 0.25 the IMAT volume equals the volume of pixels that are at least 50% fat
(1.03x, 0.94-1.11x per case), with overlap (Dice) 0.78 with those pixels and 0.77 with
pixels that are at least 30% fat. Without this check, half of the Dixon IMAT pixels were
less than 30% fat.

ENABLED = False turns it off: the IMAT check is skipped (seg.py's labels then only go through
seg.separate_sat_imat, the SAT / IMAT step that t1_seg.relabel_imat always runs). It is called
from pipeline.segment_thigh (Thigh Seg, Combined, batch_seg.py, check_results.py).
"""
import numpy as np

import t1_seg
from parallel import map_slices

ENABLED = True
RADIUS_MM = 10.0   # neighbourhood for the local muscle level
CONTRAST = 0.25    # IMAT stays IMAT if this far from local muscle (0) towards local fat (1);
                   # set so IMAT matches the pixels that are at least 50% fat (fat-fraction map)
MIN_PIXELS = 5     # smaller IMAT specks count as muscle


def apply(slices, labels, pixel_spacing, contrast_thr=None, return_state=False):
    """Return labels (n_slices, H, W) with IMAT / muscle re-decided, or unchanged when disabled.

    slices: the Dixon fat slices the labels were computed from; pixel_spacing: (row, col) mm.
    contrast_thr: IMAT contrast setting (default CONTRAST). With return_state=True also returns
    the state for t1_seg.relabel_imat() (the GUI's IMAT Contrast slider); when disabled its
    contrast maps are None, so relabelling leaves the labels unchanged.
    """
    n = len(slices)
    if ENABLED:
        contrast = map_slices(_contrast_one, slices, labels, [float(pixel_spacing[0])] * n)
    else:
        contrast = [None] * n
    state = dict(labels=np.asarray(labels, np.int16), contrast=contrast, min_pixels=MIN_PIXELS, size=None,
                 pixel_mm2=float(pixel_spacing[0]) * float(pixel_spacing[1]))
    out = t1_seg.relabel_imat(state, CONTRAST if contrast_thr is None else contrast_thr)
    return (out, state) if return_state else out


def _contrast_one(image, lab, spacing_mm):
    """Contrast map of one slice for the IMAT check (None if the slice has no fat or muscle)."""
    image = np.asarray(image, np.float32)
    level = t1_seg.local_fat_level(image, spacing_mm)
    return t1_seg.imat_contrast(image, level, lab, RADIUS_MM / spacing_mm)
