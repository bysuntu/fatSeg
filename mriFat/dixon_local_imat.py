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

ENABLED = False turns it off (Dixon results are then those of seg.py). To remove it
completely: delete this file and the three lines marked "dixon_local_imat" in
readSpace_threeButton.py.
"""
import numpy as np

import t1_seg

ENABLED = True
RADIUS_MM = 10.0   # neighbourhood for the local muscle level
CONTRAST = 0.25    # IMAT stays IMAT if this far from local muscle (0) towards local fat (1);
                   # set so IMAT matches the pixels that are at least 50% fat (fat-fraction map)
MIN_PIXELS = 5     # smaller IMAT specks count as muscle


def apply(slices, labels, pixel_spacing):
    """Return labels (n_slices, H, W) with IMAT / muscle re-decided, or unchanged when disabled.

    slices: the Dixon fat slices the labels were computed from; pixel_spacing: (row, col) mm.
    """
    if not ENABLED:
        return labels
    radius_px = RADIUS_MM / float(pixel_spacing[0])
    out = []
    for image, lab in zip(slices, labels):
        image = np.asarray(image, np.float32)
        level = t1_seg.local_fat_level(image, float(pixel_spacing[0]))
        out.append(t1_seg.refine_imat(image, level, lab, radius_px, CONTRAST, MIN_PIXELS))
    return np.array(out, dtype=np.int16)
