"""
Find the series in a case folder and segment them the way the GUI does, without the GUI.

Used by batch_seg.py and check_results.py; the GUI's Thigh Seg calls segment_thigh() too, so
the GUI and the scripts run the same steps. All settings default to the GUI's defaults.

Labels are returned laid out like the images, (rows, cols, slices):
  thigh   1 SAT, 2 IMAT, 3 muscle      (seg.py + dixon_local_imat.py for Dixon fat, t1_seg.py for T1 TSE)
  abdomen 1 SAT, 2 VAT                 (abd_seg.py)
"""
import os

import numpy as np

import abd_seg
import dixon_local_imat
import seg
import t1_seg
from dicom_io import _name_matches, _series_description, parseDicomFolder, slice_spacing_mm

# Series the GUI loads, per mode: (sequence, parseDicomFolder filter)
SERIES = {'thigh': (('DIXON_F', dict(seriesName='6pt_DIXON_VIBE_F')),
                    ('T1_TSE', dict(seriesContains='t1_tse_tra'))),
          'abdomen': (('DIXON_F', dict(seriesSuffix='_F')),)}

LABELS = {'thigh': {1: 'SAT', 2: 'IMAT', 3: 'muscle'}, 'abdomen': {1: 'SAT', 2: 'VAT'}}

DIXON_SAT_THR = 100   # thigh Dixon SAT threshold (raw signal), the GUI's default


def find_series(case_dir, mode):
    """(folder, sequence, filter) for every series of `mode` in the subfolders of case_dir.

    Reads headers only. In thigh mode a folder can give both Dixon and T1 (the GUI loads Dixon
    first); each is returned, since both are wanted for comparisons.
    """
    found = []
    for sub in sorted(os.listdir(case_dir)):
        folder = os.path.join(case_dir, sub)
        if not os.path.isdir(folder):
            continue
        names = set()
        for root, _, files in os.walk(folder):
            for f in files:
                name = _series_description(os.path.join(root, f))
                if name is not None:
                    names.add(name)
        for sequence, kw in SERIES[mode]:
            if any(_name_matches(n, **kw) for n in names):
                found.append((folder, sequence, kw))
    return found


def latest_series(options):
    """choose_series policy for scripts: the acquisition with the highest series number."""
    print('  several series match: ' + ', '.join(f"#{o['number']} {o['slices']} slices" for o in options)
          + f"; using #{options[-1]['number']}")
    return len(options) - 1


def load(folder, kw, choose_series=latest_series):
    """Slices and per-slice info of one series (dicom_io.parseDicomFolder)."""
    return parseDicomFolder(folder, choose_series=choose_series, **kw)


def clipped_stack(pixels):
    """The stack as the GUI holds it (image_stack): clipped to its 1st-99th percentile."""
    stack = np.array(pixels)
    return np.clip(stack, np.percentile(stack, 1), np.percentile(stack, 99))


def segment_thigh(pixels, info, sequence, sat_thr=None, contrast=None, radius_mm=None, sat_radius_mm=None,
                  return_state=False):
    """Thigh Seg: labels (slices, rows, cols), 1 SAT, 2 IMAT, 3 muscle (and the IMAT state with return_state).

    sequence 'DIXON_F' or 'T1_TSE'. None = default: SAT threshold DIXON_SAT_THR / t1_seg.SAT_FRACTION,
    IMAT contrast dixon_local_imat.CONTRAST / t1_seg.LOCAL_CONTRAST, IMAT radius t1_seg.LOCAL_RADIUS_MM,
    SAT radius t1_seg.SAT_RADIUS_MM (radii: T1 only). Uses the raw pixels, as the GUI does.
    """
    spacing = info[0][4]
    if sequence == 'T1_TSE':
        result = t1_seg.segment_stack(pixels, pixel_spacing=spacing, contrast_thr=contrast, return_state=True,
                                      sat_thr=sat_thr, radius_mm=radius_mm, sat_radius_mm=sat_radius_mm)
    else:
        labels = seg.segment_stack(pixels, sat_thr=DIXON_SAT_THR if sat_thr is None else sat_thr)
        result = dixon_local_imat.apply(pixels, labels, spacing, contrast_thr=contrast, return_state=True)
    return result if return_state else result[0]


def segment_abdomen(pixels, info, vat_fraction=None, sat_fraction=None):
    """SAT/AVT Seg: labels (rows, cols, slices), 1 SAT, 2 VAT, on the clipped stack as in the GUI."""
    return abd_seg.segment_abdomen_stack(clipped_stack(pixels), pixel_spacing=float(info[0][4][0]),
                                         vat_fraction=abd_seg.FAT_FRACTION if vat_fraction is None else vat_fraction,
                                         sat_fraction=abd_seg.FAT_FRACTION if sat_fraction is None else sat_fraction)


def segment(pixels, info, mode, sequence):
    """Default segmentation of one series, laid out like the images: (rows, cols, slices), int16."""
    if mode == 'thigh':
        labels = np.asarray(segment_thigh(pixels, info, sequence)).transpose(1, 2, 0)
    else:
        labels = segment_abdomen(pixels, info)
    return np.ascontiguousarray(labels, dtype=np.int16)


def slice_areas(labels, info, mode):
    """Per-slice area (cm^2) of each label: {name: array over slices}. labels: (rows, cols, slices)."""
    pixel_cm2 = np.array([float(k[4][0]) * float(k[4][1]) * 1e-2 for k in info])
    return {name: (labels == value).sum(axis=(0, 1)) * pixel_cm2 for value, name in LABELS[mode].items()}


def volumes(labels, info, mode):
    """Volume (cm^3) of each label: slice areas times the slice spacing, as the GUI's plot."""
    depth_cm = slice_spacing_mm(info) * 0.1
    return {name: float(a.sum() * depth_cm) for name, a in slice_areas(labels, info, mode).items()}
