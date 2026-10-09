"""
Compare SAT / VAT volumes between the AbdoCompL3 and T12-S1 DIXON VIBE sequences
of each case, over the region the two scans have in common.

Usage:
    python compare_overlap.py E:\\after
    python compare_overlap.py E:\\after --csv overlap_volumes.csv

Expected layout: <root>/<case>/<series folder>/ containing the DICOM files and the
seg.nii.gz saved by the GUI (labels: 1 = SAT, 2 = VAT). Series folders are found by
name: AbdoCompL3* and T12S1* (underscores ignored).

Overlap is taken in scanner coordinates (no motion correction). Each scan is cut to the slab
the other scan covers (its first to last slice, +- half a slice thickness), measured along that
scan's own slice direction, so scans tilted against each other compare the same region (e.g.
03260016NHCTHO: 3.9 deg). Voxels only partly inside count by the fraction inside. For two
axial scans this is the common z range. Differences are T12 - L3, as a percentage of the mean
of the two. 'slab_from' / 'slab_to' are the limits along the L3 slice direction (z if axial).
"""
import argparse
import contextlib
import csv
import io
import os
import sys

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mriFat'))
from readSpace_threeButton import parseDicomFolder

SERIES = (('L3', 'abdocompl3'), ('T12', 't12s1'))


def load_case(case_dir):
    """Return {'L3': ..., 'T12': ...} with segmentation and slice geometry."""
    scans = {}
    for sub in sorted(os.listdir(case_dir)):
        name = sub.lower().replace('_', '')
        key = next((k for k, prefix in SERIES if name.startswith(prefix)), None)
        if key is None:
            continue
        folder = os.path.join(case_dir, sub)
        seg_path = os.path.join(folder, 'seg.nii.gz')
        if not os.path.isfile(seg_path):
            print(f'  missing {seg_path}')
            continue
        # Geometry from the _F DICOM series, sorted the same way the GUI sorts them
        with contextlib.redirect_stdout(io.StringIO()):
            _, info = parseDicomFolder(folder, seriesSuffix='_F')
        row_dir = np.array(info[0][2][:3], float)                      # ImageOrientationPatient
        col_dir = np.array(info[0][2][3:], float)
        scans[key] = dict(seg=nib.load(seg_path).get_fdata(),
                          first=np.array(info[0][3], float),            # ImagePositionPatient
                          last=np.array(info[-1][3], float),
                          row_dir=row_dir, col_dir=col_dir, normal=np.cross(row_dir, col_dir),
                          spacing=[float(v) for v in info[0][4]],       # PixelSpacing
                          thickness=float(info[0][5]))                  # SliceThickness
    return scans


def slab(s, normal):
    """Range covered by scan s along a direction: first to last slice, +- half a thickness."""
    a, b = np.dot(s['first'], normal), np.dot(s['last'], normal)
    return min(a, b) - s['thickness'] / 2, max(a, b) + s['thickness'] / 2


def fraction_inside(s, other):
    """Share of each voxel of s (rows, cols, slices) inside the slab covered by `other`."""
    normal = other['normal']
    lo, hi = slab(other, normal)
    rows, cols, n = s['seg'].shape
    step = (s['last'] - s['first']) / max(n - 1, 1)
    rr, cc = np.mgrid[0:rows, 0:cols]
    depth = (np.dot(s['first'], normal)
             + np.dot(s['col_dir'], normal) * s['spacing'][0] * rr[..., None]
             + np.dot(s['row_dir'], normal) * s['spacing'][1] * cc[..., None]
             + np.dot(step, normal) * np.arange(n)[None, None, :])
    extent = s['thickness'] * abs(np.dot(s['normal'], normal))           # voxel extent along `normal`
    return np.clip((np.minimum(depth + extent / 2, hi) - np.maximum(depth - extent / 2, lo)) / extent, 0, 1)


def overlap_volumes(scans):
    """SAT / VAT volume (cm3) of each scan within the region both scans cover."""
    volumes = {}
    for key, s in scans.items():
        other = scans['T12' if key == 'L3' else 'L3']
        frac = fraction_inside(s, other) * fraction_inside(s, s)          # inside both slabs
        voxel_cm3 = s['spacing'][0] * s['spacing'][1] * s['thickness'] / 1000
        volumes[key] = [float(((s['seg'] == label) * frac).sum() * voxel_cm3)
                        for label in (1, 2)]   # 1 = SAT, 2 = VAT
    lo, hi = (max(a, b) if i == 0 else min(a, b)
              for i, (a, b) in enumerate(zip(slab(scans['L3'], scans['L3']['normal']),
                                             slab(scans['T12'], scans['L3']['normal']))))
    tilt = np.degrees(np.arccos(min(1.0, abs(np.dot(scans['L3']['normal'], scans['T12']['normal'])))))
    return volumes, lo, hi, tilt


def pct(a, b):
    return 100 * (b - a) / ((a + b) / 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('root', help='folder with one subfolder per case')
    parser.add_argument('--csv', help='also write the table to this CSV file')
    args = parser.parse_args()

    rows = []
    print(f'{"case":16s} | {"L3 SAT":>8s} {"T12 SAT":>8s} {"dSAT":>6s} | '
          f'{"L3 VAT":>8s} {"T12 VAT":>8s} {"dVAT":>6s} | {"dTotal":>6s} | overlap (mm) | tilt')
    for case in sorted(os.listdir(args.root)):
        case_dir = os.path.join(args.root, case)
        if not os.path.isdir(case_dir):
            continue
        scans = load_case(case_dir)
        if set(scans) != {'L3', 'T12'}:
            print(f'{case:16s} | skipped: needs both AbdoCompL3 and T12S1 with seg.nii.gz')
            continue
        volumes, lo, hi, tilt = overlap_volumes(scans)
        L, T = volumes['L3'], volumes['T12']
        row = dict(case=case, slab_from=round(lo, 2), slab_to=round(hi, 2), tilt_deg=round(tilt, 2),
                   L3_SAT=round(L[0], 1), T12_SAT=round(T[0], 1), SAT_diff_pct=round(pct(L[0], T[0]), 2),
                   L3_VAT=round(L[1], 1), T12_VAT=round(T[1], 1), VAT_diff_pct=round(pct(L[1], T[1]), 2),
                   total_diff_pct=round(pct(sum(L), sum(T)), 2))
        rows.append(row)
        print(f'{case:16s} | {L[0]:8.1f} {T[0]:8.1f} {pct(L[0], T[0]):+5.1f}% | '
              f'{L[1]:8.1f} {T[1]:8.1f} {pct(L[1], T[1]):+5.1f}% | {pct(sum(L), sum(T)):+5.1f}% | '
              f'{lo:.1f} .. {hi:.1f} | {tilt:.1f} deg')
    print('Volumes in cm3. Differences are T12 - L3, as a percentage of the mean of the two.')

    if args.csv and rows:
        with open(args.csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f'Saved {args.csv}')


if __name__ == '__main__':
    main()
