"""
Compare SAT / VAT volumes between the AbdoCompL3 and T12-S1 DIXON VIBE sequences
of each case, over the z range the two scans have in common.

Usage:
    python compare_overlap.py E:\\after
    python compare_overlap.py E:\\after --csv overlap_volumes.csv

Expected layout: <root>/<case>/<series folder>/ containing the DICOM files and the
seg.nii.gz saved by the GUI (labels: 1 = SAT, 2 = VAT). Series folders are found by
name: AbdoCompL3* and T12S1* (underscores ignored).

Overlap is taken in scanner coordinates (no motion correction): each slice covers its
z position +/- half its thickness, and slices only partly inside the overlap count by
the fraction inside. Differences are T12 - L3, as a percentage of the mean of the two.
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
        scans[key] = dict(seg=nib.load(seg_path).get_fdata(),
                          z=np.array([float(i[3][2]) for i in info]),   # ImagePositionPatient z
                          spacing=[float(v) for v in info[0][4]],       # PixelSpacing
                          thickness=float(info[0][5]))                  # SliceThickness
    return scans


def overlap_volumes(scans):
    """SAT / VAT volume (cm3) of each scan within the common z range."""
    lo = max(s['z'].min() - s['thickness'] / 2 for s in scans.values())
    hi = min(s['z'].max() + s['thickness'] / 2 for s in scans.values())
    volumes = {}
    for key, s in scans.items():
        z, th = s['z'], s['thickness']
        frac = np.clip((np.minimum(z + th / 2, hi) - np.maximum(z - th / 2, lo)) / th, 0, 1)
        voxel_cm3 = s['spacing'][0] * s['spacing'][1] * th / 1000
        volumes[key] = [float(sum((s['seg'][:, :, i] == label).sum() * voxel_cm3 * frac[i]
                                  for i in range(len(frac))))
                        for label in (1, 2)]   # 1 = SAT, 2 = VAT
    return volumes, lo, hi


def pct(a, b):
    return 100 * (b - a) / ((a + b) / 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('root', help='folder with one subfolder per case')
    parser.add_argument('--csv', help='also write the table to this CSV file')
    args = parser.parse_args()

    rows = []
    print(f'{"case":16s} | {"L3 SAT":>8s} {"T12 SAT":>8s} {"dSAT":>6s} | '
          f'{"L3 VAT":>8s} {"T12 VAT":>8s} {"dVAT":>6s} | {"dTotal":>6s} | overlap z (mm)')
    for case in sorted(os.listdir(args.root)):
        case_dir = os.path.join(args.root, case)
        if not os.path.isdir(case_dir):
            continue
        scans = load_case(case_dir)
        if set(scans) != {'L3', 'T12'}:
            print(f'{case:16s} | skipped: needs both AbdoCompL3 and T12S1 with seg.nii.gz')
            continue
        volumes, lo, hi = overlap_volumes(scans)
        L, T = volumes['L3'], volumes['T12']
        row = dict(case=case, z_from=round(lo, 2), z_to=round(hi, 2),
                   L3_SAT=round(L[0], 1), T12_SAT=round(T[0], 1), SAT_diff_pct=round(pct(L[0], T[0]), 2),
                   L3_VAT=round(L[1], 1), T12_VAT=round(T[1], 1), VAT_diff_pct=round(pct(L[1], T[1]), 2),
                   total_diff_pct=round(pct(sum(L), sum(T)), 2))
        rows.append(row)
        print(f'{case:16s} | {L[0]:8.1f} {T[0]:8.1f} {pct(L[0], T[0]):+5.1f}% | '
              f'{L[1]:8.1f} {T[1]:8.1f} {pct(L[1], T[1]):+5.1f}% | {pct(sum(L), sum(T)):+5.1f}% | '
              f'{lo:.1f} .. {hi:.1f}')
    print('Volumes in cm3. Differences are T12 - L3, as a percentage of the mean of the two.')

    if args.csv and rows:
        with open(args.csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f'Saved {args.csv}')


if __name__ == '__main__':
    main()
