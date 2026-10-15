"""
Segment every case in a folder without the GUI, with the GUI's default settings.

Usage:
    python batch_seg.py <root> --mode thigh   [--out batch_out] [--csv volumes.csv]
    python batch_seg.py <root> --mode abdomen [--out batch_out] [--csv volumes.csv]

<root> has one subfolder per case, each with one subfolder per series (as rawThigh / rawAbd).
Series are found as the GUI finds them: thigh 6pt_DIXON_VIBE_F (Dixon fat, seg.py +
dixon_local_imat.py) and t1_tse_tra (t1_seg.py), both when present; abdomen any series ending
in _F (abd_seg.py). If several acquisitions in one folder match, the latest is used and the
series number is in the CSV.

Writes, per series, <out>/<case>/<series folder>_<sequence>.nii.gz (labels laid out like the
images, with the DICOM geometry, as the GUI saves them; thigh 1 SAT, 2 IMAT, 3 muscle, abdomen
1 SAT, 2 VAT) and <out>/<case>/<series folder>_<sequence>_slices.csv (area per slice, cm^2),
and the volume of each label (cm^3, slice areas times slice spacing, as the GUI's plot) to the
--csv file (default <out>/volumes.csv). The input folders are never written to, so a hand-edited
seg.nii.gz next to the DICOM files is left alone.
"""
import argparse
import contextlib
import csv
import io
import os
import sys
import time

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mriFat'))
import parallel
import pipeline
from dicom_io import dicom_affine, slice_spacing_mm


def run_series(case, folder, sequence, kw, mode):
    """Segment one series; returns (row for the volume table, labels, info)."""
    with contextlib.redirect_stdout(io.StringIO()) as quiet:
        pixels, info = pipeline.load(folder, kw)
    for line in quiet.getvalue().splitlines():
        if line.startswith('  several series') or line.startswith('Using series'):
            print(line)
    labels = pipeline.segment(pixels, info, mode, sequence)
    row = dict(case=case, series_folder=os.path.basename(folder), sequence=sequence, series_number=info[0][8],
               series_name=info[0][1], slices=len(info), slice_mm=round(slice_spacing_mm(info), 3),
               pixel_mm=f"{float(info[0][4][0]):g}x{float(info[0][4][1]):g}")
    row.update({f'{name}_cm3': round(v, 2) for name, v in pipeline.volumes(labels, info, mode).items()})
    return row, labels, info


def iter_series(root, mode):
    for case in sorted(os.listdir(root)):
        case_dir = os.path.join(root, case)
        if os.path.isdir(case_dir):
            for folder, sequence, kw in pipeline.find_series(case_dir, mode):
                yield case, folder, sequence, kw


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('root', help='folder with one subfolder per case')
    parser.add_argument('--mode', required=True, choices=('thigh', 'abdomen'))
    parser.add_argument('--out', default='batch_out', help='output folder (default batch_out)')
    parser.add_argument('--csv', help='volume table (default <out>/volumes.csv)')
    args = parser.parse_args()

    rows = []
    for case, folder, sequence, kw in iter_series(args.root, args.mode):
        start = time.perf_counter()
        print(f'{case} / {os.path.basename(folder)} ({sequence})', flush=True)
        try:
            row, labels, info = run_series(case, folder, sequence, kw, args.mode)
        except Exception as e:   # one bad series should not stop the batch
            print(f'  FAILED: {e}', flush=True)
            continue
        name = f"{os.path.basename(folder)}_{sequence}"
        out_dir = os.path.join(args.out, case)
        os.makedirs(out_dir, exist_ok=True)
        nib.save(nib.Nifti1Image(labels, dicom_affine(info)), os.path.join(out_dir, name + '.nii.gz'))
        areas = pipeline.slice_areas(labels, info, args.mode)
        with open(os.path.join(out_dir, name + '_slices.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['slice', 'position_mm'] + [f'{n}_cm2' for n in areas])
            for k in range(len(info)):
                w.writerow([k, round(float(info[k][6]), 2)] + [round(float(a[k]), 3) for a in areas.values()])
        rows.append(row)
        print('  ' + ', '.join(f"{k[:-4]} {v:.1f}" for k, v in row.items() if k.endswith('_cm3'))
              + f' cm3  ({time.perf_counter() - start:.0f} s)', flush=True)
    parallel.shutdown()

    if not rows:
        print('No series found.')
        return
    csv_path = args.csv or os.path.join(args.out, 'volumes.csv')
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'{len(rows)} series; volumes in {csv_path}, labels in {args.out}')


if __name__ == '__main__':
    main()
