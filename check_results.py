"""
Regression check: segment the reference cases and compare with the stored results.

Usage:
    python check_results.py              compare with the reference (exit code 1 if anything changed)
    python check_results.py --update     store the current results as the new reference
    python check_results.py --only thigh (or abdomen)

Runs the GUI's default segmentation (mriFat/pipeline.py, the same steps as batch_seg.py) on
every series of rawThigh (thigh: Dixon fat and T1 TSE) and rawAbd (abdomen: every _F series), and
compares, per series, the labels (exactly, by checksum) and the volume of each label with
check_results_reference.json. For a series whose labels changed, it prints the volume changes
and how many slices differ, so a code change shows its effect on the results before it is kept.

The reference holds case folder names and volumes derived from patient data, so it stays local
(listed in .gitignore). Create it once with --update from a version whose results are trusted.
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'mriFat'))
import parallel
import pipeline

DATA = {'thigh': os.path.join(HERE, 'rawThigh'), 'abdomen': os.path.join(HERE, 'rawAbd')}
REFERENCE = os.path.join(HERE, 'check_results_reference.json')
TOLERANCE_CM3 = 0.05   # volume differences below this are reported as equal (rounding)


def run(modes):
    """{key: result} for every reference series; key mode/case/series folder/sequence."""
    results = {}
    for mode in modes:
        root = DATA[mode]
        if not os.path.isdir(root):
            print(f'{root} not found, {mode} skipped')
            continue
        for case in sorted(os.listdir(root)):
            case_dir = os.path.join(root, case)
            if not os.path.isdir(case_dir):
                continue
            for folder, sequence, kw in pipeline.find_series(case_dir, mode):
                key = f'{mode}/{case}/{os.path.basename(folder)}/{sequence}'
                start = time.perf_counter()
                with contextlib.redirect_stdout(io.StringIO()):
                    pixels, info = pipeline.load(folder, kw)
                    labels = pipeline.segment(pixels, info, mode, sequence)
                results[key] = dict(series_number=info[0][8], shape=list(labels.shape),
                                    checksum=hashlib.md5(labels.tobytes()).hexdigest(),
                                    slice_checksums=[hashlib.md5(labels[:, :, k].tobytes()).hexdigest()[:12]
                                                     for k in range(labels.shape[2])],
                                    volumes_cm3={n: round(v, 3) for n, v in pipeline.volumes(labels, info, mode).items()})
                print(f'  {key}  ({time.perf_counter() - start:.0f} s)', flush=True)
    return results


def compare(reference, results):
    """Print the differences; return the number of series that changed, were added or are missing."""
    problems = 0
    for key in sorted(set(reference) | set(results)):
        if key not in results:
            if key.split('/')[0] in {k.split('/')[0] for k in results}:
                print(f'MISSING  {key}: in the reference but not found now')
                problems += 1
            continue
        if key not in reference:
            print(f'NEW      {key}: not in the reference (run --update to add it)')
            problems += 1
            continue
        ref, now = reference[key], results[key]
        if ref['checksum'] == now['checksum']:
            continue
        problems += 1
        if ref['shape'] != now['shape'] or ref['series_number'] != now['series_number']:
            print(f"CHANGED  {key}: series #{ref['series_number']} {ref['shape']} -> #{now['series_number']} {now['shape']}")
            continue
        slices = [k for k, (a, b) in enumerate(zip(ref['slice_checksums'], now['slice_checksums'])) if a != b]
        parts = []
        for name, a in ref['volumes_cm3'].items():
            b = now['volumes_cm3'][name]
            if abs(b - a) >= TOLERANCE_CM3:
                parts.append(f'{name} {a:.1f} -> {b:.1f} ({100 * (b - a) / a if a else 0:+.2f}%)')
        where = f'slices {slices}' if len(slices) <= 10 else f'{len(slices)} of {len(ref["slice_checksums"])} slices'
        print(f"CHANGED  {key}: {where} differ; " + ('; '.join(parts) or 'volumes equal'))
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--update', action='store_true', help='store the current results as the reference')
    parser.add_argument('--only', choices=('thigh', 'abdomen'))
    args = parser.parse_args()
    modes = [args.only] if args.only else ['thigh', 'abdomen']

    start = time.perf_counter()
    results = run(modes)
    parallel.shutdown()
    print(f'{len(results)} series segmented in {time.perf_counter() - start:.0f} s')

    reference = {}
    if os.path.isfile(REFERENCE):
        with open(REFERENCE) as f:
            reference = json.load(f)
    if args.update:
        kept = {k: v for k, v in reference.items() if k.split('/')[0] not in modes}
        with open(REFERENCE, 'w') as f:
            json.dump({**kept, **results}, f, indent=1, sort_keys=True)
        print(f'Reference updated: {REFERENCE}')
        return
    if not reference:
        sys.exit(f'No reference yet: run  python check_results.py --update  with a trusted version first.')

    problems = compare({k: v for k, v in reference.items() if k.split('/')[0] in modes}, results)
    if problems:
        print(f'{problems} series differ from the reference.')
        sys.exit(1)
    print('All results identical to the reference.')


if __name__ == '__main__':
    main()
