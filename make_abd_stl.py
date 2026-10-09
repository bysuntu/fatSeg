"""
Segment each abdominal scan with the GUI's method (mriFat/abd_seg.py, BOUNDARY as set there)
and write SAT and VAT surfaces as binary STL in patient coordinates (mm), for viewing the
overlap of the AbdoCompL3 and T12S1...DIXON VIBE scans in ParaView.

Usage:
    python make_abd_stl.py E:\\fatSeg\\rawAbd
    python make_abd_stl.py E:\\fatSeg\\rawAbd --out abd_stl --smooth 0

Expected layout: <root>/<case>/<series folder>/ with the DICOMs (series folders found by name,
underscores ignored: AbdoCompL3*, T12S1*). Output per case in <out>/<case>/:
    L3_SAT.stl, L3_VAT.stl, T12_SAT.stl, T12_VAT.stl
    L3_SAT_overlap.stl, ..., T12_VAT_overlap.stl   (each scan cut to the slab the other covers)
plus segmentation_volumes.csv with the voxel and mesh volumes.

The overlap is the real slab of the other scan, along its own slice direction: the scans of a
case can be tilted against each other (03260016NHCTHO: 3.9 deg), so cutting at flat z planes
would compare different regions near the flanks.

Vertices are mapped from (slice, row, column) to patient coordinates with each scan's DICOM
position and orientation, without registration: both scans of a case open in the same frame.
"""
import argparse
import contextlib
import csv
import io
import os
import struct
import sys

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import marching_cubes

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mriFat'))
with contextlib.redirect_stdout(io.StringIO()):
    from readSpace_threeButton import parseDicomFolder
import abd_seg

SERIES = (('L3', 'abdocompl3'), ('T12', 't12s1'))
LABELS = ((1, 'SAT'), (2, 'VAT'))


def load_and_segment(folder):
    """Labels (slices, rows, cols) as the GUI makes them, and the slice geometry."""
    with contextlib.redirect_stdout(io.StringIO()):
        pixels, info = parseDicomFolder(folder, seriesSuffix='_F')
        stack = np.array(pixels, np.float32)
        stack = np.clip(stack, np.percentile(stack, 1), np.percentile(stack, 99))   # as the GUI does
        spacing = [float(v) for v in info[0][4]]
        labels = abd_seg.segment_abdomen_stack(stack, 100, spacing[0]).transpose(2, 0, 1)
    geom = dict(origin=np.array(info[0][3], float), last=np.array(info[-1][3], float),
                row_dir=np.array(info[0][2][:3], float), col_dir=np.array(info[0][2][3:], float),
                spacing=spacing, thickness=float(info[0][5]), n=len(info),
                z=np.array([float(i[3][2]) for i in info]), shape=labels.shape[1:])
    return labels, geom


def slab_mask(geom, other):
    """Voxels of `geom`'s grid inside the slab covered by `other` (its slices +- half thickness)."""
    normal = np.cross(other['row_dir'], other['col_dir'])
    start = np.dot(other['origin'], normal) - other['thickness'] / 2
    end = np.dot(other['last'], normal) + other['thickness'] / 2
    lo, hi = min(start, end), max(start, end)
    step = (geom['last'] - geom['origin']) / max(geom['n'] - 1, 1)
    rows, cols = geom['shape']
    rr, cc = np.mgrid[0:rows, 0:cols]
    plane = (np.dot(geom['col_dir'], normal) * geom['spacing'][0] * rr
             + np.dot(geom['row_dir'], normal) * geom['spacing'][1] * cc)
    depth = np.dot(geom['origin'], normal) + np.arange(geom['n'])[:, None, None] * np.dot(step, normal) + plane[None]
    return (depth >= lo) & (depth <= hi)


def to_patient(verts, geom):
    """(slice, row, col) index coordinates -> patient mm."""
    step = (geom['last'] - geom['origin']) / max(geom['n'] - 1, 1)
    s, r, c = verts[:, 0:1], verts[:, 1:2], verts[:, 2:3]
    return (geom['origin'] + s * step + c * geom['spacing'][1] * geom['row_dir']
            + r * geom['spacing'][0] * geom['col_dir'])


def surface(mask, geom, smooth):
    """Closed surface of a binary mask, in patient mm. Returns (vertices, faces) or None."""
    if mask.sum() < 10:
        return None
    vol = np.pad(mask.astype(np.float32), 1)                    # closes the surface at the edges
    if smooth > 0:
        # in voxels: less along the slices, which are already 3-4 mm apart
        vol = ndi.gaussian_filter(vol, sigma=(smooth * 0.5, smooth, smooth))
    verts, faces, _, _ = marching_cubes(vol, level=0.5)
    return to_patient(verts - 1, geom), faces


def mesh_volume(verts, faces):
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    return abs(np.einsum('ij,ij->i', v0, np.cross(v1, v2)).sum()) / 6


def write_stl(path, verts, faces, header):
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)
    data = np.zeros(len(faces), dtype=[('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')])
    data['n'] = normals
    data['v'] = np.stack([v0, v1, v2], axis=1)
    with open(path, 'wb') as f:
        f.write(header.encode('ascii', 'replace')[:80].ljust(80, b' '))
        f.write(struct.pack('<I', len(faces)))
        f.write(data.tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('root', help='folder with one subfolder per case')
    parser.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'abd_stl'),
                        help='output folder (default: abd_stl next to this script)')
    parser.add_argument('--smooth', type=float, default=0.7,
                        help='in-plane Gaussian smoothing of the masks before meshing, in voxels (0 = blocky)')
    args = parser.parse_args()

    rows = []
    print(f"Segmentation: abd_seg.py, BOUNDARY = '{abd_seg.BOUNDARY}'. Output: {args.out}")
    for case in sorted(os.listdir(args.root)):
        case_dir = os.path.join(args.root, case)
        if not os.path.isdir(case_dir):
            continue
        scans = {}
        for sub in sorted(os.listdir(case_dir)):
            key = next((k for k, p in SERIES if sub.lower().replace('_', '').startswith(p)), None)
            if key and os.path.isdir(os.path.join(case_dir, sub)):
                scans[key] = load_and_segment(os.path.join(case_dir, sub))
        if not scans:
            continue
        out_dir = os.path.join(args.out, case)
        os.makedirs(out_dir, exist_ok=True)

        both = set(scans) == {'L3', 'T12'}

        for key, (labels, geom) in scans.items():
            vox = geom['spacing'][0] * geom['spacing'][1] * geom['thickness'] / 1000
            parts = [('', np.ones(labels.shape, bool))]
            if both:
                other = scans['T12' if key == 'L3' else 'L3'][1]
                parts.append(('_overlap', slab_mask(geom, other)))
            for suffix, keep in parts:
                for value, name in LABELS:
                    mask = (labels == value) & keep
                    result = surface(mask, geom, args.smooth)
                    if result is None:
                        continue
                    verts, faces = result
                    fname = f'{key}_{name}{suffix}.stl'
                    write_stl(os.path.join(out_dir, fname), verts, faces,
                              f'{case} {key} {name}{suffix} (patient coordinates, mm)')
                    v_vox, v_mesh = mask.sum() * vox, mesh_volume(verts, faces) / 1000
                    rows.append(dict(case=case, file=fname, voxel_cm3=round(v_vox, 1), mesh_cm3=round(v_mesh, 1),
                                     triangles=len(faces)))
                    print(f'  {case} {fname:22s} voxel {v_vox:8.1f} cm3, mesh {v_mesh:8.1f} cm3, {len(faces):8d} triangles', flush=True)
        with open(os.path.join(out_dir, 'segmentation_volumes.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows([r for r in rows if r['case'] == case])
        if both:
            ov = {r['file']: r['voxel_cm3'] for r in rows if r['case'] == case and r['file'].endswith('_overlap.stl')}
            for name in ('SAT', 'VAT'):
                a, b = ov.get(f'L3_{name}_overlap.stl'), ov.get(f'T12_{name}_overlap.stl')
                if a and b:
                    print(f'  {case}: {name} in the common slab: L3 {a:.1f}, T12 {b:.1f} cm3 ({200 * (b - a) / (a + b):+.1f}%)')


if __name__ == '__main__':
    main()
