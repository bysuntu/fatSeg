"""
Abdominal SAT / VAT segmentation for axial Dixon fat (_F) slices.

Kept separate from the thigh pipeline (seg.py) so changes here cannot affect it.

Idea (adapted from the thigh method):
  0. The fat threshold is set relative to the local fat signal: the median SAT
     intensity of each slice (from a first pass at the GUI threshold), smoothed over
     neighbouring slices because coil sensitivity varies along the body. A voxel counts
     as fat when at least FAT_FRACTION of its signal is fat, so volumes agree across
     scans with different intensity scales and voxel sizes (partial volume).
  1. Fat = pixels above threshold; the largest fat component is the subcutaneous ring.
  2. Body = that ring with its holes filled. Arms are removed: the torso has a deep core
     (> ARM_CORE_MM inside its outline) that a crescent-shaped arm lacks; fat pieces lying
     mostly outside the zone around that core are dropped before gap closing can bridge
     them to the torso. Arms that really touch the torso and have their own core are split
     off at the narrow contact (watershed on the distance to the body outline).
  3. Cast rays from the body centre. On each ray, SAT is the run of fat that starts
     at the skin and ends at the first non-fat gap (the abdominal muscle wall); gaps up to
     GAP_TOL_MM are tolerated.
  4. Rays that disagree with their neighbours (bridges into VAT make them too thick;
     vessels or the navel crease inside SAT make them too thin) are replaced by the
     median of neighbouring rays.
     Rays that are too thick are judged more strictly: at the anterior midline
     (linea alba) there is little muscle to stop the SAT run.
     A wider fault shows as a sharp jump away from the neighbouring thickness and back
     again; such segments are bridged by interpolation. Dips (the run stopped early, e.g.
     at a fascia line on a flank) are bridged all around the body; bumps (e.g. a bridge
     into VAT along the linea alba) only in the anterior sector, because thick fat pads on
     the flanks and pelvis are real. A single sharp step without a return is real anatomy
     (e.g. where the muscle wall ends at the posterior flank) and is kept.
     Finally each ray is compared with the same ray on the neighbouring slices
     (XSLICE_WINDOW on each side): a ray much thinner than there (a wide wedge of VAT
     cutting into SAT on one or two slices) takes their median thickness, since real
     anatomy changes gradually along the body.
     BOUNDARY selects how the SAT inner edge is found (3-4 are the 'rays' method):
       'rays'     as above (default).
       'path'     the best closed path around the body (dynamic programming on the image
                  unwrapped around the body centre): each candidate edge scores fat on its
                  outer side minus a dark band on its inner side (PATH_OUT_MM / PATH_IN_MM;
                  a thick muscle wall scores high, a thin fascia line low), minus the darkness
                  crossed from the skin (PATH_MU per mm), minus thickness changes between
                  neighbouring rays (PATH_GAMMA per mm). One continuous boundary, so a fascia
                  line cannot stop it locally; it sits ~1 px shallower than the rays, hence
                  PATH_OFFSET_PX.
       'combined' the ray result, except where a ray is more than COMBINE_TOL_MM thinner than
                  the path (the ray stopped early): there the path is used. Keeps the rays'
                  pixel-accurate edge and the path's protection against early stops.
  5. Every fat pixel in the body is either SAT or VAT:
     SAT = fat outside the SAT inner edge, VAT = fat inside it.
  6. Bone marrow is not VAT: the vertebral body (a round, mid-intensity region on the
     posterior midline) is found and fat specks inside it are left unlabelled.

Labels: 0 background (non-fat / bone marrow), 1 SAT, 2 VAT.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.draw import polygon as draw_polygon
from skimage.measure import regionprops
from skimage.morphology import convex_hull_image, disk
from skimage.segmentation import watershed

N_RAYS = 360
MEDIAN_WINDOW = 21     # rays used to judge outliers
THICK_TOL = 0.10       # a ray is too thick if it exceeds the median by > 10% (+ 3 px)
THIN_TOL = 0.25        # a ray is too thin if it falls below the median by > 25% (+ 3 px)
JUMP_TOL = 0.15        # thickness change between adjacent rays counted as a sharp jump
MAX_PLATEAU = 20       # rays; widest jump-and-return segment that is bridged
GAP_TOL_MM = 2.0       # mm of non-fat tolerated inside the SAT run (noise); in mm so that
                       # thin muscle wall is not skipped on coarse scans
ANTERIOR_SECTOR = 45   # degrees either side of the anterior midline where thick bumps are
                       # bridged too (thin dips are bridged all around)
XSLICE_WINDOW = 3      # slices on each side used to check each ray across slices
XSLICE_THIN = 0.75     # a ray is too thin if below this share of the neighbouring slices'
XSLICE_ABS_MM = 2.0    # median thickness, minus this many mm
BOUNDARY = 'combined'      # 'rays', 'path' or 'combined' (see step 4 above)
PATH_OUT_MM = 2.0      # path: fat window outside the candidate edge
PATH_IN_MM = 4.0       # path: dark window inside it (thicker than a fascia line)
PATH_MU = 0.05         # path: cost per mm of darkness crossed between skin and edge
PATH_GAMMA = 0.05      # path: cost per mm of thickness change between neighbouring rays
PATH_MAX_JUMP_MM = 12  # path: largest thickness change between neighbouring rays
PATH_MAX_FRAC = 0.75   # path: SAT thickness at most this share of the skin radius
PATH_OFFSET_PX = 2.0   # path: move the edge this many px deeper (it sits mid-transition)
COMBINE_TOL_MM = 8.0   # combined: use the path where the ray is this much thinner than it
MIN_SIZE = 10          # px; smaller fat specks are dropped
OUTSIDE_KEEP = 0.5     # fat pieces with less than this share inside the torso zone are dropped
TORSO_MARGIN_MM = 10   # torso zone = within ARM_CORE_MM + this of the torso core
ARM_CORE_MM = 15       # body parts thicker than 2x this get their own core; arms are split
                       # from the torso where the contact is narrower than that
FIRST_PASS_THRESHOLD = 100   # rough fat threshold (raw value) for the first pass that measures the fat signal
FAT_FRACTION = 0.4     # fat threshold as a fraction of the local fat signal, for SAT and the boundary (tuned on
                       # 5 cases scanned with two sequences: flat optimum 0.3-0.45)
REF_SMOOTH = 5         # slices; median window for the per-slice fat signal
REF_PERCENTILE = 50    # percentile of SAT intensity used as the fat signal
VERT_BAND = (0.12, 0.26)   # vertebral body intensity band, as a fraction of the fat signal
VERT_AREA = (300, 5000)    # px; plausible vertebral body area
VERT_SOLIDITY = 0.7    # vertebral body is compact and round (measured with holes filled)
VERT_SEARCH = 0.35     # search the posterior midline up to this fraction of body height


def _remove_small(mask, min_size=MIN_SIZE):
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_size
    keep[0] = False
    return keep[lab]


def _split_off_arms(region, pixel_spacing):
    """Keep the torso: split touching arms off at the narrow contact and drop them."""
    dist = ndi.distance_transform_edt(region, sampling=pixel_spacing)
    cores, n = ndi.label(dist > ARM_CORE_MM)
    if n <= 1:
        return region
    parts = watershed(-dist, cores, mask=region)
    return parts == (np.argmax(np.bincount(parts.ravel())[1:]) + 1)


def _largest_filled(fat):
    """Close small gaps in the SAT ring; return the largest piece and its filled region."""
    closed = ndi.binary_closing(fat, structure=disk(5))
    lab, n = ndi.label(closed)
    if n == 0:
        return None, None
    largest = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    return largest, ndi.binary_fill_holes(largest)


def _drop_outside_pieces(fat, pixel_spacing):
    """Drop fat pieces (e.g. arms) lying mostly outside the zone around the torso core."""
    lab, n = ndi.label(fat)
    if n <= 1:
        return fat
    largest, filled = _largest_filled(fat)
    hull = convex_hull_image(largest)
    if filled.sum() < 0.8 * hull.sum():
        filled = hull    # ring open: the hull gives the torso shape
    dist = ndi.distance_transform_edt(filled, sampling=pixel_spacing)
    cores, nc = ndi.label(dist > ARM_CORE_MM)
    if nc == 0:
        return fat       # no solid torso (e.g. open ring): leave as is
    core = cores == (np.argmax(np.bincount(cores.ravel())[1:]) + 1)
    zone = ndi.distance_transform_edt(~core, sampling=pixel_spacing) <= ARM_CORE_MM + TORSO_MARGIN_MM
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    inside = np.bincount(lab[zone].ravel(), minlength=n + 1)
    keep = inside >= OUTSIDE_KEEP * sizes
    keep[np.argmax(sizes[1:]) + 1] = True    # never drop the main piece
    keep[0] = False
    return keep[lab]


def _body_mask(fat, pixel_spacing=1.0):
    fat = _drop_outside_pieces(fat, pixel_spacing)
    largest, filled = _largest_filled(fat)
    if largest is None:
        return None
    # Ring still open (e.g. cut by the FOV edge): fall back to the convex hull
    hull = convex_hull_image(largest)
    if filled.sum() < 0.8 * hull.sum():
        return hull
    # Solid body: arms resting against the torso are cut off at the narrow contact,
    # then the hull check is repeated on the torso alone
    body = _split_off_arms(filled, pixel_spacing)
    hull = convex_hull_image(largest & body)
    if body.sum() < 0.8 * hull.sum():
        body = hull
    return body


def _smooth_outliers(values):
    """Replace rays that differ too much from their neighbours by the local median."""
    med = ndi.median_filter(values, size=MEDIAN_WINDOW, mode='wrap')
    too_thick = values > med * (1 + THICK_TOL) + 3
    too_thin = values < med * (1 - THIN_TOL) - 3
    return np.where(too_thick | too_thin, med, values)


def _bridge_plateaus(values):
    """Bridge segments that jump sharply away from the neighbouring thickness and back."""
    t = values.copy()
    n = len(t)
    # Anterior midline is angle 3*pi/2 (smaller row = anterior); rays are 360/n degrees apart
    k_ant = 3 * n // 4
    half = int(ANTERIOR_SECTOR * n / 360)
    k = 0
    while k < n:
        base = t[k % n]
        tol = JUMP_TOL * base + 3
        # Dips (SAT run stopped early, e.g. at a fascia line) are bridged all around; bumps
        # only at the front, since thick fat pads on the flanks and pelvis are real
        front = abs((k - k_ant + n // 2) % n - n // 2) <= half
        jump = t[(k + 1) % n] - base
        if jump < -tol or (front and jump > tol):
            # The segment must stay away from the base level and return to it with a
            # sharp jump within MAX_PLATEAU rays; a gradual return is real anatomy
            # (e.g. thick flank SAT tapering off) and is left alone
            for j in range(k + 2, k + 2 + MAX_PLATEAU):
                end = t[j % n]
                if abs(end - base) <= tol:
                    if abs(end - t[(j - 1) % n]) > tol:
                        idx = np.arange(k + 1, j) % n
                        t[idx] = np.linspace(base, end, j - k + 1)[1:-1]
                        k = j - 1
                    break
        k += 1
    return t


def _polygon_mask(center, radii, angles, shape):
    mask = np.zeros(shape, dtype=bool)
    pr, pc = draw_polygon(center[0] + radii * np.sin(angles),
                          center[1] + radii * np.cos(angles), shape=shape)
    mask[pr, pc] = True
    return mask


def _sat_rays(fat, body, center, pixel_spacing):
    """Skin radius and corrected SAT thickness (pixels) on each ray, and the ray angles."""
    h, w = fat.shape
    step = 0.5
    radii = np.arange(0, np.hypot(h, w), step)
    angles = np.linspace(0, 2 * np.pi, N_RAYS, endpoint=False)
    r_skin = np.zeros(N_RAYS)
    r_sat = np.zeros(N_RAYS)     # radius of the SAT inner edge
    gap_steps = int(round(GAP_TOL_MM / pixel_spacing / step))

    for k, a in enumerate(angles):
        rows = np.round(center[0] + radii * np.sin(a)).astype(int)
        cols = np.round(center[1] + radii * np.cos(a)).astype(int)
        valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        rows, cols, rr = rows[valid], cols[valid], radii[valid]
        inside = np.nonzero(body[rows, cols])[0]
        if inside.size == 0:
            continue
        i_skin = inside[-1]
        r_skin[k] = rr[i_skin]
        on_fat = fat[rows[:i_skin + 1], cols[:i_skin + 1]]

        # SAT: fat run from the skin inward, tolerating small gaps
        i_end, gap, i = i_skin, 0, i_skin
        while i >= 0:
            if on_fat[i]:
                i_end, gap = i, 0
            else:
                gap += 1
                if gap > gap_steps:
                    break
            i -= 1
        r_sat[k] = rr[i_end] if on_fat[i_end] else r_skin[k]

    return r_skin, _smooth_outliers(_bridge_plateaus(r_skin - r_sat)), angles


def _path_thickness(image, fat_ref, body, center, pixel_spacing):
    """SAT thickness (px) on each ray from the best closed boundary path, or None."""
    h, w = image.shape
    step = 0.5
    mm = step * pixel_spacing
    radii = np.arange(0, np.hypot(h, w), step)
    angles = np.linspace(0, 2 * np.pi, N_RAYS, endpoint=False)
    rel = image / fat_ref
    profiles, r_skin = [], np.zeros(N_RAYS)
    for k, a in enumerate(angles):
        rows = np.round(center[0] + radii * np.sin(a)).astype(int)
        cols = np.round(center[1] + radii * np.cos(a)).astype(int)
        valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        rows, cols, rr = rows[valid], cols[valid], radii[valid]
        inside = np.nonzero(body[rows, cols])[0]
        if inside.size == 0:
            profiles.append(np.zeros(1))
            continue
        r_skin[k] = rr[inside[-1]]
        profiles.append(rel[rows[:inside[-1] + 1], cols[:inside[-1] + 1]][::-1])   # skin inward

    # Score of every candidate thickness on every ray: fat outside, dark band inside,
    # minus the darkness crossed from the skin
    w_out = max(1, int(round(PATH_OUT_MM / mm)))
    w_in = max(1, int(round(PATH_IN_MM / mm)))
    n_t = max(len(p) for p in profiles)
    score = np.full((N_RAYS, n_t), -np.inf)
    for k, p in enumerate(profiles):
        if len(p) < w_out + w_in + 2:
            continue
        cs = np.concatenate([[0], np.cumsum(p)])
        crossed = np.concatenate([[0], np.cumsum(np.clip(1 - p / FAT_FRACTION, 0, 1))]) * mm
        j = np.arange(w_out, max(w_out, min(len(p) - w_in, int(PATH_MAX_FRAC * r_skin[k] / step))))
        if j.size:
            score[k, j] = ((cs[j] - cs[j - w_out]) / w_out - (cs[j + w_in] - cs[j]) / w_in
                           - PATH_MU * crossed[j])
    if not np.isfinite(score).any():
        return None

    # Best closed path: two laps round the body, the middle lap is kept (no start/end effect)
    max_jump = max(1, int(round(PATH_MAX_JUMP_MM / mm)))
    order = np.concatenate([np.arange(N_RAYS), np.arange(N_RAYS)])
    total = score[order[0]].copy()
    back = np.zeros((len(order), n_t), np.int32)
    idx = np.arange(n_t)
    for t in range(1, len(order)):
        best = np.full(n_t, -np.inf)
        arg = np.zeros(n_t, np.int32)
        for d in range(-max_jump, max_jump + 1):
            shifted = np.full(n_t, -np.inf)
            if d >= 0:
                shifted[d:] = total[:n_t - d]
            else:
                shifted[:d] = total[-d:]
            cand = shifted - PATH_GAMMA * abs(d) * mm
            better = cand > best
            best[better] = cand[better]
            arg[better] = idx[better] - d
        total = best + score[order[t]]
        back[t] = arg
    path = np.zeros(len(order), np.int32)
    path[-1] = int(np.argmax(total))
    for t in range(len(order) - 1, 0, -1):
        path[t - 1] = back[t][path[t]]
    keep = np.arange(N_RAYS // 2, N_RAYS // 2 + N_RAYS)
    thick = np.zeros(N_RAYS)
    thick[order[keep]] = path[keep] * step
    return thick + PATH_OFFSET_PX


def _slice_rays(image, fat_ref, pixel_spacing):
    """Fat, body and SAT rays of one slice, or None if the slice has no body."""
    fat = _remove_small(image > FAT_FRACTION * fat_ref)
    if not fat.any():
        return None
    body = _body_mask(fat, pixel_spacing)
    if body is None or not body.any():
        return None
    center = ndi.center_of_mass(body)
    r_skin, thick, angles = _sat_rays(fat, body, center, pixel_spacing)
    if BOUNDARY in ('path', 'combined'):
        path = _path_thickness(image, fat_ref, body, center, pixel_spacing)
        if path is not None:
            if BOUNDARY == 'path':
                thick = path
            else:   # keep the ray unless it stopped well short of the path
                thick = np.where(thick < path - COMBINE_TOL_MM / pixel_spacing, path, thick)
    return dict(fat=fat, body=body, center=center, r_skin=r_skin, thick=thick, angles=angles)


def _labels_from_rays(image, fat_ref, rays, vat_fraction=FAT_FRACTION):
    """SAT = fat outside the SAT inner edge, VAT = fat inside it (vertebral marrow excluded).

    SAT uses FAT_FRACTION; VAT uses `vat_fraction` (the GUI's 'VAT Fat Fraction').
    """
    r_sat = np.maximum(rays['r_skin'] - rays['thick'], 0)
    sat_inner = _polygon_mask(rays['center'], r_sat, rays['angles'], rays['fat'].shape)
    sat = rays['fat'] & rays['body'] & ~sat_inner
    vat_fat = rays['fat'] if vat_fraction == FAT_FRACTION else _remove_small(image > vat_fraction * fat_ref)
    vat = vat_fat & rays['body'] & sat_inner
    vert = _vertebra_mask(image, rays['body'], rays['center'], fat_ref)
    if vert is not None:
        vat &= ~vert
    return sat, vat


def _check_across_slices(rays, pixel_spacing):
    """Replace rays whose SAT is much thinner than on the neighbouring slices.

    A run stopped early (e.g. a wide wedge of VAT cutting into SAT) usually appears on one
    or two slices only; real anatomy changes gradually along the body.
    """
    n = len(rays)
    thick = np.array([r['thick'] if r else np.full(N_RAYS, np.nan) for r in rays])
    abs_px = XSLICE_ABS_MM / pixel_spacing
    for i, r in enumerate(rays):
        if r is None:
            continue
        nb = [j for j in range(i - XSLICE_WINDOW, i + XSLICE_WINDOW + 1)
              if j != i and 0 <= j < n and rays[j] is not None]
        if len(nb) < 2:
            continue
        ref = np.median(thick[nb], axis=0)
        too_thin = thick[i] < XSLICE_THIN * ref - abs_px
        r['thick'] = np.where(too_thin, ref, thick[i])


def _vertebra_mask(image, body, center, fat_ref):
    """Vertebral body: round mid-intensity region just posterior of the body centre."""
    # Assumes standard axial orientation: anterior at the top (posterior = larger row)
    smooth = ndi.gaussian_filter(image.astype(float), 2)
    band = (smooth > VERT_BAND[0] * fat_ref) & (smooth < VERT_BAND[1] * fat_ref) & body
    band = ndi.binary_opening(band, structure=disk(3))   # cut thin links to neighbours
    lab, n = ndi.label(band)
    if n == 0:
        return None
    rows_b = np.nonzero(body)[0]
    row_end = center[0] + VERT_SEARCH * (rows_b.max() - rows_b.min())
    col = int(round(center[1]))
    tried = set()
    for row in range(int(center[0]), int(row_end)):
        lbl = lab[row, col]
        if lbl == 0 or lbl in tried:
            continue
        tried.add(lbl)
        vert = ndi.binary_fill_holes(lab == lbl)   # marrow fat specks leave holes
        r = regionprops(vert.astype(np.uint8))[0]
        if VERT_AREA[0] <= r.area <= VERT_AREA[1] and r.solidity >= VERT_SOLIDITY:
            # The band often catches only part of the body; the hull covers notches with marrow
            return ndi.binary_dilation(convex_hull_image(vert), structure=disk(2))
    return None


def segment_abdomen(image, fat_ref, pixel_spacing=1.0):
    """Segment one axial slice given the fat signal and pixel size (mm). Returns (sat, vat) masks."""
    rays = _slice_rays(image, fat_ref, pixel_spacing)
    if rays is None:
        empty = np.zeros(image.shape, dtype=bool)
        return empty, empty
    return _labels_from_rays(image, fat_ref, rays)


def fat_reference(image_stack, threshold=FIRST_PASS_THRESHOLD, pixel_spacing=1.0):
    """Per-slice fat signal: median SAT intensity from a first pass with `threshold` as the fat threshold."""
    ref = np.full(len(image_stack), np.nan)
    for i, sl in enumerate(image_stack):
        sat, _ = segment_abdomen(sl, threshold / FAT_FRACTION, pixel_spacing)
        if sat.sum() >= 100:
            ref[i] = np.percentile(sl[sat], REF_PERCENTILE)
    if np.all(np.isnan(ref)):
        return np.full(len(image_stack), threshold / FAT_FRACTION)
    ref[np.isnan(ref)] = np.nanmedian(ref)
    # Smooth along the body; the median ignores single slices with a poor SAT estimate
    return ndi.median_filter(ref, size=REF_SMOOTH, mode='nearest')


def segment_abdomen_stack(image_stack, threshold=FIRST_PASS_THRESHOLD, pixel_spacing=1.0, vat_fraction=FAT_FRACTION,
                          return_state=False):
    """Segment a (S, H, W) stack. Returns labels shaped (H, W, S): 1 SAT, 2 VAT.

    `threshold` (raw value) only needs to roughly separate fat for the first pass that
    measures the fat signal. SAT and the SAT/VAT boundary use FAT_FRACTION of that signal;
    VAT uses `vat_fraction` (the GUI's 'VAT Fat Fraction', default FAT_FRACTION).
    `pixel_spacing` is the in-plane pixel size in mm.
    With return_state=True also returns the fat signal and boundary, for relabel_vat().
    """
    fat_ref = fat_reference(image_stack, threshold, pixel_spacing)
    print(f'Abdomen: fat signal {fat_ref.min():.0f}-{fat_ref.max():.0f}; '
          f'SAT fat fraction {FAT_FRACTION:g} (threshold {FAT_FRACTION * fat_ref.min():.0f}-{FAT_FRACTION * fat_ref.max():.0f}), '
          f'VAT fat fraction {vat_fraction:g} (threshold {vat_fraction * fat_ref.min():.0f}-{vat_fraction * fat_ref.max():.0f})')
    rays = [_slice_rays(sl, fat_ref[i], pixel_spacing) for i, sl in enumerate(image_stack)]
    _check_across_slices(rays, pixel_spacing)
    state = dict(fat_ref=fat_ref, rays=rays)
    labels = relabel_vat(image_stack, state, vat_fraction)
    return (labels, state) if return_state else labels


def relabel_vat(image_stack, state, vat_fraction=FAT_FRACTION):
    """Labels (H, W, S) from a stored fat signal and SAT/VAT boundary, with VAT at `vat_fraction`.

    SAT and the boundary do not depend on the VAT fat fraction, so changing it only needs this
    step, not a new segmentation.
    """
    labels = np.zeros(image_stack.shape, dtype=np.uint8)
    for i, sl in enumerate(image_stack):
        if state['rays'][i] is None:
            continue
        sat, vat = _labels_from_rays(sl, state['fat_ref'][i], state['rays'][i], vat_fraction)
        labels[i][sat] = 1
        labels[i][vat] = 2
    return labels.transpose(1, 2, 0)
