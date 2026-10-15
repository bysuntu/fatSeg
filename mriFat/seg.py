import pydicom
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from skimage.filters import sato
import os
import cv2
from shapely.geometry import Polygon
import pygeoops

from parallel import map_slices

def load_dicom_image(path):
    dicom = pydicom.dcmread(path)
    image = dicom.pixel_array.astype(np.float32)
    
    minPixel = np.percentile(image, 3)
    maxPixel = np.percentile(image, 97)
    # Normalize to 0–255
    # image = 255 * (image - np.min(image)) / (np.max(image) - np.min(image))
    image = np.clip((image - minPixel) / (maxPixel - minPixel), 0, 1) * 255
    return image.astype(np.uint8)

def segment_bright_regions(image, threshold=200, min_size=10):
    binary = image > threshold
    binary = remove_small_objects(binary, min_size=min_size)
    labeled, _ = ndi.label(binary)
    return labeled

def segment_dark_regions(image, threshold=50, min_size=10):
    binary = image < threshold
    binary = remove_small_objects(binary, min_size=min_size)
    labeled, _ = ndi.label(binary)
    return labeled

def extract_segments_in_roi(labeled_image):
    props = regionprops(labeled_image)
    if not props:
        return np.zeros_like(labeled_image), np.zeros_like(labeled_image), None

    # Sort regions by area
    sorted_regions = sorted(props, key=lambda x: x.area, reverse=True)

    # Get the largest region
    largest = sorted_regions[0]
    minr, minc, maxr, maxc = largest.bbox

    # Create masks inside the ROI
    segment1 = np.zeros_like(labeled_image, dtype=np.uint8)
    segment2 = np.zeros_like(labeled_image, dtype=np.uint8)

    for region in sorted_regions:
        if minr <= region.bbox[0] < region.bbox[2] and minc <= region.bbox[1] < region.bbox[3]:
            if region.label == largest.label:
                segment1[labeled_image == region.label] = 1
            else:
                segment2[labeled_image == region.label] = 1

    return segment1, segment2, (minr, minc, maxr, maxc)


BONE_SCORE_RATIO = 0.35  # marrow = largest candidate scoring at least this fraction of the roundest one


def extract_bone_in_roi(labeled_image, seg1):
    """Pick the femur marrow among the bright spots inside the SAT ring; the rest is IMAT.

    Candidates: spots of more than 50 pixels not within 3 pixels of the ring. Each is scored
    for roundness, area / edge^2, with its holes filled first (dark specks inside the marrow
    would otherwise add edge and lower its score). Small round fat spots can score as high
    as the marrow, so the marrow is the largest candidate scoring at least BONE_SCORE_RATIO
    of the best score. The marrow is returned with its holes filled.
    """
    props = regionprops(labeled_image)
    if not props:
        return np.zeros_like(labeled_image), np.zeros_like(labeled_image), None

    allContours = []

    others = np.zeros_like(labeled_image)
    for i, seg in enumerate(props):

        others[labeled_image == seg.label] = 1

        cur_ = np.zeros_like(labeled_image)
        cur_[labeled_image == seg.label] = 1
        cur_ = cur_.astype(np.uint8)

        dd_ = cv2.dilate(cur_.astype(np.uint8), kernel=np.ones((3, 3), np.uint8), iterations=3)
        test_ = seg1 * dd_
        if np.sum(test_) > 0:
            continue

        if np.sum(cur_ > 0.5) > 50:
            filled_ = ndi.binary_fill_holes(cur_).astype(np.uint8)
            edge_ = np.sum((filled_ - cv2.erode(filled_, kernel=np.ones((3, 3), np.uint8), iterations=1)) > 0.5)
            area_ = np.sum(filled_ > 0.5)
            allContours.append([i, area_, edge_, area_ / (edge_ * edge_)])

    if not allContours:
        # No bone candidate (e.g. slice outside the thigh): keep every bright spot as IMAT
        return np.zeros_like(labeled_image), others, None

    best = max(c[-1] for c in allContours)
    index_ = max((c for c in allContours if c[-1] >= BONE_SCORE_RATIO * best), key=lambda c: c[1])[0]
    selected_ = props[index_]

    bone = np.zeros_like(labeled_image)
    bone[ndi.binary_fill_holes(labeled_image == selected_.label)] = 1

    return bone, others * (1 - bone), selected_.bbox


def apply_roi_mask(image, bbox, mask=None):
    masked_image = np.zeros_like(image)
    if bbox is None:
        return masked_image
    minr, minc, maxr, maxc = bbox
    masked_image[minr:maxr, minc:maxc] = image[minr:maxr, minc:maxc]

    mask = mask.astype(np.uint8)

    # return masked_image
    if mask is not None:
        dilated = cv2.dilate(mask, kernel=np.ones((3, 3), np.uint8), iterations = 2)
        masked_image[dilated == 1] = 0

    return masked_image

def visualize_segments(image, masked_image, seg1, seg2):
    firstSeg = np.stack((image * 0.5, image * 0.5, image * 0.5), axis=-1) + np.stack((seg1 * 0.5, seg1 * 0, seg1 * 0), axis=-1)*255
    firstSeg = firstSeg.astype(np.uint8)

    secondSeg = np.stack((image * 0.5, image * 0.5, image * 0.5), axis=-1) + np.stack((seg2 * 0, seg2 * 0.5, seg2 * 0), axis=-1)*255
    secondSeg = secondSeg.astype(np.uint8)

    plt.figure(figsize=(15, 4))
    plt.subplot(1, 4, 1)
    plt.title("Original")
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.title("Masked by ROI")
    plt.imshow(masked_image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.title("Segment 1 (Largest Region)")
    plt.imshow(firstSeg)
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.title("Segment 2 (Inner Spots in ROI)")
    plt.imshow(secondSeg)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def refine_fat_ring(seg):
    contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    allContours = []

    for cnt in contours:
        if len(cnt) < 10:
            continue

        minX =  1000
        maxX = -1000
        minY =  1000
        maxY = -1000
        contour_ = []
        for i in range(len(cnt) - 1):
            pt0 = cnt[i][0]
            pt1 = cnt[i + 1][0]

            minX = min(minX, pt0[0], pt1[0])
            maxX = max(maxX, pt0[0], pt1[0])
            minY = min(minY, pt0[1], pt1[1])
            maxY = max(maxY, pt0[1], pt1[1])

            contour_.append(pt0)
        
        delX = maxX - minX
        delY = maxY - minY
        allContours.append([delX * delY, contour_])

    if not allContours:
        return seg  # No valid contours found: nothing to refine (it used to return a tuple here, which
                    # made segment() fail and the slice come out empty)

    # Sort contours by bounding box area, descending
    allContours.sort(key=lambda x: x[0], reverse=True)

    selected = allContours[0][1]

    zero_like = np.zeros_like(seg)
    for i in range(len(selected)):
        pt0 = selected[i]
        pt1 = selected[(i + 1) % len(selected)]

        cv2.line(zero_like, pt0, pt1, 1, 2)
        
    kernel = np.ones((3, 3), np.uint8)

    # Erode
    eroded = cv2.erode(seg, kernel, iterations=3)

    dilated = cv2.dilate(zero_like, kernel, iterations=3)
    dilated = dilated * seg

    eroded = np.logical_or(dilated, eroded)

    labeled = label(eroded, connectivity=2)

    # Get region properties
    regions = regionprops(labeled)

    # Sort regions by area (largest first)
    sorted_regions = sorted(regions, key=lambda r: r.area, reverse=True)

    # Extract sorted masks
    sorted_masks = [(labeled == region.label).astype(np.uint8) * 255 for region in sorted_regions]

    zero_like = np.logical_or(zero_like, sorted_masks[0]).astype(np.uint8)

    dilated = cv2.dilate(zero_like, kernel, iterations=4)
    dilated = dilated * seg
    totalArea = np.sum(dilated)

    totalArea = np.sum(seg)
    for k in range(1, 3):
        newMask = cv2.dilate(zero_like, kernel, iterations=k)
        # Apply the mask to the original segmentation
        newSeg = seg * newMask
        if np.sum(newSeg) > 0.98 * totalArea:
            break

    return newSeg

def segment(inputImage,dicom_path = None, return_outer=False, sat_thr=100, imat_thr=80, sat_blur=0, sat_guard=0, sat_core=0):
    # sat_thr / imat_thr: fat thresholds for the SAT ring and for IMAT inside it.
    # The defaults are for Dixon fat (_F) pixel values; t1_seg.py passes its own.
    # sat_blur: Gaussian sigma (pixels) of the image the SAT ring is found on; IMAT and
    # muscle always use the image itself. 0 = off (Dixon); see t1_seg.SAT_BLUR_PX.
    # sat_guard: with sat_blur, the ring is the one found on the image itself plus what
    # the blurred ring adds where the image is above sat_guard (clearly fat), connected.
    # Without it, bright muscle next to a blurred fat joint could become SAT.
    # sat_core: see below; 0 = off (Dixon); see t1_seg.SAT_CORE.
    if dicom_path is None:
        image = inputImage
    else:
        image = load_dicom_image(dicom_path)
    sat_image = cv2.GaussianBlur(np.asarray(image, np.float32), (0, 0), sat_blur) if sat_blur > 0 else image
    labeledFirst = segment_bright_regions(sat_image, threshold=sat_thr, min_size=5)
    seg1, _, bbox = extract_segments_in_roi(labeledFirst)

    # Refine Fat Ring

    # plt.imshow(seg1, cmap='gray')
    # plt.show()

    seg1 = refine_fat_ring(seg1)
    if sat_blur > 0:
        sharp, _, _ = extract_segments_in_roi(segment_bright_regions(image, threshold=sat_thr, min_size=5))
        sharp = refine_fat_ring(sharp)
        seg1 = _largest_component((sharp > 0) | ((seg1 > 0) & (np.asarray(image) > sat_guard))).astype(np.uint8)

    masked_image = apply_roi_mask(image, bbox, seg1)
    labeledSecond = segment_bright_regions(masked_image, threshold=imat_thr, min_size=5)
    
    # bone, seg2, _ = extract_segments_in_roi(labeledSecond)
    bone, seg2, _ = extract_bone_in_roi(labeledSecond, seg1)

    if sat_core > 0:
        # SAT must have a clearly fat core (> sat_core) connected to the ring; pixels between
        # sat_thr and sat_core count only within 2 pixels of it (the blurred edge of fat).
        # A broad region just above sat_thr (uniformly bright muscle on T1) becomes muscle.
        core = _largest_component((seg1 > 0) & (np.asarray(image) > sat_core))
        near = cv2.dilate(core.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2) > 0
        seg1 = _largest_component((seg1 > 0) & near).astype(np.uint8)

    flood_filled = seg1.copy()
    h, w = image.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Choose a pixel inside the hole — e.g., center of the image
    seed_point = (0, 0)

    # Perform flood fill inside the hole
    cv2.floodFill(flood_filled, mask, seed_point, 1)

    # Combine the filled hole with the original image
    outer_region = (mask[1:-1, 1:-1] == 0).astype(np.uint8)

    # Bright spots outside the body are noise, not IMAT
    seg2 = seg2 * outer_region

    # refine_fat_ring trims about a pixel off the inner edge of the SAT ring, and IMAT is not
    # searched within 2 pixels of the ring, so that fat used to end up as muscle. Give fat in
    # that band that is connected to the ring back to SAT.
    band = cv2.dilate(seg1.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2) > 0
    rim = band & (sat_image > sat_thr) & (outer_region > 0) & (seg2 == 0) & (bone == 0)
    seg1 = ndi.binary_propagation(seg1 > 0, mask=(seg1 > 0) | rim).astype(np.uint8)

    muscle = np.clip(outer_region - seg1 - seg2 - bone, 0, 1)

    '''
    plt.subplot(141)
    plt.imshow(bone, cmap='gray')
    plt.suptitle('bone')
    

    plt.subplot(142)
    plt.imshow(seg1, cmap='gray')
    plt.suptitle('seg1')

    plt.subplot(143)
    plt.imshow(seg2, cmap='gray')
    plt.suptitle('seg2')
    print('seg2: ', np.max(seg2))

    plt.subplot(144)
    # plt.imshow(flood_filled, cmap='gray')
    plt.suptitle('labelfirst')

    plt.show()
    '''

    if return_outer:
        return seg1, seg2, muscle, image, outer_region
    return seg1, seg2, muscle, image
    # return seg1, seg2, dark2_ - bone, image
    # visualize_segments(image, masked_image, seg1, seg2)


BALL_DIAMETER_MM = 10.0   # SAT vs IMAT: fat a ball of this size can reach from outside is SAT; 0 = off
BALL_SLICE_VOTE = True    # ... decided by majority over the slice and its two neighbours
IMAT_SAT_BORDER = 0.5     # IMAT pieces whose border is more than this share SAT become SAT; 0 = off


def _ball_reach(lab, pixel_mm, diameter_mm):
    """Pixels of one slice a ball of diameter_mm can cover when it rolls in from outside the body.

    The ball may lie in fat (SAT or IMAT) and outside the body, not in muscle or bone. It can
    sit partly outside the skin, so a SAT ring thinner than the ball is still reached.
    """
    r = diameter_mm / 2 / pixel_mm
    ri = int(np.ceil(r))
    yy, xx = np.mgrid[-ri:ri + 1, -ri:ri + 1]
    ball = ((yy ** 2 + xx ** 2) <= r * r).astype(np.uint8)
    free = (lab == 1) | (lab == 2) | ~ndi.binary_fill_holes(lab > 0)
    pad = ri + 2
    centres = cv2.erode(np.pad(free, pad, constant_values=True).astype(np.uint8), ball) > 0
    pieces, _ = ndi.label(centres)
    edge = np.unique(np.concatenate([pieces[0], pieces[-1], pieces[:, 0], pieces[:, -1]]))
    reach = cv2.dilate(np.isin(pieces, edge[edge > 0]).astype(np.uint8), ball) > 0
    return reach[pad:-pad, pad:-pad]


def _sat_border_rule(lab, share):
    """IMAT pieces whose outer border is more than `share` SAT become SAT (one slice)."""
    pieces, n = ndi.label(lab == 2, structure=np.ones((3, 3)))
    if not n:
        return lab
    border = (cv2.dilate((pieces > 0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & (pieces == 0)
    owner = ndi.maximum_filter(pieces, size=3)[border]
    total = np.bincount(owner, minlength=n + 1)
    sat = np.bincount(owner, weights=(lab[border] == 1), minlength=n + 1)
    flip = np.nonzero((total > 0) & (sat > share * total))[0]
    out = lab.copy()
    out[np.isin(pieces, flip[flip > 0])] = 1
    return out


def separate_sat_imat(labels, pixel_mm, diameter_mm=None, slice_vote=None, border_share=None):
    """Re-decide which fat is SAT and which is IMAT (labels (n_slices, H, W); muscle and bone unchanged).

    1. Rolling ball (BALL_DIAMETER_MM): fat a ball of that diameter can reach from outside the
       body is SAT, other fat IMAT. Fat pockets between muscles with a mouth narrower than the
       ball become IMAT, whatever the width of their neck; SAT keeps thin parts of the ring.
    2. Slice vote (BALL_SLICE_VOTE): a fat pixel takes the SAT / IMAT label most of the slices
       k-1, k, k+1 give it (counting the slices where it is fat), so a pocket whose mouth is
       near the ball size does not flip from slice to slice.
    3. Border rule (IMAT_SAT_BORDER): IMAT pieces mostly surrounded by SAT become SAT.
    Tuned on rawThigh (both sequences): T1 vs Dixon IMAT Dice 0.513 -> 0.536, mean |difference|
    per scan 9.0% -> 7.5%; IMAT is ~30% larger than with seg.py's SAT step alone.
    """
    diameter_mm = BALL_DIAMETER_MM if diameter_mm is None else diameter_mm
    slice_vote = BALL_SLICE_VOTE if slice_vote is None else slice_vote
    border_share = IMAT_SAT_BORDER if border_share is None else border_share
    labels = np.asarray(labels)
    out = labels.copy()
    fat = (labels == 1) | (labels == 2)
    if diameter_mm:
        reach = np.array(map_slices(_ball_reach, list(labels), [pixel_mm] * len(labels), [diameter_mm] * len(labels)))
        sat = fat & reach
        if slice_vote and len(labels) > 1:
            s, f = sat.astype(np.int16), fat.astype(np.int16)
            s_sum, f_sum = s.copy(), f.copy()
            s_sum[1:] += s[:-1]; s_sum[:-1] += s[1:]
            f_sum[1:] += f[:-1]; f_sum[:-1] += f[1:]
            sat = np.where(2 * s_sum > f_sum, True, np.where(2 * s_sum < f_sum, False, sat)) & fat
        out[fat & sat] = 1
        out[fat & ~sat] = 2
    if border_share:
        out = np.array([_sat_border_rule(l, border_share) for l in out])
    return out


def _largest_component(mask):
    labeled, n = ndi.label(mask)
    if n <= 1:
        return mask > 0
    return labeled == (np.argmax(np.bincount(labeled.ravel())[1:]) + 1)


def _solidity(mask):
    if not mask.any():
        return 0.0
    return regionprops(mask.astype(np.uint8))[0].solidity


def _segment_parts(im, sat_thr, imat_thr, sat_blur=0, sat_guard=0, sat_core=0):
    """segment() of one slice as (SAT, IMAT, muscle, body outline); empty if it fails."""
    try:
        seg1, seg2, muscle, _, outer = segment(im, return_outer=True, sat_thr=sat_thr, imat_thr=imat_thr,
                                               sat_blur=sat_blur, sat_guard=sat_guard, sat_core=sat_core)
        return seg1.astype(np.uint8), seg2.astype(np.uint8), muscle.astype(np.uint8), outer > 0
    except Exception:
        empty = np.zeros(np.shape(im), np.uint8)
        return empty, empty, empty, empty > 0


def _fix_bone(parts, window=2):
    """Keep the marrow consistent through the stack; swap it back where a slice labelled it IMAT.

    The femur hardly moves between slices, but a single slice can pick a round fat spot as
    the bone and leave the marrow as IMAT. Start from the slice whose bone overlaps the bone
    of the most slices within `window`, then walk outward: on each slice the bone is the
    candidate (the slice's own bone or one of its IMAT spots) overlapping the previous
    slice's bone most. A replaced bone, a bright spot, becomes IMAT. Bone is what is inside
    the body outline but neither SAT, IMAT nor muscle.
    """
    n = len(parts)
    bones = [(p[3] > 0) & (p[0] == 0) & (p[1] == 0) & (p[2] == 0) for p in parts]
    if n < 2 or not any(b.any() for b in bones):
        return list(parts)
    agree = [sum(1 for j in range(max(0, i - window), min(n, i + window + 1))
                 if j != i and (bones[i] & bones[j]).any()) for i in range(n)]
    start = max(range(n), key=lambda i: (agree[i], bones[i].any(), -abs(i - n // 2)))

    fixed = list(parts)
    for order in (range(start + 1, n), range(start - 1, -1, -1)):
        prev = bones[start]
        for i in order:
            seg1, seg2, muscle, outer = parts[i]
            # A slice without bone (e.g. the marrow touched the ring and was left as IMAT) is
            # checked too, so its marrow can be taken back from the IMAT spots
            if not prev.any() or (bones[i].any() and (bones[i] & prev).sum() >= 0.5 * min(bones[i].sum(), prev.sum())):
                prev = bones[i] if bones[i].any() else prev
                continue
            pieces, m = ndi.label(seg2)
            cover = np.bincount(pieces[prev].ravel(), minlength=m + 1)
            cover[0] = 0
            k = int(np.argmax(cover))
            if cover[k] <= (bones[i] & prev).sum():
                prev = bones[i] if bones[i].any() else prev
                continue
            if not bones[i].any() and cover[k] < 0.5 * min((pieces == k).sum(), prev.sum()):
                continue   # no bone here and no IMAT spot that matches the previous bone
            new_bone = ndi.binary_fill_holes(pieces == k) & (seg1 == 0)
            seg2 = ((seg2 > 0) & ~new_bone) | (bones[i] & ~new_bone)
            muscle = (muscle > 0) & ~new_bone
            fixed[i] = (seg1, seg2.astype(np.uint8), muscle.astype(np.uint8), outer)
            prev = new_bone
    return fixed


def segment_stack(slices, max_shift=2, solidity_thr=0.98, ridge_thr=0.06, sat_thr=100, imat_thr=80, sat_blur=0, sat_guard=0, sat_core=0):
    """Segment a stack of axial thigh slices: 1 SAT, 2 IMAT, 3 muscle.

    Near the hip the other leg or the perineum lies against the thigh and is picked up
    as SAT. A clean thigh outline is nearly convex (solidity >= solidity_thr), a merged
    one is not. Clean slices are kept as they are. On merged slices the thigh is limited
    to the neighbouring slice's thigh grown by max_shift pixels, and is also cut along
    the thin dark skin line where the two legs touch (Sato dark-ridge response above
    ridge_thr). Slices are processed outward from the clean slice nearest the middle.
    Marrow missed on a slice (labelled IMAT) is restored from its neighbours (_fix_bone).

    slices: sequence of 2D arrays. sat_blur / sat_guard / sat_core: see segment().
    Returns an int16 array (n_slices, H, W).
    """
    n = len(slices)
    parts = _fix_bone(map_slices(_segment_parts, slices, [sat_thr] * n, [imat_thr] * n,
                                 [sat_blur] * n, [sat_guard] * n, [sat_core] * n))

    body = [_largest_component(p[3]) for p in parts]
    solidity = [_solidity(b) for b in body]
    clean = [i for i in range(n) if solidity[i] >= solidity_thr]
    start = min(clean, key=lambda i: abs(i - n // 2)) if clean else int(np.argmax(solidity))

    scale = np.percentile(np.asarray(slices, dtype=np.float32), 99) or 1.0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * max_shift + 1, 2 * max_shift + 1))
    thigh = [None] * n
    thigh[start] = body[start]
    for order in (range(start + 1, n), range(start - 1, -1, -1)):
        prev = thigh[start]
        for i in order:
            if solidity[i] >= solidity_thr or not prev.any() or not body[i].any():
                cur = body[i]
            else:
                allowed = (cv2.dilate(prev.astype(np.uint8), kernel) > 0) & parts[i][3]
                # Cut along the skin line between the legs, keep the piece that
                # overlaps the previous thigh most, then give back the cut line itself
                ridge = sato(np.asarray(slices[i], np.float32) / scale, sigmas=[1, 1.5, 2], black_ridges=True)
                pieces, n_pieces = ndi.label(allowed & (ridge <= ridge_thr))
                if n_pieces:
                    overlap = np.bincount(pieces[prev].ravel(), minlength=n_pieces + 1)
                    overlap[0] = 0
                    piece = (pieces == np.argmax(overlap)).astype(np.uint8)
                    cur = (cv2.dilate(piece, np.ones((3, 3), np.uint8), iterations=2) > 0) & allowed
                else:
                    cur = allowed
                cur = _largest_component(cur)
                cur = ndi.binary_fill_holes(cur) & parts[i][3]
            thigh[i] = cur
            prev = cur

    labels = np.zeros((n,) + np.shape(slices[0]), np.int16)
    for i, (seg1, seg2, muscle, _) in enumerate(parts):
        t = thigh[i].astype(np.uint8)
        labels[i] = seg1 * t + 2 * seg2 * t + 3 * muscle * t
    return labels

# Example usage:
# main("path_to_your_file.dcm")


# Example usage:
# main("path_to_your_file.dcm")

if __name__ == "__main__":

    sourceDir = r'C:\Users\hctsbo\Desktop\fatSeg\03160070NHCCJ9\6pt_DIXON_VIBE'
    sliceId = 34
    it = 0
    for f in os.listdir(sourceDir):
        # dcm = read_dcm(os.path.join(sourceDir, f))
        '''
        if it != sliceId:
            it += 1
            continue
        print(it)
        ''' 
        seg1, seg2, masked_image, image = segment('nothing', os.path.join(sourceDir, f))
        visualize_segments(image, masked_image, seg1, seg2)
        it += 1 
    
