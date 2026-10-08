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


def extract_bone_in_roi(labeled_image, seg1):
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
        edge_ = cur_ - cv2.erode(cur_.astype(np.uint8), kernel=np.ones((3, 3), np.uint8), iterations=1)
        edge_ = np.sum(edge_ > 0.5)
        area_ = np.sum(cur_ > 0.5)

        dd_ = cv2.dilate(cur_.astype(np.uint8), kernel=np.ones((3, 3), np.uint8), iterations=3)
        test_ = seg1 * dd_
        if np.sum(test_) > 0:
            continue

        if area_ > 50:
            allContours.append([i, area_, edge_, edge_ / area_, area_ / (edge_ * edge_)])

        

    if not allContours:
        # No bone candidate (e.g. slice outside the thigh): keep every bright spot as IMAT
        return np.zeros_like(labeled_image), others, None

    allContours = sorted(allContours, key=lambda x: x[-1], reverse=True)
    index_ = allContours[0][0]
    selected_ = props[index_]

    bone = np.zeros_like(labeled_image)
    bone[labeled_image == selected_.label] = 1

    return bone, others - bone, selected_.bbox


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
        return seg, None  # No valid contours found

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

def segment(inputImage,dicom_path = None, return_outer=False, sat_thr=100, imat_thr=80):
    # sat_thr / imat_thr: fat thresholds for the SAT ring and for IMAT inside it.
    # The defaults are for Dixon fat (_F) pixel values; t1_seg.py passes its own.
    if dicom_path is None:
        image = inputImage
    else:
        image = load_dicom_image(dicom_path)
    labeledFirst = segment_bright_regions(image, threshold=sat_thr, min_size=5)
    seg1, _, bbox = extract_segments_in_roi(labeledFirst)

    # Refine Fat Ring

    # plt.imshow(seg1, cmap='gray')
    # plt.show()

    seg1 = refine_fat_ring(seg1)

    masked_image = apply_roi_mask(image, bbox, seg1)
    labeledSecond = segment_bright_regions(masked_image, threshold=imat_thr, min_size=5)
    
    # bone, seg2, _ = extract_segments_in_roi(labeledSecond)
    bone, seg2, _ = extract_bone_in_roi(labeledSecond, seg1)

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
    rim = band & (image > sat_thr) & (outer_region > 0) & (seg2 == 0) & (bone == 0)
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


def _largest_component(mask):
    labeled, n = ndi.label(mask)
    if n <= 1:
        return mask > 0
    return labeled == (np.argmax(np.bincount(labeled.ravel())[1:]) + 1)


def _solidity(mask):
    if not mask.any():
        return 0.0
    return regionprops(mask.astype(np.uint8))[0].solidity


def segment_stack(slices, max_shift=2, solidity_thr=0.98, ridge_thr=0.06, sat_thr=100, imat_thr=80):
    """Segment a stack of axial thigh slices: 1 SAT, 2 IMAT, 3 muscle.

    Near the hip the other leg or the perineum lies against the thigh and is picked up
    as SAT. A clean thigh outline is nearly convex (solidity >= solidity_thr), a merged
    one is not. Clean slices are kept as they are. On merged slices the thigh is limited
    to the neighbouring slice's thigh grown by max_shift pixels, and is also cut along
    the thin dark skin line where the two legs touch (Sato dark-ridge response above
    ridge_thr). Slices are processed outward from the clean slice nearest the middle.

    slices: sequence of 2D arrays. Returns an int16 array (n_slices, H, W).
    """
    n = len(slices)
    parts = []
    for im in slices:
        try:
            seg1, seg2, muscle, _, outer = segment(im, return_outer=True, sat_thr=sat_thr, imat_thr=imat_thr)
            parts.append((seg1.astype(np.uint8), seg2.astype(np.uint8), muscle.astype(np.uint8), outer > 0))
        except Exception:
            empty = np.zeros(np.shape(im), np.uint8)
            parts.append((empty, empty, empty, empty > 0))

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
    
