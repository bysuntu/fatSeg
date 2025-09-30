import pydicom
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
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

def segment(inputImage,dicom_path = None):
    if dicom_path is None:
        image = inputImage
    else:
        image = load_dicom_image(dicom_path)
    labeledFirst = segment_bright_regions(image, threshold=100, min_size=5)
    seg1, _, bbox = extract_segments_in_roi(labeledFirst)

    # Refine Fat Ring

    # plt.imshow(seg1, cmap='gray')
    # plt.show()

    seg1 = refine_fat_ring(seg1)

    masked_image = apply_roi_mask(image, bbox, seg1)
    labeledSecond = segment_bright_regions(masked_image, threshold=80, min_size=5)
    
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

    muscle = outer_region - seg1 - seg2 - bone

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

    return seg1, seg2, muscle, image
    # return seg1, seg2, dark2_ - bone, image
    # visualize_segments(image, masked_image, seg1, seg2)

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
    
