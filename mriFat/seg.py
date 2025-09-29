import pydicom
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
import os
import torch
import torch.nn.functional as F
from torchvision import transforms

def load_unet_model(model_path):
    """Load the UNet segmentation model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model, device

def apply_unet_segmentation(masked_image, model, device):
    """Apply UNet model to the masked image"""
    # Normalize the image to [0, 1] range
    image_normalized = masked_image.astype(np.float32) / 255.0

    # Convert to tensor and add batch and channel dimensions
    image_tensor = torch.from_numpy(image_normalized).unsqueeze(0).unsqueeze(0).to(device)

    # Apply the model
    with torch.no_grad():
        prediction = model(image_tensor)

    # Convert back to numpy and remove batch/channel dimensions
    prediction = prediction.squeeze().cpu().numpy()

    # Apply sigmoid if needed (depends on your model output)
    prediction = torch.sigmoid(torch.from_numpy(prediction)).numpy()

    # Threshold the prediction to get binary mask
    segmentation_mask = (prediction > 0.5).astype(np.uint8)

    return segmentation_mask, prediction

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

def apply_roi_mask(image, bbox, mask=None):
    masked_image = np.zeros_like(image)
    if bbox is None:
        return masked_image
    minr, minc, maxr, maxc = bbox
    masked_image[minr:maxr, minc:maxc] = image[minr:maxr, minc:maxc]

    # return masked_image
    if mask is not None:
        masked_image[mask == 1] = 0

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

def segment(inputImage, dicom_path=None, unet_model_path="unet_segmentation_deployment_model.pth"):
    if dicom_path is None:
        image = inputImage
    else:
        image = load_dicom_image(dicom_path)

    # Load UNet model once
    model, device = load_unet_model(unet_model_path)

    labeledFirst = segment_bright_regions(image, threshold=100, min_size=5)
    seg1, _, bbox = extract_segments_in_roi(labeledFirst)

    masked_image = apply_roi_mask(image, bbox, seg1)

    # Apply UNet segmentation to the masked image
    unet_segmentation, unet_prediction = apply_unet_segmentation(masked_image, model, device)
    labeledSecond = segment_bright_regions(masked_image, threshold=80, min_size=5)
    
    bone, seg2, _ = extract_segments_in_roi(labeledSecond)
    masked_image = apply_roi_mask(image, bbox, bone)

    masked_image = apply_roi_mask(image, bbox, bone)
    labeledThird = segment_bright_regions(masked_image, threshold=80, min_size=5)

    _, seg2, _ = extract_segments_in_roi(labeledThird)


    labeledDark = segment_dark_regions(masked_image, threshold=100, min_size=5)
    dark1_, dark2_, _ = extract_segments_in_roi(labeledDark)

    '''
    plt.subplot(1, 4, 1)
    plt.imshow(masked_image, cmap='gray')
    plt.subplot(1, 4, 2)
    plt.imshow(dark1_, cmap='gray')
    plt.subplot(1, 4, 3)
    plt.imshow(dark2_ - bone, cmap='gray')
    plt.subplot(1, 4, 4)
    plt.imshow(image, cmap='gray')
    plt.show()
    '''

    # return seg1, seg2, masked_image, image
    return seg1, seg2, dark2_ - bone, image, unet_segmentation, unet_prediction
    # visualize_segments(image, masked_image, seg1, seg2)

# Example usage:
# main("path_to_your_file.dcm")


# Example usage:
# main("path_to_your_file.dcm")

if __name__ == "__main__":

    sourceDir = '0624-0001-NHC-RBA_(1)'
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
        seg1, seg2, masked_image, image, unet_seg, unet_pred = segment('nothing', os.path.join(sourceDir, f))
        visualize_segments(image, masked_image, seg1, seg2)

        # Optionally visualize UNet results
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("UNet Segmentation")
        plt.imshow(unet_seg, cmap='gray')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.title("UNet Prediction (Raw)")
        plt.imshow(unet_pred, cmap='viridis')
        plt.axis('off')
        plt.show()
        it += 1 
    
