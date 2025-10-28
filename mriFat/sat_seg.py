import numpy as np
from scipy.ndimage import label, find_objects

def auto_segment(image_stack, segmentation_threshold):
    """
    Perform automatic segmentation on the image stack.
    For each slice, finds the largest connected component and creates a bounding box,
    then excludes segmentations outside the box.

    Args:
        image_stack: 3D numpy array of image data
        segmentation_threshold: threshold value for segmentation

    Returns:
        segmentation: 3D numpy array with segmented data
    """
    # Initial thresholding
    segmentation = np.where(image_stack > segmentation_threshold, 1, 0)
    '''
    # Process each slice to find largest area and apply bounding box
    for i in range(segmentation.shape[0]):
        slice_seg = segmentation[i, :, :]

        if np.sum(slice_seg) == 0:  # Skip empty slices
            continue

        # Find connected components
        labeled_array, num_features = label(slice_seg)

        if num_features == 0:
            continue

        # Find the largest connected component
        component_sizes = []
        for component_id in range(1, num_features + 1):
            component_size = np.sum(labeled_array == component_id)
            component_sizes.append((component_size, component_id))

        # Get the largest component
        largest_size, largest_id = max(component_sizes, key=lambda x: x[0])

        # Create mask for largest component only
        largest_component_mask = (labeled_array == largest_id)

        # Find bounding box of largest component
        coords = np.where(largest_component_mask)
        if len(coords[0]) > 0:
            min_row, max_row = coords[0].min(), coords[0].max()
            min_col, max_col = coords[1].min(), coords[1].max()

            # Create new slice with only segmentations inside the bounding box
            new_slice = np.zeros_like(slice_seg)
            box_region = slice_seg[min_row:max_row+1, min_col:max_col+1]
            new_slice[min_row:max_row+1, min_col:max_col+1] = box_region

            segmentation[i, :, :] = new_slice
    '''
    print('before: ', segmentation.shape)
    res = np.transpose(segmentation, [2, 1, 0])
    res = np.transpose(res, [1, 0, 2])
    print('res: ', res.shape)
    return res #np.transpose(segmentation, [2, 1, 0])