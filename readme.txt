### `readmd.md` - U-Net Thigh Fat Segmentation Model Documentation

This document provides key information about the trained U-Net model for multi-class semantic segmentation of thigh fat from medical images.

#### 1. Model Overview
*   **Architecture:** U-Net
*   **Purpose:** Multi-class semantic segmentation of thigh fat.
*   **Classes (4 total):**
    *   0: Background
    *   1: SAT (Subcutaneous Adipose Tissue)
    *   2: IMAT (Intramuscular Adipose Tissue)
    *   3: Muscle
*   **Training Data:** 2D slices extracted from 3D NIfTI medical images, processed slice-by-slice.

#### 2. Data Dimensions
*   **Input Data Dimensions (to the model):**
    *   Single image slice: `(256, 256, 1)`
    *   Batched input (during training/inference): `(BATCH_SIZE, 256, 256, 1)`
    *   _Description:_ Each input is a 256x256 pixel image with a single channel (representing intensity/grayscale data).

*   **Output Data Dimensions (from the model):**
    *   Raw model output (probabilities per class): `(256, 256, 4)`
    *   Batched output: `(BATCH_SIZE, 256, 256, 4)`
    *   _Description:_ For each 256x256 pixel, the model outputs 4 probability values, one for each of the `NUM_CLASSES`. To obtain the final segmented mask (class labels), an `np.argmax` operation is applied along the last axis, resulting in a mask of shape `(256, 256, 1)` or `(256, 256)` where each pixel contains its predicted class label (0, 1, 2, or 3).

#### 3. CPU Deployment Considerations (Local Machine)
*   **Model File Location:** The trained model is saved as `unet_thighfat_segmentation_model.keras` within the `saved_models` directory.
*   **RAM Requirements for Inference:**
    *   To load the model weights and perform inference on a batch of images on a CPU, an estimated **2-4 GB of RAM** should be sufficient.
    *   This estimate is based on the model's size (U-Net with 256x256 inputs, 4 classes) and successful operation within Colab's standard RAM limits for training (12-13 GB).
    *   Actual RAM usage may vary slightly depending on the operating system, other running applications, and specific TensorFlow/Keras versions.
*   **Dependencies:** To load and use this model on a local CPU, you will need:
    *   Python 3.x
    *   TensorFlow (CPU version is sufficient)
    *   NumPy
    *   nibabel (for NIfTI file handling if you are using `.nii.gz` files for inference)

#### 4. Usage Example (Pseudo-code for Inference)
```python
import tensorflow as tf
import numpy as np
import nibabel as nib # If loading NIfTI files

# Define IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES as used during training
IMG_HEIGHT = 256
IMG_WIDTH = 256
NUM_CLASSES = 4

# Load the saved model
model = tf.keras.models.load_model('/content/drive/MyDrive/thighFat/saved_models/unet_thighfat_segmentation_model.keras')

# --- Example: Prepare a single image for prediction ---
# Replace this with your actual image loading and preprocessing logic
# For NIfTI files, you would load a slice and preprocess it similar to the training pipeline

# Placeholder for a preprocessed image (e.g., a single 256x256 grayscale slice)
# image_data_raw = nib.load('path/to/your/image.nii.gz').get_fdata()[:, :, slice_idx]
# image_data_processed = tf.image.resize(np.expand_dims(image_data_raw, axis=-1), [IMG_HEIGHT, IMG_WIDTH])
# image_data_processed = image_data_processed / tf.reduce_max(image_data_processed)

# Create a dummy image for demonstration (replace with actual image_data_processed)
dummy_image = np.random.rand(IMG_HEIGHT, IMG_WIDTH, 1).astype(np.float32)

# Add batch dimension (model expects batches)
input_image_batch = np.expand_dims(dummy_image, axis=0)

# Make prediction
prediction = model.predict(input_image_batch)

# Get the segmented mask by taking the argmax along the class axis
segmented_mask = np.argmax(prediction[0], axis=-1)

print(f"Input image shape: {input_image_batch.shape}")
print(f"Raw prediction output shape: {prediction.shape}")
print(f"Segmented mask shape: {segmented_mask.shape}")

# segmented_mask now contains integer labels (0, 1, 2, 3) for each pixel.
# You can visualize or further process this mask.
```