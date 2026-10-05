import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
sys.path.insert(0, r'C:\Users\bysu\Desktop\fatSeg\mriFat')
import numpy as np
import nibabel as nib
import tensorflow as tf
from readSpace_threeButton import parseDicomFolder

pixels_f, info_f = parseDicomFolder(r'C:\Users\bysu\Desktop\fatSeg\6pt_DIXON_VIBE', '6pt_DIXON_VIBE_F')
model = tf.keras.models.load_model(r'C:\Users\bysu\Desktop\fatSeg\unet_thighfat_segmentation_model_best_loss.keras')
res_data = nib.load(r'C:\Users\bysu\Desktop\fatSeg\res.nii.gz').get_fdata()

image_stack = np.array(pixels_f)
IMG_HEIGHT, IMG_WIDTH = 256, 256

# Reproduce ai_seg_mode exactly
fatSeg = []
for i in range(len(pixels_f)):
    cur_ = pixels_f[i].astype(np.float32)
    cur_t = cur_.T
    nifti_h, nifti_w = cur_t.shape[:2]
    resized = tf.image.resize(cur_t[..., np.newaxis], [IMG_HEIGHT, IMG_WIDTH])
    max_val = tf.reduce_max(resized)
    if max_val > 0:
        resized = resized / max_val
    input_img = tf.expand_dims(resized, axis=0)
    prediction = model.predict(input_img, verbose=0)
    mask = np.argmax(prediction[0], axis=-1).astype(np.uint8)
    mask_resized = tf.image.resize(mask[..., np.newaxis], [nifti_h, nifti_w], method='nearest')
    mask_resized = mask_resized.numpy()[:, :, 0].astype(np.uint8)
    fatSeg.append(mask_resized)

ai_seg = np.array(fatSeg).transpose(1, 2, 0)

# Simulate display for both:
test_slice = 20
mri_slice = image_stack[test_slice]  # (260, 320)

print(f"mri_slice shape: {mri_slice.shape}")
print(f"ai_seg shape: {ai_seg.shape}")
print(f"res_data shape: {res_data.shape}")

# AI Seg display path
ai_slice = ai_seg[:, :, test_slice]
print(f"\nAI seg slice shape: {ai_slice.shape}")
if ai_slice.shape != mri_slice.shape:
    print("  -> .T applied in display")
    ai_display = ai_slice.T
else:
    print("  -> no .T needed")
    ai_display = ai_slice

# res.nii.gz display path
res_slice = res_data[:, :, test_slice]
print(f"res slice shape: {res_slice.shape}")
if res_slice.shape != mri_slice.shape:
    print("  -> .T applied in display")
    res_display = res_slice.T
else:
    print("  -> no .T needed")
    res_display = res_slice

print(f"\nai_display shape: {ai_display.shape}")
print(f"res_display shape: {res_display.shape}")

# Compare what's actually shown on screen
identical = np.array_equal(ai_display, res_display)
mismatched = np.sum(ai_display != res_display)
print(f"\nDisplay identical: {identical}")
print(f"Display mismatched pixels: {mismatched} / {ai_display.size} ({mismatched/ai_display.size*100:.4f}%)")

# Check a specific pixel to verify alignment
# Find a pixel with label=1 in res_display
label1_coords = np.where(res_display == 1)
if len(label1_coords[0]) > 0:
    r, c = label1_coords[0][0], label1_coords[1][0]
    print(f"\nSpot check at display [r={r}, c={c}]:")
    print(f"  res_display: {int(res_display[r, c])}")
    print(f"  ai_display:  {int(ai_display[r, c])}")
    print(f"  mri_slice pixel value: {mri_slice[r, c]}")
