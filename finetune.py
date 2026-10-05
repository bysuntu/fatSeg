"""
Fine-tune the existing U-Net model on local labeled NIfTI data.

Data layout:  thighFat/<case>/img.nii.gz  (320, W, slices)
                              seg.nii.gz  (W, 320, slices)  — transposed vs img

Labels: 0=background, 1=SAT, 2=IMAT, 3=muscle, -1=unlabeled (masked out)
"""

import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow import keras

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(BASE_DIR, 'thighFat')
_FINETUNED     = os.path.join(BASE_DIR, 'unet_thighfat_finetuned.keras')
_ORIGINAL      = os.path.join(BASE_DIR, 'unet_thighfat_segmentation_model_best_loss.keras')
MODEL_PATH     = _FINETUNED if os.path.exists(_FINETUNED) else _ORIGINAL
OUTPUT_PATH    = os.path.join(BASE_DIR, 'unet_thighfat_finetuned.keras')

IMG_SIZE       = 256
NUM_CLASSES    = 4
BATCH_SIZE     = 4
EPOCHS         = 60
LR             = 1e-5
VAL_SPLIT      = 0.15


def load_slices():
    images, labels, weights = [], [], []

    for case in sorted(os.listdir(DATA_DIR)):
        case_dir = os.path.join(DATA_DIR, case)
        if not os.path.isdir(case_dir) or case == 'saved_models':
            continue
        img_path = os.path.join(case_dir, 'img.nii.gz')
        seg_path = os.path.join(case_dir, 'seg.nii.gz')
        if not os.path.exists(img_path) or not os.path.exists(seg_path):
            continue

        img_vol = nib.load(img_path).get_fdata().astype(np.float32)  # (320, W, S)
        seg_vol = nib.load(seg_path).get_fdata().astype(np.int32)    # (W, 320, S)

        n_slices = img_vol.shape[2]
        print(f"  {case}: {n_slices} slices", flush=True)

        for s in range(n_slices):
            img_slice = img_vol[:, :, s]          # (320, W)
            seg_slice = seg_vol[:, :, s].T        # (320, W)  — realign axes

            # Skip slices with no labeled foreground
            if np.sum(seg_slice > 0) == 0:
                continue

            # Map unlabeled (-1) to background (0); clip to valid range
            seg_slice = np.where(seg_slice < 0, 0, seg_slice)
            seg_slice = np.clip(seg_slice, 0, NUM_CLASSES - 1).astype(np.uint8)

            # Resize image to model input size
            img_r = tf.image.resize(img_slice[..., np.newaxis],
                                    [IMG_SIZE, IMG_SIZE]).numpy()
            seg_r = tf.image.resize(seg_slice[..., np.newaxis],
                                    [IMG_SIZE, IMG_SIZE],
                                    method='nearest').numpy()[:, :, 0].astype(np.uint8)

            # Normalize image to [0, 1]
            max_val = np.max(img_r)
            if max_val > 0:
                img_r = img_r / max_val

            # Per-pixel sample weight: foreground pixels weighted 5× over background
            w = np.where(seg_r > 0, 5.0, 1.0).astype(np.float32)

            images.append(img_r)
            labels.append(seg_r)
            weights.append(w)

    return np.array(images), np.array(labels), np.array(weights)


def main():
    print("Loading training data...")
    images, labels, weights = load_slices()
    print(f"Total slices: {len(images)}")

    # Class distribution for info
    for c in range(NUM_CLASSES):
        pct = 100 * np.sum(labels == c) / labels.size
        print(f"  Class {c}: {pct:.1f}%")

    print(f"\nLoading model from {MODEL_PATH}...")
    model = keras.models.load_model(MODEL_PATH)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy'],
        weighted_metrics=['accuracy'],
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            OUTPUT_PATH,
            monitor='val_loss',
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    print(f"\nFine-tuning for up to {EPOCHS} epochs (lr={LR})...")
    model.fit(
        images, labels,
        sample_weight=weights,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VAL_SPLIT,
        callbacks=callbacks,
        shuffle=True,
    )

    print(f"\nDone. Best model saved to:\n  {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
