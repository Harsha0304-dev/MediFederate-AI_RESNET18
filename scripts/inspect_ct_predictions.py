from pathlib import Path
import sys

# ============================================================
# Project path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

# ============================================================
# Imports
# ============================================================

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.medifederate.unet import UNet


# ============================================================
# Configuration
# ============================================================

PATCH_ROOT = (
    PROJECT_ROOT /
    "data" /
    "processed" /
    "ct_patches"
)

TEST_IMAGE_DIR = (
    PATCH_ROOT /
    "test" /
    "images"
)

TEST_MASK_DIR = (
    PATCH_ROOT /
    "test" /
    "masks"
)

MODEL_PATH = (
    PROJECT_ROOT /
    "models" /
    "unet_ct_patch_best.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT /
    "results" /
    "ct_prediction_inspection"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

INPUT_SIZE = 128

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Header
# ============================================================

print("=" * 60)

print(
    "MediFederate AI"
)

print(
    "CT Prediction Visual Inspection"
)

print("=" * 60)

print(
    f"Device: {DEVICE}"
)

print(
    f"Model: {MODEL_PATH}"
)

print(
    f"Test directory: {TEST_IMAGE_DIR}"
)

print("=" * 60)


# ============================================================
# Check files
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Model not found:\n"
        f"{MODEL_PATH}"
    )


test_files = sorted(
    TEST_IMAGE_DIR.glob("*.npy")
)


if len(test_files) == 0:

    raise RuntimeError(
        "No test patches found."
    )


print(
    f"Test patches found: "
    f"{len(test_files)}"
)


# ============================================================
# Load model
# ============================================================

model = UNet(
    in_channels=1,
    out_channels=1
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

# Support checkpoint dictionary
if isinstance(
    checkpoint,
    dict
) and "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

else:

    model.load_state_dict(
        checkpoint
    )

model = model.to(
    DEVICE
)

model.eval()


print(
    "Model loaded successfully."
)


# ============================================================
# Dice / IoU
# ============================================================

def calculate_metrics(
    prediction,
    target
):

    prediction = prediction.astype(
        bool
    )

    target = target.astype(
        bool
    )

    intersection = np.logical_and(
        prediction,
        target
    ).sum()

    prediction_sum = (
        prediction.sum()
    )

    target_sum = (
        target.sum()
    )

    union = np.logical_or(
        prediction,
        target
    ).sum()

    dice = (
        2.0 * intersection + 1e-8
    ) / (
        prediction_sum +
        target_sum +
        1e-8
    )

    iou = (
        intersection + 1e-8
    ) / (
        union + 1e-8
    )

    return dice, iou


# ============================================================
# Process test patches
# ============================================================

all_results = []


for index, image_path in enumerate(
    test_files
):

    mask_path = (
        TEST_MASK_DIR /
        image_path.name
    )

    if not mask_path.exists():

        print(
            f"Mask missing for "
            f"{image_path.name}"
        )

        continue

    # --------------------------------------------------------
    # Load patch
    # --------------------------------------------------------

    image = np.load(
        image_path
    ).astype(
        np.float32
    )

    mask = np.load(
        mask_path
    ).astype(
        np.float32
    )

    # --------------------------------------------------------
    # Normalize CT
    # --------------------------------------------------------

    image_display = np.clip(
        image,
        -1000,
        400
    )

    image_display = (
        image_display + 1000
    ) / 1400.0

    image_tensor = torch.from_numpy(
        image_display
    ).float()

    image_tensor = (
        image_tensor
        .unsqueeze(0)
        .unsqueeze(0)
    )

    image_tensor = image_tensor.to(
        DEVICE
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with torch.no_grad():

        logits = model(
            image_tensor
        )

        probabilities = torch.sigmoid(
            logits
        )

    probability = (
        probabilities[0, 0]
        .cpu()
        .numpy()
    )

    prediction = (
        probability >= 0.5
    ).astype(
        np.uint8
    )

    target = (
        mask > 0.5
    ).astype(
        np.uint8
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    dice, iou = calculate_metrics(
        prediction,
        target
    )

    predicted_pixels = int(
        prediction.sum()
    )

    target_pixels = int(
        target.sum()
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()

    print(
        f"[{index + 1}/{len(test_files)}] "
        f"{image_path.name}"
    )

    print(
        f"Ground truth pixels: "
        f"{target_pixels}"
    )

    print(
        f"Predicted pixels: "
        f"{predicted_pixels}"
    )

    print(
        f"Dice: {dice:.4f}"
    )

    print(
        f"IoU: {iou:.4f}"
    )

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    figure = plt.figure(
        figsize=(16, 4)
    )

    # Original CT
    ax1 = figure.add_subplot(
        1,
        4,
        1
    )

    ax1.imshow(
        image_display,
        cmap="gray"
    )

    ax1.set_title(
        "CT Patch"
    )

    ax1.axis("off")

    # Ground truth
    ax2 = figure.add_subplot(
        1,
        4,
        2
    )

    ax2.imshow(
        image_display,
        cmap="gray"
    )

    ax2.imshow(
        target,
        alpha=0.45
    )

    ax2.set_title(
        "Ground Truth"
    )

    ax2.axis("off")

    # Prediction
    ax3 = figure.add_subplot(
        1,
        4,
        3
    )

    ax3.imshow(
        image_display,
        cmap="gray"
    )

    ax3.imshow(
        prediction,
        alpha=0.45
    )

    ax3.set_title(
        "Prediction"
    )

    ax3.axis("off")

    # Overlay
    ax4 = figure.add_subplot(
        1,
        4,
        4
    )

    ax4.imshow(
        image_display,
        cmap="gray"
    )

    # Ground truth outline
    ax4.contour(
        target,
        levels=[0.5],
        linewidths=1.5
    )

    # Prediction outline
    if prediction.sum() > 0:

        ax4.contour(
            prediction,
            levels=[0.5],
            linewidths=1.5,
            linestyles="--"
        )

    ax4.set_title(
        f"Overlay\n"
        f"Dice={dice:.3f}"
    )

    ax4.axis("off")

    figure.suptitle(
        image_path.name
    )

    figure.tight_layout()

    output_file = (
        OUTPUT_DIR /
        f"prediction_{index + 1:02d}.png"
    )

    figure.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(
        figure
    )

    all_results.append(
        {
            "file":
                image_path.name,

            "dice":
                dice,

            "iou":
                iou,

            "target_pixels":
                target_pixels,

            "predicted_pixels":
                predicted_pixels
        }
    )


# ============================================================
# Summary
# ============================================================

if all_results:

    mean_dice = np.mean(
        [
            x["dice"]
            for x in all_results
        ]
    )

    mean_iou = np.mean(
        [
            x["iou"]
            for x in all_results
        ]
    )

    print()

    print("=" * 60)

    print(
        "INSPECTION SUMMARY"
    )

    print("=" * 60)

    print(
        f"Patches inspected: "
        f"{len(all_results)}"
    )

    print(
        f"Mean Dice: "
        f"{mean_dice:.4f}"
    )

    print(
        f"Mean IoU: "
        f"{mean_iou:.4f}"
    )

    print()

    print(
        "Images saved to:"
    )

    print(
        OUTPUT_DIR
    )

    print("=" * 60)

print()

print(
    "CT prediction inspection completed."
)