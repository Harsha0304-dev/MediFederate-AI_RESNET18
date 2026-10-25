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

RESULTS_DIR = (
    PROJECT_ROOT /
    "results"
)

OUTPUT_FILE = (
    RESULTS_DIR /
    "ct_threshold_results.csv"
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# Thresholds to test
THRESHOLDS = [
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90
]


# ============================================================
# Header
# ============================================================

print("=" * 60)
print("MediFederate AI")
print("CT Segmentation Threshold Evaluation")
print("=" * 60)

print(
    f"Device: {DEVICE}"
)

print(
    f"Model: {MODEL_PATH}"
)

print(
    f"Test patient: LIDC-IDRI-0010"
)

print(
    f"Test patches: "
    f"{len(list(TEST_IMAGE_DIR.glob('*.npy')))}"
)

print("=" * 60)


# ============================================================
# Check model
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
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

if (
    isinstance(checkpoint, dict)
    and
    "model_state_dict" in checkpoint
):

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
# Load test files
# ============================================================

test_files = sorted(
    TEST_IMAGE_DIR.glob("*.npy")
)

if len(test_files) == 0:

    raise RuntimeError(
        "No test patches found."
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
        2.0 * intersection +
        1e-8
    ) / (
        prediction_sum +
        target_sum +
        1e-8
    )

    iou = (
        intersection +
        1e-8
    ) / (
        union +
        1e-8
    )

    return dice, iou


# ============================================================
# Store predictions
# ============================================================

predictions = []

targets = []

file_names = []

print()
print(
    "Generating model probabilities..."
)

# ============================================================
# Inference
# ============================================================

with torch.no_grad():

    for index, image_path in enumerate(
        test_files
    ):

        mask_path = (
            TEST_MASK_DIR /
            image_path.name
        )

        if not mask_path.exists():

            print(
                f"Skipping missing mask: "
                f"{image_path.name}"
            )

            continue

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

        # ----------------------------------------------------
        # Normalize CT
        # ----------------------------------------------------

        image = np.clip(
            image,
            -1000,
            400
        )

        image = (
            image + 1000.0
        ) / 1400.0

        # ----------------------------------------------------
        # Tensor
        # ----------------------------------------------------

        image_tensor = torch.from_numpy(
            image
        ).float()

        image_tensor = (
            image_tensor
            .unsqueeze(0)
            .unsqueeze(0)
            .to(DEVICE)
        )

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        logits = model(
            image_tensor
        )

        probability = torch.sigmoid(
            logits
        )

        probability = (
            probability[0, 0]
            .cpu()
            .numpy()
        )

        predictions.append(
            probability
        )

        targets.append(
            mask
        )

        file_names.append(
            image_path.name
        )

print(
    f"Generated predictions for "
    f"{len(predictions)} patches."
)


# ============================================================
# Evaluate thresholds
# ============================================================

all_results = []

print()
print("=" * 60)
print("THRESHOLD RESULTS")
print("=" * 60)

print(
    f"{'Threshold':<12}"
    f"{'Mean Dice':<15}"
    f"{'Mean IoU':<15}"
    f"{'Pred Pixels':<15}"
)

print("-" * 60)


for threshold in THRESHOLDS:

    dice_scores = []

    iou_scores = []

    predicted_pixels = []

    target_pixels = []

    for probability, target in zip(
        predictions,
        targets
    ):

        prediction = (
            probability >= threshold
        ).astype(
            np.uint8
        )

        dice, iou = calculate_metrics(
            prediction,
            target
        )

        dice_scores.append(
            dice
        )

        iou_scores.append(
            iou
        )

        predicted_pixels.append(
            int(prediction.sum())
        )

        target_pixels.append(
            int(target.sum())
        )

    mean_dice = float(
        np.mean(
            dice_scores
        )
    )

    mean_iou = float(
        np.mean(
            iou_scores
        )
    )

    mean_predicted_pixels = float(
        np.mean(
            predicted_pixels
        )
    )

    mean_target_pixels = float(
        np.mean(
            target_pixels
        )
    )

    print(
        f"{threshold:<12.2f}"
        f"{mean_dice:<15.4f}"
        f"{mean_iou:<15.4f}"
        f"{mean_predicted_pixels:<15.1f}"
    )

    all_results.append(
        {
            "threshold":
                threshold,

            "mean_dice":
                mean_dice,

            "mean_iou":
                mean_iou,

            "mean_predicted_pixels":
                mean_predicted_pixels,

            "mean_target_pixels":
                mean_target_pixels
        }
    )


# ============================================================
# Find best threshold
# ============================================================

best_result = max(
    all_results,
    key=lambda x: x["mean_dice"]
)

print()
print("=" * 60)

print(
    "BEST THRESHOLD"
)

print("=" * 60)

print(
    f"Threshold: "
    f"{best_result['threshold']:.2f}"
)

print(
    f"Mean Dice: "
    f"{best_result['mean_dice']:.4f}"
)

print(
    f"Mean IoU: "
    f"{best_result['mean_iou']:.4f}"
)

print(
    f"Mean predicted pixels: "
    f"{best_result['mean_predicted_pixels']:.1f}"
)

print(
    f"Mean ground-truth pixels: "
    f"{best_result['mean_target_pixels']:.1f}"
)


# ============================================================
# Save CSV
# ============================================================

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "threshold,"
        "mean_dice,"
        "mean_iou,"
        "mean_predicted_pixels,"
        "mean_target_pixels\n"
    )

    for result in all_results:

        file.write(
            f"{result['threshold']},"
            f"{result['mean_dice']:.6f},"
            f"{result['mean_iou']:.6f},"
            f"{result['mean_predicted_pixels']:.2f},"
            f"{result['mean_target_pixels']:.2f}\n"
        )


print()
print(
    "Results saved to:"
)

print(
    OUTPUT_FILE
)

print("=" * 60)
