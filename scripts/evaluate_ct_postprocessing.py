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
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ct_patches"
)

TEST_IMAGE_DIR = (
    PATCH_ROOT
    / "test"
    / "images"
)

TEST_MASK_DIR = (
    PATCH_ROOT
    / "test"
    / "masks"
)

# IMPORTANT:
# Use the 128x128 model because it performed better
# than the 256x256 experiment.
MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "unet_ct_patch_best.pt"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

OUTPUT_FILE = (
    RESULTS_DIR
    / "ct_postprocessing_results.csv"
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# Best threshold discovered previously
THRESHOLD = 0.80

# Minimum connected-component sizes
MIN_COMPONENT_SIZES = [
    0,
    5,
    10,
    20,
    30,
    50,
    75,
    100
]


# ============================================================
# Header
# ============================================================

print("=" * 60)

print(
    "MediFederate AI"
)

print(
    "CT Segmentation Post-Processing Evaluation"
)

print("=" * 60)

print(
    f"Device: {DEVICE}"
)

print(
    f"Model: {MODEL_PATH}"
)

print(
    "Test patient: LIDC-IDRI-0010"
)

print(
    f"Threshold: {THRESHOLD}"
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
# Connected component filtering
# ============================================================

def connected_components(
    binary_mask
):
    """
    Find 8-connected components
    in a 2D binary mask.

    Returns a list of:
        (pixels, coordinates)
    """

    mask = (
        binary_mask.astype(bool)
    )

    height, width = (
        mask.shape
    )

    visited = np.zeros(
        mask.shape,
        dtype=bool
    )

    components = []

    neighbors = [
        (-1, -1),
        (-1, 0),
        (-1, 1),

        (0, -1),
        (0, 1),

        (1, -1),
        (1, 0),
        (1, 1)
    ]

    for y in range(height):

        for x in range(width):

            if not mask[y, x]:
                continue

            if visited[y, x]:
                continue

            stack = [
                (y, x)
            ]

            visited[y, x] = True

            pixels = []

            while stack:

                cy, cx = (
                    stack.pop()
                )

                pixels.append(
                    (cy, cx)
                )

                for dy, dx in neighbors:

                    ny = cy + dy
                    nx = cx + dx

                    if (
                        0 <= ny < height
                        and
                        0 <= nx < width
                        and
                        mask[ny, nx]
                        and
                        not visited[ny, nx]
                    ):

                        visited[ny, nx] = True

                        stack.append(
                            (ny, nx)
                        )

            components.append(
                pixels
            )

    return components


# ============================================================
# Remove small components
# ============================================================

def filter_components(
    prediction,
    minimum_size
):

    # No filtering
    if minimum_size <= 0:

        return prediction.astype(
            np.uint8
        )

    components = (
        connected_components(
            prediction
        )
    )

    filtered = np.zeros_like(
        prediction,
        dtype=np.uint8
    )

    for component in components:

        if len(component) < minimum_size:
            continue

        for y, x in component:

            filtered[y, x] = 1

    return filtered


# ============================================================
# Dice / IoU
# ============================================================

def calculate_metrics(
    prediction,
    target
):

    prediction = (
        prediction.astype(bool)
    )

    target = (
        target.astype(bool)
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

    return (
        float(dice),
        float(iou)
    )


# ============================================================
# Test files
# ============================================================

test_files = sorted(
    TEST_IMAGE_DIR.glob(
        "*.npy"
    )
)

if len(test_files) == 0:

    raise RuntimeError(
        "No test patches found."
    )


# ============================================================
# Generate predictions once
# ============================================================

predictions = []

targets = []

file_names = []

print()

print(
    "Generating predictions..."
)


with torch.no_grad():

    for image_path in test_files:

        mask_path = (
            TEST_MASK_DIR
            / image_path.name
        )

        if not mask_path.exists():

            print(
                f"Skipping missing mask: "
                f"{image_path.name}"
            )

            continue

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = np.load(
            image_path
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # Load ground truth
        # ----------------------------------------------------

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

        tensor = torch.from_numpy(
            image
        ).float()

        tensor = (
            tensor
            .unsqueeze(0)
            .unsqueeze(0)
            .to(DEVICE)
        )

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        logits = model(
            tensor
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
# Evaluate post-processing
# ============================================================

all_results = []

print()

print("=" * 60)

print(
    "POST-PROCESSING RESULTS"
)

print("=" * 60)

print(
    f"{'Min Pixels':<15}"
    f"{'Mean Dice':<15}"
    f"{'Mean IoU':<15}"
    f"{'Mean Pred':<15}"
)

print(
    "-" * 60
)


for minimum_size in (
    MIN_COMPONENT_SIZES
):

    dice_scores = []

    iou_scores = []

    predicted_pixels = []

    target_pixels = []

    # --------------------------------------------------------
    # Process every test patch
    # --------------------------------------------------------

    for probability, target in zip(
        predictions,
        targets
    ):

        # Threshold probability
        binary_prediction = (
            probability >= THRESHOLD
        ).astype(
            np.uint8
        )

        # Remove small components
        filtered_prediction = (
            filter_components(
                binary_prediction,
                minimum_size
            )
        )

        dice, iou = calculate_metrics(
            filtered_prediction,
            target
        )

        dice_scores.append(
            dice
        )

        iou_scores.append(
            iou
        )

        predicted_pixels.append(
            int(
                filtered_prediction.sum()
            )
        )

        target_pixels.append(
            int(
                target.sum()
            )
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

    mean_predicted = float(
        np.mean(
            predicted_pixels
        )
    )

    mean_target = float(
        np.mean(
            target_pixels
        )
    )

    print(
        f"{minimum_size:<15}"
        f"{mean_dice:<15.4f}"
        f"{mean_iou:<15.4f}"
        f"{mean_predicted:<15.1f}"
    )

    all_results.append(
        {
            "minimum_component_size":
                minimum_size,

            "threshold":
                THRESHOLD,

            "mean_dice":
                mean_dice,

            "mean_iou":
                mean_iou,

            "mean_predicted_pixels":
                mean_predicted,

            "mean_target_pixels":
                mean_target
        }
    )


# ============================================================
# Find best configuration
# ============================================================

best_result = max(
    all_results,
    key=lambda x: x["mean_dice"]
)


# ============================================================
# Print best result
# ============================================================

print()

print("=" * 60)

print(
    "BEST POST-PROCESSING CONFIGURATION"
)

print("=" * 60)

print(
    f"Probability threshold: "
    f"{best_result['threshold']:.2f}"
)

print(
    f"Minimum component size: "
    f"{best_result['minimum_component_size']} pixels"
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
        "minimum_component_size,"
        "threshold,"
        "mean_dice,"
        "mean_iou,"
        "mean_predicted_pixels,"
        "mean_target_pixels\n"
    )

    for result in all_results:

        file.write(
            f"{result['minimum_component_size']},"
            f"{result['threshold']:.2f},"
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

print()

print("=" * 60)

print(
    "CT post-processing evaluation completed."
)

print("=" * 60)