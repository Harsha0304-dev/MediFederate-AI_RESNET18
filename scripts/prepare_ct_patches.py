from pathlib import Path
import random

import nibabel as nib
import numpy as np


# ============================================================
# MediFederate AI
# CT Nodule-Centered Patch Preparation
# ============================================================

# Project CT preprocessing directory
CT_ROOT = Path(
    "data/processed/ct"
)

# Output directory
OUTPUT_ROOT = Path(
    "data/processed/ct_patches"
)

# ------------------------------------------------------------
# IMPORTANT:
# We now use 256x256 patches instead of 128x128.
# The larger patch provides more surrounding lung context.
# ------------------------------------------------------------

PATCH_SIZE = 256

SEED = 42


# ============================================================
# Patient-level split
# ============================================================

# Training patients
TRAIN_PATIENTS = [
    "LIDC-IDRI-0001",
    "LIDC-IDRI-0002",
    "LIDC-IDRI-0003",
    "LIDC-IDRI-0004",
    "LIDC-IDRI-0005",
    "LIDC-IDRI-0006",
    "LIDC-IDRI-0007",
]

# Validation patients
VAL_PATIENTS = [
    "LIDC-IDRI-0008",
    "LIDC-IDRI-0009",
]

# Test patient
# IMPORTANT:
# This patient is never used during training or validation.
TEST_PATIENTS = [
    "LIDC-IDRI-0010",
]


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)

np.random.seed(SEED)


# ============================================================
# Create directories
# ============================================================

for split in [
    "train",
    "val",
    "test"
]:

    image_dir = (
        OUTPUT_ROOT /
        split /
        "images"
    )

    mask_dir = (
        OUTPUT_ROOT /
        split /
        "masks"
    )

    image_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    mask_dir.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# Connected Components
# ============================================================

def connected_components(mask):
    """
    Find connected foreground regions
    in a 2D binary mask.

    Returns:
        List of tuples:
        (
            y_min,
            y_max,
            x_min,
            x_max,
            pixel_count
        )
    """

    mask = mask.astype(bool)

    height, width = mask.shape

    visited = np.zeros(
        mask.shape,
        dtype=bool
    )

    components = []

    # 8-connected neighborhood
    neighbors = [
        (-1, -1),
        (-1, 0),
        (-1, 1),

        (0, -1),
        (0, 1),

        (1, -1),
        (1, 0),
        (1, 1),
    ]

    for y in range(height):

        for x in range(width):

            # Not foreground
            if not mask[y, x]:
                continue

            # Already processed
            if visited[y, x]:
                continue

            # Start new component
            stack = [
                (y, x)
            ]

            visited[y, x] = True

            pixels = []

            while stack:

                current_y, current_x = (
                    stack.pop()
                )

                pixels.append(
                    (
                        current_y,
                        current_x
                    )
                )

                for dy, dx in neighbors:

                    ny = (
                        current_y +
                        dy
                    )

                    nx = (
                        current_x +
                        dx
                    )

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
                            (
                                ny,
                                nx
                            )
                        )

            # Ignore extremely tiny regions
            if len(pixels) < 5:
                continue

            ys = [
                pixel[0]
                for pixel in pixels
            ]

            xs = [
                pixel[1]
                for pixel in pixels
            ]

            y_min = min(ys)
            y_max = max(ys)

            x_min = min(xs)
            x_max = max(xs)

            pixel_count = len(
                pixels
            )

            components.append(
                (
                    y_min,
                    y_max,
                    x_min,
                    x_max,
                    pixel_count
                )
            )

    return components


# ============================================================
# Crop Patch
# ============================================================

def crop_patch(
    image,
    mask,
    center_y,
    center_x
):
    """
    Extract a PATCH_SIZE x PATCH_SIZE
    patch centered around the nodule.
    """

    half = PATCH_SIZE // 2

    height, width = image.shape

    # Initial crop coordinates
    start_y = (
        center_y -
        half
    )

    start_x = (
        center_x -
        half
    )

    # Keep crop inside image
    start_y = max(
        0,
        min(
            start_y,
            height - PATCH_SIZE
        )
    )

    start_x = max(
        0,
        min(
            start_x,
            width - PATCH_SIZE
        )
    )

    end_y = (
        start_y +
        PATCH_SIZE
    )

    end_x = (
        start_x +
        PATCH_SIZE
    )

    # Extract image
    image_patch = image[
        start_y:end_y,
        start_x:end_x
    ]

    # Extract mask
    mask_patch = mask[
        start_y:end_y,
        start_x:end_x
    ]

    # --------------------------------------------------------
    # Padding
    # --------------------------------------------------------

    if (
        image_patch.shape
        !=
        (
            PATCH_SIZE,
            PATCH_SIZE
        )
    ):

        padded_image = np.zeros(
            (
                PATCH_SIZE,
                PATCH_SIZE
            ),
            dtype=np.float32
        )

        padded_mask = np.zeros(
            (
                PATCH_SIZE,
                PATCH_SIZE
            ),
            dtype=np.uint8
        )

        patch_height, patch_width = (
            image_patch.shape
        )

        padded_image[
            :patch_height,
            :patch_width
        ] = image_patch

        padded_mask[
            :patch_height,
            :patch_width
        ] = mask_patch

        image_patch = (
            padded_image
        )

        mask_patch = (
            padded_mask
        )

    return (
        image_patch,
        mask_patch
    )


# ============================================================
# Process One Patient
# ============================================================

def process_patient(
    patient_id,
    split
):

    patient_dir = (
        CT_ROOT /
        patient_id
    )

    # CT volume
    ct_file = (
        patient_dir /
        "ct_hu.nii.gz"
    )

    # Reference annotation mask
    mask_file = (
        patient_dir /
        "reference_mask.nii.gz"
    )

    # --------------------------------------------------------
    # Check CT
    # --------------------------------------------------------

    if not ct_file.exists():

        print(
            f"{patient_id}: "
            f"CT volume missing"
        )

        return 0

    # --------------------------------------------------------
    # Check mask
    # --------------------------------------------------------

    if not mask_file.exists():

        print(
            f"{patient_id}: "
            f"reference mask missing"
        )

        return 0

    # --------------------------------------------------------
    # Load CT
    # --------------------------------------------------------

    ct = nib.load(
        str(ct_file)
    ).get_fdata(
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Load mask
    # --------------------------------------------------------

    mask = nib.load(
        str(mask_file)
    ).get_fdata()

    mask = (
        mask > 0
    ).astype(
        np.uint8
    )

    # --------------------------------------------------------
    # Verify dimensions
    # --------------------------------------------------------

    if ct.shape != mask.shape:

        print(
            f"{patient_id}: "
            f"CT/mask shape mismatch"
        )

        print(
            f"CT shape: "
            f"{ct.shape}"
        )

        print(
            f"Mask shape: "
            f"{mask.shape}"
        )

        return 0

    patch_count = 0

    positive_slices = 0

    # ========================================================
    # Process every CT slice
    # ========================================================

    for slice_idx in range(
        ct.shape[0]
    ):

        image = ct[
            slice_idx
        ]

        target = mask[
            slice_idx
        ]

        # ----------------------------------------------------
        # Skip slices without annotation
        # ----------------------------------------------------

        if not np.any(target):

            continue

        positive_slices += 1

        # ----------------------------------------------------
        # Find connected components
        # ----------------------------------------------------

        components = (
            connected_components(
                target
            )
        )

        # ----------------------------------------------------
        # Create one patch per component
        # ----------------------------------------------------

        for component_idx, component in enumerate(
            components
        ):

            (
                y_min,
                y_max,
                x_min,
                x_max,
                pixel_count
            ) = component

            # Center of bounding box
            center_y = (
                y_min +
                y_max
            ) // 2

            center_x = (
                x_min +
                x_max
            ) // 2

            # ------------------------------------------------
            # Extract 256x256 patch
            # ------------------------------------------------

            image_patch, mask_patch = (
                crop_patch(
                    image,
                    target,
                    center_y,
                    center_x
                )
            )

            # ------------------------------------------------
            # Filename
            # ------------------------------------------------

            filename = (
                f"{patient_id}_"
                f"slice_{slice_idx:04d}_"
                f"component_{component_idx:02d}.npy"
            )

            image_output = (
                OUTPUT_ROOT /
                split /
                "images" /
                filename
            )

            mask_output = (
                OUTPUT_ROOT /
                split /
                "masks" /
                filename
            )

            # ------------------------------------------------
            # Save image
            # ------------------------------------------------

            np.save(
                image_output,
                image_patch.astype(
                    np.float32
                )
            )

            # ------------------------------------------------
            # Save mask
            # ------------------------------------------------

            np.save(
                mask_output,
                mask_patch.astype(
                    np.uint8
                )
            )

            patch_count += 1

    print(
        f"{patient_id} | "
        f"{split} | "
        f"positive slices="
        f"{positive_slices} | "
        f"patches="
        f"{patch_count}"
    )

    return patch_count


# ============================================================
# Process Split
# ============================================================

def process_split(
    patients,
    split
):

    print()

    print(
        "=" * 60
    )

    print(
        f"Preparing "
        f"{split.upper()} patches"
    )

    print(
        "=" * 60
    )

    total_patches = 0

    for patient_id in patients:

        patches = process_patient(
            patient_id,
            split
        )

        total_patches += patches

    print()

    print(
        f"{split.upper()} patches: "
        f"{total_patches}"
    )

    return total_patches


# ============================================================
# Main
# ============================================================

def main():

    print()

    print(
        "=" * 60
    )

    print(
        "MediFederate AI"
    )

    print(
        "CT Nodule-Centered "
        "Patch Preparation"
    )

    print(
        "=" * 60
    )

    print(
        f"Patch size: "
        f"{PATCH_SIZE}x{PATCH_SIZE}"
    )

    print(
        "Train patients: "
        f"{len(TRAIN_PATIENTS)}"
    )

    print(
        "Validation patients: "
        f"{len(VAL_PATIENTS)}"
    )

    print(
        "Test patients: "
        f"{len(TEST_PATIENTS)}"
    )

    print(
        "Test patient: "
        "LIDC-IDRI-0010"
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    train_count = process_split(
        TRAIN_PATIENTS,
        "train"
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    val_count = process_split(
        VAL_PATIENTS,
        "val"
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test_count = process_split(
        TEST_PATIENTS,
        "test"
    )

    # ========================================================
    # Summary
    # ========================================================

    print()

    print(
        "=" * 60
    )

    print(
        "PATCH DATASET COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        f"Training patches: "
        f"{train_count}"
    )

    print(
        f"Validation patches: "
        f"{val_count}"
    )

    print(
        f"Test patches: "
        f"{test_count}"
    )

    print(
        f"Patch size: "
        f"{PATCH_SIZE}x{PATCH_SIZE}"
    )

    print(
        "=" * 60
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()