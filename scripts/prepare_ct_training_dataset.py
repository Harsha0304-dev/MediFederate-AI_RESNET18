from pathlib import Path
import json
import random

import nibabel as nib
import numpy as np


# ============================================================
# Configuration
# ============================================================

CT_ROOT = Path("data/processed/ct")
OUTPUT_ROOT = Path("data/processed/ct_dataset")

SEED = 42

# Patient-level split
TRAIN_PATIENTS = [
    "LIDC-IDRI-0001",
    "LIDC-IDRI-0002",
    "LIDC-IDRI-0003",
    "LIDC-IDRI-0004",
    "LIDC-IDRI-0005",
    "LIDC-IDRI-0006",
    "LIDC-IDRI-0007",
]

VAL_PATIENTS = [
    "LIDC-IDRI-0008",
    "LIDC-IDRI-0009",
]

TEST_PATIENTS = [
    "LIDC-IDRI-0010",
]


# ============================================================
# Create directories
# ============================================================

for split in ["train", "val", "test"]:

    (OUTPUT_ROOT / split / "images").mkdir(
        parents=True,
        exist_ok=True
    )

    (OUTPUT_ROOT / split / "masks").mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# CT intensity normalization
# ============================================================

def normalize_ct(ct):

    # Lung window approximately
    lower = -1000.0
    upper = 400.0

    ct = np.clip(
        ct,
        lower,
        upper
    )

    ct = (
        (ct - lower) /
        (upper - lower)
    )

    return ct.astype(np.float32)


# ============================================================
# Process one patient
# ============================================================

def process_patient(patient_id, split):

    patient_dir = (
        CT_ROOT / patient_id
    )

    ct_file = (
        patient_dir /
        "ct_hu.nii.gz"
    )

    mask_file = (
        patient_dir /
        "reference_mask.nii.gz"
    )

    if not ct_file.exists():

        print(
            f"{patient_id}: CT missing"
        )

        return 0, 0

    if not mask_file.exists():

        print(
            f"{patient_id}: mask missing"
        )

        return 0, 0

    # --------------------------------------------------------
    # Load CT and mask
    # --------------------------------------------------------

    ct_img = nib.load(
        str(ct_file)
    )

    mask_img = nib.load(
        str(mask_file)
    )

    ct = ct_img.get_fdata(
        dtype=np.float32
    )

    mask = (
        mask_img.get_fdata() > 0
    ).astype(np.uint8)

    if ct.shape != mask.shape:

        raise ValueError(
            f"{patient_id}: "
            f"CT shape {ct.shape} != "
            f"mask shape {mask.shape}"
        )

    # --------------------------------------------------------
    # Normalize CT
    # --------------------------------------------------------

    ct = normalize_ct(ct)

    positive_slices = 0
    negative_slices = []

    # --------------------------------------------------------
    # Save slices
    # --------------------------------------------------------

    for slice_idx in range(ct.shape[0]):

        image_slice = ct[slice_idx]
        mask_slice = mask[slice_idx]

        has_nodule = (
            np.sum(mask_slice) > 0
        )

        if has_nodule:

            positive_slices += 1

        else:

            negative_slices.append(
                slice_idx
            )

    # --------------------------------------------------------
    # Keep negative slices but limit them
    #
    # This prevents thousands of empty lung slices from
    # dominating the segmentation training.
    # --------------------------------------------------------

    random.seed(
        SEED + hash(patient_id) % 10000
    )

    random.shuffle(
        negative_slices
    )

    # Keep at most 3 negative slices per positive slice
    max_negative = max(
        3 * positive_slices,
        20
    )

    selected_negative = (
        negative_slices[:max_negative]
    )

    selected_slices = sorted(
        list(
            range(ct.shape[0])
        )
        if positive_slices == 0
        else
        [
            i
            for i in range(ct.shape[0])
            if np.sum(mask[i]) > 0
        ] + selected_negative
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

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

    for slice_idx in selected_slices:

        image = ct[slice_idx]
        target = mask[slice_idx]

        filename = (
            f"{patient_id}_"
            f"slice_{slice_idx:04d}.npy"
        )

        np.save(
            image_dir / filename,
            image
        )

        np.save(
            mask_dir / filename,
            target
        )

    total_saved = len(
        selected_slices
    )

    print(
        f"{patient_id} | "
        f"{split} | "
        f"CT slices={ct.shape[0]} | "
        f"positive={positive_slices} | "
        f"negative saved={len(selected_negative)} | "
        f"total saved={total_saved}"
    )

    return (
        positive_slices,
        total_saved
    )


# ============================================================
# Process split
# ============================================================

def process_split(patient_list, split):

    total_positive = 0
    total_saved = 0

    print()
    print("=" * 60)
    print(
        f"Preparing {split.upper()} dataset"
    )
    print("=" * 60)

    for patient_id in patient_list:

        positive, saved = (
            process_patient(
                patient_id,
                split
            )
        )

        total_positive += positive
        total_saved += saved

    print()
    print(
        f"{split.upper()} summary:"
    )

    print(
        f"Positive slices: "
        f"{total_positive}"
    )

    print(
        f"Total slices saved: "
        f"{total_saved}"
    )

    return {
        "patients": patient_list,
        "positive_slices": total_positive,
        "total_slices": total_saved
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print(
        "MediFederate AI"
    )
    print(
        "Patient-Level CT Dataset Preparation"
    )
    print("=" * 60)

    print()
    print(
        "TRAIN:"
    )
    print(
        ", ".join(TRAIN_PATIENTS)
    )

    print()
    print(
        "VALIDATION:"
    )
    print(
        ", ".join(VAL_PATIENTS)
    )

    print()
    print(
        "TEST:"
    )
    print(
        ", ".join(TEST_PATIENTS)
    )

    # --------------------------------------------------------
    # Prepare datasets
    # --------------------------------------------------------

    train_stats = process_split(
        TRAIN_PATIENTS,
        "train"
    )

    val_stats = process_split(
        VAL_PATIENTS,
        "val"
    )

    test_stats = process_split(
        TEST_PATIENTS,
        "test"
    )

    # --------------------------------------------------------
    # Save split information
    # --------------------------------------------------------

    split_info = {

        "seed": SEED,

        "train": train_stats,

        "validation": val_stats,

        "test": test_stats,

        "split_strategy":
            "patient_level",

        "normalization":
            "HU clipped to [-1000, 400] "
            "and scaled to [0, 1]",

        "negative_slice_strategy":
            "maximum 3 negative slices "
            "per positive slice",

        "target":
            "reference_mask.nii.gz"
    }

    with open(
        OUTPUT_ROOT /
        "dataset_split.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            split_info,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print(
        "CT DATASET PREPARATION COMPLETE"
    )
    print("=" * 60)

    print(
        f"Train patients: "
        f"{len(TRAIN_PATIENTS)}"
    )

    print(
        f"Validation patients: "
        f"{len(VAL_PATIENTS)}"
    )

    print(
        f"Test patients: "
        f"{len(TEST_PATIENTS)}"
    )

    print()
    print(
        f"Output: {OUTPUT_ROOT}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()