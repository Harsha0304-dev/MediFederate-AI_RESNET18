from pathlib import Path
import json

import nibabel as nib
import numpy as np


# ============================================================
# Paths
# ============================================================

CT_ROOT = Path("data/processed/ct")

OUTPUT_ROOT = CT_ROOT


# ============================================================
# Utility: convert NumPy values to JSON-safe Python values
# ============================================================

def json_safe(value):
    """
    Convert NumPy data types into normal Python types
    so they can be written to JSON.
    """

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, dict):
        return {
            key: json_safe(val)
            for key, val in value.items()
        }

    if isinstance(value, list):
        return [
            json_safe(item)
            for item in value
        ]

    return value


# ============================================================
# Process one patient
# ============================================================

def process_patient(patient_dir):

    patient_id = patient_dir.name

    print("\n" + "-" * 60)
    print(f"Processing {patient_id}")
    print("-" * 60)

    # --------------------------------------------------------
    # Locate CT volume
    # --------------------------------------------------------

    ct_file = patient_dir / "ct_hu.nii.gz"

    if not ct_file.exists():

        print(
            f"{patient_id}: CT volume not found"
        )

        return False

    # --------------------------------------------------------
    # Locate annotation masks
    # --------------------------------------------------------

    mask_files = sorted(
        patient_dir.glob(
            "nodule_mask_*.nii.gz"
        )
    )

    if not mask_files:

        print(
            f"{patient_id}: no annotation masks found"
        )

        return False

    print(
        f"Annotation masks found: "
        f"{len(mask_files)}"
    )

    # --------------------------------------------------------
    # Load CT
    # --------------------------------------------------------

    ct_img = nib.load(
        str(ct_file)
    )

    ct_shape = tuple(
        int(x)
        for x in ct_img.shape
    )

    print(
        f"CT shape: {ct_shape}"
    )

    # --------------------------------------------------------
    # Get physical voxel spacing
    # --------------------------------------------------------

    spacing = [
        float(x)
        for x in ct_img.header.get_zooms()[:3]
    ]

    print(
        "Voxel spacing: "
        f"{spacing[0]:.4f}, "
        f"{spacing[1]:.4f}, "
        f"{spacing[2]:.4f} mm"
    )

    # --------------------------------------------------------
    # Create empty reference mask
    # --------------------------------------------------------

    reference_mask = np.zeros(
        ct_shape,
        dtype=np.uint8
    )

    masks_used = 0

    # --------------------------------------------------------
    # Merge all annotation masks
    # --------------------------------------------------------

    for mask_file in mask_files:

        try:

            mask_img = nib.load(
                str(mask_file)
            )

            mask = (
                mask_img.get_fdata() > 0
            )

            # Check shape
            if mask.shape != ct_shape:

                print(
                    f"  SKIPPED {mask_file.name}: "
                    f"shape {mask.shape} != {ct_shape}"
                )

                continue

            # Union operation
            reference_mask |= (
                mask.astype(np.uint8)
            )

            masks_used += 1

            print(
                f"  Added: {mask_file.name}"
            )

        except Exception as e:

            print(
                f"  ERROR reading "
                f"{mask_file.name}: {e}"
            )

    # --------------------------------------------------------
    # Make sure at least one mask was usable
    # --------------------------------------------------------

    if masks_used == 0:

        print(
            f"{patient_id}: "
            "no compatible masks"
        )

        return False

    # --------------------------------------------------------
    # Save reference mask
    # --------------------------------------------------------

    reference_file = (
        patient_dir /
        "reference_mask.nii.gz"
    )

    reference_img = nib.Nifti1Image(
        reference_mask,
        ct_img.affine,
        ct_img.header.copy()
    )

    nib.save(
        reference_img,
        str(reference_file)
    )

    # --------------------------------------------------------
    # Calculate measurements
    # --------------------------------------------------------

    voxel_count = int(
        np.sum(reference_mask > 0)
    )

    voxel_z = float(spacing[0])
    voxel_y = float(spacing[1])
    voxel_x = float(spacing[2])

    voxel_volume_mm3 = (
        voxel_z *
        voxel_y *
        voxel_x
    )

    volume_mm3 = (
        float(voxel_count) *
        voxel_volume_mm3
    )

    volume_cm3 = (
        volume_mm3 / 1000.0
    )

    # --------------------------------------------------------
    # Bounding box
    # --------------------------------------------------------

    coordinates = np.argwhere(
        reference_mask > 0
    )

    if len(coordinates) > 0:

        z_min, y_min, x_min = (
            coordinates.min(axis=0)
        )

        z_max, y_max, x_max = (
            coordinates.max(axis=0)
        )

        depth_mm = (
            int(z_max - z_min + 1) *
            voxel_z
        )

        height_mm = (
            int(y_max - y_min + 1) *
            voxel_y
        )

        width_mm = (
            int(x_max - x_min + 1) *
            voxel_x
        )

        positive_slices = int(
            np.sum(
                np.any(
                    reference_mask > 0,
                    axis=(1, 2)
                )
            )
        )

    else:

        width_mm = 0.0
        height_mm = 0.0
        depth_mm = 0.0
        positive_slices = 0

    # --------------------------------------------------------
    # Print measurements
    # --------------------------------------------------------

    print()
    print(
        f"  Masks used       : "
        f"{masks_used}/{len(mask_files)}"
    )

    print(
        f"  Reference voxels : "
        f"{voxel_count}"
    )

    print(
        f"  Width            : "
        f"{width_mm:.2f} mm"
    )

    print(
        f"  Height           : "
        f"{height_mm:.2f} mm"
    )

    print(
        f"  Depth            : "
        f"{depth_mm:.2f} mm"
    )

    print(
        f"  Positive slices  : "
        f"{positive_slices}"
    )

    print(
        f"  Volume           : "
        f"{volume_cm3:.4f} cm3"
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {

        "patient_id": patient_id,

        "source_masks": int(
            len(mask_files)
        ),

        "masks_used": int(
            masks_used
        ),

        "reference_type":
            "annotation_union",

        "ct_shape": [
            int(x)
            for x in ct_shape
        ],

        "spacing_mm": [
            float(x)
            for x in spacing
        ],

        "positive_slices": int(
            positive_slices
        ),

        "reference_voxels": int(
            voxel_count
        ),

        "width_mm": float(
            width_mm
        ),

        "height_mm": float(
            height_mm
        ),

        "depth_mm": float(
            depth_mm
        ),

        "reference_volume_mm3": float(
            volume_mm3
        ),

        "reference_volume_cm3": float(
            volume_cm3
        )
    }

    # Extra safety conversion
    metadata = json_safe(
        metadata
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata_file = (
        patient_dir /
        "reference_metadata.json"
    )

    with open(
        metadata_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    print()
    print(
        f"  Saved reference mask: "
        f"{reference_file.name}"
    )

    print(
        f"  Saved metadata: "
        f"{metadata_file.name}"
    )

    return True


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)

    print(
        "MediFederate AI"
    )

    print(
        "CT Annotation Reference Mask Generation"
    )

    print("=" * 60)

    if not CT_ROOT.exists():

        print(
            f"ERROR: Directory not found: "
            f"{CT_ROOT}"
        )

        return

    # --------------------------------------------------------
    # Find patients
    # --------------------------------------------------------

    patients = sorted(
        [
            p
            for p in CT_ROOT.glob(
                "LIDC-IDRI-*"
            )
            if p.is_dir()
        ]
    )

    print(
        f"Patients found: "
        f"{len(patients)}"
    )

    if not patients:

        print(
            "No patient directories found."
        )

        return

    # --------------------------------------------------------
    # Process patients
    # --------------------------------------------------------

    successful = 0
    failed = 0

    for patient_dir in patients:

        try:

            result = process_patient(
                patient_dir
            )

            if result:

                successful += 1

            else:

                failed += 1

        except Exception as e:

            failed += 1

            print()
            print(
                f"{patient_dir.name}: "
                f"FAILED"
            )

            print(
                f"Reason: {e}"
            )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 60)

    print(
        "REFERENCE MASK PROCESSING COMPLETE"
    )

    print("=" * 60)

    print(
        f"Successful: "
        f"{successful}/{len(patients)}"
    )

    print(
        f"Failed: "
        f"{failed}/{len(patients)}"
    )

    print("=" * 60)


if __name__ == "__main__":

    main()