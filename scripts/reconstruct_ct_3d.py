from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import pydicom
from scipy import ndimage


# ============================================================
# MediFederate AI
# 3D CT Measurement from Existing Test Predictions
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PATIENT_ID = "LIDC-IDRI-0010"

PATCH_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ct_patches"
    / "test"
)

PREDICTION_DIR = (
    PROJECT_ROOT
    / "results"
    / "ct_measurements"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "ct_3d"
    / PATIENT_ID
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Current model/post-processing settings
THRESHOLD = 0.80
MIN_COMPONENT_PIXELS = 20

# Your CT dataset location
CT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ct"
    / "lidc_idri"
    / PATIENT_ID
)


# ============================================================
# FIND SLICE NUMBER
# ============================================================

def get_slice_number(filename):

    match = re.search(
        r"_slice_(\d+)_",
        filename
    )

    if match:
        return int(match.group(1))

    return None


# ============================================================
# LOAD ACTUAL DICOM SPACING
# ============================================================

def get_dicom_spacing():

    print()
    print("=" * 70)
    print("Reading DICOM spacing")
    print("=" * 70)

    dicom_files = []

    for path in CT_DIR.rglob("*.dcm"):

        try:

            ds = pydicom.dcmread(
                str(path),
                stop_before_pixels=True
            )

            if getattr(ds, "Modality", "") == "CT":
                dicom_files.append(ds)

        except Exception:
            continue

    if not dicom_files:

        raise RuntimeError(
            f"No CT DICOM files found in:\n{CT_DIR}"
        )

    # --------------------------------------------------------
    # Sort by ImagePositionPatient
    # --------------------------------------------------------

    def get_z(ds):

        if hasattr(ds, "ImagePositionPatient"):
            return float(
                ds.ImagePositionPatient[2]
            )

        return float(
            getattr(ds, "SliceLocation", 0)
        )

    dicom_files.sort(key=get_z)

    first = dicom_files[0]

    # --------------------------------------------------------
    # In-plane spacing
    # --------------------------------------------------------

    pixel_spacing = first.PixelSpacing

    spacing_y = float(
        pixel_spacing[0]
    )

    spacing_x = float(
        pixel_spacing[1]
    )

    # --------------------------------------------------------
    # Slice spacing
    # --------------------------------------------------------

    z_positions = []

    for ds in dicom_files:

        if hasattr(
            ds,
            "ImagePositionPatient"
        ):

            z_positions.append(
                float(
                    ds.ImagePositionPatient[2]
                )
            )

    if len(z_positions) > 1:

        differences = np.diff(
            z_positions
        )

        spacing_z = float(
            np.median(
                np.abs(differences)
            )
        )

    else:

        spacing_z = float(
            getattr(
                first,
                "SliceThickness",
                1.0
            )
        )

    print(
        f"CT DICOM slices found: "
        f"{len(dicom_files)}"
    )

    print(
        f"Pixel spacing X: "
        f"{spacing_x:.6f} mm"
    )

    print(
        f"Pixel spacing Y: "
        f"{spacing_y:.6f} mm"
    )

    print(
        f"Slice spacing Z: "
        f"{spacing_z:.6f} mm"
    )

    return (
        spacing_z,
        spacing_y,
        spacing_x
    )


# ============================================================
# FIND PREDICTION FILES
# ============================================================

def find_prediction_files():

    print()
    print("=" * 70)
    print("Finding CT prediction files")
    print("=" * 70)

    # Look for numpy predictions first
    candidates = list(
        PREDICTION_DIR.rglob("*.npy")
    )

    candidates = [
        p for p in candidates
        if "prediction" in p.name.lower()
        or "pred" in p.name.lower()
    ]

    if candidates:

        print(
            f"Prediction files found: "
            f"{len(candidates)}"
        )

        return sorted(
            candidates,
            key=lambda p: (
                get_slice_number(p.name)
                if get_slice_number(p.name)
                is not None
                else 999999
            )
        )

    return []


# ============================================================
# FALLBACK:
# RE-CALCULATE PREDICTIONS FROM PATCHES
# ============================================================

def create_predictions_from_masks():

    """
    If probability prediction files were not saved,
    use the previously generated measurement information
    and corresponding masks.

    This function is intentionally conservative.
    It does NOT invent missing spatial information.
    """

    mask_dir = PATCH_DIR / "masks"

    files = sorted(
        mask_dir.glob("*.npy"),
        key=lambda p: (
            get_slice_number(p.name)
            if get_slice_number(p.name)
            is not None
            else 999999
        )
    )

    print(
        f"Ground-truth mask patches found: "
        f"{len(files)}"
    )

    return files


# ============================================================
# LOAD PATCH MASK
# ============================================================

def load_mask(path):

    array = np.load(
        path
    )

    array = np.squeeze(
        array
    )

    if array.ndim != 2:

        raise ValueError(
            f"Unexpected mask shape "
            f"{array.shape}: {path}"
        )

    return (
        array > 0.5
    ).astype(np.uint8)


# ============================================================
# GROUP CONSECUTIVE SLICES
# ============================================================

def group_slices(slice_numbers):

    if not slice_numbers:
        return []

    groups = []

    current = [
        slice_numbers[0]
    ]

    for number in slice_numbers[1:]:

        previous = current[-1]

        if number == previous + 1:

            current.append(number)

        else:

            groups.append(current)

            current = [number]

    groups.append(current)

    return groups


# ============================================================
# MEASURE ONE 3D GROUP
# ============================================================

def measure_group(
    group_id,
    slice_numbers,
    masks,
    spacing
):

    spacing_z, spacing_y, spacing_x = spacing

    # --------------------------------------------------------
    # Stack masks
    # --------------------------------------------------------

    volume_mask = np.stack(
        masks,
        axis=0
    )

    # --------------------------------------------------------
    # Remove tiny 2D components
    # --------------------------------------------------------

    cleaned = np.zeros_like(
        volume_mask,
        dtype=np.uint8
    )

    structure_2d = (
        ndimage.generate_binary_structure(
            2,
            2
        )
    )

    for z in range(
        volume_mask.shape[0]
    ):

        labeled, count = ndimage.label(
            volume_mask[z],
            structure=structure_2d
        )

        if count == 0:
            continue

        sizes = np.bincount(
            labeled.ravel()
        )

        for component_id in range(
            1,
            count + 1
        ):

            if (
                sizes[component_id]
                >= MIN_COMPONENT_PIXELS
            ):

                cleaned[z][
                    labeled == component_id
                ] = 1

    # --------------------------------------------------------
    # 3D connected components
    # --------------------------------------------------------

    structure_3d = (
        ndimage.generate_binary_structure(
            3,
            2
        )
    )

    labeled_3d, count_3d = ndimage.label(
        cleaned,
        structure=structure_3d
    )

    if count_3d == 0:
        return []

    measurements = []

    voxel_volume_mm3 = (
        spacing_z
        * spacing_y
        * spacing_x
    )

    for component_id in range(
        1,
        count_3d + 1
    ):

        coordinates = np.where(
            labeled_3d == component_id
        )

        voxel_count = len(
            coordinates[0]
        )

        if voxel_count == 0:
            continue

        z_min = int(
            coordinates[0].min()
        )

        z_max = int(
            coordinates[0].max()
        )

        y_min = int(
            coordinates[1].min()
        )

        y_max = int(
            coordinates[1].max()
        )

        x_min = int(
            coordinates[2].min()
        )

        x_max = int(
            coordinates[2].max()
        )

        depth_voxels = (
            z_max - z_min + 1
        )

        height_voxels = (
            y_max - y_min + 1
        )

        width_voxels = (
            x_max - x_min + 1
        )

        depth_mm = (
            depth_voxels
            * spacing_z
        )

        height_mm = (
            height_voxels
            * spacing_y
        )

        width_mm = (
            width_voxels
            * spacing_x
        )

        volume_mm3 = (
            voxel_count
            * voxel_volume_mm3
        )

        volume_cm3 = (
            volume_mm3 / 1000.0
        )

        equivalent_diameter = (
            (
                6.0
                * volume_mm3
                / np.pi
            )
            ** (1.0 / 3.0)
        )

        measurements.append({

            "group_id":
                group_id,

            "component_id":
                int(component_id),

            "start_slice":
                int(slice_numbers[0]),

            "end_slice":
                int(slice_numbers[-1]),

            "slice_count":
                int(len(slice_numbers)),

            "voxel_count":
                int(voxel_count),

            "width_mm":
                round(
                    float(width_mm),
                    3
                ),

            "height_mm":
                round(
                    float(height_mm),
                    3
                ),

            "depth_mm":
                round(
                    float(depth_mm),
                    3
                ),

            "volume_mm3":
                round(
                    float(volume_mm3),
                    3
                ),

            "volume_cm3":
                round(
                    float(volume_cm3),
                    4
                ),

            "equivalent_diameter_mm":
                round(
                    float(
                        equivalent_diameter
                    ),
                    3
                ),

            "bbox_x_min":
                x_min,

            "bbox_x_max":
                x_max,

            "bbox_y_min":
                y_min,

            "bbox_y_max":
                y_max,

            "bbox_z_min":
                z_min,

            "bbox_z_max":
                z_max
        })

    return measurements


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("MediFederate AI")
    print("3D CT Nodule Measurement")
    print("=" * 70)

    print(
        f"Patient: {PATIENT_ID}"
    )

    print(
        f"Patch directory:\n{PATCH_DIR}"
    )

    print(
        f"Threshold: {THRESHOLD}"
    )

    print(
        f"Minimum component: "
        f"{MIN_COMPONENT_PIXELS} pixels"
    )

    # --------------------------------------------------------
    # Check directories
    # --------------------------------------------------------

    if not PATCH_DIR.exists():

        raise RuntimeError(
            f"Patch directory not found:\n"
            f"{PATCH_DIR}"
        )

    if not CT_DIR.exists():

        raise RuntimeError(
            f"CT directory not found:\n"
            f"{CT_DIR}"
        )

    # --------------------------------------------------------
    # Get actual DICOM spacing
    # --------------------------------------------------------

    spacing = get_dicom_spacing()

    # --------------------------------------------------------
    # Load the 14 test patch masks
    #
    # IMPORTANT:
    # These are the corresponding ground-truth masks.
    #
    # We use them only to validate the 3D grouping pipeline.
    # We do NOT call them model predictions.
    # --------------------------------------------------------

    mask_dir = PATCH_DIR / "masks"

    mask_files = sorted(
        mask_dir.glob("*.npy"),
        key=lambda p: (
            get_slice_number(p.name)
            if get_slice_number(p.name)
            is not None
            else 999999
        )
    )

    if not mask_files:

        raise RuntimeError(
            "No mask patches found."
        )

    print()
    print(
        f"Test mask patches: "
        f"{len(mask_files)}"
    )

    # --------------------------------------------------------
    # Extract slice numbers
    # --------------------------------------------------------

    slice_numbers = []

    masks = []

    for path in mask_files:

        slice_number = (
            get_slice_number(
                path.name
            )
        )

        if slice_number is None:

            print(
                f"Skipping unknown file: "
                f"{path.name}"
            )

            continue

        slice_numbers.append(
            slice_number
        )

        masks.append(
            load_mask(path)
        )

    # --------------------------------------------------------
    # Display slice list
    # --------------------------------------------------------

    print()
    print("Slices:")

    print(
        slice_numbers
    )

    # --------------------------------------------------------
    # Group consecutive slices
    # --------------------------------------------------------

    groups = group_slices(
        slice_numbers
    )

    print()
    print("=" * 70)
    print("Slice Groups")
    print("=" * 70)

    for index, group in enumerate(
        groups,
        start=1
    ):

        print(
            f"Group {index}: "
            f"{group[0]} -> "
            f"{group[-1]} "
            f"({len(group)} slices)"
        )

    # --------------------------------------------------------
    # Build lookup
    # --------------------------------------------------------

    mask_lookup = {
        number: mask
        for number, mask in zip(
            slice_numbers,
            masks
        )
    }

    all_measurements = []

    # --------------------------------------------------------
    # Measure each group
    # --------------------------------------------------------

    for group_id, group in enumerate(
        groups,
        start=1
    ):

        group_masks = []

        for slice_number in group:

            group_masks.append(
                mask_lookup[
                    slice_number
                ]
            )

        measurements = measure_group(
            group_id,
            group,
            group_masks,
            spacing
        )

        all_measurements.extend(
            measurements
        )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    csv_path = (
        OUTPUT_DIR
        / "reference_3d_measurements.csv"
    )

    json_path = (
        OUTPUT_DIR
        / "reference_3d_measurements.json"
    )

    summary_path = (
        OUTPUT_DIR
        / "reference_3d_summary.json"
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    if all_measurements:

        dataframe = pd.DataFrame(
            all_measurements
        )

        dataframe.to_csv(
            csv_path,
            index=False
        )

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    total_volume = sum(
        item["volume_cm3"]
        for item in all_measurements
    )

    summary = {

        "patient_id":
            PATIENT_ID,

        "source":
            "CT test ground-truth masks",

        "note":
            "Reference 3D measurement pipeline. "
            "These measurements are derived from "
            "the test masks and are not model prediction "
            "performance metrics.",

        "test_patch_count":
            len(mask_files),

        "slice_numbers":
            slice_numbers,

        "slice_groups":
            groups,

        "spacing_mm_zyx":
            list(spacing),

        "minimum_component_pixels":
            MIN_COMPONENT_PIXELS,

        "predicted_threshold":
            THRESHOLD,

        "number_of_3d_components":
            len(all_measurements),

        "total_volume_cm3":
            round(
                float(total_volume),
                4
            ),

        "measurements":
            all_measurements
    }

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary["measurements"],
            file,
            indent=4
        )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("3D CT MEASUREMENT SUMMARY")
    print("=" * 70)

    print(
        f"Patient: {PATIENT_ID}"
    )

    print(
        f"Test patches: "
        f"{len(mask_files)}"
    )

    print(
        f"Slice groups: "
        f"{len(groups)}"
    )

    print(
        f"3D components: "
        f"{len(all_measurements)}"
    )

    print(
        f"Total reference volume: "
        f"{total_volume:.4f} cm³"
    )

    print()

    for item in all_measurements:

        print(
            f"Group {item['group_id']} "
            f"Component "
            f"{item['component_id']}:"
        )

        print(
            f"  Slices: "
            f"{item['start_slice']} - "
            f"{item['end_slice']}"
        )

        print(
            f"  Width: "
            f"{item['width_mm']:.3f} mm"
        )

        print(
            f"  Height: "
            f"{item['height_mm']:.3f} mm"
        )

        print(
            f"  Depth: "
            f"{item['depth_mm']:.3f} mm"
        )

        print(
            f"  Volume: "
            f"{item['volume_cm3']:.4f} cm³"
        )

        print(
            f"  Equivalent diameter: "
            f"{item['equivalent_diameter_mm']:.3f} mm"
        )

        print()

    print(
        f"CSV:\n{csv_path}"
    )

    print(
        f"JSON:\n{json_path}"
    )

    print(
        f"Summary:\n{summary_path}"
    )

    print()
    print("=" * 70)
    print("3D CT MEASUREMENT COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()