from pathlib import Path
import sys
import csv
import json
import re

import numpy as np
import torch
import pydicom

from src.medifederate.unet import UNet


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


# ============================================================
# CONFIGURATION
# ============================================================

PATIENT_ID = "LIDC-IDRI-0010"

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

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "unet_ct_patch_best.pt"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

OUTPUT_DIR = (
    RESULTS_DIR
    / "ct_measurements"
)

PREDICTION_DIR = (
    RESULTS_DIR
    / "ct_predictions"
    / PATIENT_ID
)

CT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ct"
    / "lidc_idri"
    / PATIENT_ID
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PREDICTION_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "nodule_measurements.csv"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "nodule_measurements.json"
)

PREDICTION_SUMMARY_JSON = (
    PREDICTION_DIR
    / "prediction_summary.json"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# INFERENCE SETTINGS
# ============================================================

THRESHOLD = 0.80

MIN_COMPONENT_SIZE = 20


# ============================================================
# GET ACTUAL DICOM SPACING
# ============================================================

def get_actual_dicom_spacing():

    print()
    print("=" * 60)
    print("Reading actual CT DICOM spacing")
    print("=" * 60)

    dicom_files = []

    for path in CT_DIR.rglob("*.dcm"):

        try:

            ds = pydicom.dcmread(
                str(path),
                stop_before_pixels=True
            )

            if getattr(
                ds,
                "Modality",
                ""
            ) == "CT":

                dicom_files.append(
                    (path, ds)
                )

        except Exception:
            continue

    if not dicom_files:

        print(
            "WARNING: No CT DICOM files found."
        )

        print(
            "Using previous fallback spacing."
        )

        return (
            2.5,
            0.703125,
            0.703125
        )

    # --------------------------------------------------------
    # Sort by physical Z position
    # --------------------------------------------------------

    def get_z(item):

        ds = item[1]

        if hasattr(
            ds,
            "ImagePositionPatient"
        ):

            return float(
                ds.ImagePositionPatient[2]
            )

        return float(
            getattr(
                ds,
                "SliceLocation",
                0
            )
        )

    dicom_files.sort(
        key=get_z
    )

    first_ds = dicom_files[0][1]

    # --------------------------------------------------------
    # Pixel spacing
    # --------------------------------------------------------

    pixel_spacing = (
        first_ds.PixelSpacing
    )

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

    for _, ds in dicom_files:

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
                first_ds,
                "SliceThickness",
                1.0
            )
        )

    print(
        f"CT DICOM files: "
        f"{len(dicom_files)}"
    )

    print(
        f"X spacing: "
        f"{spacing_x:.6f} mm"
    )

    print(
        f"Y spacing: "
        f"{spacing_y:.6f} mm"
    )

    print(
        f"Z spacing: "
        f"{spacing_z:.6f} mm"
    )

    return (
        spacing_z,
        spacing_y,
        spacing_x
    )


# ============================================================
# ACTUAL SPACING
# ============================================================

(
    SLICE_SPACING_MM,
    PIXEL_SPACING_Y_MM,
    PIXEL_SPACING_X_MM
) = get_actual_dicom_spacing()


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 60)
print("MediFederate AI")
print("CT Nodule Measurement Pipeline")
print("=" * 60)

print(
    f"Device: {DEVICE}"
)

print(
    f"Model: {MODEL_PATH}"
)

print(
    f"Test patient: {PATIENT_ID}"
)

print(
    f"Probability threshold: {THRESHOLD}"
)

print(
    f"Minimum component size: "
    f"{MIN_COMPONENT_SIZE} pixels"
)

print(
    f"Pixel spacing X: "
    f"{PIXEL_SPACING_X_MM:.6f} mm"
)

print(
    f"Pixel spacing Y: "
    f"{PIXEL_SPACING_Y_MM:.6f} mm"
)

print(
    f"Slice spacing Z: "
    f"{SLICE_SPACING_MM:.6f} mm"
)

print("=" * 60)


# ============================================================
# CHECK MODEL
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Model not found:\n"
        f"{MODEL_PATH}"
    )


# ============================================================
# LOAD MODEL
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

elif (
    isinstance(checkpoint, dict)
    and
    "state_dict" in checkpoint
):

    model.load_state_dict(
        checkpoint[
            "state_dict"
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
# CONNECTED COMPONENTS
# ============================================================

def connected_components(
    binary_mask
):

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
                    (
                        cy,
                        cx
                    )
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
                        not visited[
                            ny,
                            nx
                        ]
                    ):

                        visited[
                            ny,
                            nx
                        ] = True

                        stack.append(
                            (
                                ny,
                                nx
                            )
                        )

            components.append(
                pixels
            )

    return components


# ============================================================
# FILTER COMPONENTS
# ============================================================

def filter_components(
    binary_mask,
    minimum_size
):

    components = (
        connected_components(
            binary_mask
        )
    )

    filtered = np.zeros_like(
        binary_mask,
        dtype=np.uint8
    )

    kept_components = []

    for component in components:

        if (
            len(component)
            < minimum_size
        ):

            continue

        for y, x in component:

            filtered[y, x] = 1

        kept_components.append(
            component
        )

    return (
        filtered,
        kept_components
    )


# ============================================================
# EXTRACT SLICE NUMBER
# ============================================================

def extract_slice_number(
    filename
):

    match = re.search(
        r"_slice_(\d+)_",
        filename
    )

    if match:

        return int(
            match.group(1)
        )

    return -1


# ============================================================
# MEASURE COMPONENT
# ============================================================

def measure_component(
    component,
    slice_index
):

    pixels = len(
        component
    )

    ys = np.array(
        [
            p[0]
            for p in component
        ]
    )

    xs = np.array(
        [
            p[1]
            for p in component
        ]
    )

    height_pixels = (
        ys.max()
        - ys.min()
        + 1
    )

    width_pixels = (
        xs.max()
        - xs.min()
        + 1
    )

    width_mm = (
        width_pixels
        * PIXEL_SPACING_X_MM
    )

    height_mm = (
        height_pixels
        * PIXEL_SPACING_Y_MM
    )

    area_mm2 = (
        pixels
        * PIXEL_SPACING_X_MM
        * PIXEL_SPACING_Y_MM
    )

    equivalent_diameter_mm = (
        2.0
        * np.sqrt(
            area_mm2
            / np.pi
        )
    )

    return {

        "slice_index":
            int(slice_index),

        "pixel_count":
            int(pixels),

        "width_pixels":
            int(width_pixels),

        "height_pixels":
            int(height_pixels),

        "width_mm":
            float(width_mm),

        "height_mm":
            float(height_mm),

        "area_mm2":
            float(area_mm2),

        "equivalent_diameter_mm":
            float(
                equivalent_diameter_mm
            )
    }


# ============================================================
# FIND TEST PATCHES
# ============================================================

test_files = sorted(
    TEST_IMAGE_DIR.glob(
        "*.npy"
    ),
    key=lambda p: (
        extract_slice_number(
            p.name
        )
    )
)


if len(test_files) == 0:

    raise RuntimeError(
        "No test patches found."
    )


print()

print(
    f"Test patches found: "
    f"{len(test_files)}"
)


# ============================================================
# PROCESS TEST PATCHES
# ============================================================

all_measurements = []

total_predicted_pixels = 0

slice_indices = []

prediction_records = []

print()

print("=" * 60)

print(
    "PROCESSING CT TEST PATCHES"
)

print("=" * 60)


with torch.no_grad():

    for patch_number, image_path in enumerate(
        test_files,
        start=1
    ):

        # ----------------------------------------------------
        # Load CT patch
        # ----------------------------------------------------

        image = np.load(
            image_path
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # Normalize HU
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
        # U-Net prediction
        # ----------------------------------------------------

        logits = model(
            tensor
        )

        probabilities = (
            torch.sigmoid(
                logits
            )
            [0, 0]
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        # ----------------------------------------------------
        # Threshold
        # ----------------------------------------------------

        binary_prediction = (
            probabilities
            >= THRESHOLD
        ).astype(
            np.uint8
        )

        # ----------------------------------------------------
        # Remove small components
        # ----------------------------------------------------

        (
            filtered_mask,
            components
        ) = filter_components(
            binary_prediction,
            MIN_COMPONENT_SIZE
        )

        predicted_pixels = int(
            filtered_mask.sum()
        )

        total_predicted_pixels += (
            predicted_pixels
        )

        # ----------------------------------------------------
        # Slice number
        # ----------------------------------------------------

        slice_index = (
            extract_slice_number(
                image_path.name
            )
        )

        slice_indices.append(
            slice_index
        )

        # ====================================================
        # SAVE U-NET PREDICTIONS
        # ====================================================

        prediction_name = (
            f"LIDC-IDRI-0010"
            f"_slice_{slice_index:04d}"
        )

        probability_path = (
            PREDICTION_DIR
            / f"{prediction_name}"
            f"_probability.npy"
        )

        mask_path = (
            PREDICTION_DIR
            / f"{prediction_name}"
            f"_mask.npy"
        )

        np.save(
            probability_path,
            probabilities
        )

        np.save(
            mask_path,
            filtered_mask
        )

        # ----------------------------------------------------
        # Save prediction metadata
        # ----------------------------------------------------

        prediction_records.append({

            "patient_id":
                PATIENT_ID,

            "slice_index":
                int(slice_index),

            "source_patch":
                image_path.name,

            "probability_file":
                probability_path.name,

            "mask_file":
                mask_path.name,

            "threshold":
                THRESHOLD,

            "minimum_component_size":
                MIN_COMPONENT_SIZE,

            "predicted_pixels":
                predicted_pixels
        })

        # ----------------------------------------------------
        # Console output
        # ----------------------------------------------------

        print()

        print(
            f"[{patch_number}/"
            f"{len(test_files)}] "
            f"{image_path.name}"
        )

        print(
            f"Predicted pixels: "
            f"{predicted_pixels}"
        )

        print(
            f"Detected components: "
            f"{len(components)}"
        )

        # ----------------------------------------------------
        # Measurements
        # ----------------------------------------------------

        for component_number, component in enumerate(
            components,
            start=1
        ):

            measurement = (
                measure_component(
                    component,
                    slice_index
                )
            )

            measurement[
                "patient_id"
            ] = PATIENT_ID

            measurement[
                "patch_number"
            ] = patch_number

            measurement[
                "component_number"
            ] = component_number

            measurement[
                "source_patch"
            ] = image_path.name

            all_measurements.append(
                measurement
            )

            print(
                f"  Component "
                f"{component_number}: "
                f"width="
                f"{measurement['width_mm']:.2f} mm, "
                f"height="
                f"{measurement['height_mm']:.2f} mm, "
                f"area="
                f"{measurement['area_mm2']:.2f} mm2, "
                f"equivalent diameter="
                f"{measurement['equivalent_diameter_mm']:.2f} mm"
            )


# ============================================================
# SAVE PREDICTION SUMMARY
# ============================================================

prediction_summary = {

    "patient_id":
        PATIENT_ID,

    "model":
        str(MODEL_PATH),

    "test_patch_count":
        len(test_files),

    "slice_indices":
        slice_indices,

    "threshold":
        THRESHOLD,

    "minimum_component_size":
        MIN_COMPONENT_SIZE,

    "pixel_spacing_x_mm":
        PIXEL_SPACING_X_MM,

    "pixel_spacing_y_mm":
        PIXEL_SPACING_Y_MM,

    "slice_spacing_mm":
        SLICE_SPACING_MM,

    "total_predicted_pixels":
        total_predicted_pixels,

    "predictions":
        prediction_records
}

with open(
    PREDICTION_SUMMARY_JSON,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        prediction_summary,
        file,
        indent=4
    )


# ============================================================
# CT MEASUREMENT SUMMARY
# ============================================================

print()

print("=" * 60)

print(
    "CT MEASUREMENT SUMMARY"
)

print("=" * 60)

print(
    f"Test patches: "
    f"{len(test_files)}"
)

print(
    f"Detected components: "
    f"{len(all_measurements)}"
)

print(
    f"Total predicted pixels: "
    f"{total_predicted_pixels}"
)


# ============================================================
# SAVE CSV
# ============================================================

csv_fields = [

    "patient_id",
    "patch_number",
    "source_patch",
    "slice_index",
    "component_number",
    "pixel_count",
    "width_pixels",
    "height_pixels",
    "width_mm",
    "height_mm",
    "area_mm2",
    "equivalent_diameter_mm"
]

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=csv_fields
    )

    writer.writeheader()

    for row in all_measurements:

        writer.writerow(
            row
        )


# ============================================================
# SAVE JSON SUMMARY
# ============================================================

summary = {

    "patient_id":
        PATIENT_ID,

    "test_patches":
        len(test_files),

    "probability_threshold":
        THRESHOLD,

    "minimum_component_size":
        MIN_COMPONENT_SIZE,

    "pixel_spacing_x_mm":
        PIXEL_SPACING_X_MM,

    "pixel_spacing_y_mm":
        PIXEL_SPACING_Y_MM,

    "slice_spacing_mm":
        SLICE_SPACING_MM,

    "detected_components":
        len(all_measurements),

    "total_predicted_pixels":
        total_predicted_pixels,

    "prediction_directory":
        str(PREDICTION_DIR),

    "measurements":
        all_measurements
}

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        indent=4
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()

print(
    "Measurement CSV:"
)

print(
    OUTPUT_CSV
)

print()

print(
    "Measurement JSON:"
)

print(
    OUTPUT_JSON
)

print()

print(
    "Saved U-Net predictions:"
)

print(
    PREDICTION_DIR
)

print()

print(
    "Prediction summary:"
)

print(
    PREDICTION_SUMMARY_JSON
)

print()

print("=" * 60)

print(
    "CT nodule measurement completed."
)

print("=" * 60)