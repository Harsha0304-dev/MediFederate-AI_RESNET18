from pathlib import Path
import json
import numpy as np
import pandas as pd
import pydicom


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PRED_ROOT = PROJECT_ROOT / "results" / "ct_predictions" / "LIDC-IDRI-0010"
CT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ct"
    / "lidc_idri"
    / "LIDC-IDRI-0010"
)

OUTPUT_ROOT = PROJECT_ROOT / "results" / "ct_3d_prediction"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

PATIENT_ID = "LIDC-IDRI-0010"
MIN_COMPONENT_VOXELS = 20


# ============================================================
# LOAD ACTUAL DICOM SPACING
# ============================================================

def get_ct_spacing():

    files = sorted(CT_ROOT.rglob("*.dcm"))

    if not files:
        raise FileNotFoundError(f"No CT DICOM files found in {CT_ROOT}")

    datasets = []

    for file in files:
        try:
            ds = pydicom.dcmread(file, stop_before_pixels=True)

            if getattr(ds, "Modality", "") == "CT":
                datasets.append(ds)

        except Exception:
            pass

    if not datasets:
        raise RuntimeError("No CT DICOM datasets found.")

    first = datasets[0]

    pixel_spacing = [float(x) for x in first.PixelSpacing]

    spacing_y = pixel_spacing[0]
    spacing_x = pixel_spacing[1]

    positions = []

    for ds in datasets:

        if hasattr(ds, "ImagePositionPatient"):
            positions.append(
                float(ds.ImagePositionPatient[2])
            )

    positions = sorted(set(positions))

    if len(positions) > 1:

        differences = np.diff(positions)

        spacing_z = float(np.median(np.abs(differences)))

    else:

        spacing_z = float(getattr(first, "SliceThickness", 1.0))

    return spacing_z, spacing_y, spacing_x


# ============================================================
# CONNECTED COMPONENTS IN 3D
# ============================================================

def connected_components_3d(volume):

    depth, height, width = volume.shape

    visited = np.zeros(volume.shape, dtype=bool)

    components = []

    neighbors = [
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    ]

    for z in range(depth):

        for y in range(height):

            for x in range(width):

                if not volume[z, y, x]:
                    continue

                if visited[z, y, x]:
                    continue

                stack = [(z, y, x)]
                visited[z, y, x] = True

                component = []

                while stack:

                    cz, cy, cx = stack.pop()

                    component.append((cz, cy, cx))

                    for dz, dy, dx in neighbors:

                        nz = cz + dz
                        ny = cy + dy
                        nx = cx + dx

                        if (
                            nz < 0
                            or nz >= depth
                            or ny < 0
                            or ny >= height
                            or nx < 0
                            or nx >= width
                        ):
                            continue

                        if not volume[nz, ny, nx]:
                            continue

                        if visited[nz, ny, nx]:
                            continue

                        visited[nz, ny, nx] = True

                        stack.append((nz, ny, nx))

                components.append(component)

    return components


# ============================================================
# READ PREDICTION FILES
# ============================================================

def load_predictions():

    mask_files = sorted(PRED_ROOT.glob("*_mask.npy"))

    if not mask_files:
        raise FileNotFoundError(
            f"No prediction masks found in {PRED_ROOT}"
        )

    records = []

    for file in mask_files:

        name = file.stem

        # Example:
        # LIDC-IDRI-0010_slice_0067_component_00_mask

        parts = name.split("_")

        slice_index = None

        for i, part in enumerate(parts):

            if part == "slice" and i + 1 < len(parts):

                slice_index = int(parts[i + 1])
                break

        if slice_index is None:
            continue

        mask = np.load(file)

        mask = mask.astype(bool)

        records.append(
            {
                "slice": slice_index,
                "mask": mask,
                "file": file.name,
            }
        )

    records.sort(key=lambda x: x["slice"])

    return records


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("MediFederate AI")
    print("3D CT U-Net Prediction Reconstruction")
    print("=" * 60)

    spacing_z, spacing_y, spacing_x = get_ct_spacing()

    print(f"Patient: {PATIENT_ID}")
    print()
    print("Actual CT spacing:")
    print(f"  X: {spacing_x:.6f} mm")
    print(f"  Y: {spacing_y:.6f} mm")
    print(f"  Z: {spacing_z:.6f} mm")

    records = load_predictions()

    print()
    print(f"Prediction slices found: {len(records)}")

    if not records:
        return

    # --------------------------------------------------------
    # Determine patch size
    # --------------------------------------------------------

    patch_height, patch_width = records[0]["mask"].shape

    min_slice = min(r["slice"] for r in records)
    max_slice = max(r["slice"] for r in records)

    depth = max_slice - min_slice + 1

    volume = np.zeros(
        (depth, patch_height, patch_width),
        dtype=bool,
    )

    # --------------------------------------------------------
    # Place masks into 3D volume
    # --------------------------------------------------------

    for record in records:

        z = record["slice"] - min_slice

        volume[z] = np.logical_or(
            volume[z],
            record["mask"],
        )

    print()
    print(
        f"3D volume shape: "
        f"{volume.shape}"
    )

    print(
        f"Slice range: "
        f"{min_slice} - {max_slice}"
    )

    # --------------------------------------------------------
    # Connected components
    # --------------------------------------------------------

    print()
    print("Detecting 3D connected components...")

    components = connected_components_3d(volume)

    print(
        f"Raw 3D components: "
        f"{len(components)}"
    )

    results = []

    for component_id, component in enumerate(
        components,
        start=1,
    ):

        voxel_count = len(component)

        if voxel_count < MIN_COMPONENT_VOXELS:
            continue

        coords = np.array(component)

        z_coords = coords[:, 0]
        y_coords = coords[:, 1]
        x_coords = coords[:, 2]

        z_min = int(z_coords.min())
        z_max = int(z_coords.max())

        y_min = int(y_coords.min())
        y_max = int(y_coords.max())

        x_min = int(x_coords.min())
        x_max = int(x_coords.max())

        # ----------------------------------------------------
        # Physical dimensions
        # ----------------------------------------------------

        width_mm = (
            (x_max - x_min + 1)
            * spacing_x
        )

        height_mm = (
            (y_max - y_min + 1)
            * spacing_y
        )

        depth_mm = (
            (z_max - z_min + 1)
            * spacing_z
        )

        area_mm2 = (
            voxel_count
            * spacing_x
            * spacing_y
        )

        volume_mm3 = (
            voxel_count
            * spacing_x
            * spacing_y
            * spacing_z
        )

        volume_cm3 = volume_mm3 / 1000.0

        equivalent_diameter_mm = (
            (6.0 * volume_mm3 / np.pi)
            ** (1.0 / 3.0)
        )

        first_slice = min_slice + z_min
        last_slice = min_slice + z_max

        result = {
            "patient_id": PATIENT_ID,
            "component_id": component_id,
            "voxel_count": voxel_count,
            "first_slice": first_slice,
            "last_slice": last_slice,
            "slice_count": z_max - z_min + 1,
            "width_mm": round(width_mm, 3),
            "height_mm": round(height_mm, 3),
            "depth_mm": round(depth_mm, 3),
            "area_mm2": round(area_mm2, 3),
            "volume_mm3": round(volume_mm3, 3),
            "volume_cm3": round(volume_cm3, 6),
            "equivalent_diameter_mm": round(
                equivalent_diameter_mm,
                3,
            ),
        }

        results.append(result)

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("3D PREDICTED NODULE MEASUREMENTS")
    print("=" * 60)

    if not results:

        print("No components passed the minimum voxel threshold.")

    else:

        for result in results:

            print()
            print(
                f"Component "
                f"{result['component_id']}"
            )

            print(
                f"  Slices: "
                f"{result['first_slice']} - "
                f"{result['last_slice']}"
            )

            print(
                f"  Width: "
                f"{result['width_mm']} mm"
            )

            print(
                f"  Height: "
                f"{result['height_mm']} mm"
            )

            print(
                f"  Depth: "
                f"{result['depth_mm']} mm"
            )

            print(
                f"  Volume: "
                f"{result['volume_cm3']} cm3"
            )

            print(
                f"  Equivalent diameter: "
                f"{result['equivalent_diameter_mm']} mm"
            )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    csv_path = (
        OUTPUT_ROOT
        / "predicted_3d_measurements.csv"
    )

    json_path = (
        OUTPUT_ROOT
        / "predicted_3d_measurements.json"
    )

    summary_path = (
        OUTPUT_ROOT
        / "predicted_3d_summary.json"
    )

    df = pd.DataFrame(results)

    df.to_csv(
        csv_path,
        index=False,
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    total_volume = sum(
        r["volume_cm3"]
        for r in results
    )

    summary = {
        "patient_id": PATIENT_ID,
        "prediction_source": str(PRED_ROOT),
        "number_of_prediction_slices": len(records),
        "number_of_3d_components": len(results),
        "spacing_mm": {
            "x": spacing_x,
            "y": spacing_y,
            "z": spacing_z,
        },
        "threshold": 0.80,
        "minimum_component_voxels": MIN_COMPONENT_VOXELS,
        "total_predicted_volume_cm3": round(
            total_volume,
            6,
        ),
    }

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("=" * 60)
    print("FILES SAVED")
    print("=" * 60)

    print(csv_path)
    print(json_path)
    print(summary_path)

    print()
    print("3D CT prediction reconstruction completed.")


if __name__ == "__main__":
    main()