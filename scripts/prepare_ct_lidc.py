from pathlib import Path
import glob
import json

import numpy as np
import pydicom
import nibabel as nib
from highdicom.seg import Segmentation


# ============================================================
# Paths
# ============================================================

CT_BASE = Path(
    r"C:\Users\Harsha\Downloads\LIDC_CT_Images_10\lidc_idri"
)

SEG_BASE = Path(
    r"C:\Users\Harsha\Downloads\LIDC_CT_Nodules\lidc_idri"
)

OUTPUT_BASE = Path("data/processed/ct")
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)


# ============================================================
# Patients
# ============================================================

PATIENTS = [
    f"LIDC-IDRI-{i:04d}"
    for i in range(1, 11)
]


# ============================================================
# Utility functions
# ============================================================

def load_ct_series(patient_dir):
    """Load CT slices and sort them by physical Z position."""

    files = sorted(patient_dir.rglob("*.dcm"))

    slices = []

    for f in files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=False)

            if getattr(ds, "Modality", "") != "CT":
                continue

            if not hasattr(ds, "PixelData"):
                continue

            slices.append((f, ds))

        except Exception:
            continue

    if not slices:
        raise RuntimeError("No CT slices found.")

    # Sort by physical Z coordinate
    slices.sort(
        key=lambda x: float(
            x[1].ImagePositionPatient[2]
        )
        if hasattr(x[1], "ImagePositionPatient")
        else float(getattr(x[1], "InstanceNumber", 0))
    )

    return slices


def ct_to_hu(slices):
    """Convert CT pixel values to Hounsfield Units."""

    images = []

    for _, ds in slices:
        arr = ds.pixel_array.astype(np.float32)

        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))

        hu = arr * slope + intercept
        images.append(hu)

    volume = np.stack(images, axis=0)

    return volume


def get_spacing(slices):
    """Return physical voxel spacing in mm."""

    first = slices[0][1]

    pixel_spacing = [
        float(first.PixelSpacing[0]),
        float(first.PixelSpacing[1]),
    ]

    if len(slices) > 1:
        z_positions = [
            float(ds.ImagePositionPatient[2])
            for _, ds in slices
            if hasattr(ds, "ImagePositionPatient")
        ]

        z_positions = sorted(z_positions)

        differences = np.diff(z_positions)

        if len(differences):
            slice_spacing = float(np.median(np.abs(differences)))
        else:
            slice_spacing = float(
                getattr(first, "SliceThickness", 1.0)
            )
    else:
        slice_spacing = float(
            getattr(first, "SliceThickness", 1.0)
        )

    return (
        slice_spacing,
        pixel_spacing[0],
        pixel_spacing[1],
    )


def find_seg_files(patient_dir):
    """Find DICOM SEG objects."""

    files = sorted(patient_dir.rglob("*.dcm"))

    seg_files = []

    for f in files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)

            if getattr(ds, "Modality", "") == "SEG":
                seg_files.append(f)

        except Exception:
            continue

    return seg_files


def map_seg_to_ct(seg_file, ct_slices):
    """
    Decode a SEG object and map its frames to CT slices
    using referenced SOP Instance UIDs.
    """

    seg = Segmentation.from_dataset(
        pydicom.dcmread(str(seg_file))
    )

    seg_ds = seg

    # CT SOP UID -> volume index
    sop_to_index = {}

    for idx, (_, ds) in enumerate(ct_slices):
        sop_uid = getattr(ds, "SOPInstanceUID", None)

        if sop_uid:
            sop_to_index[sop_uid] = idx

    # Decode segmentation pixel array
    pixel_array = seg_ds.pixel_array

    # SEG commonly has shape:
    # (frames, rows, columns)
    if pixel_array.ndim == 2:
        pixel_array = pixel_array[np.newaxis, ...]

    mask = np.zeros(
        (
            len(ct_slices),
            pixel_array.shape[-2],
            pixel_array.shape[-1],
        ),
        dtype=np.uint8,
    )

    mapped = 0

    # Extract referenced CT SOP UIDs
    try:
        per_frame = seg_ds.PerFrameFunctionalGroupsSequence

        for frame_idx, frame in enumerate(per_frame):

            try:
                derivation = (
                    frame.DerivationImageSequence[0]
                )

                source_image = (
                    derivation.SourceImageSequence[0]
                )

                referenced_uid = (
                    source_image.ReferencedSOPInstanceUID
                )

            except Exception:
                continue

            if referenced_uid not in sop_to_index:
                continue

            ct_idx = sop_to_index[referenced_uid]

            mask[ct_idx] |= (
                pixel_array[frame_idx] > 0
            ).astype(np.uint8)

            mapped += 1

    except Exception as e:
        print(
            f"    Warning: frame mapping issue: {e}"
        )

    return mask, mapped


def measure_mask(mask, spacing):
    """Calculate physical dimensions and volume."""

    voxel_z, voxel_y, voxel_x = spacing

    coords = np.argwhere(mask > 0)

    if len(coords) == 0:
        return {
            "width_mm": 0,
            "height_mm": 0,
            "depth_mm": 0,
            "volume_mm3": 0,
            "volume_cm3": 0,
            "voxel_count": 0,
        }

    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0)

    depth = (z_max - z_min + 1) * voxel_z
    height = (y_max - y_min + 1) * voxel_y
    width = (x_max - x_min + 1) * voxel_x

    voxel_volume = (
        voxel_z * voxel_y * voxel_x
    )

    volume_mm3 = (
        int(np.sum(mask > 0)) * voxel_volume
    )

    return {
        "width_mm": float(width),
        "height_mm": float(height),
        "depth_mm": float(depth),
        "volume_mm3": float(volume_mm3),
        "volume_cm3": float(volume_mm3 / 1000.0),
        "voxel_count": int(np.sum(mask > 0)),
    }


def save_nifti(volume, output_path, spacing):
    """
    Save volume with a spacing-aware affine.

    Orientation is kept in the processing coordinate system;
    physical voxel spacing is preserved.
    """

    z, y, x = spacing

    affine = np.diag([
        x,
        y,
        z,
        1.0,
    ])

    nii = nib.Nifti1Image(
        volume.astype(np.float32),
        affine,
    )

    nib.save(nii, str(output_path))


# ============================================================
# Process one patient
# ============================================================

def process_patient(patient_id):

    print("\n" + "=" * 60)
    print(patient_id)
    print("=" * 60)

    ct_dir = CT_BASE / patient_id
    seg_dir = SEG_BASE / patient_id

    output_dir = OUTPUT_BASE / patient_id
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    if not ct_dir.exists():
        print("ERROR: CT directory not found")
        return False

    if not seg_dir.exists():
        print("ERROR: SEG directory not found")
        return False

    # --------------------------------------------------------
    # CT
    # --------------------------------------------------------

    ct_slices = load_ct_series(ct_dir)

    print(f"CT slices: {len(ct_slices)}")

    ct_volume = ct_to_hu(ct_slices)

    spacing = get_spacing(ct_slices)

    print(
        "Voxel spacing:",
        spacing,
        "mm"
    )

    print(
        "CT volume shape:",
        ct_volume.shape
    )

    # Save CT volume
    ct_output = output_dir / "ct_hu.nii.gz"

    save_nifti(
        ct_volume,
        ct_output,
        spacing
    )

    # --------------------------------------------------------
    # SEG
    # --------------------------------------------------------

    seg_files = find_seg_files(seg_dir)

    print(
        f"SEG files: {len(seg_files)}"
    )

    measurements = []

    for idx, seg_file in enumerate(seg_files, start=1):

        print(
            f"\n  SEG {idx}: {seg_file.name}"
        )

        try:

            mask, mapped = map_seg_to_ct(
                seg_file,
                ct_slices
            )

            print(
                f"    Frames mapped: {mapped}"
            )

            voxel_count = int(
                np.sum(mask > 0)
            )

            print(
                f"    Mask voxels: {voxel_count}"
            )

            if voxel_count == 0:
                print(
                    "    WARNING: Empty mask"
                )
                continue

            mask_output = (
                output_dir /
                f"nodule_mask_{idx}.nii.gz"
            )

            save_nifti(
                mask,
                mask_output,
                spacing
            )

            measurement = measure_mask(
                mask,
                spacing
            )

            measurement["segmentation"] = (
                f"nodule_mask_{idx}.nii.gz"
            )

            measurements.append(
                measurement
            )

            print(
                f"    Width : "
                f"{measurement['width_mm']:.2f} mm"
            )

            print(
                f"    Height: "
                f"{measurement['height_mm']:.2f} mm"
            )

            print(
                f"    Depth : "
                f"{measurement['depth_mm']:.2f} mm"
            )

            print(
                f"    Volume: "
                f"{measurement['volume_cm3']:.4f} cm3"
            )

        except Exception as e:

            print(
                f"    ERROR processing SEG: {e}"
            )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata = {
        "patient_id": patient_id,
        "ct_slices": len(ct_slices),
        "ct_shape": list(ct_volume.shape),
        "spacing_mm": list(spacing),
        "segmentations": measurements,
    }

    metadata_file = (
        output_dir / "metadata.json"
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

    print(
        f"\nSaved: {output_dir}"
    )

    return True


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    successful = 0

    print(
        "MediFederate AI - LIDC-IDRI CT preprocessing"
    )

    print(
        f"Patients to process: {len(PATIENTS)}"
    )

    for patient_id in PATIENTS:

        try:

            if process_patient(patient_id):
                successful += 1

        except Exception as e:

            print(
                f"\nFAILED: {patient_id}"
            )

            print(
                f"Reason: {e}"
            )

    print("\n" + "=" * 60)

    print(
        f"Processing completed: "
        f"{successful}/{len(PATIENTS)} patients"
    )

    print("=" * 60)