import json
import os
from typing import Any, Callable

import numpy as np
import pydicom as dcm
import SimpleITK as sitk

from domain.io_types import MetaDataCCTA


def read_ct_volume(
    folder: str,
    progress_cb: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Read a CT DICOM series from `folder` (one level deep, no recursion).

    Scans all files at the top level, keeps only CT slices that carry
    ImagePositionPatient, picks the series with the most slices, sorts by
    z-position, applies HU calibration (RescaleSlope/Intercept), and stacks
    into a volume.

    Args:
        folder: directory containing the DICOM files
        progress_cb: optional callback(current, total) called after each slice load

    Returns:
        volume: (n_slices, H, W) int16 array in Hounsfield Units
        metadata: dict with pixel_spacing (mm tuple), slice_thickness (mm), n_slices

    Raises:
        ValueError: if no valid CT slices are found
    """
    entries = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    if not entries:
        raise ValueError(f'No files found in {folder}')

    # Fast header-only pass
    headers: list[tuple[str, Any, float]] = []
    for path in entries:
        try:
            ds = dcm.dcmread(path, force=True, stop_before_pixels=True)
            if getattr(ds, 'Modality', None) != 'CT':
                continue
            pos = getattr(ds, 'ImagePositionPatient', None)
            if pos is None:
                continue
            headers.append((path, ds, float(pos[2])))
        except Exception:
            continue

    if not headers:
        raise ValueError(
            'No CT slices with spatial position (ImagePositionPatient) found.\n'
            'Make sure to open the folder that contains the individual slice files.'
        )

    # Group by SeriesInstanceUID, pick series with most slices
    series: dict[str, list] = {}
    for path, ds, z in headers:
        uid = str(getattr(ds, 'SeriesInstanceUID', 'unknown'))
        series.setdefault(uid, []).append((path, ds, z))
    chosen = max(series.values(), key=len)
    chosen.sort(key=lambda t: t[2])

    paths = [t[0] for t in chosen]
    first_ds = chosen[0][1]
    total = len(paths)

    # Load pixel data and apply HU calibration
    slices = []
    for i, path in enumerate(paths):
        if progress_cb is not None:
            progress_cb(i + 1, total)
        ds = dcm.dcmread(path, force=True)
        px = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, 'RescaleSlope', 1))
        intercept = float(getattr(ds, 'RescaleIntercept', 0))
        slices.append((px * slope + intercept).astype(np.int16))

    volume = np.stack(slices, axis=0)  # (n_slices, H, W)

    px_spacing = getattr(first_ds, 'PixelSpacing', [1.0, 1.0])
    if len(chosen) > 1:
        slice_thickness = abs(float(chosen[1][2]) - float(chosen[0][2]))
    else:
        slice_thickness = float(getattr(first_ds, 'SliceThickness', 1.0))

    try:
        ccta_meta = parse_metadata_ccta(first_ds)
    except Exception:
        ccta_meta = MetaDataCCTA()

    metadata = {
        'pixel_spacing': (float(px_spacing[0]), float(px_spacing[1])),
        'slice_thickness': slice_thickness,
        'n_slices': total,
        'ccta_metadata': ccta_meta,
    }
    return volume, metadata


_CCTA_KNOWN_KEYWORDS = frozenset(
    {
        'PatientName',
        'PatientBirthDate',
        'PatientSex',
        'Manufacturer',
        'ManufacturerModelName',
        'PixelSpacing',
        'SliceThickness',
    }
)

# Binary VRs that can't be meaningfully stringified
_BINARY_VRS = frozenset({'OB', 'OW', 'UN', 'OD', 'OF', 'OL', 'OV', 'UR'})


def parse_metadata_ccta(ds: Any) -> MetaDataCCTA:
    """Extract patient, device, and all remaining DICOM tags from a pydicom CT Dataset."""

    def _str(attr: str, default: str = 'Unknown') -> str:
        val = getattr(ds, attr, None)
        return str(val).strip() if val is not None else default

    birthdate = _str('PatientBirthDate')
    if len(birthdate) == 8 and birthdate.isdigit():
        birthdate = f'{birthdate[:4]}/{birthdate[4:6]}/{birthdate[6:]}'

    px = getattr(ds, 'PixelSpacing', [0.0, 0.0])
    st = float(getattr(ds, 'SliceThickness', 0.0) or 0.0)

    # Collect all remaining tags (mirrors the intravascular "extra rows" pattern)
    raw_tags: dict = {}
    try:
        for elem in ds:
            if elem.tag == (0x7FE0, 0x0010):  # PixelData — skip
                continue
            if elem.VR in _BINARY_VRS:
                continue
            if elem.keyword in _CCTA_KNOWN_KEYWORDS:
                continue
            try:
                name = elem.name or elem.keyword or str(elem.tag)
                raw_tags[name] = str(elem.value)
            except Exception:
                pass
    except Exception:
        pass

    return MetaDataCCTA(
        modality='CCTA',
        patient_name=_str('PatientName'),
        birthdate=birthdate,
        sex=_str('PatientSex'),
        slice_thickness=st,
        pixel_spacing=(float(px[0]), float(px[1])),
        manufacturer=_str('Manufacturer'),
        model=_str('ManufacturerModelName'),
        raw_tags=raw_tags,
    )


def parse_metadata_ccta_nifti(path: str, img: Any) -> MetaDataCCTA:
    """
    Parse CCTA metadata from a NIfTI file.

    Spacing comes from the SimpleITK image header.
    Patient / device fields are read from a BIDS JSON sidecar if present
    (dcm2niix writes one with the same basename: e.g. scan.json beside scan.nii.gz).
    """
    dx, dy, dz = img.GetSpacing()

    # Strip .nii or .nii.gz to find the sidecar
    base = path
    for ext in ('.nii.gz', '.nii'):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    sidecar = base + '.json'

    patient_name = birthdate = sex = manufacturer = model = 'Unknown'
    if os.path.isfile(sidecar):
        try:
            with open(sidecar, encoding='utf-8') as f:
                info = json.load(f)
            patient_name = str(info.get('PatientName', 'Unknown'))
            raw_bd = str(info.get('PatientBirthDate', 'Unknown'))
            if len(raw_bd) == 8 and raw_bd.isdigit():
                raw_bd = f'{raw_bd[:4]}/{raw_bd[4:6]}/{raw_bd[6:]}'
            birthdate = raw_bd
            sex = str(info.get('PatientSex', 'Unknown'))
            manufacturer = str(info.get('Manufacturer', 'Unknown'))
            model = str(info.get('ManufacturerModelName', 'Unknown'))
        except Exception:
            pass

    return MetaDataCCTA(
        modality='CCTA',
        patient_name=patient_name,
        birthdate=birthdate,
        sex=sex,
        slice_thickness=dz,
        pixel_spacing=(dy, dx),
        manufacturer=manufacturer,
        model=model,
    )


def read_nifti_volume(path: str) -> tuple[np.ndarray, dict]:
    """
    Read a NIfTI file (.nii or .nii.gz) into the same format as read_ct_volume.

    Returns:
        volume: (Z, Y, X) int16 array (values passed through as-is, assumed HU)
        metadata: dict with pixel_spacing (dy, dx) in mm, slice_thickness in mm, n_slices

    Raises:
        ValueError: if the file cannot be read or is not 3-D
    """
    try:
        img = sitk.ReadImage(path)
    except Exception as e:
        raise ValueError(f'Could not read NIfTI file: {e}') from e

    if img.GetDimension() != 3:
        raise ValueError(f'Expected a 3-D NIfTI volume, got {img.GetDimension()}-D.')

    volume = sitk.GetArrayFromImage(img).astype(np.int16)  # (Z, Y, X)
    dx, dy, dz = img.GetSpacing()  # SimpleITK order: x, y, z

    metadata = {
        'pixel_spacing': (dy, dx),  # (row spacing, col spacing) — matches DICOM convention
        'slice_thickness': dz,
        'n_slices': volume.shape[0],
        'ccta_metadata': parse_metadata_ccta_nifti(path, img),
    }
    return volume, metadata


def read_mask_volume(path: str) -> tuple[np.ndarray, dict]:
    """
    Read a segmentation mask from a NIfTI file (.nii or .nii.gz).

    Integer label values are preserved as-is (e.g. 0=background, 1=lumen, …).
    Spatial metadata uses the same convention as read_ct_volume / read_nifti_volume.

    Returns:
        mask: (Z, Y, X) uint8 array of label values
        metadata: dict with pixel_spacing (dy, dx) in mm, slice_thickness in mm, n_slices

    Raises:
        ValueError: if the file cannot be read or is not 3-D
    """
    try:
        img = sitk.ReadImage(path)
    except Exception as e:
        raise ValueError(f'Could not read mask file: {e}') from e

    if img.GetDimension() != 3:
        raise ValueError(f'Expected a 3-D mask volume, got {img.GetDimension()}-D.')

    mask = sitk.GetArrayFromImage(img).astype(np.uint8)  # (Z, Y, X)
    dx, dy, dz = img.GetSpacing()

    metadata = {
        'pixel_spacing': (dy, dx),
        'slice_thickness': dz,
        'n_slices': mask.shape[0],
    }
    return mask, metadata
