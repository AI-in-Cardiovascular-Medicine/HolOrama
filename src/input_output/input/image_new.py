import os
import traceback
from typing import Optional
import pandas as pd
import numpy as np
import nibabel as nib

import pydicom as dcm
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import QFileDialog


from gui.popup_windows.message_boxes import ErrorMessage
from input_output.input.metadata_new import parse_metadata_dcm, parse_metadata_nifti
from domain.all_types import SupportedType


def read_image(main_window):
    main_window.reset_state()
    main_window.gating_display.fig.clear()
    plt.draw()
    main_window.status_bar.showMessage('Reading image file...')
    file_name, _ = QFileDialog.getOpenFileName(
        main_window, 'Open File', '..', 'All files (*)', options=QFileDialog.Option.DontUseNativeDialog
    )

    root, ext = os.path.splitext(file_name)
    if ext == '.gz':
        root = os.path.splitext(root)[0]
    main_window.file_name = root

    if file_name:
        if ext == '.gz' or ext == '.nii':
            pixel_array, metadata = _read_nifit(file_name)
            data_correct, _ = _check_integrity(metadata)
            if not data_correct:
                ErrorMessage(main_window, 'Data is corrupted. File could not be loaded.')
                return None
            pixel_array_parsed, is_oct = _parse_pixel_array(pixel_array)
            parse_metadata_nifti(metadata, pixel_array_parsed.shape[0], is_oct)
            if is_oct:
                main_window.images_rgb = pixel_array.clip(0, 255).astype(np.uint8)
                main_window.images_display = 1
        else:
            try:
                pixel_array, metadata = _read_dicom(file_name)  # can also have no file extension
                data_correct, _ = _check_integrity(metadata)
                if not data_correct:
                    ErrorMessage(main_window, 'Data is corrupted. File could not be loaded.')
                    return None
                pixel_array_parsed, is_oct = _parse_pixel_array(pixel_array)
                parse_metadata_dcm(metadata, pixel_array_parsed.shape[0])
                if is_oct:
                    main_window.images_rgb = pixel_array.clip(0, 255).astype(np.uint8)
                    main_window.images_display = 1

            except Exception:
                traceback.print_exc()
                ErrorMessage(
                    main_window,
                    f'File is not a valid {"/".join(t.value for t in SupportedType)} file and could not be loaded (DICOM or NIfTi supported)',
                )
                return None

    main_window.images = pixel_array_parsed


_PRIVATE_TAGS = {
    0x000B1001: 'BostonPullbackRate',  # Boston Scientific pullback rate (mm/s)
}


def _read_dicom(filename) -> tuple[np.ndarray, pd.DataFrame]:
    dicom = dcm.dcmread(filename, force=True, defer_size=256)
    pixel_array = dicom.pixel_array

    rows = []
    for elem in dicom:
        if elem.name == 'Pixel Data':
            continue
        rows.append(
            {
                "Tag": str(elem.tag),
                "VR": elem.VR,
                "Description": elem.name,
                "Value": elem.value,
            }
        )
    for tag, name in _PRIVATE_TAGS.items():
        if tag in dicom:
            rows.append(
                {
                    "Tag": str(dicom[tag].tag),
                    "VR": dicom[tag].VR,
                    "Description": name,
                    "Value": dicom[tag].value,
                }
            )
    metadata = pd.DataFrame(rows)

    return pixel_array, metadata


def _read_nifit(filename) -> tuple[np.ndarray, pd.DataFrame]:
    nft: nib.Nifti1Image = nib.load(filename)  # type: ignore[assignment]
    pixel_array = nft.get_fdata()
    # nibabel returns (x, y, z[, c]) — transpose to (frames, h, w[, c]) to match pydicom convention
    if pixel_array.ndim == 3:
        pixel_array = pixel_array.transpose(2, 1, 0)
    elif pixel_array.ndim == 4:
        pixel_array = pixel_array.transpose(2, 1, 0, 3)

    rows_nft = []
    for field in nft.header.structarr.dtype.names:
        rows_nft.append(
            {
                "Tag": field,
                "VR": str(nft.header.structarr.dtype[field]),
                "Description": field,
                "Value": nft.header[field].tolist(),
            }
        )
    metadata = pd.DataFrame(rows_nft)

    return pixel_array, metadata


def _check_integrity(metadata: pd.DataFrame) -> tuple[bool, Optional[str]]:
    is_dicom = not metadata[metadata['Description'] == 'Modality'].empty
    if is_dicom:
        modality = metadata[metadata['Description'] == 'Modality']['Value']
        if modality.empty or not modality.isin([t.value for t in SupportedType]).any():
            return False, None
        num_frames = metadata[metadata['Description'] == 'Number of Frames']['Value']
        if not num_frames.empty and int(num_frames.iloc[0]) < 1:
            return False, None
        return True, str(modality.iloc[0])
    else:
        dim = metadata[metadata['Description'] == 'dim']['Value']
        if not dim.empty:
            d = dim.iloc[0]
            # d[0] = number of dimensions, d[3] = z/frame count
            if hasattr(d, '__len__') and (d[0] < 3 or d[3] < 1):
                return False, None
        pixdim = metadata[metadata['Description'] == 'pixdim']['Value']
        if not pixdim.empty:
            p = pixdim.iloc[0]
            if hasattr(p, '__len__') and len(p) > 1 and p[1] <= 0:
                return False, None
        return True, None  # NIfTI — modality determined by pixel array shape


def _parse_pixel_array(pixel_array: np.ndarray) -> tuple[np.ndarray, bool]:
    if pixel_array.ndim == 4 and pixel_array.shape[-1] == 3:
        return _convert_oct_to_gray(pixel_array), True
    return pixel_array, False


def _convert_oct_to_gray(oct_array):
    """
    Converts an RGB OCT array (Frames, H, W, 3) to Grayscale (Frames, H, W).
    """
    # Define the luminosity weights
    weights = np.array([0.299, 0.587, 0.114])

    # Use dot product to apply weights to the last dimension (the 3 color channels)
    # This effectively does: (R * 0.299) + (G * 0.587) + (B * 0.114)
    gray_oct = np.dot(oct_array[..., :3], weights)

    return gray_oct.astype(np.uint8)
