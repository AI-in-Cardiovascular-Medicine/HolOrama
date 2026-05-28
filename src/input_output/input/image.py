import os
import traceback
from typing import Optional

import numpy as np
import pandas as pd
import nibabel as nib
import pydicom as dcm
import matplotlib.pyplot as plt
from skimage import measure as sk_measure
from PyQt6.QtWidgets import QFileDialog, QInputDialog

from gui.popup_windows.message_boxes import ErrorMessage
from input_output.input.metadata import (
    MetaData,
    PromptFn,
    parse_metadata_dcm,
    parse_metadata_nifti,
    populate_metadata_table,
)
from input_output.input.contours import read_contours
from domain.all_types import ContourType, SupportedType
from domain.io_types import FrameData
from domain.runtime_types import RuntimeData
from input_output.output.imgs_masks import (
    LABEL_LUMEN,
    LABEL_EEM_WALL,
    LABEL_CALCIUM,
    LABEL_LIPID,
    LABEL_MACROPHAGE,
    LABEL_BRANCH,
)
from segmentation.segment import downsample
from tools.geometry import SplineGeometry


def read_image(main_window) -> None:
    main_window.status_bar.showMessage('Reading image file...')
    file_name, _ = QFileDialog.getOpenFileName(
        main_window, 'Open File', '..', 'All files (*)', options=QFileDialog.Option.DontUseNativeDialog
    )
    if not file_name:
        main_window.status_bar.showMessage(main_window.waiting_status)
        return

    main_window.reset_state()
    main_window.gating_display.fig.clear()
    plt.draw()

    root, ext = os.path.splitext(file_name)
    if ext == '.gz':
        root = os.path.splitext(root)[0]
    main_window.file_name = root

    prompt = _make_prompt(main_window)

    if ext in ('.gz', '.nii'):
        pixel_array, metadata_df = _read_nifti(file_name)
        data_correct, _ = _check_integrity(metadata_df)
        if not data_correct:
            ErrorMessage(main_window, 'Data is corrupted. File could not be loaded.')
            main_window.status_bar.showMessage(main_window.waiting_status)
            return
        pixel_array_parsed, is_oct = _parse_pixel_array(pixel_array)
        md = parse_metadata_nifti(metadata_df, pixel_array_parsed.shape[0], is_oct, prompt)
        if is_oct:
            main_window.images_rgb = pixel_array.clip(0, 255).astype(np.uint8)
    else:
        try:
            pixel_array, metadata_df = _read_dicom(file_name)
            data_correct, _ = _check_integrity(metadata_df)
            if not data_correct:
                ErrorMessage(main_window, 'Data is corrupted. File could not be loaded.')
                main_window.status_bar.showMessage(main_window.waiting_status)
                return
            pixel_array_parsed, is_oct = _parse_pixel_array(pixel_array)
            md = parse_metadata_dcm(metadata_df, pixel_array_parsed.shape[0], prompt)
            if is_oct:
                main_window.images_rgb = pixel_array.clip(0, 255).astype(np.uint8)
        except Exception:
            traceback.print_exc()
            ErrorMessage(
                main_window,
                f'File is not a valid {"/".join(t.value for t in SupportedType)} file and could not be loaded (DICOM or NIfTI supported)',
            )
            main_window.status_bar.showMessage(main_window.waiting_status)
            return

    main_window.images = pixel_array_parsed
    num_frames = pixel_array_parsed.shape[0]

    _store_metadata(main_window, md, num_frames)
    populate_metadata_table(main_window.metadata_table, md, metadata_df)

    main_window.display_slider.blockSignals(True)
    main_window.display_slider.setMaximum(num_frames - 1)
    main_window.display_slider.blockSignals(False)

    success = read_contours(main_window, main_window.file_name)
    if success:
        for i in range(num_frames):
            if i not in main_window.data:
                main_window.data[i] = FrameData()
        main_window.segmentation = True
        main_window.gated_frames_dia = [i for i in range(num_frames) if main_window.data[i].phase == 'D']
        main_window.gated_frames_sys = [i for i in range(num_frames) if main_window.data[i].phase == 'S']
        main_window.gated_frames_oct = [i for i in range(num_frames) if main_window.data[i].phase == 'T']
        main_window.gated_frames = main_window.gated_frames_dia
    else:
        main_window.data = RuntimeData()
        main_window.data.frame_data_dct = {i: FrameData() for i in range(num_frames)}

    main_window.display.set_data(main_window.images)
    main_window.image_displayed = True
    main_window.display_slider.setValue(num_frames - 1)
    main_window.right_half.update_for_modality()
    main_window.status_bar.showMessage(main_window.waiting_status)


def read_nifti_mask(main_window, contour_type: ContourType = ContourType.LUMEN) -> None:
    if not main_window.image_displayed:
        ErrorMessage(main_window, 'Load an image before importing a mask')
        return

    _LABEL_MASK = {
        ContourType.LUMEN: lambda a: a == LABEL_LUMEN,
        ContourType.EEM: lambda a: np.isin(a, [LABEL_LUMEN, LABEL_EEM_WALL]),
        ContourType.CALCIUM: lambda a: a == LABEL_CALCIUM,
        ContourType.LIPID: lambda a: a == LABEL_LIPID,
        ContourType.MACROPHAGE: lambda a: a == LABEL_MACROPHAGE,
        ContourType.BRANCH: lambda a: a == LABEL_BRANCH,
    }
    if contour_type not in _LABEL_MASK:
        return

    file_name, _ = QFileDialog.getOpenFileName(
        main_window,
        'Open NIfTI Mask',
        '..',
        'NIfTI files (*.nii *.nii.gz)',
        options=QFileDialog.Option.DontUseNativeDialog,
    )
    if not file_name:
        return

    try:
        nft: nib.Nifti1Image = nib.load(file_name)  # type: ignore[assignment]
        mask_arr = np.asarray(nft.dataobj).transpose(2, 1, 0).astype(np.uint8)
    except Exception:
        traceback.print_exc()
        ErrorMessage(main_window, 'Could not read NIfTI mask file')
        return

    num_frames = min(mask_arr.shape[0], main_window.metadata['num_frames'])
    mask_fn = _LABEL_MASK[contour_type]
    field_name = contour_type.value
    single_contour = contour_type in (ContourType.LUMEN, ContourType.EEM)

    n_pts = main_window.display.n_interactive_points
    n_pts_contour = main_window.display.n_points_contour
    sf = main_window.display.scaling_factor
    try:
        for frame_idx in range(num_frames):
            binary = mask_fn(mask_arr[frame_idx]).astype(np.uint8)
            if not binary.any():
                continue
            found = sk_measure.find_contours(binary, 0.5)
            if not found:
                continue
            if single_contour:
                found = [max(found, key=len)]
            if frame_idx not in main_window.data:
                main_window.data[frame_idx] = FrameData()
            contour_obj = getattr(main_window.data[frame_idx], field_name)
            sparse_contours = []
            for c in found:
                xs_scaled = [float(col) * sf for col in c[:, 1]]
                ys_scaled = [float(row) * sf for row in c[:, 0]]
                geometry = SplineGeometry(xs_scaled, ys_scaled, n_pts_contour, None, None)
                if geometry.full_contour[0] is None or len(geometry.full_contour[0]) == 0:
                    continue
                downsampled = downsample(
                    ([list(geometry.full_contour[0])], [list(geometry.full_contour[1])]),
                    n_pts,
                )
                sparse_contours.append([[x / sf for x in downsampled[0]], [y / sf for y in downsampled[1]]])
            if not sparse_contours:
                continue
            contour_obj.contours = sparse_contours
            contour_obj.closed = [True] * len(sparse_contours)
    except Exception:
        traceback.print_exc()
        ErrorMessage(main_window, 'Error converting mask to contours')
        return

    main_window.segmentation = True
    main_window.display.set_frame(main_window.display.frame)


def _make_prompt(main_window) -> PromptFn:
    def prompt(title: str, message: str, default: float) -> float:
        val, ok = QInputDialog.getDouble(main_window, title, message, default, decimals=4)
        return val if ok else default

    return prompt


def _store_metadata(main_window, md: MetaData, num_frames: int) -> None:
    main_window.metadata['modality'] = md.modality
    main_window.metadata['pullback_speed'] = md.pullback_speed
    main_window.metadata['pullback_length'] = md.pullback_length
    main_window.metadata['resolution'] = md.resolution
    main_window.metadata['dimension'] = md.dimension
    main_window.metadata['manufacturer'] = md.manufacturer
    main_window.metadata['model'] = md.model
    main_window.metadata['pullback_start_frame'] = md.pullback_start_frame
    main_window.metadata['frame_rate'] = md.frame_rate
    main_window.metadata['num_frames'] = num_frames


_PRIVATE_TAGS = {
    0x000B1001: 'BostonPullbackRate',  # Boston Scientific pullback rate (mm/s)
}


def _read_dicom(filename: str) -> tuple[np.ndarray, pd.DataFrame]:
    dicom = dcm.dcmread(filename, force=True, defer_size=256)
    pixel_array = dicom.pixel_array

    rows = []
    for elem in dicom:
        if elem.name == 'Pixel Data':
            continue
        rows.append({'Tag': str(elem.tag), 'VR': elem.VR, 'Description': elem.name, 'Value': elem.value})
    for tag, name in _PRIVATE_TAGS.items():
        if tag in dicom:
            rows.append(
                {
                    'Tag': str(dicom[tag].tag),
                    'VR': dicom[tag].VR,
                    'Description': name,
                    'Value': dicom[tag].value,
                }
            )
    return pixel_array, pd.DataFrame(rows)


def _read_nifti(filename: str) -> tuple[np.ndarray, pd.DataFrame]:
    nft: nib.Nifti1Image = nib.load(filename)  # type: ignore[assignment]
    pixel_array = nft.get_fdata()
    if pixel_array.ndim == 3:
        pixel_array = pixel_array.transpose(2, 1, 0)
    elif pixel_array.ndim == 4:
        pixel_array = pixel_array.transpose(2, 1, 0, 3)

    rows_nft = []
    for field in nft.header.structarr.dtype.names:
        rows_nft.append(
            {
                'Tag': field,
                'VR': str(nft.header.structarr.dtype[field]),
                'Description': field,
                'Value': nft.header[field].tolist(),
            }
        )
    return pixel_array, pd.DataFrame(rows_nft)


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
            if hasattr(d, '__len__') and (d[0] < 3 or d[3] < 1):
                return False, None
        pixdim = metadata[metadata['Description'] == 'pixdim']['Value']
        if not pixdim.empty:
            p = pixdim.iloc[0]
            if hasattr(p, '__len__') and len(p) > 1 and p[1] <= 0:
                return False, None
        return True, None


def _parse_pixel_array(pixel_array: np.ndarray) -> tuple[np.ndarray, bool]:
    if pixel_array.ndim == 4 and pixel_array.shape[-1] == 3:
        return _convert_oct_to_gray(pixel_array), True
    return pixel_array, False


def _convert_oct_to_gray(oct_array: np.ndarray) -> np.ndarray:
    weights = np.array([0.299, 0.587, 0.114])
    return np.dot(oct_array[..., :3], weights).astype(np.uint8)
