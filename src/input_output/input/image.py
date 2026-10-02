import os
import traceback
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd
import pydicom as dcm
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QInputDialog,
    QMessageBox,
    QProgressDialog,
)

from domain.intravascular.types import SupportedType
from domain.intravascular.contour_presets import ContourPreset, active_preset, with_labels
from domain.intravascular.io_types import Contour, FrameData
from domain.intravascular.oct_display_types import OCT_LUT
from domain.intravascular.undo import push_pullback_contours_snapshot
from input_output.input.contours import read_contours
from input_output.input.mask_contours import frame_contours, label_aliases
from input_output.output.contours import write_contours
from input_output.input.metadata import (
    MetaDataIntravascular,
    PromptFn,
    extract_modality,
    parse_metadata_dcm,
    parse_metadata_nifti,
    populate_metadata_table,
)
from pages.intravascular.popup_windows.message_boxes import ErrorMessage, WarningMessage
from pages.intravascular.utils.metrics import clear_lumen_measurements


def read_image(main_window) -> None:
    from gui.active_page import ActivePage

    master = main_window.window()
    if hasattr(master, '_switch_page'):
        master._switch_page(ActivePage.INTRAVASCULAR.value)

    main_window.status_bar.showMessage('Reading image file...')
    file_name, _ = QFileDialog.getOpenFileName(main_window, 'Open File', '..', 'All files (*)')
    if not file_name:
        main_window.status_bar.showMessage(main_window.waiting_status)
        return

    master.reload_intravascular()
    main_window = master.intravascular_page

    root, ext = os.path.splitext(file_name)
    if ext == '.gz':
        root = os.path.splitext(root)[0]
    main_window.file_name = root

    progress = QProgressDialog('Reading image file...', '', 0, 3, main_window)
    progress.setWindowTitle('Loading')
    progress.setMinimumDuration(0)
    progress.setModal(True)
    progress.setValue(0)
    QApplication.processEvents()
    QApplication.processEvents()  # second flush processes the paint event queued by show

    _gray_oct_warning = False
    try:
        prompt = _make_prompt(main_window)

        if ext in ('.gz', '.nii'):
            pixel_array, metadata_df = _read_nifti(file_name)
            data_correct, _ = _check_integrity(metadata_df)
            if not data_correct:
                ErrorMessage(main_window, 'Data is corrupted. File could not be loaded.')
                main_window.status_bar.showMessage(main_window.waiting_status)
                return

            msg = QMessageBox(main_window)
            msg.setWindowTitle('Select Modality')
            msg.setText('Is this an IVUS or OCT NIfTI file?')
            msg.addButton('IVUS', QMessageBox.ButtonRole.ActionRole)
            oct_btn = msg.addButton('OCT', QMessageBox.ButtonRole.ActionRole)
            msg.exec()
            is_oct = msg.clickedButton() == oct_btn

            pixel_array_parsed, is_oct = _parse_pixel_array(pixel_array, is_oct)
            md = parse_metadata_nifti(metadata_df, pixel_array_parsed.shape[0], is_oct, prompt)
            if is_oct and pixel_array.ndim == 4 and pixel_array.shape[-1] == 3:
                # 4-D RGB NIfTI — use colour channels directly (same guard as DICOM path)
                main_window.runtime_data.images_rgb = pixel_array.clip(0, 255).astype(np.uint8)
            # 3D grayscale OCT: images_rgb stays None → _convert_gray_to_oct runs below
        else:
            try:
                pixel_array, metadata_df = _read_dicom(file_name)
                data_correct, _ = _check_integrity(metadata_df)
                if not data_correct:
                    ErrorMessage(main_window, 'Data is corrupted. File could not be loaded.')
                    main_window.status_bar.showMessage(main_window.waiting_status)
                    return
                is_oct = extract_modality(metadata_df) == 'OCT'
                pixel_array_parsed, is_oct = _parse_pixel_array(pixel_array, is_oct)
                md = parse_metadata_dcm(metadata_df, pixel_array_parsed.shape[0], prompt)
                if is_oct and pixel_array.ndim == 4 and pixel_array.shape[-1] == 3:
                    main_window.runtime_data.images_rgb = pixel_array.clip(0, 255).astype(np.uint8)
            except Exception:
                traceback.print_exc()
                ErrorMessage(
                    main_window,
                    f'File is not a valid {"/".join(t.value for t in SupportedType)} file and could not be loaded (DICOM or NIfTI supported)',
                )
                main_window.status_bar.showMessage(main_window.waiting_status)
                return

        main_window.runtime_data.images = pixel_array_parsed
        num_frames = pixel_array_parsed.shape[0]
        if is_oct and main_window.runtime_data.images_rgb is None:
            main_window.runtime_data.images_rgb = _convert_gray_to_oct(pixel_array_parsed)
            _gray_oct_warning = True

        _store_metadata(main_window, md, num_frames)
        populate_metadata_table(main_window.metadata_table, md, metadata_df)

        main_window.display_slider.blockSignals(True)
        main_window.display_slider.setMaximum(num_frames - 1)
        main_window.display_slider.blockSignals(False)

        progress.setValue(1)
        progress.setLabelText('Reading contours...')
        QApplication.processEvents()
        success = read_contours(main_window, main_window.file_name)
        if success:
            for i in range(num_frames):
                if i not in main_window.runtime_data.frame_data_dct:
                    main_window.runtime_data.frame_data_dct[i] = FrameData()
            main_window.segmentation = True
            main_window.runtime_data.gated_frames_dia = [
                i for i in range(num_frames) if main_window.runtime_data.frame_data_dct[i].phase == 'D'
            ]
            main_window.runtime_data.gated_frames_sys = [
                i for i in range(num_frames) if main_window.runtime_data.frame_data_dct[i].phase == 'S'
            ]
            main_window.runtime_data.tagged_frames = [
                i for i in range(num_frames) if main_window.runtime_data.frame_data_dct[i].phase == 'T'
            ]
            main_window.runtime_data.gated_frames = main_window.runtime_data.gated_frames_dia
        else:
            main_window.runtime_data.frame_data_dct = {i: FrameData() for i in range(num_frames)}

        progress.setValue(2)
        progress.setLabelText('Rendering...')
        QApplication.processEvents()
        main_window.display.set_data(main_window.runtime_data.images)
        main_window.image_displayed = True

        if success:
            # scaling_factor is now set; batch-compute areas for all contoured frames
            # so the pullback overviews are fully populated without requiring navigation.
            main_window.display.refresh_all_frame_metrics()

        main_window.display_slider.setValue(num_frames - 1)
        main_window.right_half.update_for_modality()
        progress.setValue(3)
        main_window.status_bar.showMessage(main_window.waiting_status)
    finally:
        progress.close()

    if _gray_oct_warning:
        WarningMessage(
            main_window,
            'The OCT file contained a grayscale image rather than true RGB.\n'
            'A sepia/copper false-colour lookup table has been applied automatically.',
        )

    if ext in ('.gz', '.nii'):
        main_window._last_image_dir = os.path.dirname(os.path.abspath(file_name))
        reply = QMessageBox.question(
            main_window,
            'Load Mask?',
            'Would you like to load a segmentation mask for this image?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            read_nifti_mask(main_window)


def read_nifti_mask(main_window) -> None:
    """Read a multi-label NIfTI mask over the whole pullback, replacing every frame's
    contours with those read off it (see mask_contours) after the active contour preset.

    Slice i of the mask is frame i of the pullback, so the two have to match in size. A
    mask holding labels the preset does not define offers a new preset with a row for each
    of them first. One Ctrl+Z undoes the whole import.
    """
    if not main_window.image_displayed:
        ErrorMessage(main_window, 'Load an image before importing a mask')
        return

    file_name, _ = QFileDialog.getOpenFileName(
        main_window,
        'Open NIfTI Mask',
        getattr(main_window, '_last_image_dir', None) or '..',
        'NIfTI files (*.nii *.nii.gz)',
    )
    if not file_name:
        return

    try:
        nft: nib.Nifti1Image = nib.load(file_name)  # type: ignore[assignment]
        mask_arr = _drop_trailing_singletons(np.rint(np.asarray(nft.dataobj)).astype(np.int32))
        mask_arr = mask_arr.transpose(2, 1, 0) if mask_arr.ndim == 3 else mask_arr.T
    except Exception:
        traceback.print_exc()
        ErrorMessage(main_window, 'Could not read NIfTI mask file')
        return

    images = main_window.runtime_data.images
    expected = tuple(images.shape[:3])
    if mask_arr.ndim == 2:
        mask_arr = mask_arr[np.newaxis]
    if tuple(mask_arr.shape) != expected:
        ErrorMessage(
            main_window,
            f'The mask holds {mask_arr.shape[0]} slices of {mask_arr.shape[2]}x{mask_arr.shape[1]} pixels, '
            f'the pullback {expected[0]} frames of {expected[2]}x{expected[1]}.\n\n'
            'A mask is read slice by slice onto the frames, so it has to cover the whole pullback.',
        )
        return

    preset = _preset_for_mask(main_window, mask_arr)
    if preset is None:
        return

    display = main_window.display
    knots = display.n_interactive_points
    handle_radius = display._angle_handle_radius() / display.scaling_factor
    frame_data_dct = main_window.runtime_data.frame_data_dct
    progress = QProgressDialog('Reading the mask...', 'Cancel', 0, len(mask_arr), main_window)
    progress.setWindowTitle('Open Intravascular Mask')
    progress.setMinimumDuration(0)
    progress.setModal(True)
    read: dict[int, tuple] = {}
    try:
        for frame in range(len(mask_arr)):
            progress.setValue(frame)
            QApplication.processEvents()
            if progress.wasCanceled():
                return  # nothing written yet
            read[frame] = frame_contours(
                mask_arr[frame], preset, lambda defn: knots if not defn.appendable else knots // 2, handle_radius
            )
    except Exception:
        traceback.print_exc()
        ErrorMessage(main_window, 'Error converting mask to contours')
        return
    finally:
        progress.close()

    push_pullback_contours_snapshot(main_window.runtime_data, display.frame)
    for frame, (contours, centroid) in read.items():
        frame_data = frame_data_dct.setdefault(frame, FrameData())
        clear_lumen_measurements(frame_data)  # derived from the lumen being replaced
        for defn in preset.types:
            frame_data.contours[defn.type.value] = contours.get(defn.type.value, Contour())
        frame_data.centroid = centroid

    main_window.segmentation = True
    main_window.contours_drawn = True
    display.set_frame(display.frame)
    display.refresh_all_frame_metrics()
    write_contours(main_window, force=True)
    main_window.status_bar.showMessage('Mask read into contours (Ctrl+Z undoes it)')


def _preset_for_mask(main_window, mask_arr: np.ndarray) -> ContourPreset | None:
    """The preset to read `mask_arr` after: the active one — or, when the mask holds labels
    it does not define and the user takes up the offer, a new one with a row for each of
    them, made in Intravascular Contour Settings. None to give up the import."""
    preset = active_preset()
    aliases = label_aliases(preset)  # read as another type, so not missing (the fibrous cap)
    values = [int(value) for value in np.unique(mask_arr) if value != 0]
    missing = [value for value in values if value not in aliases and all(defn.label != value for defn in preset.types)]
    if not missing:
        return preset

    shown = ', '.join(str(value) for value in missing[:10]) + (', …' if len(missing) > 10 else '')
    reply = QMessageBox.question(
        main_window,
        'Labels Without a Contour Type',
        f'This mask holds {len(values)} labels, {len(missing)} of which the preset {preset.name!r} does not '
        f'define ({shown}).\n\nCreate a new preset with a row for each of them? Otherwise they are left out.',
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
        QMessageBox.StandardButton.Yes,
    )
    if reply == QMessageBox.StandardButton.Cancel:
        return None
    if reply == QMessageBox.StandardButton.No:
        return preset

    # Imported here: the shortcuts module imports this one.
    from gui.shortcuts import apply_contour_preset
    from pages.intravascular.popup_windows.contour_settings_dialog import ContourSettingsDialog

    draft = with_labels(preset, missing, f'{preset.name} + {len(missing)} labels')
    dialog = ContourSettingsDialog(main_window, active_name=preset.name, draft=draft)
    if not dialog.exec() or dialog.selected_preset() is None:
        return None
    chosen = dialog.selected_preset()
    apply_contour_preset(main_window, getattr(main_window.window(), 'ccta_page', None), chosen)
    return chosen


def _make_prompt(main_window) -> PromptFn:
    def prompt(title: str, message: str, default: float) -> float:
        val, ok = QInputDialog.getDouble(main_window, title, message, default, decimals=4)
        return val if ok else default

    return prompt


def _store_metadata(main_window, md: MetaDataIntravascular, num_frames: int) -> None:
    main_window.runtime_data.metadata['modality'] = md.modality
    main_window.runtime_data.metadata['pullback_speed'] = md.pullback_speed
    main_window.runtime_data.metadata['pullback_length'] = md.pullback_length
    main_window.runtime_data.metadata['resolution'] = md.resolution
    main_window.runtime_data.metadata['dimension'] = md.dimension
    main_window.runtime_data.metadata['manufacturer'] = md.manufacturer
    main_window.runtime_data.metadata['model'] = md.model
    main_window.runtime_data.metadata['pullback_start_frame'] = md.pullback_start_frame
    main_window.runtime_data.metadata['frame_rate'] = md.frame_rate
    main_window.runtime_data.metadata['num_frames'] = num_frames


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
    try:
        pixel_array = nft.get_fdata()
    except Exception:
        # DT_RGB24 and other structured dtypes can't be cast to float by get_fdata().
        # Read raw bytes and unpack the named fields (R, G, B) into a trailing channel axis.
        raw = np.asarray(nft.dataobj)
        if raw.dtype.names:
            pixel_array = np.stack([raw[c].astype(np.float64) for c in raw.dtype.names], axis=-1)
        else:
            pixel_array = raw.astype(np.float64)
    pixel_array = _drop_trailing_singletons(pixel_array)
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


def _drop_trailing_singletons(array: np.ndarray) -> np.ndarray:
    """`array` without the axes of length 1 past its third.

    A 3-D volume is sometimes stored with a trailing axis of length 1 — a one-component
    'vector' (x, y, z, 1), as some converters write it — which would otherwise be read as
    a frame of (H, W, 1) instead of (H, W).
    """
    while array.ndim > 3 and array.shape[-1] == 1:
        array = array[..., 0]
    return array


_DICOM_MODALITY_ALIASES: dict[str, str] = {
    'US': 'IVUS',  # standard DICOM ultrasound
    'OPT': 'OCT',  # standard DICOM ophthalmic tomography
}


def _check_integrity(metadata: pd.DataFrame) -> tuple[bool, Optional[str]]:
    is_dicom = not metadata[metadata['Description'] == 'Modality'].empty
    if is_dicom:
        modality = metadata[metadata['Description'] == 'Modality']['Value']
        _accepted = {t.value for t in SupportedType} | set(_DICOM_MODALITY_ALIASES.keys())
        if modality.empty or not modality.isin(_accepted).any():
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


def _parse_pixel_array(pixel_array: np.ndarray, is_oct: bool | None = None) -> tuple[np.ndarray, bool]:
    if is_oct is None:
        # NIfTI path: auto-detect from shape (3D NIfTI OCT left for future work)
        is_oct = pixel_array.ndim == 4 and pixel_array.shape[-1] == 3
    if pixel_array.ndim == 4 and pixel_array.shape[-1] == 3:
        return _convert_oct_to_gray(pixel_array), is_oct
    return pixel_array, is_oct


def _convert_oct_to_gray(oct_array: np.ndarray) -> np.ndarray:
    weights = np.array([0.299, 0.587, 0.114])
    return np.dot(oct_array[..., :3], weights).astype(np.uint8)


def _convert_gray_to_oct(gray_array: np.ndarray) -> np.ndarray:
    """
    Convert (N, H, W) grayscale → (N, H, W, 3) uint8 with the OCT false-colour LUT.
    Normalises the full volume to [0, 255] before LUT lookup so the result is
    correct regardless of whether the input is float [0,1], raw HU, or uint8.
    """
    arr = gray_array.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi > lo:
        arr = (arr - lo) / (hi - lo) * 255.0
    return OCT_LUT[arr.clip(0, 255).astype(np.uint8)]
