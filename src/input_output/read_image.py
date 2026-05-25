import os
import traceback

import pydicom as dcm
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import QFileDialog

from skimage import measure as sk_measure

from gui.popup_windows.message_boxes import ErrorMessage
from input_output.metadata import parse_dicom, parse_nifti, parse_nifti_oct
from input_output.contours_io import read_contours
from domain.io_types import FrameData
from segmentation.save_as_nifti import (
    LABEL_LUMEN,
    LABEL_EEM_WALL,
    LABEL_CALCIUM,
    LABEL_LIPID,
    LABEL_MACROPHAGE,
    LABEL_BRANCH,
)
from segmentation.segment import downsample
from domain.all_types import ContourType
from tools.geometry import SplineGeometry


def read_image(main_window):
    """
    Reads DICOM or NIfTi images.

    Reads the DICOM/NIfTi images and metadata. Places metatdata in a table.
    Images are displayed in the graphics scene.
    """
    main_window.reset_state()
    main_window.status_bar.showMessage('Reading image file...')
    file_name, _ = QFileDialog.getOpenFileName(
        main_window, 'Open IVUS File', '..', 'All files (*)', options=QFileDialog.Option.DontUseNativeDialog
    )
    if file_name:
        main_window.gating_display.fig.clear()
        plt.draw()
        # Probe for DICOM: only dcmread + pixel_array determine the format; parse_dicom runs outside
        # the format-detection except so its errors are never silently swallowed by the NIfTI path.
        _is_dicom = False
        try:
            main_window.dicom = dcm.dcmread(file_name, force=True, defer_size=256)
            main_window.images = main_window.dicom.pixel_array
            if 'Modality' not in main_window.dicom:
                raise ValueError("Not a valid DICOM: missing Modality tag")
            _is_dicom = True
        except Exception:
            main_window.dicom = None  # prevent garbage dataset from leaking into NIfTI path
        if _is_dicom:
            parse_dicom(main_window)
            if main_window.images.ndim == 4:  # 3 channel input
                if main_window.metadata['modality'] == 'OCT':
                    main_window.images_display = 1  # add only a flag value for RAM efficiency
                    main_window.images = convert_oct_to_gray(main_window.images)
                else:
                    main_window.images = main_window.images[:, :, :, 0]
        else:
            try:  # NIfTi
                img = sitk.ReadImage(file_name)
                main_window.images = sitk.GetArrayFromImage(img)
                main_window.file_name = os.path.basename(file_name).split('_')[0]

                if main_window.images.ndim == 4:  # RGB/OCT
                    # Scalar 4D NIfTI: SimpleITK reverses axis order so channels land at dim 0 → (3, F, H, W).
                    # Vector NIfTI (GetNumberOfComponentsPerPixel > 1): channels already last → (F, H, W, 3).
                    # convert_oct_to_gray expects channels-last, so transpose the scalar case.
                    if img.GetNumberOfComponentsPerPixel() == 1:
                        main_window.images = main_window.images.transpose(1, 2, 3, 0)
                    # Store uint8 RGB for colour display (mirrors dicom.pixel_array for DICOM OCT).
                    main_window.images_rgb = main_window.images.clip(0, 255).astype(np.uint8)
                    main_window.images_display = 1
                    main_window.images = convert_oct_to_gray(main_window.images)
                    parse_nifti_oct(main_window, img)
                else:
                    parse_nifti(main_window, img)

            except Exception:
                traceback.print_exc()
                ErrorMessage(
                    main_window, 'File is not a valid IVUS file and could not be loaded (DICOM or NIfTi supported)'
                )
                return None

        root, ext = os.path.splitext(file_name)
        if ext == '.gz':
            root = os.path.splitext(root)[0]
        main_window.file_name = root
        main_window.metadata['num_frames'] = main_window.images.shape[0]
        main_window.display_slider.blockSignals(True)
        main_window.display_slider.setMaximum(main_window.metadata['num_frames'] - 1)
        main_window.display_slider.blockSignals(False)

        num_frames = main_window.metadata['num_frames']
        success = read_contours(main_window, main_window.file_name)
        if success:
            # Fill any frames absent from the JSON with empty FrameData
            for i in range(num_frames):
                if i not in main_window.data:
                    main_window.data[i] = FrameData()
            main_window.segmentation = True
            main_window.gated_frames_dia = [
                frame for frame in range(num_frames) if main_window.data[frame].phase == 'D'
            ]
            main_window.gated_frames_sys = [
                frame for frame in range(num_frames) if main_window.data[frame].phase == 'S'
            ]
            main_window.gated_frames_oct = [
                frame for frame in range(num_frames) if main_window.data[frame].phase == 'T'
            ]
            main_window.gated_frames = main_window.gated_frames_dia
        else:  # initialise empty containers
            main_window.data = {i: FrameData() for i in range(num_frames)}
        main_window.display.set_data(main_window.images)

        main_window.image_displayed = True
        main_window.display_slider.setValue(main_window.metadata['num_frames'] - 1)
        main_window.right_half.update_for_modality()
    main_window.status_bar.showMessage(main_window.waiting_status)


def read_nifti_mask(main_window, contour_type=ContourType.LUMEN):
    """Read a NIfTI segmentation mask and populate main_window.data with contours.

    The mask label scheme matches contours_to_mask in save_as_nifti.py:
      1=lumen, 2=EEM wall, 3=calcium, 4=lipid, 5=macrophage, 7=branch.
    EEM contour is derived from the outer boundary of lumen+wall (labels 1+2).
    """
    if not main_window.image_displayed:
        ErrorMessage(main_window, 'Load an image before importing a mask')
        return

    # Map each ContourType to the label(s) that form its binary mask
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
        mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(file_name)).astype(np.uint8)
    except Exception:
        traceback.print_exc()
        ErrorMessage(main_window, 'Could not read NIfTI mask file')
        return

    num_frames = min(mask_arr.shape[0], main_window.metadata['num_frames'])
    mask_fn = _LABEL_MASK[contour_type]
    field_name = contour_type.value  # matches FrameData attribute names exactly
    # Lumen and EEM are single closed boundaries; plaques/branch can be multi-region
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

            # Mirror _close_current_spline: scale → smooth via SplineGeometry → downsample → unscale.
            # find_contours returns (row, col); xs=col, ys=row.
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
                sparse_contours.append(
                    [
                        [x / sf for x in downsampled[0]],
                        [y / sf for y in downsampled[1]],
                    ]
                )
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


def convert_oct_to_gray(oct_array):
    """
    Converts an RGB OCT array (Frames, H, W, 3) to Grayscale (Frames, H, W).
    """
    # Define the luminosity weights
    weights = np.array([0.299, 0.587, 0.114])

    # Use dot product to apply weights to the last dimension (the 3 color channels)
    # This effectively does: (R * 0.299) + (G * 0.587) + (B * 0.114)
    gray_oct = np.dot(oct_array[..., :3], weights)

    return gray_oct.astype(np.uint8)
