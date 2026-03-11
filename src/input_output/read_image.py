import os

import pydicom as dcm
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from PyQt6.QtWidgets import QFileDialog

from gui.popup_windows.message_boxes import ErrorMessage
from input_output.metadata import parse_dicom
from input_output.contours_io import read_contours, FrameData


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
        try:  # DICOM
            main_window.dicom = dcm.read_file(file_name, force=True)
            main_window.images = main_window.dicom.pixel_array
            parse_dicom(main_window)
            if main_window.images.ndim == 4:  # 3 channel input
                if main_window.metadata['modality'] == 'OCT':
                    main_window.images_display = 1  # add only a flag value for RAM efficiency
                    main_window.images = convert_oct_to_gray(main_window.images)
                else:
                    main_window.images = main_window.images[:, :, :, 0]
        except AttributeError:
            try:  # NIfTi
                img = sitk.ReadImage(file_name)
                main_window.images = sitk.GetArrayFromImage(img)
                # main_window.file_name = main_window.file_name.split('_')[0]  # remove _img.nii suffix
                main_window.file_name = os.path.basename(file_name).split('_')[0]
                # TODO: Do the same as parse_dicom here
            except:
                ErrorMessage(
                    main_window, 'File is not a valid IVUS file and could not be loaded (DICOM or NIfTi supported)'
                )
                return None

        main_window.file_name = os.path.splitext(file_name)[0]  # remove file extension
        main_window.metadata['num_frames'] = main_window.images.shape[0]
        main_window.display_slider.setMaximum(main_window.metadata['num_frames'] - 1)

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
            main_window.gated_frames = main_window.gated_frames_dia
        else:  # initialise empty containers
            main_window.data = {i: FrameData() for i in range(num_frames)}
        main_window.display.set_data(main_window.images)

        main_window.image_displayed = True
        main_window.display_slider.setValue(main_window.metadata['num_frames'] - 1)
    main_window.status_bar.showMessage(main_window.waiting_status)


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
