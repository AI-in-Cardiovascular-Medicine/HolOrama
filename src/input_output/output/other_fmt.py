import os

import numpy as np
from loguru import logger
from pages.intravascular.popup_windows.message_boxes import ErrorMessage


def save_gated_images(main_window) -> None:
    """Save diastolic and systolic frames as separate .npy arrays."""
    if not main_window.image_displayed:
        ErrorMessage(main_window, 'Cannot save gated images before reading the input file.')
        return

    diastolic, systolic = [], []
    for i, frame in main_window.runtime_data.frame_data_dct.items():
        if frame.phase == 'D':
            diastolic.append(main_window.runtime_data.images[i])
        elif frame.phase == 'S':
            systolic.append(main_window.runtime_data.images[i])

    base = os.path.splitext(main_window.file_name)[0]
    np.save(f'{base}_diastolic.npy', np.array(diastolic))
    np.save(f'{base}_systolic.npy', np.array(systolic))
    logger.info(f'Saved {len(diastolic)} diastolic and {len(systolic)} systolic frames.')
