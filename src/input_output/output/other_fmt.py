import numpy as np
from loguru import logger

from pages.intravascular.popup_windows.message_boxes import ErrorMessage


def save_gated_images(main_window) -> None:
    """Save gated/tagged frames as .npy arrays — one file per phase group that has frames:
    diastolic ('D') and systolic ('S') for IVUS gating, tagged ('T') for OCT. Tagged frames
    previously weren't exported at all: only the 'D'/'S' phases were collected."""
    if not main_window.image_displayed:
        ErrorMessage(main_window, 'Cannot save gated images before reading the input file.')
        return

    # Keyed by output suffix; iteration follows frame_data_dct order, so each array stacks
    # its frames in acquisition order (the same ordering the previous dia/sys split used).
    phase_to_group = {'D': 'diastolic', 'S': 'systolic', 'T': 'tagged'}
    groups: dict[str, list] = {name: [] for name in phase_to_group.values()}
    for i, frame in main_window.runtime_data.frame_data_dct.items():
        group = phase_to_group.get(frame.phase)
        if group is not None:
            groups[group].append(main_window.runtime_data.images[i])

    base = main_window.file_name  # already the extension-free stem; see write_contours
    saved = []
    for name, images in groups.items():
        if images:
            np.save(f'{base}_{name}.npy', np.array(images))
            saved.append(f'{len(images)} {name}')

    if saved:
        logger.info(f'Saved {", ".join(saved)} frames.')
    else:
        ErrorMessage(main_window, 'No gated or tagged frames to save.')
