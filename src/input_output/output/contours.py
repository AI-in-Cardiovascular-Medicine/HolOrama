import os
import json
import shutil
import tempfile

import numpy as np
from loguru import logger
from dataclasses import asdict

from version import version_file_str
from gui.popup_windows.message_boxes import ErrorMessage

def write_contours(main_window) -> None:
    """Serialize main_window.data (Dict[int, FrameData]) to JSON."""
    if not main_window.image_displayed:
        ErrorMessage(main_window, 'Cannot write contours before reading input file.')
        return

    base = os.path.splitext(main_window.file_name)[0]
    out_path = f'{base}_contours_{version_file_str}.json'

    try:
        serializable = {str(i): asdict(frame) for i, frame in main_window.data.items()}
        serializable['gating_signal'] = main_window.gating_signal
        out_dir = os.path.dirname(out_path) or '.'
        tmp_fd, tmp_path = tempfile.mkstemp(dir=out_dir, suffix='.tmp')
        try:
            with os.fdopen(tmp_fd, 'w') as f:
                json.dump(serializable, f, default=_to_serializable, indent=2)
            shutil.move(tmp_path, out_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        logger.info(f'Wrote contours to {out_path}')
    except Exception as e:
        logger.exception(f'Failed to write contours: {e}')


def _to_serializable(obj):
    """Fallback serializer for json.dump to handle numpy types."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    try:
        return str(obj)
    except Exception:
        return None
