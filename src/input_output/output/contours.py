import os
import json
import shutil
import tempfile
import threading
import hashlib

import numpy as np
from loguru import logger
from dataclasses import asdict

from version import CONTOURS_VERSION_TAG
from pages.intravascular.popup_windows.message_boxes import ErrorMessage


def write_contours(main_window, force: bool = True) -> None:
    """Serialize main_window.runtime_data.frame_data_dct (Dict[int, FrameData]) to JSON.

    force=True  (default, used by Ctrl+S): always writes synchronously.
    force=False (used by auto-save): skips if content unchanged since last save,
                                     otherwise writes on a background thread.
    """
    if not main_window.image_displayed:
        if force:
            ErrorMessage(main_window, 'Cannot write contours before reading input file.')
        return

    base = os.path.splitext(main_window.file_name)[0]
    out_path = f'{base}_contours_{CONTOURS_VERSION_TAG}.json'

    try:
        serializable = {str(i): asdict(frame) for i, frame in main_window.runtime_data.frame_data_dct.items()}
        serializable['gating_signal'] = main_window.runtime_data.gating_signal
        content = json.dumps(serializable, default=_to_serializable, indent=2)
    except Exception as e:
        logger.exception(f'Failed to serialize contours: {e}')
        return

    if not force:
        content_hash = hashlib.md5(content.encode()).hexdigest()
        if getattr(main_window, '_last_contours_hash', None) == content_hash:
            return
        main_window._last_contours_hash = content_hash
        threading.Thread(target=_write_to_disk, args=(content, out_path), daemon=True).start()
    else:
        main_window._last_contours_hash = hashlib.md5(content.encode()).hexdigest()
        _write_to_disk(content, out_path)


def _write_to_disk(content: str, out_path: str) -> None:
    out_dir = os.path.dirname(out_path) or '.'
    tmp_fd, tmp_path = tempfile.mkstemp(dir=out_dir, suffix='.tmp')
    try:
        with os.fdopen(tmp_fd, 'w') as f:
            f.write(content)
        shutil.move(tmp_path, out_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        logger.exception('Failed to write contours to disk')
    else:
        logger.info(f'Wrote contours to {out_path}')


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
