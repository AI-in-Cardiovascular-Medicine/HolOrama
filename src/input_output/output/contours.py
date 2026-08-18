import hashlib
import json
import os
import shutil
import tempfile
import threading
from dataclasses import asdict

import numpy as np
from loguru import logger

from pages.intravascular.popup_windows.message_boxes import ErrorMessage
from version import CONTOURS_VERSION_TAG


def write_contours(main_window, force: bool = True, blocking: bool | None = None) -> None:
    """Serialize main_window.runtime_data.frame_data_dct (Dict[int, FrameData]) to JSON.

    force=True  (default, used by Ctrl+S): always writes.
    force=False (used by auto-save): skips if content unchanged since last save.
    blocking:   write on the calling thread rather than a background one. Defaults to
                `force`; pass True whenever the caller may not outlive the write — the
                app closing or a page being replaced — because the background writer is
                a daemon thread and dies with the interpreter.
    """
    if not main_window.image_displayed:
        if force:
            ErrorMessage(main_window, 'Cannot write contours before reading input file.')
        return

    # main_window.file_name is already the extension-free stem (see read_image), and
    # read_contours globs '<stem>_contours*.json' against it. Splitting an extension off
    # it again would eat a second dot-separated segment, so a pullback named after a
    # dotted DICOM UID ('1.2.840.…604688.dcm') saved to a stem the loader never looks at
    # and every contour came back empty on the next open.
    out_path = f'{main_window.file_name}_contours_{CONTOURS_VERSION_TAG}.json'

    try:
        serializable = {str(i): asdict(frame) for i, frame in main_window.runtime_data.frame_data_dct.items()}
        serializable['gating_signal'] = main_window.runtime_data.gating_signal
        content = json.dumps(serializable, default=_to_serializable, indent=2)
    except Exception as e:
        logger.exception(f'Failed to serialize contours: {e}')
        return

    content_hash = hashlib.md5(content.encode()).hexdigest()
    if not force and getattr(main_window, '_last_contours_hash', None) == content_hash:
        return
    main_window._last_contours_hash = content_hash

    write_here = force if blocking is None else blocking
    if write_here:
        _write_to_disk(content, out_path)
    else:
        threading.Thread(target=_write_to_disk, args=(content, out_path), daemon=True).start()


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
