from pathlib import Path

import numpy as np
from loguru import logger
from PyQt6.QtWidgets import QApplication, QProgressDialog

from segmentation.onnx_inference import DEFAULT_MODEL_PATH, OnnxSegmenter, SegmentationUnavailable

__all__ = ['Predict', 'SegmentationUnavailable']


def resolve_model_path(model_file: str | None) -> Path:
    """config.yaml's segmentation.model_file, or the bundled model when it is empty. Configs
    written before the switch to ONNX still point at an nnU-Net model folder (or a Keras
    .h5); those can no longer be run, so they fall back to the bundled model as well."""
    if not model_file:
        return DEFAULT_MODEL_PATH
    path = Path(model_file)
    if path.suffix.lower() != '.onnx':
        logger.warning(f'segmentation.model_file={model_file!r} is not an .onnx model; using the bundled model')
        return DEFAULT_MODEL_PATH
    return path


class Predict:
    """Automatic lumen segmentation of a (frames, y, x) image stack. Thin Qt wrapper around
    OnnxSegmenter: picks the model and settings from config.yaml and shows a cancellable
    progress dialog when a main window is given."""

    def __init__(self, main_window, config=None) -> None:
        self.main_window = main_window
        config = main_window.config if config is None else config
        seg = config.segmentation
        self.segmenter = OnnxSegmenter(
            resolve_model_path(getattr(seg, 'model_file', '')),
            use_mirroring=getattr(seg, 'use_mirroring', True),
            tile_step_size=getattr(seg, 'tile_step_size', 0.5),
            batch_size=getattr(seg, 'batch_size', 2),
            num_threads=getattr(seg, 'num_threads', 0),
            device=getattr(seg, 'device', 'auto'),
        )

    def check_available(self) -> tuple[bool, str]:
        """(True, '') if the runtime and model load, else (False, user-facing reason)."""
        try:
            self.segmenter.load()
        except SegmentationUnavailable as e:
            return False, str(e)
        return True, ''

    def _progress_label(self, lower_limit: int, upper_limit: int) -> str:
        device = {'cuda': 'GPU (CUDA)', 'directml': 'GPU (DirectML)'}.get(self.segmenter.active_device, 'CPU')
        label = f'Segmenting frames {lower_limit + 1} to {upper_limit} on the {device}...'
        if self.segmenter.fallback_reason:
            label += f'\n\nGPU not used: {self.segmenter.fallback_reason}'
        return label

    def __call__(self, images: np.ndarray, lower_limit: int, upper_limit: int) -> np.ndarray | None:
        """Masks for frames [lower_limit, upper_limit) of images (other frames stay 0), or
        None if the user cancelled. Raises SegmentationUnavailable if it cannot run."""
        progress = None
        if self.main_window is not None:
            progress = QProgressDialog(
                self._progress_label(lower_limit, upper_limit),
                'Cancel',
                0,
                max(1, upper_limit - lower_limit),
                self.main_window,
            )
            progress.setWindowTitle('Automatic segmentation')
            progress.setMinimumDuration(0)
            progress.setModal(True)
            progress.setValue(0)
            QApplication.processEvents()

        def on_progress(done: int, total: int) -> bool:
            if progress is None:
                return True
            progress.setMaximum(max(1, total))
            progress.setValue(done)
            QApplication.processEvents()
            return not progress.wasCanceled()

        try:
            return self.segmenter.predict(images, range(lower_limit, upper_limit), on_progress)
        finally:
            if progress is not None:
                progress.close()
