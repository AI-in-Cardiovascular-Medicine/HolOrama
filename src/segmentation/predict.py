import numpy as np
from PyQt6.QtWidgets import QApplication, QProgressDialog

from domain.intravascular.segmentation_types import SegmentationSettings
from segmentation.onnx_inference import OnnxSegmenter, SegmentationUnavailable

__all__ = ['Predict', 'SegmentationUnavailable']


class Predict:
    """Automatic lumen segmentation of a (frames, y, x) image stack. Thin Qt wrapper around
    OnnxSegmenter: takes the settings from config.yaml and shows a cancellable progress
    dialog when a main window is given."""

    def __init__(self, main_window, config=None) -> None:
        self.main_window = main_window
        config = main_window.config if config is None else config
        self.segmenter = OnnxSegmenter(SegmentationSettings.from_config(config))
        self._progress: QProgressDialog | None = None

    def check_available(self) -> tuple[bool, str]:
        """(True, '') if the runtime and model load, else (False, user-facing reason)."""
        try:
            self.segmenter.load()
        except SegmentationUnavailable as e:
            return False, str(e)
        return True, ''

    def __call__(self, images: np.ndarray, lower_limit: int, upper_limit: int) -> np.ndarray | None:
        """Masks for frames [lower_limit, upper_limit) of images (other frames stay 0), or
        None if the user cancelled. Raises SegmentationUnavailable if it cannot run."""
        if self.main_window is not None:
            self._progress = QProgressDialog(
                self._progress_label(lower_limit, upper_limit),
                'Cancel',
                0,
                max(1, upper_limit - lower_limit),
                self.main_window,
            )
            self._progress.setWindowTitle('Automatic segmentation')
            self._progress.setMinimumDuration(0)
            self._progress.setModal(True)
            self._progress.setValue(0)
            QApplication.processEvents()
        try:
            return self.segmenter.predict(images, range(lower_limit, upper_limit), self._on_progress)
        finally:
            if self._progress is not None:
                self._progress.close()
                self._progress = None

    def _progress_label(self, lower_limit: int, upper_limit: int) -> str:
        backend = {'cuda': 'CUDA', 'directml': 'DirectML'}.get(self.segmenter.active_device)
        device = f'GPU ({backend}, {self.segmenter.active_precision})' if backend else 'CPU'
        label = f'Segmenting frames {lower_limit + 1} to {upper_limit} on the {device}...'
        if self.segmenter.fallback_reason:
            label += f'\n\nGPU not used: {self.segmenter.fallback_reason}'
        return label

    def _on_progress(self, done: int, total: int) -> bool:
        """predict()'s progress callback: False once the user pressed Cancel."""
        if self._progress is None:
            return True
        self._progress.setMaximum(max(1, total))
        self._progress.setValue(done)
        QApplication.processEvents()
        return not self._progress.wasCanceled()
