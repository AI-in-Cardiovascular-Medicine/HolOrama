from types import SimpleNamespace

from PyQt6.QtCore import QSize, Qt, QTimer
from PyQt6.QtWidgets import QCheckBox, QPushButton, QSplitter, QTableWidget

from domain.colors import DIASTOLE_COLOR, SYSTOLE_COLOR
from domain.runtime_types import RuntimeData
from signal_processing.gating_plot import GatingPlot
from input_output.output.contours import write_contours
from pages.intravascular.brush_panel import BrushSettingsPopup
from pages.intravascular.left_half.display import Display
from pages.intravascular.left_half.left_half import LeftHalf
from pages.intravascular.right_half.gating_display import GatingDisplay
from pages.intravascular.right_half.longitudinal_view import LongitudinalView
from pages.intravascular.right_half.right_half import RightHalf
from pages.intravascular.utils.oct_plot import OCTPlot
from pages.intravascular.utils.slider import Communicate, Slider
from segmentation.predict import Predict

PENDING_SAVE_DELAY_MS = 1500  # quiet time after an edit before the pending changes are written


class IntravascularPage(QSplitter):
    def __init__(self, config: SimpleNamespace, menu_bar, status_bar) -> None:
        super().__init__(Qt.Orientation.Horizontal)
        self.config: SimpleNamespace = config
        self.menu_bar = menu_bar
        self.status_bar = status_bar

        self.file_name: str | None = None
        self.gating_plot: GatingPlot = GatingPlot(self)
        self.predictor: Predict = Predict(self)
        self.image_displayed: bool = False
        self.segmentation: bool = False
        self.contours_drawn: bool = False
        self.hide_contours: bool = False
        self.hide_special_points: bool = False
        self.colormap_enabled: bool = False
        self.runtime_data: RuntimeData = RuntimeData()
        self.diastole_color: tuple[int, int, int] = DIASTOLE_COLOR
        self.diastole_color_plt: tuple[float, ...] = tuple(x / 255 for x in self.diastole_color)
        self.systole_color: tuple[int, int, int] = SYSTOLE_COLOR
        self.systole_color_plt: tuple[float, ...] = tuple(x / 255 for x in self.systole_color)
        self.waiting_status: str = 'Waiting for user input...'
        self.small_display = None
        self.results_plot = None

        self._init_ui()

    def _init_ui(self) -> None:
        self.file_name = "default_file_name"
        self.metadata_table: QTableWidget = QTableWidget()

        self.status_bar.showMessage(self.waiting_status)

        self.display: Display = Display(self)
        self.display_frame_comms: Communicate = Communicate()
        self.display_frame_comms.updateBW[int].connect(self.display.set_frame)
        self.display_slider: Slider = Slider(self, Qt.Orientation.Horizontal)
        self.hide_contours_box: QCheckBox = QCheckBox('&Hide Contours')
        self.hide_contours_box.setChecked(False)
        self.hide_special_points_box: QCheckBox = QCheckBox('&Hide Metrics')
        self.hide_special_points_box.setChecked(False)
        self.mask_mode_box: QCheckBox = QCheckBox('&Mask mode')
        self.mask_mode_box.setChecked(False)

        self.longitudinal_view: LongitudinalView = LongitudinalView(self)
        self.brush_settings_popup: BrushSettingsPopup = BrushSettingsPopup(self)
        self.brush_settings_popup._radius_slider.valueChanged.connect(
            lambda: self.display._update_brush_cursor() if self.display._brush_active else None
        )

        self.left_half: LeftHalf = LeftHalf(self)
        self.addWidget(self.left_half())
        self.right_half: RightHalf = RightHalf(self)
        self.addWidget(self.right_half())
        self.setChildrenCollapsible(False)

        # Two save timers with different jobs. The debounce fires shortly after an edit
        # so a change is on disk while the user is still looking at it; the periodic one
        # re-serializes everything and writes if the content moved, catching any edit that
        # forgot to flag itself.
        self._pending_save_timer: QTimer = QTimer(self)
        self._pending_save_timer.setSingleShot(True)
        self._pending_save_timer.setInterval(PENDING_SAVE_DELAY_MS)
        self._pending_save_timer.timeout.connect(self._save_pending_changes)

        self._autosave_timer: QTimer = QTimer(self)
        self._autosave_timer.timeout.connect(self.auto_save)
        self._autosave_timer.start(self.config.save.autosave_interval)

    # Each right half owns the widgets only its modality shows (see right_half.py).
    # They are still addressed as main_window.<name> from everywhere else, so the page
    # forwards the names whose readers live outside the half that builds them.

    @property
    def diastolic_frame_box(self) -> QCheckBox:
        return self.right_half.ivus.diastolic_frame_box

    @property
    def systolic_frame_box(self) -> QCheckBox:
        return self.right_half.ivus.systolic_frame_box

    @property
    def use_diastolic_button(self) -> QPushButton:
        return self.right_half.ivus.use_diastolic_button

    @property
    def gating_display(self) -> GatingDisplay:
        return self.right_half.ivus.gating_display

    @property
    def oct_plot(self) -> OCTPlot:
        return self.right_half.oct.oct_plot

    def sizeHint(self) -> QSize:
        return QSize(0, 0)

    def minimumSizeHint(self) -> QSize:
        return QSize(0, 0)

    def style(self):
        from PyQt6.QtWidgets import QApplication

        return QApplication.style()

    def close(self):
        top = self.window()
        if top is not None and top is not self:
            top.close()

    def auto_save(self) -> None:
        if self.image_displayed:
            write_contours(self, force=False)

    def save_contours_soon(self) -> None:
        """Note that frame data changed and save once the edits stop coming.

        Called by every contour-changing operation (directly, or via push_contour_snapshot
        / Display.update_display). Restarting the timer on each edit means a burst of edits
        costs one write, and serializing the whole pullback — ~0.1-0.3 s — never lands in
        the middle of an interaction.
        """
        self.runtime_data.mark_unsaved()
        if self.image_displayed:
            self._pending_save_timer.start()

    def _save_pending_changes(self) -> None:
        if self.image_displayed and self.runtime_data.unsaved_changes:
            write_contours(self, force=False)
            self.runtime_data.unsaved_changes = False

    def flush_contours(self) -> None:
        """Write outstanding changes now — before this page is replaced or the app closes.

        Blocking on purpose: the background writer is a daemon thread, so a write started
        here would be killed on the way out. Unchanged content still costs nothing (the
        hash check in write_contours skips the write), and the hash is checked rather than
        the flag so an edit that never flagged itself is saved too.
        """
        self._pending_save_timer.stop()
        if self.image_displayed:
            write_contours(self, force=False, blocking=True)
            self.runtime_data.unsaved_changes = False

    def reset_state(self) -> None:
        self._pending_save_timer.stop()
        if self.results_plot is not None:
            self.results_plot.close()
        if self.small_display is not None:
            self.small_display.close()
            self.small_display = None
        self.display.reset()
        self.oct_plot.reset()
        self.file_name = None
        self.image_displayed = False
        self.segmentation = False
        self.contours_drawn = False
        self.hide_contours = False
        self.hide_special_points = False
        self.colormap_enabled = False
        self.runtime_data = RuntimeData()
        self.status_bar.showMessage(self.waiting_status)
