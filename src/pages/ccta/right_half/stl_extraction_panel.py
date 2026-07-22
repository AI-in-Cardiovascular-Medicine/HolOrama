# This combines aortic root, coronaries and allows to cut-off LVOT into one new combined mask, which can be exported as a STL for fluid dynamics
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)


def _separator() -> QFrame:
    sep = QFrame()
    sep.setFrameShape(QFrame.Shape.HLine)
    sep.setFrameShadow(QFrame.Shadow.Sunken)
    return sep


class StlExtractionPanel(QWidget):
    """Panel for extracting and exporting the aortic root with coronaries and LVOT,
    building the same cut geometry as an in-memory 3-D layer, and marking RCA/LCA
    outlet points on it for centerline computation."""

    line_draw_requested = pyqtSignal(int)  # 0 = axial LVOT, 1 = coronal LVOT, 2 = coronal aorta-top
    extract_requested = pyqtSignal(int, int, int, str)  # coronaries_label, aorta_label, lv_label, format
    build_cut_geometry_requested = pyqtSignal(int, int, int)  # coronaries_label, aorta_label, lv_label
    outlet_point_mode_requested = pyqtSignal(str)  # 'rca', 'lca', or '' to cancel
    clear_outlet_points_requested = pyqtSignal(str)  # 'rca' or 'lca'

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 8)
        root.setSpacing(4)

        title = QLabel('Aortic Root with Coronaries')
        title.setStyleSheet('font-weight: bold;')
        root.addWidget(title)
        root.addWidget(_separator())

        # Mask selectors
        self._coronaries_combo = QComboBox()
        self._aorta_combo = QComboBox()
        self._lv_combo = QComboBox()

        for text, combo in [
            ('Coronaries:', self._coronaries_combo),
            ('Aorta:', self._aorta_combo),
            ('LV:', self._lv_combo),
        ]:
            row = QHBoxLayout()
            lbl = QLabel(text)
            lbl.setFixedWidth(72)
            row.addWidget(lbl)
            row.addWidget(combo, 1)
            root.addLayout(row)
            combo.currentIndexChanged.connect(self._update_extract_btn)

        root.addWidget(_separator())

        # Cut planes: LVOT (2 lines) + aorta top. All three required — the aorta-top
        # plane is where the outlet centroid is measured, so without it there's no
        # outlet for the cut-geometry layer or the ao centerline target.
        root.addWidget(QLabel('Cut planes:'))
        self._line_status: list[QLabel] = []
        for i, text in enumerate(['LVOT cut line (axial)', 'LVOT cut line (coronal)', 'Aorta top cut (coronal)']):
            row = QHBoxLayout()
            btn = QPushButton(text)
            btn.clicked.connect(lambda _, idx=i: self.line_draw_requested.emit(idx))
            status = QLabel('○')
            status.setFixedWidth(18)
            status.setStyleSheet('color: #888; font-size: 14px;')
            row.addWidget(btn, 1)
            row.addWidget(status)
            root.addLayout(row)
            self._line_status.append(status)

        self._build_geometry_btn = QPushButton('Build Cut Geometry')
        self._build_geometry_btn.setEnabled(False)
        self._build_geometry_btn.setToolTip('Build the cut surface as a 3-D layer with inlet/outlet markers')
        self._build_geometry_btn.clicked.connect(self._on_build_cut_geometry)
        root.addWidget(self._build_geometry_btn)

        root.addWidget(_separator())

        # Outlet points (RCA / LCA), placed on the cut geometry in the 3-D view —
        # used as centerline targets by Calculate Centerlines.
        root.addWidget(QLabel('Outlet points:'))
        self._rca_btn, self._rca_count_lbl = self._outlet_row(root, 'Add RCA Outlet', 'rca')
        self._lca_btn, self._lca_count_lbl = self._outlet_row(root, 'Add LCA Outlet', 'lca')

        root.addWidget(_separator())

        # Export format + Extract && Export — kept at the very bottom of the panel.
        fmt_row = QHBoxLayout()
        fmt_row.addWidget(QLabel('Export as:'))
        self._fmt_group = QButtonGroup(self)
        for fmt, text in [('nifti', 'NIfTI'), ('stl', 'STL')]:
            rb = QRadioButton(text)
            rb.setProperty('fmt', fmt)
            self._fmt_group.addButton(rb)
            fmt_row.addWidget(rb)
        self._fmt_group.buttons()[0].setChecked(True)
        fmt_row.addStretch()
        root.addLayout(fmt_row)

        self._extract_btn = QPushButton('Extract && Export')
        self._extract_btn.setEnabled(False)
        self._extract_btn.setToolTip('Exports the smoothed cut geometry if Build Cut Geometry + Smooth were used (STL)')
        self._extract_btn.clicked.connect(self._on_extract)
        root.addWidget(self._extract_btn)

        self._lines_drawn = [False, False, False]

    def _outlet_row(self, root: QVBoxLayout, text: str, category: str) -> tuple[QPushButton, QLabel]:
        row = QHBoxLayout()
        btn = QPushButton(text)
        btn.setCheckable(True)
        btn.toggled.connect(lambda checked, cat=category: self._on_outlet_mode_toggled(cat, checked))
        count_lbl = QLabel('0 pts')
        count_lbl.setFixedWidth(46)
        count_lbl.setStyleSheet('color: #888;')
        clear_btn = QPushButton('Clear')
        clear_btn.setFixedWidth(50)
        clear_btn.clicked.connect(lambda: self.clear_outlet_points_requested.emit(category))
        row.addWidget(btn, 1)
        row.addWidget(count_lbl)
        row.addWidget(clear_btn)
        root.addLayout(row)
        return btn, count_lbl

    def _on_outlet_mode_toggled(self, category: str, checked: bool) -> None:
        # Mutually exclusive: turning one on turns the other off.
        other_btn = self._lca_btn if category == 'rca' else self._rca_btn
        if checked and other_btn.isChecked():
            other_btn.blockSignals(True)
            other_btn.setChecked(False)
            other_btn.blockSignals(False)
        self.outlet_point_mode_requested.emit(category if checked else '')

    def set_outlet_point_count(self, category: str, count: int) -> None:
        lbl = self._rca_count_lbl if category == 'rca' else self._lca_count_lbl
        lbl.setText(f'{count} pts')

    def reset_outlet_mode(self) -> None:
        """Uncheck both outlet-point toggle buttons without re-emitting their signals.
        Called by page.py when a mode request is rejected (e.g. cut geometry not
        built yet), so the buttons don't stay checked while picking is actually off."""
        for btn in (self._rca_btn, self._lca_btn):
            btn.blockSignals(True)
            btn.setChecked(False)
            btn.blockSignals(False)

    def set_building(self, building: bool) -> None:
        """Toggle the Build Cut Geometry button's busy state while a build is running."""
        self._build_geometry_btn.setText('Building…' if building else 'Build Cut Geometry')
        if building:
            self._build_geometry_btn.setEnabled(False)
        else:
            self._update_extract_btn()  # restore normal label/line-count gating

    def set_labels(self, labels: list[int], names: dict[int, str]) -> None:
        """Populate the mask-selector dropdowns."""
        for combo in (self._coronaries_combo, self._aorta_combo, self._lv_combo):
            combo.blockSignals(True)
            combo.clear()
            for lv in labels:
                combo.addItem(names.get(lv, f'Label {lv}'), userData=lv)
            combo.blockSignals(False)
        self._update_extract_btn()

    def set_selected_labels(self, cor: int, aorta: int, lv: int) -> None:
        """Restore previously chosen mask selections (used when auto-loading a saved
        cut state). No-op per combo if that label isn't present anymore."""
        for combo, label in (
            (self._coronaries_combo, cor),
            (self._aorta_combo, aorta),
            (self._lv_combo, lv),
        ):
            idx = combo.findData(label)
            if idx >= 0:
                combo.setCurrentIndex(idx)

    def update_label_name(self, label: int, name: str) -> None:
        """Sync a renamed label across all dropdowns."""
        for combo in (self._coronaries_combo, self._aorta_combo, self._lv_combo):
            for i in range(combo.count()):
                if combo.itemData(i) == label:
                    combo.setItemText(i, name)
                    break

    def set_line_drawn(self, index: int) -> None:
        """Mark a cut line as complete (called by page.py after line_drawn signal)."""
        self._lines_drawn[index] = True
        self._line_status[index].setText('●')
        self._line_status[index].setStyleSheet('color: #44cc44; font-size: 14px;')
        self._update_extract_btn()

    def _update_extract_btn(self) -> None:
        labels_ok = all(c.count() > 0 for c in (self._coronaries_combo, self._aorta_combo, self._lv_combo))
        ready = labels_ok and all(self._lines_drawn)
        self._extract_btn.setEnabled(ready)
        self._build_geometry_btn.setEnabled(ready)

    def _on_extract(self) -> None:
        cor, aorta, lv = self._selected_labels()
        fmt = next(
            (btn.property('fmt') for btn in self._fmt_group.buttons() if btn.isChecked()),
            'nifti',
        )
        self.extract_requested.emit(cor, aorta, lv, fmt)

    def _on_build_cut_geometry(self) -> None:
        cor, aorta, lv = self._selected_labels()
        self.build_cut_geometry_requested.emit(cor, aorta, lv)

    def _selected_labels(self) -> tuple[int, int, int]:
        return (
            self._coronaries_combo.currentData(),
            self._aorta_combo.currentData(),
            self._lv_combo.currentData(),
        )
