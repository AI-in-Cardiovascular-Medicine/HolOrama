"""Popup asking for vmtkcenterlinesmoothing's parameters before Calculate Centerlines
runs, separately for the aorta and the RCA/LCA coronaries, since the aorta often needs
much heavier smoothing than the finer coronary branches."""

from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QSpinBox,
    QVBoxLayout,
)

from domain.ccta.centerline_types import AORTA_SMOOTHING, CORONARY_SMOOTHING, SmoothingParams


class CenterlineSmoothingDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle('Centerline Smoothing Parameters')

        layout = QVBoxLayout(self)
        self._ao_iterations, self._ao_factor = self._build_group(layout, 'Aorta', AORTA_SMOOTHING)
        self._cor_iterations, self._cor_factor = self._build_group(layout, 'RCA / LCA', CORONARY_SMOOTHING)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def aorta_params(self) -> SmoothingParams:
        return SmoothingParams(iterations=self._ao_iterations.value(), factor=self._ao_factor.value())

    def coronary_params(self) -> SmoothingParams:
        return SmoothingParams(iterations=self._cor_iterations.value(), factor=self._cor_factor.value())

    def _build_group(
        self, layout: QVBoxLayout, title: str, defaults: SmoothingParams
    ) -> tuple[QSpinBox, QDoubleSpinBox]:
        box = QGroupBox(title)
        form = QFormLayout(box)

        iterations_spin = QSpinBox()
        iterations_spin.setRange(0, 5000)
        iterations_spin.setValue(defaults.iterations)
        iterations_spin.setToolTip('vmtkcenterlinesmoothing -iterations: number of moving-average smoothing passes.')
        form.addRow('Smoothing iterations:', iterations_spin)

        factor_spin = QDoubleSpinBox()
        factor_spin.setRange(0.0, 1.0)
        factor_spin.setSingleStep(0.01)
        factor_spin.setDecimals(2)
        factor_spin.setValue(defaults.factor)
        factor_spin.setToolTip('vmtkcenterlinesmoothing -factor: smoothing strength per iteration.')
        form.addRow('Smoothing factor:', factor_spin)

        layout.addWidget(box)
        return iterations_spin, factor_spin
