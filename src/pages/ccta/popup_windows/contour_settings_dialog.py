"""CCTA Contour Settings: the CCTA label presets, and the name and colour each gives a mask
value (see domain.ccta_presets). The preset bar and the saving are those of every preset
dialog (see gui.preset_dialog); a label is renamed here and nowhere else.
"""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtWidgets import QSpinBox

from domain.ccta_presets import MAX_LABEL, CctaLabelDef, CctaPreset
from gui.preset_dialog import PresetDialog, WorkingPreset
from input_output.preset_library import CCTA

_COLUMNS = ('Label', 'Colour', 'Name', '')
_COL_LABEL, _COL_COLOR, _COL_NAME, _COL_ACTIONS = range(len(_COLUMNS))


@dataclass
class _Row:
    label: int
    name: str
    color: str


class CctaContourSettingsDialog(PresetDialog):
    TITLE = 'CCTA Contour Settings'
    KIND = CCTA
    COLUMNS = _COLUMNS
    COLUMN_TIPS = (
        'The mask value the row names',
        'The colour the label is overlaid and rendered in',
        'Its name everywhere in the app',
        '',
    )
    NAME_COLUMN = _COL_NAME
    ADD_TEXT = '+ Add label'
    NEW_TIP = 'A new preset without labels'
    PRESET_TIP = 'The preset the CCTA page names and colours its labels with after OK'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMinimumWidth(560)

    def rows_of(self, preset) -> list:
        return [_Row(label=defn.label, name=defn.name, color=defn.color) for defn in preset.labels]

    def build(self, working: WorkingPreset) -> CctaPreset:
        return CctaPreset(
            name=working.name,
            labels=tuple(CctaLabelDef(label=row.label, name=row.name.strip(), color=row.color) for row in working.rows),
        )

    def new_row(self) -> _Row:
        return _Row(label=self._free_label(MAX_LABEL), name='New label', color=self._next_color())

    def fill_row(self, index: int, row, editable: bool) -> None:
        label = QSpinBox()
        label.setRange(1, MAX_LABEL)
        label.setValue(row.label)
        label.valueChanged.connect(lambda value, r=row: self._set(r, 'label', value))
        self._place(index, _COL_LABEL, label, editable)
        self._place(index, _COL_COLOR, self._color_swatch(row), editable)
        self._place(index, _COL_NAME, self._name_edit(row), editable)
