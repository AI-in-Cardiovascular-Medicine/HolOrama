"""CCTA Contour Settings: the CCTA label presets, and the name and colour each gives a mask
value (see domain.ccta_presets). The preset bar and the saving are those of every preset
dialog (see gui.preset_dialog); a label is renamed here and nowhere else.
"""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtWidgets import QMessageBox, QSpinBox, QWidget

from domain.ccta_presets import MAX_LABEL, CctaLabelDef, CctaPreset, active_ccta_preset, draft_for
from gui.preset_dialog import PresetDialog, WorkingPreset
from input_output.preset_library import CCTA

_COLUMNS = ('Label', 'Colour', 'Name', '')
_COL_LABEL, _COL_COLOR, _COL_NAME, _COL_ACTIONS = range(len(_COLUMNS))


def ask_for_draft(parent: QWidget, labels: list[int]) -> CctaPreset | None:
    """If the active preset lacks some of a mask's `labels`, ask whether to make a new preset
    with a row for every one of them, each in a colour of its own, to name: that draft if
    so, otherwise None."""
    preset = active_ccta_preset()
    missing = [label for label in labels if preset.get(label) is None]
    if not missing:
        return None
    shown = ', '.join(str(label) for label in missing[:10]) + (', …' if len(missing) > 10 else '')
    reply = QMessageBox.question(
        parent,
        'Labels Without a Name',
        f'This mask holds {len(labels)} labels, {len(missing)} of which the preset '
        f'{preset.name!r} does not name ({shown}).\n\n'
        f'Create a new preset with a row for each of the {len(labels)} labels?',
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.Yes,
    )
    if reply != QMessageBox.StandardButton.Yes:
        return None
    return draft_for(f'{len(labels)} labels', labels)


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
