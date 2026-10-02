"""Intravascular Contour Settings: the contour presets, and the contour types each defines.

One row per contour type — its mask label, colour, name, tools, the type it lies inside and
its layer in the mask — with the lumen and the EEM always the first two (see
domain.contour_presets). The preset bar and the saving are those of every preset dialog
(see gui.preset_dialog).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from PyQt6.QtWidgets import QComboBox, QSpinBox

from domain.all_types import ContourType
from domain.contour_presets import (
    DEFAULT_PRESET_FILE,
    ContourPreset,
    ContourTypeDef,
    ToolSet,
    active_preset,
    load_preset,
)
from domain.io_types import RESERVED_CONTOUR_IDS
from gui.preset_dialog import PresetDialog, WorkingPreset
from input_output.preset_library import INTRAVASCULAR

_TOOL_LABELS = {
    ToolSet.CLOSED: 'Closed only',
    ToolSet.OPEN_CLOSED: 'Open and closed',
    ToolSet.ANGLE: 'Angle',
}
_SPLINE_TOOLS = (ToolSet.CLOSED, ToolSet.OPEN_CLOSED)
_FIXED_IDS = (ContourType.LUMEN.value, ContourType.EEM.value)
_COLUMNS = ('Label', 'Colour', 'Name', 'Tools', 'Inside', 'Layer', '')
_COL_LABEL, _COL_COLOR, _COL_NAME, _COL_TOOLS, _COL_INSIDE, _COL_LAYER, _COL_ACTIONS = range(len(_COLUMNS))
_NO_CONTAINER = '—'


@dataclass
class _Row:
    """One contour type as the dialog edits it."""

    key: str  # what other rows refer to it by: its id, or a placeholder until it has one
    id: str | None  # None for a type added in this dialog, which gets one when saved
    name: str
    label: int
    color: str
    tools: ToolSet
    layer: int
    inside: str | None = None  # the key of the row it lies inside

    @property
    def fixed(self) -> bool:
        return self.id in _FIXED_IDS


def _rows_of(preset: ContourPreset) -> list[_Row]:
    return [
        _Row(
            key=defn.type.value,
            id=defn.type.value,
            name=defn.name,
            label=defn.label,
            color=defn.color,
            tools=defn.tools,
            layer=defn.layer,
            inside=defn.inside.value if defn.inside is not None else None,
        )
        for defn in preset.types
    ]


def _new_id(name: str, taken: set[str]) -> str:
    """A contour type id for a type called `name`: its name, lower-cased with every run of
    other characters as one '_', made unique among `taken` and the reserved names."""
    stem = re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')
    if not stem or not stem[0].isalpha():
        stem = f'type_{stem}'.rstrip('_')
    candidate, suffix = stem, 2
    while candidate in taken or candidate in RESERVED_CONTOUR_IDS:
        candidate = f'{stem}_{suffix}'
        suffix += 1
    return candidate


def build_preset(working: WorkingPreset) -> ContourPreset:
    """The ContourPreset `working` stands for, validated (raises PresetError).

    A type added in the dialog gets its id here, from its name.
    """
    taken = {row.id for row in working.rows if row.id is not None}
    ids: dict[str, str] = {}
    for row in working.rows:
        if row.id is None:
            ids[row.key] = _new_id(row.name, taken)
            taken.add(ids[row.key])
        else:
            ids[row.key] = row.id
    types = tuple(
        ContourTypeDef(
            type=ContourType(ids[row.key]),
            name=row.name.strip(),
            label=row.label,
            color=row.color,
            tools=row.tools,
            layer=row.layer,
            inside=ContourType(ids[row.inside]) if row.inside in ids else None,
        )
        for row in working.rows
    )
    return ContourPreset(name=working.name, types=types)


class ContourSettingsDialog(PresetDialog):
    TITLE = 'Intravascular Contour Settings'
    KIND = INTRAVASCULAR
    COLUMNS = _COLUMNS
    COLUMN_TIPS = (
        'The value the type is written into the exported mask as (1-255)',
        'The colour it is drawn and overlaid in',
        'Its name everywhere in the app',
        'What it can be drawn with (brush comes with every spline type)',
        'The type it lies inside: it is clipped to that one (and kept out of the lumen), and an open '
        'contour of it fills outwards from the arc to that type\'s boundary',
        'Where it sits in the mask: wherever two regions overlap, the higher layer shows. A type '
        'lying inside another has to be above it',
        '',
    )
    NAME_COLUMN = _COL_NAME
    ADD_TEXT = '+ Add contour type'
    NEW_TIP = 'A new preset with only the lumen and the EEM'
    PRESET_TIP = 'The preset the intravascular page annotates with after OK'
    FIXED_ROWS = len(_FIXED_IDS)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMinimumWidth(820)

    def rows_of(self, preset) -> list:
        return _rows_of(preset)

    def build(self, working: WorkingPreset) -> ContourPreset:
        return build_preset(working)

    def new_preset_rows(self) -> list:
        return [row for row in _rows_of(load_preset(DEFAULT_PRESET_FILE)) if row.fixed]

    def draft_rows(self, draft) -> list:
        """A draft's rows, those the active preset has not as new ones: nothing is drawn with
        them yet, so they can still become anything — an angle, say — and take their id
        from the name they are given."""
        known = {defn.type.value for defn in active_preset().types}
        rows = _rows_of(draft)
        for row in rows:
            if row.id not in known:
                row.id = None
                row.key = f'new:{next(self._new_keys)}'
        return rows

    def new_row(self) -> _Row:
        return _Row(
            key=f'new:{next(self._new_keys)}',
            id=None,
            name='New type',
            label=self._free_label(255),
            color=self._next_color(),
            tools=ToolSet.CLOSED,
            layer=max((row.layer for row in self._current.rows), default=0) + 1,  # on top, until placed
        )

    def row_removed(self, row) -> None:
        for other in self._current.rows:
            if other.inside == row.key:
                other.inside = None

    def name_edited(self) -> None:
        """Follow renamed rows in every Inside drop-down."""
        for index, row in enumerate(self._current.rows):
            combo = self._table.cellWidget(index, _COL_INSIDE)
            if isinstance(combo, QComboBox):
                self._fill_inside(combo, row)

    def fill_row(self, index: int, row, editable: bool) -> None:
        label = QSpinBox()
        label.setRange(1, 255)
        label.setValue(row.label)
        label.valueChanged.connect(lambda value, r=row: self._set(r, 'label', value))
        self._place(index, _COL_LABEL, label, editable)
        self._place(index, _COL_COLOR, self._color_swatch(row), editable)
        self._place(index, _COL_NAME, self._name_edit(row), editable)

        tools = QComboBox()
        if row.fixed:  # A saved type keeps what it is or contours already drawn with it would change their meaning.
            choices: tuple[ToolSet, ...] = (ToolSet.CLOSED,)
        elif row.id is None:
            choices = tuple(ToolSet)
        else:
            choices = (ToolSet.ANGLE,) if row.tools is ToolSet.ANGLE else _SPLINE_TOOLS
        for choice in choices:
            tools.addItem(_TOOL_LABELS[choice], choice)
        tools.setCurrentIndex(choices.index(row.tools))
        tools.currentIndexChanged.connect(lambda _i, r=row, c=tools: self._set_tools(r, c.currentData()))
        self._place(index, _COL_TOOLS, tools, editable and len(choices) > 1)

        inside = QComboBox()
        self._place(index, _COL_INSIDE, inside, editable and not row.fixed and row.tools is not ToolSet.ANGLE)
        self._fill_inside(inside, row)
        inside.currentIndexChanged.connect(lambda _i, r=row, c=inside: self._set(r, 'inside', c.currentData()))

        layer = QSpinBox()
        layer.setRange(-99, 99)
        layer.setValue(row.layer)
        layer.valueChanged.connect(lambda value, r=row: self._set(r, 'layer', value))
        self._place(index, _COL_LAYER, layer, editable)

    def _fill_inside(self, combo: QComboBox, row: _Row) -> None:
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(_NO_CONTAINER, None)
        for other in self._current.rows:
            if other is not row and other.tools is not ToolSet.ANGLE:
                combo.addItem(other.name, other.key)
        found = combo.findData(row.inside)
        combo.setCurrentIndex(max(found, 0))
        combo.blockSignals(False)

    def _set_tools(self, row: _Row, tools: ToolSet) -> None:
        row.tools = tools
        if tools is ToolSet.ANGLE:
            row.inside = None  # a sector is a wedge about the catheter: it lies in nothing
        elif tools is ToolSet.OPEN_CLOSED and row.inside is None:
            row.inside = ContourType.EEM.value  # an open contour needs a boundary to fill up to
        self._changed()
        self._load_table()
