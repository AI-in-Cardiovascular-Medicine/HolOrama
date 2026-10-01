"""Intravascular Contour Settings: the contour presets, and the contour types each defines.

One row per contour type — its mask label, colour, name, tools, the type it lies inside and
its layer in the mask — with the lumen and the EEM always the first two (see
domain.contour_presets). Every change stays in the dialog until OK, which writes the user
presets that changed, removes the deleted ones, and makes the preset chosen at the top the
active one. Built-in presets are read-only: duplicate one to change it.
"""

from __future__ import annotations

import copy
import itertools
import re
from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from domain.all_types import ContourType
from domain.colors import CATEGORICAL_PALETTE
from domain.contour_presets import (
    DEFAULT_PRESET_FILE,
    ContourPreset,
    ContourTypeDef,
    PresetError,
    ToolSet,
    load_preset,
)
from domain.io_types import RESERVED_CONTOUR_IDS
from input_output.preset_library import PresetEntry, load_library, save_user_preset, write_preset

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


@dataclass
class _WorkingPreset:
    name: str
    rows: list[_Row]
    builtin: bool
    path: Path | None  # the file it is kept in; None until it is first saved
    dirty: bool = False


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


def _working(entry: PresetEntry) -> _WorkingPreset:
    return _WorkingPreset(name=entry.name, rows=_rows_of(entry.preset), builtin=entry.builtin, path=entry.path)


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


def build_preset(working: _WorkingPreset) -> ContourPreset:
    """The ContourPreset `working` stands for, validated (raises PresetError).

    A type added in the dialog gets its id here, from its name. The lumen is always on top,
    so its layer is not edited but set just above everyone else's.
    """
    taken = {row.id for row in working.rows if row.id is not None}
    ids: dict[str, str] = {}
    for row in working.rows:
        if row.id is None:
            ids[row.key] = _new_id(row.name, taken)
            taken.add(ids[row.key])
        else:
            ids[row.key] = row.id
    others = [row.layer for row in working.rows if row.id != ContourType.LUMEN.value]
    lumen_layer = max(others, default=0) + 1
    types = tuple(
        ContourTypeDef(
            type=ContourType(ids[row.key]),
            name=row.name.strip(),
            label=row.label,
            color=row.color,
            tools=row.tools,
            layer=lumen_layer if row.id == ContourType.LUMEN.value else row.layer,
            inside=ContourType(ids[row.inside]) if row.inside in ids else None,
        )
        for row in working.rows
    )
    return ContourPreset(name=working.name, types=types)


class ContourSettingsDialog(QDialog):
    def __init__(self, parent=None, active_name: str | None = None, library: list[PresetEntry] | None = None):
        super().__init__(parent)
        self.setWindowTitle('Intravascular Contour Settings')
        self.setMinimumWidth(820)
        self.setMinimumHeight(460)

        entries = load_library() if library is None else library
        self._presets: list[_WorkingPreset] = [_working(entry) for entry in entries]
        self._removed_paths: list[Path] = []
        self._new_keys = itertools.count(1)
        self._selected: ContourPreset | None = None

        layout = QVBoxLayout(self)

        bar = QHBoxLayout()
        bar.addWidget(QLabel('Preset:'))
        self._preset_combo = QComboBox()
        self._preset_combo.setMinimumWidth(220)
        self._preset_combo.setToolTip('The preset the intravascular page annotates with after OK')
        self._preset_combo.currentIndexChanged.connect(self._load_table)
        bar.addWidget(self._preset_combo, 1)
        self._new_btn = self._bar_button(bar, 'New', 'A new preset with only the lumen and the EEM', self._new_preset)
        self._duplicate_btn = self._bar_button(bar, 'Duplicate', 'A copy of this preset to change', self._duplicate)
        self._rename_btn = self._bar_button(bar, 'Rename', 'Rename this preset', self._rename)
        self._delete_btn = self._bar_button(bar, 'Delete', 'Delete this preset', self._delete)
        self._import_btn = self._bar_button(bar, 'Import…', 'Add a preset from a JSON file', self._import)
        self._export_btn = self._bar_button(bar, 'Export…', 'Save this preset as a JSON file', self._export)
        layout.addLayout(bar)

        self._hint = QLabel('Built-in presets cannot be changed — duplicate this one to change it.')
        self._hint.setStyleSheet('color: #aaa')
        layout.addWidget(self._hint)

        self._table = QTableWidget(0, len(_COLUMNS))
        self._table.setHorizontalHeaderLabels(_COLUMNS)
        self._table.verticalHeader().setVisible(False)  # type: ignore[union-attr]
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        header = self._table.horizontalHeader()
        assert header is not None
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(_COL_NAME, QHeaderView.ResizeMode.Stretch)
        self._set_header_tips()
        layout.addWidget(self._table, 1)

        self._add_btn = QPushButton('+ Add contour type')
        self._add_btn.clicked.connect(self._add_row)
        layout.addWidget(self._add_btn, alignment=Qt.AlignmentFlag.AlignLeft)

        self._error = QLabel()
        self._error.setStyleSheet('color: #ff6b6b')
        self._error.setWordWrap(True)
        layout.addWidget(self._error)

        self._buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        names = [preset.name for preset in self._presets]
        self._refresh_presets(names.index(active_name) if active_name in names else 0)

    # -- the preset bar -------------------------------------------------------------------

    @staticmethod
    def _bar_button(bar: QHBoxLayout, text: str, tip: str, slot) -> QPushButton:
        button = QPushButton(text)
        button.setToolTip(tip)
        button.clicked.connect(slot)
        bar.addWidget(button)
        return button

    @property
    def _current(self) -> _WorkingPreset:
        return self._presets[self._preset_combo.currentIndex()]

    def _refresh_presets(self, select: int) -> None:
        self._preset_combo.blockSignals(True)
        self._preset_combo.clear()
        for preset in self._presets:
            self._preset_combo.addItem(f'{preset.name} (built-in)' if preset.builtin else preset.name)
        self._preset_combo.setCurrentIndex(select)
        self._preset_combo.blockSignals(False)
        self._load_table()

    def _ask_name(self, title: str, default: str, renaming: _WorkingPreset | None = None) -> str | None:
        """A preset name no other preset (than the one `renaming`) has, or None if the user
        gave up."""
        name = default
        while True:
            name, ok = QInputDialog.getText(self, title, 'Preset name:', QLineEdit.EchoMode.Normal, name)
            name = name.strip()
            if not ok or not name:
                return None
            if all(preset.name.casefold() != name.casefold() for preset in self._presets if preset is not renaming):
                return name
            QMessageBox.warning(self, title, f'There is already a preset called {name!r}.')

    def _add_preset(self, preset: _WorkingPreset) -> None:
        self._presets.append(preset)
        self._refresh_presets(len(self._presets) - 1)

    def _new_preset(self) -> None:
        name = self._ask_name('New Preset', 'New preset')
        if name is None:
            return
        fixed = [row for row in _rows_of(load_preset(DEFAULT_PRESET_FILE)) if row.fixed]
        self._add_preset(_WorkingPreset(name=name, rows=fixed, builtin=False, path=None, dirty=True))

    def _duplicate(self) -> None:
        source = self._current
        name = self._ask_name('Duplicate Preset', f'{source.name} copy')
        if name is None:
            return
        self._add_preset(
            _WorkingPreset(name=name, rows=copy.deepcopy(source.rows), builtin=False, path=None, dirty=True)
        )

    def _rename(self) -> None:
        name = self._ask_name('Rename Preset', self._current.name, renaming=self._current)
        if name is None:
            return
        self._current.name = name
        self._current.dirty = True
        self._refresh_presets(self._preset_combo.currentIndex())

    def _delete(self) -> None:
        preset = self._current
        answer = QMessageBox.question(self, 'Delete Preset', f'Delete the preset {preset.name!r}?')
        if answer != QMessageBox.StandardButton.Yes:
            return
        if preset.path is not None:
            self._removed_paths.append(preset.path)
        self._presets.remove(preset)
        self._refresh_presets(0)

    def _import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, 'Import Contour Preset', '', 'Contour presets (*.json)')
        if not path:
            return
        try:
            preset = load_preset(Path(path))
        except PresetError as exc:
            QMessageBox.warning(self, 'Import Contour Preset', str(exc))
            return
        name = preset.name
        if any(other.name.casefold() == name.casefold() for other in self._presets):
            name = self._ask_name('Import Contour Preset', f'{preset.name} (imported)') or ''
            if not name:
                return
        self._add_preset(_WorkingPreset(name=name, rows=_rows_of(preset), builtin=False, path=None, dirty=True))

    def _export(self) -> None:
        try:
            preset = build_preset(self._current)
        except PresetError as exc:
            QMessageBox.warning(self, 'Export Contour Preset', f'Fix this preset first: {exc}')
            return
        stem = re.sub(r'[^A-Za-z0-9]+', '_', preset.name).strip('_').lower() or 'preset'
        path, _ = QFileDialog.getSaveFileName(self, 'Export Contour Preset', f'{stem}.json', 'Contour presets (*.json)')
        if path:
            write_preset(preset, Path(path))

    # -- the table ------------------------------------------------------------------------

    def _set_header_tips(self) -> None:
        tips = (
            'The value the type is written into the exported mask as (1-255)',
            'The colour it is drawn and overlaid in',
            'Its name everywhere in the app',
            'What it can be drawn with; the brush comes with every spline type',
            'The type it lies inside: it is clipped to that one (and kept out of the lumen), and an open '
            'contour of it fills outwards from the arc to that type\'s boundary',
            'Where it sits in the mask: wherever two regions overlap, the higher layer shows. The lumen '
            'is always on top, except of what lies inside it',
            '',
        )
        for column, tip in enumerate(tips):
            item = self._table.horizontalHeaderItem(column)
            if item is not None:
                item.setToolTip(tip)

    def _load_table(self) -> None:
        preset = self._current
        editable = not preset.builtin
        self._hint.setVisible(preset.builtin)
        self._add_btn.setEnabled(editable)
        self._rename_btn.setEnabled(editable)
        self._delete_btn.setEnabled(editable)

        self._table.setRowCount(0)
        self._table.setRowCount(len(preset.rows))
        for index, row in enumerate(preset.rows):
            self._fill_row(index, row, editable)
        self._validate()

    def _fill_row(self, index: int, row: _Row, editable: bool) -> None:
        label = QSpinBox()
        label.setRange(1, 255)
        label.setValue(row.label)
        label.valueChanged.connect(lambda value, r=row: self._set(r, 'label', value))
        self._place(index, _COL_LABEL, label, editable)

        swatch = QPushButton()
        swatch.setFixedSize(36, 20)
        swatch.setStyleSheet(f'background-color: {row.color}; border: 1px solid #666;')
        swatch.clicked.connect(lambda _checked, r=row, s=swatch: self._pick_color(r, s))
        self._place(index, _COL_COLOR, swatch, editable)

        name = QLineEdit(row.name)
        name.textEdited.connect(lambda text, r=row: self._set(r, 'name', text))
        name.editingFinished.connect(self._refresh_inside_choices)
        self._place(index, _COL_NAME, name, editable)

        tools = QComboBox()
        # A saved type keeps what it is — a spline type stays a spline type and an angle an
        # angle — or contours already drawn with it would change their meaning.
        if row.fixed:
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

        if row.id == ContourType.LUMEN.value:
            top = QLabel('top')
            top.setToolTip('The lumen is always on top, except of what lies inside it')
            top.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._place(index, _COL_LAYER, top, True)
        else:
            layer = QSpinBox()
            layer.setRange(-99, 99)
            layer.setValue(row.layer)
            layer.valueChanged.connect(lambda value, r=row: self._set(r, 'layer', value))
            self._place(index, _COL_LAYER, layer, editable)

        actions = QWidget()
        box = QHBoxLayout(actions)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(2)
        if not row.fixed:
            for text, tip, slot in (
                ('▲', 'Move up: the order of the drop-downs, and which rows get a keyboard shortcut', -1),
                ('▼', 'Move down', 1),
                ('✕', 'Remove this contour type', 0),
            ):
                button = QPushButton(text)
                button.setFixedWidth(26)
                button.setToolTip(tip)
                if slot:
                    button.clicked.connect(lambda _checked, r=row, d=slot: self._move_row(r, d))
                else:
                    button.clicked.connect(lambda _checked, r=row: self._remove_row(r))
                box.addWidget(button)
        self._place(index, _COL_ACTIONS, actions, editable)

    def _place(self, index: int, column: int, widget: QWidget, enabled: bool) -> None:
        widget.setEnabled(enabled)
        self._table.setCellWidget(index, column, widget)

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

    def _refresh_inside_choices(self) -> None:
        """Follow renamed rows and changed tools in every Inside drop-down."""
        for index, row in enumerate(self._current.rows):
            combo = self._table.cellWidget(index, _COL_INSIDE)
            if isinstance(combo, QComboBox):
                self._fill_inside(combo, row)

    def _set(self, row: _Row, attribute: str, value) -> None:
        setattr(row, attribute, value)
        self._changed()

    def _set_tools(self, row: _Row, tools: ToolSet) -> None:
        row.tools = tools
        if tools is ToolSet.ANGLE:
            row.inside = None  # a sector is a wedge about the catheter: it lies in nothing
        elif tools is ToolSet.OPEN_CLOSED and row.inside is None:
            row.inside = ContourType.EEM.value  # an open contour needs a boundary to fill up to
        self._changed()
        self._load_table()

    def _pick_color(self, row: _Row, swatch: QPushButton) -> None:
        color = QColorDialog.getColor(QColor(row.color), self, f'Colour of {row.name}')
        if color.isValid():
            row.color = color.name()
            swatch.setStyleSheet(f'background-color: {row.color}; border: 1px solid #666;')
            self._changed()

    def _add_row(self) -> None:
        rows = self._current.rows
        labels = {row.label for row in rows}
        label = next(value for value in range(1, 256) if value not in labels)
        used = {QColor(row.color).name() for row in rows}
        palette = [f'#{r:02x}{g:02x}{b:02x}' for r, g, b in CATEGORICAL_PALETTE]
        color = next((hex_color for hex_color in palette if hex_color not in used), palette[len(rows) % len(palette)])
        rows.append(
            _Row(
                key=f'new:{next(self._new_keys)}',
                id=None,
                name='New type',
                label=label,
                color=color,
                tools=ToolSet.CLOSED,
                layer=max((row.layer for row in rows if row.id != ContourType.LUMEN.value), default=0) + 1,
            )
        )
        self._changed()
        self._load_table()

    def _move_row(self, row: _Row, delta: int) -> None:
        rows = self._current.rows
        index = rows.index(row)
        target = index + delta
        if target < len(_FIXED_IDS) or target >= len(rows):
            return  # the lumen and the EEM stay the first two rows
        rows[index], rows[target] = rows[target], rows[index]
        self._changed()
        self._load_table()

    def _remove_row(self, row: _Row) -> None:
        rows = self._current.rows
        rows.remove(row)
        for other in rows:
            if other.inside == row.key:
                other.inside = None
        self._changed()
        self._load_table()

    def _changed(self) -> None:
        self._current.dirty = True
        self._validate()

    def _validate(self) -> bool:
        try:
            build_preset(self._current)
        except PresetError as exc:
            self._error.setText(str(exc))
            valid = False
        else:
            self._error.setText('')
            valid = True
        ok = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok is not None:
            ok.setEnabled(valid)
        return valid

    # -- result ---------------------------------------------------------------------------

    def accept(self) -> None:
        built: list[tuple[_WorkingPreset, ContourPreset]] = []
        for index, preset in enumerate(self._presets):
            if preset.builtin or not preset.dirty:
                continue
            try:
                built.append((preset, build_preset(preset)))
            except PresetError:
                self._preset_combo.setCurrentIndex(index)  # shows the preset and what is wrong
                return
        try:
            for path in self._removed_paths:
                path.unlink(missing_ok=True)
            for preset, contour_preset in built:
                preset.path = save_user_preset(contour_preset, preset.path)
                preset.dirty = False
        except OSError as exc:
            QMessageBox.warning(self, 'Intravascular Contour Settings', f'Could not save the presets: {exc}')
            return
        self._removed_paths = []
        self._selected = build_preset(self._current)
        super().accept()

    def selected_preset(self) -> ContourPreset | None:
        """The preset chosen when OK was pressed, or None if the dialog was cancelled."""
        return self._selected
