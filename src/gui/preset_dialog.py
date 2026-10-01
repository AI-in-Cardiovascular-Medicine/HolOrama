"""The dialog every kind of preset is edited in (Intravascular Contour Settings, CCTA Contour
Settings): a preset bar — which preset, New, Duplicate, Rename, Delete, Import, Export — over
a table with one row per entry of the chosen preset.

Every change stays in the dialog until OK, which writes the user presets that changed,
removes the deleted ones, and returns the preset chosen at the top for the caller to make
active. Built-in presets are read-only: duplicate one to change it. What a row holds, and
how a preset is built from the rows, is up to the subclass (see the hooks at the bottom).
"""

from __future__ import annotations

import copy
import itertools
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from domain.colors import CATEGORICAL_PALETTE
from domain.contour_presets import PresetError
from input_output.preset_library import PresetEntry, PresetKind, load_library, save_user_preset, write_preset


@dataclass
class WorkingPreset:
    name: str
    rows: list
    builtin: bool
    path: Path | None  # the file it is kept in; None until it is first saved
    dirty: bool = False


class PresetDialog(QDialog):
    TITLE = ''
    KIND: PresetKind
    COLUMNS: tuple[str, ...] = ()
    COLUMN_TIPS: tuple[str, ...] = ()
    NAME_COLUMN = 0
    ADD_TEXT = '+ Add'
    NEW_TIP = 'A new preset'
    PRESET_TIP = 'The preset to use after OK'
    FIXED_ROWS = 0  # how many leading rows stay where they are

    def __init__(
        self,
        parent=None,
        active_name: str | None = None,
        library: list[PresetEntry] | None = None,
        draft=None,
    ):
        """`draft`: a preset to open on as a new one, not saved until OK (its name made
        unique among the others first)."""
        super().__init__(parent)
        self.setWindowTitle(self.TITLE)
        self.setMinimumHeight(460)

        entries = load_library(self.KIND) if library is None else library
        self._presets: list[WorkingPreset] = [
            WorkingPreset(name=entry.name, rows=self.rows_of(entry.preset), builtin=entry.builtin, path=entry.path)
            for entry in entries
        ]
        self._removed_paths: list[Path] = []
        self._new_keys = itertools.count(1)
        self._selected: Any = None

        layout = QVBoxLayout(self)

        bar = QHBoxLayout()
        bar.addWidget(QLabel('Preset:'))
        self._preset_combo = QComboBox()
        self._preset_combo.setMinimumWidth(220)
        self._preset_combo.setToolTip(self.PRESET_TIP)
        self._preset_combo.currentIndexChanged.connect(self._load_table)
        bar.addWidget(self._preset_combo, 1)
        self._new_btn = self._bar_button(bar, 'New', self.NEW_TIP, self._new_preset)
        self._duplicate_btn = self._bar_button(bar, 'Duplicate', 'A copy of this preset to change', self._duplicate)
        self._rename_btn = self._bar_button(bar, 'Rename', 'Rename this preset', self._rename)
        self._delete_btn = self._bar_button(bar, 'Delete', 'Delete this preset', self._delete)
        self._import_btn = self._bar_button(bar, 'Import…', 'Add a preset from a JSON file', self._import)
        self._export_btn = self._bar_button(bar, 'Export…', 'Save this preset as a JSON file', self._export)
        layout.addLayout(bar)

        self._hint = QLabel('Built-in presets cannot be changed — duplicate this one to change it.')
        self._hint.setStyleSheet('color: #aaa')
        layout.addWidget(self._hint)

        self._table = QTableWidget(0, len(self.COLUMNS))
        self._table.setHorizontalHeaderLabels(self.COLUMNS)
        self._table.verticalHeader().setVisible(False)  # type: ignore[union-attr]
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        header = self._table.horizontalHeader()
        assert header is not None
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self.NAME_COLUMN, QHeaderView.ResizeMode.Stretch)
        for column, tip in enumerate(self.COLUMN_TIPS):
            item = self._table.horizontalHeaderItem(column)
            if item is not None:
                item.setToolTip(tip)
        layout.addWidget(self._table, 1)

        self._add_btn = QPushButton(self.ADD_TEXT)
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

        if draft is not None:
            self._presets.append(
                WorkingPreset(
                    name=self._unique_name(draft.name), rows=self.rows_of(draft), builtin=False, path=None, dirty=True
                )
            )
            active_name = self._presets[-1].name
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
    def _current(self) -> WorkingPreset:
        return self._presets[self._preset_combo.currentIndex()]

    def _refresh_presets(self, select: int) -> None:
        self._preset_combo.blockSignals(True)
        self._preset_combo.clear()
        for preset in self._presets:
            self._preset_combo.addItem(f'{preset.name} (built-in)' if preset.builtin else preset.name)
        self._preset_combo.setCurrentIndex(select)
        self._preset_combo.blockSignals(False)
        self._load_table()

    def _ask_name(self, title: str, default: str, renaming: WorkingPreset | None = None) -> str | None:
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

    def _unique_name(self, name: str) -> str:
        taken = {preset.name.casefold() for preset in self._presets}
        candidate, suffix = name, 2
        while candidate.casefold() in taken:
            candidate = f'{name} ({suffix})'
            suffix += 1
        return candidate

    def _add_preset(self, preset: WorkingPreset) -> None:
        self._presets.append(preset)
        self._refresh_presets(len(self._presets) - 1)

    def _new_preset(self) -> None:
        name = self._ask_name('New Preset', 'New preset')
        if name is None:
            return
        self._add_preset(WorkingPreset(name=name, rows=self.new_preset_rows(), builtin=False, path=None, dirty=True))

    def _duplicate(self) -> None:
        source = self._current
        name = self._ask_name('Duplicate Preset', f'{source.name} copy')
        if name is None:
            return
        self._add_preset(
            WorkingPreset(name=name, rows=copy.deepcopy(source.rows), builtin=False, path=None, dirty=True)
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
        path, _ = QFileDialog.getOpenFileName(self, 'Import Preset', '', 'Presets (*.json)')
        if not path:
            return
        try:
            preset = self.KIND.load(Path(path))
        except PresetError as exc:
            QMessageBox.warning(self, 'Import Preset', str(exc))
            return
        name = preset.name
        if any(other.name.casefold() == name.casefold() for other in self._presets):
            name = self._ask_name('Import Preset', f'{preset.name} (imported)') or ''
            if not name:
                return
        self._add_preset(WorkingPreset(name=name, rows=self.rows_of(preset), builtin=False, path=None, dirty=True))

    def _export(self) -> None:
        try:
            preset = self.build(self._current)
        except PresetError as exc:
            QMessageBox.warning(self, 'Export Preset', f'Fix this preset first: {exc}')
            return
        stem = re.sub(r'[^A-Za-z0-9]+', '_', preset.name).strip('_').lower() or 'preset'
        path, _ = QFileDialog.getSaveFileName(self, 'Export Preset', f'{stem}.json', 'Presets (*.json)')
        if path:
            write_preset(preset, Path(path))

    # -- the table ------------------------------------------------------------------------

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
            self.fill_row(index, row, editable)
            self._place(index, len(self.COLUMNS) - 1, self._actions(row), editable)
        self._validate()

    def _place(self, index: int, column: int, widget: QWidget, enabled: bool) -> None:
        widget.setEnabled(enabled)
        self._table.setCellWidget(index, column, widget)

    def _color_swatch(self, row) -> QPushButton:
        swatch = QPushButton()
        swatch.setFixedSize(36, 20)
        swatch.setStyleSheet(f'background-color: {row.color}; border: 1px solid #666;')
        swatch.clicked.connect(lambda _checked: self._pick_color(row, swatch))
        return swatch

    def _name_edit(self, row) -> QLineEdit:
        edit = QLineEdit(row.name)
        edit.textEdited.connect(lambda text: self._set(row, 'name', text))
        edit.editingFinished.connect(self.name_edited)
        return edit

    def _actions(self, row) -> QWidget:
        actions = QWidget()
        box = QHBoxLayout(actions)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(2)
        if self._current.rows.index(row) < self.FIXED_ROWS:
            return actions
        for text, tip, delta in (
            ('▲', 'Move up: the order of the drop-downs, and which rows get a keyboard shortcut', -1),
            ('▼', 'Move down', 1),
            ('✕', 'Remove this row', 0),
        ):
            button = QPushButton(text)
            button.setFixedWidth(26)
            button.setToolTip(tip)
            if delta:
                button.clicked.connect(lambda _checked, d=delta: self._move_row(row, d))
            else:
                button.clicked.connect(lambda _checked: self._remove_row(row))
            box.addWidget(button)
        return actions

    def _set(self, row, attribute: str, value) -> None:
        setattr(row, attribute, value)
        self._changed()

    def _pick_color(self, row, swatch: QPushButton) -> None:
        color = QColorDialog.getColor(QColor(row.color), self, f'Colour of {row.name}')
        if color.isValid():
            row.color = color.name()
            swatch.setStyleSheet(f'background-color: {row.color}; border: 1px solid #666;')
            self._changed()

    def _next_color(self) -> str:
        """The first palette colour no row of the current preset has yet."""
        rows = self._current.rows
        used = {QColor(row.color).name() for row in rows}
        palette = [f'#{r:02x}{g:02x}{b:02x}' for r, g, b in CATEGORICAL_PALETTE]
        return next((color for color in palette if color not in used), palette[len(rows) % len(palette)])

    def _free_label(self, maximum: int) -> int:
        labels = {row.label for row in self._current.rows}
        return next(value for value in range(1, maximum + 1) if value not in labels)

    def _add_row(self) -> None:
        self._current.rows.append(self.new_row())
        self._changed()
        self._load_table()

    def _move_row(self, row, delta: int) -> None:
        rows = self._current.rows
        index = rows.index(row)
        target = index + delta
        if target < self.FIXED_ROWS or target >= len(rows):
            return
        rows[index], rows[target] = rows[target], rows[index]
        self._changed()
        self._load_table()

    def _remove_row(self, row) -> None:
        self._current.rows.remove(row)
        self.row_removed(row)
        self._changed()
        self._load_table()

    def _changed(self) -> None:
        self._current.dirty = True
        self._validate()

    def _validate(self) -> bool:
        try:
            self.build(self._current)
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
        built: list[tuple[WorkingPreset, Any]] = []
        for index, preset in enumerate(self._presets):
            if preset.builtin or not preset.dirty:
                continue
            try:
                built.append((preset, self.build(preset)))
            except PresetError:
                self._preset_combo.setCurrentIndex(index)  # shows the preset and what is wrong
                return
        try:
            for path in self._removed_paths:
                path.unlink(missing_ok=True)
            for preset, result in built:
                preset.path = save_user_preset(result, preset.path, self.KIND)
                preset.dirty = False
        except OSError as exc:
            QMessageBox.warning(self, self.TITLE, f'Could not save the presets: {exc}')
            return
        self._removed_paths = []
        self._selected = self.build(self._current)
        super().accept()

    def selected_preset(self):
        """The preset chosen when OK was pressed, or None if the dialog was cancelled."""
        return self._selected

    # -- hooks ----------------------------------------------------------------------------

    def rows_of(self, preset) -> list:
        """The rows the dialog edits `preset` as."""
        raise NotImplementedError

    def build(self, working: WorkingPreset):
        """The preset `working` stands for, validated (raises PresetError)."""
        raise NotImplementedError

    def fill_row(self, index: int, row, editable: bool) -> None:
        """Put the widgets editing `row` into table row `index`, all but the last column."""
        raise NotImplementedError

    def new_row(self):
        """The row + Add appends."""
        raise NotImplementedError

    def new_preset_rows(self) -> list:
        """The rows of a preset made with New."""
        return []

    def row_removed(self, row) -> None:
        """Called once `row` is gone from the current preset."""

    def name_edited(self) -> None:
        """Called when a name has been edited."""
