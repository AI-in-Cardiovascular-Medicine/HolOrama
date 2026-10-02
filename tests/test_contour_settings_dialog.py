"""Tests for Intravascular Contour Settings (the preset dialog) and the preset library behind it.

The dialog keeps every change in memory until OK; these drive its widgets the way a user
would and check what lands on disk and what preset comes out.
"""

import json

import pytest
from PyQt6.QtWidgets import QComboBox, QLineEdit, QMessageBox, QPushButton, QSpinBox

from domain.intravascular.types import ContourType
from domain.intravascular.contour_presets import (
    DEFAULT_PRESET_FILE,
    ToolSet,
    active_preset,
    load_preset,
    set_active_preset,
)
from input_output import preset_library
from gui import preset_dialog as dialog_module
from pages.intravascular.popup_windows.contour_settings_dialog import (
    _COL_INSIDE,
    _COL_LABEL,
    _COL_NAME,
    _COL_TOOLS,
    ContourSettingsDialog,
)


@pytest.fixture
def user_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(preset_library, 'user_presets_dir', lambda kind=None: tmp_path)
    return tmp_path


@pytest.fixture
def answers(monkeypatch):
    """Queue the names the user types into the name prompts."""
    queue: list = []

    def get_text(*_args, **_kwargs):
        return (queue.pop(0), True) if queue else ('', False)

    monkeypatch.setattr(dialog_module.QInputDialog, 'getText', get_text)
    monkeypatch.setattr(dialog_module.QMessageBox, 'warning', lambda *args: None)
    monkeypatch.setattr(dialog_module.QMessageBox, 'question', lambda *args: QMessageBox.StandardButton.Yes)
    return queue


@pytest.fixture
def restore_active_preset():
    before = active_preset()
    yield
    set_active_preset(before)


def _dialog(qt_app, active_name='Default'):
    return ContourSettingsDialog(active_name=active_name)


def _cell(dialog, row, column):
    return dialog._table.cellWidget(row, column)


def _ok(dialog):
    return dialog._buttons.button(dialog._buttons.StandardButton.Ok)


class TestBuiltInPresets:
    def test_it_opens_on_the_active_preset_with_its_rows(self, qt_app, user_dir):
        dialog = _dialog(qt_app)
        assert dialog._preset_combo.currentText() == 'Default (built-in)'
        names = [_cell(dialog, i, _COL_NAME).text() for i in range(dialog._table.rowCount())]
        assert names == [defn.name for defn in active_preset().types]

    def test_a_built_in_preset_is_read_only(self, qt_app, user_dir):
        dialog = _dialog(qt_app)
        assert not dialog._add_btn.isEnabled()
        assert not dialog._rename_btn.isEnabled() and not dialog._delete_btn.isEnabled()
        assert not _cell(dialog, 2, _COL_NAME).isEnabled()
        assert _ok(dialog).isEnabled()  # choosing it is still fine

    def test_ok_on_it_writes_nothing(self, qt_app, user_dir):
        dialog = _dialog(qt_app)
        dialog.accept()
        assert list(user_dir.iterdir()) == []
        assert dialog.selected_preset() == load_preset(DEFAULT_PRESET_FILE)


class TestEditing:
    def _duplicate(self, qt_app, answers, name='IVUS CAD'):
        dialog = _dialog(qt_app)
        answers.append(name)
        dialog._duplicate()
        return dialog

    def test_duplicate_makes_an_editable_copy(self, qt_app, user_dir, answers):
        dialog = self._duplicate(qt_app, answers)
        assert dialog._preset_combo.currentText() == 'IVUS CAD'
        assert dialog._add_btn.isEnabled()
        assert _cell(dialog, 2, _COL_NAME).isEnabled()

    def test_a_name_another_preset_has_is_refused(self, qt_app, user_dir, answers):
        dialog = _dialog(qt_app)
        answers.extend(['default'])  # then the prompt is cancelled
        dialog._duplicate()
        assert [preset.name for preset in dialog._presets] == ['Default']

    def test_a_new_type_is_saved_with_an_id_from_its_name(self, qt_app, user_dir, answers):
        dialog = self._duplicate(qt_app, answers)
        dialog._add_row()
        row = dialog._table.rowCount() - 1
        name = _cell(dialog, row, _COL_NAME)
        name.setText('Thrombus')
        name.textEdited.emit('Thrombus')
        name.editingFinished.emit()
        inside = _cell(dialog, row, _COL_INSIDE)
        inside.setCurrentIndex(inside.findData(ContourType.LUMEN.value))

        dialog.accept()

        preset = dialog.selected_preset()
        thrombus = preset['thrombus']
        assert thrombus.inside == ContourType.LUMEN
        assert preset.paint_order()[-1] == thrombus  # it lies in the lumen, so it is painted over it
        files = list(user_dir.iterdir())
        assert len(files) == 1 and load_preset(files[0]) == preset

    def test_a_clash_disables_ok_and_says_why(self, qt_app, user_dir, answers):
        dialog = self._duplicate(qt_app, answers)
        label: QSpinBox = _cell(dialog, 2, _COL_LABEL)
        label.setValue(1)  # the lumen's
        assert not _ok(dialog).isEnabled()
        assert 'mask label' in dialog._error.text()
        label.setValue(3)
        assert _ok(dialog).isEnabled()

    def test_a_saved_type_stays_a_spline_type(self, qt_app, user_dir, answers):
        dialog = self._duplicate(qt_app, answers)
        tools: QComboBox = _cell(dialog, 2, _COL_TOOLS)  # calcium
        assert [tools.itemData(i) for i in range(tools.count())] == [ToolSet.CLOSED, ToolSet.OPEN_CLOSED]

    def test_a_new_type_can_be_anything(self, qt_app, user_dir, answers):
        dialog = self._duplicate(qt_app, answers)
        dialog._add_row()
        tools: QComboBox = _cell(dialog, dialog._table.rowCount() - 1, _COL_TOOLS)
        assert tools.count() == 3

    def test_making_a_type_open_gives_it_the_eem_to_fill_up_to(self, qt_app, user_dir, answers):
        dialog = self._duplicate(qt_app, answers)
        dialog._add_row()
        row = dialog._table.rowCount() - 1
        tools: QComboBox = _cell(dialog, row, _COL_TOOLS)
        tools.setCurrentIndex(tools.findData(ToolSet.OPEN_CLOSED))
        assert dialog._current.rows[row].inside == ContourType.EEM.value
        assert _ok(dialog).isEnabled()

    def test_the_lumen_and_the_eem_cannot_be_removed_or_moved(self, qt_app, user_dir, answers):
        dialog = self._duplicate(qt_app, answers)
        for row in (0, 1):
            assert not _cell(dialog, row, 6).findChildren(QPushButton)
        calcium = dialog._current.rows[2]
        dialog._move_row(calcium, -1)
        assert dialog._current.rows[2] is calcium

    def test_removing_a_type_frees_what_lay_inside_it(self, qt_app, user_dir, answers):
        dialog = self._duplicate(qt_app, answers)
        dialog._add_row()
        new = dialog._current.rows[-1]
        new.inside = 'branch'
        dialog._remove_row(dialog._current.rows[3])  # branch
        assert new.inside is None

    def test_cancel_writes_nothing(self, qt_app, user_dir, answers):
        dialog = self._duplicate(qt_app, answers)
        dialog.reject()
        assert list(user_dir.iterdir()) == []
        assert dialog.selected_preset() is None


class TestUserPresets:
    def _saved(self, user_dir, name='OCT CAD'):
        preset = load_preset(DEFAULT_PRESET_FILE)
        raw = preset.to_dict()
        raw['name'] = name
        path = user_dir / 'oct_cad.json'
        path.write_text(json.dumps(raw), encoding='utf-8')
        return path

    def test_they_are_listed_after_the_built_in_ones(self, qt_app, user_dir):
        self._saved(user_dir)
        dialog = _dialog(qt_app, active_name='OCT CAD')
        assert [dialog._preset_combo.itemText(i) for i in range(2)] == ['Default (built-in)', 'OCT CAD']
        assert dialog._preset_combo.currentIndex() == 1

    def test_rename_keeps_the_file(self, qt_app, user_dir, answers):
        path = self._saved(user_dir)
        dialog = _dialog(qt_app, active_name='OCT CAD')
        answers.append('OCT stable CAD')
        dialog._rename()
        dialog.accept()
        assert load_preset(path).name == 'OCT stable CAD'
        assert list(user_dir.iterdir()) == [path]

    def test_delete_removes_the_file_on_ok(self, qt_app, user_dir, answers):
        path = self._saved(user_dir)
        dialog = _dialog(qt_app, active_name='OCT CAD')
        dialog._delete()
        assert path.exists()  # nothing happens before OK
        dialog.accept()
        assert not path.exists()
        assert dialog.selected_preset().name == 'Default'

    def test_import_and_export(self, qt_app, user_dir, answers, monkeypatch, tmp_path_factory):
        outside = tmp_path_factory.mktemp('shared') / 'colleague.json'
        dialog = _dialog(qt_app)
        monkeypatch.setattr(dialog_module.QFileDialog, 'getSaveFileName', lambda *a: (str(outside), ''))
        dialog._export()
        assert load_preset(outside) == load_preset(DEFAULT_PRESET_FILE)

        monkeypatch.setattr(dialog_module.QFileDialog, 'getOpenFileName', lambda *a: (str(outside), ''))
        answers.append('Colleague')  # 'Default' is taken
        dialog._import()
        dialog.accept()
        assert dialog.selected_preset().name == 'Colleague'
        assert [entry.name for entry in preset_library.load_library()] == ['Default', 'Colleague']


class TestLibrary:
    def test_a_broken_file_is_skipped(self, user_dir):
        (user_dir / 'broken.json').write_text('{', encoding='utf-8')
        assert [entry.name for entry in preset_library.load_library()] == ['Default']

    def test_a_second_preset_of_the_same_name_is_skipped(self, user_dir):
        (user_dir / 'copy.json').write_text(DEFAULT_PRESET_FILE.read_text(encoding='utf-8'), encoding='utf-8')
        assert [entry.name for entry in preset_library.load_library()] == ['Default']

    def test_the_configured_preset_becomes_active(self, user_dir, restore_active_preset):
        raw = load_preset(DEFAULT_PRESET_FILE).to_dict()
        raw['name'] = 'OCT CAD'
        (user_dir / 'oct.json').write_text(json.dumps(raw), encoding='utf-8')
        config = type('C', (), {'intravascular': type('I', (), {'contour_preset': 'OCT CAD'})()})()
        assert preset_library.activate_configured_preset(config).name == 'OCT CAD'
        assert active_preset().name == 'OCT CAD'

    def test_an_unknown_preset_falls_back_to_the_default(self, user_dir, restore_active_preset):
        config = type('C', (), {'intravascular': type('I', (), {'contour_preset': 'Gone'})()})()
        assert preset_library.activate_configured_preset(config).name == 'Default'


def test_name_edits_reach_the_inside_choices(qt_app, user_dir, answers):
    dialog = _dialog(qt_app)
    answers.append('Mine')
    dialog._duplicate()
    name: QLineEdit = _cell(dialog, 3, _COL_NAME)  # branch
    name.setText('Side branch')
    name.textEdited.emit('Side branch')
    name.editingFinished.emit()
    inside: QComboBox = _cell(dialog, 2, _COL_INSIDE)
    assert inside.findText('Side branch') >= 0
