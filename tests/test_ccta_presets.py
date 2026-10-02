"""Tests for the CCTA label presets (domain.ccta.presets), their settings dialog, the mask
panel showing them, and the Switch default button."""

import json
from types import SimpleNamespace

import pytest

from domain.ccta.presets import (
    COLORFUL_PRESET_FILE,
    PUBLICATION_PRESET_FILE,
    CctaPreset,
    active_ccta_preset,
    fallback_color,
    load_ccta_preset,
    set_active_ccta_preset,
)
from domain.colors import CATEGORICAL_PALETTE
from domain.presets import PresetError
from input_output import preset_library

ANATOMIC_NAMES = [
    'Coronaries',
    'LVM',
    'LA',
    'LV',
    'RA',
    'RV',
    'Aorta',
    'Pulmonary Arteries',
    'Pericardial Fat',
    'Epicardial Fat',
    'Pulmonary Vein',
    'SVC',
    'IVC',
    'LAA',
]


@pytest.fixture
def restore_active():
    before = active_ccta_preset()
    yield
    set_active_ccta_preset(before)


class TestBuiltIns:
    @pytest.mark.parametrize('path', [COLORFUL_PRESET_FILE, PUBLICATION_PRESET_FILE])
    def test_both_name_the_labels_anatomically(self, path):
        preset = load_ccta_preset(path)
        assert [defn.name for defn in preset.labels] == ANATOMIC_NAMES
        assert [defn.label for defn in preset.labels] == list(range(1, 15))

    def test_colorful_is_the_categorical_palette(self):
        preset = load_ccta_preset(COLORFUL_PRESET_FILE)
        assert [defn.rgb for defn in preset.labels] == list(CATEGORICAL_PALETTE)

    def test_publication_has_its_own_colours(self):
        colorful, publication = load_ccta_preset(COLORFUL_PRESET_FILE), load_ccta_preset(PUBLICATION_PRESET_FILE)
        assert publication.color_of(7) == (168, 32, 26)  # aorta
        assert colorful.color_of(7) != publication.color_of(7)

    def test_colorful_is_active_by_default(self):
        assert preset_library.CCTA.default_file == COLORFUL_PRESET_FILE


class TestLookup:
    def test_a_label_is_named_by_its_value_not_its_position(self):
        preset = load_ccta_preset(COLORFUL_PRESET_FILE)
        assert preset.name_of(7) == 'Aorta'  # whichever other labels a mask holds

    def test_a_value_the_preset_lacks(self):
        preset = load_ccta_preset(COLORFUL_PRESET_FILE)
        assert preset.name_of(42) == 'Label 42'
        assert preset.color_of(42) == fallback_color(42)

    @pytest.mark.parametrize(
        'labels, message',
        [
            ([{'label': 1, 'name': 'A', 'color': '#ffffff'}, {'label': 1, 'name': 'B', 'color': '#000000'}], 'share'),
            ([{'label': 0, 'name': 'A', 'color': '#ffffff'}], 'between 1'),
            ([{'label': 1, 'name': ' ', 'color': '#ffffff'}], 'needs a name'),
            ([{'label': 1, 'name': 'A', 'color': 'white'}], 'not a colour'),
        ],
    )
    def test_a_broken_rule_is_named(self, labels, message):
        with pytest.raises(PresetError, match=message):
            CctaPreset.from_dict({'name': 'X', 'labels': labels})


@pytest.fixture
def user_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(preset_library, 'user_presets_dir', lambda kind=None: tmp_path)
    return tmp_path


class TestDialog:
    def test_a_copy_with_a_new_label_is_saved(self, qt_app, user_dir, monkeypatch):
        from gui import preset_dialog
        from pages.ccta.popup_windows.contour_settings_dialog import CctaContourSettingsDialog

        monkeypatch.setattr(preset_dialog.QInputDialog, 'getText', lambda *a, **k: ('Whole heart', True))
        dialog = CctaContourSettingsDialog(active_name='Default 2 - Publication')
        assert dialog._preset_combo.currentText() == 'Default 2 - Publication (built-in)'
        assert dialog._table.columnCount() == 4  # label, colour, name, actions — no tools

        dialog._duplicate()
        dialog._add_row()
        dialog._current.rows[-1].name = 'Left atrial appendage occluder'
        dialog.accept()

        preset = dialog.selected_preset()
        assert preset.name_of(15) == 'Left atrial appendage occluder'
        assert preset.name_of(1) == 'Coronaries'
        assert [p.name for p in user_dir.iterdir()] == ['whole_heart.json']

    def test_a_clash_disables_ok(self, qt_app, user_dir, monkeypatch):
        from gui import preset_dialog
        from pages.ccta.popup_windows.contour_settings_dialog import CctaContourSettingsDialog

        monkeypatch.setattr(preset_dialog.QInputDialog, 'getText', lambda *a, **k: ('Mine', True))
        dialog = CctaContourSettingsDialog()
        dialog._duplicate()
        dialog._table.cellWidget(1, 0).setValue(1)
        assert not dialog._buttons.button(dialog._buttons.StandardButton.Ok).isEnabled()
        assert 'share the mask label 1' in dialog._error.text()

    def test_the_configured_preset_becomes_active(self, user_dir, restore_active):
        config = SimpleNamespace(ccta=SimpleNamespace(contour_preset='Default 2 - Publication'))
        preset_library.activate_configured_preset(config, preset_library.CCTA)
        assert active_ccta_preset() == load_ccta_preset(PUBLICATION_PRESET_FILE)


class TestMaskPanel:
    def test_names_are_shown_not_edited(self, qt_app):
        from PyQt6.QtWidgets import QLineEdit

        from pages.ccta.right_half.mask_panel import MaskPanel

        panel = MaskPanel()
        panel.set_labels([1, 7, 42])
        preset = load_ccta_preset(COLORFUL_PRESET_FILE)
        panel.set_label_appearance(
            preset.name,
            {label: preset.name_of(label) for label in (1, 7, 42)},
            {label: preset.color_of(label) for label in (1, 7, 42)},
        )
        assert panel.label_names() == {1: 'Coronaries', 7: 'Aorta', 42: 'Label 42'}
        assert not panel.findChildren(QLineEdit)

    def test_switch_default_asks_the_page(self, qt_app):
        from pages.ccta.right_half.mask_panel import MaskPanel

        panel = MaskPanel()
        asked = []
        panel.switch_default_requested.connect(lambda: asked.append(True))
        panel._btn_switch.click()
        assert asked == [True]


class TestSwitchDefault:
    def _switch(self, active):
        from domain.ccta.presets import switched_default

        return switched_default(active).name

    def test_colorful_switches_to_publication_and_back(self):
        assert self._switch(load_ccta_preset(COLORFUL_PRESET_FILE)) == 'Default 2 - Publication'
        assert self._switch(load_ccta_preset(PUBLICATION_PRESET_FILE)) == 'Default 1 - Colorful'

    def test_a_custom_preset_switches_to_colorful(self):
        raw = json.loads(PUBLICATION_PRESET_FILE.read_text(encoding='utf-8'))
        raw['name'] = 'Mine'
        assert self._switch(CctaPreset.from_dict(raw)) == 'Default 1 - Colorful'


class TestPresetForAMask:
    """Opening a mask whose labels the active preset does not all name offers a new preset."""

    def test_every_draft_colour_differs_past_the_palette(self):
        from domain.ccta.presets import distinct_colors

        colors = distinct_colors(40)
        assert len(set(colors)) == 40
        assert colors[:14] == list(CATEGORICAL_PALETTE)

    def test_a_draft_has_a_row_per_mask_value_and_names_nothing(self):
        from domain.ccta.presets import draft_for

        draft = draft_for('3 labels', [1, 2, 50])
        assert [(defn.label, defn.name) for defn in draft.labels] == [(1, 'Label 1'), (2, 'Label 2'), (50, 'Label 50')]
        assert len({defn.color for defn in draft.labels}) == 3

    def test_the_dialog_opens_on_the_draft_and_saves_it_only_on_ok(self, qt_app, user_dir):
        from domain.ccta.presets import draft_for
        from pages.ccta.popup_windows.contour_settings_dialog import CctaContourSettingsDialog

        dialog = CctaContourSettingsDialog(active_name='Default 1 - Colorful', draft=draft_for('20 labels', [1, 2, 3]))
        assert dialog._preset_combo.currentText() == '20 labels'
        assert dialog._table.rowCount() == 3 and dialog._add_btn.isEnabled()
        assert list(user_dir.iterdir()) == []

        dialog._current.rows[0].name = 'Liver'
        dialog.accept()
        assert dialog.selected_preset().name_of(1) == 'Liver'
        assert [p.name for p in user_dir.iterdir()] == ['20_labels.json']

    def test_a_draft_name_another_preset_has_is_made_unique(self, qt_app, user_dir):
        from domain.ccta.presets import draft_for
        from pages.ccta.popup_windows.contour_settings_dialog import CctaContourSettingsDialog

        dialog = CctaContourSettingsDialog(draft=draft_for('Default 1 - Colorful', [1]))
        assert dialog._current.name == 'Default 1 - Colorful (2)'

    def _offer(self, monkeypatch, labels, answer):
        from PyQt6.QtWidgets import QMessageBox

        from pages.ccta.popup_windows import contour_settings_dialog
        from pages.ccta.popup_windows.contour_settings_dialog import ask_for_draft

        asked = []

        def question(*args, **kwargs):
            asked.append(args[2])
            return answer

        monkeypatch.setattr(contour_settings_dialog.QMessageBox, 'question', question)
        draft = ask_for_draft(None, labels)  # type: ignore[arg-type]
        return asked, [] if draft is None else [draft], QMessageBox

    def test_a_mask_the_preset_names_in_full_asks_nothing(self, monkeypatch, restore_active):
        set_active_ccta_preset(load_ccta_preset(COLORFUL_PRESET_FILE))
        asked, opened, _ = self._offer(monkeypatch, [1, 4, 7], answer=None)
        assert asked == [] and opened == []

    def test_a_mask_with_labels_the_preset_lacks_is_offered_a_draft(self, monkeypatch, restore_active):
        from PyQt6.QtWidgets import QMessageBox

        set_active_ccta_preset(load_ccta_preset(COLORFUL_PRESET_FILE))
        labels = list(range(1, 21))
        asked, opened, _ = self._offer(monkeypatch, labels, answer=QMessageBox.StandardButton.Yes)
        assert '20 labels, 6 of which' in asked[0]
        assert [defn.label for defn in opened[0].labels] == labels

    def test_declining_changes_nothing(self, monkeypatch, restore_active):
        from PyQt6.QtWidgets import QMessageBox

        set_active_ccta_preset(load_ccta_preset(COLORFUL_PRESET_FILE))
        asked, opened, _ = self._offer(monkeypatch, [1, 99], answer=QMessageBox.StandardButton.No)
        assert len(asked) == 1 and opened == []
