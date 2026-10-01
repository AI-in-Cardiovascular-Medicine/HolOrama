"""The presets on disk: the built-in ones bundled with the app, and the ones a user made,
imported or copied, one JSON file each — for each kind of preset there is (the
intravascular contour presets, the CCTA label presets).

Built-in presets are read-only — they may sit under C:\\Program Files once installed — so a
user changes one by duplicating it. User presets live in a per-user directory once frozen
and in presets/ at the repository root in a dev run (see app_paths). A preset is known by
its name, which is unique among the presets of its kind.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from app_paths import IS_FROZEN, REPO_ROOT, user_data_dir
from domain.ccta_presets import BUILTIN_CCTA_PRESETS_DIR, COLORFUL_PRESET_FILE, load_ccta_preset, set_active_ccta_preset
from domain.contour_presets import (
    BUILTIN_PRESETS_DIR,
    DEFAULT_PRESET_FILE,
    PresetError,
    load_preset,
    set_active_preset,
)


@dataclass(frozen=True)
class PresetKind:
    folder: str  # the presets/ subdirectory, and the config section that names the active one
    builtin_dir: Path
    default_file: Path  # what is active when the configured preset cannot be found
    load: Callable[[Path], Any]
    activate: Callable[[Any], None]


INTRAVASCULAR = PresetKind('intravascular', BUILTIN_PRESETS_DIR, DEFAULT_PRESET_FILE, load_preset, set_active_preset)
CCTA = PresetKind('ccta', BUILTIN_CCTA_PRESETS_DIR, COLORFUL_PRESET_FILE, load_ccta_preset, set_active_ccta_preset)


def user_presets_dir(kind: PresetKind = INTRAVASCULAR) -> Path:
    return (user_data_dir() if IS_FROZEN else REPO_ROOT) / 'presets' / kind.folder


@dataclass(frozen=True)
class PresetEntry:
    preset: Any  # a ContourPreset or a CctaPreset, by kind
    path: Path
    builtin: bool

    @property
    def name(self) -> str:
        return self.preset.name


def load_library(kind: PresetKind = INTRAVASCULAR) -> list[PresetEntry]:
    """Every readable preset of `kind`: the built-in ones first, then the user's, each by name.

    A file that cannot be read, or whose name another preset already has, is skipped with
    a warning rather than taking the whole library down with it.
    """
    entries: list[PresetEntry] = []
    names: set[str] = set()
    for directory, builtin in ((kind.builtin_dir, True), (user_presets_dir(kind), False)):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob('*.json')):
            try:
                preset = kind.load(path)
            except PresetError as exc:
                logger.warning(f'Skipping preset {path}: {exc}')
                continue
            if preset.name.casefold() in names:
                logger.warning(f'Skipping preset {path}: another preset is already called {preset.name!r}')
                continue
            names.add(preset.name.casefold())
            entries.append(PresetEntry(preset, path, builtin))
    return entries


def _file_stem(name: str) -> str:
    stem = re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_').lower()
    return stem or 'preset'


def save_user_preset(preset, path: Path | None = None, kind: PresetKind = INTRAVASCULAR) -> Path:
    """Write `preset` to `path`, or to a new file in the user preset directory of `kind`
    named after it. Returns where it went."""
    if path is None:
        directory = user_presets_dir(kind)
        directory.mkdir(parents=True, exist_ok=True)
        stem = _file_stem(preset.name)
        path = directory / f'{stem}.json'
        suffix = 2
        while path.exists():
            path = directory / f'{stem}_{suffix}.json'
            suffix += 1
    write_preset(preset, path)
    return path


def write_preset(preset, path: Path) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(preset.to_dict(), f, indent=2)
        f.write('\n')


def activate_configured_preset(config, kind: PresetKind = INTRAVASCULAR):
    """Make the preset named in config.<kind>.contour_preset the active one of its kind,
    falling back to the kind's default if no preset of that name can be read."""
    wanted = getattr(getattr(config, kind.folder, None), 'contour_preset', None)
    entry = next((entry for entry in load_library(kind) if entry.name == wanted), None)
    if entry is None:
        if wanted:
            logger.warning(f'Preset {wanted!r} not found; using {kind.default_file.name}')
        preset = kind.load(kind.default_file)
    else:
        preset = entry.preset
    kind.activate(preset)
    return preset
