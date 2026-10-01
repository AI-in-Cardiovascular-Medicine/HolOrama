"""The contour presets on disk: the built-in ones bundled with the app, and the ones a user
made, imported or copied, one JSON file each.

Built-in presets are read-only — they may sit under C:\\Program Files once installed — so a
user changes one by duplicating it. User presets live in a per-user directory once frozen
and in presets/ at the repository root in a dev run (see app_paths). A preset is known by
its name, which is unique across both.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from app_paths import IS_FROZEN, REPO_ROOT, user_data_dir
from domain.contour_presets import (
    BUILTIN_PRESETS_DIR,
    DEFAULT_PRESET_FILE,
    ContourPreset,
    PresetError,
    load_preset,
    set_active_preset,
)


def user_presets_dir() -> Path:
    return (user_data_dir() if IS_FROZEN else REPO_ROOT) / 'presets' / 'intravascular'


@dataclass(frozen=True)
class PresetEntry:
    preset: ContourPreset
    path: Path
    builtin: bool

    @property
    def name(self) -> str:
        return self.preset.name


def load_library() -> list[PresetEntry]:
    """Every readable preset: the built-in ones first, then the user's, each by name.

    A file that cannot be read, or whose name another preset already has, is skipped with
    a warning rather than taking the whole library down with it.
    """
    entries: list[PresetEntry] = []
    names: set[str] = set()
    for directory, builtin in ((BUILTIN_PRESETS_DIR, True), (user_presets_dir(), False)):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob('*.json')):
            try:
                preset = load_preset(path)
            except PresetError as exc:
                logger.warning(f'Skipping contour preset {path}: {exc}')
                continue
            if preset.name.casefold() in names:
                logger.warning(f'Skipping contour preset {path}: another preset is already called {preset.name!r}')
                continue
            names.add(preset.name.casefold())
            entries.append(PresetEntry(preset, path, builtin))
    return entries


def _file_stem(name: str) -> str:
    stem = re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_').lower()
    return stem or 'preset'


def save_user_preset(preset: ContourPreset, path: Path | None = None) -> Path:
    """Write `preset` to `path`, or to a new file in the user preset directory named
    after it. Returns where it went."""
    if path is None:
        directory = user_presets_dir()
        directory.mkdir(parents=True, exist_ok=True)
        stem = _file_stem(preset.name)
        path = directory / f'{stem}.json'
        suffix = 2
        while path.exists():
            path = directory / f'{stem}_{suffix}.json'
            suffix += 1
    write_preset(preset, path)
    return path


def write_preset(preset: ContourPreset, path: Path) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(preset.to_dict(), f, indent=2)
        f.write('\n')


def activate_configured_preset(config) -> ContourPreset:
    """Make the preset named in config.intravascular.contour_preset the active one,
    falling back to the bundled default if no preset of that name can be read."""
    wanted = getattr(getattr(config, 'intravascular', None), 'contour_preset', None)
    entry = next((entry for entry in load_library() if entry.name == wanted), None)
    if entry is None:
        if wanted:
            logger.warning(f'Contour preset {wanted!r} not found; using the default preset')
        preset = load_preset(DEFAULT_PRESET_FILE)
    else:
        preset = entry.preset
    set_active_preset(preset)
    return preset
