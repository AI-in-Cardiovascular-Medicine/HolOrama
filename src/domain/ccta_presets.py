"""CCTA label presets: the name and colour each value of a CCTA mask stands for.

A CCTA mask arrives labelled already, so unlike the intravascular presets (see
domain.contour_presets) there is nothing to draw or layer: a row is a mask value, a name
and a colour. A value a mask holds that its preset lacks is shown as 'Label <value>' in a
palette colour. Two presets are built in: Colorful and Publication, both with the
anatomic names of a cardiac CCTA segmentation.
"""

from __future__ import annotations

import colorsys
import json
import re
from dataclasses import dataclass
from pathlib import Path

from domain.ccta_display_types import LABEL_COLORS
from domain.contour_presets import PRESET_FORMAT, PresetError

BUILTIN_CCTA_PRESETS_DIR = Path(__file__).resolve().parent.parent / 'presets' / 'ccta'
COLORFUL_PRESET_FILE = BUILTIN_CCTA_PRESETS_DIR / 'colorful.json'
PUBLICATION_PRESET_FILE = BUILTIN_CCTA_PRESETS_DIR / 'publication.json'

MAX_LABEL = 65535  # a CCTA mask can be uint16 (e.g. a whole-body segmentation)
_HEX_COLOR = re.compile(r'^#[0-9a-fA-F]{6}$')

Rgb = tuple[int, int, int]


def hex_to_rgb(color: str) -> Rgb:
    return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)


def rgb_to_hex(rgb: Rgb) -> str:
    r, g, b = rgb
    return f'#{r:02x}{g:02x}{b:02x}'


def fallback_color(label: int) -> Rgb:
    """The colour of a mask value its preset does not define."""
    return LABEL_COLORS[(label - 1) % len(LABEL_COLORS)]


_GOLDEN_RATIO = 0.618033988749895


def distinct_colors(count: int) -> list[Rgb]:
    """`count` colours that all differ: the palette first, then hues spread by the golden
    ratio (each new one lands in the widest gap left), alternating in brightness."""
    colors = list(LABEL_COLORS[:count])
    hue = 0.0
    for i in range(count - len(colors)):
        hue = (hue + _GOLDEN_RATIO) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95 if i % 2 == 0 else 0.7)
        colors.append((round(r * 255), round(g * 255), round(b * 255)))
    return colors


def draft_for(name: str, labels: list[int]) -> CctaPreset:
    """A preset naming nothing yet: one row per mask value in `labels`, each 'Label <value>'
    in a colour of its own — for the user to fill in."""
    colors = distinct_colors(len(labels))
    return CctaPreset(
        name=name,
        labels=tuple(
            CctaLabelDef(label=label, name=f'Label {label}', color=rgb_to_hex(color))
            for label, color in zip(labels, colors)
        ),
    )


@dataclass(frozen=True)
class CctaLabelDef:
    """One row of a CCTA preset."""

    label: int  # the mask value
    name: str
    color: str  # '#rrggbb'

    @property
    def rgb(self) -> Rgb:
        return hex_to_rgb(self.color)

    def to_dict(self) -> dict:
        return {'label': self.label, 'name': self.name, 'color': self.color}

    @classmethod
    def from_dict(cls, raw: dict) -> CctaLabelDef:
        try:
            return cls(label=int(raw['label']), name=str(raw['name']), color=str(raw['color']))
        except (KeyError, TypeError, ValueError) as exc:
            raise PresetError(f'Invalid label {raw!r}: {exc}') from exc


@dataclass(frozen=True)
class CctaPreset:
    """A named set of mask labels, in the order they are listed (rows)."""

    name: str
    labels: tuple[CctaLabelDef, ...]

    def __post_init__(self) -> None:
        seen: set[int] = set()
        for defn in self.labels:
            if not 1 <= defn.label <= MAX_LABEL:
                raise PresetError(f'The mask label of {defn.name} has to be between 1 and {MAX_LABEL}')
            if defn.label in seen:
                raise PresetError(f'Two labels share the mask label {defn.label}')
            seen.add(defn.label)
            if not defn.name.strip():
                raise PresetError(f'Mask label {defn.label} needs a name')
            if not _HEX_COLOR.match(defn.color):
                raise PresetError(f'{defn.color!r} is not a colour (#rrggbb) for {defn.name}')

    def get(self, label: int) -> CctaLabelDef | None:
        return next((defn for defn in self.labels if defn.label == label), None)

    def name_of(self, label: int) -> str:
        defn = self.get(label)
        return defn.name if defn is not None else f'Label {label}'

    def color_of(self, label: int) -> Rgb:
        defn = self.get(label)
        return defn.rgb if defn is not None else fallback_color(label)

    def to_dict(self) -> dict:
        return {'format': PRESET_FORMAT, 'name': self.name, 'labels': [defn.to_dict() for defn in self.labels]}

    @classmethod
    def from_dict(cls, raw: dict) -> CctaPreset:
        if not isinstance(raw, dict):
            raise PresetError(f'A preset is a JSON object, not {type(raw).__name__}')
        if raw.get('format', PRESET_FORMAT) > PRESET_FORMAT:
            raise PresetError(f'Preset format {raw.get("format")} is newer than this version reads')
        labels = raw.get('labels')
        if not isinstance(labels, list):
            raise PresetError('A CCTA preset needs a list of "labels"')
        return cls(name=str(raw.get('name', 'Unnamed')), labels=tuple(CctaLabelDef.from_dict(r) for r in labels))


def load_ccta_preset(path: Path) -> CctaPreset:
    """Read and validate the CCTA preset stored at `path`."""
    try:
        with open(path, encoding='utf-8') as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise PresetError(f'Could not read the preset {path}: {exc}') from exc
    return CctaPreset.from_dict(raw)


_active: CctaPreset | None = None


def active_ccta_preset() -> CctaPreset:
    """The preset the CCTA page names and colours its labels with; Colorful until another
    one is set."""
    global _active
    if _active is None:
        _active = load_ccta_preset(COLORFUL_PRESET_FILE)
    return _active


def set_active_ccta_preset(preset: CctaPreset) -> None:
    global _active
    _active = preset
