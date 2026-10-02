"""Contour presets: which contour types a pullback is annotated with, and how each one is
drawn, painted into the exported mask, and read back out of one.

A preset is data — a JSON file (see BUILTIN_PRESETS_DIR) — so a new kind of annotation
needs no code, only a row. Two rows are fixed because the software itself relies on them:
the lumen, which every measurement starts from, and the EEM, which the vessel wall and
the plaque burden are read against. They are the first two rows of every preset.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable

from domain.all_types import ContourType, SegmentationTool
from domain.io_types import RESERVED_CONTOUR_IDS

PRESET_FORMAT = 1

# Bundled next to the source in development and next to the executable once compiled
# (see build_nuitka.ps1); this module sits one level below either.
BUILTIN_PRESETS_DIR = Path(__file__).resolve().parent.parent / 'presets' / 'intravascular'
DEFAULT_PRESET_FILE = BUILTIN_PRESETS_DIR / 'default.json'

# The keyboard shortcuts belong to HolOrama, not to a preset: they go to a preset's rows in
# the order it lists them, (new, append) per row, so the default preset keeps the keys it
# always had and a type added past the last slot simply has none. The lumen and the EEM
# are always the first two spline rows, and are never appended to.
SPLINE_SHORTCUTS: tuple[tuple[str, str | None], ...] = (
    ('E', None),
    ('Q', None),
    ('7', 'Ctrl+7'),
    ('8', 'Ctrl+8'),
    ('9', 'Ctrl+9'),
    ('0', 'Ctrl+0'),
)
ANGLE_SHORTCUTS: tuple[tuple[str, str | None], ...] = (('3', 'Ctrl+3'), ('B', 'Ctrl+B'))

_ID_PATTERN = re.compile(r'^[a-z][a-z0-9_]*$')


class PresetError(ValueError):
    """A preset that cannot be used as it stands; the message says which rule it breaks."""


class ToolSet(Enum):
    """What a contour type can be drawn with (preset's Tools column).

    The brush comes with every spline type (it paints the region a closed contour
    encloses) and never with an angular sector, which is stored as the angles bounding it.
    """

    CLOSED = 'closed'
    OPEN_CLOSED = 'open_closed'
    ANGLE = 'angle'

    @property
    def tools(self) -> frozenset[SegmentationTool]:
        return _TOOLS[self]


_TOOLS = {
    ToolSet.CLOSED: frozenset({SegmentationTool.CLOSED_SPLINE, SegmentationTool.BRUSH}),
    ToolSet.OPEN_CLOSED: frozenset(
        {SegmentationTool.OPEN_SPLINE, SegmentationTool.CLOSED_SPLINE, SegmentationTool.BRUSH}
    ),
    ToolSet.ANGLE: frozenset({SegmentationTool.ANGLE}),
}


@dataclass(frozen=True)
class ContourTypeDef:
    """One row of a preset."""

    type: ContourType
    name: str
    label: int  # the value this type's pixels carry in the exported mask
    color: str  # a Qt colour name or hex code
    tools: ToolSet
    # Where it sits in the mask, bottom (lowest) to top: wherever two regions overlap, the
    # higher one is what the mask shows. Whatever lies inside another type is layered above
    # it, as the lumen is above the EEM.
    layer: int
    # The type this one lies within. Its region is clipped to that type's (and, unless that
    # type is the lumen itself, kept out of the lumen), and an open contour of it is filled
    # outwards from the arc up to that type's boundary — the arc marks its luminal side.
    inside: ContourType | None = None

    @property
    def is_angle(self) -> bool:
        return self.tools is ToolSet.ANGLE

    @property
    def appendable(self) -> bool:
        """Whether a frame can hold several of this type; it holds one lumen and one EEM."""
        return self.type not in (ContourType.LUMEN, ContourType.EEM)

    def to_dict(self) -> dict:
        return {
            'id': self.type.value,
            'name': self.name,
            'label': self.label,
            'color': self.color,
            'tools': self.tools.value,
            'layer': self.layer,
            'inside': self.inside.value if self.inside is not None else None,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> ContourTypeDef:
        try:
            inside = raw.get('inside')
            return cls(
                type=ContourType(str(raw['id'])),
                name=str(raw['name']),
                label=int(raw['label']),
                color=str(raw['color']),
                tools=ToolSet(raw['tools']),
                layer=int(raw['layer']),
                inside=ContourType(str(inside)) if inside else None,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PresetError(f'Invalid contour type {raw!r}: {exc}') from exc


def _key(contour_type: ContourType | str) -> str:
    return contour_type if isinstance(contour_type, str) else contour_type.value


@dataclass(frozen=True)
class ContourPreset:
    """A named set of contour types, in the order they are listed (rows)."""

    name: str
    types: tuple[ContourTypeDef, ...]

    def __post_init__(self) -> None:
        _validate(self)

    # -- lookup ---------------------------------------------------------------------------

    def get(self, contour_type: ContourType | str) -> ContourTypeDef | None:
        """The row for `contour_type` (a ContourType or its id), or None if not in this preset."""
        key = _key(contour_type)
        return next((defn for defn in self.types if defn.type.value == key), None)

    def __getitem__(self, contour_type: ContourType | str) -> ContourTypeDef:
        defn = self.get(contour_type)
        if defn is None:
            raise KeyError(f'{_key(contour_type)} is not a contour type of preset {self.name!r}')
        return defn

    def __contains__(self, contour_type: object) -> bool:
        return isinstance(contour_type, (ContourType, str)) and self.get(contour_type) is not None

    def allowed_tools(self, contour_type: ContourType | str) -> frozenset[SegmentationTool]:
        """The tools `contour_type` can be drawn with; none for a type outside this preset."""
        defn = self.get(contour_type)
        return defn.tools.tools if defn is not None else frozenset()

    def is_angle(self, contour_type: ContourType | str) -> bool:
        defn = self.get(contour_type)
        return defn is not None and defn.is_angle

    @property
    def spline_types(self) -> tuple[ContourTypeDef, ...]:
        """Every type drawn as a spline (open or closed), in row order."""
        return tuple(defn for defn in self.types if not defn.is_angle)

    @property
    def angle_types(self) -> tuple[ContourTypeDef, ...]:
        """Every type drawn as an angular sector around the image centre, in row order:
        two radial boundaries and the region between them (the guide-wire shadow, the
        blood artefact)."""
        return tuple(defn for defn in self.types if defn.is_angle)

    def with_tool(self, tool: SegmentationTool) -> tuple[ContourType, ...]:
        """Every type that can be drawn with `tool`, in row order."""
        return tuple(defn.type for defn in self.types if tool in defn.tools.tools)

    def shortcuts(self, contour_type: ContourType | str) -> tuple[str | None, str | None]:
        """(new, append) keyboard shortcut of `contour_type`; see SPLINE_SHORTCUTS."""
        defn = self.get(contour_type)
        if defn is None:
            return None, None
        rows, slots = (self.angle_types, ANGLE_SHORTCUTS) if defn.is_angle else (self.spline_types, SPLINE_SHORTCUTS)
        index = rows.index(defn)
        return slots[index] if index < len(slots) else (None, None)

    # -- the mask -------------------------------------------------------------------------

    def paint_order(self) -> tuple[ContourTypeDef, ...]:
        """Every type, bottom to top — by layer — the order the mask is painted in."""
        return tuple(sorted(self.types, key=lambda defn: defn.layer))

    def contents(self, contour_type: ContourType | str) -> tuple[ContourTypeDef, ...]:
        """The types lying directly inside `contour_type`, in row order."""
        key = _key(contour_type)
        return tuple(defn for defn in self.types if defn.inside is not None and defn.inside.value == key)

    def mask_labels(self, contour_type: ContourType | str) -> frozenset[int]:
        """Every mask value a region of `contour_type` shows up as once the mask is painted.

        Its own label, and those of whatever lies inside it and so is painted over it — as
        does the lumen over the EEM, which is why reading an EEM back takes the lumen too.
        """
        defn = self[contour_type]
        labels = {defn.label}
        for inner in self.contents(defn.type):
            labels |= self.mask_labels(inner.type)
        if defn.type == ContourType.EEM:
            labels |= self.mask_labels(ContourType.LUMEN)
        return frozenset(labels)

    def _lies_in(self, defn: ContourTypeDef, container: ContourType) -> bool:
        """Whether `defn` lies inside `container`, directly or through the types between."""
        seen: set[ContourType] = set()
        current: ContourTypeDef | None = defn
        while current is not None and current.inside is not None and current.inside not in seen:
            if current.inside == container:
                return True
            seen.add(current.inside)
            current = self.get(current.inside)
        return False

    # -- persistence ----------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {'format': PRESET_FORMAT, 'name': self.name, 'types': [defn.to_dict() for defn in self.types]}

    @classmethod
    def from_dict(cls, raw: dict) -> ContourPreset:
        if not isinstance(raw, dict):
            raise PresetError(f'A preset is a JSON object, not {type(raw).__name__}')
        if raw.get('format', PRESET_FORMAT) > PRESET_FORMAT:
            raise PresetError(f'Preset format {raw.get("format")} is newer than this version reads')
        types = raw.get('types')
        if not isinstance(types, list):
            raise PresetError('A preset needs a list of "types"')
        return cls(name=str(raw.get('name', 'Unnamed')), types=tuple(ContourTypeDef.from_dict(t) for t in types))


def _validate(preset: ContourPreset) -> None:
    """Raise PresetError for the first rule `preset` breaks."""
    types = preset.types
    ids = [defn.type.value for defn in types]
    _require(len(types) >= 2, 'A preset needs at least the lumen and the EEM')
    _require(ids[0] == ContourType.LUMEN.value, 'The lumen has to be the first contour type')
    _require(ids[1] == ContourType.EEM.value, 'The EEM has to be the second contour type')
    _require_unique(ids, 'id')
    _require_unique((defn.label for defn in types), 'mask label')
    _require_unique((defn.layer for defn in types), 'layer')

    for defn in types:
        key = defn.type.value
        _require(bool(_ID_PATTERN.match(key)), f'{key!r} is not a valid id (lower-case letters, digits and _)')
        _require(key not in RESERVED_CONTOUR_IDS, f'{key!r} is reserved and cannot name a contour type')
        _require(bool(defn.name.strip()), f'{key} needs a name')
        _require(bool(defn.color.strip()), f'{key} needs a colour')
        _require(1 <= defn.label <= 255, f'The mask label of {defn.name} has to be between 1 and 255')
        if defn.type in (ContourType.LUMEN, ContourType.EEM):
            _require(defn.tools is ToolSet.CLOSED, f'{defn.name} can only be drawn as a closed contour')
            _require(defn.inside is None, f'{defn.name} cannot lie inside another type')
        if defn.inside is None:
            _require(defn.tools is not ToolSet.OPEN_CLOSED, f'{defn.name} is drawn open, so it needs a type to lie in')
            continue
        _require(not defn.is_angle, f'{defn.name} is an angle, which cannot lie inside another type')
        container = preset.get(defn.inside)
        _require(container is not None, f'{defn.name} lies inside {defn.inside.value}, which the preset lacks')
        assert container is not None
        _require(container.tools is not ToolSet.ANGLE, f'{defn.name} cannot lie inside the angle {container.name}')
        _require(container.type != defn.type, f'{defn.name} cannot lie inside itself')
        _require(not preset._lies_in(container, defn.type), f'{defn.name} and {container.name} lie inside each other')
        # Painted below its container it would never show: the container covers all of it.
        _require(defn.layer > container.layer, f'{defn.name} has to be layered above {container.name}, its container')

    lumen, eem = types[0], types[1]
    _require(lumen.layer > eem.layer, f'{lumen.name} has to be layered above {eem.name}, which it lies inside')


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PresetError(message)


def _require_unique(values: Iterable, what: str) -> None:
    seen: set = set()
    for value in values:
        _require(value not in seen, f'Two contour types share the {what} {value!r}')
        seen.add(value)


def with_labels(preset: ContourPreset, labels: Iterable[int], name: str) -> ContourPreset:
    """`preset` with a row added for every mask value in `labels` it lacks — closed,
    'Label <value>', on top, in a colour none of its rows has — for the user to fill in."""
    from domain.colors import distinct_colors

    missing = [label for label in labels if all(defn.label != label for defn in preset.types)]
    used = {defn.color.lower() for defn in preset.types}
    colors = [f'#{r:02x}{g:02x}{b:02x}' for r, g, b in distinct_colors(len(preset.types) + len(missing) + len(used))]
    fresh = [color for color in colors if color not in used]
    ids = {defn.type.value for defn in preset.types}
    top = max(defn.layer for defn in preset.types)
    added = []
    for i, label in enumerate(missing):
        key = f'label_{label}'
        while key in ids:
            key += '_'
        ids.add(key)
        added.append(
            ContourTypeDef(
                type=ContourType(key),
                name=f'Label {label}',
                label=label,
                color=fresh[i],
                tools=ToolSet.CLOSED,
                layer=top + 1 + i,
            )
        )
    return ContourPreset(name=name, types=preset.types + tuple(added))


def load_preset(path: Path) -> ContourPreset:
    """Read and validate the preset stored at `path`."""
    try:
        with open(path, encoding='utf-8') as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise PresetError(f'Could not read the preset {path}: {exc}') from exc
    return ContourPreset.from_dict(raw)


_active: ContourPreset | None = None


def active_preset() -> ContourPreset:
    """The preset the intravascular page annotates with; the bundled default until another
    one is set."""
    global _active
    if _active is None:
        _active = load_preset(DEFAULT_PRESET_FILE)
    return _active


def set_active_preset(preset: ContourPreset) -> None:
    global _active
    _active = preset
