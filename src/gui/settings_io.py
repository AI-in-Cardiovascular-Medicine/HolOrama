"""Shared load/apply/save plumbing for per-page Settings dialogs (Display Settings,
CCTA Settings, ...). Each dialog defines its own defaults plus a key -> top-level
config-section mapping — some keys are shared across pages (e.g. windowing/zoom
sensitivity live under 'common'), others are page-specific ('intravascular', 'ccta') —
and this module does the generic SimpleNamespace read and round-tripped YAML write-back
so that logic isn't duplicated per dialog."""

from pathlib import Path
from typing import Any


def current_values(config, key_sections: dict[str, str], defaults: dict[str, Any]) -> dict[str, Any]:
    """Read each key from config.<section>, falling back to its default if the section
    or key isn't present."""
    values = {}
    for key, default in defaults.items():
        section = getattr(config, key_sections[key], None)
        values[key] = getattr(section, key, default) if section is not None else default
    return values


def apply_values(config, key_sections: dict[str, str], values: dict[str, Any]) -> None:
    """Write each value onto the live config namespace (config.<section>.<key> = value)."""
    for key, value in values.items():
        section = getattr(config, key_sections[key])
        setattr(section, key, value)


def _to_yaml_value(value: Any):
    """A list of (r, g, b)-style rows (e.g. label_colors) round-trips through ruamel as
    an exploded block sequence of block sequences unless each row is explicitly marked
    flow-style — this rebuilds any list-of-lists/tuples value that way so it dumps as
    '- [r, g, b]' rather than '- - r\\n  - g\\n  - b'. Any other value passes through as-is."""
    from ruamel.yaml.comments import CommentedSeq

    if isinstance(value, list) and value and all(isinstance(row, (list, tuple)) for row in value):
        rows = CommentedSeq()
        for row in value:
            seq_row = CommentedSeq(row)
            seq_row.fa.set_flow_style()
            rows.append(seq_row)
        return rows
    return value


def save_values(config_path: Path, key_sections: dict[str, str], values: dict[str, Any]) -> None:
    """Round-trip config.yaml via ruamel.yaml, updating only the given keys (each routed
    to its own top-level section) while preserving comments/formatting elsewhere."""
    from ruamel.yaml import YAML
    from ruamel.yaml.comments import CommentedSeq

    yaml = YAML(typ='rt')
    yaml.preserve_quotes = True
    yaml.boolean_representation = ['False', 'True']  # type: ignore[attr-defined]  # config.yaml uses capitalized booleans
    yaml.indent(mapping=2, sequence=4, offset=2)  # matches this file's hand-written list style
    with open(config_path, encoding='utf-8') as f:
        data = yaml.load(f)

    for key, value in values.items():
        section = data[key_sections[key]]
        new_value = _to_yaml_value(value)
        existing = section.get(key)
        if isinstance(existing, CommentedSeq) and isinstance(new_value, CommentedSeq):
            # Mutate the existing sequence node in place rather than replacing it outright,
            # so any comments/blank lines ruamel attached to that node survive the save.
            existing[:] = new_value
        else:
            section[key] = new_value

    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f)


def resolve_config_path(config) -> Path:
    """The path _load_config stashed on the live config namespace, or a best-effort
    fallback to the in-repo default if that's somehow missing."""
    config_path = getattr(config, '_config_path', None)
    if config_path is not None:
        return config_path
    return Path(__file__).resolve().parent.parent / 'config.yaml'
