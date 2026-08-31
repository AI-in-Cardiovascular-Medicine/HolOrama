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


def save_values(config_path: Path, key_sections: dict[str, str], values: dict[str, Any]) -> None:
    """Round-trip config.yaml via ruamel.yaml, updating only the given keys (each routed
    to its own top-level section) while preserving comments/formatting elsewhere."""
    from ruamel.yaml import YAML

    yaml = YAML(typ='rt')
    yaml.preserve_quotes = True
    yaml.boolean_representation = ['False', 'True']  # type: ignore[attr-defined]  # config.yaml uses capitalized booleans
    with open(config_path, encoding='utf-8') as f:
        data = yaml.load(f)

    for key, value in values.items():
        data[key_sections[key]][key] = value

    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f)


def resolve_config_path(config) -> Path:
    """The path _load_config stashed on the live config namespace, or a best-effort
    fallback to the in-repo default if that's somehow missing."""
    config_path = getattr(config, '_config_path', None)
    if config_path is not None:
        return config_path
    return Path(__file__).resolve().parent.parent / 'config.yaml'
