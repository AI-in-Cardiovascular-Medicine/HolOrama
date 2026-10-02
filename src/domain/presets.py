"""What the intravascular contour presets and the CCTA label presets share."""

PRESET_FORMAT = 1  # the version of the preset JSON files


class PresetError(ValueError):
    """A preset that cannot be used as it stands; the message says which rule it breaks."""
