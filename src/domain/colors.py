"""Color primitives shared across ccta/intravascular/fusion (see
ccta_display_types.py, contour_presets.py, and fusion_display_types.py for the
domain-specific palettes built on top of these)."""

# Generic qualitative palette for indexed/categorical coloring (segmentation labels,
# branch/side-branch coloring, ...). Index with `CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)]`.
CATEGORICAL_PALETTE: tuple[tuple[int, int, int], ...] = (
    (255, 60, 60),  # red
    (60, 220, 60),  # green
    (60, 60, 255),  # blue
    (255, 220, 0),  # yellow
    (220, 60, 220),  # magenta
    (0, 210, 210),  # cyan
    (255, 140, 0),  # orange
    (160, 60, 255),  # purple
    (0, 180, 255),  # sky blue
    (255, 60, 140),  # pink
    (0, 200, 120),  # mint
    (180, 255, 0),  # lime
    (255, 180, 100),  # peach
    (140, 140, 255),  # lavender
)

_GOLDEN_RATIO = 0.618033988749895


def distinct_colors(count: int) -> list[tuple[int, int, int]]:
    """`count` colours that all differ: the categorical palette first, then hues spread by
    the golden ratio (each new one lands in the widest gap left), alternating in brightness."""
    import colorsys

    colors = list(CATEGORICAL_PALETTE[:count])
    hue = 0.0
    for i in range(count - len(colors)):
        hue = (hue + _GOLDEN_RATIO) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95 if i % 2 == 0 else 0.7)
        colors.append((round(r * 255), round(g * 255), round(b * 255)))
    return colors


DEFAULT_MASK_ALPHA: float = 0.45

# shared by the intravascular page, its plots
# (gating, longitudinal view, results plot), and the fusion viewer's aligned-geometry
# rendering, so "diastole"/"systole" always mean the same colors app-wide.
DIASTOLE_COLOR: tuple[int, int, int] = (39, 69, 219)
SYSTOLE_COLOR: tuple[int, int, int] = (209, 55, 38)


def branch_ramp_color(base: tuple[int, int, int], index: int, count: int) -> tuple[int, int, int]:
    """Shade base color progressively lighter for branch index `index` of `count`."""
    if count <= 1:
        return base
    t = index / max(count - 1, 1)
    r, g, b = base
    return (
        int(r + (255 - r) * 0.5 * t),
        int(g + (255 - g) * 0.5 * t),
        int(b + (255 - b) * 0.5 * t),
    )
