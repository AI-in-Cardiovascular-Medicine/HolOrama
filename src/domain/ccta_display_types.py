LABEL_COLORS: tuple[tuple[int, int, int], ...] = (
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

DEFAULT_MASK_ALPHA: float = 0.45

DEFAULT_CT_LEVEL: int = 200  # HU center — cardiac soft tissue
DEFAULT_CT_WIDTH: int = 700  # HU range
