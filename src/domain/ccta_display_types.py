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

LABEL_NAMES_ANATOMIC: tuple[str, ...] = (
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
)

LABEL_COLORS_ANATOMIC: tuple[tuple[int, int, int], ...] = (
    (255, 60, 60),  # Coronaries
    (123, 36, 28),  # LVM
    (240, 138, 138),  # LA
    (231, 76, 60),  # LV
    (131, 153, 226),  # RA
    (68, 83, 219),  # RV
    (168, 32, 26),  # Ao
    (27, 79, 114),  # Pulmonary Arteries
    (250, 229, 195),  # Pericardial Fat
    (241, 196, 15),  # Epicardial Fat
    (244, 181, 181),  # Pulmonary Vein
    (72, 201, 176),  # SVC
    (14, 102, 85),  # IVC
    (200, 96, 74),  # LAA
)

DEFAULT_MASK_ALPHA: float = 0.45

DEFAULT_CT_LEVEL: int = 200  # HU center — cardiac soft tissue
DEFAULT_CT_WIDTH: int = 700  # HU range

# Post-cut geometry layer (Build Cut Geometry / Smooth / outlet points)
CUT_MESH_COLOR: tuple[int, int, int] = (210, 210, 210)  # light gray, distinct from LABEL_COLORS
INLET_COLOR: tuple[int, int, int] = (0, 210, 210)  # cyan — lower-Z cut plane (LV / LVOT side)
OUTLET_COLOR: tuple[int, int, int] = (220, 60, 220)  # magenta — higher-Z cut plane (aorta side)
RCA_POINT_COLOR: tuple[int, int, int] = (255, 60, 60)  # red
LCA_POINT_COLOR: tuple[int, int, int] = (60, 120, 255)  # blue

# Computed centerlines (Calculate Centerlines) — distinct from the outlet-point colors
# above so the (thin) centerline lines don't get visually confused with the (round)
# outlet-point markers sharing the same vessel.
CENTERLINE_AO_COLOR: tuple[int, int, int] = (255, 255, 255)  # white
CENTERLINE_RCA_COLOR: tuple[int, int, int] = (255, 160, 60)  # orange
CENTERLINE_LCA_COLOR: tuple[int, int, int] = (60, 200, 255)  # cyan-blue
