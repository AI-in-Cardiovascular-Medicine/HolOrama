from domain.colors import DEFAULT_MASK_ALPHA as DEFAULT_MASK_ALPHA

# The names and colours of the labels are the CCTA presets' (src/presets/ccta, domain.ccta.presets).

DEFAULT_CT_LEVEL: int = 200  # HU center — cardiac soft tissue
DEFAULT_CT_WIDTH: int = 700  # HU range

# Post-cut geometry layer (Build Cut Geometry / Smooth / outlet points)
CUT_MESH_COLOR: tuple[int, int, int] = (210, 210, 210)  # light gray, distinct from the label colours
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
