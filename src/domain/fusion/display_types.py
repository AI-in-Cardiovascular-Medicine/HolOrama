"""Color legend for the fusion viewer, matching multimodars' debug plots
(multimodars/ccta/debug_plots.py)."""

from domain.colors import CATEGORICAL_PALETTE, DIASTOLE_COLOR, SYSTOLE_COLOR, branch_ramp_color

__all__ = [
    'REGION_COLORS',
    'REGION_LABELS',
    'BOUNDARY_RING_COLORS',
    'region_label',
    'CENTERLINE_COLORS',
    'TREE_AORTA_COLOR',
    'TREE_RCA_MAIN_COLOR',
    'TREE_LCA_MAIN_COLOR',
    'TREE_CENTROID_COLOR',
    'TREE_REF_COLORS',
    'BRANCH_COLORS_RCA',
    'BRANCH_COLORS_LCA',
    'SHARP_ANGLE_COLOR',
    'SHARP_ANGLE_LABEL_COLOR',
    'POINT_SIZE',
    'CCTA_MESH_COLOR',
    'CCTA_MESH_OPACITY',
    'BRANCH_LABELS_MESH_OPACITY',
    'STITCHED_MESH_COLOR',
    'FINAL_MESH_COLOR',
    'BRANCHES_AORTA_POINT_SIZE',
    'SHARP_ANGLE_POINT_SIZE',
    'TREE_CENTROID_POINT_SIZE',
    'TREE_REF_POINT_SIZE',
    'SELECTED_REF_COLORS',
    'SELECTED_REF_POINT_SIZE',
    'IV_LUMEN_OPACITY',
    'IV_WALL_COLOR',
    'IV_WALL_OPACITY',
    'ALIGNED_CENTERLINE_COLOR',
    'ALIGNED_CCTA_MESH_COLOR',
    'ALIGNED_CCTA_MESH_OPACITY',
    'BOUNDARY_RING_POINT_SIZE',
    'CUT_MESH_COLOR',
    'CUT_MESH_OPACITY',
    'CUT_CAP_COLOR',
    'CUT_SLICE_COLORS',
    'CUT_SELECTED_SLICE_COLOR',
    'CUT_SELECTED_SLICE_LINE_WIDTH',
    'branch_ramp_color',
    'DIASTOLE_COLOR',
    'SYSTOLE_COLOR',
]

# results dict point-cloud keys -> RGB, from plot_results_key().
REGION_COLORS: dict[str, tuple[int, int, int]] = {
    'aorta_points': (255, 255, 0),  # yellow
    'rca_points': (0, 0, 255),  # blue
    'lca_points': (0, 255, 0),  # green
    'rca_removed_points': (255, 0, 0),  # red
    'lca_removed_points': (255, 0, 0),  # red
    'proximal_points': (0, 255, 255),  # cyan
    'distal_points': (255, 0, 255),  # magenta
    'anomalous_points': (255, 165, 0),  # orange
    # Vessel vertices off the pullback path (label_anomalous_region), never removed.
    'off_path_points': (128, 0, 128),  # purple
}

# boundary_points_1, boundary_points_2, ... (the rims left by Remove Labeled Points),
# shown in the Intravascular Aligned scene. Indexed by ring number - 1, wrapping.
BOUNDARY_RING_COLORS: tuple[tuple[int, int, int], ...] = (
    (255, 60, 60),  # red
    (0, 210, 210),  # cyan
    (255, 220, 0),  # yellow
    (220, 60, 220),  # magenta
)

REGION_LABELS: dict[str, str] = {
    'anomalous_points': 'overlap_points',
}


def region_label(key: str) -> str:
    """The user-facing name for a results-dict key (the key itself unless overridden)."""
    return REGION_LABELS.get(key, key)


# Centerline overlay colors, from plot_results_key()'s cl_rca/cl_lca/cl_aorta.
CENTERLINE_COLORS: dict[str, tuple[int, int, int]] = {
    'centerline_aorta': (200, 200, 0),  # olive-yellow
    'centerline_rca': (0, 100, 200),  # blue
    'centerline_lca': (0, 150, 0),  # green
}

# Vessel-tree scene, from plot_vessel_tree().
TREE_AORTA_COLOR = (192, 192, 192)  # silver
TREE_RCA_MAIN_COLOR = (70, 130, 180)  # steel-blue
TREE_LCA_MAIN_COLOR = (255, 127, 80)  # coral
TREE_CENTROID_COLOR = (255, 255, 0)  # yellow
# Reference-triplet colors, by position: main/ostium reference, then the two points that fix rotation.
TREE_REF_COLORS: tuple[tuple[int, int, int], ...] = (
    (255, 0, 0),  # main/ostium ref (red)
    (255, 165, 0),  # secondary ref (orange)
    (255, 0, 255),  # tertiary ref (magenta)
)


# Centerline Branches scene, from plot_centerline_branches()/plot_centerline_edges().
# Drawn from the app's shared categorical palette (every second with shifted start)
BRANCH_COLORS_RCA: tuple[tuple[int, int, int], ...] = CATEGORICAL_PALETTE[0::2]
BRANCH_COLORS_LCA: tuple[tuple[int, int, int], ...] = CATEGORICAL_PALETTE[1::2]
SHARP_ANGLE_COLOR: tuple[int, int, int] = (255, 0, 0)  # red
SHARP_ANGLE_LABEL_COLOR: tuple[int, int, int] = (255, 255, 255)  # white

# Scene styles. Point clouds without an explicit size use the viewer's default.
POINT_SIZE = 4.0  # centerlines, tree contours, branch points, cut slices

# CCTA Geometry / Branch Labels
CCTA_MESH_COLOR = (200, 200, 200)  # light gray
CCTA_MESH_OPACITY = 0.4
BRANCH_LABELS_MESH_OPACITY = 0.3
STITCHED_MESH_COLOR = (230, 180, 60)  # amber
FINAL_MESH_COLOR = (230, 230, 230)  # off-white

# Centerline Branches
BRANCHES_AORTA_POINT_SIZE = 3.0
SHARP_ANGLE_POINT_SIZE = 9.0

# Vessel Tree
TREE_CENTROID_POINT_SIZE = 5.0
TREE_REF_POINT_SIZE = 10.0
SELECTED_REF_COLORS = {'rca': (255, 255, 255), 'lca': (0, 255, 255)}  # white, cyan
SELECTED_REF_POINT_SIZE = 14.0  # also the ostium point in Intravascular Aligned

# Intravascular Loaded / Aligned (lumen color is the phase color)
IV_LUMEN_OPACITY = 0.6
IV_WALL_COLOR = (220, 220, 220)  # light gray
IV_WALL_OPACITY = 0.25
ALIGNED_CENTERLINE_COLOR = (0, 200, 0)  # green
ALIGNED_CCTA_MESH_COLOR = (160, 160, 160)  # gray
ALIGNED_CCTA_MESH_OPACITY = 0.35
BOUNDARY_RING_POINT_SIZE = 8.0

# Inlets & Outlets
CUT_MESH_COLOR = (220, 220, 220)  # light gray
CUT_MESH_OPACITY = 0.5
CUT_CAP_COLOR = (255, 140, 0)  # orange
CUT_SLICE_COLORS = {'aorta': TREE_AORTA_COLOR, 'rca': TREE_RCA_MAIN_COLOR, 'lca': TREE_LCA_MAIN_COLOR}
CUT_SELECTED_SLICE_COLOR = (255, 255, 255)  # white
CUT_SELECTED_SLICE_LINE_WIDTH = 4.0
