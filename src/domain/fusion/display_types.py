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
