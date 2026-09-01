"""Color legend for the fusion viewer, ported from multimodars' own debug/control
plots (multimodars/ccta/debug_plots.py) so our VTK scenes read the same way the
package's own (now-disabled) matplotlib/trimesh plots would have.
"""

from domain.colors import CATEGORICAL_PALETTE, DIASTOLE_COLOR, SYSTOLE_COLOR, branch_ramp_color

__all__ = [
    'REGION_COLORS',
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

# results dict point-cloud keys -> RGB, from plot_results_key(). Region keys not listed
# here (rca_points_main/side_N, boundary_points, prox/dist_boundary_points) aren't part
# of the documented legend and aren't visualized yet.
REGION_COLORS: dict[str, tuple[int, int, int]] = {
    'aorta_points': (255, 255, 0),  # yellow
    'rca_points': (0, 0, 255),  # blue
    'lca_points': (0, 255, 0),  # green
    'rca_removed_points': (255, 0, 0),  # red
    'lca_removed_points': (255, 0, 0),  # red — upstream uses the same key/color for both
    'proximal_points': (0, 255, 255),  # cyan
    'distal_points': (255, 0, 255),  # magenta
    'overlap_points': (255, 165, 0),  # orange
}

# Centerline overlay colors, from plot_results_key()'s cl_rca/cl_lca/cl_aorta.
CENTERLINE_COLORS: dict[str, tuple[int, int, int]] = {
    'centerline_aorta': (200, 200, 0),
    'centerline_rca': (0, 100, 200),
    'centerline_lca': (0, 150, 0),
}

# Vessel-tree scene, from plot_vessel_tree().
TREE_AORTA_COLOR = (192, 192, 192)  # silver
TREE_RCA_MAIN_COLOR = (70, 130, 180)  # steel-blue
TREE_LCA_MAIN_COLOR = (255, 127, 80)  # coral
TREE_CENTROID_COLOR = (255, 255, 0)  # yellow
# Reference-triplet colors. multimodars' own plotting code and this app's alignment code
# both treat position 0 as the main/ostium reference; positions 1-2 as off-axis references
# used only to fix rotation, not because their CW/CCW handedness is guaranteed — the
# Rust->Python binding and every Python consumer disagree on which is which. Don't read
# these two colors as authoritative clock/counterclock; they're just visually distinct.
TREE_REF_COLORS: tuple[tuple[int, int, int], ...] = (
    (255, 0, 0),  # main/ostium ref — red
    (255, 165, 0),  # secondary ref — orange
    (255, 0, 255),  # tertiary ref — magenta
)


# Centerline Branches scene, from plot_centerline_branches()/plot_centerline_edges().
# Drawn from the app's shared categorical palette (interleaved so RCA/LCA — both shown
# in the same scene — never land on the same color) rather than two separately hand-picked
# 5-color sets, which also gives each vessel more distinct branch colors before any ramp
# shading has to repeat.
BRANCH_COLORS_RCA: tuple[tuple[int, int, int], ...] = CATEGORICAL_PALETTE[0::2]
BRANCH_COLORS_LCA: tuple[tuple[int, int, int], ...] = CATEGORICAL_PALETTE[1::2]
SHARP_ANGLE_COLOR: tuple[int, int, int] = (255, 0, 0)
SHARP_ANGLE_LABEL_COLOR: tuple[int, int, int] = (255, 255, 255)
