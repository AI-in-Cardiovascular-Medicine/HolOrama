"""Color legend for the fusion viewer, ported from multimodars' own debug/control
plots (multimodars/ccta/debug_plots.py) so our VTK scenes read the same way the
package's own (now-disabled) matplotlib/trimesh plots would have.
"""

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
    'anomalous_points': (255, 165, 0),  # orange
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


def branch_ramp_color(base: tuple[int, int, int], index: int, count: int) -> tuple[int, int, int]:
    """Shade base color progressively lighter for branch index `index` of `count`
    (mirrors the 4-shade ramps plot_vessel_tree uses for side branches)."""
    if count <= 1:
        return base
    t = index / max(count - 1, 1)
    r, g, b = base
    return (
        int(r + (255 - r) * 0.5 * t),
        int(g + (255 - g) * 0.5 * t),
        int(b + (255 - b) * 0.5 * t),
    )


# Reuse the app's existing diastole/systole palette (see IntravascularPage) so the
# aligned intravascular geometry in the fusion viewer matches the rest of the app.
DIASTOLE_COLOR = (39, 69, 219)
SYSTOLE_COLOR = (209, 55, 38)
