from types import SimpleNamespace

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import QFileDialog, QHBoxLayout, QMessageBox, QPushButton, QSplitter, QVBoxLayout, QWidget

from domain.fusion.runtime_types import FusionRuntimeData
from domain.fusion.types import FusionScene
from pages.fusion.controllers.alignment import AlignmentController
from pages.fusion.controllers.branches import BranchController
from pages.fusion.controllers.cutting import CutController
from pages.fusion.controllers.final_mesh import FinalMeshController
from pages.fusion.controllers.fusion import FusionController
from pages.fusion.controllers.geometry import GeometryController
from pages.fusion.controllers.tree import TreeController
from pages.fusion.left_half.left_half import LeftHalf
from pages.fusion.right_half.right_half import RightHalf


class FusionPage(QWidget):
    """Builds the viewer and the columns, wires their signals to one controller per pipeline
    stage, and dispatches the viewer signals several tabs share."""

    def __init__(self, config: SimpleNamespace, status_bar) -> None:
        super().__init__()
        self.config: SimpleNamespace = config
        self.status_bar = status_bar
        self.data = FusionRuntimeData()

        self.left_half = LeftHalf(self)
        self.right_half = RightHalf(self)
        self.ccta_geometry = GeometryController(self)
        self.branches = BranchController(self)
        self.tree = TreeController(self)
        self.alignment = AlignmentController(self)
        self.fusion = FusionController(self)
        self.final_mesh = FinalMeshController(self)
        self.cutting = CutController(self)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.left_half())
        splitter.addWidget(self.right_half())
        # 50/50 between the 3D view and the columns, also when the window is resized.
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([1, 1])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addLayout(self._build_top_bar())
        layout.addWidget(splitter)

        self._connect_signals()

    def shutdown(self) -> None:
        self.final_mesh.shutdown()
        self.left_half.viewer.shutdown()

    # ------------------------------------------------------------------

    def _build_top_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setContentsMargins(6, 4, 6, 4)
        open_case_btn = QPushButton('Open Case Folder…')
        open_case_btn.setToolTip('Sets the default browse folder for the file pickers below')
        open_case_btn.clicked.connect(self._on_open_case)
        bar.addWidget(open_case_btn)
        bar.addStretch(1)
        return bar

    def _on_open_case(self) -> None:
        path = QFileDialog.getExistingDirectory(self, 'Open Case Folder')
        if not path:
            return
        self.data.case_dir = path
        self.right_half.geometry_column.set_default_dir(path)
        self.right_half.intravascular_column.set_default_dir(path)
        self.status_bar.showMessage(f'Case folder: {path}')

    def _on_clear_all_data(self) -> None:
        reply = QMessageBox.question(
            self,
            'Clear All Data',
            'Discard every loaded/computed fusion result (centerlines, vessel tree, '
            'alignment, scaling, stitched mesh) and clear the 3D viewer?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self.data = FusionRuntimeData()
        self.final_mesh.reset_smoothing_state()
        self.fusion.update_removal_undo_button()
        self.cutting.reset()
        self.branches.reset()
        self.right_half.intravascular_column.set_centerline_choices([('RCA', 'rca', 0), ('LCA', 'lca', 0)], 'rca', 0)
        self.right_half.intravascular_column.set_reference_choices([])
        for vessel in ('rca', 'lca'):
            self.left_half.tree_toolbar.set_branches(vessel, [], 0)
            self.left_half.tree_toolbar.set_references(vessel, [], -1)
        for scene in FusionScene:
            self.left_half.viewer.clear_scene(scene)
            self.left_half.refresh_toolbar(scene)
        self.status_bar.showMessage('Fusion data cleared.')

    def _connect_signals(self) -> None:
        lh, viewer = self.left_half, self.left_half.viewer

        gc = self.right_half.geometry_column
        gc.geometry_files_changed.connect(self.ccta_geometry.on_preview)
        gc.prepare_centerlines_requested.connect(self.ccta_geometry.on_prepare_centerlines)
        gc.run_label_geometry_requested.connect(self.ccta_geometry.on_label_geometry)
        gc.run_label_branches_requested.connect(self.branches.on_label_branches)
        gc.run_discretize_tree_requested.connect(self.tree.on_discretize)

        lh.geometry_toolbar.lasso_mode_toggled.connect(self.ccta_geometry.on_lasso_mode_toggled)
        lh.branch_labels_toolbar.lasso_mode_toggled.connect(self.branches.on_lasso_mode_toggled)
        lh.branch_labels_toolbar.rename_requested.connect(self.branches.on_rename_requested)
        lh.branch_labels_toolbar.delete_requested.connect(self.branches.on_delete_requested)
        lh.branch_toolbar.cos_threshold_changed.connect(self.branches.on_cos_threshold_changed)
        lh.branch_toolbar.split_requested.connect(self.branches.on_split_requested)
        lh.branch_toolbar.merge_requested.connect(self.branches.on_merge_requested)
        lh.tree_toolbar.branch_selected.connect(self.tree.select_branch)
        lh.tree_toolbar.reference_selected.connect(self.tree.select_reference)
        viewer.point_picked.connect(self._on_point_picked)
        viewer.lasso_closed.connect(self._on_lasso_closed)
        lh.on_scene_shown(self._on_scene_shown)
        for toolbar in (
            lh.geometry_toolbar,
            lh.branch_toolbar,
            lh.branch_labels_toolbar,
            lh.intravascular_loaded_toolbar,
            lh.alignment_toolbar,
            lh.tree_toolbar,
            lh.cut_toolbar,
        ):
            toolbar.clear_all_data_requested.connect(self._on_clear_all_data)

        ct = lh.cut_toolbar
        ct.source_changed.connect(self.cutting.on_source_changed)
        ct.step_requested.connect(self.cutting.on_step)
        ct.cut_requested.connect(self.cutting.on_cut)
        ct.rename_requested.connect(self.cutting.on_rename)
        ct.remove_requested.connect(self.cutting.on_remove)

        ic = self.right_half.intravascular_column
        ic.run_load_requested.connect(self.alignment.on_load_pullback)
        ic.run_align_requested.connect(self.alignment.on_align)
        ic.run_align_manual_requested.connect(self.alignment.on_align_manual)
        ic.run_label_anomalous_requested.connect(self.alignment.on_label_anomalous)
        ic.reference_centerline_changed.connect(self.tree.on_reference_centerline_changed)
        ic.reference_index_changed.connect(self.tree.on_reference_index_changed)

        fc = self.right_half.fusion_column
        fc.run_compute_scaling_requested.connect(self.fusion.on_compute_scaling)
        fc.run_apply_scaling_requested.connect(self.fusion.on_apply_scaling)
        fc.run_remove_points_requested.connect(self.fusion.on_remove_points)
        fc.undo_remove_points_requested.connect(self.fusion.on_undo_remove_points)
        fc.run_stitch_requested.connect(self.fusion.on_stitch)
        fc.run_remesh_requested.connect(self.final_mesh.on_remesh)
        fc.run_smooth_requested.connect(self.final_mesh.on_smooth)
        fc.export_requested.connect(self.final_mesh.on_export)
        fc.export_cut_requested.connect(self.cutting.on_export)

        gt = lh.geometry_toolbar
        gt.sphere_mode_toggled.connect(self.final_mesh.on_sphere_mode_toggled)
        gt.sphere_radius_changed.connect(self.final_mesh.on_sphere_radius_changed)
        gt.undo_requested.connect(self.final_mesh.on_undo_smoothing)
        viewer.sphere_hovered.connect(self.final_mesh.on_sphere_hovered)
        viewer.sphere_clicked.connect(self.final_mesh.on_sphere_clicked)
        viewer.sphere_radius_step.connect(gt.step_sphere_radius)
        undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        undo_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)  # fusion page only
        undo_shortcut.activated.connect(self._on_undo_shortcut)

    # ------------------------------------------------------------------
    # Viewer signals shared by several tabs
    # ------------------------------------------------------------------

    def _on_scene_shown(self, scene: FusionScene) -> None:
        # The cuts replay lazily, here and on export, not on every upstream mesh change.
        if scene == FusionScene.INLETS_OUTLETS:
            self.cutting.refresh()

    def _on_point_picked(self, x: float, y: float, z: float, scene_value: str) -> None:
        if scene_value == FusionScene.CENTERLINE_BRANCHES.value:
            self.branches.on_marker_picked(x, y, z)
        elif scene_value == FusionScene.INLETS_OUTLETS.value:
            self.cutting.on_point_picked(x, y, z)
        elif scene_value == FusionScene.VESSEL_TREE.value:
            self.tree.on_point_picked(x, y, z)

    def _on_lasso_closed(self) -> None:
        if self.left_half.branch_labels_toolbar.lasso_active():
            self.branches.on_lasso_closed()
        else:
            self.ccta_geometry.on_lasso_closed()

    def _on_undo_shortcut(self) -> None:
        """Ctrl+Z: undo smoothing first, then point removals, but only while nothing is stitched (that
        would drop the stitched mesh). The Undo button next to Remove still does it."""
        if self.data.final_mesh_undo:
            self.final_mesh.on_undo_smoothing()
        elif self.data.stitched is None and self.data.final_mesh is None:
            self.fusion.on_undo_remove_points()
