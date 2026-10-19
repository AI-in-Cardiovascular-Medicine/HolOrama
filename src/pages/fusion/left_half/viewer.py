from dataclasses import dataclass, field

import numpy as np
import trimesh
import vtkmodules.vtkInteractionStyle  # noqa: F401
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
from PyQt6.QtCore import QEvent, QPoint, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import QVBoxLayout, QWidget
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.util import numpy_support
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkFiltersCore import vtkTriangleFilter
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkBillboardTextActor3D,
    vtkCellPicker,
    vtkLightKit,
    vtkPolyDataMapper,
    vtkProp3D,
    vtkRenderer,
)

from domain.fusion.types import FusionScene
from pages.intravascular.popup_windows.message_boxes import ErrorMessage
from tools.lasso import Lasso2D, project_world_batch
from tools.sphere_smooth import SphereBrush


@dataclass
class _Layer:
    # One mesh/points/polyline actor, or a list of text actors for a label layer.
    # Label actors have no GetProperty(), so opacity is skipped for lists.
    actor: vtkActor | list[vtkProp3D]
    visible: bool = True
    opacity: float = 1.0
    color: tuple[int, int, int] = (200, 200, 200)


@dataclass
class _SceneLayers:
    layers: dict[str, _Layer] = field(default_factory=dict)


def _mesh_to_polydata(mesh: trimesh.Trimesh) -> vtkPolyData:
    pts = vtkPoints()
    pts.SetData(numpy_support.numpy_to_vtk(np.ascontiguousarray(mesh.vertices, dtype=np.float64)))

    faces = np.asarray(mesh.faces, dtype=np.int64)
    cells = np.hstack([np.full((len(faces), 1), 3, dtype=np.int64), faces]).ravel()
    id_array = numpy_support.numpy_to_vtkIdTypeArray(cells, deep=True)
    cell_array = vtkCellArray()
    cell_array.SetCells(len(faces), id_array)

    poly = vtkPolyData()
    poly.SetPoints(pts)
    poly.SetPolys(cell_array)

    tri = vtkTriangleFilter()
    tri.SetInputData(poly)
    tri.Update()
    return tri.GetOutput()


def _points_to_polydata(points: np.ndarray, as_polyline: bool) -> vtkPolyData:
    pts = vtkPoints()
    pts.SetData(numpy_support.numpy_to_vtk(np.ascontiguousarray(points, dtype=np.float64)))

    n = len(points)
    cells = vtkCellArray()
    if as_polyline:
        cells.InsertNextCell(n)
        for i in range(n):
            cells.InsertCellPoint(i)
    else:
        for i in range(n):
            cells.InsertNextCell(1)
            cells.InsertCellPoint(i)

    poly = vtkPolyData()
    poly.SetPoints(pts)
    if as_polyline:
        poly.SetLines(cells)
    else:
        poly.SetVerts(cells)
    return poly


class FusionViewer3D(QWidget):
    """Single shared VTK renderer for all fusion scenes. Layers are keyed by (scene, key).
    set_scene() only toggles actor visibility, it never tears down the GL context."""

    point_picked = pyqtSignal(float, float, float, str)  # x, y, z, scene.value
    lasso_closed = pyqtSignal()  # lasso polygon closed, caller decides what "inside" means
    # Sphere brush: hovered/clicked surface point, and Ctrl+wheel radius steps (+1/-1).
    # The caller owns the radius value.
    sphere_hovered = pyqtSignal(float, float, float)
    sphere_clicked = pyqtSignal(float, float, float)
    sphere_radius_step = pyqtSignal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self._vtk_widget = QVTKRenderWindowInteractor(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._vtk_widget)

        self._ren = vtkRenderer()
        self._ren.SetBackground(0.0, 0.0, 0.0)
        self._ren.AutomaticLightCreationOff()

        light_kit = vtkLightKit()
        light_kit.SetKeyLightElevation(30)
        light_kit.SetKeyLightAzimuth(-60)
        light_kit.SetKeyLightIntensity(1.0)
        light_kit.SetFillLightWarmth(0.4)
        light_kit.SetBackLightWarmth(0.35)
        light_kit.AddLightsToRenderer(self._ren)

        self._vtk_widget.GetRenderWindow().AddRenderer(self._ren)
        self._vtk_widget.Initialize()
        self._vtk_widget.Start()
        trackball = vtkInteractorStyleTrackballCamera()
        trackball.SetMotionFactor(7.5)
        self._vtk_widget.SetInteractorStyle(trackball)

        self._picker = vtkCellPicker()
        self._picker.SetTolerance(0.005)
        self._pick_mode = False
        self._press_qt = QPoint()
        self._vtk_widget.installEventFilter(self)

        self._lasso_mode = False
        self._lasso = Lasso2D(self._ren)

        # Sphere brush: hover preview (affected vertices come back via set_sphere_highlight)
        # and click-to-apply, picking only against the target layer's actor.
        self._sphere_mode = False
        self._sphere_target: tuple[FusionScene, str] | None = None
        self._sphere_brush = SphereBrush(self._ren)
        # Hover picks are coalesced via this timer (one pick per ~30 ms, not per mouse move).
        self._sphere_hover_pos = QPoint()
        self._sphere_hover_timer = QTimer(self)
        self._sphere_hover_timer.setSingleShot(True)
        self._sphere_hover_timer.setInterval(30)
        self._sphere_hover_timer.timeout.connect(self._on_sphere_hover_timer)
        self._sphere_press_qt = QPoint()

        self._scenes: dict[FusionScene, _SceneLayers] = {scene: _SceneLayers() for scene in FusionScene}
        self._current_scene: FusionScene = FusionScene.CCTA_GEOMETRY

    # ------------------------------------------------------------------
    # Layer management
    # ------------------------------------------------------------------

    def add_mesh(
        self,
        scene: FusionScene,
        key: str,
        mesh: trimesh.Trimesh,
        color: tuple[int, int, int] = (200, 200, 200),
        opacity: float = 1.0,
        visible: bool = True,
    ) -> None:
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(_mesh_to_polydata(mesh))
        self._add_actor(scene, key, mapper, color, opacity, visible)

    def add_polyline(
        self,
        scene: FusionScene,
        key: str,
        points: np.ndarray,
        color: tuple[int, int, int] = (255, 255, 0),
        line_width: float = 2.0,
        visible: bool = True,
        on_top: bool = False,
    ) -> None:
        """on_top draws the line in front of coincident geometry (e.g. a highlighted contour)
        via a large polygon offset towards the camera."""
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(_points_to_polydata(points, as_polyline=True))
        if on_top:
            mapper.SetResolveCoincidentTopologyToPolygonOffset()
            mapper.SetRelativeCoincidentTopologyLineOffsetParameters(0, -66000)
        actor = self._add_actor(scene, key, mapper, color, 1.0, visible)
        actor.GetProperty().SetLineWidth(line_width)

    def add_points(
        self,
        scene: FusionScene,
        key: str,
        points: np.ndarray,
        color: tuple[int, int, int] = (255, 0, 0),
        size: float = 6.0,
        visible: bool = True,
    ) -> None:
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(_points_to_polydata(points, as_polyline=False))
        actor = self._add_actor(scene, key, mapper, color, 1.0, visible)
        actor.GetProperty().SetPointSize(size)

    def add_labels(
        self,
        scene: FusionScene,
        key: str,
        points: np.ndarray,
        texts: list[str],
        color: tuple[int, int, int] = (255, 255, 255),
        font_size: int = 14,
        visible: bool = True,
    ) -> None:
        """One camera-facing text actor per (point, text) pair."""
        is_first_layer_in_scene = not self._scenes[scene].layers

        # Re-adding a layer keeps the user's visibility (see _add_actor).
        existing = self._scenes[scene].layers.get(key)
        if existing is not None:
            visible = existing.visible

        self.remove_layer(scene, key)
        group: list[vtkProp3D] = []
        r, g, b = color
        for point, text in zip(points, texts):
            actor = vtkBillboardTextActor3D()
            actor.SetPosition(float(point[0]), float(point[1]), float(point[2]))
            actor.SetInput(str(text))
            text_prop = actor.GetTextProperty()
            text_prop.SetColor(r / 255.0, g / 255.0, b / 255.0)
            text_prop.SetFontSize(font_size)
            text_prop.SetJustificationToCentered()
            text_prop.SetBold(True)
            actor.SetVisibility(int(visible and scene == self._current_scene))
            self._ren.AddActor(actor)
            group.append(actor)
        self._scenes[scene].layers[key] = _Layer(actor=group, visible=visible, opacity=1.0, color=color)
        if is_first_layer_in_scene and scene == self._current_scene:
            self._ren.ResetCamera()
        self._vtk_widget.GetRenderWindow().Render()

    def _add_actor(self, scene, key, mapper, color, opacity, visible) -> vtkActor:
        # Fit the camera only for a scene's first layer. Re-fitting on later adds would
        # reset the user's zoom on every update.
        is_first_layer_in_scene = not self._scenes[scene].layers

        # Re-adding a key keeps the user's visibility/opacity. The caller's defaults apply
        # only to the first add, otherwise every pipeline step would reset the toolbar.
        existing = self._scenes[scene].layers.get(key)
        if existing is not None:
            visible = existing.visible
            opacity = existing.opacity

        self.remove_layer(scene, key)
        actor = vtkActor()
        actor.SetMapper(mapper)
        r, g, b = color
        actor.GetProperty().SetColor(r / 255.0, g / 255.0, b / 255.0)
        actor.GetProperty().SetOpacity(opacity)
        actor.SetVisibility(int(visible and scene == self._current_scene))
        self._ren.AddActor(actor)
        self._scenes[scene].layers[key] = _Layer(actor=actor, visible=visible, opacity=opacity, color=color)
        if is_first_layer_in_scene and scene == self._current_scene:
            self._ren.ResetCamera()
        self._vtk_widget.GetRenderWindow().Render()
        return actor

    def remove_layer(self, scene: FusionScene, key: str) -> None:
        layer = self._scenes[scene].layers.pop(key, None)
        if layer is not None:
            for actor in layer.actor if isinstance(layer.actor, list) else [layer.actor]:
                self._ren.RemoveActor(actor)

    def clear_scene(self, scene: FusionScene) -> None:
        for key in list(self._scenes[scene].layers):
            self.remove_layer(scene, key)
        self._vtk_widget.GetRenderWindow().Render()

    def set_layer_visible(self, scene: FusionScene, key: str, visible: bool) -> None:
        layer = self._scenes[scene].layers.get(key)
        if layer is None:
            return
        layer.visible = visible
        if scene == self._current_scene:
            for actor in layer.actor if isinstance(layer.actor, list) else [layer.actor]:
                actor.SetVisibility(int(visible))
            self._vtk_widget.GetRenderWindow().Render()

    def isolate_layer(self, scene: FusionScene, key: str) -> None:
        """Hide every layer in `scene` except `key`. Updates the stored .visible flags too,
        so a toolbar refresh shows the right checkboxes."""
        for k, layer in self._scenes[scene].layers.items():
            layer.visible = k == key
            if scene == self._current_scene:
                for actor in layer.actor if isinstance(layer.actor, list) else [layer.actor]:
                    actor.SetVisibility(int(layer.visible))
        if scene == self._current_scene:
            self._vtk_widget.GetRenderWindow().Render()

    def set_layer_opacity(self, scene: FusionScene, key: str, opacity: float) -> None:
        layer = self._scenes[scene].layers.get(key)
        if layer is None:
            return
        layer.opacity = opacity
        # Text-label layers have no vtkProperty, so they only support visibility.
        if not isinstance(layer.actor, list):
            layer.actor.GetProperty().SetOpacity(opacity)
        if scene == self._current_scene:
            self._vtk_widget.GetRenderWindow().Render()

    def layer_states(self, scene: FusionScene) -> dict[str, tuple[bool, float, tuple[int, int, int]]]:
        """(visible, opacity, color) per layer key, for initializing toolbar controls."""
        return {key: (layer.visible, layer.opacity, layer.color) for key, layer in self._scenes[scene].layers.items()}

    # ------------------------------------------------------------------
    # Scene switching
    # ------------------------------------------------------------------

    def set_scene(self, scene: FusionScene) -> None:
        self._current_scene = scene
        self.hide_sphere_preview(render=False)
        for s, scene_layers in self._scenes.items():
            for layer in scene_layers.layers.values():
                visible = int(layer.visible and s == scene)
                for actor in layer.actor if isinstance(layer.actor, list) else [layer.actor]:
                    actor.SetVisibility(visible)
        self._ren.ResetCamera()
        self._vtk_widget.GetRenderWindow().Render()

    def reset_camera(self) -> None:
        self._ren.ResetCamera()
        self._vtk_widget.GetRenderWindow().Render()

    def shutdown(self) -> None:
        """Finalize the VTK OpenGL context while the HWND is still valid."""
        rw = self._vtk_widget.GetRenderWindow()
        rw.Finalize()
        rw.SetInteractor(None)

    # ------------------------------------------------------------------
    # Point picking (for reference-point / measurement tools)
    # ------------------------------------------------------------------

    def set_pick_mode(self, enabled: bool) -> None:
        self._pick_mode = enabled

    def eventFilter(self, obj, event) -> bool:
        if obj is self._vtk_widget:
            if self._sphere_mode and self._sphere_target is not None and self._sphere_target[0] == self._current_scene:
                if self._sphere_event(event):
                    return True
            elif self._lasso_mode:
                if event.type() == QEvent.Type.MouseButtonPress:
                    vtk_y = self._vtk_widget.height() - 1 - event.pos().y()
                    if event.button() == Qt.MouseButton.LeftButton:
                        if self._lasso.add_point(event.pos().x(), vtk_y):
                            self.lasso_closed.emit()
                        else:
                            self._lasso.redraw()
                            self._vtk_widget.GetRenderWindow().Render()
                        return True  # block VTK camera move
                    if event.button() == Qt.MouseButton.RightButton:
                        if len(self._lasso.points) >= 3:
                            self.lasso_closed.emit()
                        else:
                            ErrorMessage(self, 'Draw at least 3 points to define a lasso.')
                        return True
            elif self._pick_mode:
                if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                    self._press_qt = event.pos()
                elif event.type() == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
                    dp = event.pos() - self._press_qt
                    if abs(dp.x()) <= 3 and abs(dp.y()) <= 3:
                        vtk_y = self._vtk_widget.height() - 1 - event.pos().y()
                        if self._picker.Pick(event.pos().x(), vtk_y, 0, self._ren):
                            x, y, z = self._picker.GetPickPosition()
                            self.point_picked.emit(x, y, z, self._current_scene.value)
        return super().eventFilter(obj, event)

    # ------------------------------------------------------------------
    # Lasso (reclassify points between labels, see tools.lasso)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Sphere brush (local smoothing)
    # ------------------------------------------------------------------

    def set_sphere_mode(self, enabled: bool, scene: FusionScene | None = None, key: str | None = None) -> None:
        """Turn the sphere brush on for layer `key` of `scene`, or off. Camera controls keep
        working while it's on."""
        self._sphere_mode = enabled
        self._sphere_target = (scene, key) if enabled and scene is not None and key is not None else None
        if not enabled:
            self._sphere_hover_timer.stop()
            self.hide_sphere_preview()

    def set_sphere_radius(self, radius_mm: float) -> None:
        self._sphere_brush.set_radius(radius_mm)
        if self._sphere_brush.visible:
            self._vtk_widget.GetRenderWindow().Render()

    def set_sphere_highlight(self, points: np.ndarray) -> None:
        """Show `points` (N, 3) as the vertices the brush would move at the hovered spot."""
        self._sphere_brush.set_highlight(points)
        self._vtk_widget.GetRenderWindow().Render()

    def hide_sphere_preview(self, render: bool = True) -> None:
        self._sphere_brush.hide()
        if render:
            self._vtk_widget.GetRenderWindow().Render()

    def refresh_sphere_hover(self) -> None:
        """Re-run the hover pick at the last cursor position (e.g. after the mesh changed)."""
        if self._sphere_mode:
            self._sphere_hover_timer.start()

    def _sphere_event(self, event) -> bool:
        """Handle one widget event in sphere mode. Returns True if consumed (VTK never sees it).
        Presses/drags pass through so the camera still rotates. A click moves at most 3 px."""
        etype = event.type()
        if etype == QEvent.Type.MouseMove:
            if event.buttons() == Qt.MouseButton.NoButton:
                self._sphere_hover_pos = event.pos()
                self._sphere_hover_timer.start()
            elif self._sphere_brush.visible:
                self.hide_sphere_preview()  # rotating/panning, the preview would lag behind
        elif etype == QEvent.Type.Leave:
            self._sphere_hover_timer.stop()
            self.hide_sphere_preview()
        elif etype == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            self._sphere_press_qt = event.pos()
        elif etype == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
            dp = event.pos() - self._sphere_press_qt
            if abs(dp.x()) <= 3 and abs(dp.y()) <= 3:
                hit = self._sphere_pick(event.pos())
                if hit is not None:
                    self.sphere_clicked.emit(*hit)
        elif etype == QEvent.Type.Wheel and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            if delta:
                self.sphere_radius_step.emit(1 if delta > 0 else -1)
            return True  # Ctrl+wheel resizes the brush instead of zooming
        return False

    def _sphere_pick(self, pos: QPoint) -> tuple[float, float, float] | None:
        if self._sphere_target is None:
            return None
        scene, key = self._sphere_target
        layer = self._scenes[scene].layers.get(key)
        if layer is None or isinstance(layer.actor, list) or not layer.visible:
            return None
        return self._sphere_brush.pick(layer.actor, pos.x(), self._vtk_widget.height() - 1 - pos.y())

    def _on_sphere_hover_timer(self) -> None:
        if not self._sphere_mode:
            return
        hit = self._sphere_pick(self._sphere_hover_pos)
        if hit is None:
            self.hide_sphere_preview()
            return
        self._sphere_brush.show_at(hit)
        self.sphere_hovered.emit(*hit)  # the caller answers with set_sphere_highlight (which renders)

    def set_lasso_mode(self, enabled: bool) -> None:
        self._lasso_mode = enabled
        if not enabled:
            self._lasso.clear()
            self._vtk_widget.GetRenderWindow().Render()

    def clear_lasso(self) -> None:
        self._lasso.clear()
        self._vtk_widget.GetRenderWindow().Render()

    def points_inside_lasso(self, points: np.ndarray) -> np.ndarray:
        """Boolean mask of `points` (N, 3) inside the closed lasso. Call after lasso_closed."""
        screen = project_world_batch(
            self._ren, self._vtk_widget.GetRenderWindow().GetSize(), points[:, 0], points[:, 1], points[:, 2]
        )
        return self._lasso.contains(screen)
