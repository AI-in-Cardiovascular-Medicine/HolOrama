from PyQt6.QtWidgets import QTabWidget

from pages.ccta.left_half.segmentation.display import CctaDisplay
from pages.ccta.left_half.segmentation.display_3d import CctaViewer3D
from pages.ccta.left_half.segmentation.views import SegmentationViews
from pages.ccta.left_half.cut_geometry.viewer import CutGeometryViewer3D


class LeftHalf:
    """Manages the left-side tabbed interface: Segmentation (4-view grid) and Cut Geometry."""

    def __init__(
        self,
        label_colors: tuple[tuple[int, int, int], ...],
        mask_alpha: float,
        windowing_sensitivity: float,
        zoom_sensitivity: float,
        parent=None,
    ) -> None:
        self.widget = QTabWidget(parent)

        def display(orientation: str) -> CctaDisplay:
            return CctaDisplay(
                orientation,
                label_colors=label_colors,
                mask_alpha=mask_alpha,
                windowing_sensitivity=windowing_sensitivity,
                zoom_sensitivity=zoom_sensitivity,
            )

        self.axial = display('axial')
        self.coronal = display('coronal')
        self.sagittal = display('sagittal')

        self.segmentation_viewer_3d = CctaViewer3D(label_colors=label_colors)

        self.segmentation_views = SegmentationViews(
            self.axial, self.coronal, self.sagittal, self.segmentation_viewer_3d
        )

        # Cut geometry gets its own tab (own VTK render window) rather than sharing
        # the segmentation 3D view — it has its own mask/picking, unrelated to the
        # per-label segmentation actors shown in "Segmentation".
        self.cut_geometry_viewer = CutGeometryViewer3D()

        self.widget.addTab(self.segmentation_views, 'Segmentation')
        self.widget.addTab(self.cut_geometry_viewer, 'Cut Geometry')

    def __call__(self):
        return self.widget
