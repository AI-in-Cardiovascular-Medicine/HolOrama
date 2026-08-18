from enum import Enum


class FusionScene(Enum):
    """The scenes shown in the fusion page's shared 3D viewer (one VTK renderer,
    actors grouped per scene and swapped on tab change)."""

    CCTA_GEOMETRY = 'ccta_geometry'
    CENTERLINE_BRANCHES = 'centerline_branches'
    INTRAVASCULAR_LOADED = 'intravascular_loaded'
    INTRAVASCULAR_ALIGNED = 'intravascular_aligned'
    VESSEL_TREE = 'vessel_tree'

    @classmethod
    def label(cls, scene: 'FusionScene') -> str:
        mapping = {
            cls.CCTA_GEOMETRY: 'CCTA Geometry',
            cls.CENTERLINE_BRANCHES: 'Centerline Branches',
            cls.INTRAVASCULAR_LOADED: 'Intravascular Loaded',
            cls.INTRAVASCULAR_ALIGNED: 'Intravascular Aligned',
            cls.VESSEL_TREE: 'Vessel Tree',
        }
        return mapping[scene]
