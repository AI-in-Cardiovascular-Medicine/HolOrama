from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class MetaDataCCTA:
    modality: str = 'CCTA'
    patient_name: str = 'Unknown'
    birthdate: str = 'Unknown'
    sex: str = 'Unknown'
    slice_thickness: float = 0.0
    pixel_spacing: Tuple[float, float] = (0.0, 0.0)
    manufacturer: str = 'Unknown'
    model: str = 'Unknown'
    raw_tags: dict = field(default_factory=dict)  # all remaining DICOM / NIfTI tags
    ...


CANONICAL_ORIENTATION = 'LAS'

# Direction cosines of CANONICAL_ORIENTATION, in SimpleITK's row-major order.
CANONICAL_DIRECTION: Tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True)
class VolumeGeometry:
    """Voxel-grid geometry of a loaded CCTA volume, in the app's canonical orientation."""

    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # sitk order: (x, y, z)
    direction: Tuple[float, ...] = CANONICAL_DIRECTION
    source_orientation: str = CANONICAL_ORIENTATION  # the file's, so masks are written back the same way


def geometry_from_spacing(voxel_spacing: Tuple[float, float, float]) -> VolumeGeometry:
    """Canonical geometry with only `voxel_spacing` (dz, dy, dx), for when the source's is unknown."""
    dz, dy, dx = voxel_spacing
    return VolumeGeometry(spacing=(dx, dy, dz))
