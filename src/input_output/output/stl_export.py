"""Export a combined binary mask as NIfTI or binary STL."""

import struct
import numpy as np
import SimpleITK as sitk
from skimage.measure import marching_cubes


def export_nifti(mask: np.ndarray, voxel_spacing: tuple[float, float, float], output_path: str) -> None:
    """Write a binary mask as NIfTI .nii.gz, preserving voxel spacing."""
    dz, dy, dx = voxel_spacing
    img = sitk.GetImageFromArray(mask.astype(np.uint8))
    img.SetSpacing((dx, dy, dz))  # sitk order: (x, y, z)
    sitk.WriteImage(img, output_path)


def export_stl(mask: np.ndarray, voxel_spacing: tuple[float, float, float], output_path: str) -> None:
    """Run marching cubes on the binary mask and write a binary STL."""
    verts, faces, _, _ = marching_cubes(mask > 0, level=0.5, spacing=voxel_spacing)
    _write_binary_stl(verts, faces, output_path)


def _write_binary_stl(verts: np.ndarray, faces: np.ndarray, path: str) -> None:
    n_tri = len(faces)
    with open(path, 'wb') as f:
        f.write(b'\0' * 80)  # 80-byte header
        f.write(struct.pack('<I', n_tri))
        for face in faces:
            v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
            normal = np.cross(v1 - v0, v2 - v0)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal /= norm
            f.write(struct.pack('<fff', *normal))
            f.write(struct.pack('<fff', *v0))
            f.write(struct.pack('<fff', *v1))
            f.write(struct.pack('<fff', *v2))
            f.write(struct.pack('<H', 0))  # attribute byte count
