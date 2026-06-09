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
    """Run marching cubes on the binary mask and write a binary STL.

    A 1-voxel zero-padding is added before marching cubes so every surface
    has a closed exterior even when the mask touches the volume boundary.
    The padding offset is subtracted from vertex coordinates afterwards.
    """
    _, dy, _ = voxel_spacing
    padded = np.pad(mask > 0, pad_width=1, mode='constant', constant_values=0)
    verts, faces, _, _ = marching_cubes(padded, level=0.5, spacing=voxel_spacing)
    verts -= np.array(voxel_spacing)  # undo 1-voxel padding offset

    y_ext = (mask.shape[1] - 1) * dy
    verts = np.column_stack(
        [
            verts[:, 2],
            y_ext - verts[:, 1],
            verts[:, 0],
        ]
    )
    faces = faces[:, [0, 2, 1]]
    _write_binary_stl(verts, faces, output_path)


def _write_binary_stl(verts: np.ndarray, faces: np.ndarray, path: str) -> None:
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0).astype(np.float32)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.where(norms > 0, norms, 1.0)

    # Pack all triangles at once: 50 bytes each (12 normal + 36 verts + 2 attr)
    buf = np.zeros(
        len(faces), dtype=[('n', '<f4', 3), ('v0', '<f4', 3), ('v1', '<f4', 3), ('v2', '<f4', 3), ('a', '<u2')]
    )
    buf['n'] = normals
    buf['v0'] = v0.astype(np.float32)
    buf['v1'] = v1.astype(np.float32)
    buf['v2'] = v2.astype(np.float32)

    with open(path, 'wb') as f:
        f.write(b'\0' * 80)
        f.write(struct.pack('<I', len(faces)))
        f.write(buf.tobytes())
