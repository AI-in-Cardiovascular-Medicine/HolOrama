"""Export a combined binary mask as NIfTI or STL (ASCII default, binary available)."""

import struct

import numpy as np
import SimpleITK as sitk
from skimage.measure import marching_cubes

from domain.io_types import CANONICAL_ORIENTATION, VolumeGeometry


def export_nifti(mask: np.ndarray, geometry: VolumeGeometry, output_path: str) -> None:
    """Write a mask drawn on a canonicalized volume as NIfTI (.nii / .nii.gz).

    The mask is put back into `geometry`'s source orientation before writing, so the file
    lands on the exact voxel grid of the image it was drawn on — and so it overlays that
    image both here (the readers canonicalize it right back) and in any other viewer.
    """
    img = sitk.GetImageFromArray(mask.astype(np.uint8))  # (Z, Y, X) -> sitk (X, Y, Z)
    img.SetOrigin(geometry.origin)
    img.SetSpacing(geometry.spacing)
    img.SetDirection(geometry.direction)
    if geometry.source_orientation != CANONICAL_ORIENTATION:
        img = sitk.DICOMOrient(img, geometry.source_orientation)
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
    _write_ascii_stl(verts, faces, output_path)


def _write_ascii_stl(verts: np.ndarray, faces: np.ndarray, path: str) -> None:
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0).astype(np.float32)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.where(norms > 0, norms, 1.0)

    with open(path, 'w') as f:
        f.write('solid mesh\n')
        for n, a, b, c in zip(normals, v0, v1, v2):
            f.write(f'  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n')
            f.write('    outer loop\n')
            f.write(f'      vertex {a[0]:.6e} {a[1]:.6e} {a[2]:.6e}\n')
            f.write(f'      vertex {b[0]:.6e} {b[1]:.6e} {b[2]:.6e}\n')
            f.write(f'      vertex {c[0]:.6e} {c[1]:.6e} {c[2]:.6e}\n')
            f.write('    endloop\n')
            f.write('  endfacet\n')
        f.write('endsolid mesh\n')


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
