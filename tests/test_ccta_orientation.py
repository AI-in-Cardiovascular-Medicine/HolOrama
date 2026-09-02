"""Tests for how CCTA volumes and masks are oriented on the way in and out
(input_output.input.ccta_io, input_output.output.stl_export.export_nifti).

The displays index the volume as (z, y, x) and flip axes by hand, which only reads
correctly when the in-memory array is in CANONICAL_ORIENTATION ('LAS'). Files arrive in
whatever orientation they were written in — plain axial CT and most TotalSegmentator
output are 'LPS', which displays flipped front-to-back — so every reader canonicalizes
and every writer reverses it.

Volumes here are asymmetric ramps: every axis carries a different gradient, so any missed
flip or transposed pair of axes changes the array rather than leaving it looking plausible.
"""

import numpy as np
import pytest
import SimpleITK as sitk

from domain.io_types import CANONICAL_ORIENTATION, VolumeGeometry, geometry_from_spacing
from input_output.input.ccta_io import read_ct_volume, read_mask_volume, read_nifti_volume
from input_output.output.stl_export import export_nifti

SPACING = (0.4, 0.5, 0.8)  # sitk order (x, y, z) — all different, so a transpose shows up
ORIGIN = (-72.9, -90.8, 1639.6)


def _ramp(shape=(9, 7, 5)) -> np.ndarray:
    """A (Z, Y, X) volume whose value encodes its own index on every axis."""
    z, y, x = np.indices(shape)
    return (100 * z + 10 * y + x).astype(np.int16)


def _write(tmp_path, name, array, orientation, spacing=SPACING, origin=ORIGIN):
    """Write `array` (in `orientation`'s index order) as a NIfTI file, returning its path."""
    img = sitk.GetImageFromArray(array)
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    img.SetDirection(sitk.DICOMOrientImageFilter.GetDirectionCosinesFromOrientation(orientation))
    path = str(tmp_path / name)
    sitk.WriteImage(img, path)
    return path


def _in_orientation(array, orientation):
    """`array` (canonical order) re-expressed in `orientation`'s index order."""
    img = sitk.GetImageFromArray(array)
    img.SetSpacing(SPACING)
    img.SetOrigin(ORIGIN)
    img.SetDirection(sitk.DICOMOrientImageFilter.GetDirectionCosinesFromOrientation(CANONICAL_ORIENTATION))
    return sitk.GetArrayFromImage(sitk.DICOMOrient(img, orientation))


# Every orientation a CT is realistically stored in, plus the canonical one itself.
ORIENTATIONS = ['LAS', 'LPS', 'RAS', 'RAI', 'LPI', 'PSL']


@pytest.mark.parametrize('orientation', ORIENTATIONS)
def test_volume_is_canonicalized_whatever_the_file_says(tmp_path, orientation):
    """Reading the same anatomy stored in any orientation yields the same array."""
    canonical = _ramp()
    path = _write(tmp_path, 'vol.nii.gz', _in_orientation(canonical, orientation), orientation)

    volume, metadata = read_nifti_volume(path)

    assert np.array_equal(volume, canonical)
    assert metadata['geometry'].source_orientation == orientation


@pytest.mark.parametrize('orientation', ORIENTATIONS)
def test_mask_is_canonicalized_whatever_the_file_says(tmp_path, orientation):
    mask = (_ramp() % 4).astype(np.uint8)
    path = _write(tmp_path, 'mask.nii.gz', _in_orientation(mask, orientation), orientation)

    read, metadata = read_mask_volume(path)

    assert np.array_equal(read, mask)
    assert metadata['geometry'].source_orientation == orientation


def test_mask_lines_up_with_a_volume_stored_the_other_way_round(tmp_path):
    """A mask and the image it segments need not agree on orientation — an LPS
    TotalSegmentator mask over an LAS image is the case that motivated this."""
    canonical = _ramp()
    vol_path = _write(tmp_path, 'vol.nii.gz', _in_orientation(canonical, 'LAS'), 'LAS')
    mask_path = _write(tmp_path, 'mask.nii.gz', _in_orientation((canonical % 4).astype(np.uint8), 'LPS'), 'LPS')

    volume, _ = read_nifti_volume(vol_path)
    mask, _ = read_mask_volume(mask_path)

    assert mask.shape == volume.shape
    assert np.array_equal(mask, (volume % 4).astype(np.uint8))


@pytest.mark.parametrize('orientation', ORIENTATIONS)
def test_spacing_follows_the_axes_it_belongs_to(tmp_path, orientation):
    """Reorienting a sagittally-stored volume permutes its spacings too, so the reported
    pixel spacing / slice thickness must come off the canonicalized grid."""
    path = _write(tmp_path, 'vol.nii.gz', _in_orientation(_ramp(), orientation), orientation)

    _, metadata = read_nifti_volume(path)

    dz = metadata['slice_thickness']
    dy, dx = metadata['pixel_spacing']
    assert sorted((dx, dy, dz)) == pytest.approx(sorted(SPACING))
    assert (dx, dy, dz) == pytest.approx(metadata['geometry'].spacing)


@pytest.mark.parametrize('orientation', ORIENTATIONS)
def test_saved_mask_round_trips_and_keeps_the_source_grid(tmp_path, orientation):
    """A mask drawn here is written back the way the image was stored, so it overlays that
    image in other viewers — and comes back unchanged when this app reads it again."""
    vol_path = _write(tmp_path, 'vol.nii.gz', _in_orientation(_ramp(), orientation), orientation)
    volume, metadata = read_nifti_volume(vol_path)
    geometry = metadata['geometry']
    mask = np.zeros(volume.shape, np.uint8)
    mask[1:3, 2:5, 0:2] = 1  # asymmetric on every axis
    mask[7, 6, 4] = 2

    out_path = str(tmp_path / 'mask_out.nii.gz')
    export_nifti(mask, geometry, out_path)

    source, written = sitk.ReadImage(vol_path), sitk.ReadImage(out_path)
    assert written.GetSize() == source.GetSize()
    assert written.GetOrigin() == pytest.approx(source.GetOrigin())
    assert written.GetSpacing() == pytest.approx(source.GetSpacing())
    assert written.GetDirection() == pytest.approx(source.GetDirection())
    assert np.array_equal(read_mask_volume(out_path)[0], mask)


@pytest.mark.parametrize('orientation', ORIENTATIONS)
def test_masks_saved_before_geometry_was_preserved_still_line_up(tmp_path, orientation):
    """Masks this app wrote earlier carry sitk's placeholder grid (identity direction,
    zero origin) with voxels in the order the volume file happened to be stored in.
    Given the volume's geometry they must land on the canonicalized volume; without it
    they must at least come back as they were written."""
    canonical = _ramp()
    vol_path = _write(tmp_path, 'vol.nii.gz', _in_orientation(canonical, orientation), orientation)
    volume, metadata = read_nifti_volume(vol_path)
    geometry = metadata['geometry']

    as_stored = _in_orientation((canonical % 4).astype(np.uint8), orientation)
    legacy = sitk.GetImageFromArray(as_stored)  # the old writer set spacing and nothing else
    legacy.SetSpacing(SPACING)
    legacy_path = str(tmp_path / 'legacy_seg.nii.gz')
    sitk.WriteImage(legacy, legacy_path)

    mask, legacy_meta = read_mask_volume(legacy_path, geometry)
    assert np.array_equal(mask, (volume % 4).astype(np.uint8))
    assert legacy_meta['geometry'] == geometry  # re-saving restores the volume's grid

    without_volume, _ = read_mask_volume(legacy_path)
    assert np.array_equal(without_volume, as_stored)


def test_geometry_from_spacing_is_canonical():
    """The fallback grid used when no source geometry is at hand must not itself
    reorient a mask on the way out."""
    geometry = geometry_from_spacing((0.8, 0.5, 0.4))  # (dz, dy, dx)

    assert geometry.spacing == (0.4, 0.5, 0.8)  # sitk (x, y, z)
    assert geometry.source_orientation == CANONICAL_ORIENTATION
    assert geometry == VolumeGeometry(spacing=(0.4, 0.5, 0.8))


def _write_ct_series(folder, array, iop, spacing=SPACING, origin=ORIGIN):
    """Write `array` (Z, Y, X) as a CT DICOM series whose slices carry `iop`.

    Slices are written in shuffled filename order, since read_ct_volume is supposed to
    sort them by ImagePositionPatient z rather than trust the directory listing.
    """
    import pydicom as dcm
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

    folder.mkdir(exist_ok=True)
    dx, dy, dz = spacing
    series_uid = generate_uid()
    order = list(range(array.shape[0]))
    for out_index, z in enumerate(reversed(order)):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(f'{out_index}.dcm', {}, file_meta=meta, preamble=b'\0' * 128)
        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.SeriesInstanceUID = series_uid
        ds.Modality = 'CT'
        ds.ImageOrientationPatient = list(iop)
        ds.ImagePositionPatient = [origin[0], origin[1], origin[2] + z * dz]
        ds.PixelSpacing = [dy, dx]  # DICOM order: (row, column) spacing
        ds.SliceThickness = dz
        ds.RescaleSlope = 1
        ds.RescaleIntercept = 0
        ds.Rows, ds.Columns = array.shape[1], array.shape[2]
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = 'MONOCHROME2'
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.PixelData = array[z].astype(np.int16).tobytes()
        dcm.dcmwrite(str(folder / f'slice_{out_index}.dcm'), ds, write_like_original=False)
    return str(folder)


def test_axial_dicom_series_is_canonicalized(tmp_path):
    """A plain axial CT series is stored LPS, which the displays would show flipped
    front-to-back, so the DICOM reader canonicalizes it like the NIfTI one does."""
    canonical = _ramp()
    folder = _write_ct_series(tmp_path / 'series', _in_orientation(canonical, 'LPS'), (1, 0, 0, 0, 1, 0))

    volume, metadata = read_ct_volume(folder)

    assert np.array_equal(volume, canonical)
    assert metadata['geometry'].source_orientation == 'LPS'
    assert metadata['n_slices'] == canonical.shape[0]
    dy, dx = metadata['pixel_spacing']
    assert (dx, dy, metadata['slice_thickness']) == pytest.approx(SPACING)


def test_coronally_acquired_dicom_series_is_canonicalized(tmp_path):
    """ImageOrientationPatient, not the axial assumption, decides which array axis is
    which — here rows run head-to-foot, so the stack comes in as 'LIP'."""
    canonical = _ramp()
    iop = (1, 0, 0, 0, 0, -1)  # columns -> Left, rows -> Inferior, slices -> Posterior
    folder = _write_ct_series(tmp_path / 'series', _in_orientation(canonical, 'LIP'), iop)

    volume, metadata = read_ct_volume(folder)

    assert np.array_equal(volume, canonical)
    assert metadata['geometry'].source_orientation == 'LIP'
