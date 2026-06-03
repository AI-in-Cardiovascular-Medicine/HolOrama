import pytest
import pandas as pd
import numpy as np
from types import SimpleNamespace

from input_output.input.metadata import (
    parse_metadata_dcm,
    parse_metadata_nifti,
    extract_modality,
    extract_patient_info,
    extract_pullback_rate,
    extract_resolution,
    extract_frame_time_ms,
    extract_frame_time_vector,
    extract_pullback_length_ivus,
    extract_frame_rate,
    extract_dimension,
    extract_pullback_start_frame,
    extract_manufacturer,
    extract_nifti_spacing,
    extract_nifti_frame_rate,
    extract_nifti_dimension,
)


def _df(*rows):
    return pd.DataFrame(list(rows), columns=['Description', 'Value'])


class TestExtractHelpers:
    def test_modality(self):
        assert extract_modality(_df(('Modality', 'IVUS'))) == 'IVUS'

    def test_modality_missing_returns_none(self):
        assert extract_modality(_df(('Other', 'x'))) is None

    def test_patient_info_apostrophe_keys(self):
        d = _df(("Patient's Name", 'Doe^John'), ("Patient's Birth Date", '19800101'), ("Patient's Sex", 'M'))
        assert extract_patient_info(d) == ('Doe^John', '1980/01/01', 'M')

    def test_patient_info_plain_keys(self):
        d = _df(('Patient Name', 'Smith'), ('Patient Birth Date', '19900215'), ('Patient Sex', 'F'))
        assert extract_patient_info(d) == ('Smith', '1990/02/15', 'F')

    def test_patient_info_defaults_to_unknown(self):
        assert extract_patient_info(_df(('Other', 'x'))) == ('Unknown', 'Unknown', 'Unknown')

    def test_pullback_rate_ivus_field(self):
        assert extract_pullback_rate(_df(('IVUS Pullback Rate', '0.5'))) == pytest.approx(0.5)

    def test_pullback_rate_boston_field(self):
        assert extract_pullback_rate(_df(('BostonPullbackRate', '1.0'))) == pytest.approx(1.0)

    def test_pullback_rate_missing_returns_none(self):
        assert extract_pullback_rate(_df(('Other', 'x'))) is None

    def test_resolution_pixel_spacing_scalar(self):
        assert extract_resolution(_df(('Pixel Spacing', 0.15))) == pytest.approx(0.15)

    def test_resolution_pixel_spacing_array(self):
        assert extract_resolution(_df(('Pixel Spacing', [0.12, 0.12]))) == pytest.approx(0.12)

    def test_resolution_ultrasound_region_mm(self):
        region = SimpleNamespace(PhysicalUnitsXDirection=0, PhysicalDeltaX=0.05)
        assert extract_resolution(_df(('Sequence of Ultrasound Regions', [region]))) == pytest.approx(0.05)

    def test_resolution_ultrasound_region_cm_multiplied_by_10(self):
        # PhysicalUnitsXDirection==3 means cm → multiply by 10 to get mm
        region = SimpleNamespace(PhysicalUnitsXDirection=3, PhysicalDeltaX=0.005)
        assert extract_resolution(_df(('Sequence of Ultrasound Regions', [region]))) == pytest.approx(0.05)

    def test_resolution_missing_returns_none(self):
        assert extract_resolution(_df(('Other', 'x'))) is None

    def test_frame_time_ms(self):
        assert extract_frame_time_ms(_df(('Frame Time', '33.3'))) == pytest.approx(33.3)

    def test_frame_time_ms_missing_returns_none(self):
        assert extract_frame_time_ms(_df(('Other', 'x'))) is None

    def test_frame_time_vector(self):
        assert extract_frame_time_vector(_df(('Frame Time Vector', [100.0, 100.0]))) == [100.0, 100.0]

    def test_frame_time_vector_missing_returns_none(self):
        assert extract_frame_time_vector(_df(('Other', 'x'))) is None

    def test_pullback_length_ivus_with_ftv(self):
        d = _df(('Frame Time Vector', [100.0, 100.0, 100.0]))
        result = extract_pullback_length_ivus(d, pullback_rate=0.5, num_frames=3)
        # cumsum([100, 100, 100]) / 1000 * 0.5 = [0.1, 0.2, 0.3] * 0.5
        np.testing.assert_allclose(result, [0.05, 0.10, 0.15])

    def test_pullback_length_ivus_no_ftv_returns_zeros(self):
        result = extract_pullback_length_ivus(_df(('Other', 'x')), pullback_rate=0.5, num_frames=5)
        np.testing.assert_array_equal(result, np.zeros(5))

    def test_frame_rate(self):
        assert extract_frame_rate(_df(('Cine Rate', '30'))) == 30.0

    def test_frame_rate_missing_returns_none(self):
        assert extract_frame_rate(_df(('Other', 'x'))) is None

    def test_dimension(self):
        assert extract_dimension(_df(('Rows', '512'))) == 512

    def test_dimension_missing_returns_none(self):
        assert extract_dimension(_df(('Other', 'x'))) is None

    def test_pullback_start_frame(self):
        assert extract_pullback_start_frame(_df(('IVUS Pullback Start Frame Number', '5'))) == 5

    def test_pullback_start_frame_defaults_to_zero(self):
        assert extract_pullback_start_frame(_df(('Other', 'x'))) == 0

    def test_manufacturer(self):
        d = _df(('Manufacturer', 'Philips'), ("Manufacturer's Model Name", 'Eagle Eye'))
        assert extract_manufacturer(d) == ('Philips', 'Eagle Eye')

    def test_manufacturer_alt_model_key(self):
        d = _df(('Manufacturer', 'Boston'), ('Manufacturer Model Name', 'Atlantis'))
        assert extract_manufacturer(d) == ('Boston', 'Atlantis')

    def test_manufacturer_missing_returns_unknown(self):
        assert extract_manufacturer(_df(('Other', 'x'))) == ('Unknown', 'Unknown')

    def test_nifti_spacing(self):
        xy, z = extract_nifti_spacing(_df(('pixdim', [1.0, 0.12, 0.12, 0.2, 0.0])))
        assert xy == pytest.approx(0.12)
        assert z == pytest.approx(0.2)

    def test_nifti_spacing_missing_returns_none_pair(self):
        assert extract_nifti_spacing(_df(('Other', 'x'))) == (None, None)

    def test_nifti_frame_rate(self):
        dt = 1 / 180
        assert extract_nifti_frame_rate(_df(('pixdim', [1.0, 0.12, 0.12, 0.2, dt]))) == pytest.approx(180.0, rel=1e-3)

    def test_nifti_frame_rate_missing_returns_none(self):
        assert extract_nifti_frame_rate(_df(('Other', 'x'))) is None

    def test_nifti_dimension(self):
        assert extract_nifti_dimension(_df(('dim', [3, 512, 512, 100]))) == 512

    def test_nifti_dimension_missing_returns_none(self):
        assert extract_nifti_dimension(_df(('Other', 'x'))) is None


class TestParseMetadataDcm:
    def _ivus_df(self):
        return pd.DataFrame(
            [
                ('Modality', 'IVUS'),
                ("Patient's Name", 'Doe'),
                ("Patient's Birth Date", '19800101'),
                ("Patient's Sex", 'M'),
                ('IVUS Pullback Rate', '0.5'),
                ('Pixel Spacing', 0.1),
                ('Rows', '512'),
                ('Cine Rate', '30'),
                ('IVUS Pullback Start Frame Number', '0'),
            ],
            columns=['Description', 'Value'],
        )

    def test_ivus_all_fields(self):
        meta = parse_metadata_dcm(self._ivus_df(), num_frames=10)
        assert meta.modality == 'IVUS'
        assert meta.pullback_speed == pytest.approx(0.5)
        assert meta.resolution == pytest.approx(0.1)
        assert meta.frame_rate == 30.0
        assert meta.dimension == 512
        assert meta.pullback_start_frame == 0

    def test_oct_frame_rate_from_frame_time(self):
        d = pd.DataFrame(
            [
                ('Modality', 'OCT'),
                ('Frame Time', str(1000 / 180)),
                ('IVUS Pullback Rate', '36.0'),
                ('Pixel Spacing', 0.01),
            ],
            columns=['Description', 'Value'],
        )
        meta = parse_metadata_dcm(d, num_frames=180)
        assert meta.modality == 'OCT'
        assert meta.frame_rate == pytest.approx(180.0, rel=1e-2)
        assert meta.pullback_length == pytest.approx(36.0, rel=1e-2)

    def test_oct_abbott_optis_overrides_frame_rate_to_180(self):
        d = pd.DataFrame(
            [
                ('Modality', 'OCT'),
                ('Manufacturer', 'Abbott'),
                ("Manufacturer's Model Name", 'OPTIS Mobile'),
                ('Frame Time', '100.0'),
                ('IVUS Pullback Rate', '36.0'),
                ('Pixel Spacing', 0.01),
            ],
            columns=['Description', 'Value'],
        )
        meta = parse_metadata_dcm(d, num_frames=180)
        assert meta.frame_rate == 180.0

    def test_prompt_fn_called_for_missing_pullback_rate(self):
        d = pd.DataFrame([('Modality', 'IVUS'), ('Pixel Spacing', 0.1)], columns=['Description', 'Value'])
        calls = []

        def prompt_fn(title, msg, default):
            calls.append(default)
            return 2.5

        meta = parse_metadata_dcm(d, num_frames=5, prompt_fn=prompt_fn)
        assert meta.pullback_speed == 2.5
        assert len(calls) == 1

    def test_no_prompt_fn_leaves_missing_fields_as_none(self):
        d = pd.DataFrame([('Modality', 'IVUS')], columns=['Description', 'Value'])
        meta = parse_metadata_dcm(d, num_frames=5, prompt_fn=None)
        assert meta.pullback_speed is None
        assert meta.resolution is None


class TestParseMetadataNifti:
    def test_ivus_with_z_spacing(self):
        d = pd.DataFrame(
            [
                ('pixdim', [1.0, 0.1, 0.1, 0.05, 0.0]),
                ('dim', [3, 512, 512, 50]),
            ],
            columns=['Description', 'Value'],
        )
        meta = parse_metadata_nifti(d, num_frames=50, is_oct=False, prompt_fn=lambda t, m, d: 0.5)
        assert meta.modality == 'IVUS'
        assert meta.resolution == pytest.approx(0.1)
        assert meta.pullback_speed == 0.5
        assert len(meta.pullback_length) == 50

    def test_oct_with_z_spacing_computes_frame_rate_and_length(self):
        z_per_frame = 0.2  # 36 mm/s / 180 fps
        d = pd.DataFrame([('pixdim', [1.0, 0.01, 0.01, z_per_frame, 0.0])], columns=['Description', 'Value'])
        meta = parse_metadata_nifti(d, num_frames=180, is_oct=True, prompt_fn=lambda t, m, d: 36.0)
        assert meta.modality == 'OCT'
        assert meta.frame_rate == pytest.approx(180.0, rel=1e-3)
        assert meta.pullback_length == pytest.approx(z_per_frame * 180)
