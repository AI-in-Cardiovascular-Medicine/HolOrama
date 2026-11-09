"""The ONNX re-implementation of nnU-Net's 2D inference (src/segmentation/onnx_inference.py).

Numerical parity with nnUNetPredictor itself is checked once per exported model by
scripts/model_export/verify_onnx.py (it needs torch + nnUZoo, which the app no longer has).
These tests cover the pieces that can drift without torch: nnU-Net's helper maths, the
frame-range / cancel / error behaviour, the GPU and precision choice, and loading the bundled
model."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('onnxruntime')

from domain.intravascular.segmentation_types import SegmentationSettings  # noqa: E402
from segmentation import onnx_inference as oi  # noqa: E402
from segmentation.predict import Predict  # noqa: E402

CPU_ONE_FRAME_PER_CALL = SegmentationSettings(use_mirroring=False, batch_size=1, device='cpu')


def test_compute_steps_matches_nnunet_examples():
    # Example from nnU-Net's own docstring: image 110, patch 64, step 0.5 -> 0, 23, 46
    assert oi.compute_steps((110,), (64,), 0.5) == [[0, 23, 46]]
    # The bundled model on a 512x512 frame: one row, two overlapping columns
    assert oi.compute_steps((512, 512), (512, 448), 0.5) == [[0], [0, 64]]
    assert oi.compute_steps((512, 448), (512, 448), 0.5) == [[0], [0]]


def test_gaussian_is_peaked_positive_and_scaled():
    g = oi.compute_gaussian((64, 48))
    assert g.shape == (64, 48) and g.dtype == np.float32
    assert g.max() == pytest.approx(10.0)
    assert g.min() > 0
    assert np.unravel_index(g.argmax(), g.shape) == (32, 24)


def test_nonzero_bbox_ignores_holes_and_handles_empty():
    vol = np.zeros((4, 10, 12))
    vol[1:3, 2:8, 3:9] = 1
    vol[1:3, 4:6, 5:7] = 0  # an interior hole must not shrink the box
    assert oi.nonzero_bbox(vol) == (slice(1, 3), slice(2, 8), slice(3, 9))
    assert oi.nonzero_bbox(np.zeros((2, 3, 4))) == (slice(0, 2), slice(0, 3), slice(0, 4))


def test_nonzero_bbox_equals_box_of_hole_filled_mask():
    # nnU-Net hole-fills before taking the box, skipping the fill must give the same box.
    from scipy.ndimage import binary_fill_holes

    rng = np.random.default_rng(1)
    for _ in range(20):
        vol = (rng.random((6, 20, 24)) > 0.9).astype(np.uint8)
        filled = binary_fill_holes(vol != 0)
        assert oi.nonzero_bbox(vol) == oi.nonzero_bbox(filled.astype(np.uint8))


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16, np.int16, np.float32])
def test_zscore_stats_match_numpy(dtype):
    rng = np.random.default_rng(2)
    vol = (rng.random((70, 30, 40)) * 250).astype(dtype)  # 70 frames: spans several chunks
    mean, std = oi.zscore_stats(vol)
    ref = vol.astype(np.float64)
    assert mean == pytest.approx(ref.mean(), rel=1e-6)
    assert std == pytest.approx(ref.std(), rel=1e-6)


@pytest.fixture(scope='module')
def segmenter():
    # CPU: deterministic everywhere (a GPU-less CI runner may still offer DirectML's slow software adapter)
    s = oi.OnnxSegmenter(CPU_ONE_FRAME_PER_CALL)
    s.load()
    return s


def test_bundled_model_metadata(segmenter):
    info = segmenter.info
    assert info.patch_size == (512, 448)
    assert info.mirror_axes == (0, 1)
    assert info.num_classes == 2


def test_predict_shape_dtype_and_frame_range(segmenter):
    vol = _ring_volume()
    full = segmenter.predict(vol)
    assert full is not None and full.shape == vol.shape and full.dtype == np.uint8
    assert set(np.unique(full)) <= {0, 1}

    # Only the requested frames are filled in, and they match a full run exactly (the crop and
    # normalisation statistics always come from the whole volume).
    part = segmenter.predict(vol, frames=range(1, 2))
    assert part is not None
    assert np.array_equal(part[1], full[1])
    assert not part[0].any() and not part[2].any()


def test_all_zero_frames_are_background(segmenter):
    vol = _ring_volume()
    vol[0] = 0  # cropped away by the non-zero bounding box, like nnU-Net does
    out = segmenter.predict(vol)
    assert out is not None and not out[0].any()


def test_cancel_returns_none(segmenter):
    progress = _CancellingProgress()
    assert segmenter.predict(_ring_volume(), progress_cb=progress) is None
    assert progress.calls == [(1, 3)]


def test_missing_model_is_reported_as_unavailable(tmp_path):
    s = oi.OnnxSegmenter(SegmentationSettings(model_file=str(tmp_path / 'nope.onnx')))
    with pytest.raises(oi.SegmentationUnavailable, match='not found'):
        s.load()


def test_missing_onnxruntime_is_reported_as_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, 'onnxruntime', None)  # makes `import onnxruntime` raise ImportError
    ok, reason = Predict(None, config=SimpleNamespace(segmentation=SimpleNamespace())).check_available()
    assert not ok and 'onnxruntime' in reason


def test_invalid_device_is_reported_as_unavailable():
    with pytest.raises(oi.SegmentationUnavailable, match='segmentation.device'):
        oi.OnnxSegmenter(SegmentationSettings(device='tpu')).load()


def test_cpu_device_never_tries_the_gpu(monkeypatch):
    gpus = _FakeGpus({})
    monkeypatch.setattr(oi.OnnxSegmenter, '_try_gpu_session', gpus.try_gpu_session)
    s = oi.OnnxSegmenter(SegmentationSettings(device='cpu'))
    s.load()
    assert gpus.tried == []
    assert s.active_device == 'cpu' and s.fallback_reason == ''


def test_auto_tries_cuda_then_directml_then_cpu(monkeypatch):
    gpus = _FakeGpus({'cuda': oi._MISSING_RUNTIME['cuda'], 'directml': 'DirectML test inference failed'})
    monkeypatch.setattr(oi.OnnxSegmenter, '_try_gpu_session', gpus.try_gpu_session)
    auto = oi.OnnxSegmenter(SegmentationSettings(device='auto'))
    auto.load()
    assert gpus.tried == ['cuda', 'directml']
    assert auto.active_device == 'cpu'
    # the missing CUDA package is normal, the installed-but-broken DirectML is reported
    assert auto.fallback_reason == 'DirectML test inference failed'


@pytest.mark.parametrize('device', ['cuda', 'directml'])
def test_explicit_gpu_device_refuses_without_it(monkeypatch, device):
    gpus = _FakeGpus({'cuda': 'no CUDA', 'directml': 'no DirectML'})
    monkeypatch.setattr(oi.OnnxSegmenter, '_try_gpu_session', gpus.try_gpu_session)
    with pytest.raises(oi.SegmentationUnavailable, match=f'"{device}"'):
        oi.OnnxSegmenter(SegmentationSettings(device=device)).load()


def test_cpu_only_package_is_not_reported_as_a_fallback(monkeypatch):
    # A CPU-only onnxruntime (Linux/macOS default) is a normal setup, not a degraded GPU one.
    gpus = _FakeGpus(dict(oi._MISSING_RUNTIME))
    monkeypatch.setattr(oi.OnnxSegmenter, '_try_gpu_session', gpus.try_gpu_session)
    s = oi.OnnxSegmenter(SegmentationSettings(device='auto'))
    s.load()
    assert s.active_device == 'cpu' and s.fallback_reason == ''


def test_invalid_precision_is_reported_as_unavailable():
    with pytest.raises(oi.SegmentationUnavailable, match='segmentation.precision'):
        oi.OnnxSegmenter(SegmentationSettings(precision='int8')).load()


def test_fp16_model_path_sits_next_to_the_model():
    assert oi.fp16_model_path('C:/m/ivus.onnx').as_posix() == 'C:/m/ivus_fp16.onnx'
    assert oi.fp16_model_path(oi.DEFAULT_MODEL_PATH).name == 'ivus_lumen_u2netp_fp16.onnx'


def test_bundled_fp16_model_keeps_float32_io_and_metadata():
    import onnxruntime as ort

    fp32 = ort.InferenceSession(str(oi.DEFAULT_MODEL_PATH), providers=['CPUExecutionProvider'])
    fp16 = ort.InferenceSession(str(oi.fp16_model_path(oi.DEFAULT_MODEL_PATH)), providers=['CPUExecutionProvider'])
    assert [i.type for i in fp16.get_inputs()] == ['tensor(float)']
    assert [o.type for o in fp16.get_outputs()] == ['tensor(float)']
    meta32, meta16 = fp32.get_modelmeta().custom_metadata_map, fp16.get_modelmeta().custom_metadata_map
    assert {k: v for k, v in meta16.items() if k != 'precision'} == meta32


def test_fp16_masks_agree_with_fp32(segmenter):
    # The fp16 copy runs on the CPU too (slowly), which checks it belongs to the fp32 model.
    fp16_file = str(oi.fp16_model_path(oi.DEFAULT_MODEL_PATH))
    fp16 = oi.OnnxSegmenter(SegmentationSettings(model_file=fp16_file, use_mirroring=False, batch_size=1, device='cpu'))
    vol = _ring_volume()
    ref, out = segmenter.predict(vol), fp16.predict(vol)
    assert ref is not None and out is not None
    assert (ref != out).sum() <= 0.001 * ref.size  # a few border pixels at most


@pytest.fixture
def fake_gpu(monkeypatch, tmp_path):
    gpu = _FakeGpuSessions(tmp_path)
    monkeypatch.setattr(oi.OnnxSegmenter, '_gpu_session', gpu.gpu_session)
    monkeypatch.setattr(oi.OnnxSegmenter, '_test_input', gpu.test_input)
    monkeypatch.setattr(oi, '_median_run_time', gpu.median_run_time)
    return gpu


def test_auto_precision_takes_fp16_only_when_clearly_faster(fake_gpu):
    s, session = fake_gpu.run('auto')
    assert s.active_precision == 'fp16' and session.path.name == 'm_fp16.onnx'
    fake_gpu.times['fp16'] = 0.95  # 5% faster: not worth the changed border pixels
    s, session = fake_gpu.run('auto')
    assert s.active_precision == 'fp32' and session.path.name == 'm.onnx'


def test_fp32_precision_never_loads_the_fp16_model(fake_gpu):
    s, session = fake_gpu.run('fp32')
    assert s.active_precision == 'fp32' and fake_gpu.created == ['m.onnx']


def test_fp16_precision_skips_the_timing(fake_gpu):
    fake_gpu.times['fp16'] = 5.0
    s, session = fake_gpu.run('fp16')
    assert s.active_precision == 'fp16' and session.path.name == 'm_fp16.onnx'


@pytest.mark.parametrize('precision', ['auto', 'fp16'])
def test_missing_or_broken_fp16_model_keeps_fp32(fake_gpu, precision):
    fake_gpu.fp16_broken = True
    s, session = fake_gpu.run(precision)
    assert s.active_precision == 'fp32' and session.path.name == 'm.onnx'
    fake_gpu.fp16_broken = False
    oi.fp16_model_path(fake_gpu.model).unlink()
    s, session = fake_gpu.run(precision)
    assert s.active_precision == 'fp32' and session.path.name == 'm.onnx'


def test_cpu_runs_fp32_whatever_the_precision():
    s = oi.OnnxSegmenter(SegmentationSettings(device='cpu', precision='fp16'))
    s.load()
    assert s.active_device == 'cpu' and s.active_precision == 'fp32'
    assert 'precision' not in s._session.get_modelmeta().custom_metadata_map  # set only in the fp16 copy


def test_settings_from_config_fill_in_missing_and_empty_keys():
    defaults = SegmentationSettings()
    assert SegmentationSettings.from_config(SimpleNamespace()) == defaults
    assert SegmentationSettings.from_config(SimpleNamespace(segmentation=SimpleNamespace())) == defaults
    section = SimpleNamespace(model_file=None, device='cuda', batch_size=4, input_dir='/old/key')  # stale key ignored
    settings = SegmentationSettings.from_config(SimpleNamespace(segmentation=section))
    assert settings == SegmentationSettings(device='cuda', batch_size=4)


def test_bundled_config_matches_the_settings_defaults():
    import yaml

    config = yaml.safe_load((Path(__file__).resolve().parents[1] / 'src' / 'config.yaml').read_text(encoding='utf-8'))
    section = SimpleNamespace(**config['segmentation'])
    assert SegmentationSettings.from_config(SimpleNamespace(segmentation=section)) == SegmentationSettings()


def test_old_configs_fall_back_to_bundled_model():
    assert oi.resolve_model_path('') == oi.DEFAULT_MODEL_PATH
    assert oi.resolve_model_path(None) == oi.DEFAULT_MODEL_PATH
    # pre-ONNX configs pointed at an nnU-Net folder or a Keras .h5
    assert oi.resolve_model_path('./models/nnUNetTrainerU2NetP__nnUNetResEncUNetLPlans__2d') == oi.DEFAULT_MODEL_PATH
    assert oi.resolve_model_path('./models/old/u2net.h5') == oi.DEFAULT_MODEL_PATH
    assert str(oi.resolve_model_path('C:/models/custom.onnx')).endswith('custom.onnx')


def _ring_volume(frames=3, size=320, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size] - size / 2
    r = np.hypot(yy, xx)
    ring = ((r > 60) & (r < 80)) * 180.0
    return np.stack([ring + rng.normal(30, 10, (size, size)) for _ in range(frames)]).clip(0, 255).astype(np.uint8)


class _CancellingProgress:
    """predict() progress callback that records its calls and cancels at the first one."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def __call__(self, done: int, total: int) -> bool:
        self.calls.append((done, total))
        return False


class _FakeGpus:
    """Stands in for OnnxSegmenter._try_gpu_session: every GPU fails with the given reason."""

    def __init__(self, reasons: dict[str, str]) -> None:
        self.reasons = reasons
        self.tried: list[str] = []

    def try_gpu_session(self, ort, device):
        self.tried.append(device)
        return None, self.reasons[device]


class _FakeSession:
    def __init__(self, path) -> None:
        self.path = Path(path)


class _FakeGpuSessions:
    """A DirectML 'GPU' whose sessions are _FakeSession(model path), with a set run time per
    precision. Stands in for OnnxSegmenter._gpu_session, ._test_input and _median_run_time."""

    def __init__(self, tmp_path: Path) -> None:
        self.model = tmp_path / 'm.onnx'
        self.model.touch()
        oi.fp16_model_path(self.model).touch()
        self.times = {'fp32': 1.0, 'fp16': 0.5}
        self.created: list[str] = []
        self.fp16_broken = False
        self.ort = SimpleNamespace(get_available_providers=lambda: ['DmlExecutionProvider', 'CPUExecutionProvider'])

    def run(self, precision: str):
        """(segmenter, session) after choosing the DirectML session at this precision."""
        s = oi.OnnxSegmenter(SegmentationSettings(model_file=str(self.model), precision=precision))
        session, reason = s._try_gpu_session(self.ort, 'directml')
        assert reason == ''
        return s, session

    def gpu_session(self, ort, device, path, fp16):
        self.created.append(Path(path).name)
        if fp16 and self.fp16_broken:
            return None, 'fp16 broken'
        return _FakeSession(path), ''

    def test_input(self, session):
        return None

    def median_run_time(self, session, x):
        return self.times['fp16' if session.path.stem.endswith('_fp16') else 'fp32']
