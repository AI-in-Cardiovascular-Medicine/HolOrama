"""The ONNX re-implementation of nnU-Net's 2D inference (src/segmentation/onnx_inference.py).

Numerical parity with nnUNetPredictor itself is checked once per exported model by
scripts/model_export/verify_onnx.py (it needs torch + nnUZoo, which the app no longer has);
these tests cover the pieces that can drift without torch: nnU-Net's helper maths, the
frame-range / cancel / error behaviour, and loading the bundled model."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('onnxruntime')

from segmentation import onnx_inference as oi  # noqa: E402
from segmentation.predict import Predict, resolve_model_path  # noqa: E402


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


def test_nonzero_bbox_fills_holes_and_handles_empty():
    vol = np.zeros((4, 10, 12))
    vol[1:3, 2:8, 3:9] = 1
    vol[1:3, 4:6, 5:7] = 0  # an interior hole must not shrink the box
    assert oi.nonzero_bbox(vol) == (slice(1, 3), slice(2, 8), slice(3, 9))
    assert oi.nonzero_bbox(np.zeros((2, 3, 4))) == (slice(0, 2), slice(0, 3), slice(0, 4))


def _ring_volume(frames=3, size=320, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size] - size / 2
    r = np.hypot(yy, xx)
    ring = ((r > 60) & (r < 80)) * 180.0
    return np.stack([ring + rng.normal(30, 10, (size, size)) for _ in range(frames)]).clip(0, 255).astype(np.uint8)


@pytest.fixture(scope='module')
def segmenter():
    s = oi.OnnxSegmenter(use_mirroring=False, batch_size=1)  # one frame per network call
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
    calls = []

    def progress(done, total):
        calls.append((done, total))
        return False

    assert segmenter.predict(_ring_volume(), progress_cb=progress) is None
    assert calls == [(1, 3)]


def test_missing_model_is_reported_as_unavailable(tmp_path):
    s = oi.OnnxSegmenter(tmp_path / 'nope.onnx')
    with pytest.raises(oi.SegmentationUnavailable, match='not found'):
        s.load()


def test_missing_onnxruntime_is_reported_as_unavailable(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == 'onnxruntime':
            raise ImportError('no onnxruntime')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', fake_import)
    ok, reason = Predict(None, config=SimpleNamespace(segmentation=SimpleNamespace())).check_available()
    assert not ok and 'onnxruntime' in reason


def test_invalid_device_is_reported_as_unavailable():
    with pytest.raises(oi.SegmentationUnavailable, match='segmentation.device'):
        oi.OnnxSegmenter(device='tpu').load()


def test_cpu_device_never_tries_the_gpu(monkeypatch):
    monkeypatch.setattr(oi.OnnxSegmenter, '_try_cuda_session', lambda self, ort: pytest.fail('GPU was tried'))
    s = oi.OnnxSegmenter(device='cpu')
    s.load()
    assert s.active_device == 'cpu' and s.fallback_reason == ''


def test_auto_falls_back_to_cpu_and_cuda_refuses(monkeypatch):
    monkeypatch.setattr(oi.OnnxSegmenter, '_try_cuda_session', lambda self, ort: (None, 'CUDA test inference failed'))
    auto = oi.OnnxSegmenter(device='auto')
    auto.load()
    assert auto.active_device == 'cpu' and 'CUDA test inference failed' in auto.fallback_reason
    with pytest.raises(oi.SegmentationUnavailable, match='"cuda"'):
        oi.OnnxSegmenter(device='cuda').load()


def test_cpu_only_package_is_not_reported_as_a_fallback(monkeypatch):
    # The plain CPU package is the normal case (compiled installer), not a degraded GPU setup.
    monkeypatch.setattr(oi.OnnxSegmenter, '_try_cuda_session', lambda self, ort: (None, 'needs "onnxruntime-gpu"'))
    s = oi.OnnxSegmenter(device='auto')
    s.load()
    assert s.active_device == 'cpu' and s.fallback_reason == ''


def test_old_configs_fall_back_to_bundled_model():
    assert resolve_model_path('') == oi.DEFAULT_MODEL_PATH
    assert resolve_model_path(None) == oi.DEFAULT_MODEL_PATH
    # pre-ONNX configs pointed at an nnU-Net folder or a Keras .h5
    assert resolve_model_path('./models/nnUNetTrainerU2NetP__nnUNetResEncUNetLPlans__2d') == oi.DEFAULT_MODEL_PATH
    assert resolve_model_path('./models/old/u2net.h5') == oi.DEFAULT_MODEL_PATH
    assert str(resolve_model_path('C:/models/custom.onnx')).endswith('custom.onnx')
