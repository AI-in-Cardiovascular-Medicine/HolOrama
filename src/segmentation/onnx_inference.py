"""nnU-Net 2D inference re-implemented on numpy + onnxruntime.

The automatic IVUS segmentation model was trained with nnU-Net (nnUZoo's U2NetP trainer).
Running it through nnU-Net itself needs torch, nnunetv2 and a patched nnUZoo, which made the
install fragile and the app impossible to compile. Instead, the network is exported once to
ONNX (scripts/model_export/export_onnx.py) and this module reproduces exactly the steps
nnUNetPredictor.predict_single_npy_array performs for a 2D configuration:

1. crop the (frames, y, x) volume to the bounding box of its non-zero region,
   as nnU-Net's crop_to_nonzero does
2. z-score normalise with the mean/std of that whole cropped volume
   (resampling is a no-op: the model's target spacing is 1 and frames are fed with spacing 1)
3. per frame: centre-pad to at least the patch size, tile with a sliding window
   (step = tile_step_size * patch), run every tile plus its mirrored variants, weight each
   tile's logits by nnU-Net's Gaussian importance map and average
4. argmax over classes, undo padding and cropping

Verified against nnUNetPredictor on the model's own test frames by
scripts/model_export/verify_onnx.py. The only intentional difference: nnU-Net accumulates
logits in float16, this module in float32, which can flip a handful of borderline pixels.

On a GPU the network can run in float16 from a converted copy of the model next to it
(<name>_fp16.onnx, scripts/model_export/convert_fp16.py), with float32 input and output, so
everything around the network stays float32. On a 1163 frame pullback this moved 2 to 4 border
pixels per frame (min Dice 0.9998) for 1.3x (DirectML) to 1.5x (CUDA) faster segmentation on a
Quadro RTX 4000. The CPU always runs float32.

Only the requested frames are run through the network, but normalisation statistics and the
crop are always computed on the whole volume, so a frame's mask does not depend on which
range was requested — the same result nnU-Net gave when it was fed the whole pullback.
"""

from __future__ import annotations

import itertools
import json
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger
from scipy.ndimage import gaussian_filter

# Bundled model, shipped next to this module (and next to the compiled module in the Nuitka build).
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / 'weights' / 'ivus_lumen_u2netp.onnx'
# 'auto': the first GPU runtime that is installed and passes a test inference, else the CPU.
DEVICES = ('auto', 'cpu', 'cuda', 'directml')
# 'auto': float16 on a GPU when it is clearly faster there than float32, else float32.
PRECISIONS = ('auto', 'fp32', 'fp16')
# 'auto' keeps float32 unless float16 takes at most this share of its time per call:
# float16 shifts a few border pixels, so it has to buy a real speedup.
_FP16_MAX_TIME_RATIO = 0.9
_GPU_PROVIDERS = {'cuda': 'CUDAExecutionProvider', 'directml': 'DmlExecutionProvider'}  # in 'auto' order
_MISSING_RUNTIME = {
    'cuda': 'CUDA inference needs the "onnxruntime-gpu" package (source installs: install script with '
    '-Gpu / --gpu, or uv sync --extra segmentation-gpu).',
    'directml': 'DirectML inference needs the "onnxruntime-directml" package (Windows only; installed '
    'by the "segmentation" extra there).',
}
_INIT_FAILED = {
    'cuda': "CUDA could not be initialised (no NVIDIA GPU, or the driver is too old: CUDA 12 needs driver "
    ">= 527.41 on Windows / 525.60 on Linux). See the log for onnxruntime's message.",
    'directml': "DirectML could not be initialised (no DirectX 12 capable GPU). See the log for onnxruntime's message.",
}


class SegmentationUnavailable(RuntimeError):
    """Raised when automatic segmentation cannot run on this installation (runtime missing,
    model missing or unreadable). The message is meant to be shown to the user as is."""


@dataclass(frozen=True)
class ModelInfo:
    patch_size: tuple[int, int]
    mirror_axes: tuple[int, ...]
    num_classes: int
    labels: dict


def _import_onnxruntime():
    try:
        import onnxruntime  # noqa: PLC0415 - optional dependency, imported lazily
    except ImportError as e:
        raise SegmentationUnavailable(
            'Automatic segmentation needs the optional "onnxruntime" package, which is not installed.\n\n'
            'From source: run the install script again (it installs the "segmentation" extra), or\n'
            '    uv sync --extra segmentation'
        ) from e
    return onnxruntime


def fp16_model_path(model_path: str | os.PathLike) -> Path:
    """The float16 copy that belongs to a model: <name>_fp16.onnx in the same folder."""
    path = Path(model_path)
    return path.with_name(f'{path.stem}_fp16{path.suffix}')


def _median_run_time(session, x: np.ndarray, runs: int = 3) -> float:
    """Median seconds per session.run on x (the session is already warmed up)."""
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run(None, {'image': x})
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def compute_gaussian(tile_size: tuple[int, ...], sigma_scale: float = 1.0 / 8, value_scaling_factor: float = 10):
    """nnU-Net's Gaussian importance map (sliding_window_prediction.compute_gaussian)."""
    tmp = np.zeros(tile_size)
    tmp[tuple(i // 2 for i in tile_size)] = 1
    gaussian = gaussian_filter(tmp, [i * sigma_scale for i in tile_size], 0, mode='constant', cval=0)
    gaussian = (gaussian / gaussian.max() * value_scaling_factor).astype(np.float32)
    gaussian[gaussian == 0] = gaussian[gaussian != 0].min()
    return gaussian


def compute_steps(image_size: tuple[int, ...], tile_size: tuple[int, ...], tile_step_size: float) -> list[list[int]]:
    """nnU-Net's compute_steps_for_sliding_window."""
    steps = []
    for size, tile in zip(image_size, tile_size):
        num_steps = int(np.ceil((size - tile) / (tile * tile_step_size))) + 1
        actual = (size - tile) / (num_steps - 1) if num_steps > 1 else 0
        steps.append([int(np.round(actual * i)) for i in range(num_steps)])
    return steps


def nonzero_bbox(volume: np.ndarray) -> tuple[slice, ...]:
    """Half-open bounding box of the non-zero region: the box nnU-Net's crop_to_nonzero crops
    to. nnU-Net hole-fills the mask in 3D first, but that cannot change the box: a zero voxel
    outside the box lies in an all-zero slab that touches the volume's border, so it is never
    a hole. Skipping the fill saves minutes on a full pullback. An all-zero volume keeps its
    full extent, as get_bbox_from_mask does."""
    mask = volume != 0
    slicer = []
    for axis in range(mask.ndim):
        other = tuple(a for a in range(mask.ndim) if a != axis)
        hits = np.flatnonzero(mask.any(axis=other))
        slicer.append(slice(int(hits[0]), int(hits[-1]) + 1) if hits.size else slice(0, mask.shape[axis]))
    return tuple(slicer)


def zscore_stats(volume: np.ndarray, chunk: int = 32) -> tuple[np.float32, float]:
    """Mean and std of the whole volume (float64 accumulation), without a float copy of it."""
    if volume.dtype.kind in 'ub' and volume.dtype.itemsize <= 2:
        counts = np.zeros(1 << (8 * volume.dtype.itemsize), dtype=np.int64)
        for start in range(0, volume.shape[0], chunk):
            part = volume[start : start + chunk].ravel()
            counts += np.bincount(part, minlength=counts.size)
        values = np.arange(counts.size, dtype=np.float64)
        n = counts.sum()
        mean = float((counts * values).sum() / n)
        var = float((counts * (values - mean) ** 2).sum() / n)
    else:
        n = volume.size
        mean = sum(float(volume[s : s + chunk].sum(dtype=np.float64)) for s in range(0, volume.shape[0], chunk)) / n
        var = (
            sum(
                float(((volume[s : s + chunk].astype(np.float64) - mean) ** 2).sum())
                for s in range(0, volume.shape[0], chunk)
            )
            / n
        )
    # float32 values, as nnU-Net's normalisation of a float32 image yields.
    return np.float32(mean), max(float(np.float32(np.sqrt(var))), 1e-8)


class OnnxSegmenter:
    def __init__(
        self,
        model_path: str | os.PathLike | None = None,
        use_mirroring: bool = True,
        tile_step_size: float = 0.5,
        batch_size: int = 2,
        num_threads: int = 0,
        device: str = 'auto',
        precision: str = 'auto',
    ) -> None:
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.use_mirroring = use_mirroring
        self.tile_step_size = tile_step_size
        self.batch_size = max(1, int(batch_size))
        self.num_threads = num_threads
        self.device = str(device or 'auto').lower()
        self.precision = str(precision or 'auto').lower()
        self._session = None
        self._info: ModelInfo | None = None
        self.active_device = ''  # 'cuda', 'directml' or 'cpu' once loaded
        self.active_precision = ''  # 'fp32' or 'fp16' once loaded
        self.fallback_reason = ''  # why 'auto' ended up on the CPU although a GPU runtime is installed

    # -- loading ---------------------------------------------------------------------------
    def _create_session(self, ort, providers: list, options=None, model_path: Path | None = None):
        options = options or ort.SessionOptions()
        if self.num_threads:
            options.intra_op_num_threads = int(self.num_threads)
        return ort.InferenceSession(str(model_path or self.model_path), options, providers=providers)

    def _test_input(self, session) -> np.ndarray:
        """Zero tiles of the one shape predict() always uses: DirectML specialises on the first
        shape it runs."""
        meta = session.get_modelmeta().custom_metadata_map
        patch = [int(s) for s in meta.get('patch_size', '512,448').split(',')]
        return np.zeros((self.batch_size, 1, *patch), dtype=np.float32)

    def _try_gpu_session(self, ort, device: str):
        """A session running on `device` ('cuda' or 'directml'), or (None, reason). Picks the
        float16 model when segmentation.precision asks for it (see _pick_precision)."""
        provider = _GPU_PROVIDERS[device]
        if provider not in ort.get_available_providers():
            return None, _MISSING_RUNTIME[device]
        if device == 'cuda' and hasattr(ort, 'preload_dlls'):  # CUDA/cuDNN from the nvidia-* pip wheels
            try:
                ort.preload_dlls()
            except Exception as e:  # noqa: BLE001 - reported, then the session attempt decides
                logger.warning(f'onnxruntime.preload_dlls failed: {e}')
        session, reason = self._gpu_session(ort, device, self.model_path, fp16=False)
        if session is None:
            return None, reason
        return self._pick_precision(ort, device, session), ''

    def _pick_precision(self, ort, device: str, fp32_session):
        """The float32 session, or a float16 one of the model's _fp16 copy: always for
        precision 'fp16', for 'auto' only when it runs clearly faster on this GPU (some GPUs,
        e.g. GTX 10xx, run float16 far slower). Any problem with float16 keeps float32."""
        self.active_precision = 'fp32'
        if self.precision == 'fp32':
            return fp32_session
        path = fp16_model_path(self.model_path)
        if not path.is_file():
            if self.precision == 'fp16':
                logger.warning(f'segmentation.precision is "fp16", but {path.name} does not exist; using fp32')
            return fp32_session
        fp16_session, reason = self._gpu_session(ort, device, path, fp16=True)
        if fp16_session is None:
            logger.warning(f'fp16 model not usable, using fp32: {reason}')
            return fp32_session
        if self.precision == 'auto':
            x = self._test_input(fp32_session)
            t32, t16 = _median_run_time(fp32_session, x), _median_run_time(fp16_session, x)
            logger.info(f'{device}: fp32 {t32 * 1000:.0f} ms, fp16 {t16 * 1000:.0f} ms per network call')
            if t16 > _FP16_MAX_TIME_RATIO * t32:
                return fp32_session
        self.active_precision = 'fp16'
        return fp16_session

    def _gpu_session(self, ort, device: str, model_path: Path, fp16: bool):
        """A session of model_path on `device`, or (None, reason). onnxruntime silently falls
        back to the CPU when a GPU provider cannot be initialised or fails on the first run, so
        the provider the session actually got is checked and a test inference is run with the
        fallback disabled: 'GPU' then really means GPU."""
        provider = _GPU_PROVIDERS[device]
        name = 'CUDA' if device == 'cuda' else 'DirectML'
        options = ort.SessionOptions()
        if device == 'cuda':
            # use_tf32=0: TF32 convolutions (RTX 30xx and newer) shift hundreds of mask pixels per
            # frame away from the validated float32 result; full float32 is still ~16x the CPU.
            # The float16 model has no float32 convolutions left, and use_tf32=0 would also keep
            # its float16 ones off the tensor cores (2.5x slower than float32 on a Quadro RTX 4000).
            provider_options = {'use_tf32': '1' if fp16 else '0'}
        else:
            # high_performance: the discrete GPU on laptops that also have an integrated one
            # (device 0 is often the iGPU, ~6x slower). The DirectML provider requires the two
            # session settings below.
            provider_options = {'performance_preference': 'high_performance'}
            options.enable_mem_pattern = False
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        try:
            session = self._create_session(
                ort, [(provider, provider_options), 'CPUExecutionProvider'], options, model_path
            )
        except Exception as e:
            return None, f'Could not start {name} inference: {e}'
        if session.get_providers()[0] != provider:
            return None, _INIT_FAILED[device]
        session.disable_fallback()
        try:
            session.run(None, {'image': self._test_input(session)})
        except Exception as e:
            logger.warning(f'{name} test inference failed: {e}')
            return None, f'{name} inference failed on the GPU ({str(e).splitlines()[0][:200]})'
        return session, ''

    def load(self) -> None:
        """Create the onnxruntime session. Raises SegmentationUnavailable with a user-facing
        reason when that is not possible, so callers can block the feature cleanly.
        device 'auto' tries CUDA, then DirectML (any GPU on Windows), then the CPU; 'cuda' and
        'directml' require that GPU runtime and refuse otherwise."""
        if self._session is not None:
            return
        if self.device not in DEVICES:
            raise SegmentationUnavailable(
                f'segmentation.device must be one of {", ".join(DEVICES)}, not {self.device!r}'
            )
        if self.precision not in PRECISIONS:
            raise SegmentationUnavailable(
                f'segmentation.precision must be one of {", ".join(PRECISIONS)}, not {self.precision!r}'
            )
        ort = _import_onnxruntime()
        if not self.model_path.is_file():
            raise SegmentationUnavailable(f'Segmentation model not found:\n{self.model_path}')
        session = None
        reasons: list[str] = []  # only from GPU runtimes that are installed but did not work
        try:
            if self.device in _GPU_PROVIDERS:
                session, reason = self._try_gpu_session(ort, self.device)
                if session is None:
                    raise SegmentationUnavailable(
                        f'segmentation.device is "{self.device}", but {reason[0].lower()}{reason[1:]}'
                    )
            elif self.device == 'auto':
                for device in _GPU_PROVIDERS:
                    session, reason = self._try_gpu_session(ort, device)
                    if session is not None:
                        break
                    if reason != _MISSING_RUNTIME[device]:  # installed but unusable: worth reporting
                        reasons.append(reason)
            if session is None:
                session = self._create_session(ort, ['CPUExecutionProvider'])
        except SegmentationUnavailable:
            raise
        except Exception as e:  # corrupt file, unsupported opset, CPU without required instructions, ...
            raise SegmentationUnavailable(f'Could not load the segmentation model {self.model_path.name}:\n{e}') from e
        provider = session.get_providers()[0]
        self.active_device = next((d for d, p in _GPU_PROVIDERS.items() if p == provider), 'cpu')
        if self.active_device == 'cpu':
            self.active_precision = 'fp32'
        # Only worth mentioning when a GPU runtime is installed but unusable; a CPU-only package
        # (or a machine without a GPU) is a normal setup, not a fallback.
        self.fallback_reason = ' '.join(reasons) if self.active_device == 'cpu' else ''
        if self.fallback_reason:
            logger.warning(f'Automatic segmentation falls back to the CPU: {self.fallback_reason}')
        logger.info(
            f'Segmentation model {self.model_path.name} loaded on {self.active_device.upper()} ({self.active_precision})'
        )
        meta = session.get_modelmeta().custom_metadata_map
        try:
            if meta.get('normalization') != 'zscore':
                raise ValueError(f'unsupported normalization {meta.get("normalization")!r}')
            info = ModelInfo(
                patch_size=tuple(int(s) for s in meta['patch_size'].split(',')),  # type: ignore[arg-type]
                mirror_axes=tuple(int(a) for a in meta.get('mirror_axes', '').split(',') if a != ''),
                num_classes=int(meta['num_classes']),
                labels=json.loads(meta.get('labels', '{}')),
            )
        except (KeyError, ValueError) as e:
            raise SegmentationUnavailable(
                f'{self.model_path.name} is not a HolOrama segmentation model (metadata: {e}).\n'
                'Export models with scripts/model_export/export_onnx.py.'
            ) from e
        self._session, self._info = session, info

    @property
    def info(self) -> ModelInfo:
        self.load()
        assert self._info is not None
        return self._info

    # -- inference -------------------------------------------------------------------------
    def _mirror_combinations(self) -> list[tuple[int, ...]]:
        if not self.use_mirroring or not self.info.mirror_axes:
            return [()]
        axes = [a + 2 for a in self.info.mirror_axes]  # (batch, channel, y, x)
        return [()] + [c for i in range(len(axes)) for c in itertools.combinations(axes, i + 1)]

    def _run(self, tiles: np.ndarray) -> np.ndarray:
        """Mirror-averaged logits for a (n, 1, py, px) stack of tiles."""
        assert self._session is not None
        combos = self._mirror_combinations()
        result = None
        for axes in combos:
            batch = np.ascontiguousarray(np.flip(tiles, axes) if axes else tiles)
            logits = self._session.run(None, {'image': batch})[0]
            logits = np.flip(logits, axes) if axes else logits
            result = logits if result is None else result + logits
        assert result is not None
        return result / len(combos)

    def predict(
        self,
        volume: np.ndarray,
        frames: Iterable[int] | None = None,
        progress_cb: Callable[[int, int], bool] | None = None,
    ) -> np.ndarray | None:
        """Segment `frames` (default: all) of a (frames, y, x) volume. Returns a uint8 label
        volume of the same shape (frames not requested stay 0), or None if progress_cb
        returned False (cancelled). progress_cb(frames done, total) is called after each network call."""
        self.load()
        info = self.info
        if volume.ndim != 3:
            raise ValueError(f'Expected a (frames, y, x) volume, got shape {volume.shape}')

        crop = nonzero_bbox(volume)
        cropped = volume[crop]  # a view; frames are normalised one at a time below
        mean, std = zscore_stats(cropped)

        out = np.zeros(volume.shape, dtype=np.uint8)
        z0, z1 = crop[0].start, crop[0].stop
        wanted = range(volume.shape[0]) if frames is None else frames
        todo = sorted({z for z in wanted if z0 <= z < z1})
        total = len(todo)

        patch = info.patch_size
        frame_shape = cropped.shape[1:]
        padded_shape = tuple(max(p, s) for p, s in zip(patch, frame_shape))
        pad = [((p - s) // 2, (p - s) - (p - s) // 2) for p, s in zip(padded_shape, frame_shape)]
        unpad = tuple(slice(lo, size - hi) for (lo, hi), size in zip(pad, padded_shape))
        steps = compute_steps(padded_shape, patch, self.tile_step_size)
        tile_slicers = [(slice(sy, sy + patch[0]), slice(sx, sx + patch[1])) for sy in steps[0] for sx in steps[1]]
        gaussian = compute_gaussian(patch)
        weight = np.zeros(padded_shape, dtype=np.float32)
        for sl in tile_slicers:
            weight[sl] += gaussian

        # Every network call gets exactly batch_size tiles (the last one padded with zero tiles whose
        # output is dropped), whatever the frame size: DirectML optimises for the first input shape
        # it sees and runs any other batch size 2-6x slower. Tiles of consecutive frames share calls.
        n_tiles = len(tile_slicers)
        tiles = [(z, j) for z in todo for j in range(n_tiles)]
        batch = np.zeros((self.batch_size, 1, *patch), dtype=np.float32)
        padded: dict[int, np.ndarray] = {}  # frames whose tiles are in flight
        acc: dict[int, np.ndarray] = {}
        done = 0
        for start in range(0, len(tiles), self.batch_size):
            chunk = tiles[start : start + self.batch_size]
            batch[:] = 0
            for i, (z, j) in enumerate(chunk):
                if z not in padded:
                    frame = (cropped[z - z0].astype(np.float32) - mean) / std
                    padded[z] = np.pad(frame, pad, mode='constant', constant_values=0)
                batch[i, 0] = padded[z][tile_slicers[j]]
            logits = self._run(batch)
            for i, (z, j) in enumerate(chunk):
                if z not in acc:
                    acc[z] = np.zeros((info.num_classes, *padded_shape), dtype=np.float32)
                acc[z][(slice(None), *tile_slicers[j])] += logits[i] * gaussian
                if j == n_tiles - 1:  # frame complete
                    frame_acc = acc.pop(z) / weight
                    del padded[z]
                    out[z, crop[1], crop[2]] = frame_acc[(slice(None), *unpad)].argmax(axis=0).astype(np.uint8)
                    done += 1
            if progress_cb is not None and progress_cb(done, total) is False:
                return None
        return out
