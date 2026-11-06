"""nnU-Net 2D inference re-implemented on numpy + onnxruntime.

The automatic IVUS segmentation model was trained with nnU-Net (nnUZoo's U2NetP trainer).
Running it through nnU-Net itself needs torch, nnunetv2 and a patched nnUZoo, which made the
install fragile and the app impossible to compile. Instead, the network is exported once to
ONNX (scripts/model_export/export_onnx.py) and this module reproduces exactly the steps
nnUNetPredictor.predict_single_npy_array performs for a 2D configuration:

1. crop the (frames, y, x) volume to the bounding box of its non-zero region
   (binary-hole-filled in 3D), as nnU-Net's crop_to_nonzero does
2. z-score normalise with the mean/std of that whole cropped volume
   (resampling is a no-op: the model's target spacing is 1 and frames are fed with spacing 1)
3. per frame: centre-pad to at least the patch size, tile with a sliding window
   (step = tile_step_size * patch), run every tile plus its mirrored variants, weight each
   tile's logits by nnU-Net's Gaussian importance map and average
4. argmax over classes, undo padding and cropping

Verified against nnUNetPredictor on the model's own test frames by
scripts/model_export/verify_onnx.py. The only intentional difference: nnU-Net accumulates
logits in float16, this module in float32, which can flip a handful of borderline pixels.

Only the requested frames are run through the network, but normalisation statistics and the
crop are always computed on the whole volume, so a frame's mask does not depend on which
range was requested — the same result nnU-Net gave when it was fed the whole pullback.
"""

from __future__ import annotations

import itertools
import json
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger
from scipy.ndimage import binary_fill_holes, gaussian_filter

# Bundled model, shipped next to this module (and next to the compiled module in the Nuitka build).
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / 'weights' / 'ivus_lumen_u2netp.onnx'
# 'auto': NVIDIA GPU if the onnxruntime-gpu package is installed and CUDA works, else CPU.
DEVICES = ('auto', 'cpu', 'cuda')


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
    """Half-open bounding box of the hole-filled non-zero region (nnU-Net's crop_to_nonzero).
    An all-zero volume keeps its full extent, as get_bbox_from_mask does."""
    mask = binary_fill_holes(volume != 0)
    slicer = []
    for axis in range(mask.ndim):
        other = tuple(a for a in range(mask.ndim) if a != axis)
        hits = np.flatnonzero(mask.any(axis=other))
        slicer.append(slice(int(hits[0]), int(hits[-1]) + 1) if hits.size else slice(0, mask.shape[axis]))
    return tuple(slicer)


class OnnxSegmenter:
    def __init__(
        self,
        model_path: str | os.PathLike | None = None,
        use_mirroring: bool = True,
        tile_step_size: float = 0.5,
        batch_size: int = 2,
        num_threads: int = 0,
        device: str = 'auto',
    ) -> None:
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.use_mirroring = use_mirroring
        self.tile_step_size = tile_step_size
        self.batch_size = max(1, int(batch_size))
        self.num_threads = num_threads
        self.device = str(device or 'auto').lower()
        self._session = None
        self._info: ModelInfo | None = None
        self.active_device = ''  # 'cuda' or 'cpu' once loaded
        self.fallback_reason = ''  # why 'auto' ended up on the CPU although a GPU runtime is installed

    # -- loading ---------------------------------------------------------------------------
    def _create_session(self, ort, providers: list[str | tuple[str, dict[str, str]]]):
        options = ort.SessionOptions()
        if self.num_threads:
            options.intra_op_num_threads = int(self.num_threads)
        return ort.InferenceSession(str(self.model_path), options, providers=providers)

    def _try_cuda_session(self, ort):
        """A session running on the GPU, or (None, reason). onnxruntime silently falls back to
        the CPU when CUDA cannot be initialised (old driver, missing DLLs), so the providers the
        session actually got are checked rather than trusted."""
        if 'CUDAExecutionProvider' not in ort.get_available_providers():
            return None, (
                'GPU inference needs the "onnxruntime-gpu" package (source installs: '
                'install script with -Gpu / --gpu, or uv sync --extra segmentation-gpu).'
            )
        if hasattr(ort, 'preload_dlls'):  # CUDA/cuDNN from the nvidia-* pip wheels
            try:
                ort.preload_dlls()
            except Exception as e:  # noqa: BLE001 - reported, then the session attempt decides
                logger.warning(f'onnxruntime.preload_dlls failed: {e}')
        # use_tf32=0: TF32 convolutions (RTX 30xx and newer) shift hundreds of mask pixels per
        # frame away from the validated float32 result; full float32 is still ~30x the CPU.
        cuda = ('CUDAExecutionProvider', {'use_tf32': '0'})
        try:
            session = self._create_session(ort, [cuda, 'CPUExecutionProvider'])
        except Exception as e:
            return None, f'Could not start CUDA inference: {e}'
        if session.get_providers()[0] != 'CUDAExecutionProvider':
            return None, (
                'CUDA could not be initialised (no NVIDIA GPU, or the driver is too old: CUDA 12 needs '
                'driver >= 527.41 on Windows / 525.60 on Linux). See the log for onnxruntime\'s message.'
            )
        # A CUDA session can still fail on its first run (e.g. a cuDNN that does not match the
        # onnxruntime build), and onnxruntime's Python wrapper would then silently rebuild the
        # session on the CPU. Disable that and run once now, so 'GPU' really means GPU.
        session.disable_fallback()
        patch = [int(s) for s in session.get_modelmeta().custom_metadata_map.get('patch_size', '512,448').split(',')]
        try:
            session.run(None, {'image': np.zeros((1, 1, *patch), dtype=np.float32)})
        except Exception as e:
            logger.warning(f'CUDA test inference failed: {e}')
            return None, f'CUDA inference failed on the GPU ({str(e).splitlines()[0][:200]})'
        return session, ''

    def load(self) -> None:
        """Create the onnxruntime session. Raises SegmentationUnavailable with a user-facing
        reason when that is not possible, so callers can block the feature cleanly.
        device 'auto' uses the GPU when it works and the CPU otherwise; 'cuda' requires the GPU."""
        if self._session is not None:
            return
        if self.device not in DEVICES:
            raise SegmentationUnavailable(
                f'segmentation.device must be one of {", ".join(DEVICES)}, not {self.device!r}'
            )
        ort = _import_onnxruntime()
        if not self.model_path.is_file():
            raise SegmentationUnavailable(f'Segmentation model not found:\n{self.model_path}')
        session, reason = None, ''
        try:
            if self.device in ('auto', 'cuda'):
                session, reason = self._try_cuda_session(ort)
                if session is None and self.device == 'cuda':
                    raise SegmentationUnavailable(f'segmentation.device is "cuda", but {reason[0].lower()}{reason[1:]}')
            if session is None:
                session = self._create_session(ort, ['CPUExecutionProvider'])
        except SegmentationUnavailable:
            raise
        except Exception as e:  # corrupt file, unsupported opset, CPU without required instructions, ...
            raise SegmentationUnavailable(f'Could not load the segmentation model {self.model_path.name}:\n{e}') from e
        self.active_device = 'cuda' if session.get_providers()[0] == 'CUDAExecutionProvider' else 'cpu'
        # Only worth mentioning when a GPU runtime is installed but unusable; the CPU package is
        # the normal case, not a fallback.
        self.fallback_reason = reason if self.active_device == 'cpu' and 'onnxruntime-gpu' not in reason else ''
        if self.fallback_reason:
            logger.warning(f'Automatic segmentation falls back to the CPU: {self.fallback_reason}')
        logger.info(f'Segmentation model {self.model_path.name} loaded on {self.active_device.upper()}')
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
        returned False (cancelled). progress_cb(done, total) is called after each frame."""
        self.load()
        info = self.info
        if volume.ndim != 3:
            raise ValueError(f'Expected a (frames, y, x) volume, got shape {volume.shape}')

        crop = nonzero_bbox(volume)
        cropped = volume[crop].astype(np.float32)
        mean, std = cropped.mean(), cropped.std()
        cropped = (cropped - mean) / max(float(std), 1e-8)

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

        # Tiles of consecutive frames are batched together; each batch covers whole frames.
        frames_per_batch = max(1, self.batch_size // len(tile_slicers))
        for start in range(0, total, frames_per_batch):
            chunk = todo[start : start + frames_per_batch]
            padded = [np.pad(cropped[z - z0], pad, mode='constant', constant_values=0) for z in chunk]
            tiles = np.stack([p[sl] for p in padded for sl in tile_slicers])[:, None]
            logits = self._run(tiles)
            for i, z in enumerate(chunk):
                acc = np.zeros((info.num_classes, *padded_shape), dtype=np.float32)
                for j, sl in enumerate(tile_slicers):
                    acc[(slice(None), *sl)] += logits[i * len(tile_slicers) + j] * gaussian
                acc /= weight
                seg = acc[(slice(None), *unpad)].argmax(axis=0).astype(np.uint8)
                out[z, crop[1], crop[2]] = seg
            if progress_cb is not None and progress_cb(min(start + len(chunk), total), total) is False:
                return None
        return out
