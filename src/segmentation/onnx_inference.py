"""nnU-Net 2D/3D inference re-implemented on numpy + onnxruntime.

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
(<name>_fp16.onnx, written by scripts/model_export/export_onnx.py), with float32 input and output, so
everything around the network stays float32. On a 1163 frame pullback this moved 2 to 4 border
pixels per frame (min Dice 0.9998) for 1.3x (DirectML) to 1.5x (CUDA) faster segmentation on a
Quadro RTX 4000. The CPU always runs float32.

Only the requested frames are run through the network, but normalisation statistics and the
crop are always computed on the whole volume, so a frame's mask does not depend on which
range was requested, the same result nnU-Net gave when it was fed the whole pullback.

Third-party code
----------------
Several functions are copied or adapted from MIC-DKFZ projects, all under the Apache License 2.0
(Copyright Division of Medical Image Computing, German Cancer Research Center (DKFZ),
Heidelberg, Germany). Each one names its original in its docstring.

- nnunetv2 (https://github.com/MIC-DKFZ/nnUNet; compared against 2.8.1, the resampling code
  comes from an earlier 2.x release):
    get_lowres_axis, get_do_separate_z, resample_data_or_seg_to_shape, resample_data_or_seg
        nnunetv2/preprocessing/resampling/default_resampling.py
    compute_steps, compute_gaussian
        nnunetv2/inference/sliding_window_prediction.py
    ct_normalization
        nnunetv2/preprocessing/normalization/default_normalization_schemes.py (CTNormalization)
- batchgenerators (https://github.com/MIC-DKFZ/batchgenerators; 0.25.3):
    resize_segmentation
        batchgenerators/augmentations/utils.py

Written for this module but reproducing nnU-Net step for step (see their docstrings):
nonzero_bbox, zscore_stats, resample_logits_to_labels, OnnxSegmenter._run,
OnnxSegmenter._sliding_window_3d and the tile loop of OnnxSegmenter._ivus_predict.
"""

from __future__ import annotations

from typing import Any, List, Literal, Tuple, Union
from os.path import join, split
import itertools
import json
import os
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from copy import deepcopy

from scipy.ndimage import map_coordinates
from skimage.transform import resize
import pandas as pd
import numpy as np
from deep_utils import JsonUtils
from loguru import logger
from scipy.ndimage import gaussian_filter

from domain.ccta.io_types import CANONICAL_ORIENTATION
from domain.intravascular.segmentation_types import (
    DEVICES,
    FP16_MAX_TIME_RATIO,
    PRECISIONS,
    ModelInfo,
    SegmentationSettings,
)
from .utils import compute_new_shape

# Bundled model, shipped next to this module (and next to the compiled module in the Nuitka build).
DEFAULT_IVUS_MODEL_PATH = Path(__file__).resolve().parent / "weights" / "ivus_lumen" / "ivus_lumen_u2netp.onnx"
DEFAULT_CT_SEG_MODEL_PATH = Path(__file__).resolve().parent / "weights" / "ct_seg" / "ct_seg_u2net_small.onnx"
_GPU_PROVIDERS = {
    "cuda": "CUDAExecutionProvider",
    "directml": "DmlExecutionProvider",
}  # in 'auto' order
_MISSING_RUNTIME = {
    "cuda": 'CUDA inference needs the "onnxruntime-gpu" package (source installs: install script with '
    "-Gpu / --gpu, or uv sync --extra segmentation-gpu).",
    "directml": 'DirectML inference needs the "onnxruntime-directml" package (Windows only; installed '
    'by the "segmentation" extra there).',
}
_INIT_FAILED = {
    "cuda": "CUDA could not be initialised (no NVIDIA GPU, or the driver is too old: CUDA 12 needs driver "
    ">= 527.41 on Windows / 525.60 on Linux). See the log for onnxruntime's message.",
    "directml": "DirectML could not be initialised (no DirectX 12 capable GPU). See the log for onnxruntime's message.",
}


class SegmentationUnavailable(RuntimeError):
    """Raised when automatic segmentation cannot run on this installation (runtime missing,
    model missing or unreadable). The message is meant to be shown to the user as is."""


class OnnxSegmenter:
    def __init__(self, settings: SegmentationSettings | None = None) -> None:
        settings = settings or SegmentationSettings()
        self.model_path = resolve_model_path(settings.model_file, seg_type=settings.seg_type)
        self.device = str(settings.device or "auto").lower()
        self.precision = str(settings.precision or "auto").lower()
        self.use_mirroring = settings.use_mirroring

        self.num_threads = settings.num_threads
        self.seg_type = settings.seg_type

        self._session = None
        self._info: ModelInfo | None = None
        self.active_device = ""  # 'cuda', 'directml' or 'cpu' once loaded
        self.active_precision = ""  # 'fp32' or 'fp16' once loaded
        self.fallback_reason = ""  # why 'auto' ended up on the CPU although a GPU runtime is installed
        self.config = (
            JsonUtils.load(join(split(self.model_path)[0], "config.json")) if self.seg_type == "ct_seg" else {}
        )
        self.tile_step_size = self.config.get("tile_step_size", settings.tile_step_size)
        self.batch_size = max(1, int(self.config.get("batch_size", settings.batch_size)))

    def predict(
        self,
        volume: np.ndarray,
        spacing: list[float] | None = None,
        frames: Iterable[int] | None = None,
        progress_cb: Callable[[int, int], bool] | None = None,
    ) -> np.ndarray | None:
        """Segment `frames` (default: all) of a (frames, y, x) volume. Returns a uint8 label
        volume of the same shape (frames not requested stay 0), or None if progress_cb
        returned False (cancelled). progress_cb(frames done, total) is called after each network call."""
        self.load()
        info = self.info
        if volume.ndim != 3:
            raise ValueError(f"Expected a (frames, y, x) volume, got shape {volume.shape}")
        if self.seg_type == "ivus":
            output = self._ivus_predict(volume, info, frames, progress_cb)
        elif self.seg_type == "ct_seg":
            output = self._ct_seg_predict(volume, spacing, progress_cb)
        else:
            raise ValueError("seg_type is not supported")

        return output

    def _ivus_predict(self, volume, info, frames, progress_cb):
        crop = nonzero_bbox(volume)
        cropped = volume[crop]  # a view, frames are normalised one at a time below
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
                    padded[z] = np.pad(frame, pad, mode="constant", constant_values=0)
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

    def _ct_seg_predict(self, volume, spacing, progress_cb):
        """nnU-Net's predict_single_npy_array for a 3D full-resolution configuration, step by
        step (plans' transpose_forward assumed to be the identity: the app's (Z, Y, X) volume and
        (dz, dy, dx) spacing are already in nnU-Net's array order):

        1. crop to the non-zero bounding box (nonzero_bbox: the same box as crop_to_nonzero)
        2. CT normalisation with the dataset's clipping percentiles, mean and std
        3. resample to the model's spacing (order 3, separate z only for anisotropic spacing)
        4. sliding window with Gaussian weighting
        5. resample the logits back to the cropped shape (order 1)
        6. argmax (softmax would not change it), undo the crop

        The app hands over volumes in CANONICAL_ORIENTATION; nnU-Net reads its training images
        without reorienting them, so the volume is brought into the orientation of the model's
        training data (config.json "orientation") first and the mask back afterwards. A no-op
        while both are the same.

        As for IVUS, logits are accumulated in float32 where nnU-Net uses float16."""
        if spacing is None:
            raise ValueError("CT segmentation needs the volume's voxel spacing")
        if self.config is None:
            raise ValueError("config for the ct model is required")

        model_orientation = self.config.get("orientation", CANONICAL_ORIENTATION)
        app_volume = volume
        volume, perm = reorient_array(app_volume, CANONICAL_ORIENTATION, model_orientation)
        original_spacing = [float(spacing[p]) for p in perm]
        target_spacing = [float(s) for s in self.config["spacing"]]

        tic = time.perf_counter()
        crop = nonzero_bbox(volume)
        data = volume[crop][None].astype(np.float32)  # (1, z, y, x)
        shape_after_cropping = data.shape[1:]
        # Normalisation before resampling, as nnU-Net does.
        data = ct_normalization(
            data,
            mean_intensity=self.config["mean"],
            std_intensity=self.config["std"],
            percentile_00_5=self.config["percentile_00_5"],
            percentile_99_5=self.config["percentile_99_5"],
        )
        new_shape = compute_new_shape(shape_after_cropping, original_spacing, target_spacing)
        data = resample_data_or_seg_to_shape(
            data,
            new_shape,
            original_spacing,
            target_spacing,
            is_seg=False,
            order=3,
            order_z=0,
            force_separate_z=None,
        )
        logger.info(
            f"CT preprocessing {time.perf_counter() - tic:.1f} s: {volume.shape} at {original_spacing} mm "
            f"-> {data.shape[1:]} at {target_spacing} mm"
        )

        tic = time.perf_counter()
        logits = self._sliding_window_3d(data[0], progress_cb)
        if logits is None:
            return None
        del data
        logger.info(f"CT sliding window {time.perf_counter() - tic:.1f} s")

        tic = time.perf_counter()
        labels = resample_logits_to_labels(logits, shape_after_cropping, target_spacing, original_spacing)
        del logits
        out = np.zeros(volume.shape, dtype=np.uint8)
        out[crop] = labels
        out, _ = reorient_array(out, model_orientation, CANONICAL_ORIENTATION)
        out = np.ascontiguousarray(out)
        assert out.shape == app_volume.shape
        logger.info(f"CT postprocessing {time.perf_counter() - tic:.1f} s")
        return out

    def _sliding_window_3d(self, image, progress_cb):
        """nnU-Net's predict_sliding_window_return_logits for one (z, y, x) image: centre-pad to
        at least the patch size, run every tile, weight its logits by the Gaussian importance
        map, divide by the summed weights, undo the padding. Returns (classes, z, y, x) float32
        logits, or None if progress_cb returned False."""
        patch = self.config["patch_size"]
        padded_shape = tuple(max(p, s) for p, s in zip(patch, image.shape))
        pad = [((p - s) // 2, (p - s) - (p - s) // 2) for p, s in zip(padded_shape, image.shape)]
        if any(lo or hi for lo, hi in pad):
            image = np.pad(image, pad, mode="constant", constant_values=0)

        unpad = tuple(slice(lo, size - hi) for (lo, hi), size in zip(pad, padded_shape))
        steps = compute_steps(padded_shape, patch, self.tile_step_size)
        tile_slicers = [tuple(slice(s, s + p) for s, p in zip(start, patch)) for start in itertools.product(*steps)]
        gaussian = compute_gaussian(patch)
        logits = np.zeros((len(self.config["labels"]), *padded_shape), dtype=np.float32)
        weight = np.zeros(padded_shape, dtype=np.float32)

        # Fixed batch shape (last call padded with zero tiles whose output is dropped), see _ivus_predict.
        batch = np.zeros((self.batch_size, 1, *patch), dtype=np.float32)
        total = len(tile_slicers)
        for start in range(0, total, self.batch_size):
            chunk = tile_slicers[start : start + self.batch_size]
            batch[:] = 0
            for i, sl in enumerate(chunk):
                batch[i, 0] = image[sl]
            tile_logits = self._run(batch)
            for i, sl in enumerate(chunk):
                logits[(slice(None), *sl)] += tile_logits[i] * gaussian
                weight[sl] += gaussian
            if progress_cb is not None and progress_cb(start + len(chunk), total) is False:
                return None
        logits /= weight
        return logits[(slice(None), *unpad)]

    # -- loading ---------------------------------------------------------------------------
    def load(self) -> None:
        """Create the onnxruntime session. Raises SegmentationUnavailable with a user-facing
        reason when that is not possible, so callers can block the feature cleanly."""
        if self._session is not None:
            return
        if self.device not in DEVICES:
            raise SegmentationUnavailable(
                f"segmentation.device must be one of {', '.join(DEVICES)}, not {self.device!r}"
            )
        if self.precision not in PRECISIONS:
            raise SegmentationUnavailable(
                f"segmentation.precision must be one of {', '.join(PRECISIONS)}, not {self.precision!r}"
            )
        ort = _import_onnxruntime()
        if not self.model_path.is_file():
            raise SegmentationUnavailable(f"Segmentation model not found:\n{self.model_path}")
        session = self._open_session(ort)
        self._info = self._read_model_info(session)
        self._session = session

    @property
    def info(self) -> ModelInfo:
        self.load()
        assert self._info is not None
        return self._info

    def _open_session(self, ort):
        """device 'auto' tries CUDA, then DirectML (any GPU on Windows), then the CPU; 'cuda' and
        'directml' require that GPU runtime and refuse otherwise."""
        session = None
        reasons: list[str] = []  # only from GPU runtimes that are installed but did not work
        try:
            if self.device in _GPU_PROVIDERS:
                session, reason = self._try_gpu_session(ort, self.device)
                if session is None:
                    raise SegmentationUnavailable(
                        f'segmentation.device is "{self.device}", but {reason[0].lower()}{reason[1:]}'
                    )
            elif self.device == "auto":
                for device in _GPU_PROVIDERS:
                    session, reason = self._try_gpu_session(ort, device)
                    if session is not None:
                        break
                    if reason != _MISSING_RUNTIME[device]:  # installed but unusable: worth reporting
                        reasons.append(reason)
            if session is None:
                session = self._create_session(ort, ["CPUExecutionProvider"])
        except SegmentationUnavailable:
            raise
        except Exception as e:  # corrupt file, unsupported opset, CPU without required instructions, ...
            raise SegmentationUnavailable(f"Could not load the segmentation model {self.model_path.name}:\n{e}") from e
        provider = session.get_providers()[0]
        self.active_device = next((d for d, p in _GPU_PROVIDERS.items() if p == provider), "cpu")
        if self.active_device == "cpu":
            self.active_precision = "fp32"
        # Only worth mentioning when a GPU runtime is installed but unusable. A CPU-only package
        # (or a machine without a GPU) is a normal setup, not a fallback.
        self.fallback_reason = " ".join(reasons) if self.active_device == "cpu" else ""
        if self.fallback_reason:
            logger.warning(f"Automatic segmentation falls back to the CPU: {self.fallback_reason}")
        logger.info(
            f"Segmentation model {self.model_path.name} loaded on {self.active_device.upper()} ({self.active_precision})"
        )
        return session

    def _try_gpu_session(self, ort, device: str):
        """A session running on `device` ('cuda' or 'directml'), or (None, reason). Picks the
        float16 model when segmentation.precision asks for it (see _pick_precision)."""
        provider = _GPU_PROVIDERS[device]
        if provider not in ort.get_available_providers():
            return None, _MISSING_RUNTIME[device]
        if device == "cuda" and hasattr(ort, "preload_dlls"):  # CUDA/cuDNN from the nvidia-* pip wheels
            try:
                ort.preload_dlls()
            except Exception as e:  # noqa: BLE001 - reported, then the session attempt decides
                logger.warning(f"onnxruntime.preload_dlls failed: {e}")

        return self._pick_precision(ort, device)

    def _gpu_session(self, ort, device: str, model_path: Path, fp16: bool):
        """A session of model_path on `device`, or (None, reason). onnxruntime silently falls
        back to the CPU when a GPU provider cannot be initialised or fails on the first run, so
        the provider the session actually got is checked and a test inference is run with the
        fallback disabled: 'GPU' then really means GPU."""
        provider = _GPU_PROVIDERS[device]
        name = "CUDA" if device == "cuda" else "DirectML"
        options = ort.SessionOptions()
        if device == "cuda":
            # use_tf32=0: TF32 convolutions (RTX 30xx and newer) shift hundreds of mask pixels per
            # frame away from the validated float32 result, full float32 is still ~16x the CPU.
            # The float16 model has no float32 convolutions left, and use_tf32=0 would also keep
            # its float16 ones off the tensor cores (2.5x slower than float32 on a Quadro RTX 4000).
            provider_options = {"use_tf32": "1" if fp16 else "0"}
        else:
            # high_performance: the discrete GPU on laptops that also have an integrated one
            # (device 0 is often the iGPU, ~6x slower). The DirectML provider requires the two
            # session settings below.
            provider_options = {"performance_preference": "high_performance"}
            options.enable_mem_pattern = False
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        try:
            session = self._create_session(
                ort,
                [(provider, provider_options), "CPUExecutionProvider"],
                options,
                model_path,
            )
        except Exception as e:
            return None, f"Could not start {name} inference: {e}"
        if session.get_providers()[0] != provider:
            return None, _INIT_FAILED[device]
        session.disable_fallback()
        try:
            session.run(None, {"image": self._test_input(session)})
        except Exception as e:
            logger.warning(f"{name} test inference failed: {e}")
            return (
                None,
                f"{name} inference failed on the GPU ({str(e).splitlines()[0][:200]})",
            )
        return session, ""

    def _create_session(self, ort, providers: list, options=None, model_path: Path | None = None):
        options = options or ort.SessionOptions()
        if self.num_threads:
            options.intra_op_num_threads = int(self.num_threads)
        return ort.InferenceSession(str(model_path or self.model_path), options, providers=providers)

    def _test_input(self, session) -> np.ndarray:
        """Zero tiles of the one shape predict() always uses: DirectML specialises on the first
        shape it runs."""
        meta = session.get_modelmeta().custom_metadata_map
        patch = [int(s) for s in meta.get("patch_size", "512,448").split(",")]
        return np.zeros((self.batch_size, 1, *patch), dtype=np.float32)

    def _pick_precision(self, ort, device: str):
        """The float32 session, or a float16 one of the model's _fp16 copy: always for
        precision 'fp16', for 'auto' only when it runs clearly faster on this GPU (some GPUs,
        e.g. GTX 10xx, run float16 far slower). Any problem with float16 keeps float32. float16
        alone is used when float32 does not run, e.g. with too little GPU memory. (None, reason)
        when neither runs."""
        self.active_precision = 'fp32'
        path = fp16_model_path(self.model_path)
        if self.precision == 'fp16' and not path.is_file():
            logger.warning(f'segmentation.precision is "fp16", but {path.name} does not exist, using fp32')
        fp32_session, reason = self._gpu_session(ort, device, self.model_path, fp16=False)
        if self.precision == 'fp32' or not path.is_file():
            return fp32_session, reason
        fp16_session, fp16_reason = self._gpu_session(ort, device, path, fp16=True)
        if fp16_session is None:
            if fp32_session is not None:
                logger.warning(f'fp16 model not usable, using fp32: {fp16_reason}')
            return fp32_session, reason
        if fp32_session is not None and self.precision == 'auto':
            x = self._test_input(fp32_session)
            t32, t16 = _median_run_time(fp32_session, x), _median_run_time(fp16_session, x)
            logger.info(f'{device}: fp32 {t32 * 1000:.0f} ms, fp16 {t16 * 1000:.0f} ms per network call')
            if t16 > FP16_MAX_TIME_RATIO * t32:
                return fp32_session, ''
        self.active_precision = 'fp16'
        return fp16_session, ''

    def _read_model_info(self, session) -> ModelInfo:
        meta = session.get_modelmeta().custom_metadata_map
        try:
            return ModelInfo(
                patch_size=tuple(int(s) for s in meta["patch_size"].split(",")),  # type: ignore[arg-type]
                mirror_axes=tuple(int(a) for a in meta.get("mirror_axes", "").split(",") if a != ""),
                num_classes=int(meta["num_classes"]),
                labels=json.loads(meta.get("labels", "{}")),
            )
        except (KeyError, ValueError) as e:
            raise SegmentationUnavailable(
                f"{self.model_path.name} is not a HolOrama segmentation model (metadata: {e}).\n"
                "Export models with scripts/model_export/export_onnx.py."
            ) from e

    # -- inference -------------------------------------------------------------------------
    def _run(self, tiles: np.ndarray) -> np.ndarray:
        """Mirror-averaged logits for a (n, 1, py, px) stack of tiles."""
        assert self._session is not None
        combos = self._mirror_combinations()
        result = None
        for axes in combos:
            batch = np.ascontiguousarray(np.flip(tiles, axes) if axes else tiles)
            logits = self._session.run(None, {"image": batch})[0]
            logits = np.flip(logits, axes) if axes else logits
            result = logits if result is None else result + logits
        assert result is not None
        return result / len(combos)

    def _mirror_combinations(self) -> list[tuple[int, ...]]:
        if not self.use_mirroring or not self.info.mirror_axes:
            return [()]
        axes = [a + 2 for a in self.info.mirror_axes]  # (batch, channel, y, x)
        return [()] + [c for i in range(len(axes)) for c in itertools.combinations(axes, i + 1)]


def get_lowres_axis(new_spacing: Union[Tuple[float, ...], List[float], np.ndarray]):
    """Adapted from nnunetv2.preprocessing.resampling.default_resampling.get_lowres_axis
    (same result, ratio written the other way round)."""
    axis = np.where(np.array(new_spacing) / max(new_spacing) == 1)[0]  # find which axis is anisotropic
    # if isinstance(axis, np.ndarray):
    #     axis = axis[0]
    # if not isinstance(axis, (np.int64, np.int32, np.int16, np.int8)):
    #     axis = axis[0]
    return axis


def get_do_separate_z(spacing: Union[Tuple[float, ...], List[float], np.ndarray], anisotropy_threshold=3):
    """From nnunetv2.preprocessing.resampling.default_resampling.get_do_separate_z
    (anisotropy_threshold = nnunetv2.configuration.ANISO_THRESHOLD)."""
    do_separate_z = (np.max(spacing) / np.min(spacing)) > anisotropy_threshold
    return do_separate_z


def resample_data_or_seg_to_shape(
    data: Union[np.ndarray],
    new_shape: Union[Tuple[int, ...], List[int], np.ndarray],
    current_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
    new_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
    is_seg: bool = False,
    order: int = 3,
    order_z: int = 0,
    force_separate_z: Union[bool, None] = False,
    separate_z_anisotropy_threshold: float = 3,
    verbose=False,
):
    """
    needed for segmentation export. Stupid, I know. Maybe we can fix that with Leos new resampling functions

    Adapted from nnunetv2.preprocessing.resampling.default_resampling.resample_data_or_seg_to_shape
    (an earlier 2.x release, before the separate-z decision moved into determine_do_sep_z_and_axis).
    Changes: numpy only (no torch.Tensor input), and a single low-resolution axis is unpacked
    from get_lowres_axis's array, verbose passed through.
    """

    if force_separate_z is not None:
        do_separate_z = force_separate_z
        if force_separate_z:
            axis = get_lowres_axis(current_spacing)
        else:
            axis = None
    else:
        if get_do_separate_z(current_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(current_spacing)
        elif get_do_separate_z(new_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(new_spacing)
        else:
            do_separate_z = False
            axis = None

    if axis is not None:
        if len(axis) == 3:
            print("every axis has the same spacing, this should never happen, why is this code here?")
            do_separate_z = False
        elif len(axis) == 2:
            print(
                "this happens for spacings like (0.24, 1.25, 1.25) for example. In that case we do not want to resample separately in the out of plane axis"
            )
            do_separate_z = False
        else:
            # I changed pass to this. To prevent errors.
            axis = axis[0]

    if data is not None:
        assert data.ndim == 4, "data must be c x y z"

    data_reshaped = resample_data_or_seg(
        data,
        new_shape,
        is_seg,
        axis,
        order,
        do_separate_z,
        order_z=order_z,
        verbose=verbose,
    )
    return data_reshaped


def resample_logits_to_labels(
    logits: np.ndarray,
    new_shape: Union[Tuple[int, ...], List[int], np.ndarray],
    current_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
    new_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
) -> np.ndarray:
    """argmax over classes of nnU-Net's resampling_fn_probabilities applied to (C, z, y, x)
    logits (order 1, order 0 along z for anisotropic spacing), as a uint8 label map of
    new_shape. Same result as resampling all channels and taking argmax, with far less memory:

    - one channel at a time: resample_data_or_seg resamples every channel independently anyway,
      but converts the whole input to float64 first (2x the logits for 14 classes)
    - running argmax: only the best value so far (float32) and its class (uint8) are kept at
      new_shape, never all resampled channels. Updating on a strict > keeps the lower class on
      ties, as np.argmax does."""
    best: np.ndarray | None = None
    labels = np.zeros(tuple(int(s) for s in new_shape), dtype=np.uint8)
    for c in range(logits.shape[0]):
        channel = resample_data_or_seg_to_shape(
            logits[c : c + 1],
            new_shape,
            current_spacing,
            new_spacing,
            is_seg=False,
            order=1,
            order_z=0,
            force_separate_z=None,
        )[0]
        if best is None:
            # A copy: without resampling, channel is a view of the logits.
            best = np.array(channel, dtype=np.float32, copy=True)
            continue
        better = channel > best
        labels[better] = c
        np.copyto(best, channel, where=better)
        del channel, better
    return labels


def resize_segmentation(segmentation, new_shape, order=3):
    """
    Resizes a segmentation map. Supports all orders (see skimage documentation). Will transform segmentation map to one
    hot encoding which is resized and transformed back to a segmentation map.
    This prevents interpolation artifacts ([0, 0, 2] -> [0, 1, 2])

    Copied from batchgenerators.augmentations.utils.resize_segmentation (batchgenerators 0.25.3),
    which nnunetv2's resampling uses.
    :param segmentation:
    :param new_shape:
    :param order:
    :return:
    """
    tpe = segmentation.dtype
    assert len(segmentation.shape) == len(new_shape), "new shape must have same dimensionality as segmentation"
    if order == 0:
        return resize(
            segmentation.astype(float),
            new_shape,
            order,
            mode="edge",
            clip=True,
            anti_aliasing=False,
        ).astype(tpe)
    else:
        reshaped = np.zeros(new_shape, dtype=segmentation.dtype)

        unique_labels = np.sort(pd.unique(segmentation.ravel()))
        for i, c in enumerate(unique_labels):
            mask = segmentation == c
            reshaped_multihot = resize(
                mask.astype(float),
                new_shape,
                order,
                mode="edge",
                clip=True,
                anti_aliasing=False,
            )
            reshaped[reshaped_multihot >= 0.5] = c
        return reshaped


def resample_data_or_seg(
    data: np.ndarray,
    new_shape: Union[Tuple[int, ...], List[int], np.ndarray],
    is_seg: bool = False,
    axis: Union[None, int] = None,
    order: int = 3,
    do_separate_z: bool = False,
    order_z: int = 0,
    dtype_out=None,
    verbose=False,
):
    """
    separate_z=True will resample with order 0 along z

    Adapted from nnunetv2.preprocessing.resampling.default_resampling.resample_data_or_seg.
    Changes: optional verbose timing output.
    :param data:
    :param new_shape:
    :param is_seg:
    :param axis:
    :param order:
    :param do_separate_z:
    :param order_z: only applies if do_separate_z is True
    :return:
    """
    tic = time.time()
    assert data.ndim == 4, "data must be (c, x, y, z)"
    assert len(new_shape) == data.ndim - 1, f"new_shape: {new_shape}, data.ndim: {data.ndim}"

    if is_seg:
        resize_fn = resize_segmentation
        kwargs: dict[str, Any] = {}
    else:
        resize_fn = resize
        kwargs = {"mode": "edge", "anti_aliasing": False}
    shape = np.array(data[0].shape)
    new_shape = np.array(new_shape)
    old_shape = data.shape
    dtype_out = dtype_out or data.dtype

    reshaped_final = np.zeros((data.shape[0], *new_shape), dtype=dtype_out)
    # print("resample_data_or_seg -> shape: ", shape)
    # print("resample_data_or_seg -> new_shape: ", new_shape)
    if np.any(shape != new_shape):
        data = data.astype(np.float32, copy=False)
        if do_separate_z:
            # print("separate z, order in z is", order_z, "order inplane is", order)
            assert axis is not None, "If do_separate_z, we need to know what axis is anisotropic"
            if axis == 0:
                new_shape_2d = new_shape[1:]
            elif axis == 1:
                new_shape_2d = new_shape[[0, 2]]
            else:
                new_shape_2d = new_shape[:-1]

            for c in range(data.shape[0]):
                tmp = deepcopy(new_shape)
                tmp[axis] = shape[axis]
                reshaped_here = np.zeros(tmp)
                for slice_id in range(shape[axis]):
                    if axis == 0:
                        reshaped_here[slice_id] = resize_fn(data[c, slice_id], new_shape_2d, order, **kwargs)
                    elif axis == 1:
                        reshaped_here[:, slice_id] = resize_fn(data[c, :, slice_id], new_shape_2d, order, **kwargs)
                    else:
                        reshaped_here[:, :, slice_id] = resize_fn(
                            data[c, :, :, slice_id], new_shape_2d, order, **kwargs
                        )
                if shape[axis] != new_shape[axis]:
                    # The following few lines are blatantly copied and modified from sklearn's resize()
                    rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
                    orig_rows, orig_cols, orig_dim = reshaped_here.shape

                    # align_corners=False
                    row_scale = float(orig_rows) / rows
                    col_scale = float(orig_cols) / cols
                    dim_scale = float(orig_dim) / dim

                    map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
                    map_rows = row_scale * (map_rows + 0.5) - 0.5
                    map_cols = col_scale * (map_cols + 0.5) - 0.5
                    map_dims = dim_scale * (map_dims + 0.5) - 0.5

                    coord_map = np.array([map_rows, map_cols, map_dims])
                    if not is_seg or order_z == 0:
                        reshaped_final[c] = map_coordinates(reshaped_here, coord_map, order=order_z, mode="nearest")[
                            None
                        ]
                    else:
                        unique_labels = np.sort(pd.unique(reshaped_here.ravel()))  # np.unique(reshaped_data)
                        for i, cl in enumerate(unique_labels):
                            reshaped_final[c][
                                np.round(
                                    map_coordinates(
                                        (reshaped_here == cl).astype(float),
                                        coord_map,
                                        order=order_z,
                                        mode="nearest",
                                    )
                                )
                                > 0.5
                            ] = cl
                else:
                    reshaped_final[c] = reshaped_here
        else:
            # print("no separate z, order", order)
            for c in range(data.shape[0]):
                reshaped_final[c] = resize_fn(data[c], new_shape, order, **kwargs)
        if verbose:
            print(
                f"[INFO] resample_data_or_seg with reshaped final took: {time.time() - tic},"
                f" reshaped_final: {reshaped_final.shape} from old_shape: {old_shape}"
            )
        return reshaped_final
    else:
        # print("no resampling necessary")
        if verbose:
            print(
                f"[INFO] resample_data_or_seg with no resampling took: {time.time() - tic}," f" and data: {data.shape}"
            )
        return data


def resolve_model_path(model_file: str | os.PathLike | None, seg_type: Literal["ivus", "ct_seg"]) -> Path:
    """segmentation.model_file, or the bundled model when it is empty. Configs written before
    the switch to ONNX still point at an nnU-Net model folder (or a Keras .h5). Those can no
    longer be run, so they fall back to the bundled model as well."""
    if not model_file:
        if seg_type == "ivus":
            return DEFAULT_IVUS_MODEL_PATH
        elif seg_type == "ct_seg":
            return DEFAULT_CT_SEG_MODEL_PATH
    path = Path(model_file)
    if path.suffix.lower() != ".onnx":
        logger.warning(f"segmentation.model_file={str(model_file)!r} is not an .onnx model, using the bundled model")
        if seg_type == "ivus":
            return DEFAULT_IVUS_MODEL_PATH
        elif seg_type == "ct_seg":
            return DEFAULT_CT_SEG_MODEL_PATH
    return path


def nonzero_bbox(volume: np.ndarray) -> tuple[slice, ...]:
    """Half-open bounding box of the non-zero region: the box nnU-Net's crop_to_nonzero
    (nnunetv2.preprocessing.cropping.cropping, using acvl_utils' get_bbox_from_mask) crops to. nnU-Net hole-fills the mask in 3D first, but that cannot change the box: a zero voxel
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


_OPPOSITE_DIRECTION = {"L": "R", "R": "L", "A": "P", "P": "A", "S": "I", "I": "S"}


def reorient_array(array: np.ndarray, source: str, target: str) -> tuple[np.ndarray, list[int]]:
    """A (z, y, x) array of an image in orientation `source` (SimpleITK/DICOM code such as
    'LAS', letters in x, y, z order), as SimpleITK would give it after
    sitk.DICOMOrient(image, target): only axis permutations and flips, so voxel values are
    untouched. Also returns the axis permutation, to reorder a (dz, dy, dx) spacing the same
    way: [spacing[p] for p in perm]. A view when no copy is needed."""
    source, target = source.upper(), target.upper()
    perm, flips = [], []
    for k in range(3):  # numpy axis k is SimpleITK axis 2 - k
        letter = target[2 - k]
        j = next(
            (i for i, s in enumerate(source) if s in (letter, _OPPOSITE_DIRECTION[letter])),
            None,
        )
        if j is None:
            raise ValueError(f"Cannot reorient {source!r} to {target!r}")
        perm.append(2 - j)
        flips.append(source[j] != letter)
    out = np.transpose(array, perm)
    axes = tuple(k for k, flip in enumerate(flips) if flip)
    return (np.flip(out, axes) if axes else out), perm


def zscore_stats(volume: np.ndarray, chunk: int = 32) -> tuple[np.float32, float]:
    """Mean and std of the whole volume (float64 accumulation), without a float copy of it."""
    if volume.dtype.kind in "ub" and volume.dtype.itemsize <= 2:
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


def compute_steps(image_size: tuple[int, ...], tile_size: tuple[int, ...], tile_step_size: float) -> list[list[int]]:
    """nnU-Net's compute_steps_for_sliding_window, condensed from
    nnunetv2.inference.sliding_window_prediction.compute_steps_for_sliding_window."""
    steps = []
    for size, tile in zip(image_size, tile_size):
        num_steps = int(np.ceil((size - tile) / (tile * tile_step_size))) + 1
        actual = (size - tile) / (num_steps - 1) if num_steps > 1 else 0
        steps.append([int(np.round(actual * i)) for i in range(num_steps)])
    return steps


def compute_gaussian(
    tile_size: tuple[int, ...],
    sigma_scale: float = 1.0 / 8,
    value_scaling_factor: float = 10,
):
    """nnU-Net's Gaussian importance map, adapted from
    nnunetv2.inference.sliding_window_prediction.compute_gaussian: numpy float32 instead of a
    torch tensor, and value_scaling_factor defaults to 10, the value nnUNetPredictor passes."""
    tmp = np.zeros(tile_size)
    tmp[tuple(i // 2 for i in tile_size)] = 1
    gaussian = gaussian_filter(tmp, [i * sigma_scale for i in tile_size], 0, mode="constant", cval=0)
    gaussian = (gaussian / gaussian.max() * value_scaling_factor).astype(np.float32)
    gaussian[gaussian == 0] = gaussian[gaussian != 0].min()
    return gaussian


def _import_onnxruntime():
    try:
        import onnxruntime  # noqa: PLC0415 - optional dependency, imported lazily
    except ImportError as e:
        raise SegmentationUnavailable(
            'Automatic segmentation needs the optional "onnxruntime" package, which is not installed.\n\n'
            'From source: run the install script again (it installs the "segmentation" extra), or\n'
            "    uv sync --extra segmentation"
        ) from e
    return onnxruntime


def fp16_model_path(model_path: str | os.PathLike) -> Path:
    """The float16 copy that belongs to a model: <name>_fp16.onnx in the same folder."""
    path = Path(model_path)
    return path.with_name(f"{path.stem}_fp16{path.suffix}")


def _median_run_time(session, x: np.ndarray, runs: int = 3) -> float:
    """Median seconds per session.run on x (the session is already warmed up)."""
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run(None, {"image": x})
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def ct_normalization(image: np.ndarray, mean_intensity, std_intensity, percentile_00_5, percentile_99_5) -> np.ndarray:
    """Clip to the dataset's 0.5/99.5 percentiles, then z-score with the dataset's mean and std.
    Adapted from nnunetv2.preprocessing.normalization.default_normalization_schemes
    .CTNormalization.run, with the intensity properties passed in instead of read from the plans."""
    lower_bound = percentile_00_5
    upper_bound = percentile_99_5
    image = np.clip(image, lower_bound, upper_bound)
    image = (image - mean_intensity) / max(std_intensity, 1e-8)
    return image
