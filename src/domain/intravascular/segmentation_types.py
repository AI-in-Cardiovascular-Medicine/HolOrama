"""Settings of the automatic IVUS lumen segmentation (config.yaml's segmentation section) and
what the runtime reads from a model file (segmentation.onnx_inference)."""

from __future__ import annotations

from dataclasses import dataclass, fields

DEVICES = ('auto', 'cpu', 'cuda', 'directml')  # 'auto': the first GPU that passes a test inference, else the CPU
PRECISIONS = ('auto', 'fp32', 'fp16')  # 'auto': fp16 on a GPU when it is clearly faster there, else fp32
# 'auto' precision keeps fp32 unless fp16 takes at most this share of its time per network
# call: fp16 shifts a few border pixels, so it has to buy a real speedup.
FP16_MAX_TIME_RATIO = 0.9

# Frames the network finds no lumen in get a catheter-sized circle at the image centre.
FALLBACK_CONTOUR_RADIUS_MM = 0.5
FALLBACK_RESOLUTION_MM = 0.1  # mm per pixel, when the file does not state its resolution


@dataclass(frozen=True)
class SegmentationSettings:
    model_file: str = ''  # '' = the bundled model
    device: str = 'auto'  # one of DEVICES
    precision: str = 'auto'  # one of PRECISIONS, GPU only (the CPU always runs fp32)
    use_mirroring: bool = True  # nnU-Net's test-time mirroring, ~4x slower, noticeably better masks
    tile_step_size: float = 0.5  # sliding-window overlap as a fraction of the patch
    batch_size: int = 2  # tiles per network call (2 = one 512x512 frame)
    num_threads: int = 0  # CPU threads, 0 = all cores

    @classmethod
    def from_config(cls, config) -> SegmentationSettings:
        """The segmentation section of config.yaml. Missing or empty keys keep their default."""
        section = getattr(config, 'segmentation', None)
        values = {f.name: getattr(section, f.name, None) for f in fields(cls)}
        return cls(**{name: value for name, value in values.items() if value is not None})


@dataclass(frozen=True)
class ModelInfo:
    """Inference parameters stored as metadata in an exported model (export_onnx.py)."""

    patch_size: tuple[int, int]
    mirror_axes: tuple[int, ...]
    num_classes: int
    labels: dict
