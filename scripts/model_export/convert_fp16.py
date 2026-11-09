"""Make the float16 copy of an exported segmentation model, used on GPUs (segmentation.precision).

Usage:
    python convert_fp16.py <model.onnx> [<output.onnx>]

The output defaults to <model>_fp16.onnx next to the model, where onnx_inference.py looks for
it. Every operator runs in float16, Resize included (the converter's default keeps Resize in
float32, which adds two casts per upsampling and made the network slower). Input and output
stay float32, so the inference code around the network is unchanged. The model metadata
(patch size, mirror axes, ...) is copied.

Needs only `onnx` and `onnxconverter-common` (e.g. in the export environment), no torch.
Check the result on a real pullback with scripts/benchmark_segmentation.py --precision fp16 --save.
"""

import argparse
import warnings
from pathlib import Path

import onnx
from onnxconverter_common import float16


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model', type=Path)
    parser.add_argument('output', type=Path, nargs='?')
    args = parser.parse_args()
    # Same name as onnx_inference.fp16_model_path (not imported: that pulls in the app's dependencies).
    output = args.output or args.model.with_name(f'{args.model.stem}_fp16{args.model.suffix}')

    model = onnx.load(str(args.model))
    meta = {p.key: p.value for p in model.metadata_props}
    with warnings.catch_warnings():
        # "float32 number ... will be truncated": a few tiny weights below float16's range, harmless
        warnings.simplefilter('ignore', UserWarning)
        converted = float16.convert_float_to_float16(model, keep_io_types=True, op_block_list=[])
    del converted.metadata_props[:]
    for key, value in meta.items():
        entry = converted.metadata_props.add()
        entry.key, entry.value = key, value
    meta_entry = converted.metadata_props.add()
    meta_entry.key, meta_entry.value = 'precision', 'fp16'
    onnx.checker.check_model(converted)
    onnx.save(converted, str(output))
    print(f'{output} ({output.stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
