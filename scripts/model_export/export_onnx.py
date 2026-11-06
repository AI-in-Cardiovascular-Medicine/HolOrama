"""Export a trained nnU-Net (nnUZoo) 2D model folder to a single, self-describing ONNX file
that HolOrama's runtime (src/segmentation/onnx_inference.py) runs with onnxruntime alone.

This is a developer-only, one-off step: it is the ONLY place in the project that still needs
torch + nnUZoo. The app itself never imports either. See README.md next to this file for how
to set up the throwaway export environment.

Usage:
    python export_onnx.py <nnUNet model folder> <output .onnx> [--fold 0] [--checkpoint checkpoint_final.pth]

The network is rebuilt exactly as nnUNetPredictor builds it for inference (deep supervision
off, weights from the chosen fold/checkpoint), and the inference parameters the runtime needs
are stored as ONNX metadata so the .onnx file is all the app ever has to know about:
    patch_size            "512,448"
    mirror_axes           "0,1"      (empty = mirroring not allowed for this model)
    normalization         "zscore"   (the only scheme the runtime implements)
    num_classes           "2"
    labels                JSON of dataset.json's label map
    source                the model folder / fold / checkpoint it was exported from
"""

import argparse
import json
from pathlib import Path

import onnx
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

SUPPORTED_NORMALIZATION = {'ZScoreNormalization': 'zscore'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model_folder', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--checkpoint', default='checkpoint_final.pth')
    parser.add_argument('--opset', type=int, default=17)
    args = parser.parse_args()

    predictor = nnUNetPredictor(device=torch.device('cpu'), verbose=False, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(
        str(args.model_folder), use_folds=(args.fold,), checkpoint_name=args.checkpoint
    )
    network = predictor.network
    network.load_state_dict(predictor.list_of_parameters[0])
    network.eval()

    cm = predictor.configuration_manager
    pm = predictor.plans_manager
    if len(cm.patch_size) != 2:
        raise SystemExit(f'Only 2D configurations are supported, got patch size {cm.patch_size}')
    if list(pm.transpose_forward) != [0, 1, 2]:
        raise SystemExit(f'Non-identity transpose_forward {pm.transpose_forward} is not supported by the runtime')
    schemes = cm.normalization_schemes
    if len(schemes) != 1 or schemes[0] not in SUPPORTED_NORMALIZATION or cm.use_mask_for_norm[0]:
        raise SystemExit(f'Unsupported normalization {schemes} (use_mask_for_norm={cm.use_mask_for_norm})')
    if list(cm.spacing) != [1.0, 1.0]:
        raise SystemExit(f'Runtime assumes a target spacing of [1, 1] (no resampling), got {cm.spacing}')

    num_input_channels = 1
    dummy = torch.zeros((1, num_input_channels, *cm.patch_size), dtype=torch.float32)
    with torch.no_grad():
        out = network(dummy)
    num_classes = int(out.shape[1])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_kwargs = dict(
        input_names=['image'],
        output_names=['logits'],
        dynamic_axes={'image': {0: 'batch'}, 'logits': {0: 'batch'}},
        opset_version=args.opset,
        do_constant_folding=True,
    )
    if 'dynamo' in torch.onnx.export.__code__.co_varnames:  # torch >= 2.5 defaults may switch exporters
        export_kwargs['dynamo'] = False
    torch.onnx.export(network, dummy, str(args.output), **export_kwargs)

    mirror_axes = predictor.allowed_mirroring_axes or ()
    metadata = {
        'patch_size': ','.join(str(int(s)) for s in cm.patch_size),
        'mirror_axes': ','.join(str(int(a)) for a in mirror_axes),
        'normalization': SUPPORTED_NORMALIZATION[schemes[0]],
        'num_classes': str(num_classes),
        'labels': json.dumps(predictor.dataset_json.get('labels', {})),
        'source': f'{args.model_folder.name} fold_{args.fold}/{args.checkpoint}',
    }
    model = onnx.load(str(args.output))
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.checker.check_model(model)
    onnx.save(model, str(args.output))
    print(f'Wrote {args.output} ({args.output.stat().st_size / 1e6:.1f} MB) with metadata {metadata}')


if __name__ == '__main__':
    main()
