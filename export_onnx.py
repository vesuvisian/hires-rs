"""
Export HiRes detection and segmentation models to ONNX.

Usage:
    # Export both models (defaults match shipped weights/)
    python export_onnx.py

    # Detection only
    python export_onnx.py --model detection

    # Segmentation only, custom encoder / size
    python export_onnx.py --model segmentation \\
        --seg_weights weights/segmentation/efficientnet-b2_best.pt \\
        --encoder efficientnet-b2 --seg_imgsz 512

    # Skip numerical check against ONNX Runtime
    python export_onnx.py --no-verify
"""

from __future__ import annotations

import argparse
import inspect
import os
import shutil
import sys

import numpy as np
import torch


def _repo_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def export_detection(
    weights: str,
    output: str,
    imgsz: int = 640,
    opset: int = 12,
    simplify: bool = True,
) -> str:
    """Export YOLOv8 detection weights via Ultralytics → ONNX."""
    from ultralytics import YOLO

    if not os.path.isfile(weights):
        raise FileNotFoundError(f'Detection weights not found: {weights}')

    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)

    model = YOLO(weights)
    exported = model.export(
        format='onnx',
        imgsz=imgsz,
        opset=opset,
        simplify=simplify,
        dynamic=False,
    )
    # Ultralytics writes next to the .pt; move/copy to the requested path.
    exported = str(exported)
    if os.path.abspath(exported) != os.path.abspath(output):
        shutil.move(exported, output)

    print(f'Detection ONNX → {output}')
    return output


def export_segmentation(
    weights: str,
    output: str,
    encoder: str = 'efficientnet-b2',
    imgsz: int = 512,
    opset: int = 18,
    batch_size: int = 1,
) -> str:
    """Export SMP UNet++ segmentation state_dict → ONNX."""
    seg_dir = os.path.join(_repo_root(), 'segmentation')
    if seg_dir not in sys.path:
        sys.path.insert(0, seg_dir)

    from train import build_model  # noqa: E402

    if not os.path.isfile(weights):
        raise FileNotFoundError(f'Segmentation weights not found: {weights}')

    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)

    device = torch.device('cpu')
    model = build_model(encoder).to(device)
    state = torch.load(weights, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(batch_size, 3, imgsz, imgsz, device=device)

    # Prefer the torch.export / dynamo exporter (clean fixed shapes for UNet++).
    # Fall back to TorchScript export on older torch or when onnxscript is missing.
    export_kwargs = dict(
        input_names=['input'],
        output_names=['logits'],
        opset_version=opset,
        do_constant_folding=True,
    )
    supports_dynamo = 'dynamo' in inspect.signature(torch.onnx.export).parameters

    if supports_dynamo:
        try:
            torch.onnx.export(
                model, dummy, output,
                dynamo=True,
                external_data=False,
                **export_kwargs,
            )
        except ModuleNotFoundError as exc:
            if 'onnxscript' not in str(exc):
                raise
            print('onnxscript not installed; falling back to TorchScript ONNX export')
            torch.onnx.export(model, dummy, output, dynamo=False, **export_kwargs)
    else:
        torch.onnx.export(model, dummy, output, **export_kwargs)

    print(f'Segmentation ONNX → {output}  (encoder={encoder}, imgsz={imgsz})')
    return output


def verify_segmentation_onnx(
    onnx_path: str,
    weights: str,
    encoder: str = 'efficientnet-b2',
    imgsz: int = 512,
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> None:
    """Compare PyTorch vs ONNX Runtime logits on a random input."""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            'onnxruntime is required for --verify. '
            'Install with: pip install onnxruntime'
        ) from exc

    seg_dir = os.path.join(_repo_root(), 'segmentation')
    if seg_dir not in sys.path:
        sys.path.insert(0, seg_dir)
    from train import build_model  # noqa: E402

    device = torch.device('cpu')
    model = build_model(encoder).to(device)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.eval()

    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((1, 3, imgsz, imgsz), dtype=np.float32)
    x = torch.from_numpy(x_np)

    with torch.no_grad():
        pt_out = model(x).numpy()

    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    ort_out = session.run(None, {'input': x_np})[0]

    if not np.allclose(pt_out, ort_out, atol=atol, rtol=rtol):
        max_diff = float(np.max(np.abs(pt_out - ort_out)))
        raise AssertionError(
            f'Segmentation ONNX mismatch (max abs diff={max_diff:.6g}, '
            f'atol={atol}, rtol={rtol})'
        )
    print(f'Segmentation verify OK  (max abs diff={float(np.max(np.abs(pt_out - ort_out))):.6g})')


def verify_detection_onnx(onnx_path: str) -> None:
    """Basic graph load check for the detection ONNX file."""
    try:
        import onnx
    except ImportError as exc:
        raise ImportError(
            'onnx is required for --verify. Install with: pip install onnx'
        ) from exc

    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print(f'Detection verify OK  ({onnx_path})')


def main(args: argparse.Namespace) -> None:
    models = {'detection', 'segmentation'} if args.model == 'all' else {args.model}

    if 'detection' in models:
        det_out = args.det_output or _default_onnx_path(args.det_weights)
        export_detection(
            weights=args.det_weights,
            output=det_out,
            imgsz=args.det_imgsz,
            opset=args.det_opset,
            simplify=not args.no_simplify,
        )
        if args.verify:
            verify_detection_onnx(det_out)

    if 'segmentation' in models:
        seg_out = args.seg_output or _default_onnx_path(args.seg_weights)
        export_segmentation(
            weights=args.seg_weights,
            output=seg_out,
            encoder=args.encoder,
            imgsz=args.seg_imgsz,
            opset=args.seg_opset,
        )
        if args.verify:
            verify_segmentation_onnx(
                onnx_path=seg_out,
                weights=args.seg_weights,
                encoder=args.encoder,
                imgsz=args.seg_imgsz,
            )

    print('Done.')


def _default_onnx_path(weights: str) -> str:
    """Place <stem>.onnx next to the source .pt checkpoint."""
    stem, _ = os.path.splitext(weights)
    return f'{stem}.onnx'


if __name__ == '__main__':
    root = _repo_root()
    parser = argparse.ArgumentParser(
        description='Export HiRes detection / segmentation models to ONNX'
    )
    parser.add_argument(
        '--model',
        choices=('all', 'detection', 'segmentation'),
        default='all',
        help='Which model(s) to export',
    )
    parser.add_argument(
        '--det_weights',
        default=os.path.join(root, 'weights', 'detection', 'best.pt'),
    )
    parser.add_argument(
        '--seg_weights',
        default=os.path.join(root, 'weights', 'segmentation', 'efficientnet-b2_best.pt'),
    )
    parser.add_argument(
        '--det_output',
        default=None,
        help='Override detection ONNX path '
             '(default: same folder as --det_weights, .pt → .onnx)',
    )
    parser.add_argument(
        '--seg_output',
        default=None,
        help='Override segmentation ONNX path '
             '(default: same folder as --seg_weights, .pt → .onnx)',
    )
    parser.add_argument(
        '--encoder',
        default='efficientnet-b2',
        help='SMP encoder used when the segmentation checkpoint was trained',
    )
    parser.add_argument('--det_imgsz', type=int, default=640)
    parser.add_argument('--seg_imgsz', type=int, default=512)
    parser.add_argument('--det_opset', type=int, default=12)
    parser.add_argument(
        '--seg_opset',
        type=int,
        default=18,
        help='ONNX opset for segmentation (18+ recommended with dynamo exporter)',
    )
    parser.add_argument(
        '--no-simplify',
        action='store_true',
        help='Disable ONNX graph simplify for the detection export',
    )
    parser.add_argument(
        '--verify',
        dest='verify',
        action='store_true',
        default=True,
        help='Validate exported graphs (default: on)',
    )
    parser.add_argument(
        '--no-verify',
        dest='verify',
        action='store_false',
        help='Skip ONNX / ONNX Runtime checks',
    )

    main(parser.parse_args())
