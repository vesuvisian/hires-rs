"""Dump PyTorch forward tensors for native Burn parity checks.

Writes float32 .npy files that tools/parity (or numpy) can compare.

    python scripts/parity_forward.py --det-pt weights/detection/best.pt \\
        --seg-pt weights/segmentation/efficientnet-b2_best.pt \\
        --out /tmp/hires-pt
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch


def dump_detection(pt: str, out_dir: str, seed: int) -> None:
    from ultralytics import YOLO

    rng = np.random.default_rng(seed)
    x = rng.random((1, 3, 640, 640), dtype=np.float32)
    model = YOLO(pt).model.float().eval()
    with torch.no_grad():
        t = torch.from_numpy(x)
        y = model(t)
        if isinstance(y, (tuple, list)):
            y = y[0]
        y = y.cpu().numpy()
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "det_input.npy"), x)
    np.save(os.path.join(out_dir, "det_output.npy"), y)
    print(f"detection out {y.shape} max={y.max():.6g}  → {out_dir}")


def dump_segmentation(pt: str, out_dir: str, seed: int, encoder: str) -> None:
    import sys

    hires = os.environ.get("HIRES_ROOT")
    if hires:
        sys.path.insert(0, os.path.join(hires, "segmentation"))
    from train import build_model  # type: ignore

    rng = np.random.default_rng(seed)
    x = rng.standard_normal((1, 3, 64, 64), dtype=np.float32)
    model = build_model(encoder).float().eval()
    state = torch.load(pt, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    with torch.no_grad():
        y = model(torch.from_numpy(x)).cpu().numpy()
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "seg_input.npy"), x)
    np.save(os.path.join(out_dir, "seg_output.npy"), y)
    print(f"segmentation out {y.shape} max={y.max():.6g}  → {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--det-pt", default=None)
    p.add_argument("--seg-pt", default=None)
    p.add_argument("--out", default="/tmp/hires-pt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--encoder", default="efficientnet-b2")
    args = p.parse_args()
    if args.det_pt:
        dump_detection(args.det_pt, args.out, args.seed)
    if args.seg_pt:
        dump_segmentation(args.seg_pt, args.out, args.seed, args.encoder)


if __name__ == "__main__":
    main()
