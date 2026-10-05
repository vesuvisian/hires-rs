"""Export a raw PyTorch state_dict for Burn PytorchStore.

Ultralytics `.pt` files are full pickled models. Burn cannot unpickle those
classes; this script writes a tensor-only dict.

Usage (from the HiRes repo, or pass explicit paths):

    python extract_state_dict.py
    python extract_state_dict.py --model detection --pt weights/detection/best.pt
"""

from __future__ import annotations

import argparse
import os

import torch


def extract_detection(src: str, dst: str) -> None:
    from ultralytics import YOLO

    yolo = YOLO(src)
    model = yolo.model
    # Prefer EMA weights when the checkpoint still carries them.
    ckpt = getattr(yolo, "ckpt", None) or {}
    ema = ckpt.get("ema") if isinstance(ckpt, dict) else None
    if ema is not None:
        try:
            state = ema.float().state_dict()
        except AttributeError:
            state = ema.state_dict() if hasattr(ema, "state_dict") else dict(ema)
    else:
        state = model.state_dict()
    # Drop non-tensor buffers Burn will not consume.
    state = {
        k: v.detach().cpu()
        for k, v in state.items()
        if torch.is_tensor(v) and not k.endswith("num_batches_tracked")
    }
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    torch.save(state, dst)
    print(f"detection state_dict → {dst}  ({len(state)} tensors)")


def extract_segmentation(src: str, dst: str) -> None:
    raw = torch.load(src, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict):
        raise TypeError(f"Expected a state_dict dict in {src}")
    state = {
        k: v.detach().cpu()
        for k, v in raw.items()
        if torch.is_tensor(v) and not k.endswith("num_batches_tracked")
    }
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    torch.save(state, dst)
    print(f"segmentation state_dict → {dst}  ({len(state)} tensors)")


def main() -> None:
    root = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(root)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("all", "detection", "segmentation"), default="all")
    parser.add_argument(
        "--det-pt",
        default=os.path.join(repo, "weights", "detection", "best.pt"),
    )
    parser.add_argument(
        "--seg-pt",
        default=os.path.join(repo, "weights", "segmentation", "efficientnet-b2_best.pt"),
    )
    parser.add_argument(
        "--det-out",
        default=os.path.join(repo, "weights", "detection", "best.state_dict.pt"),
    )
    parser.add_argument(
        "--seg-out",
        default=os.path.join(repo, "weights", "segmentation", "efficientnet-b2_best.state_dict.pt"),
    )
    args = parser.parse_args()
    models = {"detection", "segmentation"} if args.model == "all" else {args.model}
    if "detection" in models:
        extract_detection(args.det_pt, args.det_out)
    if "segmentation" in models:
        extract_segmentation(args.seg_pt, args.seg_out)


if __name__ == "__main__":
    main()
