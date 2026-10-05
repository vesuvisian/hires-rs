#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="$ROOT/docs/app"

mkdir -p "$APP/weights/detection" "$APP/weights/segmentation"
cp "$ROOT/web/index.html" "$ROOT/web/demo.js" "$ROOT/web/demo.css" "$APP/"

# Default: native HiRes-style packs. Set HIRES_ONNX=1 to stage the ONNX backup.
if [[ "${HIRES_ONNX:-}" == "1" ]]; then
  DET_SRC="$ROOT/weights/detection/best.onnx.bpk"
  SEG_SRC="$ROOT/weights/segmentation/efficientnet-b2_best.onnx.bpk"
  DET_DST="$APP/weights/detection/best.onnx.bpk"
  SEG_DST="$APP/weights/segmentation/efficientnet-b2_best.onnx.bpk"
else
  DET_SRC="$ROOT/weights/detection/best.bpk"
  SEG_SRC="$ROOT/weights/segmentation/efficientnet-b2_best.bpk"
  DET_DST="$APP/weights/detection/best.bpk"
  SEG_DST="$APP/weights/segmentation/efficientnet-b2_best.bpk"
fi

if [[ -f "$DET_SRC" && -f "$SEG_SRC" ]]; then
  cp "$DET_SRC" "$DET_DST"
  cp "$SEG_SRC" "$SEG_DST"
else
  echo "warning: weights/*.bpk not found; the demo cannot load models until they are copied to docs/app/weights/" >&2
fi
