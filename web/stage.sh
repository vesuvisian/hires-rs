#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="$ROOT/docs/app"

mkdir -p "$APP/weights/detection" "$APP/weights/segmentation"
cp "$ROOT/web/index.html" "$ROOT/web/demo.js" "$ROOT/web/demo.css" "$APP/"

DET="$ROOT/weights/detection/best.bpk"
SEG="$ROOT/weights/segmentation/efficientnet-b2_best.bpk"
if [[ -f "$DET" && -f "$SEG" ]]; then
  cp "$DET" "$APP/weights/detection/"
  cp "$SEG" "$APP/weights/segmentation/"
else
  echo "warning: weights/*.bpk not found; the demo cannot load models until they are copied to docs/app/weights/" >&2
fi
