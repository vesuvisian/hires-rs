# hires-rs

Rust port of the [HiRes](https://github.com/HiRes491/HiRes) end-to-end resistor value pipeline, using **Burn `0.22.0-pre.4`**. Method described in [*HiRes: A Hierarchical Cascaded Method for Resistor Value Identification*](https://arxiv.org/abs/2606.30179) (arXiv:2606.30179).

Cascade:

1. **Detection** — YOLOv8n (burn-onnx import) → boxes + NMS  
2. **Segmentation** — UNet++ / EfficientNet-B2 (burn-onnx import) → 13-class band mask  
3. **Decode** — PCA projection + color-code tables → ohms ± tolerance  

## Setup

Weights are vendored under `weights/`:

```text
weights/
├── detection/best.bpk
└── segmentation/efficientnet-b2_best.bpk
```

## Features (Burn backends)

Both the CLI (`hires-rs`) and the GStreamer plugin (`hires`) share the same feature flags. Default is `wgpu`. For a different backend, pass `--no-default-features --features <name>` so only one backend is enabled.

| Feature | Backend | Notes |
|---------|---------|--------|
| `wgpu` (default) | Burn wgpu | Cross-platform GPU. On macOS this typically uses Metal *through* wgpu. |
| `metal` | Burn CubeCL Metal | Native Metal (`Device::metal`). Prefer this on Apple Silicon for best FPS. |
| `cuda` | Burn CubeCL CUDA | Native CUDA (`Device::cuda(0)`). |
| `flex` | Burn flex | CPU / portable path when no GPU backend is wanted. |

Always build with `--release` for realtime use.

## Run

```bash
# GPU (wgpu) — default
cargo run --release -- path/to/image.jpg

# Directory of images
cargo run --release -- path/to/images/ --output-dir results/pipeline_output

# CPU (flex)
cargo run --release --no-default-features --features flex -- path/to/image.jpg

# Native Metal (macOS) / CUDA
cargo run --release --no-default-features --features metal -- path/to/image.jpg
cargo run --release --no-default-features --features cuda -- path/to/image.jpg
```

CLI flags mirror `pipeline.py`: `--det-weights`, `--seg-weights`, `--output-dir`, `--size`.

Outputs:

- `{stem}_resistor{N}.png` — 3-panel composite  
- `results.txt` — summary of decoded values  

## GStreamer plugin

The `hires` VideoFilter (RGB in/out) embeds the detection and segmentation burnpack weights in `libgsthires`, so no weight-path properties are needed. Each frame runs the lean overlay path and draws a box plus value/tolerance badge next to each detection.

```bash
# Default (wgpu). For native Metal on macOS:
#   cargo cbuild -p hires --release --no-default-features --features metal
cargo cbuild -p hires --release

# Point GStreamer at the built plugin (use your Cargo target-dir; default is target/)
export GST_PLUGIN_PATH="${CARGO_TARGET_DIR:-target}/release"

gst-inspect-1.0 hires

gst-launch-1.0 filesrc location=path/to/image.jpg ! decodebin ! videoconvert \
  ! video/x-raw,format=RGB ! hires ! videoconvert ! autovideosink
```

Element properties:

- `size` — segmentation canvas (default `512`)
- `band-overlay` — when `true`, blend inferred band colors onto each detection (default `false`)
- `confidence` — minimum YOLO detection score (default `0.005`). Raise toward `0.01` to cut false positives; lower toward `0.001` if true detections are missed.

```bash
gst-launch-1.0 ... ! hires confidence=0.01 band-overlay=true ! ...
```

## Export Original Models to ONNX

[`export_onnx.py`](export_onnx.py) exports the PyTorch YOLOv8n and UNet++ / EfficientNet-B2 checkpoints to ONNX. It needs the HiRes Python stack, plus ONNX (`onnx`, `onnxruntime`, `onnxslim`, `onnxscript`). The easiest approach is to copy the script into the HiRes repo and run it there (defaults already point at `weights/`).

```bash
python export_onnx.py                          # both models
python export_onnx.py --model detection        # or segmentation
python export_onnx.py --no-verify              # skip ONNX checks
```

Each `.onnx` is written next to its `.pt`. Detection: `(1, 3, 640, 640)` → `(1, 5, 8400)`. Segmentation: `(1, 3, 512, 512)` → logits `(1, 13, 512, 512)`. The ONNX graphs can then be imported into Burn.

## Notes

- HiRes (`pipeline.py`) gates Ultralytics detection at `conf=0.01`. The Burn/ONNX path scores lower (true positives often ≈0.006–0.009), so the library/CLI default is `0.001`. The GStreamer element defaults to `0.005` and exposes `confidence` so you can tune live false positives without rebuilding.
- Models are generated Burn modules from the HiRes ONNX exports; architectures are not hand-written.
