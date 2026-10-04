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

Both the CLI (`hires-rs`) and the GStreamer plugin (`hires`) share the same backend flags. Default is `wgpu` plus `cli` (clap / walkdir for the binary). For a different backend, pass `--no-default-features --features <backend>,cli` when running the CLI so only one GPU/CPU backend is enabled. The wasm demo and plugin use `--no-default-features --features <backend>` (no `cli`).

| Feature | Backend | Notes |
|---------|---------|--------|
| `wgpu` (default) | Burn wgpu | Cross-platform GPU. On macOS this typically uses Metal *through* wgpu. |
| `metal` | Burn CubeCL Metal | Native Metal (`Device::metal`). Prefer this on Apple Silicon for best FPS. |
| `cuda` | Burn CubeCL CUDA | Native CUDA (`Device::cuda(0)`). |
| `flex` | Burn flex | CPU / portable path when no GPU backend is wanted. |
| `cli` | — | CLI only (`clap`, `walkdir`). On by default with `wgpu`; enable it whenever you `--no-default-features` the binary. |

Always build with `--release` for realtime use.

## Run

```bash
# GPU (wgpu) — default
cargo run --release -- path/to/image.jpg

# Directory of images
cargo run --release -- path/to/images/ --output-dir results/pipeline_output

# CPU (flex)
cargo run --release --no-default-features --features flex,cli -- path/to/image.jpg

# Native Metal (macOS) / CUDA
cargo run --release --no-default-features --features metal,cli -- path/to/image.jpg
cargo run --release --no-default-features --features cuda,cli -- path/to/image.jpg
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

## Browser demo

The docs site is [Zensical](https://zensical.org/). An iframe loads a wasm-bindgen app (`web/`) that runs `process_rgb_overlay` / `process_rgb_overlay_bands` in the browser.

- WebGPU (`burn/webgpu`) when `navigator.gpu` is present; otherwise Flex CPU. Default CLI `wgpu` on macOS is wgpu’s Metal HAL with CubeCL **WGSL**, not CubeCL MSL — different lowering than the browser, so YOLO scores (and boxes after NMS) can disagree on the same JPEG. See [How it works](docs/how-it-works.md#backends).
- Upload or drop a still image, optional band-color overlay
- Weights are fetched at runtime from `docs/app/weights/` (not `include_bytes!`)

### Local preview

You need the vendored `weights/*.bpk` files, a recent Rust toolchain, [wasm-pack](https://rustwasm.github.io/wasm-pack/), and [Zensical](https://zensical.org/docs/get-started/). Do not open `index.html` as `file://` (wasm requires http://localhost).

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-pack

# If `pip install zensical` is blocked (PEP 668), use a venv:
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install zensical

# From the repo root — release wasm is slow the first time
wasm-pack build web --release --target web --out-dir "$PWD/docs/app/pkg-wgpu" \
  -- --no-default-features --features wgpu
wasm-pack build web --release --target web --out-dir "$PWD/docs/app/pkg-flex" \
  -- --no-default-features --features flex
bash web/stage.sh            # copies web/index.html, demo.js/css, and weights into docs/app/

zensical build --clean
python3 web/preview.py       # http://127.0.0.1:8000/hires-rs/  (wasm MIME + Pages path)

# `zensical serve` is fine for editing docs, but its preview server often
# serves .wasm as the wrong MIME type. instantiateStreaming then falls
# back (slower). GitHub Pages sends the correct type.

```

`stage.sh` warns if `weights/` is missing; the iframe will load but inference will fail until those burnpacks are present. After changing `web/index.html`, `demo.js`, or `demo.css`, re-run `bash web/stage.sh` (no wasm rebuild). After changing `web/src` or `hires-rs`, re-run the `wasm-pack` commands.

For a faster iteration loop you can swap `--release` for `--dev` on `wasm-pack`; the GitHub Pages workflow always builds `--release`.

### GitHub Pages

The demo is meant to live at [https://vesuvisian.com/hires-rs/](https://vesuvisian.com/hires-rs/) (`zensical.toml` `site_url`). That path is the user-site custom domain (`vesuvisian.com`) plus this repo name — GitHub serves project Pages there automatically. Do **not** set a custom domain on *this* repository, or it would take over the apex instead of `/hires-rs/`.

Set this repo’s Pages source to **GitHub Actions**. The workflow builds both wasm packs, copies `weights/*.bpk` into `docs/app/`, then `zensical build`. Each `.bpk` must be under GitHub’s 100 MB file limit (or copied from a Release in CI).

