---
icon: lucide/waypoints
---

# How it works

This project is a Rust port of the [HiRes](https://github.com/HiRes491/HiRes) pipeline, using the **Burn** machine learning framework. The original algorithm is described in [*HiRes: A Hierarchical Cascaded Method for Resistor Value Identification*](https://arxiv.org/abs/2606.30179).

The method is a **cascade**, not a single network. A detector finds resistor bodies in the photo; a segmenter labels color bands on each crop; a geometric decoder turns that mask into ohms and tolerance. Later stages never see the full image — they only see what the previous stage kept.

```
photo → YOLOv8n boxes → crop (+10%) → UNet++ mask → PCA bands → color-code tables → Ω ± %
```

## Detection

YOLOv8n is a single-class body detector. The image is scaled so the long edge fits a 640 max side, bilinearly resized, then centered on a gray canvas (`114/255`):

- **Native** (`native-models`, default): Ultralytics **auto** letterbox — pad only until `H` and `W` are multiples of stride 32 (e.g. 1280×720 → 640×384). Anchor count follows the feature maps.
- **ONNX** (`onnx-models`, backup): pad to a fixed **640×640** square (export shape). Output length is `8400` anchors.

Aspect ratio is preserved so boxes map back with a uniform scale and pad. Candidates below the confidence gate are dropped on-device, then:

1. Convert xywh to xyxy and **undo letterbox** (subtract pad, divide by scale, clamp to the original image).
2. Keep at most 100 highest-scoring boxes.
3. **NMS** at IoU 0.5.

Each surviving box is expanded by **10%** on every side (clamped to the image) so bands near the body edge are not clipped. Crops smaller than 30 px on the short side are discarded.

Native matches HiRes auto letterbox and defaults to `conf=0.01`. ONNX square letterbox scores lower (≈0.006–0.009), so that backup path defaults to `0.001`. Raise **Conf** if you see false boxes (the GStreamer element exposes `confidence` for live video).

## Segmentation

Each crop is independently labeled by UNet++ with an EfficientNet-B2 encoder. Input is **resize-padded** to a 512×512 canvas (black pad, ImageNet mean/std), not letterboxed like YOLO. Logits are 13 channels; the class at each pixel is `argmax`.

| Id | Class |
| --- | --- |
| 0 | background |
| 1–12 | black, blue, brown, gold, green, grey, orange, violet, red, silver, white, yellow |

The prediction is unpadded and nearest-neighbor resized back to the crop.

A **refine pass** then zooms in: take the bounding box of all non-background pixels, pad it by 20% (at least 8 px), and run the same model on that tighter crop if it is at least 20 px. The refined mask is pasted back into a full-size canvas. If refine loses too much band area (less than half the first-pass foreground), the first pass is kept — thin gold/silver stripes sometimes collapse when zoomed.

The live demo still runs refine (decode needs a clean mask) but skips the RGB class visualization and softmax confidence that the CLI composites use.

## Band extraction

The mask is a 2-D label field, not an ordered list of stripes. Extraction recovers that order with a 1-D projection:

1. Collect every non-background pixel. Fewer than 20 pixels → no bands.
2. Run **PCA** on those `(x, y)` points. The principal eigenvector is the band axis (the long direction of the color-code); its sign is flipped so `x` is non-negative, which makes left-to-right and top-to-bottom consistent enough to sort.
3. Project each pixel onto that axis and histogram into **200 bins**.
4. In each bin, vote by class. Smooth each class independently with a 1-D Gaussian (`σ = 2.5`). Background is ignored.
5. Walk the sequence of dominant classes. A run becomes a band only if it spans enough bins and enough pixels (thresholds scale down on tiny live crops so thin stripes survive).

Each band stores class/color, pixel area, bounding box, and a centroid on the axis. Tiny blobs and `unknown` labels are dropped. Gold and silver are allowed to be much smaller than digit bands (they often are). If more than five bands remain, the five largest by area are kept.

## Decode

Colors along the axis still have an unknown **reading direction**. Real through-hole resistors put the tolerance stripe at one end, usually with a slightly larger gap.

Direction is chosen, in order, by:

- Gold/silver position (must sit at an end, not strictly interior; two metal bands must be the last two or the first two)
- For 5-band parts without a metal stripe: whether brown/red/green/blue/violet/grey sits at an end
- **Gap ratio**: if the first or last inter-band gap is ≥ 1.5× the median interior gap, that wide gap is the tolerance side
- Fallbacks: a black stripe cannot lead, and a much narrower first band is treated as the far end

Black on the outer edge is rejected (it is neither a valid leading digit nor a tolerance color). Extra leading gold/silver is stripped; a leftover pair of metal bands at the tail is trimmed to one.

Standard color-code arithmetic then applies:

| Bands | Value |
| --- | --- |
| 3 | `(d1×10 + d2) × multiplier`, tolerance assumed ±20% |
| 4 | `(d1×10 + d2) × multiplier`, last band is tolerance |
| 5 | `(d1×100 + d2×10 + d3) × multiplier`, last band is tolerance |
| >5 | treated as 5-band using the first five after ordering |

Digit, multiplier, and tolerance maps are IEC 60062 (black=0 … white=9; gold×0.1 / ±5%; silver×0.01 / ±10%; and so on). Invalid digit or tolerance colors are reported as errors rather than guessed.

If the decoded ohms are **not** within 1% of an **E24** mantissa and more than three bands were found, the decoder tries dropping the one or two smallest-area bands. Segmentation often invents a sliver; the E24 check pulls the reading back onto a real series value.

The overlay then shows `{value} ±{tolerance}%` next to each box. Optional **Band overlay** blends the 13-class colors onto the crop so you can see what the segmenter thought each stripe was.

## Backends

Burn backends are compile-time exclusive. The site ships two wasm packs:

- **wgpu** — used when `navigator.gpu` is present (Chrome, Edge, recent Firefox/Safari). The wasm crate enables `burn/webgpu` so CubeCL targets **browser WebGPU**.
- **flex** — CPU fallback; still images only, often tens of seconds per photo

Default CLI `wgpu` is not that stack. On macOS it still talks to the GPU through **wgpu’s Metal HAL**, but CubeCL emits **WGSL** (`wgpu<wgsl>`) unless you build `--features metal` (native CubeCL MSL). The demo’s WGSL is compiled by the **browser**, not by native wgpu/Naga. Letterbox, NMS, and the rest of the cascade are the same Rust; the kernels are not bit-identical.

On the ONNX backup path, objectness often sits just above the floor (**0.006–0.009**), so a small backend drift can drop the body box under `0.001` while weaker proposals remain. Native’s higher default gate (`0.01`) assumes auto letterbox; if a peak disappears on one GPU path, lowering **Conf** only adds noise. Flex CPU is a third numeric path.

Other CLI vs demo differences:

- The demo skips RGB class vis and softmax `seg_conf` (decode still uses the refined mask). That does not change boxes.
- Canvas `MAX_SIDE` (1280) is display-only when wasm decoded the file; inference uses that full-resolution RGB.
- If the `image` crate cannot decode the file, the demo falls back to `createImageBitmap` pixels (optional color management / EXIF). That **can** change the RGB the models see.

## Weights

Burnpacks are fetched at runtime (not embedded in the `.wasm`):

```text
weights/detection/best.bpk
weights/segmentation/efficientnet-b2_best.bpk
```

`bash web/stage.sh` stages those native packs by default. `HIRES_ONNX=1` stages the `*.onnx.bpk` backup packs instead (and the wasm build must use `--features onnx-models`; point `demo.js` at the `.onnx.bpk` paths).

Default builds use hand-written YOLOv8n + SMP UNet++ (`native-models`) loaded from plain `*.bpk` (from original `.pt`). `onnx-models` is an exclusive backup that uses generated graphs from the HiRes ONNX exports (`*.onnx.bpk`).
