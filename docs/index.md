---
icon: lucide/scan-search
hide:
  - toc
---

# HiRes

Identify through-hole resistor values in the browser using the [*HiRes*](https://arxiv.org/abs/2606.30179) algorithm.

See [how it works](how-it-works.md), or the [repo](https://github.com/vesuvisian/hires-rs) for the CLI tool and GStreamer plugin.

This is a work in progress and still needs further refinement.

## How to use

- Wait until the status line says **Ready**. The first run downloads the model weights (tens of MB) and may stall while shaders compile.
- **Upload** or drop a photo of a resistor.
- Optionally enable **Band overlay** to blend inferred color-code classes onto each detection.
- Raise **Conf** if you see false positive bounding box detections. The default matches the CLI (`0.01`).

<div class="hires-frame-wrap" markdown="0">
  <iframe
    class="hires-frame"
    src="app/"
    title="HiRes"
    allow="fullscreen"
  ></iframe>
</div>
