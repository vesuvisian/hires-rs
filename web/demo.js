const MAX_SIDE = 1280;
const BAND_ALPHA = 0.55;
const DET_W = "./weights/detection/best.bpk";
const SEG_W = "./weights/segmentation/efficientnet-b2_best.bpk";
const BITMAP_OPTS = { imageOrientation: "none", colorSpaceConversion: "none" };

const statusEl = document.getElementById("status");
const metaEl = document.getElementById("meta");
const hintEl = document.getElementById("hint");
const canvas = document.getElementById("view");
const ctx =
  canvas.getContext("2d", { colorSpace: "srgb", willReadFrequently: true }) ||
  canvas.getContext("2d");
const fileEl = document.getElementById("file");
const dropEl = document.getElementById("drop");
const bandsEl = document.getElementById("bands");
const confEl = document.getElementById("conf");
const confVal = document.getElementById("conf-val");

let app = null;
let vis = [];
let busy = false;
let lastBitmap = null;
let lastRgba = null;
let lastNativeW = 0;
let lastNativeH = 0;
let lastDets = null;
let useWasmImage = false;
let backend = "";

function setStatus(msg) {
  statusEl.textContent = msg;
}

function conf() {
  // Integer milliconf (1 → 0.001) so range inputs cannot snap to 0.
  const milli = Math.max(1, Math.round(Number(confEl.value) || 1));
  return milli / 1000;
}

confEl.addEventListener("input", () => {
  confVal.textContent = conf().toFixed(3);
  lastDets = null;
});

async function fetchProgress(url, label) {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`${label}: HTTP ${res.status}`);
  }
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.byteLength;
    if (total) {
      setStatus(`Fetching ${label}… ${Math.round((100 * received) / total)}%`);
    } else {
      setStatus(`Fetching ${label}… ${(received / 1e6).toFixed(1)} MB`);
    }
  }
  const out = new Uint8Array(received);
  let offset = 0;
  for (const c of chunks) {
    out.set(c, offset);
    offset += c.byteLength;
  }
  return out;
}

async function loadModule() {
  const useGpu = Boolean(navigator.gpu);
  const pkg = useGpu ? "./pkg-wgpu/hires_web.js" : "./pkg-flex/hires_web.js";
  setStatus(`Loading ${useGpu ? "WebGPU" : "Flex CPU"} wasm…`);
  const mod = await import(pkg);
  await mod.default();
  vis = mod.visColors();
  return mod;
}

async function boot() {
  try {
    const mod = await loadModule();
    setStatus("Initializing device…");
    app = await mod.HiresApp.init();
    backend = app.backendName();
    const det = await fetchProgress(DET_W, "detection weights");
    const seg = await fetchProgress(SEG_W, "segmentation weights");
    setStatus("Loading models…");
    await app.loadWeights(det, seg);
    setStatus(`Ready (${backend}).`);
    metaEl.textContent =
      backend === "flex" ? "Flex CPU fallback — inference may take tens of seconds." : "";
  } catch (err) {
    console.error(err);
    setStatus(`Error: ${err.message || err}`);
  }
}

function hideHint() {
  hintEl.classList.add("hidden");
}

function viewScale() {
  return {
    sx: lastNativeW ? canvas.width / lastNativeW : 1,
    sy: lastNativeH ? canvas.height / lastNativeH : 1,
  };
}

function drawImageToFit(source) {
  const w = source.width;
  const h = source.height;
  if (!w || !h) return;
  const scale = Math.min(1, MAX_SIDE / Math.max(w, h));
  const dw = Math.max(1, Math.round(w * scale));
  const dh = Math.max(1, Math.round(h * scale));
  canvas.width = dw;
  canvas.height = dh;
  ctx.drawImage(source, 0, 0, dw, dh);
}

function rgbaFromBitmap(bitmap) {
  const c = document.createElement("canvas");
  c.width = bitmap.width;
  c.height = bitmap.height;
  const cctx =
    c.getContext("2d", { colorSpace: "srgb", willReadFrequently: true }) ||
    c.getContext("2d");
  cctx.drawImage(bitmap, 0, 0);
  return new Uint8Array(cctx.getImageData(0, 0, c.width, c.height).data);
}

async function bitmapFromFile(file) {
  try {
    return await createImageBitmap(file, BITMAP_OPTS);
  } catch {
    return await createImageBitmap(file);
  }
}

function detsHaveMasks(dets) {
  return Array.isArray(dets) && dets.some((d) => d.mask);
}

function blendMask(det, sx, sy) {
  if (!det.mask || !vis.length) return;
  const { mask, mask_w: mw, mask_h: mh, crop_x: ox, crop_y: oy } = det;
  const img = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const data = img.data;
  const inv = 1 - BAND_ALPHA;
  const x0 = Math.max(0, Math.floor(ox * sx));
  const y0 = Math.max(0, Math.floor(oy * sy));
  const x1 = Math.min(canvas.width, Math.ceil((ox + mw) * sx));
  const y1 = Math.min(canvas.height, Math.ceil((oy + mh) * sy));
  for (let py = y0; py < y1; py++) {
    const my = Math.min(mh - 1, Math.max(0, Math.floor(py / sy - oy)));
    for (let px = x0; px < x1; px++) {
      const mx = Math.min(mw - 1, Math.max(0, Math.floor(px / sx - ox)));
      const cls = mask[my * mw + mx];
      if (!cls) continue;
      const c = vis[Math.min(cls, vis.length - 1)];
      const i = (py * canvas.width + px) * 4;
      data[i] = BAND_ALPHA * c[0] + inv * data[i];
      data[i + 1] = BAND_ALPHA * c[1] + inv * data[i + 1];
      data[i + 2] = BAND_ALPHA * c[2] + inv * data[i + 2];
    }
  }
  ctx.putImageData(img, 0, 0);
}

function drawDets(dets) {
  const { sx, sy } = viewScale();
  if (bandsEl.checked) {
    for (const det of dets) blendMask(det, sx, sy);
  }
  ctx.save();
  ctx.strokeStyle = "rgb(0, 210, 0)";
  ctx.lineWidth = 2;
  ctx.font = "14px sans-serif";
  for (const det of dets) {
    const x1 = det.x1 * sx;
    const y1 = det.y1 * sy;
    const w = Math.max(1, (det.x2 - det.x1) * sx);
    const h = Math.max(1, (det.y2 - det.y1) * sy);
    ctx.strokeRect(x1, y1, w, h);
    const label = det.tolerance
      ? `${det.value}  ${det.tolerance}`
      : det.value;
    const pad = 4;
    const tw = ctx.measureText(label).width;
    const x = x1 + 4;
    const y = Math.max(16, y1);
    ctx.fillStyle = "rgba(0, 40, 0, 0.7)";
    ctx.fillRect(x, y - 14 - pad, tw + pad * 2, 16 + pad);
    ctx.fillStyle = "rgb(0, 220, 80)";
    ctx.fillText(label, x + pad, y - 4);
  }
  ctx.restore();
}

function redrawDets(dets) {
  if (lastBitmap) {
    drawImageToFit(lastBitmap);
  }
  drawDets(dets);
}

function canInfer() {
  return useWasmImage || lastRgba;
}

async function runOnCanvas() {
  if (!app || busy || !canInfer()) return;
  if (lastDets && (!bandsEl.checked || detsHaveMasks(lastDets))) {
    redrawDets(lastDets);
    return;
  }
  busy = true;
  const t0 = performance.now();
  try {
    const dets = useWasmImage
      ? await app.infer(conf(), bandsEl.checked)
      : await app.inferRgba(
          lastNativeW,
          lastNativeH,
          lastRgba,
          conf(),
          bandsEl.checked,
        );
    lastDets = dets;
    redrawDets(dets);
    const ms = Math.round(performance.now() - t0);
    const n = dets.length;
    const scores = Array.from(dets)
      .map((d) => {
        const w = Math.round(d.x2 - d.x1);
        const h = Math.round(d.y2 - d.y1);
        return `${Number(d.det_conf).toFixed(4)} ${w}×${h}`;
      })
      .join(", ");
    metaEl.textContent = `${n} detection${n === 1 ? "" : "s"} · conf≥${conf().toFixed(3)} · det=[${scores || "—"}] · ${ms} ms · ${backend}`;
  } catch (err) {
    console.error(err);
    setStatus(`Infer error: ${err.message || err}`);
  } finally {
    busy = false;
  }
}

async function ingestFile(file) {
  if (!app) return;
  hideHint();
  lastDets = null;
  lastRgba = null;
  useWasmImage = false;

  const bytes = new Uint8Array(await file.arrayBuffer());
  try {
    const dim = app.loadImage(bytes);
    lastNativeW = dim.width;
    lastNativeH = dim.height;
    useWasmImage = true;
    const rgba = app.imageRgba();
    const imageData = new ImageData(
      new Uint8ClampedArray(rgba),
      lastNativeW,
      lastNativeH,
    );
    lastBitmap = await createImageBitmap(imageData);
  } catch (err) {
    console.warn("wasm image decode failed, using bitmap pixels", err);
  }

  if (!useWasmImage) {
    lastBitmap = await bitmapFromFile(file);
    lastNativeW = lastBitmap.width;
    lastNativeH = lastBitmap.height;
    lastRgba = rgbaFromBitmap(lastBitmap);
  }
  drawImageToFit(lastBitmap);
  await runOnCanvas();
}

fileEl.addEventListener("change", async (ev) => {
  const file = ev.target.files?.[0];
  if (file) await ingestFile(file);
});

["dragenter", "dragover"].forEach((name) => {
  dropEl.addEventListener(name, (ev) => {
    ev.preventDefault();
    dropEl.classList.add("drag");
  });
});
["dragleave", "drop"].forEach((name) => {
  dropEl.addEventListener(name, (ev) => {
    ev.preventDefault();
    dropEl.classList.remove("drag");
  });
});
dropEl.addEventListener("drop", async (ev) => {
  const file = ev.dataTransfer?.files?.[0];
  if (file) await ingestFile(file);
});

bandsEl.addEventListener("change", async () => {
  if (lastBitmap) await runOnCanvas();
});

boot();
