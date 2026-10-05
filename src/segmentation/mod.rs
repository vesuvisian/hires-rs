#[cfg(feature = "native-models")]
pub mod native;
#[cfg(feature = "onnx-models")]
pub mod onnx;

pub mod model {
    #[cfg(feature = "native-models")]
    pub use super::native::Model;
    #[cfg(feature = "onnx-models")]
    pub use super::onnx::Model;
}

#[cfg(feature = "onnx-models")]
pub const SEG_WEIGHTS_DEFAULT: &str = "weights/segmentation/efficientnet-b2_best.onnx.bpk";
#[cfg(feature = "native-models")]
pub const SEG_WEIGHTS_DEFAULT: &str = "weights/segmentation/efficientnet-b2_best.bpk";

use burn::prelude::*;
use burn::tensor::activation::softmax;
use image::{Rgb, RgbImage};

use crate::preprocess::resize_pad_batch_to_tensor;

pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

pub const VIS_COLORS: [[u8; 3]; 13] = [
    [0, 0, 0],
    [30, 30, 30],
    [0, 74, 173],
    [94, 56, 49],
    [235, 191, 124],
    [0, 191, 99],
    [88, 90, 89],
    [255, 145, 77],
    [148, 0, 211],
    [228, 50, 50],
    [192, 192, 192],
    [255, 255, 255],
    [244, 203, 36],
];

#[derive(Clone, Debug)]
pub struct PadInfo {
    pub pad_y: usize,
    pub pad_x: usize,
    pub new_h: usize,
    pub new_w: usize,
}

/// Controls segmentation cost for CLI vs live overlay.
#[derive(Clone, Copy, Debug)]
pub struct SegOptions {
    /// Run a second pass on the band bbox crop (higher quality, ~2× GPU).
    pub refine: bool,
    /// Build RGB class visualization (needed for CLI composites).
    pub build_vis: bool,
    /// Softmax mean confidence over band pixels (extra D2H).
    pub compute_conf: bool,
}

impl Default for SegOptions {
    fn default() -> Self {
        Self {
            refine: true,
            build_vis: true,
            compute_conf: true,
        }
    }
}

impl SegOptions {
    /// Live overlay: keep refine (needed for band decode) but skip vis/softmax.
    pub const fn overlay() -> Self {
        Self {
            refine: true,
            build_vis: false,
            compute_conf: false,
        }
    }
}

/// Kept for callers that still want an RGB padded canvas (e.g. debugging).
pub fn resize_pad(img: &RgbImage, size: usize) -> (RgbImage, PadInfo) {
    let (w, h) = img.dimensions();
    let scale = size as f32 / (h.max(w) as f32);
    let new_h = ((h as f32 * scale) as usize).max(1);
    let new_w = ((w as f32 * scale) as usize).max(1);
    let resized = crate::image_io::resize_rgb_bilinear(img, new_w as u32, new_h as u32);
    let pad_y = (size - new_h) / 2;
    let pad_x = (size - new_w) / 2;
    let mut canvas = RgbImage::from_pixel(size as u32, size as u32, Rgb([0, 0, 0]));
    image::imageops::replace(&mut canvas, &resized, pad_x as i64, pad_y as i64);
    (
        canvas,
        PadInfo {
            pad_y,
            pad_x,
            new_h,
            new_w,
        },
    )
}

fn nearest_resize_mask(
    mask: &[u8],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<u8> {
    let mut out = vec![0u8; dst_w * dst_h];
    for y in 0..dst_h {
        let sy = ((y as f32 * src_h as f32 / dst_h as f32).floor() as usize).min(src_h - 1);
        let src_row = sy * src_w;
        let dst_row = y * dst_w;
        for x in 0..dst_w {
            let sx = ((x as f32 * src_w as f32 / dst_w as f32).floor() as usize).min(src_w - 1);
            out[dst_row + x] = mask[src_row + sx];
        }
    }
    out
}

fn unpad_pred(pred_data: &[i32], pad: &PadInfo, s: usize) -> Vec<u8> {
    let mut unpad = Vec::with_capacity(pad.new_h * pad.new_w);
    for y in 0..pad.new_h {
        let row = (pad.pad_y + y) * s + pad.pad_x;
        for x in 0..pad.new_w {
            unpad.push(pred_data[row + x] as u8);
        }
    }
    unpad
}

fn sample_seg_conf(
    probs_data: &[f32],
    pred_data: &[i32],
    pad: &PadInfo,
    s: usize,
    batch_offset: usize,
) -> f32 {
    let plane = s * s;
    let base = batch_offset * 13 * plane;
    let pred_base = batch_offset * plane;
    let mut sum = 0.0f32;
    let mut count = 0usize;
    for y in 0..pad.new_h {
        for x in 0..pad.new_w {
            let idx = (pad.pad_y + y) * s + (pad.pad_x + x);
            let cls = pred_data[pred_base + idx] as usize;
            if cls > 0 {
                let mut best = f32::NEG_INFINITY;
                for c in 0..13 {
                    let v = probs_data[base + c * plane + idx];
                    if v > best {
                        best = v;
                    }
                }
                sum += best;
                count += 1;
            }
        }
    }
    if count > 0 { sum / count as f32 } else { 0.0 }
}

struct PassOut {
    mask: Vec<u8>,
    width: u32,
    height: u32,
    seg_conf: f32,
}

/// Batched inference pass. Returns one mask (original crop size) per input.
async fn run_pass_batch_async(
    model: &model::Model,
    imgs: &[&RgbImage],
    device: &Device,
    size: usize,
    compute_conf: bool,
) -> Vec<PassOut> {
    if imgs.is_empty() {
        return Vec::new();
    }
    let (input, pads) = resize_pad_batch_to_tensor(imgs, size, device);
    let logits = model.forward(input);
    let pred = logits.clone().argmax(1);
    let pred_data = pred
        .into_data_async()
        .await
        .expect("argmax i32")
        .try_to_vec::<i32>()
        .expect("argmax i32");
    let s = size;
    let plane = s * s;

    let probs_data = if compute_conf {
        let probs = softmax(logits, 1);
        Some(
            probs
                .into_data_async()
                .await
                .expect("probs f32")
                .try_to_vec::<f32>()
                .expect("probs f32"),
        )
    } else {
        drop(logits);
        None
    };

    imgs.iter()
        .zip(pads.iter())
        .enumerate()
        .map(|(b, (img, pad))| {
            let (w, h) = img.dimensions();
            let pred_slice = &pred_data[b * plane..(b + 1) * plane];
            let unpad = unpad_pred(pred_slice, pad, s);
            let seg_conf = probs_data
                .as_ref()
                .map(|p| sample_seg_conf(p, &pred_data, pad, s, b))
                .unwrap_or(0.0);
            let mask = nearest_resize_mask(&unpad, pad.new_w, pad.new_h, w as usize, h as usize);
            PassOut {
                mask,
                width: w,
                height: h,
                seg_conf,
            }
        })
        .collect()
}

pub struct SegResult {
    pub mask: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub vis: Option<RgbImage>,
    pub seg_conf: f32,
}

pub fn run_inference(
    model: &model::Model,
    img: &RgbImage,
    device: &Device,
    size: usize,
) -> SegResult {
    run_inference_with(model, img, device, size, SegOptions::default())
}

pub fn run_inference_with(
    model: &model::Model,
    img: &RgbImage,
    device: &Device,
    size: usize,
    opts: SegOptions,
) -> SegResult {
    run_inference_batch(model, &[img], device, size, opts)
        .into_iter()
        .next()
        .expect("single-image seg batch")
}

/// Run segmentation on many crops in one (or two, if refine) GPU forward(s).
pub fn run_inference_batch(
    model: &model::Model,
    imgs: &[&RgbImage],
    device: &Device,
    size: usize,
    opts: SegOptions,
) -> Vec<SegResult> {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (model, imgs, device, size, opts);
        panic!("run_inference_batch is sync; use run_inference_batch_async on wasm");
    }
    #[cfg(not(target_arch = "wasm32"))]
    pollster::block_on(run_inference_batch_async(model, imgs, device, size, opts))
}

pub async fn run_inference_batch_async(
    model: &model::Model,
    imgs: &[&RgbImage],
    device: &Device,
    size: usize,
    opts: SegOptions,
) -> Vec<SegResult> {
    if imgs.is_empty() {
        return Vec::new();
    }

    let mut passes = run_pass_batch_async(model, imgs, device, size, opts.compute_conf).await;

    if opts.refine {
        struct RefineJob {
            idx: usize,
            r0: usize,
            c0: usize,
            crop_w: usize,
            crop_h: usize,
            crop: RgbImage,
        }

        let mut jobs = Vec::new();
        for (i, pass) in passes.iter().enumerate() {
            let width = pass.width as usize;
            let height = pass.height as usize;
            let mask = &pass.mask;

            let mut rmin = None;
            let mut rmax = 0usize;
            let mut cmin = None;
            let mut cmax = 0usize;
            for y in 0..height {
                for x in 0..width {
                    if mask[y * width + x] > 0 {
                        if rmin.is_none() {
                            rmin = Some(y);
                        }
                        rmax = y;
                        cmin = Some(cmin.map_or(x, |c: usize| c.min(x)));
                        cmax = cmax.max(x);
                    }
                }
            }

            if let (Some(rmin), Some(cmin)) = (rmin, cmin) {
                let bh = (rmax - rmin).max(1);
                let bw = (cmax - cmin).max(1);
                let py = ((bh as f32 * 0.20) as usize).max(8);
                let px = ((bw as f32 * 0.20) as usize).max(8);
                let r0 = rmin.saturating_sub(py);
                let r1 = (rmax + py + 1).min(height);
                let c0 = cmin.saturating_sub(px);
                let c1 = (cmax + px + 1).min(width);
                let crop_h = r1 - r0;
                let crop_w = c1 - c0;
                if crop_h.min(crop_w) >= 20 {
                    let crop = image::imageops::crop_imm(
                        imgs[i],
                        c0 as u32,
                        r0 as u32,
                        crop_w as u32,
                        crop_h as u32,
                    )
                    .to_image();
                    jobs.push(RefineJob {
                        idx: i,
                        r0,
                        c0,
                        crop_w,
                        crop_h,
                        crop,
                    });
                }
            }
        }

        if !jobs.is_empty() {
            let refine_refs: Vec<&RgbImage> = jobs.iter().map(|j| &j.crop).collect();
            let refine_outs =
                run_pass_batch_async(model, &refine_refs, device, size, opts.compute_conf).await;
            for (job, out) in jobs.iter().zip(refine_outs) {
                let crop_mask =
                    if out.width as usize != job.crop_w || out.height as usize != job.crop_h {
                        nearest_resize_mask(
                            &out.mask,
                            out.width as usize,
                            out.height as usize,
                            job.crop_w,
                            job.crop_h,
                        )
                    } else {
                        out.mask
                    };
                let pass = &mut passes[job.idx];
                let width = pass.width as usize;
                let height = pass.height as usize;
                let first_fg = pass.mask.iter().filter(|&&c| c > 0).count();
                let mut full = vec![0u8; width * height];
                for y in 0..job.crop_h {
                    let src = y * job.crop_w;
                    let dst = (job.r0 + y) * width + job.c0;
                    full[dst..dst + job.crop_w].copy_from_slice(&crop_mask[src..src + job.crop_w]);
                }
                let refine_fg = full.iter().filter(|&&c| c > 0).count();
                // Refine occasionally collapses thin bands; keep the first pass
                // when it had substantially more band signal.
                if refine_fg * 2 >= first_fg {
                    pass.mask = full;
                    pass.seg_conf = out.seg_conf;
                }
            }
        }
    }

    passes
        .into_iter()
        .map(|pass| {
            let vis = if opts.build_vis {
                let w = pass.width;
                let h = pass.height;
                let width = w as usize;
                let height = h as usize;
                let mut vis = RgbImage::new(w, h);
                for y in 0..height {
                    for x in 0..width {
                        let cls = pass.mask[y * width + x] as usize;
                        let c = VIS_COLORS[cls.min(12)];
                        vis.put_pixel(x as u32, y as u32, Rgb(c));
                    }
                }
                Some(vis)
            } else {
                None
            };
            SegResult {
                mask: pass.mask,
                width: pass.width,
                height: pass.height,
                vis,
                seg_conf: pass.seg_conf,
            }
        })
        .collect()
}
