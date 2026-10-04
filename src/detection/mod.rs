pub mod model;

use burn::prelude::*;
use image::RgbImage;

use crate::preprocess::letterbox_to_tensor;

#[derive(Clone, Debug)]
pub struct Detection {
    pub x1: i32,
    pub y1: i32,
    pub x2: i32,
    pub y2: i32,
    pub conf: f32,
}

#[derive(Clone, Debug)]
pub struct LetterboxMeta {
    pub scale: f32,
    pub pad_x: f32,
    pub pad_y: f32,
    pub orig_w: u32,
    pub orig_h: u32,
}

pub const DET_SIZE: u32 = 640;
/// Default YOLO confidence gate.
///
/// ONNX-exported YOLOv8 scores run lower than Ultralytics `.pt` (HiRes uses
/// `conf=0.01` on `.pt`; ONNX/Burn true positives often land around 0.006–0.009),
/// so the library default stays low. Raise it (e.g. `0.005`–`0.01`) to cut
/// false positives on live video.
pub const DEFAULT_CONF_THRESH: f32 = 0.001;
const NMS_TOP_K: usize = 100;

/// Ultralytics-style letterbox to square canvas with gray (114) padding.
/// Prefer [`detect_resistors`] which fuses letterbox into the tensor path.
pub fn letterbox(img: &RgbImage, size: u32) -> (RgbImage, LetterboxMeta) {
    let (orig_w, orig_h) = img.dimensions();
    let scale = (size as f32 / orig_w as f32).min(size as f32 / orig_h as f32);
    let new_w = (orig_w as f32 * scale).round().max(1.0) as u32;
    let new_h = (orig_h as f32 * scale).round().max(1.0) as u32;
    let resized = crate::image_io::resize_rgb_bilinear(img, new_w, new_h);

    let pad_x = (size - new_w) as f32 / 2.0;
    let pad_y = (size - new_h) as f32 / 2.0;
    let mut canvas = image::RgbImage::from_pixel(size, size, image::Rgb([114, 114, 114]));
    let ox = pad_x.floor() as u32;
    let oy = pad_y.floor() as u32;
    image::imageops::replace(&mut canvas, &resized, ox as i64, oy as i64);

    (
        canvas,
        LetterboxMeta {
            scale,
            pad_x,
            pad_y,
            orig_w,
            orig_h,
        },
    )
}

/// Decode YOLO export `(1, 5, N)` → boxes in original image coordinates.
///
/// Filters on-device with a confidence mask before a small D2H transfer of survivors.
/// On wasm, use [`decode_yolo_async`] — WebGPU cannot read tensors synchronously.
pub fn decode_yolo(output: Tensor<3>, meta: &LetterboxMeta, conf_thresh: f32) -> Vec<Detection> {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (output, meta, conf_thresh);
        panic!("decode_yolo is sync; use decode_yolo_async on wasm");
    }
    #[cfg(not(target_arch = "wasm32"))]
    pollster::block_on(decode_yolo_async(output, meta, conf_thresh))
}

/// Async decode for WebGPU/wasm (no blocking D2H).
pub async fn decode_yolo_async(
    output: Tensor<3>,
    meta: &LetterboxMeta,
    conf_thresh: f32,
) -> Vec<Detection> {
    // [1, 5, N] → [5, N]
    let pred = output.squeeze_dim::<2>(0);
    // Slice keeps rank ([1, N]); squeeze so argwhere yields [K, 1], not [K, 2].
    let conf = pred.clone().slice(s![4..5, ..]).squeeze_dim::<1>(0); // [N]
    let boxes = pred.slice(s![0..4, ..]).swap_dims(0, 1); // [N, 4] cx,cy,w,h

    let idx2d = conf.clone().greater_elem(conf_thresh).argwhere_async().await; // [K, 1]
    let k = idx2d.dims()[0];
    if k == 0 {
        return Vec::new();
    }
    let indices = idx2d.squeeze_dim::<1>(1); // [K]

    let boxes_f = boxes.select(0, indices.clone()); // [K, 4]
    let conf_f = conf.select(0, indices); // [K]

    let box_vals = boxes_f
        .into_data_async()
        .await
        .expect("f32 yolo boxes")
        .try_to_vec::<f32>()
        .expect("f32 yolo boxes");
    let conf_vals = conf_f
        .into_data_async()
        .await
        .expect("f32 yolo conf")
        .try_to_vec::<f32>()
        .expect("f32 yolo conf");

    // Re-check on the host: wgpu/WebGPU `greater_elem` + `argwhere` can leak
    // almost every anchor, which then looks like det_conf=0 (many tiny boxes).
    detections_from_xywh(&box_vals, &conf_vals, k, meta, conf_thresh)
}

fn detections_from_xywh(
    box_vals: &[f32],
    conf_vals: &[f32],
    k: usize,
    meta: &LetterboxMeta,
    conf_thresh: f32,
) -> Vec<Detection> {
    let mut dets = Vec::with_capacity(k);
    for i in 0..k {
        let cx = box_vals[i * 4];
        let cy = box_vals[i * 4 + 1];
        let bw = box_vals[i * 4 + 2];
        let bh = box_vals[i * 4 + 3];
        let conf = conf_vals[i];
        if !(conf > conf_thresh) {
            continue;
        }
        // xywh in letterboxed space → xyxy
        let mut x1 = cx - bw / 2.0;
        let mut y1 = cy - bh / 2.0;
        let mut x2 = cx + bw / 2.0;
        let mut y2 = cy + bh / 2.0;

        // undo letterbox
        x1 = (x1 - meta.pad_x) / meta.scale;
        y1 = (y1 - meta.pad_y) / meta.scale;
        x2 = (x2 - meta.pad_x) / meta.scale;
        y2 = (y2 - meta.pad_y) / meta.scale;

        let x1 = x1.clamp(0.0, meta.orig_w as f32 - 1.0).round() as i32;
        let y1 = y1.clamp(0.0, meta.orig_h as f32 - 1.0).round() as i32;
        let x2 = x2.clamp(0.0, meta.orig_w as f32 - 1.0).round() as i32;
        let y2 = y2.clamp(0.0, meta.orig_h as f32 - 1.0).round() as i32;
        if x2 <= x1 || y2 <= y1 {
            continue;
        }
        dets.push(Detection {
            x1,
            y1,
            x2,
            y2,
            conf,
        });
    }
    dets
}

pub fn nms(dets: &[Detection], iou_thresh: f32) -> Vec<usize> {
    if dets.is_empty() {
        return Vec::new();
    }
    let mut order: Vec<usize> = (0..dets.len()).collect();
    order.sort_by(|&a, &b| {
        dets[b]
            .conf
            .partial_cmp(&dets[a].conf)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if order.len() > NMS_TOP_K {
        order.truncate(NMS_TOP_K);
    }

    let mut kept = Vec::with_capacity(order.len().min(32));
    let mut suppressed = vec![false; dets.len()];
    for &i in &order {
        if suppressed[i] {
            continue;
        }
        kept.push(i);
        let ai = ((dets[i].x2 - dets[i].x1).max(0) * (dets[i].y2 - dets[i].y1).max(0)) as f32;
        for &j in &order {
            if suppressed[j] || j == i {
                continue;
            }
            let xx1 = dets[i].x1.max(dets[j].x1);
            let yy1 = dets[i].y1.max(dets[j].y1);
            let xx2 = dets[i].x2.min(dets[j].x2);
            let yy2 = dets[i].y2.min(dets[j].y2);
            let inter = ((xx2 - xx1).max(0) * (yy2 - yy1).max(0)) as f32;
            if inter == 0.0 {
                continue;
            }
            let aj = ((dets[j].x2 - dets[j].x1).max(0) * (dets[j].y2 - dets[j].y1).max(0)) as f32;
            if inter / (ai + aj - inter + 1e-6) > iou_thresh {
                suppressed[j] = true;
            }
        }
    }
    kept
}

pub fn expand_box(det: &Detection, img_w: u32, img_h: u32, frac: f32) -> (u32, u32, u32, u32) {
    let bw = (det.x2 - det.x1) as f32;
    let bh = (det.y2 - det.y1) as f32;
    let cx1 = (det.x1 as f32 - bw * frac).max(0.0) as u32;
    let cy1 = (det.y1 as f32 - bh * frac).max(0.0) as u32;
    let cx2 = ((det.x2 as f32 + bw * frac) as u32).min(img_w);
    let cy2 = ((det.y2 as f32 + bh * frac) as u32).min(img_h);
    (cx1, cy1, cx2, cy2)
}

pub fn detect_resistors(model: &model::Model, img: &RgbImage, device: &Device) -> Vec<Detection> {
    detect_resistors_with(model, img, device, DEFAULT_CONF_THRESH)
}

pub fn detect_resistors_with(
    model: &model::Model,
    img: &RgbImage,
    device: &Device,
    conf_thresh: f32,
) -> Vec<Detection> {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (model, img, device, conf_thresh);
        panic!("detect_resistors_with is sync; use detect_resistors_with_async on wasm");
    }
    #[cfg(not(target_arch = "wasm32"))]
    pollster::block_on(detect_resistors_with_async(
        model,
        img,
        device,
        conf_thresh,
    ))
}

pub async fn detect_resistors_with_async(
    model: &model::Model,
    img: &RgbImage,
    device: &Device,
    conf_thresh: f32,
) -> Vec<Detection> {
    let (input, meta) = letterbox_to_tensor(img, DET_SIZE, device);
    let output = model.forward(input);
    let raw = decode_yolo_async(output, &meta, conf_thresh).await;
    let keep = nms(&raw, 0.5);
    keep.into_iter().map(|i| raw[i].clone()).collect()
}


