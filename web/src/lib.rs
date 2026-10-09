#![recursion_limit = "512"]

use burn::tensor::{AllocationProperty, Bytes};
use hires_rs::Device;
use hires_rs::detection::model::Model as DetectionModel;
use hires_rs::image_io::load_rgb_from_memory;
use hires_rs::pipeline::{ProcessOptions, process_rgb_with_async};
use hires_rs::segmentation::VIS_COLORS;
use hires_rs::segmentation::model::Model as SegmentationModel;
use image::RgbImage;
use js_sys::{Array, Object, Reflect, Uint8Array};
use wasm_bindgen::prelude::*;

const SEG_SIZE: usize = 512;

fn owned_bytes(data: Vec<u8>) -> Bytes {
    Bytes::from_shared(bytes::Bytes::from(data), AllocationProperty::Native)
}

fn js_err(msg: impl ToString) -> JsValue {
    JsValue::from_str(&msg.to_string())
}

fn set(obj: &Object, key: &str, value: &JsValue) -> Result<(), JsValue> {
    Reflect::set(obj, &JsValue::from_str(key), value)?;
    Ok(())
}

fn process_opts(det_conf: f32, band_overlay: bool) -> ProcessOptions {
    let det_conf = if det_conf.is_finite() && det_conf > 0.0 {
        det_conf
    } else {
        hires_rs::detection::DEFAULT_CONF_THRESH
    };
    if band_overlay {
        ProcessOptions::overlay_bands()
    } else {
        ProcessOptions::overlay()
    }
    .with_det_conf(det_conf)
}

fn rgba_to_rgb(width: u32, height: u32, rgba: &[u8]) -> Result<RgbImage, JsValue> {
    let n = width as usize * height as usize;
    let need = n.saturating_mul(4);
    if rgba.len() < need {
        return Err(js_err(format!(
            "rgba buffer too small: got {}, need {need}",
            rgba.len()
        )));
    }
    let mut rgb = Vec::with_capacity(n * 3);
    for px in rgba.chunks_exact(4).take(n) {
        rgb.extend_from_slice(&px[0..3]);
    }
    RgbImage::from_raw(width, height, rgb).ok_or_else(|| js_err("invalid RGB image"))
}

#[wasm_bindgen]
pub struct HiresApp {
    device: Device,
    det: Option<DetectionModel>,
    seg: Option<SegmentationModel>,
    image: Option<RgbImage>,
}

#[wasm_bindgen]
impl HiresApp {
    pub async fn init() -> Result<HiresApp, JsValue> {
        console_error_panic_hook::set_once();
        // On wasm, CubeCL/wgpu adapter request is async. Device::default() only
        // names a device and load_weights panics with "call init() before load()".
        #[cfg(all(target_arch = "wasm32", feature = "wgpu"))]
        let device = Device::wgpu_async(hires_rs::DeviceKind::DefaultDevice).await;
        #[cfg(not(all(target_arch = "wasm32", feature = "wgpu")))]
        let device = hires_rs::resolve_device();
        Ok(Self {
            device,
            det: None,
            seg: None,
            image: None,
        })
    }

    #[wasm_bindgen(js_name = loadWeights)]
    pub fn load_weights(&mut self, det: &[u8], seg: &[u8]) -> Result<(), JsValue> {
        if det.is_empty() || seg.is_empty() {
            return Err(js_err("weight buffers are empty"));
        }
        self.det = Some(DetectionModel::from_bytes(
            owned_bytes(det.to_vec()),
            &self.device,
        ));
        self.seg = Some(SegmentationModel::from_bytes(
            owned_bytes(seg.to_vec()),
            &self.device,
        ));
        Ok(())
    }

    #[wasm_bindgen(js_name = backendName)]
    pub fn backend_name(&self) -> String {
        if cfg!(feature = "wgpu") {
            "wgpu".into()
        } else {
            "flex".into()
        }
    }

    /// Decode with the same `image` crate path as `cargo run`.
    #[wasm_bindgen(js_name = loadImage)]
    pub fn load_image(&mut self, bytes: &[u8]) -> Result<JsValue, JsValue> {
        if bytes.is_empty() {
            return Err(js_err("empty image"));
        }
        let img = load_rgb_from_memory(bytes).map_err(|e| js_err(e))?;
        if img.width() == 0 || img.height() == 0 {
            return Err(js_err("empty image"));
        }
        let obj = Object::new();
        set(&obj, "width", &JsValue::from(img.width()))?;
        set(&obj, "height", &JsValue::from(img.height()))?;
        self.image = Some(img);
        Ok(obj.into())
    }

    /// Packed RGBA of the decoded RGB the models see (no browser color management).
    #[wasm_bindgen(js_name = imageRgba)]
    pub fn image_rgba(&self) -> Result<Uint8Array, JsValue> {
        let img = self
            .image
            .as_ref()
            .ok_or_else(|| js_err("no image loaded"))?;
        let mut rgba = Vec::with_capacity(img.len() * 4 / 3);
        for px in img.pixels() {
            rgba.extend_from_slice(&[px[0], px[1], px[2], 255]);
        }
        let ua = Uint8Array::new_with_length(rgba.len() as u32);
        ua.copy_from(&rgba);
        Ok(ua)
    }

    pub async fn infer(&self, det_conf: f32, band_overlay: bool) -> Result<JsValue, JsValue> {
        let det = self
            .det
            .as_ref()
            .ok_or_else(|| js_err("models not loaded"))?;
        let seg = self
            .seg
            .as_ref()
            .ok_or_else(|| js_err("models not loaded"))?;
        // Clone before any `.await` so a concurrent `load_image` from JS cannot
        // drop the buffer while the pipeline still holds a reference (wasm-bindgen
        // does not enforce Rust borrows across the JS event loop).
        let img = self
            .image
            .as_ref()
            .ok_or_else(|| js_err("no image loaded"))?
            .clone();
        let opts = process_opts(det_conf, band_overlay);
        let results = process_rgb_with_async(&img, det, seg, &self.device, SEG_SIZE, opts).await;
        detections_to_js(&results, band_overlay)
    }

    /// Canvas/bitmap fallback when the `image` crate cannot decode the file.
    #[wasm_bindgen(js_name = inferRgba)]
    pub async fn infer_rgba(
        &self,
        width: u32,
        height: u32,
        rgba: &[u8],
        det_conf: f32,
        band_overlay: bool,
    ) -> Result<JsValue, JsValue> {
        let det = self
            .det
            .as_ref()
            .ok_or_else(|| js_err("models not loaded"))?;
        let seg = self
            .seg
            .as_ref()
            .ok_or_else(|| js_err("models not loaded"))?;
        if width == 0 || height == 0 {
            return Err(js_err("empty image"));
        }
        let img = rgba_to_rgb(width, height, rgba)?;
        let opts = process_opts(det_conf, band_overlay);
        let results = process_rgb_with_async(&img, det, seg, &self.device, SEG_SIZE, opts).await;
        detections_to_js(&results, band_overlay)
    }
}

#[wasm_bindgen(js_name = visColors)]
pub fn vis_colors() -> Array {
    let out = Array::new();
    for c in VIS_COLORS {
        let rgb = Array::new();
        rgb.push(&JsValue::from(c[0]));
        rgb.push(&JsValue::from(c[1]));
        rgb.push(&JsValue::from(c[2]));
        out.push(&rgb);
    }
    out
}

fn detections_to_js(
    results: &[hires_rs::pipeline::AnnotatedDetection],
    band_overlay: bool,
) -> Result<JsValue, JsValue> {
    let arr = Array::new();
    for ann in results {
        let obj = Object::new();
        set(&obj, "x1", &JsValue::from_f64(f64::from(ann.det.x1)))?;
        set(&obj, "y1", &JsValue::from_f64(f64::from(ann.det.y1)))?;
        set(&obj, "x2", &JsValue::from_f64(f64::from(ann.det.x2)))?;
        set(&obj, "y2", &JsValue::from_f64(f64::from(ann.det.y2)))?;
        set(&obj, "det_conf", &JsValue::from(f64::from(ann.det_conf)))?;
        set(&obj, "value", &JsValue::from_str(&ann.value))?;
        set(&obj, "tolerance", &JsValue::from_str(&ann.tolerance))?;
        set(&obj, "crop_x", &JsValue::from(ann.crop_x))?;
        set(&obj, "crop_y", &JsValue::from(ann.crop_y))?;
        set(&obj, "mask_w", &JsValue::from(ann.mask_w))?;
        set(&obj, "mask_h", &JsValue::from(ann.mask_h))?;
        if band_overlay {
            if let Some(mask) = ann.mask.as_ref() {
                let ua = Uint8Array::new_with_length(mask.len() as u32);
                ua.copy_from(mask);
                set(&obj, "mask", &ua)?;
            }
        }
        arr.push(&obj);
    }
    Ok(arr.into())
}
