use burn::prelude::Device;
use image::RgbImage;

use crate::band::{AxisInfo, BandInfo, DecodeOutcome, calculate_resistance_with_axis_info};
use crate::detection::model::Model as DetectionModel;
use crate::detection::{self, Detection, expand_box};
use crate::segmentation::model::Model as SegmentationModel;
use crate::segmentation::{SegOptions, run_inference_batch_async};

#[cfg(feature = "cli")]
use std::path::{Path, PathBuf};

#[cfg(feature = "cli")]
use anyhow::{Context, Result};
#[cfg(feature = "cli")]
use walkdir::WalkDir;

#[cfg(feature = "cli")]
use crate::image_io::{is_image_path, load_rgb, save_rgb};
#[cfg(feature = "cli")]
use crate::viz::{CompositeArgs, make_composite};

#[cfg(feature = "cli")]
pub struct PipelineArgs {
    pub output_dir: PathBuf,
    pub size: usize,
    pub det_conf: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct ProcessOptions {
    pub seg: SegOptions,
    /// YOLO detection confidence threshold (see [`detection::DEFAULT_CONF_THRESH`]).
    pub det_conf: f32,
    /// Keep crop/vis/bands for CLI composites.
    pub keep_composite_inputs: bool,
    /// Keep crop mask + origin for live band-color overlay.
    pub band_overlay: bool,
}

impl Default for ProcessOptions {
    fn default() -> Self {
        Self {
            seg: SegOptions::default(),
            det_conf: detection::DEFAULT_CONF_THRESH,
            keep_composite_inputs: true,
            band_overlay: false,
        }
    }
}

impl ProcessOptions {
    /// Lean live path: boxes + value labels only.
    pub const fn overlay() -> Self {
        Self {
            seg: SegOptions::overlay(),
            det_conf: detection::DEFAULT_CONF_THRESH,
            keep_composite_inputs: false,
            band_overlay: false,
        }
    }

    /// Live path that also keeps the seg mask for band-color blending.
    pub const fn overlay_bands() -> Self {
        Self {
            seg: SegOptions::overlay(),
            det_conf: detection::DEFAULT_CONF_THRESH,
            keep_composite_inputs: false,
            band_overlay: true,
        }
    }

    pub const fn with_det_conf(mut self, det_conf: f32) -> Self {
        self.det_conf = det_conf;
        self
    }
}

#[derive(Clone, Debug)]
pub struct DetectionResult {
    pub crop: usize,
    pub det_conf: f32,
    pub seg_conf: f32,
    pub value: String,
    pub tolerance: String,
}

/// In-memory result for one detected resistor (no disk I/O).
#[derive(Clone, Debug)]
pub struct AnnotatedDetection {
    pub crop_idx: usize,
    pub det: Detection,
    pub crop: Option<RgbImage>,
    pub vis: Option<RgbImage>,
    /// Top-left of the expanded crop in full-image coordinates.
    pub crop_x: u32,
    pub crop_y: u32,
    /// Class mask aligned to the crop (only when `band_overlay`).
    pub mask: Option<Vec<u8>>,
    pub mask_w: u32,
    pub mask_h: u32,
    pub bands: Vec<BandInfo>,
    pub axis_info: Option<AxisInfo>,
    pub det_conf: f32,
    pub seg_conf: f32,
    pub value: String,
    pub tolerance: String,
}

impl AnnotatedDetection {
    pub fn to_detection_result(&self) -> DetectionResult {
        DetectionResult {
            crop: self.crop_idx,
            det_conf: self.det_conf,
            seg_conf: self.seg_conf,
            value: self.value.clone(),
            tolerance: self.tolerance.clone(),
        }
    }
}

#[cfg(feature = "cli")]
pub fn collect_inputs(input: &Path) -> Result<Vec<PathBuf>> {
    if input.is_dir() {
        let mut paths: Vec<_> = WalkDir::new(input)
            .into_iter()
            .filter_map(|e| e.ok())
            .map(|e| e.into_path())
            .filter(|p| p.is_file() && is_image_path(p))
            .collect();
        paths.sort();
        Ok(paths)
    } else {
        Ok(vec![input.to_path_buf()])
    }
}

fn format_g(v: f64) -> String {
    if (v - v.round()).abs() < 1e-9 {
        format!("{}", v as i64)
    } else {
        let s = format!("{:.4}", v);
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

/// Run detection → segmentation → decode on an in-memory RGB image.
pub fn process_rgb(
    img: &RgbImage,
    det_model: &DetectionModel,
    seg_model: &SegmentationModel,
    device: &Device,
    size: usize,
) -> Vec<AnnotatedDetection> {
    process_rgb_with(
        img,
        det_model,
        seg_model,
        device,
        size,
        ProcessOptions::default(),
    )
}

/// Lean path for live overlay (no refine / vis / composite inputs).
pub fn process_rgb_overlay(
    img: &RgbImage,
    det_model: &DetectionModel,
    seg_model: &SegmentationModel,
    device: &Device,
    size: usize,
) -> Vec<AnnotatedDetection> {
    process_rgb_with(
        img,
        det_model,
        seg_model,
        device,
        size,
        ProcessOptions::overlay(),
    )
}

/// Live overlay that also returns seg masks for band-color blending.
pub fn process_rgb_overlay_bands(
    img: &RgbImage,
    det_model: &DetectionModel,
    seg_model: &SegmentationModel,
    device: &Device,
    size: usize,
) -> Vec<AnnotatedDetection> {
    process_rgb_with(
        img,
        det_model,
        seg_model,
        device,
        size,
        ProcessOptions::overlay_bands(),
    )
}

pub fn process_rgb_with(
    img: &RgbImage,
    det_model: &DetectionModel,
    seg_model: &SegmentationModel,
    device: &Device,
    size: usize,
    opts: ProcessOptions,
) -> Vec<AnnotatedDetection> {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (img, det_model, seg_model, device, size, opts);
        panic!("process_rgb_with is sync; use process_rgb_with_async on wasm");
    }
    #[cfg(not(target_arch = "wasm32"))]
    pollster::block_on(process_rgb_with_async(
        img, det_model, seg_model, device, size, opts,
    ))
}

pub async fn process_rgb_with_async(
    img: &RgbImage,
    det_model: &DetectionModel,
    seg_model: &SegmentationModel,
    device: &Device,
    size: usize,
    opts: ProcessOptions,
) -> Vec<AnnotatedDetection> {
    let (img_w, img_h) = img.dimensions();
    let dets = detection::detect_resistors_with_async(det_model, img, device, opts.det_conf).await;
    if dets.is_empty() {
        return Vec::new();
    }

    struct CropJob {
        crop_idx: usize,
        det: Detection,
        crop: RgbImage,
        crop_x: u32,
        crop_y: u32,
        crop_w: u32,
        crop_h: u32,
    }

    let mut jobs = Vec::new();
    for (i, det) in dets.into_iter().enumerate() {
        let (cx1, cy1, cx2, cy2) = expand_box(&det, img_w, img_h, 0.10);
        let crop_w = cx2.saturating_sub(cx1);
        let crop_h = cy2.saturating_sub(cy1);
        if crop_w.min(crop_h) < 30 {
            continue;
        }
        let crop = image::imageops::crop_imm(img, cx1, cy1, crop_w, crop_h).to_image();
        jobs.push(CropJob {
            crop_idx: i,
            det,
            crop,
            crop_x: cx1,
            crop_y: cy1,
            crop_w,
            crop_h,
        });
    }
    if jobs.is_empty() {
        return Vec::new();
    }

    let crop_refs: Vec<&RgbImage> = jobs.iter().map(|j| &j.crop).collect();
    let segs = run_inference_batch_async(seg_model, &crop_refs, device, size, opts.seg).await;

    let keep_bands = opts.keep_composite_inputs;
    let mut results = Vec::with_capacity(jobs.len());
    for (job, seg) in jobs.into_iter().zip(segs) {
        let (decode, axis_info) =
            calculate_resistance_with_axis_info(&seg.mask, seg.height as usize, seg.width as usize);

        let (value_str, tol_str, bands) = match &decode {
            DecodeOutcome::Ok(r) => (
                r.formatted(),
                format!("±{}%", format_g(r.tolerance)),
                if keep_bands {
                    r.bands.clone()
                } else {
                    Vec::new()
                },
            ),
            DecodeOutcome::Err(e) => (
                format!("ERROR({}): {}", e.error_type, e.message),
                String::new(),
                if keep_bands {
                    e.detected_bands.clone()
                } else {
                    Vec::new()
                },
            ),
        };

        let mask = if opts.band_overlay {
            Some(seg.mask)
        } else {
            None
        };

        results.push(AnnotatedDetection {
            crop_idx: job.crop_idx,
            det_conf: job.det.conf,
            det: job.det,
            crop: opts.keep_composite_inputs.then_some(job.crop),
            vis: seg.vis,
            crop_x: job.crop_x,
            crop_y: job.crop_y,
            mask,
            mask_w: job.crop_w,
            mask_h: job.crop_h,
            bands,
            axis_info: opts.keep_composite_inputs.then_some(axis_info).flatten(),
            seg_conf: seg.seg_conf,
            value: value_str,
            tolerance: tol_str,
        });
    }

    results
}

#[cfg(feature = "cli")]
pub fn run_pipeline(
    image_path: &Path,
    det_model: &DetectionModel,
    seg_model: &SegmentationModel,
    device: &Device,
    args: &PipelineArgs,
) -> Result<Vec<DetectionResult>> {
    let img = load_rgb(image_path)?;
    let annotated = process_rgb_with(
        &img,
        det_model,
        seg_model,
        device,
        args.size,
        ProcessOptions {
            det_conf: args.det_conf,
            ..ProcessOptions::default()
        },
    );
    if annotated.is_empty() {
        return Ok(Vec::new());
    }

    std::fs::create_dir_all(&args.output_dir)?;
    let stem = image_path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "image".into());

    let mut results = Vec::new();
    for ann in &annotated {
        let (Some(crop), Some(vis)) = (ann.crop.as_ref(), ann.vis.as_ref()) else {
            anyhow::bail!("composite inputs missing; use default ProcessOptions for CLI");
        };
        let composite = make_composite(CompositeArgs {
            full: &img,
            crop,
            vis,
            bands: &ann.bands,
            axis_info: ann.axis_info.as_ref(),
            det: &ann.det,
            value_str: &ann.value,
            tol_str: &ann.tolerance,
        });
        let out_path = args
            .output_dir
            .join(format!("{stem}_resistor{}.png", ann.crop_idx));
        save_rgb(&out_path, &composite)
            .with_context(|| format!("saving {}", out_path.display()))?;

        results.push(ann.to_detection_result());
    }

    Ok(results)
}
