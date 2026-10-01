use std::sync::{Arc, LazyLock, Mutex};

use burn::tensor::{AllocationProperty, Bytes};
use gst::glib;
use gst::prelude::*;
use gst::subclass::prelude::*;
use gst_base::subclass::prelude::*;
use gst_video::VideoFormat;
use gst_video::prelude::*;
use gst_video::subclass::prelude::*;
use image::RgbImage;
use smart_default::SmartDefault;

use hires_rs::Device;
use hires_rs::detection::model::Model as DetectionModel;
use hires_rs::pipeline::{ProcessOptions, process_rgb_with};
use hires_rs::resolve_device;
use hires_rs::segmentation::model::Model as SegmentationModel;
use hires_rs::viz::{annotate_frame, annotate_rgb_slice};

use crate::weights::{DET_WEIGHTS, SEG_WEIGHTS};

const DEFAULT_SIZE: u32 = 512;
const DEFAULT_BAND_OVERLAY: bool = false;
/// Higher than the library default (`0.001`) to cut live-video false positives.
/// HiRes uses `0.01` on Ultralytics `.pt`; ONNX scores run lower, so `0.008`
/// is a practical middle ground. Raise toward `0.01` if still too noisy.
const DEFAULT_CONFIDENCE: f32 = 0.008;

static CAT: LazyLock<gst::DebugCategory> = LazyLock::new(|| {
    gst::DebugCategory::new(
        "hires",
        gst::DebugColorFlags::empty(),
        Some("HiRes resistor detector"),
    )
});

#[derive(Debug, Clone, Copy, SmartDefault)]
struct Settings {
    #[default(DEFAULT_SIZE)]
    size: u32,
    #[default(DEFAULT_BAND_OVERLAY)]
    band_overlay: bool,
    #[default(DEFAULT_CONFIDENCE)]
    confidence: f32,
}

struct Models {
    device: Device,
    det: DetectionModel,
    seg: SegmentationModel,
}

#[derive(Default)]
struct State {
    models: Option<Arc<Models>>,
    /// Reused packed-RGB scratch (avoids per-frame alloc).
    frame_buf: Vec<u8>,
}

#[derive(Default)]
pub struct Hires {
    settings: Mutex<Settings>,
    state: Mutex<State>,
}

fn embedded_bytes(data: &'static [u8]) -> Bytes {
    Bytes::from_shared(bytes::Bytes::from_static(data), AllocationProperty::Native)
}

#[glib::object_subclass]
impl ObjectSubclass for Hires {
    const NAME: &'static str = "GstHires";
    type Type = super::Hires;
    type ParentType = gst_video::VideoFilter;
}

impl ObjectImpl for Hires {
    fn properties() -> &'static [glib::ParamSpec] {
        static PROPERTIES: LazyLock<Vec<glib::ParamSpec>> = LazyLock::new(|| {
            vec![
                glib::ParamSpecUInt::builder("size")
                    .nick("Segmentation Size")
                    .blurb("Segmentation canvas size")
                    .default_value(DEFAULT_SIZE)
                    .build(),
                glib::ParamSpecBoolean::builder("band-overlay")
                    .nick("Band Overlay")
                    .blurb("Blend inferred band colors onto each detected resistor")
                    .default_value(DEFAULT_BAND_OVERLAY)
                    .build(),
                glib::ParamSpecFloat::builder("confidence")
                    .nick("Detection Confidence")
                    .blurb(
                        "Minimum YOLO detection confidence (0–1). Raise to reduce false positives; \
                         HiRes uses 0.01 on .pt weights, ONNX scores are typically lower.",
                    )
                    .minimum(0.0)
                    .maximum(1.0)
                    .default_value(DEFAULT_CONFIDENCE)
                    .build(),
            ]
        });

        PROPERTIES.as_ref()
    }

    fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
        let mut settings = self.settings.lock().unwrap();
        match pspec.name() {
            "size" => settings.size = value.get().unwrap(),
            "band-overlay" => settings.band_overlay = value.get().unwrap(),
            "confidence" => settings.confidence = value.get().unwrap(),
            _ => unimplemented!(),
        }
    }

    fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
        let settings = self.settings.lock().unwrap();
        match pspec.name() {
            "size" => settings.size.to_value(),
            "band-overlay" => settings.band_overlay.to_value(),
            "confidence" => settings.confidence.to_value(),
            _ => unimplemented!(),
        }
    }
}

impl GstObjectImpl for Hires {}

impl ElementImpl for Hires {
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static ELEMENT_METADATA: LazyLock<gst::subclass::ElementMetadata> = LazyLock::new(|| {
            gst::subclass::ElementMetadata::new(
                "HiRes Resistor Detector",
                "Filter/Video",
                "Detect resistors and overlay decoded value/tolerance on each box",
                "Andrew Martin",
            )
        });

        Some(&*ELEMENT_METADATA)
    }

    fn pad_templates() -> &'static [gst::PadTemplate] {
        static PAD_TEMPLATES: LazyLock<Vec<gst::PadTemplate>> = LazyLock::new(|| {
            let caps = gst_video::VideoCapsBuilder::new()
                .format(VideoFormat::Rgb)
                .build();

            let sink_pad_template = gst::PadTemplate::new(
                "sink",
                gst::PadDirection::Sink,
                gst::PadPresence::Always,
                &caps,
            )
            .unwrap();

            let src_pad_template = gst::PadTemplate::new(
                "src",
                gst::PadDirection::Src,
                gst::PadPresence::Always,
                &caps,
            )
            .unwrap();

            vec![sink_pad_template, src_pad_template]
        });

        PAD_TEMPLATES.as_ref()
    }
}

impl BaseTransformImpl for Hires {
    const MODE: gst_base::subclass::BaseTransformMode =
        gst_base::subclass::BaseTransformMode::AlwaysInPlace;
    const PASSTHROUGH_ON_SAME_CAPS: bool = false;
    const TRANSFORM_IP_ON_PASSTHROUGH: bool = true;

    fn start(&self) -> Result<(), gst::ErrorMessage> {
        let device = resolve_device();
        let det = DetectionModel::from_bytes(embedded_bytes(DET_WEIGHTS), &device);
        let seg = SegmentationModel::from_bytes(embedded_bytes(SEG_WEIGHTS), &device);

        let mut state = self.state.lock().unwrap();
        state.models = Some(Arc::new(Models { device, det, seg }));
        Ok(())
    }

    fn stop(&self) -> Result<(), gst::ErrorMessage> {
        let mut state = self.state.lock().unwrap();
        state.models = None;
        state.frame_buf.clear();
        Ok(())
    }
}

impl VideoFilterImpl for Hires {
    fn transform_frame_ip(
        &self,
        frame: &mut gst_video::VideoFrameRef<&mut gst::BufferRef>,
    ) -> Result<gst::FlowSuccess, gst::FlowError> {
        let settings = self.settings.lock().unwrap();
        let size = settings.size as usize;
        let band_overlay = settings.band_overlay;
        let confidence = settings.confidence;
        drop(settings);

        let (models, mut frame_buf) = {
            let mut state = self.state.lock().unwrap();
            let models = state.models.clone();
            let buf = std::mem::take(&mut state.frame_buf);
            (models, buf)
        };
        let Some(models) = models else {
            gst::error!(CAT, "models not loaded");
            return Err(gst::FlowError::Error);
        };

        let width = frame.width() as u32;
        let height = frame.height() as u32;
        let stride = frame.plane_stride()[0] as usize;
        let data = frame.plane_data_mut(0).unwrap();
        let row_bytes = (width as usize) * 3;
        let need = row_bytes * height as usize;
        let tight = stride == row_bytes;

        // Contiguous RGB: copy once for inference, annotate in-place on the Gst plane
        // (no full-frame write-back). Padded stride: pack → process → annotate → unpack.
        if tight {
            if frame_buf.len() != need {
                frame_buf.resize(need, 0);
            }
            frame_buf.copy_from_slice(&data[..need]);
        } else {
            frame_buf.clear();
            if frame_buf.capacity() < need {
                frame_buf.reserve(need);
            }
            for y in 0..height as usize {
                let start = y * stride;
                frame_buf.extend_from_slice(&data[start..start + row_bytes]);
            }
        }

        let img = RgbImage::from_raw(width, height, frame_buf).ok_or_else(|| {
            gst::error!(CAT, "failed to build RgbImage");
            gst::FlowError::Error
        })?;

        let opts = if band_overlay {
            ProcessOptions::overlay_bands()
        } else {
            ProcessOptions::overlay()
        }
        .with_det_conf(confidence);
        let results = process_rgb_with(&img, &models.det, &models.seg, &models.device, size, opts);
        let out = img.into_raw();

        if tight {
            annotate_rgb_slice(&mut data[..need], width, height, &results);
        } else {
            let mut packed = RgbImage::from_raw(width, height, out).ok_or_else(|| {
                gst::error!(CAT, "failed to rebuild RgbImage for annotate");
                gst::FlowError::Error
            })?;
            annotate_frame(&mut packed, &results);
            let out = packed.into_raw();
            for y in 0..height as usize {
                let src_start = y * row_bytes;
                let dst_start = y * stride;
                data[dst_start..dst_start + row_bytes]
                    .copy_from_slice(&out[src_start..src_start + row_bytes]);
            }
            let mut state = self.state.lock().unwrap();
            state.frame_buf = out;
            return Ok(gst::FlowSuccess::Ok);
        }

        {
            let mut state = self.state.lock().unwrap();
            state.frame_buf = out;
        }

        Ok(gst::FlowSuccess::Ok)
    }
}
