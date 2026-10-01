use std::sync::LazyLock;

use ab_glyph::{FontRef, PxScale};
use image::{GenericImage, ImageBuffer, Rgb, RgbImage};
use imageproc::drawing::{
    draw_filled_circle_mut, draw_filled_rect_mut, draw_line_segment_mut, draw_text_mut,
};
use imageproc::rect::Rect;

use crate::band::{AxisInfo, BandInfo};
use crate::detection::Detection;
use crate::pipeline::AnnotatedDetection;
use crate::segmentation::VIS_COLORS;

const PANEL_H: u32 = 300;
const GAP_PX: u32 = 8;
const BAND_OVERLAY_ALPHA: f32 = 0.55;

static FONT_DATA: &[u8] = include_bytes!("DejaVuSans.ttf");
static FONT: LazyLock<Option<FontRef<'static>>> =
    LazyLock::new(|| FontRef::try_from_slice(FONT_DATA).ok());

fn color_hex(name: &str) -> [u8; 3] {
    match name.to_lowercase().as_str() {
        "black" => [0, 0, 0],
        "blue" => [0, 74, 173],
        "brown" => [94, 56, 49],
        "gold" => [235, 191, 124],
        "green" => [0, 191, 99],
        "grey" | "gray" => [88, 90, 89],
        "orange" => [255, 145, 77],
        "violet" => [94, 23, 235],
        "red" => [228, 50, 50],
        "silver" => [205, 205, 205],
        "white" => [255, 255, 255],
        "yellow" => [244, 203, 36],
        _ => [255, 255, 255],
    }
}

fn resize_to_height(img: &RgbImage, h: u32) -> (RgbImage, f32) {
    let (w0, h0) = img.dimensions();
    let s = h as f32 / h0 as f32;
    let nw = ((w0 as f32 * s).round() as u32).max(1);
    let out = crate::image_io::resize_rgb_lanczos3(img, nw, h);
    (out, s)
}

fn blend_overlay(crop: &RgbImage, vis: &RgbImage, alpha: f32) -> RgbImage {
    let (w, h) = crop.dimensions();
    let mut out = crop.clone();
    for y in 0..h {
        for x in 0..w {
            let v = vis.get_pixel(x, y).0;
            if v[0] as u16 + v[1] as u16 + v[2] as u16 > 0 {
                let c = crop.get_pixel(x, y).0;
                let blended = [
                    (alpha * v[0] as f32 + (1.0 - alpha) * c[0] as f32) as u8,
                    (alpha * v[1] as f32 + (1.0 - alpha) * c[1] as f32) as u8,
                    (alpha * v[2] as f32 + (1.0 - alpha) * c[2] as f32) as u8,
                ];
                out.put_pixel(x, y, Rgb(blended));
            }
        }
    }
    out
}

fn draw_badge<I>(img: &mut I, x1: i32, y1: i32, text: &str, color: Rgb<u8>, font: &FontRef<'_>)
where
    I: GenericImage<Pixel = Rgb<u8>>,
{
    let scale = PxScale::from(16.0);
    let (tw, th) = imageproc::drawing::text_size(scale, font, text);
    let pad = 4i32;
    let bx1 = x1.max(0);
    let by2 = y1.max(0);
    let by1 = (by2 - th as i32 - 2 * pad).max(0);
    let bx2 = bx1 + tw as i32 + 2 * pad;
    let rect = Rect::at(bx1, by1).of_size((bx2 - bx1).max(1) as u32, (by2 - by1).max(1) as u32);
    draw_filled_rect_mut(img, rect, color);
    draw_text_mut(
        img,
        Rgb([255, 255, 255]),
        bx1 + pad,
        by2 - pad - th as i32,
        scale,
        font,
        text,
    );
}

fn draw_bands_panel(
    base: &RgbImage,
    bands: &[BandInfo],
    axis_info: Option<&AxisInfo>,
    s3: f32,
    crop_w: u32,
    crop_h: u32,
) -> RgbImage {
    let mut out = base.clone();
    let (Some(axis), true) = (axis_info, !bands.is_empty()) else {
        return out;
    };
    let av = axis.axis_vector;
    let ao = [
        axis.axis_origin[0] * s3 as f64,
        axis.axis_origin[1] * s3 as f64,
    ];
    let perp = [-av[1], av[0]];
    let tick = ((crop_h.min(crop_w) as f32 * 0.20) * s3).max(14.0) as f64;
    let (ow, oh) = out.dimensions();
    let t = ow.max(oh) as f64;
    draw_line_segment_mut(
        &mut out,
        ((ao[0] - t * av[0]) as f32, (ao[1] - t * av[1]) as f32),
        ((ao[0] + t * av[0]) as f32, (ao[1] + t * av[1]) as f32),
        Rgb([180, 180, 180]),
    );

    // Embedded DejaVu Sans; if parse fails, skip band-number labels.
    let font_data = include_bytes!("DejaVuSans.ttf");
    let font = FontRef::try_from_slice(font_data).ok();

    for (j, band) in bands.iter().enumerate() {
        let cx = (band.centroid.0 * s3 as f64) as i32;
        let cy = (band.centroid.1 * s3 as f64) as i32;
        let col = color_hex(&band.color_name);
        draw_line_segment_mut(
            &mut out,
            (
                (cx as f64 - perp[0] * tick) as f32,
                (cy as f64 - perp[1] * tick) as f32,
            ),
            (
                (cx as f64 + perp[0] * tick) as f32,
                (cy as f64 + perp[1] * tick) as f32,
            ),
            Rgb([255, 255, 255]),
        );
        draw_filled_circle_mut(&mut out, (cx, cy), 9, Rgb(col));
        if let Some(ref f) = font {
            let tc = if matches!(
                band.color_name.to_lowercase().as_str(),
                "white" | "gold" | "yellow" | "silver"
            ) {
                Rgb([20, 20, 20])
            } else {
                Rgb([255, 255, 255])
            };
            draw_text_mut(
                &mut out,
                tc,
                cx - 4,
                cy - 4,
                PxScale::from(12.0),
                f,
                &format!("{}", j + 1),
            );
        }
    }
    let _ = oh;
    out
}

/// Draw boxes, value badges, and optional band-color mask overlays.
pub fn annotate_frame<I>(img: &mut I, results: &[AnnotatedDetection])
where
    I: GenericImage<Pixel = Rgb<u8>>,
{
    let font = FONT.as_ref();

    for ann in results {
        if let Some(mask) = ann.mask.as_ref() {
            blend_mask_onto(
                img,
                mask,
                ann.mask_w,
                ann.mask_h,
                ann.crop_x,
                ann.crop_y,
                BAND_OVERLAY_ALPHA,
            );
        }

        let det = &ann.det;
        imageproc::drawing::draw_hollow_rect_mut(
            img,
            Rect::at(det.x1, det.y1).of_size(
                (det.x2 - det.x1).max(1) as u32,
                (det.y2 - det.y1).max(1) as u32,
            ),
            Rgb([0, 210, 0]),
        );

        let Some(font) = font else {
            continue;
        };
        let label = if ann.tolerance.is_empty() {
            ann.value.clone()
        } else {
            format!("{}  {}", ann.value, ann.tolerance)
        };
        let label = label.replace('Ω', "Ohm").replace('±', "+/-");
        draw_badge(img, det.x1 + 4, det.y1, &label, Rgb([0, 180, 0]), font);
    }
}

/// Annotate a tightly-packed RGB plane (`stride == width * 3`) in place.
pub fn annotate_rgb_slice(buf: &mut [u8], width: u32, height: u32, results: &[AnnotatedDetection]) {
    let Some(mut img) = ImageBuffer::<Rgb<u8>, _>::from_raw(width, height, buf) else {
        return;
    };
    annotate_frame(&mut img, results);
}

fn blend_mask_onto<I>(
    img: &mut I,
    mask: &[u8],
    mask_w: u32,
    mask_h: u32,
    ox: u32,
    oy: u32,
    alpha: f32,
) where
    I: GenericImage<Pixel = Rgb<u8>>,
{
    let img_w = img.width();
    let img_h = img.height();
    let inv = 1.0 - alpha;
    for y in 0..mask_h {
        let py = oy + y;
        if py >= img_h {
            break;
        }
        let row = (y * mask_w) as usize;
        for x in 0..mask_w {
            let cls = mask[row + x as usize] as usize;
            if cls == 0 {
                continue;
            }
            let px = ox + x;
            if px >= img_w {
                break;
            }
            let c = VIS_COLORS[cls.min(12)];
            let p = img.get_pixel(px, py).0;
            img.put_pixel(
                px,
                py,
                Rgb([
                    (alpha * c[0] as f32 + inv * p[0] as f32) as u8,
                    (alpha * c[1] as f32 + inv * p[1] as f32) as u8,
                    (alpha * c[2] as f32 + inv * p[2] as f32) as u8,
                ]),
            );
        }
    }
}

pub struct CompositeArgs<'a> {
    pub full: &'a RgbImage,
    pub crop: &'a RgbImage,
    pub vis: &'a RgbImage,
    pub bands: &'a [BandInfo],
    pub axis_info: Option<&'a AxisInfo>,
    pub det: &'a Detection,
    pub value_str: &'a str,
    pub tol_str: &'a str,
}

pub fn make_composite(args: CompositeArgs<'_>) -> RgbImage {
    let CompositeArgs {
        full,
        crop,
        vis,
        bands,
        axis_info,
        det,
        value_str,
        tol_str,
    } = args;

    // Panel 1: detection box on full image
    let mut det_img = full.clone();
    imageproc::drawing::draw_hollow_rect_mut(
        &mut det_img,
        Rect::at(det.x1, det.y1).of_size(
            (det.x2 - det.x1).max(1) as u32,
            (det.y2 - det.y1).max(1) as u32,
        ),
        Rgb([0, 210, 0]),
    );
    let (mut p1, s1) = resize_to_height(&det_img, PANEL_H);

    let font = FONT.as_ref().expect("embedded font");
    draw_badge(
        &mut p1,
        ((det.x1 + 4) as f32 * s1) as i32,
        (det.y1 as f32 * s1) as i32,
        &format!("resistor {:.2}", det.conf),
        Rgb([0, 210, 0]),
        font,
    );

    let p2_src = blend_overlay(crop, vis, 0.6);
    let (p2, _) = resize_to_height(&p2_src, PANEL_H);

    let p3_src = blend_overlay(crop, vis, 0.30);
    let (p3b, s3) = resize_to_height(&p3_src, PANEL_H);
    let (cw, ch) = crop.dimensions();
    let mut p3 = draw_bands_panel(&p3b, bands, axis_info, s3, cw, ch);

    let total_w = p1.width() + GAP_PX + p2.width() + GAP_PX + p3.width();
    let mut row = RgbImage::from_pixel(total_w, PANEL_H, Rgb([245, 245, 245]));
    image::imageops::replace(&mut row, &p1, 0, 0);
    image::imageops::replace(&mut row, &p2, (p1.width() + GAP_PX) as i64, 0);
    let p3_off = p1.width() + GAP_PX + p2.width() + GAP_PX;
    image::imageops::replace(&mut row, &p3, p3_off as i64, 0);

    let label = format!("{value_str}  {tol_str}")
        .replace('Ω', "Ohm")
        .replace('±', "+/-");
    draw_badge(
        &mut row,
        (p3_off + 4) as i32,
        PANEL_H as i32,
        &label,
        Rgb([0, 180, 0]),
        font,
    );

    let _ = &mut p3;
    row
}
