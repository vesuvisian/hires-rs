use std::cell::RefCell;
use std::path::Path;

use anyhow::{Context, Result};
use fast_image_resize::{FilterType, ResizeAlg, ResizeOptions, Resizer};
use image::{DynamicImage, RgbImage};

pub const IMG_EXTS: &[&str] = &["jpg", "jpeg", "png", "bmp", "webp"];

thread_local! {
    static RESIZER: RefCell<Resizer> = RefCell::new(Resizer::new());
}

pub(crate) fn with_resizer<R>(f: impl FnOnce(&mut Resizer) -> R) -> R {
    RESIZER.with(|r| f(&mut r.borrow_mut()))
}

/// Bilinear resize — stand-in for `image::imageops::FilterType::Triangle`.
pub fn resize_rgb_bilinear(src: &RgbImage, new_w: u32, new_h: u32) -> RgbImage {
    resize_rgb(
        src,
        new_w,
        new_h,
        ResizeAlg::Convolution(FilterType::Bilinear),
    )
}

/// Lanczos3 resize — stand-in for `image::imageops::FilterType::Lanczos3`.
pub fn resize_rgb_lanczos3(src: &RgbImage, new_w: u32, new_h: u32) -> RgbImage {
    resize_rgb(
        src,
        new_w,
        new_h,
        ResizeAlg::Convolution(FilterType::Lanczos3),
    )
}

fn resize_rgb(src: &RgbImage, new_w: u32, new_h: u32, alg: ResizeAlg) -> RgbImage {
    let new_w = new_w.max(1);
    let new_h = new_h.max(1);
    if src.width() == new_w && src.height() == new_h {
        return src.clone();
    }
    let mut dst = RgbImage::new(new_w, new_h);
    let opts = ResizeOptions::new().resize_alg(alg);
    with_resizer(|r| {
        r.resize(src, &mut dst, Some(&opts)).expect("rgb resize");
    });
    dst
}

pub fn load_rgb(path: &Path) -> Result<RgbImage> {
    let img = image::open(path)
        .with_context(|| format!("failed to open {}", path.display()))?
        .to_rgb8();
    Ok(img)
}

pub fn save_rgb(path: &Path, img: &RgbImage) -> Result<()> {
    DynamicImage::ImageRgb8(img.clone())
        .save(path)
        .with_context(|| format!("failed to save {}", path.display()))?;
    Ok(())
}

pub fn is_image_path(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| IMG_EXTS.iter().any(|x| e.eq_ignore_ascii_case(x)))
        .unwrap_or(false)
}
