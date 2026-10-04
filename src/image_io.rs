use std::cell::RefCell;

use fast_image_resize::{FilterType, ResizeAlg, ResizeOptions, Resizer};
use image::RgbImage;

pub const IMG_EXTS: &[&str] = &["jpg", "jpeg", "png", "bmp", "webp"];

thread_local! {
    static RESIZER: RefCell<Resizer> = RefCell::new(Resizer::new());
}

pub(crate) fn with_resizer<R>(f: impl FnOnce(&mut Resizer) -> R) -> R {
    RESIZER.with(|r| f(&mut r.borrow_mut()))
}

/// Decode JPEG/PNG/BMP/WebP bytes the same way the CLI opens files.
pub fn load_rgb_from_memory(bytes: &[u8]) -> anyhow::Result<RgbImage> {
    Ok(image::load_from_memory(bytes)?.to_rgb8())
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

#[cfg(not(target_arch = "wasm32"))]
pub fn load_rgb(path: &std::path::Path) -> anyhow::Result<RgbImage> {
    use anyhow::Context;

    let img = image::open(path)
        .with_context(|| format!("failed to open {}", path.display()))?
        .to_rgb8();
    Ok(img)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn save_rgb(path: &std::path::Path, img: &RgbImage) -> anyhow::Result<()> {
    use anyhow::Context;
    use image::DynamicImage;

    DynamicImage::ImageRgb8(img.clone())
        .save(path)
        .with_context(|| format!("failed to save {}", path.display()))?;
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
pub fn is_image_path(path: &std::path::Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| IMG_EXTS.iter().any(|x| e.eq_ignore_ascii_case(x)))
        .unwrap_or(false)
}
