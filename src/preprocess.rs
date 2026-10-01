//! Reusable letterbox / pad → NCHW float buffers (Argus-style scratch).

use std::cell::RefCell;

use burn::prelude::*;
use fast_image_resize::images::Image as FirImage;
use fast_image_resize::pixels::PixelType;
use fast_image_resize::{FilterType, ResizeAlg, ResizeOptions};
use image::RgbImage;

use crate::detection::LetterboxMeta;
use crate::image_io::with_resizer;
use crate::segmentation::{IMAGENET_MEAN, IMAGENET_STD, PadInfo};

/// Ensure `buf` has length `need` without zero-filling when capacity already covers it.
/// Caller must overwrite every element before reading.
fn ensure_chw_len(buf: &mut Vec<f32>, need: usize) {
    if buf.len() == need {
        return;
    }
    if buf.capacity() < need {
        buf.reserve(need - buf.capacity());
    }
    // SAFETY: pack_rgb_to_chw writes every element before the buffer is uploaded.
    unsafe {
        buf.set_len(need);
    }
}

pub struct DetScratch {
    size: u32,
    canvas: Vec<u8>,
    resized: Vec<u8>,
    /// Ping-pong slots so H2D can take ownership of one while the other is packed next.
    chw: [Vec<f32>; 2],
    chw_idx: usize,
}

impl DetScratch {
    fn with_size(size: u32) -> Self {
        let n = (size as usize) * (size as usize) * 3;
        let mut a = Vec::with_capacity(n);
        let mut b = Vec::with_capacity(n);
        ensure_chw_len(&mut a, n);
        ensure_chw_len(&mut b, n);
        Self {
            size,
            canvas: vec![114; n],
            resized: Vec::new(),
            chw: [a, b],
            chw_idx: 0,
        }
    }

    fn ensure(&mut self, size: u32) {
        if self.size == size {
            return;
        }
        *self = Self::with_size(size);
    }

    fn pack_take_chw(&mut self, size: usize, normalize: bool) -> Vec<f32> {
        let need = size * size * 3;
        let i = self.chw_idx;
        ensure_chw_len(&mut self.chw[i], need);
        pack_rgb_to_chw(&self.canvas, &mut self.chw[i], size, size, normalize);
        let packed = std::mem::take(&mut self.chw[i]);
        self.chw_idx ^= 1;
        // Warm the alternate slot for the next frame (no zero-fill).
        ensure_chw_len(&mut self.chw[self.chw_idx], need);
        packed
    }
}

pub struct SegScratch {
    size: usize,
    canvas: Vec<u8>,
    resized: Vec<u8>,
    /// Ping-pong CHW slots sized for the current batch (`B * 3 * S * S`).
    chw: [Vec<f32>; 2],
    chw_idx: usize,
}

impl SegScratch {
    fn with_size(size: usize) -> Self {
        let n = size * size * 3;
        let mut a = Vec::with_capacity(n);
        let mut b = Vec::with_capacity(n);
        ensure_chw_len(&mut a, n);
        ensure_chw_len(&mut b, n);
        Self {
            size,
            canvas: vec![0; n],
            resized: Vec::new(),
            chw: [a, b],
            chw_idx: 0,
        }
    }

    fn ensure(&mut self, size: usize) {
        if self.size == size {
            return;
        }
        *self = Self::with_size(size);
    }
}

thread_local! {
    static DET_SCRATCH: RefCell<DetScratch> = RefCell::new(DetScratch::with_size(640));
    static SEG_SCRATCH: RefCell<SegScratch> = RefCell::new(SegScratch::with_size(512));
}

fn resize_rgb_into(src: &RgbImage, dst: &mut [u8], new_w: u32, new_h: u32) {
    let need = (new_w as usize) * (new_h as usize) * 3;
    assert_eq!(dst.len(), need);
    let mut dst_img = FirImage::from_slice_u8(new_w, new_h, dst, PixelType::U8x3).expect("fir dst");
    let opts = ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Bilinear));
    with_resizer(|r| {
        r.resize(src, &mut dst_img, Some(&opts))
            .expect("fir resize");
    });
}

fn pack_rgb_to_chw(rgb: &[u8], chw: &mut [f32], w: usize, h: usize, normalize: bool) {
    let plane = w * h;
    debug_assert_eq!(rgb.len(), plane * 3);
    debug_assert_eq!(chw.len(), plane * 3);
    if normalize {
        pack_rgb_to_chw_imagenet(rgb, chw, plane);
    } else {
        pack_rgb_to_chw_01(rgb, chw, plane);
    }
}

/// Interleaved RGB u8 → planar float in `[0, 1]`.
fn pack_rgb_to_chw_01(rgb: &[u8], chw: &mut [f32], plane: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: rgb/chw lengths checked by caller; neon path respects plane bounds.
        unsafe {
            pack_rgb_to_chw_01_neon(rgb, chw, plane);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        pack_rgb_to_chw_01_scalar(rgb, chw, plane);
    }
}

/// Interleaved RGB u8 → ImageNet-normalized planar float.
fn pack_rgb_to_chw_imagenet(rgb: &[u8], chw: &mut [f32], plane: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            pack_rgb_to_chw_imagenet_neon(rgb, chw, plane);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        pack_rgb_to_chw_imagenet_scalar(rgb, chw, plane);
    }
}

#[inline(always)]
#[allow(dead_code)] // non-aarch64 path + unit tests
fn pack_rgb_to_chw_01_scalar(rgb: &[u8], chw: &mut [f32], plane: usize) {
    let inv = 1.0 / 255.0;
    let (r_plane, rest) = chw.split_at_mut(plane);
    let (g_plane, b_plane) = rest.split_at_mut(plane);
    for i in 0..plane {
        let o = i * 3;
        r_plane[i] = rgb[o] as f32 * inv;
        g_plane[i] = rgb[o + 1] as f32 * inv;
        b_plane[i] = rgb[o + 2] as f32 * inv;
    }
}

#[inline(always)]
#[allow(dead_code)] // non-aarch64 path + unit tests
fn pack_rgb_to_chw_imagenet_scalar(rgb: &[u8], chw: &mut [f32], plane: usize) {
    // (x/255 - mean) / std = x * (1/(255*std)) + (-mean/std)
    let scale = [
        1.0 / (255.0 * IMAGENET_STD[0]),
        1.0 / (255.0 * IMAGENET_STD[1]),
        1.0 / (255.0 * IMAGENET_STD[2]),
    ];
    let bias = [
        -IMAGENET_MEAN[0] / IMAGENET_STD[0],
        -IMAGENET_MEAN[1] / IMAGENET_STD[1],
        -IMAGENET_MEAN[2] / IMAGENET_STD[2],
    ];
    let (r_plane, rest) = chw.split_at_mut(plane);
    let (g_plane, b_plane) = rest.split_at_mut(plane);
    for i in 0..plane {
        let o = i * 3;
        r_plane[i] = rgb[o] as f32 * scale[0] + bias[0];
        g_plane[i] = rgb[o + 1] as f32 * scale[1] + bias[1];
        b_plane[i] = rgb[o + 2] as f32 * scale[2] + bias[2];
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn pack_rgb_to_chw_01_neon(rgb: &[u8], chw: &mut [f32], plane: usize) {
    use std::arch::aarch64::*;
    // SAFETY: caller guarantees rgb/chw lengths; neon is enabled via target_feature.
    unsafe {
        let inv = 1.0 / 255.0;
        let (r_plane, rest) = chw.split_at_mut(plane);
        let (g_plane, b_plane) = rest.split_at_mut(plane);
        let mut i = 0usize;
        while i + 8 <= plane {
            let px = vld3_u8(rgb.as_ptr().add(i * 3));
            for (lane, dst) in [
                (px.0, r_plane.as_mut_ptr().add(i)),
                (px.1, g_plane.as_mut_ptr().add(i)),
                (px.2, b_plane.as_mut_ptr().add(i)),
            ] {
                let u16v = vmovl_u8(lane);
                let lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(u16v)));
                let hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(u16v)));
                vst1q_f32(dst, vmulq_n_f32(lo, inv));
                vst1q_f32(dst.add(4), vmulq_n_f32(hi, inv));
            }
            i += 8;
        }
        while i < plane {
            let o = i * 3;
            r_plane[i] = rgb[o] as f32 * inv;
            g_plane[i] = rgb[o + 1] as f32 * inv;
            b_plane[i] = rgb[o + 2] as f32 * inv;
            i += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn pack_rgb_to_chw_imagenet_neon(rgb: &[u8], chw: &mut [f32], plane: usize) {
    use std::arch::aarch64::*;
    // SAFETY: caller guarantees rgb/chw lengths; neon is enabled via target_feature.
    unsafe {
        let scale = [
            1.0 / (255.0 * IMAGENET_STD[0]),
            1.0 / (255.0 * IMAGENET_STD[1]),
            1.0 / (255.0 * IMAGENET_STD[2]),
        ];
        let bias = [
            -IMAGENET_MEAN[0] / IMAGENET_STD[0],
            -IMAGENET_MEAN[1] / IMAGENET_STD[1],
            -IMAGENET_MEAN[2] / IMAGENET_STD[2],
        ];
        let (r_plane, rest) = chw.split_at_mut(plane);
        let (g_plane, b_plane) = rest.split_at_mut(plane);
        let mut i = 0usize;
        while i + 8 <= plane {
            let px = vld3_u8(rgb.as_ptr().add(i * 3));
            for (lane, dst, s, b) in [
                (px.0, r_plane.as_mut_ptr().add(i), scale[0], bias[0]),
                (px.1, g_plane.as_mut_ptr().add(i), scale[1], bias[1]),
                (px.2, b_plane.as_mut_ptr().add(i), scale[2], bias[2]),
            ] {
                let u16v = vmovl_u8(lane);
                let lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(u16v)));
                let hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(u16v)));
                let bq = vdupq_n_f32(b);
                vst1q_f32(dst, vfmaq_n_f32(bq, lo, s));
                vst1q_f32(dst.add(4), vfmaq_n_f32(bq, hi, s));
            }
            i += 8;
        }
        while i < plane {
            let o = i * 3;
            r_plane[i] = rgb[o] as f32 * scale[0] + bias[0];
            g_plane[i] = rgb[o + 1] as f32 * scale[1] + bias[1];
            b_plane[i] = rgb[o + 2] as f32 * scale[2] + bias[2];
            i += 1;
        }
    }
}

/// Fill only the letterbox/pad border (skip when the content covers the whole canvas).
fn fill_pad_border(
    canvas: &mut [u8],
    canvas_w: usize,
    canvas_h: usize,
    new_w: usize,
    new_h: usize,
    ox: usize,
    oy: usize,
    val: u8,
) {
    debug_assert_eq!(canvas.len(), canvas_w * canvas_h * 3);
    if ox == 0 && oy == 0 && new_w == canvas_w && new_h == canvas_h {
        return;
    }
    // Top band.
    if oy > 0 {
        canvas[..oy * canvas_w * 3].fill(val);
    }
    // Bottom band.
    let bottom = (oy + new_h) * canvas_w * 3;
    if bottom < canvas.len() {
        canvas[bottom..].fill(val);
    }
    // Left / right on content rows.
    if ox > 0 || ox + new_w < canvas_w {
        for y in oy..(oy + new_h).min(canvas_h) {
            let row = y * canvas_w * 3;
            if ox > 0 {
                canvas[row..row + ox * 3].fill(val);
            }
            let right = ox + new_w;
            if right < canvas_w {
                canvas[row + right * 3..row + canvas_w * 3].fill(val);
            }
        }
    }
}

fn paste_rect(
    canvas: &mut [u8],
    resized: &[u8],
    canvas_w: usize,
    new_w: usize,
    new_h: usize,
    ox: usize,
    oy: usize,
) {
    for y in 0..new_h {
        let src_row = y * new_w * 3;
        let dst_row = ((oy + y) * canvas_w + ox) * 3;
        canvas[dst_row..dst_row + new_w * 3]
            .copy_from_slice(&resized[src_row..src_row + new_w * 3]);
    }
}

/// Letterbox to square and upload as NCHW float `[0,1]` (pad = 114/255).
pub fn letterbox_to_tensor(
    img: &RgbImage,
    size: u32,
    device: &Device,
) -> (Tensor<4>, LetterboxMeta) {
    DET_SCRATCH.with(|cell| {
        let mut sp = cell.borrow_mut();
        let sp = &mut *sp;
        sp.ensure(size);
        let (orig_w, orig_h) = img.dimensions();
        let scale = (size as f32 / orig_w as f32).min(size as f32 / orig_h as f32);
        let new_w = (orig_w as f32 * scale).round().max(1.0) as u32;
        let new_h = (orig_h as f32 * scale).round().max(1.0) as u32;
        let pad_x = (size - new_w) as f32 / 2.0;
        let pad_y = (size - new_h) as f32 / 2.0;
        let ox = pad_x.floor() as usize;
        let oy = pad_y.floor() as usize;

        let need = (new_w as usize) * (new_h as usize) * 3;
        if sp.resized.len() != need {
            sp.resized.resize(need, 0);
        }
        resize_rgb_into(img, &mut sp.resized, new_w, new_h);

        let size_usize = size as usize;
        fill_pad_border(
            &mut sp.canvas,
            size_usize,
            size_usize,
            new_w as usize,
            new_h as usize,
            ox,
            oy,
            114,
        );
        paste_rect(
            &mut sp.canvas,
            &sp.resized,
            size_usize,
            new_w as usize,
            new_h as usize,
            ox,
            oy,
        );
        let chw = sp.pack_take_chw(size_usize, false);
        let td = TensorData::new(chw, [1, 3, size_usize, size_usize]);
        let tensor = Tensor::<4>::from_data(td, device);
        let meta = LetterboxMeta {
            scale,
            pad_x,
            pad_y,
            orig_w,
            orig_h,
        };
        (tensor, meta)
    })
}

/// Resize-pad to square and upload as ImageNet-normalized NCHW.
pub fn resize_pad_to_tensor(img: &RgbImage, size: usize, device: &Device) -> (Tensor<4>, PadInfo) {
    let (tensor, pads) = resize_pad_batch_to_tensor(&[img], size, device);
    (tensor, pads.into_iter().next().expect("one pad"))
}

/// Batch resize-pad → ImageNet-normalized NCHW `[B, 3, S, S]`.
pub fn resize_pad_batch_to_tensor(
    imgs: &[&RgbImage],
    size: usize,
    device: &Device,
) -> (Tensor<4>, Vec<PadInfo>) {
    assert!(!imgs.is_empty(), "resize_pad_batch_to_tensor: empty batch");
    SEG_SCRATCH.with(|cell| {
        let mut sp = cell.borrow_mut();
        let sp = &mut *sp;
        sp.ensure(size);
        let batch = imgs.len();
        let plane = size * size * 3;
        let need = batch * plane;
        let slot = sp.chw_idx;
        ensure_chw_len(&mut sp.chw[slot], need);

        let mut pads = Vec::with_capacity(batch);
        for (b, img) in imgs.iter().enumerate() {
            let (w, h) = img.dimensions();
            let scale = size as f32 / (h.max(w) as f32);
            let new_h = ((h as f32 * scale) as usize).max(1);
            let new_w = ((w as f32 * scale) as usize).max(1);
            let pad_y = (size - new_h) / 2;
            let pad_x = (size - new_w) / 2;

            let rneed = new_w * new_h * 3;
            if sp.resized.len() != rneed {
                sp.resized.resize(rneed, 0);
            }
            resize_rgb_into(img, &mut sp.resized, new_w as u32, new_h as u32);

            fill_pad_border(&mut sp.canvas, size, size, new_w, new_h, pad_x, pad_y, 0);
            paste_rect(
                &mut sp.canvas,
                &sp.resized,
                size,
                new_w,
                new_h,
                pad_x,
                pad_y,
            );
            let dst = &mut sp.chw[slot][b * plane..(b + 1) * plane];
            pack_rgb_to_chw(&sp.canvas, dst, size, size, true);
            pads.push(PadInfo {
                pad_y,
                pad_x,
                new_h,
                new_w,
            });
        }

        let chw = std::mem::take(&mut sp.chw[slot]);
        sp.chw_idx ^= 1;
        ensure_chw_len(&mut sp.chw[sp.chw_idx], need);
        let td = TensorData::new(chw, [batch, 3, size, size]);
        let tensor = Tensor::<4>::from_data(td, device);
        (tensor, pads)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rgb(plane: usize, seed: u32) -> Vec<u8> {
        let mut rgb = vec![0u8; plane * 3];
        let mut s = seed;
        for px in rgb.iter_mut() {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *px = (s >> 24) as u8;
        }
        rgb
    }

    #[test]
    fn pack_01_matches_scalar() {
        for plane in [1usize, 7, 8, 9, 64, 640 * 640] {
            let rgb = make_rgb(plane, 1);
            let mut got = vec![0.0f32; plane * 3];
            let mut exp = vec![0.0f32; plane * 3];
            pack_rgb_to_chw_01(&rgb, &mut got, plane);
            pack_rgb_to_chw_01_scalar(&rgb, &mut exp, plane);
            for (i, (a, b)) in got.iter().zip(exp.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-6,
                    "mismatch at {i} plane={plane}: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn pack_imagenet_matches_scalar() {
        for plane in [1usize, 7, 8, 9, 64, 512 * 512] {
            let rgb = make_rgb(plane, 2);
            let mut got = vec![0.0f32; plane * 3];
            let mut exp = vec![0.0f32; plane * 3];
            pack_rgb_to_chw_imagenet(&rgb, &mut got, plane);
            pack_rgb_to_chw_imagenet_scalar(&rgb, &mut exp, plane);
            for (i, (a, b)) in got.iter().zip(exp.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-5,
                    "mismatch at {i} plane={plane}: {a} vs {b}"
                );
            }
        }
    }
}
