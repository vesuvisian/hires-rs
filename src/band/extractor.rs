use nalgebra::{Matrix2, SymmetricEigen, Vector2};

use super::color_tables::{BandInfo, MIN_BAND_AREA, get_color_name};

#[derive(Clone, Debug)]
pub struct AxisInfo {
    pub axis_vector: [f64; 2],
    pub axis_origin: [f64; 2],
}

fn compute_pca_axis(points: &[[f64; 2]]) -> ([f64; 2], [f64; 2]) {
    if points.is_empty() {
        return ([1.0, 0.0], [128.0, 128.0]);
    }
    if points.len() == 1 {
        return ([1.0, 0.0], points[0]);
    }

    let n = points.len() as f64;
    let mut cx = 0.0;
    let mut cy = 0.0;
    for p in points {
        cx += p[0];
        cy += p[1];
    }
    cx /= n;
    cy /= n;

    let mut var_x = 0.0;
    let mut var_y = 0.0;
    let mut cov_xy = 0.0;
    for p in points {
        let dx = p[0] - cx;
        let dy = p[1] - cy;
        var_x += dx * dx;
        var_y += dy * dy;
        cov_xy += dx * dy;
    }
    let denom = n - 1.0;
    var_x /= denom;
    var_y /= denom;
    cov_xy /= denom;

    let cov = Matrix2::new(var_x, cov_xy, cov_xy, var_y);
    let eig = SymmetricEigen::new(cov);
    // Select eigenvector for the largest eigenvalue (principal axis).
    let mut best_i = 0usize;
    let mut best_val = eig.eigenvalues[0];
    for i in 1..eig.eigenvalues.len() {
        if eig.eigenvalues[i] > best_val {
            best_val = eig.eigenvalues[i];
            best_i = i;
        }
    }
    let v = eig.eigenvectors.column(best_i);
    let mut axis = Vector2::new(v[0], v[1]);
    let norm = axis.norm();
    if norm > 0.0 {
        axis /= norm;
    }
    if axis[0] < 0.0 {
        axis = -axis;
    }
    ([axis[0], axis[1]], [cx, cy])
}

fn gaussian_kernel(sigma: f64, radius: usize) -> Vec<f64> {
    let mut k = Vec::with_capacity(2 * radius + 1);
    let s2 = 2.0 * sigma * sigma;
    let mut sum = 0.0;
    for i in 0..=2 * radius {
        let x = i as f64 - radius as f64;
        let v = (-x * x / s2).exp();
        k.push(v);
        sum += v;
    }
    for v in &mut k {
        *v /= sum;
    }
    k
}

fn gaussian_filter1d(data: &[f32], sigma: f64) -> Vec<f32> {
    let radius = (sigma * 3.0).ceil() as usize;
    let kernel = gaussian_kernel(sigma, radius);
    let n = data.len();
    let mut out = vec![0.0f32; n];
    for (i, out_val) in out.iter_mut().enumerate() {
        let mut acc = 0.0;
        for (ki, &kv) in kernel.iter().enumerate() {
            let j = i as isize + ki as isize - radius as isize;
            let j = j.clamp(0, (n as isize) - 1) as usize;
            acc += data[j] as f64 * kv;
        }
        *out_val = acc as f32;
    }
    out
}

pub fn extract_bands_by_projection(
    mask: &[u8],
    height: usize,
    width: usize,
) -> (Vec<BandInfo>, Option<AxisInfo>) {
    const N_BINS: usize = 200;
    const SMOOTH_SIGMA: f64 = 2.5;

    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for y in 0..height {
        for x in 0..width {
            if mask[y * width + x] > 0 {
                xs.push(x as f64);
                ys.push(y as f64);
            }
        }
    }
    if xs.len() < 20 {
        return (Vec::new(), None);
    }

    // Small live crops often have thin bands that fail the fixed HiRes floors.
    let fg = xs.len();
    let min_band_bins: usize = if fg < 800 { 3 } else { 4 };
    let min_band_area: i32 = if fg < 1500 {
        (fg as i32 / 60).clamp(12, MIN_BAND_AREA)
    } else {
        MIN_BAND_AREA
    };

    let points: Vec<[f64; 2]> = xs.iter().zip(ys.iter()).map(|(&x, &y)| [x, y]).collect();
    let (axis_vector, axis_origin) = compute_pca_axis(&points);

    let mut t_values = Vec::with_capacity(points.len());
    for p in &points {
        let dx = p[0] - axis_origin[0];
        let dy = p[1] - axis_origin[1];
        t_values.push(dx * axis_vector[0] + dy * axis_vector[1]);
    }

    let class_ids: Vec<usize> = xs
        .iter()
        .zip(ys.iter())
        .map(|(&x, &y)| mask[y as usize * width + x as usize] as usize)
        .collect();

    let t_min = t_values.iter().cloned().fold(f64::INFINITY, f64::min);
    let t_max = t_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let t_range = t_max - t_min;
    if t_range < 1.0 {
        return (Vec::new(), None);
    }

    let bin_idx: Vec<usize> = t_values
        .iter()
        .map(|&t| {
            (((t - t_min) / t_range * N_BINS as f64) as isize).clamp(0, (N_BINS as isize) - 1)
                as usize
        })
        .collect();

    let mut vote_counts = vec![0.0f32; N_BINS * 13];
    for (b, &c) in bin_idx.iter().zip(class_ids.iter()) {
        if c < 13 {
            vote_counts[b * 13 + c] += 1.0;
        }
    }

    let mut smoothed = vec![0.0f32; N_BINS * 13];
    for c in 1..13 {
        let col: Vec<f32> = (0..N_BINS).map(|b| vote_counts[b * 13 + c]).collect();
        let filt = gaussian_filter1d(&col, SMOOTH_SIGMA);
        for b in 0..N_BINS {
            smoothed[b * 13 + c] = filt[b];
        }
    }

    let mut dominant = vec![0usize; N_BINS];
    for b in 0..N_BINS {
        let mut best_c = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for c in 1..13 {
            let v = smoothed[b * 13 + c];
            if v > best_v {
                best_v = v;
                best_c = c;
            }
        }
        let raw_sum: f32 = (1..13).map(|c| vote_counts[b * 13 + c]).sum();
        dominant[b] = if raw_sum == 0.0 { 0 } else { best_c };
    }

    let mut bands = Vec::new();
    let mut i = 0;
    while i < N_BINS {
        if dominant[i] == 0 {
            i += 1;
            continue;
        }
        let seg_cls = dominant[i];
        let start = i;
        while i < N_BINS && dominant[i] == seg_cls {
            i += 1;
        }
        let end = i;
        if end - start < min_band_bins {
            continue;
        }

        let mut in_run = Vec::new();
        for (pi, &b) in bin_idx.iter().enumerate() {
            if b >= start && b < end {
                in_run.push(pi);
            }
        }
        let area = in_run.len() as i32;
        if area < min_band_area {
            continue;
        }

        let mut bx_min = i32::MAX;
        let mut by_min = i32::MAX;
        let mut bx_max = i32::MIN;
        let mut by_max = i32::MIN;
        for &pi in &in_run {
            let bx = xs[pi] as i32;
            let by = ys[pi] as i32;
            bx_min = bx_min.min(bx);
            by_min = by_min.min(by);
            bx_max = bx_max.max(bx);
            by_max = by_max.max(by);
        }

        let t_center = t_min + (start + end) as f64 / 2.0 * t_range / N_BINS as f64;
        let centroid = (
            axis_origin[0] + t_center * axis_vector[0],
            axis_origin[1] + t_center * axis_vector[1],
        );

        let cls = seg_cls as i32;
        bands.push(BandInfo {
            class_id: cls,
            color_name: get_color_name(cls),
            centroid,
            area,
            bounding_box: (bx_min, by_min, bx_max, by_max),
        });
    }

    let axis_info = AxisInfo {
        axis_vector,
        axis_origin,
    };
    (bands, Some(axis_info))
}

pub fn filter_bands_by_area(bands: &[BandInfo], max_bands: usize) -> Vec<BandInfo> {
    let mut filtered: Vec<_> = bands
        .iter()
        .filter(|b| b.area >= MIN_BAND_AREA)
        .cloned()
        .collect();
    if filtered.len() > max_bands {
        filtered.sort_by_key(|a| std::cmp::Reverse(a.area));
        filtered.truncate(max_bands);
    }
    filtered
}
