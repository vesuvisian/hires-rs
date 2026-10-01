use super::color_tables::{
    BandInfo, CalculationError, DEFAULT_TOLERANCE, ResistanceResult, color_to_digit,
    color_to_multiplier, color_to_tolerance,
};
use super::extractor::{AxisInfo, extract_bands_by_projection, filter_bands_by_area};

const E24: [f64; 24] = [
    1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0, 3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6,
    6.2, 6.8, 7.5, 8.2, 9.1,
];

const GAP_RATIO_THRESHOLD: f64 = 1.5;

const VALID_TOLERANCE_COLORS: &[&str] = &[
    "brown", "red", "green", "blue", "violet", "grey", "gold", "silver",
];

#[derive(Clone, Debug)]
pub enum DecodeOutcome {
    Ok(ResistanceResult),
    Err(CalculationError),
}

fn is_e24(value: f64, tol: f64) -> bool {
    if value <= 0.0 {
        return false;
    }
    let mut m = value;
    while m >= 10.0 {
        m /= 10.0;
    }
    while m < 1.0 {
        m *= 10.0;
    }
    E24.iter().any(|&e| (m - e).abs() / e < tol)
}

fn compute_axis_confidence(bands: &[BandInfo]) -> f64 {
    if bands.len() < 2 {
        return 0.0;
    }
    let n = bands.len() as f64;
    let mx = bands.iter().map(|b| b.centroid.0).sum::<f64>() / n;
    let my = bands.iter().map(|b| b.centroid.1).sum::<f64>() / n;
    let sx = (bands
        .iter()
        .map(|b| (b.centroid.0 - mx).powi(2))
        .sum::<f64>()
        / n)
        .sqrt();
    let sy = (bands
        .iter()
        .map(|b| (b.centroid.1 - my).powi(2))
        .sum::<f64>()
        / n)
        .sqrt();
    sx.max(sy) / (sx + sy + 1e-6)
}

fn compute_area_confidence(mask: &[u8], bands: &[BandInfo]) -> f64 {
    let non_bg = mask.iter().filter(|&&c| c > 0).count() as f64;
    if non_bg == 0.0 {
        return 0.0;
    }
    let area: f64 = bands.iter().map(|b| b.area as f64).sum();
    (area / non_bg).min(1.0)
}

fn detect_tolerance_side_by_gap(bands: &[BandInfo]) -> &'static str {
    if bands.len() < 4 {
        return "ambiguous";
    }
    let mut gaps = Vec::new();
    for i in 0..bands.len() - 1 {
        let dx = bands[i + 1].centroid.0 - bands[i].centroid.0;
        let dy = bands[i + 1].centroid.1 - bands[i].centroid.1;
        gaps.push((dx * dx + dy * dy).sqrt());
    }
    if gaps.len() < 3 {
        return "ambiguous";
    }
    let first = gaps[0];
    let last = *gaps.last().unwrap();
    let interior = &gaps[1..gaps.len() - 1];
    let mut sorted = interior.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = if sorted.len() % 2 == 1 {
        sorted[sorted.len() / 2]
    } else {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    };
    if med == 0.0 {
        return "ambiguous";
    }
    let first_q = first >= GAP_RATIO_THRESHOLD * med;
    let last_q = last >= GAP_RATIO_THRESHOLD * med;
    match (first_q, last_q) {
        (false, true) => "forward",
        (true, false) => "reverse",
        (true, true) => {
            if last >= first {
                "forward"
            } else {
                "reverse"
            }
        }
        _ => "ambiguous",
    }
}

fn apply_secondary_heuristics(bands: &[BandInfo]) -> &'static str {
    let first = &bands[0];
    let last = &bands[bands.len() - 1];
    if first.color_name.eq_ignore_ascii_case("black")
        && !last.color_name.eq_ignore_ascii_case("black")
    {
        return "reverse";
    }
    let first_w = first.bounding_box.2 - first.bounding_box.0;
    let last_w = last.bounding_box.2 - last.bounding_box.0;
    if (first_w as f64) < (last_w as f64) * 0.7 {
        return "reverse";
    }
    "forward"
}

fn determine_reading_direction(bands: &[BandInfo]) -> &'static str {
    if bands.is_empty() {
        return "error";
    }
    let n = bands.len();
    let first_color = bands[0].color_name.to_lowercase();
    let last_color = bands[n - 1].color_name.to_lowercase();

    if first_color == "black" || last_color == "black" {
        return "error_black_edge";
    }

    let tol_indices: Vec<usize> = bands
        .iter()
        .enumerate()
        .filter(|(_, b)| {
            let c = b.color_name.to_lowercase();
            c == "gold" || c == "silver"
        })
        .map(|(i, _)| i)
        .collect();

    if tol_indices.len() > 2 {
        return "error_multiple_tolerance_bands";
    }
    if tol_indices.len() == 2 {
        let valid_last = tol_indices == [n - 2, n - 1];
        let valid_first = tol_indices == [0, 1];
        if !(valid_last || valid_first) {
            return "error_multiple_tolerance_bands";
        }
    }
    if tol_indices.len() == 1 {
        let pos = tol_indices[0];
        if pos != 0 && pos != 1 && pos != n - 2 && pos != n - 1 {
            return "error_interior_tolerance_band";
        }
    }

    let first_is_4tol = first_color == "gold" || first_color == "silver";
    let last_is_4tol = last_color == "gold" || last_color == "silver";
    if last_is_4tol && !first_is_4tol {
        return "forward";
    }
    if first_is_4tol && !last_is_4tol {
        return "reverse";
    }
    if first_is_4tol && last_is_4tol {
        return "forward";
    }
    if !tol_indices.is_empty() {
        return apply_secondary_heuristics(bands);
    }

    if n == 5 {
        let five_tol: &[&str] = &["brown", "red", "green", "blue", "violet", "grey"];
        let first_is_5 = five_tol.contains(&first_color.as_str());
        let last_is_5 = five_tol.contains(&last_color.as_str());
        if last_is_5 && !first_is_5 {
            return "forward";
        }
        if first_is_5 && !last_is_5 {
            return "reverse";
        }
        let gap = detect_tolerance_side_by_gap(bands);
        if gap != "ambiguous" {
            return gap;
        }
        let mut gaps = Vec::new();
        for i in 0..n - 1 {
            let dx = bands[i + 1].centroid.0 - bands[i].centroid.0;
            let dy = bands[i + 1].centroid.1 - bands[i].centroid.1;
            gaps.push((dx * dx + dy * dy).sqrt());
        }
        let max_idx = gaps
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        if max_idx == n - 2 {
            return "forward";
        }
        if max_idx == 0 {
            return "reverse";
        }
    }
    apply_secondary_heuristics(bands)
}

fn calculate_4_band(bands: &[BandInfo]) -> DecodeOutcome {
    if bands.len() < 4 {
        return DecodeOutcome::Err(CalculationError {
            error_type: "INSUFFICIENT_BANDS".into(),
            message: format!("4-band calculation requires 4 bands, got {}", bands.len()),
            detected_bands: bands.to_vec(),
        });
    }
    let digits = color_to_digit();
    let mults = color_to_multiplier();
    let tols = color_to_tolerance();
    let c1 = bands[0].color_name.to_lowercase();
    let c2 = bands[1].color_name.to_lowercase();
    let cm = bands[2].color_name.to_lowercase();
    let ct = bands[3].color_name.to_lowercase();
    let Some(d1) = digits.get(c1.as_str()).copied() else {
        return DecodeOutcome::Err(CalculationError {
            error_type: "INVALID_COLOR".into(),
            message: format!("'{c1}' is not a valid digit color for band 1"),
            detected_bands: bands.to_vec(),
        });
    };
    let Some(d2) = digits.get(c2.as_str()).copied() else {
        return DecodeOutcome::Err(CalculationError {
            error_type: "INVALID_COLOR".into(),
            message: format!("'{c2}' is not a valid digit color for band 2"),
            detected_bands: bands.to_vec(),
        });
    };
    let Some(mult) = mults.get(cm.as_str()).copied() else {
        return DecodeOutcome::Err(CalculationError {
            error_type: "INVALID_COLOR".into(),
            message: format!("'{cm}' is not a valid multiplier color for band 3"),
            detected_bands: bands.to_vec(),
        });
    };
    if !VALID_TOLERANCE_COLORS.contains(&ct.as_str()) {
        return DecodeOutcome::Err(CalculationError {
            error_type: "INVALID_TOLERANCE_COLOR".into(),
            message: format!(
                "'{ct}' is not a valid tolerance color — likely a segmentation artifact."
            ),
            detected_bands: bands.to_vec(),
        });
    }
    let tol = *tols.get(ct.as_str()).unwrap_or(&DEFAULT_TOLERANCE);
    DecodeOutcome::Ok(ResistanceResult {
        value: (d1 * 10 + d2) as f64 * mult,
        tolerance: tol,
        bands: bands[..4].to_vec(),
        axis_confidence: 0.0,
        area_confidence: 0.0,
    })
}

fn calculate_5_band(bands: &[BandInfo]) -> DecodeOutcome {
    if bands.len() < 5 {
        return DecodeOutcome::Err(CalculationError {
            error_type: "INSUFFICIENT_BANDS".into(),
            message: format!("5-band calculation requires 5 bands, got {}", bands.len()),
            detected_bands: bands.to_vec(),
        });
    }
    let digits = color_to_digit();
    let mults = color_to_multiplier();
    let tols = color_to_tolerance();
    let c1 = bands[0].color_name.to_lowercase();
    let c2 = bands[1].color_name.to_lowercase();
    let c3 = bands[2].color_name.to_lowercase();
    let cm = bands[3].color_name.to_lowercase();
    let ct = bands[4].color_name.to_lowercase();
    let Some(d1) = digits.get(c1.as_str()).copied() else {
        return DecodeOutcome::Err(CalculationError {
            error_type: "INVALID_COLOR".into(),
            message: format!("'{c1}' is not a valid digit color for band 1"),
            detected_bands: bands.to_vec(),
        });
    };
    let Some(d2) = digits.get(c2.as_str()).copied() else {
        return DecodeOutcome::Err(CalculationError {
            error_type: "INVALID_COLOR".into(),
            message: format!("'{c2}' is not a valid digit color for band 2"),
            detected_bands: bands.to_vec(),
        });
    };
    let Some(d3) = digits.get(c3.as_str()).copied() else {
        return DecodeOutcome::Err(CalculationError {
            error_type: "INVALID_COLOR".into(),
            message: format!("'{c3}' is not a valid digit color for band 3"),
            detected_bands: bands.to_vec(),
        });
    };
    let Some(mult) = mults.get(cm.as_str()).copied() else {
        return DecodeOutcome::Err(CalculationError {
            error_type: "INVALID_COLOR".into(),
            message: format!("'{cm}' is not a valid multiplier color for band 4"),
            detected_bands: bands.to_vec(),
        });
    };
    if !VALID_TOLERANCE_COLORS.contains(&ct.as_str()) {
        return DecodeOutcome::Err(CalculationError {
            error_type: "INVALID_TOLERANCE_COLOR".into(),
            message: format!(
                "'{ct}' is not a valid tolerance color — likely a segmentation artifact."
            ),
            detected_bands: bands.to_vec(),
        });
    }
    let tol = *tols.get(ct.as_str()).unwrap_or(&DEFAULT_TOLERANCE);
    DecodeOutcome::Ok(ResistanceResult {
        value: (d1 * 100 + d2 * 10 + d3) as f64 * mult,
        tolerance: tol,
        bands: bands[..5].to_vec(),
        axis_confidence: 0.0,
        area_confidence: 0.0,
    })
}

fn calculate_3_band(bands: &[BandInfo]) -> DecodeOutcome {
    if bands.len() < 3 {
        return DecodeOutcome::Err(CalculationError {
            error_type: "INSUFFICIENT_BANDS".into(),
            message: format!("3-band calculation requires 3 bands, got {}", bands.len()),
            detected_bands: bands.to_vec(),
        });
    }
    let digits = color_to_digit();
    let mults = color_to_multiplier();
    let c1 = bands[0].color_name.to_lowercase();
    let c2 = bands[1].color_name.to_lowercase();
    let cm = bands[2].color_name.to_lowercase();
    let Some(d1) = digits.get(c1.as_str()).copied() else {
        return DecodeOutcome::Err(CalculationError {
            error_type: "INVALID_COLOR".into(),
            message: format!("'{c1}' is not a valid digit color for band 1"),
            detected_bands: bands.to_vec(),
        });
    };
    let Some(d2) = digits.get(c2.as_str()).copied() else {
        return DecodeOutcome::Err(CalculationError {
            error_type: "INVALID_COLOR".into(),
            message: format!("'{c2}' is not a valid digit color for band 2"),
            detected_bands: bands.to_vec(),
        });
    };
    let Some(mult) = mults.get(cm.as_str()).copied() else {
        return DecodeOutcome::Err(CalculationError {
            error_type: "INVALID_COLOR".into(),
            message: format!("'{cm}' is not a valid multiplier color for band 3"),
            detected_bands: bands.to_vec(),
        });
    };
    DecodeOutcome::Ok(ResistanceResult {
        value: (d1 * 10 + d2) as f64 * mult,
        tolerance: DEFAULT_TOLERANCE,
        bands: bands[..3].to_vec(),
        axis_confidence: 0.0,
        area_confidence: 0.0,
    })
}

fn decode_ordered_bands(bands: &[BandInfo]) -> DecodeOutcome {
    let mut bands = bands.to_vec();
    let direction = determine_reading_direction(&bands);
    let error_map = [
        (
            "error",
            "NO_TOLERANCE_BAND",
            "Could not determine reading direction — no recognisable tolerance band at either end.",
        ),
        (
            "error_black_edge",
            "BLACK_BOUNDARY_BAND",
            "Black band at a boundary position — invalid as leading digit and as tolerance color.",
        ),
        (
            "error_multiple_tolerance_bands",
            "MULTIPLE_TOLERANCE_BANDS",
            "Impossible gold/silver arrangement — likely a segmentation artifact.",
        ),
        (
            "error_interior_tolerance_band",
            "INTERIOR_TOLERANCE_BAND",
            "Gold/silver in a strictly interior position — not valid on any real resistor.",
        ),
    ];
    for (key, et, msg) in error_map {
        if direction == key {
            return DecodeOutcome::Err(CalculationError {
                error_type: et.into(),
                message: msg.into(),
                detected_bands: bands,
            });
        }
    }
    if direction == "reverse" {
        bands.reverse();
    }

    while bands.len() > 3 {
        let c = bands[0].color_name.to_lowercase();
        if c == "gold" || c == "silver" {
            bands.remove(0);
        } else {
            break;
        }
    }
    while bands.len() > 4 {
        let c1 = bands[bands.len() - 1].color_name.to_lowercase();
        let c2 = bands[bands.len() - 2].color_name.to_lowercase();
        if (c1 == "gold" || c1 == "silver") && (c2 == "gold" || c2 == "silver") {
            bands.pop();
        } else {
            break;
        }
    }

    match bands.len() {
        4 => calculate_4_band(&bands),
        5 => calculate_5_band(&bands),
        3 => calculate_3_band(&bands),
        n if n > 5 => calculate_5_band(&bands[..5]),
        _ => DecodeOutcome::Err(CalculationError {
            error_type: "INSUFFICIENT_BANDS".into(),
            message: format!("Need at least 3 bands, found {}", bands.len()),
            detected_bands: bands,
        }),
    }
}

pub fn calculate_resistance_with_axis_info(
    mask: &[u8],
    height: usize,
    width: usize,
) -> (DecodeOutcome, Option<AxisInfo>) {
    let (mut bands, axis_info) = extract_bands_by_projection(mask, height, width);

    if !bands.is_empty() {
        let mut areas: Vec<i32> = bands.iter().map(|b| b.area).collect();
        areas.sort();
        let median = areas[areas.len() / 2];
        // Thin tolerance bands (gold/silver) often sit well below 20% of median
        // digit-band area; keep them if they clear a small absolute floor.
        // Floor matches the adaptive extractor minimum (can be < MIN_BAND_AREA on
        // small live crops).
        const AREA_FLOOR: i32 = 12;
        bands.retain(|b| {
            if b.area < AREA_FLOOR {
                return false;
            }
            matches!(b.color_name.as_str(), "gold" | "silver")
                || b.area as f64 >= 0.15 * median as f64
        });
    }
    bands.retain(|b| b.color_name != "unknown");

    if bands.len() < 3 {
        return (
            DecodeOutcome::Err(CalculationError {
                error_type: "INSUFFICIENT_BANDS".into(),
                message: format!("Need at least 3 bands, found {}", bands.len()),
                detected_bands: bands,
            }),
            axis_info,
        );
    }

    if bands.len() > 5 {
        bands = filter_bands_by_area(&bands, 5);
    }

    let sorted = bands.clone();
    let mut result = decode_ordered_bands(&sorted);

    if let DecodeOutcome::Ok(ref r) = result
        && !is_e24(r.value, 0.01)
        && sorted.len() > 3
    {
        let mut area_asc: Vec<usize> = (0..sorted.len()).collect();
        area_asc.sort_by_key(|&i| sorted[i].area);
        for &drop_i in area_asc.iter().take(2) {
            let candidate: Vec<_> = sorted
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != drop_i)
                .map(|(_, b)| b.clone())
                .collect();
            let alt = decode_ordered_bands(&candidate);
            if let DecodeOutcome::Ok(ref a) = alt
                && is_e24(a.value, 0.01)
            {
                result = alt;
                break;
            }
        }
    }

    let axis_conf = compute_axis_confidence(&sorted);
    let area_conf = compute_area_confidence(mask, &bands);
    if let DecodeOutcome::Ok(ref mut r) = result {
        r.axis_confidence = axis_conf;
        r.area_confidence = area_conf;
    }

    (result, axis_info)
}
