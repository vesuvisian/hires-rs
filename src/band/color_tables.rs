use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct BandInfo {
    #[allow(dead_code)]
    pub class_id: i32,
    pub color_name: String,
    pub centroid: (f64, f64),
    pub area: i32,
    pub bounding_box: (i32, i32, i32, i32),
}

#[derive(Clone, Debug)]
pub struct ResistanceResult {
    pub value: f64,
    pub tolerance: f64,
    pub bands: Vec<BandInfo>,
    #[allow(dead_code)]
    pub axis_confidence: f64,
    #[allow(dead_code)]
    pub area_confidence: f64,
}

impl ResistanceResult {
    pub fn formatted(&self) -> String {
        format_resistance(self.value)
    }
}

#[derive(Clone, Debug)]
pub struct CalculationError {
    pub error_type: String,
    pub message: String,
    pub detected_bands: Vec<BandInfo>,
}

pub const ID_TO_COLOR_NAME: [&str; 13] = [
    "background",
    "black",
    "blue",
    "brown",
    "gold",
    "green",
    "grey",
    "orange",
    "violet",
    "red",
    "silver",
    "white",
    "yellow",
];

pub const DEFAULT_TOLERANCE: f64 = 20.0;
pub const MIN_BAND_AREA: i32 = 40;

pub fn get_color_name(class_id: i32) -> String {
    if (0..13).contains(&class_id) {
        ID_TO_COLOR_NAME[class_id as usize].to_string()
    } else {
        "unknown".to_string()
    }
}

pub fn color_to_digit() -> HashMap<&'static str, i32> {
    HashMap::from([
        ("black", 0),
        ("brown", 1),
        ("red", 2),
        ("orange", 3),
        ("yellow", 4),
        ("green", 5),
        ("blue", 6),
        ("violet", 7),
        ("grey", 8),
        ("white", 9),
    ])
}

pub fn color_to_multiplier() -> HashMap<&'static str, f64> {
    HashMap::from([
        ("black", 1.0),
        ("brown", 10.0),
        ("red", 100.0),
        ("orange", 1_000.0),
        ("yellow", 10_000.0),
        ("green", 100_000.0),
        ("blue", 1_000_000.0),
        ("violet", 10_000_000.0),
        ("grey", 100_000_000.0),
        ("white", 1_000_000_000.0),
        ("gold", 0.1),
        ("silver", 0.01),
    ])
}

pub fn color_to_tolerance() -> HashMap<&'static str, f64> {
    HashMap::from([
        ("brown", 1.0),
        ("red", 2.0),
        ("green", 0.5),
        ("blue", 0.25),
        ("violet", 0.1),
        ("grey", 0.05),
        ("gold", 5.0),
        ("silver", 10.0),
    ])
}

pub fn format_resistance(value: f64) -> String {
    let (formatted, unit) = if value >= 1_000_000.0 {
        (value / 1_000_000.0, "MΩ")
    } else if value >= 1_000.0 {
        (value / 1_000.0, "kΩ")
    } else if value >= 1.0 {
        (value, "Ω")
    } else {
        (value * 1000.0, "mΩ")
    };

    if (formatted - formatted.round()).abs() < 1e-9 {
        format!("{} {}", formatted as i64, unit)
    } else if (formatted * 10.0 - (formatted * 10.0).round()).abs() < 1e-9 {
        format!("{:.1} {}", formatted, unit)
    } else {
        format!("{:.2} {}", formatted, unit)
    }
}
