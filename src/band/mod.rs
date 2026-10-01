pub mod color_tables;
pub mod extractor;
pub mod resistance;

pub use color_tables::*;
pub use extractor::AxisInfo;
pub use resistance::{DecodeOutcome, calculate_resistance_with_axis_info};
