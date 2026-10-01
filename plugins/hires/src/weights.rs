//! Model weights compiled into the plugin dylib.

pub const DET_WEIGHTS: &[u8] = include_bytes!("../../../weights/detection/best.bpk");
pub const SEG_WEIGHTS: &[u8] =
    include_bytes!("../../../weights/segmentation/efficientnet-b2_best.bpk");
