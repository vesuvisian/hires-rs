//! Model weights compiled into the plugin dylib.

#[cfg(feature = "onnx-models")]
pub const DET_WEIGHTS: &[u8] = include_bytes!("../../../weights/detection/best.onnx.bpk");
#[cfg(feature = "onnx-models")]
pub const SEG_WEIGHTS: &[u8] =
    include_bytes!("../../../weights/segmentation/efficientnet-b2_best.onnx.bpk");

#[cfg(feature = "native-models")]
pub const DET_WEIGHTS: &[u8] = include_bytes!("../../../weights/detection/best.bpk");
#[cfg(feature = "native-models")]
pub const SEG_WEIGHTS: &[u8] =
    include_bytes!("../../../weights/segmentation/efficientnet-b2_best.bpk");
