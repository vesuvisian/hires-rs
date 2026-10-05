//! Optional numeric check vs a PyTorch dump (see scripts/parity_forward.py).
//!
//! Run with:
//! `cargo test --release --no-default-features --features native-models,flex --test native_parity -- --ignored --nocapture`

#![cfg(feature = "native-models")]

use burn::prelude::*;
use hires_rs::detection::model::Model as DetectionModel;
use hires_rs::segmentation::model::Model as SegmentationModel;

#[test]
#[ignore]
fn native_det_ones_is_finite() {
    let device = Device::default();
    let model = DetectionModel::from_file("weights/detection/best.bpk", &device);
    let x = Tensor::<4>::ones([1, 3, 640, 640], &device);
    let y = model.forward(x);
    assert_eq!(y.dims(), [1, 5, 8400]);
    let vals = y.to_data().as_slice::<f32>().unwrap().to_vec();
    let max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let conf_max = (0..8400)
        .map(|i| vals[4 * 8400 + i])
        .fold(f32::NEG_INFINITY, f32::max);
    eprintln!("det max={max:.6} conf_max={conf_max:.6}");
    assert!(vals.iter().all(|v| v.is_finite()));
}

#[test]
#[ignore]
fn native_seg_ones_is_finite() {
    let device = Device::default();
    let model =
        SegmentationModel::from_file("weights/segmentation/efficientnet-b2_best.bpk", &device);
    let x = Tensor::<4>::ones([1, 3, 64, 64], &device);
    let y = model.forward(x);
    assert_eq!(y.dims(), [1, 13, 64, 64]);
    let vals = y.to_data().as_slice::<f32>().unwrap().to_vec();
    let max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    eprintln!("seg max={max:.6}");
    assert!(vals.iter().all(|v| v.is_finite()));
}
