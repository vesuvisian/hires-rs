#![recursion_limit = "512"]

pub mod band;
pub mod detection;
pub mod image_io;
pub mod pipeline;
pub mod preprocess;
pub mod segmentation;
pub mod viz;

pub use burn::prelude::Device;
pub use burn::tensor::DeviceKind;

/// Resolve the Burn device from the active crate features.
pub fn resolve_device() -> Device {
    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    {
        use burn::tensor::DeviceKind;
        return Device::metal(DeviceKind::DefaultDevice);
    }
    #[cfg(feature = "cuda")]
    {
        return Device::cuda(0);
    }
    #[cfg(any(feature = "wgpu", feature = "flex"))]
    {
        Device::default()
    }
    #[cfg(not(any(
        feature = "wgpu",
        feature = "flex",
        feature = "metal",
        feature = "cuda"
    )))]
    compile_error!("Enable at least one of: wgpu, flex, metal, cuda")
}
