#![allow(clippy::all)]
#![allow(dead_code, unused_imports, unused_variables)]
// Generated from ONNX by burn-onnx — do not hand-edit.

extern crate alloc;
use burn::nn::PaddingConfig2d;
use burn::nn::conv::Conv2d;
use burn::nn::conv::Conv2dConfig;
use burn::prelude::*;
use burn::tensor::Bytes;
use burn_store::BurnpackStore;
use burn_store::ModuleSnapshot;

#[derive(Module, Debug)]
pub struct Submodule1 {
    conv2d1: Conv2d,
    conv2d2: Conv2d,
    conv2d3: Conv2d,
    conv2d4: Conv2d,
    conv2d5: Conv2d,
    conv2d6: Conv2d,
    conv2d7: Conv2d,
    conv2d8: Conv2d,
    conv2d9: Conv2d,
    conv2d10: Conv2d,
    conv2d11: Conv2d,
    conv2d12: Conv2d,
    conv2d13: Conv2d,
    conv2d14: Conv2d,
    conv2d15: Conv2d,
    conv2d16: Conv2d,
    conv2d17: Conv2d,
    conv2d18: Conv2d,
    conv2d19: Conv2d,
    conv2d20: Conv2d,
    conv2d21: Conv2d,
    conv2d22: Conv2d,
    conv2d23: Conv2d,
    conv2d24: Conv2d,
    #[module(skip)]
    device: Device,
}
impl Submodule1 {
    #[allow(unused_variables)]
    pub fn new(device: &Device) -> Self {
        let conv2d1 = Conv2dConfig::new([3, 32], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(0, 0, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d2 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(32)
            .with_bias(true)
            .init(device);
        let conv2d3 = Conv2dConfig::new([32, 8], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d4 = Conv2dConfig::new([8, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d5 = Conv2dConfig::new([32, 16], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d6 = Conv2dConfig::new([16, 16], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(16)
            .with_bias(true)
            .init(device);
        let conv2d7 = Conv2dConfig::new([16, 4], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d8 = Conv2dConfig::new([4, 16], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d9 = Conv2dConfig::new([16, 16], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d10 = Conv2dConfig::new([16, 96], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d11 = Conv2dConfig::new([96, 96], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(0, 0, 1, 1))
            .with_dilation([1, 1])
            .with_groups(96)
            .with_bias(true)
            .init(device);
        let conv2d12 = Conv2dConfig::new([96, 4], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d13 = Conv2dConfig::new([4, 96], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d14 = Conv2dConfig::new([96, 24], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d15 = Conv2dConfig::new([24, 144], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d16 = Conv2dConfig::new([144, 144], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(144)
            .with_bias(true)
            .init(device);
        let conv2d17 = Conv2dConfig::new([144, 6], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d18 = Conv2dConfig::new([6, 144], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d19 = Conv2dConfig::new([144, 24], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d20 = Conv2dConfig::new([24, 144], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d21 = Conv2dConfig::new([144, 144], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(144)
            .with_bias(true)
            .init(device);
        let conv2d22 = Conv2dConfig::new([144, 6], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d23 = Conv2dConfig::new([6, 144], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d24 = Conv2dConfig::new([144, 24], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        Self {
            conv2d1,
            conv2d2,
            conv2d3,
            conv2d4,
            conv2d5,
            conv2d6,
            conv2d7,
            conv2d8,
            conv2d9,
            conv2d10,
            conv2d11,
            conv2d12,
            conv2d13,
            conv2d14,
            conv2d15,
            conv2d16,
            conv2d17,
            conv2d18,
            conv2d19,
            conv2d20,
            conv2d21,
            conv2d22,
            conv2d23,
            conv2d24,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, input: Tensor<4>) -> (Tensor<4>, Tensor<4>) {
        let conv2d1_out1 = self.conv2d1.forward(input);
        let sigmoid1_out1 = burn::tensor::activation::sigmoid(conv2d1_out1.clone());
        let mul1_out1 = conv2d1_out1.mul(sigmoid1_out1);
        let conv2d2_out1 = self.conv2d2.forward(mul1_out1.clone());
        let sigmoid2_out1 = burn::tensor::activation::sigmoid(conv2d2_out1.clone());
        let mul2_out1 = conv2d2_out1.mul(sigmoid2_out1);
        let reducemean1_out1 = { mul2_out1.clone().mean_dim(2usize).mean_dim(3usize) };
        let conv2d3_out1 = self.conv2d3.forward(reducemean1_out1);
        let sigmoid3_out1 = burn::tensor::activation::sigmoid(conv2d3_out1.clone());
        let mul3_out1 = conv2d3_out1.mul(sigmoid3_out1);
        let conv2d4_out1 = self.conv2d4.forward(mul3_out1);
        let sigmoid4_out1 = burn::tensor::activation::sigmoid(conv2d4_out1);
        let mul4_out1 = sigmoid4_out1.mul(mul2_out1);
        let conv2d5_out1 = self.conv2d5.forward(mul4_out1);
        let conv2d6_out1 = self.conv2d6.forward(conv2d5_out1.clone());
        let sigmoid5_out1 = burn::tensor::activation::sigmoid(conv2d6_out1.clone());
        let mul5_out1 = conv2d6_out1.mul(sigmoid5_out1);
        let reducemean2_out1 = { mul5_out1.clone().mean_dim(2usize).mean_dim(3usize) };
        let conv2d7_out1 = self.conv2d7.forward(reducemean2_out1);
        let sigmoid6_out1 = burn::tensor::activation::sigmoid(conv2d7_out1.clone());
        let mul6_out1 = conv2d7_out1.mul(sigmoid6_out1);
        let conv2d8_out1 = self.conv2d8.forward(mul6_out1);
        let sigmoid7_out1 = burn::tensor::activation::sigmoid(conv2d8_out1);
        let mul7_out1 = sigmoid7_out1.mul(mul5_out1);
        let conv2d9_out1 = self.conv2d9.forward(mul7_out1);
        let add1_out1 = conv2d9_out1.add(conv2d5_out1);
        let conv2d10_out1 = self.conv2d10.forward(add1_out1);
        let sigmoid8_out1 = burn::tensor::activation::sigmoid(conv2d10_out1.clone());
        let mul8_out1 = conv2d10_out1.mul(sigmoid8_out1);
        let conv2d11_out1 = self.conv2d11.forward(mul8_out1);
        let sigmoid9_out1 = burn::tensor::activation::sigmoid(conv2d11_out1.clone());
        let mul9_out1 = conv2d11_out1.mul(sigmoid9_out1);
        let reducemean3_out1 = { mul9_out1.clone().mean_dim(2usize).mean_dim(3usize) };
        let conv2d12_out1 = self.conv2d12.forward(reducemean3_out1);
        let sigmoid10_out1 = burn::tensor::activation::sigmoid(conv2d12_out1.clone());
        let mul10_out1 = conv2d12_out1.mul(sigmoid10_out1);
        let conv2d13_out1 = self.conv2d13.forward(mul10_out1);
        let sigmoid11_out1 = burn::tensor::activation::sigmoid(conv2d13_out1);
        let mul11_out1 = sigmoid11_out1.mul(mul9_out1);
        let conv2d14_out1 = self.conv2d14.forward(mul11_out1);
        let conv2d15_out1 = self.conv2d15.forward(conv2d14_out1.clone());
        let sigmoid12_out1 = burn::tensor::activation::sigmoid(conv2d15_out1.clone());
        let mul12_out1 = conv2d15_out1.mul(sigmoid12_out1);
        let conv2d16_out1 = self.conv2d16.forward(mul12_out1);
        let sigmoid13_out1 = burn::tensor::activation::sigmoid(conv2d16_out1.clone());
        let mul13_out1 = conv2d16_out1.mul(sigmoid13_out1);
        let reducemean4_out1 = { mul13_out1.clone().mean_dim(2usize).mean_dim(3usize) };
        let conv2d17_out1 = self.conv2d17.forward(reducemean4_out1);
        let sigmoid14_out1 = burn::tensor::activation::sigmoid(conv2d17_out1.clone());
        let mul14_out1 = conv2d17_out1.mul(sigmoid14_out1);
        let conv2d18_out1 = self.conv2d18.forward(mul14_out1);
        let sigmoid15_out1 = burn::tensor::activation::sigmoid(conv2d18_out1);
        let mul15_out1 = sigmoid15_out1.mul(mul13_out1);
        let conv2d19_out1 = self.conv2d19.forward(mul15_out1);
        let add2_out1 = conv2d19_out1.add(conv2d14_out1);
        let conv2d20_out1 = self.conv2d20.forward(add2_out1.clone());
        let sigmoid16_out1 = burn::tensor::activation::sigmoid(conv2d20_out1.clone());
        let mul16_out1 = conv2d20_out1.mul(sigmoid16_out1);
        let conv2d21_out1 = self.conv2d21.forward(mul16_out1);
        let sigmoid17_out1 = burn::tensor::activation::sigmoid(conv2d21_out1.clone());
        let mul17_out1 = conv2d21_out1.mul(sigmoid17_out1);
        let reducemean5_out1 = { mul17_out1.clone().mean_dim(2usize).mean_dim(3usize) };
        let conv2d22_out1 = self.conv2d22.forward(reducemean5_out1);
        let sigmoid18_out1 = burn::tensor::activation::sigmoid(conv2d22_out1.clone());
        let mul18_out1 = conv2d22_out1.mul(sigmoid18_out1);
        let conv2d23_out1 = self.conv2d23.forward(mul18_out1);
        let sigmoid19_out1 = burn::tensor::activation::sigmoid(conv2d23_out1);
        let mul19_out1 = sigmoid19_out1.mul(mul17_out1);
        let conv2d24_out1 = self.conv2d24.forward(mul19_out1);
        let add3_out1 = conv2d24_out1.add(add2_out1);
        (add3_out1, mul1_out1)
    }
}
#[derive(Module, Debug)]
pub struct Submodule2 {
    conv2d25: Conv2d,
    conv2d26: Conv2d,
    conv2d27: Conv2d,
    conv2d28: Conv2d,
    conv2d29: Conv2d,
    conv2d30: Conv2d,
    conv2d31: Conv2d,
    conv2d32: Conv2d,
    conv2d33: Conv2d,
    conv2d34: Conv2d,
    conv2d35: Conv2d,
    conv2d36: Conv2d,
    conv2d37: Conv2d,
    conv2d38: Conv2d,
    conv2d39: Conv2d,
    conv2d40: Conv2d,
    conv2d41: Conv2d,
    conv2d42: Conv2d,
    conv2d43: Conv2d,
    conv2d44: Conv2d,
    conv2d45: Conv2d,
    conv2d46: Conv2d,
    conv2d47: Conv2d,
    conv2d48: Conv2d,
    conv2d49: Conv2d,
    #[module(skip)]
    device: Device,
}
impl Submodule2 {
    #[allow(unused_variables)]
    pub fn new(device: &Device) -> Self {
        let conv2d25 = Conv2dConfig::new([24, 144], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d26 = Conv2dConfig::new([144, 144], [5, 5])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(144)
            .with_bias(true)
            .init(device);
        let conv2d27 = Conv2dConfig::new([144, 6], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d28 = Conv2dConfig::new([6, 144], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d29 = Conv2dConfig::new([144, 48], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d30 = Conv2dConfig::new([48, 288], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d31 = Conv2dConfig::new([288, 288], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(288)
            .with_bias(true)
            .init(device);
        let conv2d32 = Conv2dConfig::new([288, 12], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d33 = Conv2dConfig::new([12, 288], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d34 = Conv2dConfig::new([288, 48], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d35 = Conv2dConfig::new([48, 288], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d36 = Conv2dConfig::new([288, 288], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(288)
            .with_bias(true)
            .init(device);
        let conv2d37 = Conv2dConfig::new([288, 12], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d38 = Conv2dConfig::new([12, 288], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d39 = Conv2dConfig::new([288, 48], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d40 = Conv2dConfig::new([48, 288], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d41 = Conv2dConfig::new([288, 288], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(288)
            .with_bias(true)
            .init(device);
        let conv2d42 = Conv2dConfig::new([288, 12], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d43 = Conv2dConfig::new([12, 288], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d44 = Conv2dConfig::new([288, 88], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d45 = Conv2dConfig::new([88, 528], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d46 = Conv2dConfig::new([528, 528], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(528)
            .with_bias(true)
            .init(device);
        let conv2d47 = Conv2dConfig::new([528, 22], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d48 = Conv2dConfig::new([22, 528], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d49 = Conv2dConfig::new([528, 88], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        Self {
            conv2d25,
            conv2d26,
            conv2d27,
            conv2d28,
            conv2d29,
            conv2d30,
            conv2d31,
            conv2d32,
            conv2d33,
            conv2d34,
            conv2d35,
            conv2d36,
            conv2d37,
            conv2d38,
            conv2d39,
            conv2d40,
            conv2d41,
            conv2d42,
            conv2d43,
            conv2d44,
            conv2d45,
            conv2d46,
            conv2d47,
            conv2d48,
            conv2d49,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, add3_out1: Tensor<4>) -> (Tensor<4>, Tensor<4>) {
        let conv2d25_out1 = self.conv2d25.forward(add3_out1);
        let sigmoid20_out1 = burn::tensor::activation::sigmoid(conv2d25_out1.clone());
        let mul20_out1 = conv2d25_out1.mul(sigmoid20_out1);
        let conv2d26_out1 = self.conv2d26.forward(mul20_out1);
        let sigmoid21_out1 = burn::tensor::activation::sigmoid(conv2d26_out1.clone());
        let mul21_out1 = conv2d26_out1.mul(sigmoid21_out1);
        let reducemean6_out1 = { mul21_out1.clone().mean_dim(2usize).mean_dim(3usize) };
        let conv2d27_out1 = self.conv2d27.forward(reducemean6_out1);
        let sigmoid22_out1 = burn::tensor::activation::sigmoid(conv2d27_out1.clone());
        let mul22_out1 = conv2d27_out1.mul(sigmoid22_out1);
        let conv2d28_out1 = self.conv2d28.forward(mul22_out1);
        let sigmoid23_out1 = burn::tensor::activation::sigmoid(conv2d28_out1);
        let mul23_out1 = sigmoid23_out1.mul(mul21_out1);
        let conv2d29_out1 = self.conv2d29.forward(mul23_out1);
        let conv2d30_out1 = self.conv2d30.forward(conv2d29_out1.clone());
        let sigmoid24_out1 = burn::tensor::activation::sigmoid(conv2d30_out1.clone());
        let mul24_out1 = conv2d30_out1.mul(sigmoid24_out1);
        let conv2d31_out1 = self.conv2d31.forward(mul24_out1);
        let sigmoid25_out1 = burn::tensor::activation::sigmoid(conv2d31_out1.clone());
        let mul25_out1 = conv2d31_out1.mul(sigmoid25_out1);
        let reducemean7_out1 = { mul25_out1.clone().mean_dim(2usize).mean_dim(3usize) };
        let conv2d32_out1 = self.conv2d32.forward(reducemean7_out1);
        let sigmoid26_out1 = burn::tensor::activation::sigmoid(conv2d32_out1.clone());
        let mul26_out1 = conv2d32_out1.mul(sigmoid26_out1);
        let conv2d33_out1 = self.conv2d33.forward(mul26_out1);
        let sigmoid27_out1 = burn::tensor::activation::sigmoid(conv2d33_out1);
        let mul27_out1 = sigmoid27_out1.mul(mul25_out1);
        let conv2d34_out1 = self.conv2d34.forward(mul27_out1);
        let add4_out1 = conv2d34_out1.add(conv2d29_out1);
        let conv2d35_out1 = self.conv2d35.forward(add4_out1.clone());
        let sigmoid28_out1 = burn::tensor::activation::sigmoid(conv2d35_out1.clone());
        let mul28_out1 = conv2d35_out1.mul(sigmoid28_out1);
        let conv2d36_out1 = self.conv2d36.forward(mul28_out1);
        let sigmoid29_out1 = burn::tensor::activation::sigmoid(conv2d36_out1.clone());
        let mul29_out1 = conv2d36_out1.mul(sigmoid29_out1);
        let reducemean8_out1 = { mul29_out1.clone().mean_dim(2usize).mean_dim(3usize) };
        let conv2d37_out1 = self.conv2d37.forward(reducemean8_out1);
        let sigmoid30_out1 = burn::tensor::activation::sigmoid(conv2d37_out1.clone());
        let mul30_out1 = conv2d37_out1.mul(sigmoid30_out1);
        let conv2d38_out1 = self.conv2d38.forward(mul30_out1);
        let sigmoid31_out1 = burn::tensor::activation::sigmoid(conv2d38_out1);
        let mul31_out1 = sigmoid31_out1.mul(mul29_out1);
        let conv2d39_out1 = self.conv2d39.forward(mul31_out1);
        let add5_out1 = conv2d39_out1.add(add4_out1);
        let conv2d40_out1 = self.conv2d40.forward(add5_out1.clone());
        let sigmoid32_out1 = burn::tensor::activation::sigmoid(conv2d40_out1.clone());
        let mul32_out1 = conv2d40_out1.mul(sigmoid32_out1);
        let conv2d41_out1 = self.conv2d41.forward(mul32_out1);
        let sigmoid33_out1 = burn::tensor::activation::sigmoid(conv2d41_out1.clone());
        let mul33_out1 = conv2d41_out1.mul(sigmoid33_out1);
        let reducemean9_out1 = { mul33_out1.clone().mean_dim(2usize).mean_dim(3usize) };
        let conv2d42_out1 = self.conv2d42.forward(reducemean9_out1);
        let sigmoid34_out1 = burn::tensor::activation::sigmoid(conv2d42_out1.clone());
        let mul34_out1 = conv2d42_out1.mul(sigmoid34_out1);
        let conv2d43_out1 = self.conv2d43.forward(mul34_out1);
        let sigmoid35_out1 = burn::tensor::activation::sigmoid(conv2d43_out1);
        let mul35_out1 = sigmoid35_out1.mul(mul33_out1);
        let conv2d44_out1 = self.conv2d44.forward(mul35_out1);
        let conv2d45_out1 = self.conv2d45.forward(conv2d44_out1.clone());
        let sigmoid36_out1 = burn::tensor::activation::sigmoid(conv2d45_out1.clone());
        let mul36_out1 = conv2d45_out1.mul(sigmoid36_out1);
        let conv2d46_out1 = self.conv2d46.forward(mul36_out1);
        let sigmoid37_out1 = burn::tensor::activation::sigmoid(conv2d46_out1.clone());
        let mul37_out1 = conv2d46_out1.mul(sigmoid37_out1);
        let reducemean10_out1 = { mul37_out1.clone().mean_dim(2usize).mean_dim(3usize) };
        let conv2d47_out1 = self.conv2d47.forward(reducemean10_out1);
        let sigmoid38_out1 = burn::tensor::activation::sigmoid(conv2d47_out1.clone());
        let mul38_out1 = conv2d47_out1.mul(sigmoid38_out1);
        let conv2d48_out1 = self.conv2d48.forward(mul38_out1);
        let sigmoid39_out1 = burn::tensor::activation::sigmoid(conv2d48_out1);
        let mul39_out1 = sigmoid39_out1.mul(mul37_out1);
        let conv2d49_out1 = self.conv2d49.forward(mul39_out1);
        let add6_out1 = conv2d49_out1.add(conv2d44_out1);
        (add6_out1, add5_out1)
    }
}
#[derive(Module, Debug)]
pub struct Submodule3 {
    conv2d50: Conv2d,
    conv2d51: Conv2d,
    conv2d52: Conv2d,
    conv2d53: Conv2d,
    conv2d54: Conv2d,
    conv2d55: Conv2d,
    conv2d56: Conv2d,
    conv2d57: Conv2d,
    conv2d58: Conv2d,
    conv2d59: Conv2d,
    conv2d60: Conv2d,
    conv2d61: Conv2d,
    conv2d62: Conv2d,
    conv2d63: Conv2d,
    conv2d64: Conv2d,
    conv2d65: Conv2d,
    conv2d66: Conv2d,
    conv2d67: Conv2d,
    conv2d68: Conv2d,
    conv2d69: Conv2d,
    conv2d70: Conv2d,
    conv2d71: Conv2d,
    conv2d72: Conv2d,
    conv2d73: Conv2d,
    conv2d74: Conv2d,
    #[module(skip)]
    device: Device,
}
impl Submodule3 {
    #[allow(unused_variables)]
    pub fn new(device: &Device) -> Self {
        let conv2d50 = Conv2dConfig::new([88, 528], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d51 = Conv2dConfig::new([528, 528], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(528)
            .with_bias(true)
            .init(device);
        let conv2d52 = Conv2dConfig::new([528, 22], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d53 = Conv2dConfig::new([22, 528], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d54 = Conv2dConfig::new([528, 88], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d55 = Conv2dConfig::new([88, 528], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d56 = Conv2dConfig::new([528, 528], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(528)
            .with_bias(true)
            .init(device);
        let conv2d57 = Conv2dConfig::new([528, 22], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d58 = Conv2dConfig::new([22, 528], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d59 = Conv2dConfig::new([528, 88], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d60 = Conv2dConfig::new([88, 528], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d61 = Conv2dConfig::new([528, 528], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(528)
            .with_bias(true)
            .init(device);
        let conv2d62 = Conv2dConfig::new([528, 22], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d63 = Conv2dConfig::new([22, 528], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d64 = Conv2dConfig::new([528, 120], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d65 = Conv2dConfig::new([120, 720], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d66 = Conv2dConfig::new([720, 720], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(720)
            .with_bias(true)
            .init(device);
        let conv2d67 = Conv2dConfig::new([720, 30], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d68 = Conv2dConfig::new([30, 720], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d69 = Conv2dConfig::new([720, 120], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d70 = Conv2dConfig::new([120, 720], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d71 = Conv2dConfig::new([720, 720], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(720)
            .with_bias(true)
            .init(device);
        let conv2d72 = Conv2dConfig::new([720, 30], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d73 = Conv2dConfig::new([30, 720], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d74 = Conv2dConfig::new([720, 120], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        Self {
            conv2d50,
            conv2d51,
            conv2d52,
            conv2d53,
            conv2d54,
            conv2d55,
            conv2d56,
            conv2d57,
            conv2d58,
            conv2d59,
            conv2d60,
            conv2d61,
            conv2d62,
            conv2d63,
            conv2d64,
            conv2d65,
            conv2d66,
            conv2d67,
            conv2d68,
            conv2d69,
            conv2d70,
            conv2d71,
            conv2d72,
            conv2d73,
            conv2d74,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, add6_out1: Tensor<4>) -> Tensor<4> {
        let conv2d50_out1 = self.conv2d50.forward(add6_out1.clone());
        let sigmoid40_out1 = burn::tensor::activation::sigmoid(conv2d50_out1.clone());
        let mul40_out1 = conv2d50_out1.mul(sigmoid40_out1);
        let conv2d51_out1 = self.conv2d51.forward(mul40_out1);
        let sigmoid41_out1 = burn::tensor::activation::sigmoid(conv2d51_out1.clone());
        let mul41_out1 = conv2d51_out1.mul(sigmoid41_out1);
        let reducemean11_out1 = { mul41_out1.clone().mean_dim(2usize).mean_dim(3usize) };
        let conv2d52_out1 = self.conv2d52.forward(reducemean11_out1);
        let sigmoid42_out1 = burn::tensor::activation::sigmoid(conv2d52_out1.clone());
        let mul42_out1 = conv2d52_out1.mul(sigmoid42_out1);
        let conv2d53_out1 = self.conv2d53.forward(mul42_out1);
        let sigmoid43_out1 = burn::tensor::activation::sigmoid(conv2d53_out1);
        let mul43_out1 = sigmoid43_out1.mul(mul41_out1);
        let conv2d54_out1 = self.conv2d54.forward(mul43_out1);
        let add7_out1 = conv2d54_out1.add(add6_out1);
        let conv2d55_out1 = self.conv2d55.forward(add7_out1.clone());
        let sigmoid44_out1 = burn::tensor::activation::sigmoid(conv2d55_out1.clone());
        let mul44_out1 = conv2d55_out1.mul(sigmoid44_out1);
        let conv2d56_out1 = self.conv2d56.forward(mul44_out1);
        let sigmoid45_out1 = burn::tensor::activation::sigmoid(conv2d56_out1.clone());
        let mul45_out1 = conv2d56_out1.mul(sigmoid45_out1);
        let reducemean12_out1 = { mul45_out1.clone().mean_dim(2usize).mean_dim(3usize) };
        let conv2d57_out1 = self.conv2d57.forward(reducemean12_out1);
        let sigmoid46_out1 = burn::tensor::activation::sigmoid(conv2d57_out1.clone());
        let mul46_out1 = conv2d57_out1.mul(sigmoid46_out1);
        let conv2d58_out1 = self.conv2d58.forward(mul46_out1);
        let sigmoid47_out1 = burn::tensor::activation::sigmoid(conv2d58_out1);
        let mul47_out1 = sigmoid47_out1.mul(mul45_out1);
        let conv2d59_out1 = self.conv2d59.forward(mul47_out1);
        let add8_out1 = conv2d59_out1.add(add7_out1);
        let conv2d60_out1 = self.conv2d60.forward(add8_out1);
        let sigmoid48_out1 = burn::tensor::activation::sigmoid(conv2d60_out1.clone());
        let mul48_out1 = conv2d60_out1.mul(sigmoid48_out1);
        let conv2d61_out1 = self.conv2d61.forward(mul48_out1);
        let sigmoid49_out1 = burn::tensor::activation::sigmoid(conv2d61_out1.clone());
        let mul49_out1 = conv2d61_out1.mul(sigmoid49_out1);
        let reducemean13_out1 = { mul49_out1.clone().mean_dim(2usize).mean_dim(3usize) };
        let conv2d62_out1 = self.conv2d62.forward(reducemean13_out1);
        let sigmoid50_out1 = burn::tensor::activation::sigmoid(conv2d62_out1.clone());
        let mul50_out1 = conv2d62_out1.mul(sigmoid50_out1);
        let conv2d63_out1 = self.conv2d63.forward(mul50_out1);
        let sigmoid51_out1 = burn::tensor::activation::sigmoid(conv2d63_out1);
        let mul51_out1 = sigmoid51_out1.mul(mul49_out1);
        let conv2d64_out1 = self.conv2d64.forward(mul51_out1);
        let conv2d65_out1 = self.conv2d65.forward(conv2d64_out1.clone());
        let sigmoid52_out1 = burn::tensor::activation::sigmoid(conv2d65_out1.clone());
        let mul52_out1 = conv2d65_out1.mul(sigmoid52_out1);
        let conv2d66_out1 = self.conv2d66.forward(mul52_out1);
        let sigmoid53_out1 = burn::tensor::activation::sigmoid(conv2d66_out1.clone());
        let mul53_out1 = conv2d66_out1.mul(sigmoid53_out1);
        let reducemean14_out1 = { mul53_out1.clone().mean_dim(2usize).mean_dim(3usize) };
        let conv2d67_out1 = self.conv2d67.forward(reducemean14_out1);
        let sigmoid54_out1 = burn::tensor::activation::sigmoid(conv2d67_out1.clone());
        let mul54_out1 = conv2d67_out1.mul(sigmoid54_out1);
        let conv2d68_out1 = self.conv2d68.forward(mul54_out1);
        let sigmoid55_out1 = burn::tensor::activation::sigmoid(conv2d68_out1);
        let mul55_out1 = sigmoid55_out1.mul(mul53_out1);
        let conv2d69_out1 = self.conv2d69.forward(mul55_out1);
        let add9_out1 = conv2d69_out1.add(conv2d64_out1);
        let conv2d70_out1 = self.conv2d70.forward(add9_out1.clone());
        let sigmoid56_out1 = burn::tensor::activation::sigmoid(conv2d70_out1.clone());
        let mul56_out1 = conv2d70_out1.mul(sigmoid56_out1);
        let conv2d71_out1 = self.conv2d71.forward(mul56_out1);
        let sigmoid57_out1 = burn::tensor::activation::sigmoid(conv2d71_out1.clone());
        let mul57_out1 = conv2d71_out1.mul(sigmoid57_out1);
        let reducemean15_out1 = { mul57_out1.clone().mean_dim(2usize).mean_dim(3usize) };
        let conv2d72_out1 = self.conv2d72.forward(reducemean15_out1);
        let sigmoid58_out1 = burn::tensor::activation::sigmoid(conv2d72_out1.clone());
        let mul58_out1 = conv2d72_out1.mul(sigmoid58_out1);
        let conv2d73_out1 = self.conv2d73.forward(mul58_out1);
        let sigmoid59_out1 = burn::tensor::activation::sigmoid(conv2d73_out1);
        let mul59_out1 = sigmoid59_out1.mul(mul57_out1);
        let conv2d74_out1 = self.conv2d74.forward(mul59_out1);
        let add10_out1 = conv2d74_out1.add(add9_out1);
        add10_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule4 {
    conv2d75: Conv2d,
    conv2d76: Conv2d,
    conv2d77: Conv2d,
    conv2d78: Conv2d,
    conv2d79: Conv2d,
    conv2d80: Conv2d,
    conv2d81: Conv2d,
    conv2d82: Conv2d,
    conv2d83: Conv2d,
    conv2d84: Conv2d,
    conv2d85: Conv2d,
    conv2d86: Conv2d,
    conv2d87: Conv2d,
    conv2d88: Conv2d,
    conv2d89: Conv2d,
    conv2d90: Conv2d,
    conv2d91: Conv2d,
    conv2d92: Conv2d,
    conv2d93: Conv2d,
    conv2d94: Conv2d,
    conv2d95: Conv2d,
    conv2d96: Conv2d,
    conv2d97: Conv2d,
    conv2d98: Conv2d,
    conv2d99: Conv2d,
    #[module(skip)]
    device: Device,
}
impl Submodule4 {
    #[allow(unused_variables)]
    pub fn new(device: &Device) -> Self {
        let conv2d75 = Conv2dConfig::new([120, 720], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d76 = Conv2dConfig::new([720, 720], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(720)
            .with_bias(true)
            .init(device);
        let conv2d77 = Conv2dConfig::new([720, 30], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d78 = Conv2dConfig::new([30, 720], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d79 = Conv2dConfig::new([720, 120], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d80 = Conv2dConfig::new([120, 720], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d81 = Conv2dConfig::new([720, 720], [5, 5])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(720)
            .with_bias(true)
            .init(device);
        let conv2d82 = Conv2dConfig::new([720, 30], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d83 = Conv2dConfig::new([30, 720], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d84 = Conv2dConfig::new([720, 208], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d85 = Conv2dConfig::new([208, 1248], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d86 = Conv2dConfig::new([1248, 1248], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(1248)
            .with_bias(true)
            .init(device);
        let conv2d87 = Conv2dConfig::new([1248, 52], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d88 = Conv2dConfig::new([52, 1248], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d89 = Conv2dConfig::new([1248, 208], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d90 = Conv2dConfig::new([208, 1248], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d91 = Conv2dConfig::new([1248, 1248], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(1248)
            .with_bias(true)
            .init(device);
        let conv2d92 = Conv2dConfig::new([1248, 52], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d93 = Conv2dConfig::new([52, 1248], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d94 = Conv2dConfig::new([1248, 208], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d95 = Conv2dConfig::new([208, 1248], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d96 = Conv2dConfig::new([1248, 1248], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(1248)
            .with_bias(true)
            .init(device);
        let conv2d97 = Conv2dConfig::new([1248, 52], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d98 = Conv2dConfig::new([52, 1248], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d99 = Conv2dConfig::new([1248, 208], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        Self {
            conv2d75,
            conv2d76,
            conv2d77,
            conv2d78,
            conv2d79,
            conv2d80,
            conv2d81,
            conv2d82,
            conv2d83,
            conv2d84,
            conv2d85,
            conv2d86,
            conv2d87,
            conv2d88,
            conv2d89,
            conv2d90,
            conv2d91,
            conv2d92,
            conv2d93,
            conv2d94,
            conv2d95,
            conv2d96,
            conv2d97,
            conv2d98,
            conv2d99,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, add10_out1: Tensor<4>) -> (Tensor<4>, Tensor<4>) {
        let conv2d75_out1 = self.conv2d75.forward(add10_out1.clone());
        let sigmoid60_out1 = burn::tensor::activation::sigmoid(conv2d75_out1.clone());
        let mul60_out1 = conv2d75_out1.mul(sigmoid60_out1);
        let conv2d76_out1 = self.conv2d76.forward(mul60_out1);
        let sigmoid61_out1 = burn::tensor::activation::sigmoid(conv2d76_out1.clone());
        let mul61_out1 = conv2d76_out1.mul(sigmoid61_out1);
        let reducemean16_out1 = { mul61_out1.clone().mean_dim(2usize).mean_dim(3usize) };
        let conv2d77_out1 = self.conv2d77.forward(reducemean16_out1);
        let sigmoid62_out1 = burn::tensor::activation::sigmoid(conv2d77_out1.clone());
        let mul62_out1 = conv2d77_out1.mul(sigmoid62_out1);
        let conv2d78_out1 = self.conv2d78.forward(mul62_out1);
        let sigmoid63_out1 = burn::tensor::activation::sigmoid(conv2d78_out1);
        let mul63_out1 = sigmoid63_out1.mul(mul61_out1);
        let conv2d79_out1 = self.conv2d79.forward(mul63_out1);
        let add11_out1 = conv2d79_out1.add(add10_out1);
        let conv2d80_out1 = self.conv2d80.forward(add11_out1.clone());
        let sigmoid64_out1 = burn::tensor::activation::sigmoid(conv2d80_out1.clone());
        let mul64_out1 = conv2d80_out1.mul(sigmoid64_out1);
        let conv2d81_out1 = self.conv2d81.forward(mul64_out1);
        let sigmoid65_out1 = burn::tensor::activation::sigmoid(conv2d81_out1.clone());
        let mul65_out1 = conv2d81_out1.mul(sigmoid65_out1);
        let reducemean17_out1 = { mul65_out1.clone().mean_dim(2usize).mean_dim(3usize) };
        let conv2d82_out1 = self.conv2d82.forward(reducemean17_out1);
        let sigmoid66_out1 = burn::tensor::activation::sigmoid(conv2d82_out1.clone());
        let mul66_out1 = conv2d82_out1.mul(sigmoid66_out1);
        let conv2d83_out1 = self.conv2d83.forward(mul66_out1);
        let sigmoid67_out1 = burn::tensor::activation::sigmoid(conv2d83_out1);
        let mul67_out1 = sigmoid67_out1.mul(mul65_out1);
        let conv2d84_out1 = self.conv2d84.forward(mul67_out1);
        let conv2d85_out1 = self.conv2d85.forward(conv2d84_out1.clone());
        let sigmoid68_out1 = burn::tensor::activation::sigmoid(conv2d85_out1.clone());
        let mul68_out1 = conv2d85_out1.mul(sigmoid68_out1);
        let conv2d86_out1 = self.conv2d86.forward(mul68_out1);
        let sigmoid69_out1 = burn::tensor::activation::sigmoid(conv2d86_out1.clone());
        let mul69_out1 = conv2d86_out1.mul(sigmoid69_out1);
        let reducemean18_out1 = { mul69_out1.clone().mean_dim(2usize).mean_dim(3usize) };
        let conv2d87_out1 = self.conv2d87.forward(reducemean18_out1);
        let sigmoid70_out1 = burn::tensor::activation::sigmoid(conv2d87_out1.clone());
        let mul70_out1 = conv2d87_out1.mul(sigmoid70_out1);
        let conv2d88_out1 = self.conv2d88.forward(mul70_out1);
        let sigmoid71_out1 = burn::tensor::activation::sigmoid(conv2d88_out1);
        let mul71_out1 = sigmoid71_out1.mul(mul69_out1);
        let conv2d89_out1 = self.conv2d89.forward(mul71_out1);
        let add12_out1 = conv2d89_out1.add(conv2d84_out1);
        let conv2d90_out1 = self.conv2d90.forward(add12_out1.clone());
        let sigmoid72_out1 = burn::tensor::activation::sigmoid(conv2d90_out1.clone());
        let mul72_out1 = conv2d90_out1.mul(sigmoid72_out1);
        let conv2d91_out1 = self.conv2d91.forward(mul72_out1);
        let sigmoid73_out1 = burn::tensor::activation::sigmoid(conv2d91_out1.clone());
        let mul73_out1 = conv2d91_out1.mul(sigmoid73_out1);
        let reducemean19_out1 = { mul73_out1.clone().mean_dim(2usize).mean_dim(3usize) };
        let conv2d92_out1 = self.conv2d92.forward(reducemean19_out1);
        let sigmoid74_out1 = burn::tensor::activation::sigmoid(conv2d92_out1.clone());
        let mul74_out1 = conv2d92_out1.mul(sigmoid74_out1);
        let conv2d93_out1 = self.conv2d93.forward(mul74_out1);
        let sigmoid75_out1 = burn::tensor::activation::sigmoid(conv2d93_out1);
        let mul75_out1 = sigmoid75_out1.mul(mul73_out1);
        let conv2d94_out1 = self.conv2d94.forward(mul75_out1);
        let add13_out1 = conv2d94_out1.add(add12_out1);
        let conv2d95_out1 = self.conv2d95.forward(add13_out1.clone());
        let sigmoid76_out1 = burn::tensor::activation::sigmoid(conv2d95_out1.clone());
        let mul76_out1 = conv2d95_out1.mul(sigmoid76_out1);
        let conv2d96_out1 = self.conv2d96.forward(mul76_out1);
        let sigmoid77_out1 = burn::tensor::activation::sigmoid(conv2d96_out1.clone());
        let mul77_out1 = conv2d96_out1.mul(sigmoid77_out1);
        let reducemean20_out1 = { mul77_out1.clone().mean_dim(2usize).mean_dim(3usize) };
        let conv2d97_out1 = self.conv2d97.forward(reducemean20_out1);
        let sigmoid78_out1 = burn::tensor::activation::sigmoid(conv2d97_out1.clone());
        let mul78_out1 = conv2d97_out1.mul(sigmoid78_out1);
        let conv2d98_out1 = self.conv2d98.forward(mul78_out1);
        let sigmoid79_out1 = burn::tensor::activation::sigmoid(conv2d98_out1);
        let mul79_out1 = sigmoid79_out1.mul(mul77_out1);
        let conv2d99_out1 = self.conv2d99.forward(mul79_out1);
        let add14_out1 = conv2d99_out1.add(add13_out1);
        (add14_out1, add11_out1)
    }
}
#[derive(Module, Debug)]
pub struct Submodule5 {
    conv2d100: Conv2d,
    conv2d101: Conv2d,
    conv2d102: Conv2d,
    conv2d103: Conv2d,
    conv2d104: Conv2d,
    conv2d105: Conv2d,
    conv2d106: Conv2d,
    conv2d107: Conv2d,
    conv2d108: Conv2d,
    conv2d109: Conv2d,
    conv2d110: Conv2d,
    conv2d111: Conv2d,
    conv2d112: Conv2d,
    conv2d113: Conv2d,
    conv2d114: Conv2d,
    resize1: burn::nn::interpolate::Interpolate2d,
    conv2d115: Conv2d,
    conv2d116: Conv2d,
    resize2: burn::nn::interpolate::Interpolate2d,
    conv2d117: Conv2d,
    conv2d118: Conv2d,
    resize3: burn::nn::interpolate::Interpolate2d,
    conv2d119: Conv2d,
    conv2d120: Conv2d,
    resize4: burn::nn::interpolate::Interpolate2d,
    conv2d121: Conv2d,
    conv2d122: Conv2d,
    resize5: burn::nn::interpolate::Interpolate2d,
    conv2d123: Conv2d,
    conv2d124: Conv2d,
    resize6: burn::nn::interpolate::Interpolate2d,
    conv2d125: Conv2d,
    conv2d126: Conv2d,
    resize7: burn::nn::interpolate::Interpolate2d,
    conv2d127: Conv2d,
    conv2d128: Conv2d,
    resize8: burn::nn::interpolate::Interpolate2d,
    conv2d129: Conv2d,
    conv2d130: Conv2d,
    resize9: burn::nn::interpolate::Interpolate2d,
    conv2d131: Conv2d,
    conv2d132: Conv2d,
    resize10: burn::nn::interpolate::Interpolate2d,
    conv2d133: Conv2d,
    conv2d134: Conv2d,
    resize11: burn::nn::interpolate::Interpolate2d,
    conv2d135: Conv2d,
    conv2d136: Conv2d,
    conv2d137: Conv2d,
    #[module(skip)]
    device: Device,
}
impl Submodule5 {
    #[allow(unused_variables)]
    pub fn new(device: &Device) -> Self {
        let conv2d100 = Conv2dConfig::new([208, 1248], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d101 = Conv2dConfig::new([1248, 1248], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(1248)
            .with_bias(true)
            .init(device);
        let conv2d102 = Conv2dConfig::new([1248, 52], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d103 = Conv2dConfig::new([52, 1248], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d104 = Conv2dConfig::new([1248, 208], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d105 = Conv2dConfig::new([208, 1248], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d106 = Conv2dConfig::new([1248, 1248], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1248)
            .with_bias(true)
            .init(device);
        let conv2d107 = Conv2dConfig::new([1248, 52], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d108 = Conv2dConfig::new([52, 1248], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d109 = Conv2dConfig::new([1248, 352], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d110 = Conv2dConfig::new([352, 2112], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d111 = Conv2dConfig::new([2112, 2112], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(2112)
            .with_bias(true)
            .init(device);
        let conv2d112 = Conv2dConfig::new([2112, 88], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d113 = Conv2dConfig::new([88, 2112], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d114 = Conv2dConfig::new([2112, 352], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let resize1 = burn::nn::interpolate::Interpolate2dConfig::new()
            .with_output_size(None)
            .with_scale_factor(Some([2.0, 2.0]))
            .with_mode(burn::nn::interpolate::InterpolateMode::Nearest)
            .with_align_corners(false)
            .init();
        let conv2d115 = Conv2dConfig::new([472, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d116 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let resize2 = burn::nn::interpolate::Interpolate2dConfig::new()
            .with_output_size(None)
            .with_scale_factor(Some([2.0, 2.0]))
            .with_mode(burn::nn::interpolate::InterpolateMode::Nearest)
            .with_align_corners(false)
            .init();
        let conv2d117 = Conv2dConfig::new([168, 48], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d118 = Conv2dConfig::new([48, 48], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let resize3 = burn::nn::interpolate::Interpolate2dConfig::new()
            .with_output_size(None)
            .with_scale_factor(Some([2.0, 2.0]))
            .with_mode(burn::nn::interpolate::InterpolateMode::Nearest)
            .with_align_corners(false)
            .init();
        let conv2d119 = Conv2dConfig::new([72, 24], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d120 = Conv2dConfig::new([24, 24], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let resize4 = burn::nn::interpolate::Interpolate2dConfig::new()
            .with_output_size(None)
            .with_scale_factor(Some([2.0, 2.0]))
            .with_mode(burn::nn::interpolate::InterpolateMode::Nearest)
            .with_align_corners(false)
            .init();
        let conv2d121 = Conv2dConfig::new([56, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d122 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let resize5 = burn::nn::interpolate::Interpolate2dConfig::new()
            .with_output_size(None)
            .with_scale_factor(Some([2.0, 2.0]))
            .with_mode(burn::nn::interpolate::InterpolateMode::Nearest)
            .with_align_corners(false)
            .init();
        let conv2d123 = Conv2dConfig::new([352, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d124 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let resize6 = burn::nn::interpolate::Interpolate2dConfig::new()
            .with_output_size(None)
            .with_scale_factor(Some([2.0, 2.0]))
            .with_mode(burn::nn::interpolate::InterpolateMode::Nearest)
            .with_align_corners(false)
            .init();
        let conv2d125 = Conv2dConfig::new([96, 24], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d126 = Conv2dConfig::new([24, 24], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let resize7 = burn::nn::interpolate::Interpolate2dConfig::new()
            .with_output_size(None)
            .with_scale_factor(Some([2.0, 2.0]))
            .with_mode(burn::nn::interpolate::InterpolateMode::Nearest)
            .with_align_corners(false)
            .init();
        let conv2d127 = Conv2dConfig::new([88, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d128 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let resize8 = burn::nn::interpolate::Interpolate2dConfig::new()
            .with_output_size(None)
            .with_scale_factor(Some([2.0, 2.0]))
            .with_mode(burn::nn::interpolate::InterpolateMode::Nearest)
            .with_align_corners(false)
            .init();
        let conv2d129 = Conv2dConfig::new([200, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d130 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let resize9 = burn::nn::interpolate::Interpolate2dConfig::new()
            .with_output_size(None)
            .with_scale_factor(Some([2.0, 2.0]))
            .with_mode(burn::nn::interpolate::InterpolateMode::Nearest)
            .with_align_corners(false)
            .init();
        let conv2d131 = Conv2dConfig::new([120, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d132 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let resize10 = burn::nn::interpolate::Interpolate2dConfig::new()
            .with_output_size(None)
            .with_scale_factor(Some([2.0, 2.0]))
            .with_mode(burn::nn::interpolate::InterpolateMode::Nearest)
            .with_align_corners(false)
            .init();
        let conv2d133 = Conv2dConfig::new([192, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d134 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let resize11 = burn::nn::interpolate::Interpolate2dConfig::new()
            .with_output_size(None)
            .with_scale_factor(Some([2.0, 2.0]))
            .with_mode(burn::nn::interpolate::InterpolateMode::Nearest)
            .with_align_corners(false)
            .init();
        let conv2d135 = Conv2dConfig::new([32, 16], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d136 = Conv2dConfig::new([16, 16], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d137 = Conv2dConfig::new([16, 13], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        Self {
            conv2d100,
            conv2d101,
            conv2d102,
            conv2d103,
            conv2d104,
            conv2d105,
            conv2d106,
            conv2d107,
            conv2d108,
            conv2d109,
            conv2d110,
            conv2d111,
            conv2d112,
            conv2d113,
            conv2d114,
            resize1,
            conv2d115,
            conv2d116,
            resize2,
            conv2d117,
            conv2d118,
            resize3,
            conv2d119,
            conv2d120,
            resize4,
            conv2d121,
            conv2d122,
            resize5,
            conv2d123,
            conv2d124,
            resize6,
            conv2d125,
            conv2d126,
            resize7,
            conv2d127,
            conv2d128,
            resize8,
            conv2d129,
            conv2d130,
            resize9,
            conv2d131,
            conv2d132,
            resize10,
            conv2d133,
            conv2d134,
            resize11,
            conv2d135,
            conv2d136,
            conv2d137,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add14_out1: Tensor<4>,
        add11_out1: Tensor<4>,
        add5_out1: Tensor<4>,
        add3_out1: Tensor<4>,
        mul1_out1: Tensor<4>,
    ) -> Tensor<4> {
        let conv2d100_out1 = self.conv2d100.forward(add14_out1.clone());
        let sigmoid80_out1 = burn::tensor::activation::sigmoid(conv2d100_out1.clone());
        let mul80_out1 = conv2d100_out1.mul(sigmoid80_out1);
        let conv2d101_out1 = self.conv2d101.forward(mul80_out1);
        let sigmoid81_out1 = burn::tensor::activation::sigmoid(conv2d101_out1.clone());
        let mul81_out1 = conv2d101_out1.mul(sigmoid81_out1);
        let reducemean21_out1 = { mul81_out1.clone().mean_dim(2usize).mean_dim(3usize) };
        let conv2d102_out1 = self.conv2d102.forward(reducemean21_out1);
        let sigmoid82_out1 = burn::tensor::activation::sigmoid(conv2d102_out1.clone());
        let mul82_out1 = conv2d102_out1.mul(sigmoid82_out1);
        let conv2d103_out1 = self.conv2d103.forward(mul82_out1);
        let sigmoid83_out1 = burn::tensor::activation::sigmoid(conv2d103_out1);
        let mul83_out1 = sigmoid83_out1.mul(mul81_out1);
        let conv2d104_out1 = self.conv2d104.forward(mul83_out1);
        let add15_out1 = conv2d104_out1.add(add14_out1);
        let conv2d105_out1 = self.conv2d105.forward(add15_out1);
        let sigmoid84_out1 = burn::tensor::activation::sigmoid(conv2d105_out1.clone());
        let mul84_out1 = conv2d105_out1.mul(sigmoid84_out1);
        let conv2d106_out1 = self.conv2d106.forward(mul84_out1);
        let sigmoid85_out1 = burn::tensor::activation::sigmoid(conv2d106_out1.clone());
        let mul85_out1 = conv2d106_out1.mul(sigmoid85_out1);
        let reducemean22_out1 = { mul85_out1.clone().mean_dim(2usize).mean_dim(3usize) };
        let conv2d107_out1 = self.conv2d107.forward(reducemean22_out1);
        let sigmoid86_out1 = burn::tensor::activation::sigmoid(conv2d107_out1.clone());
        let mul86_out1 = conv2d107_out1.mul(sigmoid86_out1);
        let conv2d108_out1 = self.conv2d108.forward(mul86_out1);
        let sigmoid87_out1 = burn::tensor::activation::sigmoid(conv2d108_out1);
        let mul87_out1 = sigmoid87_out1.mul(mul85_out1);
        let conv2d109_out1 = self.conv2d109.forward(mul87_out1);
        let conv2d110_out1 = self.conv2d110.forward(conv2d109_out1.clone());
        let sigmoid88_out1 = burn::tensor::activation::sigmoid(conv2d110_out1.clone());
        let mul88_out1 = conv2d110_out1.mul(sigmoid88_out1);
        let conv2d111_out1 = self.conv2d111.forward(mul88_out1);
        let sigmoid89_out1 = burn::tensor::activation::sigmoid(conv2d111_out1.clone());
        let mul89_out1 = conv2d111_out1.mul(sigmoid89_out1);
        let reducemean23_out1 = { mul89_out1.clone().mean_dim(2usize).mean_dim(3usize) };
        let conv2d112_out1 = self.conv2d112.forward(reducemean23_out1);
        let sigmoid90_out1 = burn::tensor::activation::sigmoid(conv2d112_out1.clone());
        let mul90_out1 = conv2d112_out1.mul(sigmoid90_out1);
        let conv2d113_out1 = self.conv2d113.forward(mul90_out1);
        let sigmoid91_out1 = burn::tensor::activation::sigmoid(conv2d113_out1);
        let mul91_out1 = sigmoid91_out1.mul(mul89_out1);
        let conv2d114_out1 = self.conv2d114.forward(mul91_out1);
        let add16_out1 = conv2d114_out1.add(conv2d109_out1);
        let resize1_out1 = self.resize1.forward(add16_out1);
        let concat1_out1 = burn::tensor::Tensor::cat([resize1_out1, add11_out1.clone()].into(), 1);
        let conv2d115_out1 = self.conv2d115.forward(concat1_out1);
        let relu1_out1 = burn::tensor::activation::relu(conv2d115_out1);
        let conv2d116_out1 = self.conv2d116.forward(relu1_out1);
        let relu2_out1 = burn::tensor::activation::relu(conv2d116_out1);
        let resize2_out1 = self.resize2.forward(add11_out1);
        let concat2_out1 = burn::tensor::Tensor::cat([resize2_out1, add5_out1.clone()].into(), 1);
        let conv2d117_out1 = self.conv2d117.forward(concat2_out1);
        let relu3_out1 = burn::tensor::activation::relu(conv2d117_out1);
        let conv2d118_out1 = self.conv2d118.forward(relu3_out1);
        let relu4_out1 = burn::tensor::activation::relu(conv2d118_out1);
        let resize3_out1 = self.resize3.forward(add5_out1.clone());
        let concat3_out1 = burn::tensor::Tensor::cat([resize3_out1, add3_out1.clone()].into(), 1);
        let conv2d119_out1 = self.conv2d119.forward(concat3_out1);
        let relu5_out1 = burn::tensor::activation::relu(conv2d119_out1);
        let conv2d120_out1 = self.conv2d120.forward(relu5_out1);
        let relu6_out1 = burn::tensor::activation::relu(conv2d120_out1);
        let resize4_out1 = self.resize4.forward(add3_out1.clone());
        let concat4_out1 = burn::tensor::Tensor::cat([resize4_out1, mul1_out1.clone()].into(), 1);
        let conv2d121_out1 = self.conv2d121.forward(concat4_out1);
        let relu7_out1 = burn::tensor::activation::relu(conv2d121_out1);
        let conv2d122_out1 = self.conv2d122.forward(relu7_out1);
        let relu8_out1 = burn::tensor::activation::relu(conv2d122_out1);
        let concat5_out1 = burn::tensor::Tensor::cat([relu4_out1.clone(), add5_out1].into(), 1);
        let resize5_out1 = self.resize5.forward(relu2_out1);
        let concat6_out1 = burn::tensor::Tensor::cat([resize5_out1, concat5_out1].into(), 1);
        let conv2d123_out1 = self.conv2d123.forward(concat6_out1);
        let relu9_out1 = burn::tensor::activation::relu(conv2d123_out1);
        let conv2d124_out1 = self.conv2d124.forward(relu9_out1);
        let relu10_out1 = burn::tensor::activation::relu(conv2d124_out1);
        let concat7_out1 =
            burn::tensor::Tensor::cat([relu6_out1.clone(), add3_out1.clone()].into(), 1);
        let resize6_out1 = self.resize6.forward(relu4_out1);
        let concat8_out1 = burn::tensor::Tensor::cat([resize6_out1, concat7_out1].into(), 1);
        let conv2d125_out1 = self.conv2d125.forward(concat8_out1);
        let relu11_out1 = burn::tensor::activation::relu(conv2d125_out1);
        let conv2d126_out1 = self.conv2d126.forward(relu11_out1);
        let relu12_out1 = burn::tensor::activation::relu(conv2d126_out1);
        let concat9_out1 =
            burn::tensor::Tensor::cat([relu8_out1.clone(), mul1_out1.clone()].into(), 1);
        let resize7_out1 = self.resize7.forward(relu6_out1.clone());
        let concat10_out1 = burn::tensor::Tensor::cat([resize7_out1, concat9_out1].into(), 1);
        let conv2d127_out1 = self.conv2d127.forward(concat10_out1);
        let relu13_out1 = burn::tensor::activation::relu(conv2d127_out1);
        let conv2d128_out1 = self.conv2d128.forward(relu13_out1);
        let relu14_out1 = burn::tensor::activation::relu(conv2d128_out1);
        let concat11_out1 =
            burn::tensor::Tensor::cat([relu12_out1.clone(), relu6_out1, add3_out1].into(), 1);
        let resize8_out1 = self.resize8.forward(relu10_out1);
        let concat12_out1 = burn::tensor::Tensor::cat([resize8_out1, concat11_out1].into(), 1);
        let conv2d129_out1 = self.conv2d129.forward(concat12_out1);
        let relu15_out1 = burn::tensor::activation::relu(conv2d129_out1);
        let conv2d130_out1 = self.conv2d130.forward(relu15_out1);
        let relu16_out1 = burn::tensor::activation::relu(conv2d130_out1);
        let concat13_out1 = burn::tensor::Tensor::cat(
            [relu14_out1.clone(), relu8_out1.clone(), mul1_out1.clone()].into(),
            1,
        );
        let resize9_out1 = self.resize9.forward(relu12_out1);
        let concat14_out1 = burn::tensor::Tensor::cat([resize9_out1, concat13_out1].into(), 1);
        let conv2d131_out1 = self.conv2d131.forward(concat14_out1);
        let relu17_out1 = burn::tensor::activation::relu(conv2d131_out1);
        let conv2d132_out1 = self.conv2d132.forward(relu17_out1);
        let relu18_out1 = burn::tensor::activation::relu(conv2d132_out1);
        let concat15_out1 =
            burn::tensor::Tensor::cat([relu18_out1, relu14_out1, relu8_out1, mul1_out1].into(), 1);
        let resize10_out1 = self.resize10.forward(relu16_out1);
        let concat16_out1 = burn::tensor::Tensor::cat([resize10_out1, concat15_out1].into(), 1);
        let conv2d133_out1 = self.conv2d133.forward(concat16_out1);
        let relu19_out1 = burn::tensor::activation::relu(conv2d133_out1);
        let conv2d134_out1 = self.conv2d134.forward(relu19_out1);
        let relu20_out1 = burn::tensor::activation::relu(conv2d134_out1);
        let resize11_out1 = self.resize11.forward(relu20_out1);
        let conv2d135_out1 = self.conv2d135.forward(resize11_out1);
        let relu21_out1 = burn::tensor::activation::relu(conv2d135_out1);
        let conv2d136_out1 = self.conv2d136.forward(relu21_out1);
        let relu22_out1 = burn::tensor::activation::relu(conv2d136_out1);
        let conv2d137_out1 = self.conv2d137.forward(relu22_out1);
        conv2d137_out1
    }
}

#[derive(Module, Debug)]
pub struct Model {
    submodule1: Submodule1,
    submodule2: Submodule2,
    submodule3: Submodule3,
    submodule4: Submodule4,
    submodule5: Submodule5,
    #[module(skip)]
    device: Device,
}

extern crate std;

impl Default for Model {
    fn default() -> Self {
        Self::from_file(
            "weights/segmentation/efficientnet-b2_best.bpk",
            &Default::default(),
        )
    }
}

impl Model {
    /// Load model weights from a burnpack file.
    pub fn from_file<P: AsRef<std::path::Path>>(file: P, device: &Device) -> Self {
        let mut model = Self::new(device);
        let mut store = BurnpackStore::from_file(&file);
        model.load_from(&mut store).unwrap_or_else(|e| {
            panic!(
                "Failed to load burnpack file {}: {e}",
                file.as_ref().display()
            )
        });
        model
    }

    /// Load model weights from in-memory bytes.
    ///
    /// The bytes must be the contents of a `.bpk` file.
    pub fn from_bytes(bytes: Bytes, device: &Device) -> Self {
        let mut model = Self::new(device);
        let mut store = BurnpackStore::from_bytes(Some(bytes));
        model
            .load_from(&mut store)
            .unwrap_or_else(|e| panic!("Failed to load burnpack bytes: {e}"));
        model
    }
}

impl Model {
    #[allow(unused_variables)]
    pub fn new(device: &Device) -> Self {
        let submodule1 = Submodule1::new(device);
        let submodule2 = Submodule2::new(device);
        let submodule3 = Submodule3::new(device);
        let submodule4 = Submodule4::new(device);
        let submodule5 = Submodule5::new(device);
        Self {
            submodule1,
            submodule2,
            submodule3,
            submodule4,
            submodule5,
            device: device.clone(),
        }
    }

    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        let (add3_out1, mul1_out1) = self.submodule1.forward(input);
        let (add6_out1, add5_out1) = self.submodule2.forward(add3_out1.clone());
        let add10_out1 = self.submodule3.forward(add6_out1);
        let (add14_out1, add11_out1) = self.submodule4.forward(add10_out1);
        let conv2d137_out1 = self
            .submodule5
            .forward(add14_out1, add11_out1, add5_out1, add3_out1, mul1_out1);
        conv2d137_out1
    }
}
