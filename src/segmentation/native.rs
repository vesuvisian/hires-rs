//! SMP `UnetPlusPlus` + classic EfficientNet-B2 encoder (`classes=13`).
//!
//! Module names follow the PyTorch `state_dict` (`encoder._conv_stem`,
//! `decoder.blocks.x_0_0.conv1.0`, `segmentation_head.0`).

use burn::nn::PaddingConfig2d;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::interpolate::{Interpolate2dConfig, InterpolateMode};
use burn::nn::pool::AdaptiveAvgPool2dConfig;
use burn::nn::{BatchNorm, BatchNormConfig, Identity};
use burn::prelude::*;
use burn::tensor::Bytes;
use burn::tensor::activation::{relu, sigmoid, silu};
use burn_store::BurnpackStore;
use burn_store::ModuleSnapshot;

const BN_EPS: f64 = 1e-3;
const BN_MOM: f64 = 0.01;
const WIDTH: f64 = 1.1;
const DEPTH: f64 = 1.2;
const NUM_CLASSES: usize = 13;
const OUT_INDEXES: [usize; 4] = [4, 7, 15, 22];

fn conv(
    cin: usize,
    cout: usize,
    k: usize,
    stride: usize,
    groups: usize,
    bias: bool,
    device: &Device,
) -> Conv2d {
    let p = k / 2;
    Conv2dConfig::new([cin, cout], [k, k])
        .with_stride([stride, stride])
        .with_padding(PaddingConfig2d::Explicit(p, p, p, p))
        .with_groups(groups)
        .with_bias(bias)
        .init(device)
}

fn bn(c: usize, device: &Device) -> BatchNorm {
    BatchNormConfig::new(c)
        .with_epsilon(BN_EPS)
        .with_momentum(BN_MOM)
        .init(device)
}

fn round_filters(filters: usize) -> usize {
    let divisor = 8.0;
    let filters = filters as f64 * WIDTH;
    let mut new_filters = ((filters + divisor / 2.0) / divisor).floor() * divisor;
    new_filters = new_filters.max(divisor);
    if new_filters < 0.9 * filters {
        new_filters += divisor;
    }
    new_filters as usize
}

fn round_repeats(repeats: usize) -> usize {
    (DEPTH * repeats as f64).ceil() as usize
}

fn upsample_x2(x: Tensor<4>) -> Tensor<4> {
    let dims = x.dims();
    Interpolate2dConfig::new()
        .with_output_size(Some([dims[2] * 2, dims[3] * 2]))
        .with_mode(InterpolateMode::Nearest)
        .with_align_corners(false)
        .init()
        .forward(x)
}

#[derive(Module, Debug)]
pub struct MBConvBlock {
    _expand_conv: Option<Conv2d>,
    _bn0: Option<BatchNorm>,
    _depthwise_conv: Conv2d,
    _bn1: BatchNorm,
    _se_reduce: Conv2d,
    _se_expand: Conv2d,
    _project_conv: Conv2d,
    _bn2: BatchNorm,
    #[module(skip)]
    stride: usize,
    #[module(skip)]
    id_skip: bool,
    #[module(skip)]
    inp: usize,
    #[module(skip)]
    oup: usize,
}

impl MBConvBlock {
    fn new(
        inp: usize,
        oup: usize,
        k: usize,
        stride: usize,
        expand: usize,
        se_ratio: f64,
        device: &Device,
    ) -> Self {
        let hidden = inp * expand;
        let (_expand_conv, _bn0) = if expand != 1 {
            (
                Some(conv(inp, hidden, 1, 1, 1, false, device)),
                Some(bn(hidden, device)),
            )
        } else {
            (None, None)
        };
        let squeezed = 1.max((inp as f64 * se_ratio) as usize);
        Self {
            _expand_conv,
            _bn0,
            _depthwise_conv: conv(hidden, hidden, k, stride, hidden, false, device),
            _bn1: bn(hidden, device),
            _se_reduce: conv(hidden, squeezed, 1, 1, 1, true, device),
            _se_expand: conv(squeezed, hidden, 1, 1, 1, true, device),
            _project_conv: conv(hidden, oup, 1, 1, 1, false, device),
            _bn2: bn(oup, device),
            stride,
            id_skip: stride == 1 && inp == oup,
            inp,
            oup,
        }
    }

    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let inputs = x.clone();
        let mut y = x;
        if let (Some(expand), Some(bn0)) = (&self._expand_conv, &self._bn0) {
            y = silu(bn0.forward(expand.forward(y)));
        }
        y = silu(self._bn1.forward(self._depthwise_conv.forward(y)));
        let se = AdaptiveAvgPool2dConfig::new([1, 1])
            .init()
            .forward(y.clone());
        let se = silu(self._se_reduce.forward(se));
        let se = sigmoid(self._se_expand.forward(se));
        y = y.mul(se);
        y = self._bn2.forward(self._project_conv.forward(y));
        if self.id_skip {
            y = y.add(inputs);
        }
        let _ = (self.inp, self.oup, self.stride);
        y
    }
}

#[derive(Module, Debug)]
pub struct Encoder {
    _conv_stem: Conv2d,
    _bn0: BatchNorm,
    _blocks: Vec<MBConvBlock>,
}

impl Encoder {
    fn new(device: &Device) -> Self {
        // lukemelas EfficientNet-B2 block args after width/depth rounding.
        let base: [(usize, usize, usize, usize, usize, usize); 7] = [
            (1, 3, 1, 1, 32, 16),
            (2, 3, 2, 6, 16, 24),
            (2, 5, 2, 6, 24, 40),
            (3, 3, 2, 6, 40, 80),
            (3, 5, 1, 6, 80, 112),
            (4, 5, 2, 6, 112, 192),
            (1, 3, 1, 6, 192, 320),
        ];
        let mut blocks = Vec::new();
        let mut in_ch = round_filters(32);
        for (repeats, k, stride, expand, _i, o) in base {
            let out_ch = round_filters(o);
            let n = round_repeats(repeats);
            for j in 0..n {
                let s = if j == 0 { stride } else { 1 };
                let inp = if j == 0 { in_ch } else { out_ch };
                blocks.push(MBConvBlock::new(inp, out_ch, k, s, expand, 0.25, device));
            }
            in_ch = out_ch;
        }
        Self {
            _conv_stem: conv(3, round_filters(32), 3, 2, 1, false, device),
            _bn0: bn(round_filters(32), device),
            _blocks: blocks,
        }
    }

    /// Feature pyramid: input, stem, then skips at `OUT_INDEXES` (depth 5 → 6 tensors).
    fn forward(&self, x: Tensor<4>) -> Vec<Tensor<4>> {
        let mut features = vec![x.clone()];
        let mut y = silu(self._bn0.forward(self._conv_stem.forward(x)));
        features.push(y.clone());
        for (i, block) in self._blocks.iter().enumerate() {
            y = block.forward(y);
            if OUT_INDEXES.contains(&i) {
                features.push(y.clone());
            }
        }
        features
    }
}

/// SMP `Conv2dReLU` is `nn.Sequential(Conv2d, BatchNorm2d, ReLU)`.
type Conv2dReLU = (Conv2d, BatchNorm);

fn conv2d_relu(cin: usize, cout: usize, device: &Device) -> Conv2dReLU {
    (conv(cin, cout, 3, 1, 1, false, device), bn(cout, device))
}

fn conv2d_relu_forward(m: &Conv2dReLU, x: Tensor<4>) -> Tensor<4> {
    relu(m.1.forward(m.0.forward(x)))
}

#[derive(Module, Debug)]
pub struct DecoderBlock {
    conv1: Conv2dReLU,
    conv2: Conv2dReLU,
    #[module(skip)]
    has_skip: bool,
}

impl DecoderBlock {
    fn new(in_ch: usize, skip_ch: usize, out_ch: usize, device: &Device) -> Self {
        Self {
            conv1: conv2d_relu(in_ch + skip_ch, out_ch, device),
            conv2: conv2d_relu(out_ch, out_ch, device),
            has_skip: skip_ch > 0,
        }
    }

    fn forward(&self, x: Tensor<4>, skip: Option<Tensor<4>>) -> Tensor<4> {
        let mut x = upsample_x2(x);
        if let Some(skip) = skip {
            x = Tensor::cat(vec![x, skip], 1);
        }
        conv2d_relu_forward(&self.conv2, conv2d_relu_forward(&self.conv1, x))
    }
}

#[derive(Module, Debug)]
pub struct DecoderBlocks {
    x_0_0: DecoderBlock,
    x_0_1: DecoderBlock,
    x_1_1: DecoderBlock,
    x_0_2: DecoderBlock,
    x_1_2: DecoderBlock,
    x_2_2: DecoderBlock,
    x_0_3: DecoderBlock,
    x_1_3: DecoderBlock,
    x_2_3: DecoderBlock,
    x_3_3: DecoderBlock,
    x_0_4: DecoderBlock,
}

impl DecoderBlocks {
    fn new(device: &Device) -> Self {
        // UnetPlusPlusDecoder channel math for EfficientNet-B2 skips
        // [32, 24, 48, 120, 352] reversed head-first.
        Self {
            x_0_0: DecoderBlock::new(352, 120, 256, device),
            x_0_1: DecoderBlock::new(256, 96, 128, device),
            x_1_1: DecoderBlock::new(120, 48, 48, device),
            x_0_2: DecoderBlock::new(128, 72, 64, device),
            x_1_2: DecoderBlock::new(48, 48, 24, device),
            x_2_2: DecoderBlock::new(48, 24, 24, device),
            x_0_3: DecoderBlock::new(64, 128, 32, device),
            x_1_3: DecoderBlock::new(24, 96, 32, device),
            x_2_3: DecoderBlock::new(24, 64, 32, device),
            x_3_3: DecoderBlock::new(24, 32, 32, device),
            x_0_4: DecoderBlock::new(32, 0, 16, device),
        }
    }
}

#[derive(Module, Debug)]
pub struct Decoder {
    blocks: DecoderBlocks,
}

impl Decoder {
    fn new(device: &Device) -> Self {
        Self {
            blocks: DecoderBlocks::new(device),
        }
    }

    fn forward(&self, features: Vec<Tensor<4>>) -> Tensor<4> {
        // Drop same-resolution input skip; reverse so head is encoder bottom.
        let mut features: Vec<Tensor<4>> = features.into_iter().skip(1).collect();
        features.reverse();
        let b = &self.blocks;
        let mut dense: std::collections::HashMap<(usize, usize), Tensor<4>> =
            std::collections::HashMap::new();

        let in_len = 5;
        let depth = in_len - 1;
        for layer_idx in 0..depth {
            for depth_idx in 0..(depth - layer_idx) {
                if layer_idx == 0 {
                    let output = match depth_idx {
                        0 => b
                            .x_0_0
                            .forward(features[0].clone(), Some(features[1].clone())),
                        1 => b
                            .x_1_1
                            .forward(features[1].clone(), Some(features[2].clone())),
                        2 => b
                            .x_2_2
                            .forward(features[2].clone(), Some(features[3].clone())),
                        3 => b
                            .x_3_3
                            .forward(features[3].clone(), Some(features[4].clone())),
                        _ => unreachable!(),
                    };
                    dense.insert((depth_idx, depth_idx), output);
                } else {
                    let dense_l_i = depth_idx + layer_idx;
                    let mut cat_feats = Vec::new();
                    for idx in (depth_idx + 1)..=dense_l_i {
                        cat_feats.push(dense[&(idx, dense_l_i)].clone());
                    }
                    cat_feats.push(features[dense_l_i + 1].clone());
                    let skip = Tensor::cat(cat_feats, 1);
                    let prev = dense[&(depth_idx, dense_l_i - 1)].clone();
                    let output = match (depth_idx, dense_l_i) {
                        (0, 1) => b.x_0_1.forward(prev, Some(skip)),
                        (0, 2) => b.x_0_2.forward(prev, Some(skip)),
                        (0, 3) => b.x_0_3.forward(prev, Some(skip)),
                        (1, 2) => b.x_1_2.forward(prev, Some(skip)),
                        (1, 3) => b.x_1_3.forward(prev, Some(skip)),
                        (2, 3) => b.x_2_3.forward(prev, Some(skip)),
                        _ => unreachable!(),
                    };
                    dense.insert((depth_idx, dense_l_i), output);
                }
            }
        }
        b.x_0_4.forward(dense[&(0, depth - 1)].clone(), None)
    }
}

#[derive(Module, Debug)]
pub struct Model {
    encoder: Encoder,
    decoder: Decoder,
    segmentation_head: (Conv2d, Identity),
    #[module(skip)]
    device: Device,
}

impl Model {
    pub fn new(device: &Device) -> Self {
        Self {
            encoder: Encoder::new(device),
            decoder: Decoder::new(device),
            segmentation_head: (
                conv(16, NUM_CLASSES, 3, 1, 1, true, device),
                Identity::new(),
            ),
            device: device.clone(),
        }
    }

    pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        let feats = self.encoder.forward(input);
        let x = self.decoder.forward(feats);
        self.segmentation_head.0.forward(x)
    }

    #[cfg(not(target_arch = "wasm32"))]
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

    pub fn from_bytes(bytes: Bytes, device: &Device) -> Self {
        let mut model = Self::new(device);
        let mut store = BurnpackStore::from_bytes(Some(bytes));
        model
            .load_from(&mut store)
            .unwrap_or_else(|e| panic!("Failed to load burnpack bytes: {e}"));
        model
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Default for Model {
    fn default() -> Self {
        Self::from_file(super::SEG_WEIGHTS_DEFAULT, &Default::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unetpp_output_shape() {
        let device = Device::default();
        let model = Model::new(&device);
        let x = Tensor::<4>::zeros([1, 3, 64, 64], &device);
        let y = model.forward(x);
        assert_eq!(y.dims(), [1, 13, 64, 64]);
        let data = y.to_data();
        let vals = data.as_slice::<f32>().unwrap();
        assert!(vals.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn efficientnet_b2_block_count() {
        let device = Device::default();
        let enc = Encoder::new(&device);
        assert_eq!(enc._blocks.len(), 23);
    }
}
