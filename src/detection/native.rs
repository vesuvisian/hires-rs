//! Hand-written YOLOv8n (`nc=1`) matching Ultralytics module names after remap.
//!
//! Forward contract matches the ONNX export: `[1, 3, 640, 640]` → `[1, 5, 8400]`.

use burn::nn::PaddingConfig2d;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::interpolate::{Interpolate2dConfig, InterpolateMode};
use burn::nn::pool::{MaxPool2d, MaxPool2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig};
use burn::prelude::*;
use burn::tensor::Bytes;
use burn::tensor::activation::{sigmoid, silu, softmax};
use burn_store::BurnpackStore;
use burn_store::ModuleSnapshot;

const BN_EPS: f64 = 1e-3;
const BN_MOM: f64 = 0.03;
const REG_MAX: usize = 16;
const NC: usize = 1;

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

fn upsample_x2(x: Tensor<4>) -> Tensor<4> {
    let dims = x.dims();
    Interpolate2dConfig::new()
        .with_output_size(Some([dims[2] * 2, dims[3] * 2]))
        .with_mode(InterpolateMode::Nearest)
        .with_align_corners(false)
        .init()
        .forward(x)
}

/// Ultralytics `Conv`: Conv2d + BatchNorm2d + SiLU.
#[derive(Module, Debug)]
pub struct ConvBnAct {
    conv: Conv2d,
    bn: BatchNorm,
}

impl ConvBnAct {
    pub fn new(cin: usize, cout: usize, k: usize, stride: usize, device: &Device) -> Self {
        Self {
            conv: conv(cin, cout, k, stride, 1, false, device),
            bn: bn(cout, device),
        }
    }

    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        silu(self.bn.forward(self.conv.forward(x)))
    }
}

#[derive(Module, Debug)]
pub struct Bottleneck {
    cv1: ConvBnAct,
    cv2: ConvBnAct,
    #[module(skip)]
    add: bool,
}

impl Bottleneck {
    pub fn new(c: usize, shortcut: bool, device: &Device) -> Self {
        Self {
            cv1: ConvBnAct::new(c, c, 3, 1, device),
            cv2: ConvBnAct::new(c, c, 3, 1, device),
            add: shortcut,
        }
    }

    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let y = self.cv2.forward(self.cv1.forward(x.clone()));
        if self.add { x.add(y) } else { y }
    }
}

#[derive(Module, Debug)]
pub struct C2f {
    cv1: ConvBnAct,
    cv2: ConvBnAct,
    m: Vec<Bottleneck>,
    #[module(skip)]
    hidden: usize,
}

impl C2f {
    pub fn new(cin: usize, cout: usize, n: usize, shortcut: bool, device: &Device) -> Self {
        let hidden = cout / 2;
        let m = (0..n)
            .map(|_| Bottleneck::new(hidden, shortcut, device))
            .collect();
        Self {
            cv1: ConvBnAct::new(cin, 2 * hidden, 1, 1, device),
            cv2: ConvBnAct::new((2 + n) * hidden, cout, 1, 1, device),
            m,
            hidden,
        }
    }

    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let y = self.cv1.forward(x);
        let parts = y.split_with_sizes([self.hidden, self.hidden].into(), 1);
        let [a, mut b] = parts.try_into().unwrap();
        let mut seq = vec![a, b.clone()];
        for block in &self.m {
            b = block.forward(b);
            seq.push(b.clone());
        }
        self.cv2.forward(Tensor::cat(seq, 1))
    }
}

#[derive(Module, Debug)]
pub struct Sppf {
    cv1: ConvBnAct,
    cv2: ConvBnAct,
    m: MaxPool2d,
}

impl Sppf {
    pub fn new(c: usize, device: &Device) -> Self {
        let hidden = c / 2;
        Self {
            cv1: ConvBnAct::new(c, hidden, 1, 1, device),
            cv2: ConvBnAct::new(hidden * 4, c, 1, 1, device),
            m: MaxPool2dConfig::new([5, 5])
                .with_strides([1, 1])
                .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
                .init(),
        }
    }

    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let x = self.cv1.forward(x);
        let y1 = self.m.forward(x.clone());
        let y2 = self.m.forward(y1.clone());
        let y3 = self.m.forward(y2.clone());
        self.cv2.forward(Tensor::cat(vec![x, y1, y2, y3], 1))
    }
}

/// Last 1×1 of a Detect tower is a plain Conv2d (bias, no BN).
type DetectTower = (ConvBnAct, ConvBnAct, Conv2d);

fn detect_tower(cin: usize, mid: usize, cout: usize, device: &Device) -> DetectTower {
    (
        ConvBnAct::new(cin, mid, 3, 1, device),
        ConvBnAct::new(mid, mid, 3, 1, device),
        conv(mid, cout, 1, 1, 1, true, device),
    )
}

fn tower_forward(t: &DetectTower, x: Tensor<4>) -> Tensor<4> {
    t.2.forward(t.1.forward(t.0.forward(x)))
}

#[derive(Module, Debug)]
pub struct Dfl {
    conv: Conv2d,
}

impl Dfl {
    pub fn new(device: &Device) -> Self {
        Self {
            conv: Conv2dConfig::new([REG_MAX, 1], [1, 1])
                .with_bias(false)
                .init(device),
        }
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let dims = x.dims();
        let b = dims[0];
        let a = dims[2];
        let x = x.reshape([b, 4, REG_MAX, a]).swap_dims(1, 2);
        let x = softmax(x, 1);
        self.conv.forward(x).reshape([b, 4, a])
    }
}

#[derive(Module, Debug)]
pub struct Detect {
    cv2: [DetectTower; 3],
    cv3: [DetectTower; 3],
    dfl: Dfl,
}

impl Detect {
    pub fn new(ch: [usize; 3], device: &Device) -> Self {
        let c2 = 16.max(ch[0] / 4).max(REG_MAX * 4);
        let c3 = ch[0].max(NC.min(100));
        Self {
            cv2: [
                detect_tower(ch[0], c2, 4 * REG_MAX, device),
                detect_tower(ch[1], c2, 4 * REG_MAX, device),
                detect_tower(ch[2], c2, 4 * REG_MAX, device),
            ],
            cv3: [
                detect_tower(ch[0], c3, NC, device),
                detect_tower(ch[1], c3, NC, device),
                detect_tower(ch[2], c3, NC, device),
            ],
            dfl: Dfl::new(device),
        }
    }

    pub fn forward(&self, feats: [Tensor<4>; 3]) -> Tensor<3> {
        let mut box_cls = Vec::with_capacity(3);
        for (i, feat) in feats.into_iter().enumerate() {
            let b = tower_forward(&self.cv2[i], feat.clone());
            let c = tower_forward(&self.cv3[i], feat);
            box_cls.push(Tensor::cat(vec![b, c], 1));
        }

        let strides = [8.0f32, 16.0, 32.0];
        let mut box_parts = Vec::new();
        let mut cls_parts = Vec::new();
        let mut anchors = Vec::new();
        let mut stride_t = Vec::new();
        let device = box_cls[0].device();
        let batch = box_cls[0].dims()[0];

        for (i, pred) in box_cls.iter().enumerate() {
            let dims = pred.dims();
            let h = dims[2];
            let w = dims[3];
            let n = h * w;
            let flat = pred.clone().reshape([batch, 4 * REG_MAX + NC, n]);
            box_parts.push(flat.clone().slice([0..batch, 0..(4 * REG_MAX), 0..n]));
            cls_parts.push(flat.slice([0..batch, (4 * REG_MAX)..(4 * REG_MAX + NC), 0..n]));
            anchors.push(make_anchors(h, w, strides[i], &device));
            stride_t.push(Tensor::<2>::full([1, n], strides[i], &device).reshape([1, 1, n]));
        }

        let boxes = Tensor::cat(box_parts, 2);
        let cls = Tensor::cat(cls_parts, 2);
        let anchor = Tensor::cat(anchors, 2);
        let stride = Tensor::cat(stride_t, 2);

        let dist = self.dfl.forward(boxes);
        let decoded = dist2bbox(dist, anchor).mul(stride);
        Tensor::cat(vec![decoded, sigmoid(cls)], 1)
    }
}

fn make_anchors(h: usize, w: usize, stride: f32, device: &Device) -> Tensor<3> {
    let _ = stride;
    let sy = Tensor::<1, Int>::arange(0..h as i64, device)
        .float()
        .add_scalar(0.5);
    let sx = Tensor::<1, Int>::arange(0..w as i64, device)
        .float()
        .add_scalar(0.5);
    let sy = sy.reshape([h, 1]).repeat_dim(1, w).reshape([1, 1, h * w]);
    let sx = sx.reshape([1, w]).repeat_dim(0, h).reshape([1, 1, h * w]);
    Tensor::cat(vec![sx, sy], 1)
}

fn dist2bbox(distance: Tensor<3>, anchor: Tensor<3>) -> Tensor<3> {
    let dims = distance.dims();
    let parts = distance.split_with_sizes([2, 2].into(), 1);
    let [lt, rb] = parts.try_into().unwrap();
    let x1y1 = anchor.clone().sub(lt);
    let x2y2 = anchor.add(rb);
    let c_xy = x1y1.clone().add(x2y2.clone()).div_scalar(2.0);
    let wh = x2y2.sub(x1y1);
    let _ = dims;
    Tensor::cat(vec![c_xy, wh], 1)
}

#[derive(Module, Debug)]
pub struct Model {
    layer0: ConvBnAct,
    layer1: ConvBnAct,
    layer2: C2f,
    layer3: ConvBnAct,
    layer4: C2f,
    layer5: ConvBnAct,
    layer6: C2f,
    layer7: ConvBnAct,
    layer8: C2f,
    layer9: Sppf,
    layer12: C2f,
    layer15: C2f,
    layer16: ConvBnAct,
    layer18: C2f,
    layer19: ConvBnAct,
    layer21: C2f,
    layer22: Detect,
    #[module(skip)]
    device: Device,
}

impl Model {
    pub fn new(device: &Device) -> Self {
        Self {
            layer0: ConvBnAct::new(3, 16, 3, 2, device),
            layer1: ConvBnAct::new(16, 32, 3, 2, device),
            layer2: C2f::new(32, 32, 1, true, device),
            layer3: ConvBnAct::new(32, 64, 3, 2, device),
            layer4: C2f::new(64, 64, 2, true, device),
            layer5: ConvBnAct::new(64, 128, 3, 2, device),
            layer6: C2f::new(128, 128, 2, true, device),
            layer7: ConvBnAct::new(128, 256, 3, 2, device),
            layer8: C2f::new(256, 256, 1, true, device),
            layer9: Sppf::new(256, device),
            layer12: C2f::new(384, 128, 1, false, device),
            layer15: C2f::new(192, 64, 1, false, device),
            layer16: ConvBnAct::new(64, 64, 3, 2, device),
            layer18: C2f::new(192, 128, 1, false, device),
            layer19: ConvBnAct::new(128, 128, 3, 2, device),
            layer21: C2f::new(384, 256, 1, false, device),
            layer22: Detect::new([64, 128, 256], device),
            device: device.clone(),
        }
    }

    pub fn forward(&self, images: Tensor<4>) -> Tensor<3> {
        let x0 = self.layer0.forward(images);
        let x1 = self.layer1.forward(x0);
        let x2 = self.layer2.forward(x1);
        let x3 = self.layer3.forward(x2);
        let x4 = self.layer4.forward(x3);
        let x5 = self.layer5.forward(x4.clone());
        let x6 = self.layer6.forward(x5);
        let x7 = self.layer7.forward(x6.clone());
        let x8 = self.layer8.forward(x7);
        let x9 = self.layer9.forward(x8);

        let p4 = self
            .layer12
            .forward(Tensor::cat(vec![upsample_x2(x9.clone()), x6], 1));
        let p3 = self
            .layer15
            .forward(Tensor::cat(vec![upsample_x2(p4.clone()), x4], 1));
        let p4 = self
            .layer18
            .forward(Tensor::cat(vec![self.layer16.forward(p3.clone()), p4], 1));
        let p5 = self
            .layer21
            .forward(Tensor::cat(vec![self.layer19.forward(p4.clone()), x9], 1));
        self.layer22.forward([p3, p4, p5])
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
        Self::from_file(super::DET_WEIGHTS_DEFAULT, &Default::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yolo_detect_output_shape() {
        let device = Device::default();
        let detect = Detect::new([64, 128, 256], &device);
        let p3 = Tensor::<4>::zeros([1, 64, 80, 80], &device);
        let p4 = Tensor::<4>::zeros([1, 128, 40, 40], &device);
        let p5 = Tensor::<4>::zeros([1, 256, 20, 20], &device);
        let y = detect.forward([p3, p4, p5]);
        assert_eq!(y.dims(), [1, 5, 8400]);
        let data = y.to_data();
        let vals = data.as_slice::<f32>().unwrap();
        assert!(vals.iter().all(|v| v.is_finite()));
    }
}
