//! Load cleaned PyTorch state_dicts into native Burn modules and write burnpacks.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use burn::prelude::Device;
use burn_store::{BurnpackStore, ModuleSnapshot, PytorchStore};
use clap::Parser;
use hires_rs::detection::native::Model as DetectionModel;
use hires_rs::segmentation::native::Model as SegmentationModel;

#[derive(Parser, Debug)]
struct Cli {
    /// Detection state_dict from scripts/extract_state_dict.py
    #[arg(long, default_value = "weights/detection/best.state_dict.pt")]
    det_pt: PathBuf,

    /// Segmentation SMP state_dict (.pt)
    #[arg(long, default_value = "weights/segmentation/efficientnet-b2_best.pt")]
    seg_pt: PathBuf,

    #[arg(long, default_value = "weights/detection/best.bpk")]
    det_out: PathBuf,

    #[arg(long, default_value = "weights/segmentation/efficientnet-b2_best.bpk")]
    seg_out: PathBuf,

    /// Convert only detection or segmentation
    #[arg(long, value_parser = ["all", "detection", "segmentation"], default_value = "all")]
    model: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let device = Device::default();

    if cli.model == "all" || cli.model == "detection" {
        convert_detection(&cli.det_pt, &cli.det_out, &device)?;
    }
    if cli.model == "all" || cli.model == "segmentation" {
        convert_segmentation(&cli.seg_pt, &cli.seg_out, &device)?;
    }
    Ok(())
}

fn convert_detection(pt: &Path, out: &Path, device: &Device) -> Result<()> {
    if !pt.is_file() {
        bail!("detection state_dict not found: {}", pt.display());
    }
    println!("Loading detection {}", pt.display());
    let mut model = DetectionModel::new(device);
    let mut store = PytorchStore::from_file(pt)
        .map_indices_contiguous(false)
        .with_key_remapping(r"^model\.(\d+)\.", "layer$1.")
        .allow_partial(true);
    let result = model
        .load_from(&mut store)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .context("load detection pytorch weights")?;
    print_apply("detection", &result);
    save_bpk(&mut model, out)
}

fn convert_segmentation(pt: &Path, out: &Path, device: &Device) -> Result<()> {
    if !pt.is_file() {
        bail!("segmentation state_dict not found: {}", pt.display());
    }
    println!("Loading segmentation {}", pt.display());
    let mut model = SegmentationModel::new(device);
    let mut store = PytorchStore::from_file(pt)
        .map_indices_contiguous(false)
        .allow_partial(true);
    let result = model
        .load_from(&mut store)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .context("load segmentation pytorch weights")?;
    print_apply("segmentation", &result);
    save_bpk(&mut model, out)
}

fn print_apply(name: &str, result: &burn_store::ApplyResult) {
    println!(
        "{name}: applied={} unused={} missing={} errors={}",
        result.applied.len(),
        result.unused.len(),
        result.missing.len(),
        result.errors.len()
    );
    let missing: Vec<_> = result
        .missing
        .iter()
        .filter(|(p, _)| !p.ends_with("training") && !p.contains(".training"))
        .take(40)
        .collect();
    if !missing.is_empty() {
        eprintln!("{name} missing (first {}):", missing.len());
        for (p, _) in missing {
            eprintln!("  {p}");
        }
    }
    if !result.unused.is_empty() {
        eprintln!("{name} unused ({}):", result.unused.len());
        for p in result.unused.iter().take(20) {
            eprintln!("  {p}");
        }
    }
    if !result.errors.is_empty() {
        for e in result.errors.iter().take(20) {
            eprintln!("{name} error: {e}");
        }
    }
}

fn save_bpk<M: ModuleSnapshot>(model: &mut M, out: &Path) -> Result<()> {
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut store = BurnpackStore::from_file(out).overwrite(true);
    model
        .save_into(&mut store)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .with_context(|| format!("save {}", out.display()))?;
    println!("Wrote {}", out.display());
    Ok(())
}
