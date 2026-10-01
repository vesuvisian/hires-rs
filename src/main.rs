use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::Parser;

use hires_rs::detection::model::Model as DetectionModel;
use hires_rs::pipeline::{PipelineArgs, collect_inputs, run_pipeline};
use hires_rs::resolve_device;
use hires_rs::segmentation::model::Model as SegmentationModel;

#[derive(Parser, Debug)]
#[command(
    name = "hires-rs",
    about = "End-to-end resistor value detection (Burn)"
)]
struct Cli {
    /// Image file or directory
    input: PathBuf,

    /// Detection burnpack weights
    #[arg(long, default_value = "weights/detection/best.bpk")]
    det_weights: PathBuf,

    /// Segmentation burnpack weights
    #[arg(long, default_value = "weights/segmentation/efficientnet-b2_best.bpk")]
    seg_weights: PathBuf,

    /// Output directory for composites and results.txt
    #[arg(long, default_value = "results/pipeline_output")]
    output_dir: PathBuf,

    /// Segmentation canvas size
    #[arg(long, default_value_t = 512)]
    size: usize,

    /// Minimum YOLO detection confidence (ONNX scores run lower than HiRes .pt)
    #[arg(long, default_value_t = hires_rs::detection::DEFAULT_CONF_THRESH)]
    det_conf: f32,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let device = resolve_device();

    #[cfg(feature = "wgpu")]
    println!("Device: wgpu");
    #[cfg(all(feature = "flex", not(feature = "wgpu")))]
    println!("Device: flex");

    println!("Detection weights:    {}", cli.det_weights.display());
    println!("Segmentation weights: {}", cli.seg_weights.display());
    println!();

    if !cli.det_weights.exists() {
        bail!("Detection weights not found: {}", cli.det_weights.display());
    }
    if !cli.seg_weights.exists() {
        bail!(
            "Segmentation weights not found: {}",
            cli.seg_weights.display()
        );
    }

    let det_model = DetectionModel::from_file(&cli.det_weights, &device);
    let seg_model = SegmentationModel::from_file(&cli.seg_weights, &device);

    let inputs = collect_inputs(&cli.input)?;
    println!("Processing {} image(s)...\n", inputs.len());

    let args = PipelineArgs {
        output_dir: cli.output_dir.clone(),
        size: cli.size,
        det_conf: cli.det_conf,
    };

    let mut all_results = Vec::new();
    for img_path in &inputs {
        let fname = img_path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| img_path.display().to_string());
        let detections = run_pipeline(img_path, &det_model, &seg_model, &device, &args)
            .with_context(|| format!("pipeline failed on {}", img_path.display()))?;
        if detections.is_empty() {
            println!("{fname}  →  [no detection]");
        }
        for d in &detections {
            println!(
                "{fname}  →  resistor {}: {} {}  (det={:.2}, seg={:.2})",
                d.crop, d.value, d.tolerance, d.det_conf, d.seg_conf
            );
        }
        all_results.push((fname, detections));
    }

    std::fs::create_dir_all(&cli.output_dir)?;
    let n_det: usize = all_results.iter().map(|(_, d)| d.len()).sum();
    let n_decoded = all_results
        .iter()
        .flat_map(|(_, d)| d)
        .filter(|d| !d.value.starts_with("ERROR"))
        .count();

    let txt_path = cli.output_dir.join("results.txt");
    let mut txt = String::from("Resistor Detection Results\n==========================\n\n");
    for (fname, detections) in &all_results {
        txt.push_str(fname);
        txt.push('\n');
        if detections.is_empty() {
            txt.push_str("  [no detection]\n");
        }
        for d in detections {
            let line = format!("  resistor {}: {} {}", d.crop, d.value, d.tolerance.trim());
            txt.push_str(line.trim_end());
            txt.push('\n');
        }
        txt.push('\n');
    }
    txt.push_str(&format!(
        "Summary: {} images  |  {} resistors detected  |  {} values decoded\n",
        inputs.len(),
        n_det,
        n_decoded
    ));
    std::fs::write(&txt_path, txt)?;

    println!("\nOutputs saved to: {}", cli.output_dir.display());
    println!("Results:          {}", txt_path.display());
    Ok(())
}
