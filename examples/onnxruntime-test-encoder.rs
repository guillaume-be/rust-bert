use std::path::PathBuf;
use std::sync::Arc;

use ort::{Environment, ExecutionProvider};
use rust_bert::pipelines::onnx::config::ONNXEnvironmentConfig;
use rust_bert::pipelines::onnx::encoder::ONNXEncoder;
use tch::{Kind, Tensor};

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    // Initial set-up, load ONNX sessions
    let environment = Arc::new(
        Environment::builder()
            .with_name("Marian")
            .with_execution_providers([ExecutionProvider::cpu()])
            .build()?,
    );
    let onnx_config = ONNXEnvironmentConfig::default();

    // Initial encoder forward pass
    let input_ids = Tensor::of_slice(&[1, 10537, 240, 1129, 32, 2211, 3, 0])
        .unsqueeze(0)
        .totype(Kind::Int64);
    let attention_mask = &Tensor::of_slice(&[1, 1, 1, 1, 1, 1, 1, 1])
        .unsqueeze(0)
        .totype(Kind::Int64);

    let encoder = ONNXEncoder::new(
        PathBuf::from("E:/Coding/opus-mt-en-fr-onnx/encoder_model.onnx"),
        &environment,
        &onnx_config,
    )?;

    let encoder_outputs =
        encoder.forward(Some(&input_ids), Some(&attention_mask), None, None, None)?;

    println!("{}", encoder_outputs.last_hidden_state);

    // let decoder = ONNXDecoder::new(
    //     PathBuf::from("E:/Coding/distilgpt2-onnx/decoder_model.onnx"),
    //     true,
    //     &environment,
    //     &onnx_config,
    // )?;
    //
    // let decoder_with_past = ONNXDecoder::new(
    //     PathBuf::from("E:/Coding/distilgpt2-onnx/decoder_with_past_model.onnx"),
    //     true,
    //     &environment,
    //     &onnx_config,
    // )?;
    //

    //
    // let outputs = decoder.forward(Some(&input_ids), Some(&attention_mask), None)?;
    //

    //
    // // Second decoder forward pass (without past)
    // let input_ids = Tensor::of_slice(&[649]).unsqueeze(0);
    // let attention_mask = &Tensor::of_slice(&[1, 1, 1, 1]).unsqueeze(0);
    //
    // let cache = match outputs.cache {
    //     Cache::ONNXCache(ref cache) => cache,
    //     _ => unreachable!(),
    // };
    //
    // let outputs =
    //     decoder_with_past.forward(Some(&input_ids), Some(&attention_mask), Some(cache))?;
    //
    // println!("{} - {:?}", outputs.lm_logits, outputs.cache);

    Ok(())
}
