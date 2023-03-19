use rust_tokenizers::tokenizer::Gpt2Tokenizer;
use std::path::PathBuf;

use rust_bert::pipelines::generation_utils::GenerateConfig;
use rust_bert::pipelines::onnx::config::ONNXEnvironmentConfig;

use rust_bert::gpt2::Gpt2Config;
use rust_bert::pipelines::common::{ConfigOption, TokenizerOption};
use rust_bert::pipelines::onnx::models::ONNXCausalDecoder;
use tch::{Kind, Tensor};

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let onnx_causal_decoder = ONNXCausalDecoder::new(
        Some(PathBuf::from(
            "E:/Coding/distilgpt2-onnx/decoder_model.onnx",
        )),
        Some(PathBuf::from(
            "E:/Coding/distilgpt2-onnx/decoder_with_past_model.onnx",
        )),
        &ONNXEnvironmentConfig::default(),
        GenerateConfig::default(),
        TokenizerOption::GPT2(
            Gpt2Tokenizer::from_file(
                "E:/Coding/distilgpt2-onnx/vocab.json",
                "E:/Coding/distilgpt2-onnx/merges.txt",
                false,
            )
            .unwrap(),
        ),
        ConfigOption::GPT2(Gpt2Config::default()),
        None,
    )?;

    // Initial decoder forward pass (without past)
    let input_ids = Tensor::of_slice(&[8888, 318, 257])
        .unsqueeze(0)
        .to_kind(Kind::Int64);
    let attention_mask = Tensor::of_slice(&[1, 1, 1])
        .unsqueeze(0)
        .to_kind(Kind::Int64);

    let outputs =
        onnx_causal_decoder.forward(Some(&input_ids), Some(&attention_mask), None, None, None)?;

    println!("{} - {:?}", outputs.lm_logits, outputs.cache);

    // Second decoder forward pass (without past)
    let input_ids = Tensor::of_slice(&[649]).unsqueeze(0).to_kind(Kind::Int64);
    let attention_mask = Tensor::of_slice(&[1, 1, 1, 1])
        .unsqueeze(0)
        .to_kind(Kind::Int64);

    let outputs = onnx_causal_decoder.forward(
        Some(&input_ids),
        Some(&attention_mask),
        None,
        None,
        Some(&outputs.cache),
    )?;

    println!("{} - {:?}", outputs.lm_logits, outputs.cache);

    Ok(())
}
