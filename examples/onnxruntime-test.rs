use ort::ExecutionProvider;
use rust_tokenizers::tokenizer::Gpt2Tokenizer;
use std::path::PathBuf;

use rust_bert::pipelines::generation_utils::{GenerateConfig, LanguageGenerator};
use rust_bert::pipelines::onnx::config::ONNXEnvironmentConfig;

use rust_bert::gpt2::Gpt2Config;
use rust_bert::pipelines::common::{ConfigOption, TokenizerOption};
use rust_bert::pipelines::onnx::models::ONNXCausalDecoder;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let onnx_causal_decoder = ONNXCausalDecoder::new(
        Some(PathBuf::from(
            "E:/Coding/distilgpt2-onnx/decoder_model.onnx",
        )),
        Some(PathBuf::from(
            "E:/Coding/distilgpt2-onnx/decoder_with_past_model.onnx",
        )),
        &ONNXEnvironmentConfig {
            execution_providers: Some(vec![ExecutionProvider::cuda()]),
            ..Default::default()
        },
        GenerateConfig {
            num_beams: 3,
            do_sample: false,
            ..Default::default()
        },
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

    let output = onnx_causal_decoder.generate(Some(&["Today is a"]), None);
    println!("{:?}", output);

    Ok(())
}
