use std::path::PathBuf;

use ort::ExecutionProvider;
use rust_bert::marian::MarianConfig;
use rust_bert::pipelines::common::{ConfigOption, TokenizerOption};
use rust_bert::pipelines::generation_utils::{GenerateConfig, LanguageGenerator};
use rust_bert::pipelines::onnx::config::ONNXEnvironmentConfig;

use rust_bert::pipelines::onnx::models::ONNXConditionalGenerator;
use rust_bert::Config;
use rust_tokenizers::tokenizer::MarianTokenizer;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let onnx_causal_decoder = ONNXConditionalGenerator::new(
        PathBuf::from("E:/Coding/opus-mt-en-fr-onnx/encoder_model.onnx"),
        Some(PathBuf::from(
            "E:/Coding/opus-mt-en-fr-onnx/decoder_model.onnx",
        )),
        Some(PathBuf::from(
            "E:/Coding/opus-mt-en-fr-onnx/decoder_with_past_model.onnx",
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
        TokenizerOption::Marian(
            MarianTokenizer::from_files(
                "E:/Coding/opus-mt-en-fr-onnx/vocab.json",
                "E:/Coding/opus-mt-en-fr-onnx/source.spm",
                false,
            )
            .unwrap(),
        ),
        ConfigOption::Marian(MarianConfig::from_file(
            "E:/Coding/opus-mt-en-fr-onnx/config.json",
        )),
        None,
    )?;

    let output = onnx_causal_decoder.generate(Some(&["Hello my name is John."]), None);
    println!("{:?}", output);
    Ok(())
}
