use std::path::PathBuf;

use rust_bert::pipelines::common::{ModelResources, ModelType, ONNXModelResources};
use rust_bert::pipelines::generation_utils::{GenerateConfig, LanguageGenerator};
use rust_bert::pipelines::onnx::models::ONNXCausalGenerator;
use rust_bert::resources::LocalResource;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let onnx_causal_generator = ONNXCausalGenerator::new(
        GenerateConfig {
            model_type: ModelType::GPT2,
            model_resource: ModelResources::ONNX(ONNXModelResources {
                encoder_resource: None,
                decoder_resource: Some(Box::new(LocalResource::from(PathBuf::from(
                    "E:/Coding/distilgpt2-onnx/decoder_model.onnx",
                )))),
                decoder_with_past_resource: Some(Box::new(LocalResource::from(PathBuf::from(
                    "E:/Coding/distilgpt2-onnx/decoder_with_past_model.onnx",
                )))),
            }),
            config_resource: Box::new(LocalResource::from(PathBuf::from(
                "E:/Coding/distilgpt2-onnx/config.json",
            ))),
            vocab_resource: Box::new(LocalResource::from(PathBuf::from(
                "E:/Coding/distilgpt2-onnx/vocab.json",
            ))),
            merges_resource: Some(Box::new(LocalResource::from(PathBuf::from(
                "E:/Coding/distilgpt2-onnx/merges.txt",
            )))),
            num_beams: 3,
            do_sample: false,
            ..Default::default()
        },
        None,
        None,
    )?;

    let output = onnx_causal_generator.generate(Some(&["Today is a"]), None);
    println!("{:?}", output);

    Ok(())
}
