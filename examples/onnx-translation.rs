use std::path::PathBuf;
use tch::Device;

use rust_bert::pipelines::common::{ModelResources, ModelType, ONNXModelResources};
use rust_bert::pipelines::translation::{Language, TranslationConfig, TranslationModel};
use rust_bert::resources::LocalResource;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let translation_model = TranslationModel::new(TranslationConfig::new(
        ModelType::Marian,
        ModelResources::ONNX(ONNXModelResources {
            encoder_resource: Some(Box::new(LocalResource::from(PathBuf::from(
                "E:/Coding/opus-mt-en-fr-onnx/encoder_model.onnx",
            )))),
            decoder_resource: Some(Box::new(LocalResource::from(PathBuf::from(
                "E:/Coding/opus-mt-en-fr-onnx/decoder_model.onnx",
            )))),
            decoder_with_past_resource: Some(Box::new(LocalResource::from(PathBuf::from(
                "E:/Coding/opus-mt-en-fr-onnx/decoder_with_past_model.onnx",
            )))),
        }),
        LocalResource::from(PathBuf::from("E:/Coding/opus-mt-en-fr-onnx/config.json")),
        LocalResource::from(PathBuf::from("E:/Coding/opus-mt-en-fr-onnx/vocab.json")),
        Some(LocalResource::from(PathBuf::from(
            "E:/Coding/opus-mt-en-fr-onnx/source.spm",
        ))),
        &[Language::English],
        &[Language::French],
        Device::cuda_if_available(),
    ))?;

    let output = translation_model.translate(&["Hello my name is John."], None, None);
    println!("{:?}", output);
    Ok(())
}
