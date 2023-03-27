use tch::Device;

use rust_bert::pipelines::common::{ModelResources, ModelType, ONNXModelResources};
use rust_bert::pipelines::translation::{Language, TranslationConfig, TranslationModel};
use rust_bert::resources::RemoteResource;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let translation_model = TranslationModel::new(TranslationConfig::new(
        ModelType::M2M100,
        ModelResources::ONNX(ONNXModelResources {
            encoder_resource: Some(Box::new(RemoteResource::new(
            "https://huggingface.co/optimum/m2m100_418M/resolve/main/encoder_model.onnx",
            "onnx-m2m100_418M",
            ))),
            decoder_resource: Some(Box::new(RemoteResource::new(
            "https://huggingface.co/optimum/m2m100_418M/resolve/main/decoder_model.onnx",
            "onnx-m2m100_418M",
            ))),
            decoder_with_past_resource: Some(Box::new(RemoteResource::new(
            "https://huggingface.co/optimum/m2m100_418M/resolve/main/decoder_with_past_model.onnx",
            "onnx-m2m100_418M",
            ))),
        }),
        RemoteResource::new(
            "https://huggingface.co/optimum/m2m100_418M/resolve/main/config.json",
            "onnx-m2m100_418M",
        ),
        RemoteResource::new(
            "https://huggingface.co/optimum/m2m100_418M/resolve/main/vocab.json",
    "onnx-m2m100_418M",
        ),
        Some(RemoteResource::new(
    "https://huggingface.co/optimum/m2m100_418M/resolve/main/sentencepiece.bpe.model",
    "onnx-m2m100_418M",
    )),
        &[Language::English],
        &[Language::French],
        Device::cuda_if_available(),
    ))?;

    let output = translation_model.translate(
        &["Hello my name is John."],
        Language::English,
        Language::French,
    )?;
    println!("{:?}", output);
    Ok(())
}
