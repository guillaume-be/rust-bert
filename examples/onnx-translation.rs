use rust_bert::m2m_100::{M2M100SourceLanguages, M2M100TargetLanguages};
use tch::Device;

use rust_bert::pipelines::common::{ModelResource, ModelType, ONNXModelResources};
use rust_bert::pipelines::translation::{Language, TranslationConfig, TranslationModel};
use rust_bert::resources::RemoteResource;

fn main() -> anyhow::Result<()> {
    let translation_model = TranslationModel::new(TranslationConfig::new(
        ModelType::M2M100,
        ModelResource::ONNX(ONNXModelResources {
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
        M2M100SourceLanguages::M2M100_418M,
        M2M100TargetLanguages::M2M100_418M,
        Device::cuda_if_available(),
    ))?;

    let source_sentence = "This sentence will be translated in multiple languages.";

    let mut outputs = Vec::new();
    outputs.extend(translation_model.translate(
        &[source_sentence],
        Language::English,
        Language::French,
    )?);
    outputs.extend(translation_model.translate(
        &[source_sentence],
        Language::English,
        Language::Spanish,
    )?);
    outputs.extend(translation_model.translate(
        &[source_sentence],
        Language::English,
        Language::Hindi,
    )?);

    println!("{:?}", outputs);
    Ok(())
}
