use rust_bert::pipelines::common::{ModelResource, ModelType, ONNXModelResources};
use rust_bert::pipelines::masked_language::{MaskedLanguageConfig, MaskedLanguageModel};
use rust_bert::resources::RemoteResource;

fn main() -> anyhow::Result<()> {
    let masked_lm = MaskedLanguageModel::new(MaskedLanguageConfig::new(
        ModelType::Bert,
        ModelResource::ONNX(ONNXModelResources {
            encoder_resource: Some(Box::new(RemoteResource::new(
                "https://huggingface.co/optimum/bert-base-uncased-for-masked-lm/resolve/main/model.onnx",
                "onnx-bert-base-uncased-for-masked-lm",
            ))),
            ..Default::default()
        }),
        RemoteResource::new(
            "https://huggingface.co/optimum/bert-base-uncased-for-masked-lm/resolve/main/config.json",
            "onnx-bert-base-uncased-for-masked-lm",
        ),
        RemoteResource::new(
            "https://huggingface.co/optimum/bert-base-uncased-for-masked-lm/resolve/main/vocab.txt",
            "onnx-bert-base-uncased-for-masked-lm",
        ),
        None,
        false,
        None,
        None,
        Some(String::from("<mask>")),
    ))?;
    let input = [
        "Hello I am a <mask> student",
        "Paris is the <mask> of France. It is <mask> in Europe.",
    ];
    let output = masked_lm.predict(input)?;
    println!("{:?}", output);
    Ok(())
}
