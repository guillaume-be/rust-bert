use std::path::PathBuf;

use rust_bert::pipelines::common::{ModelResources, ModelType, ONNXModelResources};
use rust_bert::pipelines::masked_language::{MaskedLanguageConfig, MaskedLanguageModel};
use rust_bert::resources::LocalResource;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let masked_lm = MaskedLanguageModel::new(MaskedLanguageConfig::new(
        ModelType::Roberta,
        ModelResources::ONNX(ONNXModelResources {
            encoder_resource: Some(Box::new(LocalResource::from(PathBuf::from(
                "E:/Coding/distilroberta-base/model.onnx",
            )))),
            ..Default::default()
        }),
        LocalResource::from(PathBuf::from("E:/Coding/distilroberta-base/config.json")),
        LocalResource::from(PathBuf::from("E:/Coding/distilroberta-base/vocab.json")),
        Some(LocalResource::from(PathBuf::from(
            "E:/Coding/distilroberta-base/merges.txt",
        ))),
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
