use std::path::PathBuf;

use rust_bert::pipelines::common::{ModelResources, ModelType, ONNXModelResources};
use rust_bert::pipelines::zero_shot_classification::{
    ZeroShotClassificationConfig, ZeroShotClassificationModel,
};
use rust_bert::resources::LocalResource;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let classification_model =
        ZeroShotClassificationModel::new(ZeroShotClassificationConfig::new(
            ModelType::Deberta,
            ModelResources::ONNX(ONNXModelResources {
                encoder_resource: Some(Box::new(LocalResource::from(PathBuf::from(
                    "E:/Coding/deberta-base-mnli/model.onnx",
                )))),
                ..Default::default()
            }),
            LocalResource::from(PathBuf::from("E:/Coding/ddeberta-base-mnli/config.json")),
            LocalResource::from(PathBuf::from("E:/Coding/deberta-base-mnli/vocab.json")),
            Some(LocalResource::from(PathBuf::from(
                "E:/Coding/deberta-base-mnli/merges.txt",
            ))),
            false,
            None,
            None,
        ))?;

    let input_sentence = "Who are you voting for in 2020?";
    let input_sequence_2 = "The prime minister has announced a stimulus package which was widely criticized by the opposition.";
    let candidate_labels = &["politics", "public health", "economy", "sports"];

    let output = classification_model.predict(
        [input_sentence, input_sequence_2],
        candidate_labels,
        Some(Box::new(|label: &str| {
            format!("This example is about {label}.")
        })),
        128,
    )?;
    println!("{:?}", output);
    Ok(())
}
