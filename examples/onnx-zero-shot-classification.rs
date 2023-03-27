use rust_bert::pipelines::common::{ModelResources, ModelType, ONNXModelResources};
use rust_bert::pipelines::zero_shot_classification::{
    ZeroShotClassificationConfig, ZeroShotClassificationModel,
};
use rust_bert::resources::RemoteResource;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let classification_model =
        ZeroShotClassificationModel::new(ZeroShotClassificationConfig::new(
            ModelType::DistilBert,
            ModelResources::ONNX(ONNXModelResources {
                encoder_resource: Some(Box::new(RemoteResource::new(
                    "https://huggingface.co/optimum/optimum/distilbert-base-uncased-mnli/resolve/main/model.onnx",
                    "onnx-distilbert-base-uncased-mnli",
                ))),
                ..Default::default()
            }),
            RemoteResource::new(
                "https://huggingface.co/optimum/optimum/distilbert-base-uncased-mnli/resolve/main/config.json",
                "onnx-distilbert-base-uncased-mnli",
            ),
            RemoteResource::new(
                "https://huggingface.co/optimum/optimum/distilbert-base-uncased-mnli/resolve/main/vocab.txt",
                "onnx-distilbert-base-uncased-mnli",
            ),
            None,
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
