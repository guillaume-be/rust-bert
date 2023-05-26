use rust_bert::pipelines::common::{ModelResource, ModelType, ONNXModelResources};
use rust_bert::pipelines::ner::NERModel;
use rust_bert::pipelines::token_classification::{
    LabelAggregationOption, TokenClassificationConfig,
};
use rust_bert::resources::RemoteResource;

fn main() -> anyhow::Result<()> {
    let token_classification_model = NERModel::new(TokenClassificationConfig::new(
        ModelType::Bert,
        ModelResource::ONNX(ONNXModelResources {
            encoder_resource: Some(Box::new(RemoteResource::new(
                "https://huggingface.co/optimum/bert-base-NER/resolve/main/model.onnx",
                "onnx-bert-base-NER",
            ))),
            ..Default::default()
        }),
        RemoteResource::new(
            "https://huggingface.co/optimum/bert-base-NER/resolve/main/config.json",
            "onnx-bert-base-NER",
        ),
        RemoteResource::new(
            "https://huggingface.co/optimum/bert-base-NER/resolve/main/vocab.txt",
            "onnx-bert-base-NER",
        ),
        None,
        false,
        None,
        None,
        LabelAggregationOption::First,
    ))?;
    let input = ["Asked John Smith about Acme Corp", "Let's go to New York!"];
    let output = token_classification_model.predict_full_entities(&input);
    println!("{:?}", output);
    Ok(())
}
