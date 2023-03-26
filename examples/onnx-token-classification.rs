use std::path::PathBuf;

use rust_bert::pipelines::common::{ModelResources, ModelType, ONNXModelResources};
use rust_bert::pipelines::pos_tagging::POSModel;
use rust_bert::pipelines::token_classification::{
    LabelAggregationOption, TokenClassificationConfig,
};
use rust_bert::resources::LocalResource;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let classification_model = POSModel::new(
        TokenClassificationConfig::new(
            ModelType::MobileBert,
            ModelResources::ONNX(ONNXModelResources {
                encoder_resource: Some(Box::new(LocalResource::from(PathBuf::from(
                    "E:/Coding/mobilebert-finetuned-pos/model.onnx",
                )))),
                ..Default::default()
            }),
            LocalResource::from(PathBuf::from(
                "E:/Coding/mobilebert-finetuned-pos/config.json",
            )),
            LocalResource::from(PathBuf::from(
                "E:/Coding/mobilebert-finetuned-pos/vocab.txt",
            )),
            None,
            true,
            Some(true),
            None,
            LabelAggregationOption::First,
        )
        .into(),
    )?;
    let input = [
        "My name is Amélie. My email is amelie@somemail.com.",
        "A liter of milk costs 0.95 Euros!",
    ];
    let output = classification_model.predict(&input);
    println!("{:?}", output);
    Ok(())
}
