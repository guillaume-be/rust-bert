use std::path::PathBuf;

use rust_bert::pipelines::common::{ModelResources, ModelType, ONNXModelResources};
use rust_bert::pipelines::sequence_classification::{
    SequenceClassificationConfig, SequenceClassificationModel,
};
use rust_bert::resources::LocalResource;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let classification_model =
        SequenceClassificationModel::new(SequenceClassificationConfig::new(
            ModelType::DistilBert,
            ModelResources::ONNX(ONNXModelResources {
                encoder_resource: Some(Box::new(LocalResource::from(PathBuf::from(
                    "E:/Coding/distilbert-base-uncased-finetuned-sst-2-english/model.onnx",
                )))),
                ..Default::default()
            }),
            LocalResource::from(PathBuf::from(
                "E:/Coding/distilbert-base-uncased-finetuned-sst-2-english/config.json",
            )),
            LocalResource::from(PathBuf::from(
                "E:/Coding/distilbert-base-uncased-finetuned-sst-2-english/vocab.txt",
            )),
            None,
            true,
            None,
            None,
        ))?;
    let input = [
        "Probably my all-time favorite movie, a story of selflessness, sacrifice and dedication to a noble cause, but it's not preachy or boring.",
        "This film tried to be too many things all at once: stinging political satire, Hollywood blockbuster, sappy romantic comedy, family values promo...",
        "If you like original gut wrenching laughter you will like this movie. If you are young or old then you will love this movie, hell even my mom liked it.",
    ];
    let output = classification_model.predict(input);
    println!("{:?}", output);
    Ok(())
}
