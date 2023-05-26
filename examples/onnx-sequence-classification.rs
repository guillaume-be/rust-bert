use rust_bert::pipelines::common::{ModelResource, ModelType, ONNXModelResources};
use rust_bert::pipelines::sentiment::SentimentModel;
use rust_bert::pipelines::sequence_classification::SequenceClassificationConfig;
use rust_bert::resources::RemoteResource;

fn main() -> anyhow::Result<()> {
    let classification_model = SentimentModel::new(SequenceClassificationConfig::new(
        ModelType::DistilBert,
        ModelResource::ONNX(ONNXModelResources {
            encoder_resource: Some(Box::new(RemoteResource::new(
                "https://huggingface.co/optimum/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/model.onnx",
                "onnx-distilbert-base-uncased-finetuned-sst-2-english",
            ))),
            ..Default::default()
        }),
        RemoteResource::new(
            "https://huggingface.co/optimum/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/config.json",
            "onnx-distilbert-base-uncased-finetuned-sst-2-english",
        ),
        RemoteResource::new(
            "https://huggingface.co/optimum/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/vocab.txt",
            "onnx-distilbert-base-uncased-finetuned-sst-2-english",
        ),
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
