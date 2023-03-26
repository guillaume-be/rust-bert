use std::path::PathBuf;

use rust_bert::pipelines::common::{ModelResources, ModelType, ONNXModelResources};
use rust_bert::pipelines::question_answering::{
    QaInput, QuestionAnsweringConfig, QuestionAnsweringModel,
};
use rust_bert::resources::LocalResource;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let qa_model = QuestionAnsweringModel::new(QuestionAnsweringConfig::new(
        ModelType::DistilBert,
        ModelResources::ONNX(ONNXModelResources {
            encoder_resource: Some(Box::new(LocalResource::from(PathBuf::from(
                "E:/Coding/distilbert-base-cased-distilled-squad/model.onnx",
            )))),
            ..Default::default()
        }),
        LocalResource::from(PathBuf::from(
            "E:/Coding/distilbert-base-cased-distilled-squad/config.json",
        )),
        LocalResource::from(PathBuf::from(
            "E:/Coding/distilbert-base-cased-distilled-squad/vocab.txt",
        )),
        None,
        false,
        None,
        None,
    ))?;
    let question = String::from("Where does Amy live ?");
    let context = String::from("Amy lives in Amsterdam");
    let qa_input = QaInput { question, context };

    let output = qa_model.predict(&[qa_input], 1, 32);
    println!("{:?}", output);
    Ok(())
}
