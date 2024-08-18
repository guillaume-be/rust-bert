use rust_bert::pipelines::common::{ModelResource, ModelType, ONNXModelResources};
use rust_bert::pipelines::question_answering::{
    QaInput, QuestionAnsweringConfig, QuestionAnsweringModel,
};
use rust_bert::resources::RemoteResource;

fn main() -> anyhow::Result<()> {
    let qa_model = QuestionAnsweringModel::new(QuestionAnsweringConfig::new(
        ModelType::Roberta,
        ModelResource::ONNX(ONNXModelResources {
            encoder_resource: Some(Box::new(RemoteResource::new(
                "https://huggingface.co/optimum/roberta-base-squad2/resolve/main/model.onnx",
                "onnx-roberta-base-squad2",
            ))),
            ..Default::default()
        }),
        RemoteResource::new(
            "https://huggingface.co/optimum/roberta-base-squad2/resolve/main/config.json",
            "onnx-roberta-base-squad2",
        ),
        RemoteResource::new(
            "https://huggingface.co/optimum/roberta-base-squad2/resolve/main/vocab.json",
            "onnx-roberta-base-squad2",
        ),
        Some(RemoteResource::new(
            "https://huggingface.co/optimum/roberta-base-squad2/resolve/main/merges.txt",
            "onnx-roberta-base-squad2",
        )),
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
