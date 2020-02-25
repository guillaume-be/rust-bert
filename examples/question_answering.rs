extern crate failure;
extern crate dirs;

use std::path::PathBuf;
use rust_bert::pipelines::question_answering::QuestionAnsweringModel;
use tch::Device;


fn main() -> failure::Fallible<()> {
    //    Resources paths
    let mut home: PathBuf = dirs::home_dir().unwrap();
    home.push("rustbert");
    home.push("distilbert-qa");
    let config_path = &home.as_path().join("config.json");
    let vocab_path = &home.as_path().join("vocab.txt");
    let weights_path = &home.as_path().join("model.ot");

//    Set-up Question Answering model
    let device = Device::Cpu;
    let qa_model = QuestionAnsweringModel::new(vocab_path,
                                               config_path,
                                               weights_path, device)?;

//    Define input
    let question = "Where does Amy live ?";
    let context = "Amy lives in Amsterdam";

//    Get answer
    let answers = qa_model.predict(question, context, 1);
    println!("{:?}", answers);
    Ok(())
}