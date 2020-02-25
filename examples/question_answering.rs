extern crate failure;
extern crate dirs;

use std::path::PathBuf;
use rust_bert::pipelines::question_answering::{QaExample, QuestionAnsweringModel};
use tch::Device;


fn main() -> failure::Fallible<()> {
    //    Resources paths
    let mut home: PathBuf = dirs::home_dir().unwrap();
    home.push("rustbert");
    home.push("distilbert-qa");
    let config_path = &home.as_path().join("config.json");
    let vocab_path = &home.as_path().join("vocab.txt");
    let weights_path = &home.as_path().join("model.ot");

//    Set-up masked LM model
    let device = Device::Cpu;
    let qa_model = QuestionAnsweringModel::new(vocab_path, config_path, weights_path, device)?;


//    Define input
    let question = "Where does Amy live ?";
    let answer = "Amy lives in Amsterdam.";

    let qa_example = QaExample::new(question, answer);
    println!("{:?}", qa_example);

    qa_model.generate_features(qa_example, 384, 128, 64);

    Ok(())
}

