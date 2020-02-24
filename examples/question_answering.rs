extern crate failure;
extern crate dirs;

use std::path::PathBuf;
use rust_bert::pipelines::question_answering::QaExample;


fn main() -> failure::Fallible<()> {
    //    Resources paths
    let mut home: PathBuf = dirs::home_dir().unwrap();
    home.push("rustbert");
    home.push("distilbert-qa");
    let _config_path = &home.as_path().join("config.json");
    let _vocab_path = &home.as_path().join("vocab.txt");
    let _weights_path = &home.as_path().join("model.ot");

//    Set-up masked LM model
    let question = "Where does Amy live ?";
    let answer = "Amy lives in Amsterdam.";

    let qa_example = QaExample::new(question, answer);
    println!("{:?}", qa_example);

    Ok(())
}

