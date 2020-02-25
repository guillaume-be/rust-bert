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
    let answer = "The Tale of Timmy Tiptoes is a children's book written and illustrated by Beatrix Potter, and published by Frederick Warne & Co. in October 1911. Timmy Tiptoes is a squirrel believed to be a nut-thief by his fellows, and imprisoned by them in a hollow tree with the expectation that he will confess under confinement. Timmy is tended by Chippy Hackee, a friendly, mischievous chipmunk who has run away from his wife and is camping-out in the tree. Chippy urges the prisoner to eat the nuts stored in the tree, and Timmy does so but grows so fat he cannot escape the tree. He regains his freedom when a storm topples part of the tree. The tale contrasts the harmonious marriage of its title character with the less than harmonious marriage of the chipmunk.";

    let qa_example = QaExample::new(question, answer);
    println!("{:?}", qa_example);

    qa_model.generate_features(qa_example, 128, 45, 64);

    Ok(())
}

