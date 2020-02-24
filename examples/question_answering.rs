extern crate failure;
extern crate dirs;

use std::path::PathBuf;
use tch::{Device, nn};
use rust_tokenizers::{BertTokenizer};
use rust_bert::pipelines::question_answering::QaExample;
//use rust_bert::{DistilBertForQuestionAnswering, DistilBertConfig};


fn main() -> failure::Fallible<()> {
    //    Resources paths
    let mut home: PathBuf = dirs::home_dir().unwrap();
    home.push("rustbert");
    home.push("distilbert-qa");
    let _config_path = &home.as_path().join("config.json");
    let vocab_path = &home.as_path().join("vocab.txt");
    let _weights_path = &home.as_path().join("model.ot");

//    Set-up masked LM model
    let device = Device::Cpu;
    let _vs = nn::VarStore::new(device);
    let _tokenizer: BertTokenizer = BertTokenizer::from_file(vocab_path.to_str().unwrap(), true);
//    let config = DistilBertConfig::from_file(config_path);
//    let _distilbert_model = DistilBertForQuestionAnswering::new(&vs.root(), &config);
//    vs.load(weights_path);

//    Define input
//    let input = [
//        "Looks like one thing is missing", "It\'s like comparing oranges to apples"
//    ];
//    let tokenized_input = tokenizer.encode_list(input.to_vec(), 128, &TruncationStrategy::LongestFirst, 0);
//    let max_len = tokenized_input.iter().map(|input| input.token_ids.len()).max().unwrap();
//    let tokenized_input = tokenized_input.
//        iter().
//        map(|input| input.token_ids.clone()).
//        map(|mut input| {
//            input.extend(vec![0; max_len - input.len()]);
//            input
//        }).
//        map(|input|
//            Tensor::of_slice(&(input))).
//        collect::<Vec<_>>();
//    let _input_tensor = Tensor::stack(tokenized_input.as_slice(), 0).to(device);

    let question = "Where does Amy live ?";
    let answer = "Amy lives in Amsterdam.";

    let qa_example = QaExample::new(question, answer);
    println!("{:?}", qa_example);

    Ok(())
}

