extern crate failure;
extern crate dirs;

use std::path::PathBuf;
use tch::{Device, nn, Tensor, no_grad};
use rust_tokenizers::{BertTokenizer, TruncationStrategy, MultiThreadedTokenizer};
use rust_bert::bert::bert::{BertConfig, BertModel};
use rust_bert::common::config::Config;

fn main() -> failure::Fallible<()> {
    //    Resources paths
    let mut home: PathBuf = dirs::home_dir().unwrap();
    home.push("rustbert");
    home.push("bert");
    let config_path = &home.as_path().join("config.json");
    let vocab_path = &home.as_path().join("vocab.txt");
    let weights_path = &home.as_path().join("model.ot");

//    Set-up masked LM model
    let device = Device::Cpu;
    let mut vs = nn::VarStore::new(device);
    let tokenizer: BertTokenizer = BertTokenizer::from_file(vocab_path.to_str().unwrap());
    let config = BertConfig::from_file(config_path);


//    Define input
    let input = ["Looks like one thing is missing", "It\'s like comparing oranges to apples"];
    let tokenized_input = tokenizer.encode_list(input.to_vec(), 128, &TruncationStrategy::LongestFirst, 0);
    let max_len = tokenized_input.iter().map(|input| input.token_ids.len()).max().unwrap();
    let tokenized_input = tokenized_input.
        iter().
        map(|input| input.token_ids.clone()).
        map(|mut input| {
            input.extend(vec![0; max_len - input.len()]);
            input
        }).
        map(|input|
            Tensor::of_slice(&(input))).
        collect::<Vec<_>>();
    let input_tensor = Tensor::stack(tokenized_input.as_slice(), 0).to(device);

//    Forward pass

    let bert_model = BertModel::new(&vs.root(), &config);
    vs.load(weights_path)?;
    let output = no_grad(|| {
        bert_model
            .forward_t(Some(input_tensor),
                       None,
                       None,
                       None,
                       None,
                       &None,
                       &None,
                       false)
            .unwrap()
    });

    println!("{:?}", output);
    let (_hidden_state, _pooled_output, _, _) = output;
//    _hidden_state.print();
    _pooled_output.print();

    Ok(())
}