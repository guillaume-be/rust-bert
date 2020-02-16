extern crate failure;
extern crate dirs;

use std::path::PathBuf;
use tch::{Device, nn, Tensor, no_grad};
use rust_tokenizers::{BertTokenizer, TruncationStrategy, MultiThreadedTokenizer};
use rust_bert::bert::bert::BertConfig;
use rust_bert::bert::embeddings::BertEmbeddings;
use rust_bert::common::config::Config;
use rust_bert::bert::attention::BertSelfAttention;

fn main() -> failure::Fallible<()> {
    //    Resources paths
    let mut home: PathBuf = dirs::home_dir().unwrap();
    home.push("rustbert");
    home.push("bert");
    let config_path = &home.as_path().join("config.json");
    let vocab_path = &home.as_path().join("vocab.txt");
    let _weights_path = &home.as_path().join("model.ot");

//    Set-up masked LM model
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
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

    let embeddings = BertEmbeddings::new(&vs.root(), &config);
    let bert_self_attention = BertSelfAttention::new(vs.root(), &config);

    let output = no_grad(|| {
        embeddings
            .forward_t(Some(input_tensor), None, None, None, false)
            .unwrap()
    });

    println!("{:?}", output);

    let output = no_grad(|| {
        bert_self_attention
            .forward_t(&output, &None, &None, &None, false)
    });

    println!("{:?}", output);

    Ok(())
}