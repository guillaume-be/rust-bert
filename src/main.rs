use crate::distilbert::{DistilBertConfig, embeddings};
use std::path::Path;
use std::env;
use std::sync::Arc;
use rust_transformers::preprocessing::vocab::base_vocab::Vocab;
use rust_transformers::bert_tokenizer::BertTokenizer;
use rust_transformers::preprocessing::tokenizer::base_tokenizer::{Tokenizer, TruncationStrategy};
use tch::{Device, nn, Tensor};
mod distilbert;

fn main() {

//    Config & set-up var store
    let config_path = env::var("distilbert_config_path").unwrap();
    let config_path = Path::new(&config_path);
    let config = DistilBertConfig::from_file(config_path);
//    println!("{:?}", config);
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);

//    Creation of tokenizer
    let vocab_path = "E:/Coding/backup-rust/rust-transformers/resources/vocab/bert-base-uncased-vocab.txt";
    let vocab = Arc::new(rust_transformers::BertVocab::from_file(vocab_path));
    let tokenizer: BertTokenizer = BertTokenizer::from_existing_vocab(vocab.clone());

//    Creation of sample input for testing purposes
    let input = "Hello, world! This is a tokenization test";
    let tokenized_input = tokenizer.encode(input, None, 128, &TruncationStrategy::LongestFirst, 0);

//    Pass tokenized input through embeddings
    let embeddings = embeddings(vs.root(), config);
    let input_tensor = Tensor::of_slice(&tokenized_input.token_ids).unsqueeze(0).to(device);

    let output = input_tensor.apply_t(&embeddings, true);
    println!("{:?}", output);

}
