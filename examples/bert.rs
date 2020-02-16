extern crate failure;
extern crate dirs;

use std::path::PathBuf;
use tch::{Device, nn};
use rust_tokenizers::BertTokenizer;
use rust_bert::bert::bert::BertConfig;
use rust_bert::common::config::Config;

fn main() -> failure::Fallible<()> {
    //    Resources paths
    let mut home: PathBuf = dirs::home_dir().unwrap();
    home.push("rustbert");
    home.push("bert");
    let config_path = &home.as_path().join("config.json");
    let vocab_path = &home.as_path().join("vocab.txt");
    let _weights_path = &home.as_path().join("model.ot");

    let device = Device::Cpu;
    let _vs = nn::VarStore::new(device);
    let _tokenizer: BertTokenizer = BertTokenizer::from_file(vocab_path.to_str().unwrap());
    let _config = BertConfig::from_file(config_path);


    Ok(())
}