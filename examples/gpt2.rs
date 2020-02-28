// Copyright 2019-present, the HuggingFace Inc. team, The Google AI Language Team and Facebook, Inc.
// Copyright 2019 Guillaume Becquin
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

extern crate failure;
extern crate dirs;

use std::path::PathBuf;
use tch::{Device, nn, Tensor};
use rust_tokenizers::{TruncationStrategy, Tokenizer, Gpt2Tokenizer};
use rust_bert::gpt2::gpt2::{Gpt2Config, Gpt2Model};
use rust_bert::common::config::Config;
use tch::kind::Kind::Float;
use rust_bert::gpt2::transformer::Block;


fn main() -> failure::Fallible<()> {
    //    Resources paths
    let mut home: PathBuf = dirs::home_dir().unwrap();
    home.push("rustbert");
    home.push("distilgpt2");
    let config_path = &home.as_path().join("config.json");
    let vocab_path = &home.as_path().join("vocab.txt");
    let merges_path = &home.as_path().join("merges.txt");
    let _weights_path = &home.as_path().join("model.ot");

//    Set-up masked LM model
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let tokenizer: Gpt2Tokenizer = Gpt2Tokenizer::from_file(vocab_path.to_str().unwrap(), merges_path.to_str().unwrap(), true);
    let config = Gpt2Config::from_file(config_path);
    let _gpt2_model = Gpt2Model::new(&vs.root(), &config);
//    vs.load(weights_path)?;

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
    let _input_tensor = Tensor::stack(tokenized_input.as_slice(), 0).to(device);

//    Forward pass
    let attention = Block::new(&vs.root(), &config, false);
    let _input_tensor = Tensor::ones(&[32, 56, 768], (Float, vs.device()));

    let output = attention.forward_t(&_input_tensor, &None, &None, false);
    println!("{:?}", output);

    Ok(())
}