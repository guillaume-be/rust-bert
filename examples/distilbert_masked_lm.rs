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
use tch::{Device, Tensor, nn, no_grad};
use rust_tokenizers::preprocessing::tokenizer::base_tokenizer::{Tokenizer, TruncationStrategy};
use rust_tokenizers::bert_tokenizer::BertTokenizer;
use rust_tokenizers::preprocessing::vocab::base_vocab::Vocab;
use failure::err_msg;
use rust_bert::distilbert::distilbert::{DistilBertConfig, DistilBertModelMaskedLM};
use rust_bert::Config;


fn main() -> failure::Fallible<()> {

//    Resources paths
    let mut home: PathBuf = dirs::home_dir().unwrap();
    home.push("rustbert");
    home.push("distilbert");
    let config_path = &home.as_path().join("config.json");
    let vocab_path = &home.as_path().join("vocab.txt");
    let weights_path = &home.as_path().join("model.ot");

    if !config_path.is_file() | !vocab_path.is_file() | !weights_path.is_file() {
        return Err(
            err_msg("Could not find required resources to run example. \
                          Please run ../utils/download_dependencies_distilbert.py \
                          in a Python environment with dependencies listed in ../requirements.txt"));
    }

//    Set-up masked LM model
    let device = Device::Cpu;
    let mut vs = nn::VarStore::new(device);
    let tokenizer: BertTokenizer = BertTokenizer::from_file(vocab_path.to_str().unwrap(), true);
    let config = DistilBertConfig::from_file(config_path);
    let distil_bert_model = DistilBertModelMaskedLM::new(&vs.root(), &config);
    vs.load(weights_path)?;

//    Define input
    let input = ["Looks like one thing is missing", "It\'s like comparing oranges to apples"];
    let tokenized_input = tokenizer.encode_list(input.to_vec(), 128, &TruncationStrategy::LongestFirst, 0);
    let max_len = tokenized_input.iter().map(|input| input.token_ids.len()).max().unwrap();
    let mut tokenized_input = tokenized_input.
        iter().
        map(|input| input.token_ids.clone()).
        map(|mut input| {
            input.extend(vec![0; max_len - input.len()]);
            input
        }).
        collect::<Vec<_>>();

//    Masking the token [thing] of sentence 1 and [oranges] of sentence 2
    tokenized_input[0][4] = 103;
    tokenized_input[1][6] = 103;
    let tokenized_input = tokenized_input.
        iter().
        map(|input|
            Tensor::of_slice(&(input))).
        collect::<Vec<_>>();
    let input_tensor = Tensor::stack(tokenized_input.as_slice(), 0).to(device);


//    Forward pass
    let (output, _, _) = no_grad(|| {
        distil_bert_model
            .forward_t(Some(input_tensor), None, None, false)
            .unwrap()
    });

//    Print masked tokens
    let index_1 = output.get(0).get(4).argmax(0, false);
    let index_2 = output.get(1).get(6).argmax(0, false);
    let word_1 = tokenizer.vocab().id_to_token(&index_1.int64_value(&[]));
    let word_2 = tokenizer.vocab().id_to_token(&index_2.int64_value(&[]));

    println!("{}", word_1); // Outputs "person" : "Looks like one [person] is missing"
    println!("{}", word_2);// Outputs "pear" : "It\'s like comparing [pear] to apples"

    Ok(())
}
