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

use tch::{Device, nn, Tensor, no_grad};
use rust_tokenizers::{TruncationStrategy, Tokenizer, Vocab, RobertaTokenizer};
use rust_bert::Config;
use rust_bert::bert::BertConfig;
use rust_bert::roberta::{RobertaForMaskedLM, RobertaVocabResources, RobertaConfigResources, RobertaMergesResources, RobertaModelResources};
use rust_bert::common::resources::{Resource, download_resource, RemoteResource};


fn main() -> failure::Fallible<()> {
    //    Resources paths
    let config_resource = Resource::Remote(RemoteResource::from_pretrained(RobertaConfigResources::ROBERTA));
    let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(RobertaVocabResources::ROBERTA));
    let merges_resource = Resource::Remote(RemoteResource::from_pretrained(RobertaMergesResources::ROBERTA));
    let weights_resource = Resource::Remote(RemoteResource::from_pretrained(RobertaModelResources::ROBERTA));
    let config_path = download_resource(&config_resource)?;
    let vocab_path = download_resource(&vocab_resource)?;
    let merges_path = download_resource(&merges_resource)?;
    let weights_path = download_resource(&weights_resource)?;

//    Set-up masked LM model
    let device = Device::Cpu;
    let mut vs = nn::VarStore::new(device);
    let tokenizer: RobertaTokenizer = RobertaTokenizer::from_file(vocab_path.to_str().unwrap(), merges_path.to_str().unwrap(), true);
    let config = BertConfig::from_file(config_path);
    let bert_model = RobertaForMaskedLM::new(&vs.root(), &config);
    vs.load(weights_path)?;

//    Define input
    let input = ["<pad> Looks like one thing is missing", "It\'s like comparing oranges to apples"];
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
    tokenized_input[1][5] = 103;
    let tokenized_input = tokenized_input.
        iter().
        map(|input|
            Tensor::of_slice(&(input))).
        collect::<Vec<_>>();
    let input_tensor = Tensor::stack(tokenized_input.as_slice(), 0).to(device);

//    Forward pass
    let (output, _, _) = no_grad(|| {
        bert_model
            .forward_t(Some(input_tensor),
                       None,
                       None,
                       None,
                       None,
                       &None,
                       &None,
                       false)
    });

//    Print masked tokens
    let index_1 = output.get(0).get(4).argmax(0, false);
    let index_2 = output.get(1).get(5).argmax(0, false);
    let word_1 = tokenizer.vocab().id_to_token(&index_1.int64_value(&[]));
    let word_2 = tokenizer.vocab().id_to_token(&index_2.int64_value(&[]));

    println!("{}", word_1); // Outputs "some" : "Looks like [some] thing is missing"
    println!("{}", word_2);// Outputs "apple" : "It\'s like comparing [apple] to apples"

    Ok(())
}