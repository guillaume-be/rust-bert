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

use rust_bert::gpt2::{
    GPT2LMHeadModel, Gpt2Config, Gpt2ConfigResources, Gpt2MergesResources, Gpt2ModelResources,
    Gpt2VocabResources,
};
use rust_bert::pipelines::generation::{Cache, LMHeadModel};
use rust_bert::resources::{download_resource, RemoteResource, Resource};
use rust_bert::Config;
use rust_tokenizers::{Gpt2Tokenizer, Tokenizer, TruncationStrategy};
use tch::{nn, Device, Tensor};

fn main() -> failure::Fallible<()> {
    //    Resources set-up
    let config_resource =
        Resource::Remote(RemoteResource::from_pretrained(Gpt2ConfigResources::GPT2));
    let vocab_resource =
        Resource::Remote(RemoteResource::from_pretrained(Gpt2VocabResources::GPT2));
    let merges_resource =
        Resource::Remote(RemoteResource::from_pretrained(Gpt2MergesResources::GPT2));
    let weights_resource =
        Resource::Remote(RemoteResource::from_pretrained(Gpt2ModelResources::GPT2));
    let config_path = download_resource(&config_resource)?;
    let vocab_path = download_resource(&vocab_resource)?;
    let merges_path = download_resource(&merges_resource)?;
    let weights_path = download_resource(&weights_resource)?;

    //    Set-up masked LM model
    let device = Device::Cpu;
    let mut vs = nn::VarStore::new(device);
    let tokenizer: Gpt2Tokenizer = Gpt2Tokenizer::from_file(
        vocab_path.to_str().unwrap(),
        merges_path.to_str().unwrap(),
        false,
    );
    let config = Gpt2Config::from_file(config_path);
    let gpt2_model = GPT2LMHeadModel::new(&vs.root(), &config);
    vs.load(weights_path)?;

    //    Define input
    let input = ["One two three four five six seven eight nine ten eleven"];
    let tokenized_input =
        tokenizer.encode_list(input.to_vec(), 128, &TruncationStrategy::LongestFirst, 0);
    let max_len = tokenized_input
        .iter()
        .map(|input| input.token_ids.len())
        .max()
        .unwrap();
    let tokenized_input = tokenized_input
        .iter()
        .map(|input| input.token_ids.clone())
        .map(|mut input| {
            input.extend(vec![0; max_len - input.len()]);
            input
        })
        .map(|input| Tensor::of_slice(&(input)))
        .collect::<Vec<_>>();
    let input_tensor = Tensor::stack(tokenized_input.as_slice(), 0).to(device);

    //    Forward pass
    let (output, _, _, _, _) = gpt2_model
        .forward_t(
            &Some(input_tensor),
            Cache::None,
            &None,
            &None,
            &None,
            &None,
            None,
            &None,
            false,
        )
        .unwrap();

    let next_word_id = output.get(0).get(-1).argmax(-1, true).int64_value(&[0]);
    let next_word = tokenizer.decode(vec![next_word_id], true, true);
    println!("Provided input: {}", input[0]);
    println!("Next word: {}", next_word);

    Ok(())
}
