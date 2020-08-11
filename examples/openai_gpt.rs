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

extern crate anyhow;

use rust_bert::gpt2::Gpt2Config;
use rust_bert::openai_gpt::{
    OpenAIGPTLMHeadModel, OpenAiGptConfigResources, OpenAiGptMergesResources,
    OpenAiGptModelResources, OpenAiGptVocabResources,
};
use rust_bert::pipelines::generation::{Cache, LMHeadModel};
use rust_bert::resources::{download_resource, RemoteResource, Resource};
use rust_bert::Config;
use rust_tokenizers::{OpenAiGptTokenizer, Tokenizer, TruncationStrategy};
use tch::{nn, Device, Tensor};

fn main() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = Resource::Remote(RemoteResource::from_pretrained(
        OpenAiGptConfigResources::GPT,
    ));
    let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(
        OpenAiGptVocabResources::GPT,
    ));
    let merges_resource = Resource::Remote(RemoteResource::from_pretrained(
        OpenAiGptMergesResources::GPT,
    ));
    let weights_resource = Resource::Remote(RemoteResource::from_pretrained(
        OpenAiGptModelResources::GPT,
    ));
    let config_path = download_resource(&config_resource)?;
    let vocab_path = download_resource(&vocab_resource)?;
    let merges_path = download_resource(&merges_resource)?;
    let weights_path = download_resource(&weights_resource)?;

    //    Set-up masked LM model
    let device = Device::Cpu;
    let mut vs = nn::VarStore::new(device);
    let tokenizer = OpenAiGptTokenizer::from_file(
        vocab_path.to_str().unwrap(),
        merges_path.to_str().unwrap(),
        true,
    );
    let config = Gpt2Config::from_file(config_path);
    let openai_gpt = OpenAIGPTLMHeadModel::new(&vs.root(), &config);
    vs.load(weights_path)?;

    //    Define input
    let input = ["Wondering what the next word will"];
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
    let (output, _, _, _, _) = openai_gpt
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
