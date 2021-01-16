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

use rust_bert::prophetnet::{
    ProphetNetConfig, ProphetNetConfigResources, ProphetNetModel, ProphetNetModelResources,
    ProphetNetVocabResources,
};
use rust_bert::resources::{RemoteResource, Resource};
use rust_bert::Config;
use rust_tokenizers::tokenizer::{ProphetNetTokenizer, Tokenizer, TruncationStrategy};
use tch::{nn, no_grad, Device, Tensor};

fn main() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = Resource::Remote(RemoteResource::from_pretrained(
        ProphetNetConfigResources::PROPHETNET_LARGE_UNCASED,
    ));
    let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(
        ProphetNetVocabResources::PROPHETNET_LARGE_UNCASED,
    ));
    let weights_resource = Resource::Remote(RemoteResource::from_pretrained(
        ProphetNetModelResources::PROPHETNET_LARGE_UNCASED,
    ));
    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;
    let weights_path = weights_resource.get_local_path()?;

    //    Set-up masked LM model
    let device = Device::cuda_if_available();
    let mut vs = nn::VarStore::new(device);
    let tokenizer = ProphetNetTokenizer::from_file(vocab_path.to_str().unwrap(), true, true)?;
    let config = ProphetNetConfig::from_file(config_path);

    let model = ProphetNetModel::new(&(&vs.root() / "prophetnet"), &config)?;
    vs.load(weights_path)?;

    //    Define input
    let input = ["One two three four"];

    let tokenized_input = tokenizer.encode_list(input, 1024, &TruncationStrategy::LongestFirst, 0);
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
    let model_output = no_grad(|| {
        model
            .forward_t(
                Some(&input_tensor),
                None,
                None,
                Some(&input_tensor),
                None,
                None,
                None,
                None,
                None,
                false,
            )
            .unwrap()
    });

    println!("{:?}", model_output.last_hidden_states);
    model_output.last_hidden_states.print();
    Ok(())
}
