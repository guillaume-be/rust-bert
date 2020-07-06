// Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
// Copyright 2020 Guillaume Becquin
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

use rust_bert::resources::{download_resource, RemoteResource, Resource};
use rust_bert::t5::{
    T5Config, T5ConfigResources, T5ForConditionalGeneration, T5Model, T5ModelResources,
    T5VocabResources,
};
use rust_bert::Config;
use rust_tokenizers::preprocessing::tokenizer::t5_tokenizer::T5Tokenizer;
use rust_tokenizers::{Tokenizer, TruncationStrategy};
use tch::{nn, no_grad, Device, Kind, Tensor};

fn main() -> failure::Fallible<()> {
    //    Resources paths
    let config_resource =
        Resource::Remote(RemoteResource::from_pretrained(T5ConfigResources::T5_SMALL));
    let vocab_resource =
        Resource::Remote(RemoteResource::from_pretrained(T5VocabResources::T5_SMALL));
    let weights_resource =
        Resource::Remote(RemoteResource::from_pretrained(T5ModelResources::T5_SMALL));
    let config_path = download_resource(&config_resource)?;
    let vocab_path = download_resource(&vocab_resource)?;
    let weights_path = download_resource(&weights_resource)?;

    //    Set-up masked LM model
    let device = Device::Cpu;
    let mut vs = nn::VarStore::new(device);
    let tokenizer: T5Tokenizer = T5Tokenizer::from_file(vocab_path.to_str().unwrap(), false);
    let config = T5Config::from_file(config_path);

    let t5_model = T5ForConditionalGeneration::new(&vs.root(), &config, false, false);
    vs.load(weights_path)?;

    //    Define input
    let input = ["This is a test sentence"];
    let tokenized_input =
        tokenizer.encode_list(input.to_vec(), 128, &TruncationStrategy::LongestFirst, 0);
    let tokenized_input = tokenized_input
        .iter()
        .map(|input| input.token_ids.clone())
        .map(|input| Tensor::of_slice(&(input)))
        .collect::<Vec<_>>();
    let input_tensor = Tensor::stack(tokenized_input.as_slice(), 0).to(device);
    let decoder_inputs = Tensor::zeros(&[1, 1], (Kind::Int64, input_tensor.device()));
    //    Forward pass
    let output = no_grad(|| {
        t5_model.forward_t(
            Some(&input_tensor),
            None,
            None,
            Some(&decoder_inputs),
            None,
            None,
            None,
            None,
            false,
        )
    });
    output.0.print();

    Ok(())
}
