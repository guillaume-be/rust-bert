// Copyright 2020 The Google Research Authors.
// Copyright 2019-present, the HuggingFace Inc. team
// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

use rust_bert::electra::{
    ElectraConfig, ElectraConfigResources, ElectraDiscriminator, ElectraModelResources,
    ElectraVocabResources,
};
use rust_bert::resources::{download_resource, RemoteResource, Resource};
use rust_bert::Config;
use rust_tokenizers::{BertTokenizer, Tokenizer, TruncationStrategy};
use tch::{nn, no_grad, Device, Tensor};

fn main() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = Resource::Remote(RemoteResource::from_pretrained(
        ElectraConfigResources::BASE_DISCRIMINATOR,
    ));
    let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(
        ElectraVocabResources::BASE_DISCRIMINATOR,
    ));
    let weights_resource = Resource::Remote(RemoteResource::from_pretrained(
        ElectraModelResources::BASE_DISCRIMINATOR,
    ));
    let config_path = download_resource(&config_resource)?;
    let vocab_path = download_resource(&vocab_resource)?;
    let weights_path = download_resource(&weights_resource)?;

    //    Set-up masked LM model
    let device = Device::Cpu;
    let mut vs = nn::VarStore::new(device);
    let tokenizer: BertTokenizer = BertTokenizer::from_file(vocab_path.to_str().unwrap(), true)?;
    let config = ElectraConfig::from_file(config_path);
    let electra_model = ElectraDiscriminator::new(&vs.root(), &config);
    vs.load(weights_path)?;

    //    Define input
    let input = ["One Two Three Ten Five Six Seven Eight"];
    let tokenized_input =
        tokenizer.encode_list(input.to_vec(), 128, &TruncationStrategy::LongestFirst, 0);
    let max_len = tokenized_input
        .iter()
        .map(|input| input.token_ids.len())
        .max()
        .unwrap();
    let encoded_input = tokenized_input
        .iter()
        .map(|input| input.token_ids.clone())
        .map(|mut input| {
            input.extend(vec![0; max_len - input.len()]);
            input
        })
        .map(|input| Tensor::of_slice(&(input)))
        .collect::<Vec<_>>();
    let input_tensor = Tensor::stack(encoded_input.as_slice(), 0).to(device);

    //    Forward pass
    let (output, _, _) =
        no_grad(|| electra_model.forward_t(Some(input_tensor), None, None, None, None, false));

    //    Print model predictions
    for (position, token) in tokenized_input[0].token_ids.iter().enumerate() {
        let probability = output.double_value(&[position as i64]);
        let generated = if probability > 0.5 {
            "generated"
        } else {
            "original"
        };
        println!(
            "{:?}: {} ({:.1}%)",
            tokenizer.decode([*token].to_vec(), false, false),
            generated,
            100f64 * probability
        )
    }

    Ok(())
}
