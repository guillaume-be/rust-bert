// Copyright 2018 Google AI and Google Brain team.
// Copyright 2018 Carnegie Mellon University Authors.
// Copyright 2020-present, the HuggingFace Inc. team.
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

extern crate anyhow;

use rust_bert::resources::{RemoteResource, Resource};
use rust_bert::xlnet::{
    XLNetConfig, XLNetConfigResources, XLNetLMHeadModel, XLNetModelResources, XLNetVocabResources,
};
use rust_bert::Config;
use rust_tokenizers::tokenizer::{MultiThreadedTokenizer, TruncationStrategy, XLNetTokenizer};
use rust_tokenizers::vocab::Vocab;
use tch::{nn, no_grad, Device, Kind, Tensor};

fn main() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = Resource::Remote(RemoteResource::from_pretrained(
        XLNetConfigResources::XLNET_BASE_CASED,
    ));
    let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(
        XLNetVocabResources::XLNET_BASE_CASED,
    ));
    let weights_resource = Resource::Remote(RemoteResource::from_pretrained(
        XLNetModelResources::XLNET_BASE_CASED,
    ));
    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;
    let weights_path = weights_resource.get_local_path()?;

    //    Set-up masked LM model
    let device = Device::cuda_if_available();
    let mut vs = nn::VarStore::new(device);
    let tokenizer: XLNetTokenizer =
        XLNetTokenizer::from_file(vocab_path.to_str().unwrap(), false, true)?;
    let config = XLNetConfig::from_file(config_path);
    let xlnet_model = XLNetLMHeadModel::new(&vs.root(), &config);
    vs.load(weights_path)?;

    //    Define input
    let input = ["One two three four"];
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
        .map(|input| Tensor::of_slice(&(input[..input.len() - 2])))
        .collect::<Vec<_>>();
    let input_tensor = Tensor::stack(tokenized_input.as_slice(), 0).to(device);

    // Forward pass
    let perm_mask = Tensor::zeros(&[1, 4, 4], (Kind::Float, device));
    let _ = perm_mask.narrow(2, 3, 1).fill_(1.0);

    let target_mapping = Tensor::zeros(&[1, 1, 4], (Kind::Float, device));
    let _ = target_mapping.narrow(2, 3, 1).fill_(1.0);
    let model_output = no_grad(|| {
        xlnet_model
            .forward_t(
                Some(&input_tensor),
                None,
                None,
                Some(perm_mask.as_ref()),
                Some(target_mapping.as_ref()),
                None,
                None,
                false,
            )
            .unwrap()
    });

    let index_1 = model_output
        .lm_logits
        .get(0)
        .argmax(1, false)
        .int64_value(&[]);
    let score_1 = model_output.lm_logits.double_value(&[0, 0, index_1]);
    let word_1 = tokenizer.vocab().id_to_token(&index_1);
    println!("{}, {}, {}", index_1, score_1, word_1);
    Ok(())
}
