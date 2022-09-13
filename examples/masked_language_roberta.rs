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
use rust_bert::pipelines::common::ModelType;
use rust_bert::pipelines::masked_language::{MaskedLanguageConfig, MaskedLanguageModel};
use rust_bert::resources::RemoteResource;
use rust_bert::roberta::{
    RobertaConfigResources, RobertaMergesResources, RobertaModelResources, RobertaVocabResources,
};
fn main() -> anyhow::Result<()> {
    //    Set-up model
    let config = MaskedLanguageConfig::new(
        ModelType::Roberta,
        RemoteResource::from_pretrained(RobertaModelResources::DISTILROBERTA_BASE),
        None,
        RemoteResource::from_pretrained(RobertaConfigResources::DISTILROBERTA_BASE),
        RemoteResource::from_pretrained(RobertaVocabResources::DISTILROBERTA_BASE),
        RemoteResource::from_pretrained(RobertaMergesResources::DISTILROBERTA_BASE),
        true,
        None,
        None,
    );

    let mask_language_model = MaskedLanguageModel::new(config)?;
    //    Define input
    let input = [
        "Looks like one <mask> is missing!",
        "The goal of life is <mask>.",
    ];

    //    Run model
    let output = mask_language_model.predict(&input, vec![5, 6]);
    for word in output {
        println!("{:?}", word);
    }

    Ok(())
}
