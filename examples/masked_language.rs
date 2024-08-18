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
use rust_bert::bert::{BertConfigResources, BertModelResources, BertVocabResources};
use rust_bert::pipelines::common::{ModelResource, ModelType};
use rust_bert::pipelines::masked_language::{MaskedLanguageConfig, MaskedLanguageModel};
use rust_bert::resources::RemoteResource;
fn main() -> anyhow::Result<()> {
    //    Set-up model
    let config = MaskedLanguageConfig::new(
        ModelType::Bert,
        ModelResource::Torch(Box::new(RemoteResource::from_pretrained(
            BertModelResources::BERT,
        ))),
        RemoteResource::from_pretrained(BertConfigResources::BERT),
        RemoteResource::from_pretrained(BertVocabResources::BERT),
        None,
        true,
        None,
        None,
        Some(String::from("<mask>")),
    );

    let mask_language_model = MaskedLanguageModel::new(config)?;
    //    Define input
    let input = [
        "Hello I am a <mask> student",
        "Paris is the <mask> of France. It is <mask> in Europe.",
    ];

    //    Run model
    let output = mask_language_model.predict(input)?;
    for sentence_output in output {
        println!("{sentence_output:?}");
    }

    Ok(())
}
