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

use rust_bert::mbart::{
    MBartConfigResources, MBartModelResources, MBartSourceLanguages, MBartTargetLanguages,
    MBartVocabResources,
};
use rust_bert::pipelines::common::ModelType;
use rust_bert::pipelines::translation::{Language, TranslationConfig, TranslationModel};
use rust_bert::resources::RemoteResource;
use tch::Device;

fn main() -> anyhow::Result<()> {
    let model_resource = RemoteResource::from_pretrained(MBartModelResources::MBART50_MANY_TO_MANY);
    let config_resource =
        RemoteResource::from_pretrained(MBartConfigResources::MBART50_MANY_TO_MANY);
    let vocab_resource = RemoteResource::from_pretrained(MBartVocabResources::MBART50_MANY_TO_MANY);
    let merges_resource =
        RemoteResource::from_pretrained(MBartVocabResources::MBART50_MANY_TO_MANY);

    let source_languages = MBartSourceLanguages::MBART50_MANY_TO_MANY;
    let target_languages = MBartTargetLanguages::MBART50_MANY_TO_MANY;

    let translation_config = TranslationConfig::new(
        ModelType::MBart,
        model_resource,
        config_resource,
        vocab_resource,
        merges_resource,
        source_languages,
        target_languages,
        Device::cuda_if_available(),
    );
    let model = TranslationModel::new(translation_config)?;

    let source_sentence = "This sentence will be translated in multiple languages.";

    let mut outputs = Vec::new();
    outputs.extend(model.translate(&[source_sentence], Language::English, Language::French)?);
    outputs.extend(model.translate(&[source_sentence], Language::English, Language::Spanish)?);
    outputs.extend(model.translate(&[source_sentence], Language::English, Language::Hindi)?);

    for sentence in outputs {
        println!("{}", sentence);
    }
    Ok(())
}
