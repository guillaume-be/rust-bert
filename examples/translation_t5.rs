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

extern crate anyhow;

use rust_bert::pipelines::common::ModelType;
use rust_bert::pipelines::translation::{Language, TranslationConfig, TranslationModel};
use rust_bert::resources::RemoteResource;
use rust_bert::t5::{T5ConfigResources, T5ModelResources, T5VocabResources};
use tch::Device;

fn main() -> anyhow::Result<()> {
    let model_resource = RemoteResource::from_pretrained(T5ModelResources::T5_BASE);
    let config_resource = RemoteResource::from_pretrained(T5ConfigResources::T5_BASE);
    let vocab_resource = RemoteResource::from_pretrained(T5VocabResources::T5_BASE);
    let merges_resource = RemoteResource::from_pretrained(T5VocabResources::T5_BASE);

    let source_languages = [
        Language::English,
        Language::French,
        Language::German,
        Language::Romanian,
    ];
    let target_languages = [
        Language::English,
        Language::French,
        Language::German,
        Language::Romanian,
    ];

    let translation_config = TranslationConfig::new(
        ModelType::T5,
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
    outputs.extend(model.translate(&[source_sentence], Language::English, Language::German)?);
    outputs.extend(model.translate(&[source_sentence], Language::English, Language::Romanian)?);

    for sentence in outputs {
        println!("{}", sentence);
    }
    Ok(())
}
