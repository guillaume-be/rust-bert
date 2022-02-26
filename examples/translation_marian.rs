// Copyright 2018-2020 The HuggingFace Inc. team.
// Copyright 2020 Marian Team Authors
// Copyright 2019-2020 Guillaume Becquin
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

use rust_bert::marian::{
    MarianConfigResources, MarianModelResources, MarianSourceLanguages, MarianSpmResources,
    MarianTargetLanguages, MarianVocabResources,
};
use rust_bert::pipelines::common::ModelType;
use rust_bert::pipelines::translation::{TranslationConfig, TranslationModel};
use rust_bert::resources::RemoteResource;
use tch::Device;

fn main() -> anyhow::Result<()> {
    let model_resource = RemoteResource::from_pretrained(MarianModelResources::ENGLISH2CHINESE);
    let config_resource = RemoteResource::from_pretrained(MarianConfigResources::ENGLISH2CHINESE);
    let vocab_resource = RemoteResource::from_pretrained(MarianVocabResources::ENGLISH2CHINESE);
    let merges_resource = RemoteResource::from_pretrained(MarianSpmResources::ENGLISH2CHINESE);

    let source_languages = MarianSourceLanguages::ENGLISH2CHINESE;
    let target_languages = MarianTargetLanguages::ENGLISH2CHINESE;

    let translation_config = TranslationConfig::new(
        ModelType::Marian,
        model_resource,
        config_resource,
        vocab_resource,
        merges_resource,
        source_languages,
        target_languages,
        Device::cuda_if_available(),
    );
    let model = TranslationModel::new(translation_config)?;

    let input_context_1 = "The quick brown fox jumps over the lazy dog";
    let input_context_2 = "The dog did not wake up";

    let output = model.translate(&[input_context_1, input_context_2], None, None)?;

    for sentence in output {
        println!("{}", sentence);
    }
    Ok(())
}
