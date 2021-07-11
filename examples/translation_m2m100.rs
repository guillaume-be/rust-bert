// Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
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

use rust_bert::m2m_100::{
    M2M100ConfigResources, M2M100Generator, M2M100MergesResources, M2M100ModelResources,
    M2M100VocabResources,
};
use rust_bert::pipelines::generation_utils::{GenerateConfig, LanguageGenerator};
use rust_bert::resources::{RemoteResource, Resource};

fn main() -> anyhow::Result<()> {
    let generate_config = GenerateConfig {
        max_length: 512,
        min_length: 0,
        model_resource: Resource::Remote(RemoteResource::from_pretrained(
            M2M100ModelResources::M2M100_418M,
        )),
        config_resource: Resource::Remote(RemoteResource::from_pretrained(
            M2M100ConfigResources::M2M100_418M,
        )),
        vocab_resource: Resource::Remote(RemoteResource::from_pretrained(
            M2M100VocabResources::M2M100_418M,
        )),
        merges_resource: Resource::Remote(RemoteResource::from_pretrained(
            M2M100MergesResources::M2M100_418M,
        )),
        do_sample: false,
        early_stopping: true,
        num_beams: 3,
        no_repeat_ngram_size: 0,
        ..Default::default()
    };

    let model = M2M100Generator::new(generate_config)?;

    let input_context_1 = ">>en.<< The dog did not wake up.";
    let target_language = model.get_tokenizer().convert_tokens_to_ids([">>es.<<"])[0];

    println!("{:?} - {:?}", input_context_1, target_language);
    let output = model.generate(
        Some(&[input_context_1]),
        None,
        None,
        None,
        None,
        target_language,
        None,
        false,
    );

    for sentence in output {
        println!("{:?}", sentence);
    }
    Ok(())
}
