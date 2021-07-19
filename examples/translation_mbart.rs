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
    MBartConfigResources, MBartGenerator, MBartModelResources, MBartVocabResources,
};
use rust_bert::pipelines::generation_utils::{GenerateConfig, LanguageGenerator};
use rust_bert::resources::{RemoteResource, Resource};

fn main() -> anyhow::Result<()> {
    let generate_config = GenerateConfig {
        max_length: 56,
        model_resource: Resource::Remote(RemoteResource::from_pretrained(
            MBartModelResources::MBART50_MANY_TO_MANY,
        )),
        config_resource: Resource::Remote(RemoteResource::from_pretrained(
            MBartConfigResources::MBART50_MANY_TO_MANY,
        )),
        vocab_resource: Resource::Remote(RemoteResource::from_pretrained(
            MBartVocabResources::MBART50_MANY_TO_MANY,
        )),
        merges_resource: Resource::Remote(RemoteResource::from_pretrained(
            MBartVocabResources::MBART50_MANY_TO_MANY,
        )),
        do_sample: false,
        num_beams: 1,
        ..Default::default()
    };
    let model = MBartGenerator::new(generate_config)?;

    let input_context_1 = ">>en<< The quick brown fox jumps over the lazy dog.";
    let target_language = model.get_tokenizer().convert_tokens_to_ids([">>de<<"])[0];

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
        println!("{:?}", sentence.text);
    }
    Ok(())
}
