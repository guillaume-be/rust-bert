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

use rust_bert::pipelines::generation_utils::{GenerateConfig, LanguageGenerator, T5Generator};
use rust_bert::resources::{RemoteResource, Resource};
use rust_bert::t5::{T5ConfigResources, T5ModelResources, T5VocabResources};

fn main() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource =
        Resource::Remote(RemoteResource::from_pretrained(T5ConfigResources::T5_BASE));
    let vocab_resource =
        Resource::Remote(RemoteResource::from_pretrained(T5VocabResources::T5_BASE));
    let weights_resource =
        Resource::Remote(RemoteResource::from_pretrained(T5ModelResources::T5_BASE));

    let generate_config = GenerateConfig {
        model_resource: weights_resource,
        vocab_resource,
        config_resource,
        max_length: 40,
        do_sample: false,
        num_beams: 4,
        ..Default::default()
    };

    //    Set-up masked LM model
    let t5_model = T5Generator::new(generate_config)?;

    //    Define input
    let input = ["translate English to German: This sentence will get translated to German"];

    let output = t5_model.generate(Some(input.to_vec()), None);
    println!("{:?}", output);

    Ok(())
}
