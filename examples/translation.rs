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

extern crate failure;

use rust_bert::pipelines::generation::{LanguageGenerator, GenerateConfig, MarianGenerator};
use rust_bert::resources::{Resource, LocalResource};
use std::path::PathBuf;

fn main() -> failure::Fallible<()> {

//    Set-up masked LM model
    let generate_config = GenerateConfig {
        config_resource: Resource::Local(LocalResource { local_path: PathBuf::from("E:/Coding/cache/rustbert/marian-mt-en-fr/config.json")}),
        model_resource: Resource::Local(LocalResource { local_path: PathBuf::from("E:/Coding/cache/rustbert/marian-mt-en-fr/model.ot")}),
        vocab_resource: Resource::Local(LocalResource { local_path: PathBuf::from("E:/Coding/cache/rustbert/marian-mt-en-fr/vocab.json")}),
        merges_resource: Resource::Local(LocalResource { local_path: PathBuf::from("E:/Coding/cache/rustbert/marian-mt-en-fr/spiece.model")}),
        max_length: 512,
        do_sample: false,
        num_beams: 6,
        temperature: 1.0,
        num_return_sequences: 1,
        ..Default::default()
    };
    let mut model = MarianGenerator::new(generate_config)?;

    let input_context = "The quick brown fox jumps over the lazy dog";
    let output = model.generate(Some(vec!(input_context)), None);

    for sentence in output {
        println!("{:?}", sentence);
    }
    Ok(())
}