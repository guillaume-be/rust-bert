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

use rust_bert::pipelines::common::{ModelType, TokenizerOption};
use rust_bert::pipelines::text_generation::{TextGenerationConfig, TextGenerationModel};
use rust_bert::resources::{RemoteResource, ResourceProvider};
use std::fs::File;
use std::io::Write;
use tempfile::TempDir;

fn main() -> anyhow::Result<()> {
    //    Set-up model
    let generate_config = TextGenerationConfig {
        model_type: ModelType::GPT2,
        max_length: Some(30),
        do_sample: false,
        num_beams: 1,
        temperature: 1.0,
        num_return_sequences: 1,
        ..Default::default()
    };

    // Create tokenizer
    let tmp_dir = TempDir::new()?;
    let special_token_map_path = tmp_dir.path().join("special_token_map.json");
    let mut tmp_file = File::create(&special_token_map_path)?;
    writeln!(
        tmp_file,
        r#"{{"bos_token": "<|endoftext|>", "eos_token": "<|endoftext|>", "unk_token": "<|endoftext|>"}}"#
    )?;

    let tokenizer_path = RemoteResource::from_pretrained((
        "gpt2/tokenizer",
        "https://huggingface.co/gpt2/resolve/main/tokenizer.json",
    ))
    .get_local_path()?;
    let tokenizer =
        TokenizerOption::from_hf_tokenizer_file(tokenizer_path, special_token_map_path)?;

    let model = TextGenerationModel::new_with_tokenizer(generate_config, tokenizer)?;

    let input_context = "The dog";
    // let second_input_context = "The cat was";
    let output = model.generate(&[input_context], None)?;

    for sentence in output {
        println!("{sentence:?}");
    }
    Ok(())
}
