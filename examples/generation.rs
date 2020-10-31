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
use rust_bert::pipelines::text_generation::{TextGenerationConfig, TextGenerationModel};

fn main() -> anyhow::Result<()> {
    //    Set-up masked LM model
    let generate_config = TextGenerationConfig {
        model_type: ModelType::GPT2,
        max_length: 30,
        do_sample: true,
        num_beams: 5,
        temperature: 1.1,
        num_return_sequences: 3,
        ..Default::default()
    };
    let model = TextGenerationModel::new(generate_config)?;

    let input_context = "The dog";
    let second_input_context = "The cat was";
    let output = model.generate(&[input_context, second_input_context], None);

    for sentence in output {
        println!("{:?}", sentence);
    }
    Ok(())
}
