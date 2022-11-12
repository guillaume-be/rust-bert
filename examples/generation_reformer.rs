// Copyright 2018 Google AI and Google Brain team.
// Copyright 2018 Carnegie Mellon University Authors.
// Copyright 2020-present, the HuggingFace Inc. team.
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
use rust_bert::pipelines::text_generation::{TextGenerationConfig, TextGenerationModel};
use rust_bert::reformer::{
    ReformerConfigResources, ReformerModelResources, ReformerVocabResources,
};
use rust_bert::resources::RemoteResource;

fn main() -> anyhow::Result<()> {
    //    Set-up model
    //    Resources paths
    let config_resource = Box::new(RemoteResource::from_pretrained(
        ReformerConfigResources::CRIME_AND_PUNISHMENT,
    ));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(
        ReformerVocabResources::CRIME_AND_PUNISHMENT,
    ));
    let model_resource = Box::new(RemoteResource::from_pretrained(
        ReformerModelResources::CRIME_AND_PUNISHMENT,
    ));
    let generate_config = TextGenerationConfig {
        model_type: ModelType::Reformer,
        model_resource,
        config_resource,
        vocab_resource,
        merges_resource: None,
        min_length: 100,
        max_length: Some(100),
        do_sample: true,
        early_stopping: false,
        num_beams: 3,
        num_return_sequences: 1,
        ..Default::default()
    };

    let model = TextGenerationModel::new(generate_config)?;

    let input_context_1 = "The really great men must, I think,";
    let input_context_2 = "It was a gloom winter night, and";
    let output = model.generate(&[input_context_1, input_context_2], None);

    for sentence in output {
        println!("{}", sentence);
    }
    Ok(())
}
