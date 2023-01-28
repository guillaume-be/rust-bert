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
use rust_bert::resources::RemoteResource;
use rust_bert::xlnet::{XLNetConfigResources, XLNetModelResources, XLNetVocabResources};

fn main() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = Box::new(RemoteResource::from_pretrained(
        XLNetConfigResources::XLNET_BASE_CASED,
    ));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(
        XLNetVocabResources::XLNET_BASE_CASED,
    ));
    let model_resource = Box::new(RemoteResource::from_pretrained(
        XLNetModelResources::XLNET_BASE_CASED,
    ));

    let generate_config = TextGenerationConfig {
        model_type: ModelType::XLNet,
        model_resource,
        config_resource,
        vocab_resource,
        merges_resource: None,
        max_length: Some(32),
        do_sample: false,
        num_beams: 3,
        temperature: 1.0,
        num_return_sequences: 1,
        ..Default::default()
    };
    let model = TextGenerationModel::new(generate_config)?;

    let input_context = "Once upon a time,";
    let output = model.generate(&[input_context], None);

    for sentence in output {
        println!("{}", sentence);
    }
    Ok(())
}
