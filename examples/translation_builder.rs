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

use rust_bert::pipelines::common::ModelType;
use rust_bert::pipelines::translation::{Language, TranslationModelBuilder};
use tch::Device;

fn main() -> anyhow::Result<()> {
    let model = TranslationModelBuilder::new()
        .with_device(Device::cuda_if_available())
        .with_model_type(ModelType::Marian)
        // .with_large_model()
        .with_source_languages(vec![Language::English])
        .with_target_languages(vec![Language::Spanish])
        .create_model()?;

    let input_context_1 = "This is a sentence to be translated";
    let input_context_2 = "The dog did not wake up.";

    let output = model.translate(&[input_context_1, input_context_2], None, Language::Spanish)?;

    for sentence in output {
        println!("{sentence}");
    }
    Ok(())
}
