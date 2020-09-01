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

use rust_bert::pipelines::question_answering::{squad_processor, QuestionAnsweringModel};
use std::env;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    //    Set-up Question Answering model
    let qa_model = QuestionAnsweringModel::new(Default::default())?;

    //    Define input
    let mut squad_path = PathBuf::from(env::var("squad_dataset")
        .expect("Please set the \"squad_dataset\" environment variable pointing to the SQuAD dataset folder"));
    squad_path.push("dev-v2.0.json");
    let qa_inputs = squad_processor(squad_path);

    //    Get answer
    let answers = qa_model.predict(&qa_inputs, 1, 64);
    println!("Sample answer: {:?}", answers.first().unwrap());
    println!("{}", answers.len());
    Ok(())
}
