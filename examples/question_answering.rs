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
extern crate dirs;

use std::path::PathBuf;
use rust_bert::pipelines::question_answering::QuestionAnsweringModel;
use tch::Device;


fn main() -> failure::Fallible<()> {
    //    Resources paths
    let mut home: PathBuf = dirs::home_dir().unwrap();
    home.push("rustbert");
    home.push("distilbert-qa");
    let config_path = &home.as_path().join("config.json");
    let vocab_path = &home.as_path().join("vocab.txt");
    let weights_path = &home.as_path().join("model.ot");

//    Set-up Question Answering model
    let device = Device::Cpu;
    let qa_model = QuestionAnsweringModel::new(vocab_path,
                                               config_path,
                                               weights_path, device)?;

//    Define input
    let question = "Where does Amy live ?";
    let context = "Amy lives in Amsterdam";

//    Get answer
    let answers = qa_model.predict(question, context, 1);
    println!("{:?}", answers);
    Ok(())
}