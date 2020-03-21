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
use tch::Device;
use failure::err_msg;
use rust_bert::{GPT2Generator, LanguageGenerator};


fn main() -> failure::Fallible<()> {
    //    Resources paths
    let mut home: PathBuf = dirs::home_dir().unwrap();
    home.push("rustbert");
    home.push("gpt2");
    let config_path = &home.as_path().join("config.json");
    let vocab_path = &home.as_path().join("vocab.txt");
    let merges_path = &home.as_path().join("merges.txt");
    let weights_path = &home.as_path().join("model.ot");

    if !config_path.is_file() | !vocab_path.is_file() | !merges_path.is_file() | !weights_path.is_file() {
        return Err(
            err_msg("Could not find required resources to run example. \
                          Please run ../utils/download_dependencies_gpt2.py \
                          in a Python environment with dependencies listed in ../requirements.txt"));
    }

//    Set-up masked LM model
    let device = Device::cuda_if_available();

    let model = GPT2Generator::new(vocab_path, merges_path, config_path, weights_path, device)?;

    let input_context = "The dog";
    let second_input_context = "The cat was";
    let output = model.generate(Some(vec!(input_context, second_input_context)), 0, 30, true, false, 5, 1.2,
                                0, 0.9, 1.0, 1.0, 3, 3, None);

    for sentence in output {
        println!("{:?}", sentence);
    }
    Ok(())
}