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
use rust_bert::pipelines::sentiment::{SentimentClassifier, ss2_processor};
use std::env;


fn main() -> failure::Fallible<()> {
//    Resources paths
    let mut home: PathBuf = dirs::home_dir().unwrap();
    home.push("rustbert");
    home.push("distilbert-sst2");
    let config_path = &home.as_path().join("config.json");
    let vocab_path = &home.as_path().join("vocab.txt");
    let weights_path = &home.as_path().join("model.ot");

    if !config_path.is_file() | !vocab_path.is_file() | !weights_path.is_file() {
        return Err(
            err_msg("Could not find required resources to run example. \
                          Please run ../utils/download_dependencies_sst2_sentiment.py \
                          in a Python environment with dependencies listed in ../requirements.txt"));
    }

//    Set-up classifier
    let device = Device::cuda_if_available();
    let sentiment_classifier = SentimentClassifier::new(vocab_path,
                                                        config_path,
                                                        weights_path, device)?;

//    Define input
    let mut sst2_path = PathBuf::from(env::var("SST2_PATH")
        .expect("Please set the \"squad_dataset\" environment variable pointing to the SQuAD dataset folder"));
    sst2_path.push("train.tsv");
    let inputs = ss2_processor(sst2_path).unwrap();

//    Run model
    let batch_size = 64;
    let mut output = vec!();
    for batch in inputs.chunks(batch_size) {
        output.push(sentiment_classifier.predict(batch.iter().map(|v| v.as_str()).collect::<Vec<&str>>().as_slice()));
    }
    let mut flat_outputs = vec!();
    for batch_output in output.iter_mut() {
        flat_outputs.append(batch_output);
    }
    println!("{:?}", flat_outputs.len());

    Ok(())
}