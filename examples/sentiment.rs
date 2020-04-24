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
use rust_bert::pipelines::sentiment::SentimentClassifier;


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
    let input = [
        "Probably my all-time favorite movie, a story of selflessness, sacrifice and dedication to a noble cause, but it's not preachy or boring.",
        "This film tried to be too many things all at once: stinging political satire, Hollywood blockbuster, sappy romantic comedy, family values promo...",
        "If you like original gut wrenching laughter you will like this movie. If you are young or old then you will love this movie, hell even my mom liked it.",
    ];

//    Run model
    let output = sentiment_classifier.predict(&input);
    for sentiment in output {
        println!("{:?}", sentiment);
    }

    Ok(())
}