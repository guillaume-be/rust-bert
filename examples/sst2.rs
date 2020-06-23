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

extern crate dirs;
extern crate failure;

use rust_bert::pipelines::sentiment::{ss2_processor, SentimentModel};
use std::env;
use std::path::PathBuf;

fn main() -> failure::Fallible<()> {
    //    Set-up classifier
    let sentiment_classifier = SentimentModel::new(Default::default())?;

    //    Define input
    let mut sst2_path = PathBuf::from(env::var("SST2_PATH")
        .expect("Please set the \"squad_dataset\" environment variable pointing to the SQuAD dataset folder"));
    sst2_path.push("train.tsv");
    let inputs = ss2_processor(sst2_path).unwrap();

    //    Run model
    let batch_size = 64;
    let mut output = vec![];
    for batch in inputs.chunks(batch_size) {
        output.push(
            sentiment_classifier.predict(
                batch
                    .iter()
                    .map(|v| v.as_str())
                    .collect::<Vec<&str>>()
                    .as_slice(),
            ),
        );
    }
    let mut flat_outputs = vec![];
    for batch_output in output.iter_mut() {
        flat_outputs.append(batch_output);
    }
    println!("{:?}", flat_outputs.len());

    Ok(())
}
