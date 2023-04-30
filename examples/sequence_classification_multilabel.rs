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

use rust_bert::pipelines::sequence_classification::SequenceClassificationModel;

fn main() -> anyhow::Result<()> {
    //    Set-up model
    let sequence_classification_model = SequenceClassificationModel::new(Default::default())?;

    //    Define input
    let input = [
        "Probably my all-time favorite movie, a story of selflessness, sacrifice and dedication to a noble cause, but it's not preachy or boring.",
        "This is a neutral sentence.",
        "If you like original gut wrenching laughter you will like this movie. If you are young or old then you will love this movie, hell even my mom liked it.",
    ];

    //    Run model
    let output = sequence_classification_model.predict_multilabel(&input, 0.05);
    if let Ok(labels) = output {
        for label in labels {
            println!("{label:?}");
        }
    }

    Ok(())
}
