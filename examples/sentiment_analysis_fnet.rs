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

use rust_bert::fnet::{FNetConfigResources, FNetModelResources, FNetVocabResources};
use rust_bert::pipelines::common::ModelType;
use rust_bert::pipelines::sentiment::{SentimentConfig, SentimentModel};
use rust_bert::resources::RemoteResource;

fn main() -> anyhow::Result<()> {
    //    Set-up classifier
    let config_resource = Box::new(RemoteResource::from_pretrained(
        FNetConfigResources::BASE_SST2,
    ));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(
        FNetVocabResources::BASE_SST2,
    ));
    let model_resource = Box::new(RemoteResource::from_pretrained(
        FNetModelResources::BASE_SST2,
    ));

    let sentiment_config = SentimentConfig {
        model_type: ModelType::FNet,
        model_resource,
        config_resource,
        vocab_resource,
        ..Default::default()
    };

    let sentiment_classifier = SentimentModel::new(sentiment_config)?;

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
