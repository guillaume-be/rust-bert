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

use rust_bert::bert::{BertConfigResources, BertModelResources, BertVocabResources};
use rust_bert::pipelines::common::ModelType;
use rust_bert::pipelines::ner::NERModel;
use rust_bert::pipelines::token_classification::{
    LabelAggregationOption, TokenClassificationConfig,
};
use rust_bert::resources::RemoteResource;

fn main() -> anyhow::Result<()> {
    //    Load a configuration
    let config = TokenClassificationConfig::new(
        ModelType::Bert,
        RemoteResource::from_pretrained(BertModelResources::BERT_NER),
        RemoteResource::from_pretrained(BertConfigResources::BERT_NER),
        RemoteResource::from_pretrained(BertVocabResources::BERT_NER),
        None,  //merges resource only relevant with ModelType::Roberta
        false, //lowercase
        false,
        None,
        LabelAggregationOption::Mode,
    );

    //    Create the model
    let token_classification_model = NERModel::new(config)?;
    let input = [
        "My name is Amélie. I live in Москва.",
        "Chongqing is a city in China.",
    ];
    let token_outputs = token_classification_model.predict(&input);

    for token in token_outputs {
        println!("{:?}", token);
    }

    Ok(())
}
