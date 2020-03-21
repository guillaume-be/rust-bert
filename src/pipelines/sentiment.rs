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


use rust_tokenizers::bert_tokenizer::BertTokenizer;
use std::path::Path;
use tch::{Device, Tensor, Kind, no_grad};
use tch::nn::VarStore;
use rust_tokenizers::preprocessing::tokenizer::base_tokenizer::{TruncationStrategy, MultiThreadedTokenizer};
use crate::common::config::Config;
use crate::distilbert::distilbert::{DistilBertConfig, DistilBertModelClassifier};


#[derive(Debug, PartialEq)]
pub enum SentimentPolarity {
    Positive,
    Negative,
}

#[derive(Debug)]
pub struct Sentiment {
    pub polarity: SentimentPolarity,
    pub score: f64,
}

pub struct SentimentClassifier {
    tokenizer: BertTokenizer,
    distil_bert_classifier: DistilBertModelClassifier,
    var_store: VarStore,
}

impl SentimentClassifier {
    pub fn new(vocab_path: &Path, model_config_path: &Path, model_weight_path: &Path, device: Device)
               -> failure::Fallible<SentimentClassifier> {
        let tokenizer = BertTokenizer::from_file(vocab_path.to_str().unwrap(), true);
        let mut var_store = VarStore::new(device);
        let config = DistilBertConfig::from_file(model_config_path);
        let distil_bert_classifier = DistilBertModelClassifier::new(&var_store.root(), &config);
        var_store.load(model_weight_path)?;
        Ok(SentimentClassifier { tokenizer, distil_bert_classifier, var_store })
    }

    fn prepare_for_model(&self, input: Vec<&str>) -> Tensor {
        let tokenized_input = self.tokenizer.encode_list(input.to_vec(),
                                                         128,
                                                         &TruncationStrategy::LongestFirst,
                                                         0);
        let max_len = tokenized_input.iter().map(|input| input.token_ids.len()).max().unwrap();
        let tokenized_input = tokenized_input.
            iter().
            map(|input| input.token_ids.clone()).
            map(|mut input| {
                input.extend(vec![0; max_len - input.len()]);
                input
            }).
            map(|input|
                Tensor::of_slice(&(input))).
            collect::<Vec<_>>();
        Tensor::stack(tokenized_input.as_slice(), 0).to(self.var_store.device())
    }

    pub fn predict(&self, input: &[&str]) -> Vec<Sentiment> {
        let input_tensor = self.prepare_for_model(input.to_vec());
        let (output, _, _) = no_grad(|| {
            self.distil_bert_classifier
                .forward_t(Some(input_tensor),
                           None,
                           None,
                           false)
                .unwrap()
        });
        let output = output.softmax(-1, Kind::Float);

        let mut sentiments: Vec<Sentiment> = vec!();
        for record_index in 0..output.size()[0] {
            let mut score = output.double_value(&[record_index, 0]);
            let polarity = if score < 0.5 {SentimentPolarity::Positive} else {SentimentPolarity::Negative};
            if &SentimentPolarity::Positive == &polarity {score = 1.0 - score};
            sentiments.push(Sentiment {polarity, score})
        };
        sentiments
    }
}