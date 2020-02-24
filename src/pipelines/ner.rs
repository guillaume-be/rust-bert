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
use tch::nn::VarStore;
use rust_tokenizers::preprocessing::tokenizer::base_tokenizer::{TruncationStrategy, MultiThreadedTokenizer};
use crate::{BertForTokenClassification, BertConfig};
use std::collections::HashMap;
use crate::common::config::Config;
use tch::{Tensor, no_grad, Device};
use tch::kind::Kind::Float;


#[derive(Debug)]
pub struct Entity {
    pub word: String,
    pub score: f64,
    pub label: String,
}

pub struct NERModel {
    tokenizer: BertTokenizer,
    bert_sequence_classifier: BertForTokenClassification,
    label_mapping: HashMap<i64, String>,
    var_store: VarStore,
}

impl NERModel {
    pub fn new(vocab_path: &Path, model_config_path: &Path, model_weight_path: &Path, device: Device)
               -> failure::Fallible<NERModel> {
        let tokenizer = BertTokenizer::from_file(vocab_path.to_str().unwrap(), false);
        let mut var_store = VarStore::new(device);
        let config = BertConfig::from_file(model_config_path);
        let bert_sequence_classifier = BertForTokenClassification::new(&var_store.root(), &config);
        let label_mapping = config.id2label.expect("No label dictionary (id2label) provided in configuration file");
        var_store.load(model_weight_path)?;
        Ok(NERModel { tokenizer, bert_sequence_classifier, label_mapping, var_store })
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

    pub fn predict(&self, input: Vec<&str>) -> Vec<Entity> {
        let input_tensor = self.prepare_for_model(input);
        let (output, _, _) = no_grad(|| {
            self.bert_sequence_classifier
                .forward_t(Some(input_tensor.copy()),
                           None,
                           None,
                           None,
                           None,
                           false)
        });
        let output = output.detach().to(Device::Cpu);
        let score: Tensor = output.exp() / output.exp().sum1(&[-1], true, Float);
        let labels_idx = &score.argmax(-1, true);

        let mut entities: Vec<Entity> = vec!();
        for sentence_idx in 0..labels_idx.size()[0] {
            let labels = labels_idx.get(sentence_idx);
            for position_idx in 0..labels.size()[0] {
                let label = labels.int64_value(&[position_idx]);
                if label != 0 {
                    entities.push(Entity {
                        word: rust_tokenizers::preprocessing::tokenizer::base_tokenizer::Tokenizer::decode(&self.tokenizer, vec!(input_tensor.int64_value(&[sentence_idx, position_idx])), true, true),
                        score: score.double_value(&[sentence_idx, position_idx, label]),
                        label: self.label_mapping.get(&label).expect("Index out of vocabulary bounds.").to_owned(),
                    });
                }
            }
        }
        entities
    }
}