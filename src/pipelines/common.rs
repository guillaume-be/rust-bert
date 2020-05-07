// Copyright 2019-present, the HuggingFace Inc. team, The Google AI Language Team and Facebook, Inc.
// Copyright 2019-2020 Guillaume Becquin
// Copyright 2020 Maarten van Gompel
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! # Common blocks for generic pipelines (e.g. token classification or sequence classification)
//! Provides Enums holding configuration or tokenization resources that can be used to create
//! generic pipelines. The model component is defined in the generic pipeline itself as the
//! pre-processing, forward pass and postprocessing differs between pipelines while basic config and
//! tokenization objects don't.
//!
use crate::bert::BertConfig;
use crate::distilbert::DistilBertConfig;
use rust_tokenizers::{BertTokenizer, RobertaTokenizer, TokenizedInput, TruncationStrategy};
use rust_tokenizers::preprocessing::tokenizer::base_tokenizer::Tokenizer;
use std::path::Path;
use crate::Config;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

#[derive(Clone, Copy, Serialize, Deserialize)]
/// # Identifies the type of model
pub enum ModelType {
    Bert,
    DistilBert,
    Roberta,
}

/// # Abstraction that holds a model configuration, can be of any of the supported models
pub enum ConfigOption {
    /// Bert configuration
    Bert(BertConfig),
    /// DistilBert configuration
    DistilBert(DistilBertConfig),
}

/// # Abstraction that holds a particular tokenizer, can be of any of the supported models
pub enum TokenizerOption {
    /// Bert Tokenizer
    Bert(BertTokenizer),
    /// Roberta Tokenizer
    Roberta(RobertaTokenizer),
}

impl ConfigOption {
    /// Interface method to load a configuration from file
    pub fn from_file(model_type: ModelType, path: &Path) -> Self {
        match model_type {
            ModelType::Bert | ModelType::Roberta => ConfigOption::Bert(BertConfig::from_file(path)),
            ModelType::DistilBert => ConfigOption::DistilBert(DistilBertConfig::from_file(path))
        }
    }

    pub fn get_label_mapping(self) -> HashMap<i64, String> {
        match self {
            Self::Bert(config) => config.id2label.expect("No label dictionary (id2label) provided in configuration file"),
            Self::DistilBert(config) => config.id2label.expect("No label dictionary (id2label) provided in configuration file"),
        }
    }
}

impl TokenizerOption {
    /// Interface method to load a tokenizer from file
    pub fn from_file(model_type: ModelType, vocab_path: &str, merges_path: Option<&str>, lower_case: bool) -> Self {
        match model_type {
            ModelType::Bert | ModelType::DistilBert => TokenizerOption::Bert(BertTokenizer::from_file(vocab_path, lower_case)),
            ModelType::Roberta => TokenizerOption::Roberta(RobertaTokenizer::from_file(vocab_path, merges_path.expect("No merges specified!"), lower_case)),
        }
    }

    /// Returns the model type
    pub fn model_type(&self) -> ModelType {
        match *self {
            Self::Bert(_) => ModelType::Bert,
            Self::Roberta(_) => ModelType::Roberta
        }
    }

    /// Interface method
    pub fn encode_list(&self, text_list: Vec<&str>, max_len: usize, truncation_strategy: &TruncationStrategy, stride: usize) -> Vec<TokenizedInput> {
        match *self {
            Self::Bert(ref tokenizer) => tokenizer.encode_list(text_list, max_len, truncation_strategy, stride),
            Self::Roberta(ref tokenizer) => tokenizer.encode_list(text_list, max_len, truncation_strategy, stride)
        }
    }
}
