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
use crate::albert::AlbertConfig;
use crate::bart::BartConfig;
use crate::bert::BertConfig;
use crate::common::error::RustBertError;
use crate::distilbert::DistilBertConfig;
use crate::electra::ElectraConfig;
use crate::t5::T5Config;
use crate::Config;
use rust_tokenizers::preprocessing::tokenizer::base_tokenizer::{
    Mask, Offset, OffsetSize, Tokenizer,
};
use rust_tokenizers::preprocessing::tokenizer::marian_tokenizer::MarianTokenizer;
use rust_tokenizers::preprocessing::tokenizer::t5_tokenizer::T5Tokenizer;
use rust_tokenizers::preprocessing::tokenizer::xlm_roberta_tokenizer::XLMRobertaTokenizer;
use rust_tokenizers::preprocessing::vocab::albert_vocab::AlbertVocab;
use rust_tokenizers::preprocessing::vocab::marian_vocab::MarianVocab;
use rust_tokenizers::preprocessing::vocab::t5_vocab::T5Vocab;
use rust_tokenizers::{
    AlbertTokenizer, BertTokenizer, BertVocab, RobertaTokenizer, RobertaVocab, TokenizedInput,
    TruncationStrategy, XLMRobertaVocab,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
/// # Identifies the type of model
pub enum ModelType {
    Bert,
    DistilBert,
    Roberta,
    XLMRoberta,
    Electra,
    Marian,
    T5,
    Albert,
}

/// # Abstraction that holds a model configuration, can be of any of the supported models
pub enum ConfigOption {
    /// Bert configuration
    Bert(BertConfig),
    /// DistilBert configuration
    DistilBert(DistilBertConfig),
    /// Electra configuration
    Electra(ElectraConfig),
    /// Marian configuration
    Marian(BartConfig),
    /// T5 configuration
    T5(T5Config),
    /// Albert configuration
    Albert(AlbertConfig),
}

/// # Abstraction that holds a particular tokenizer, can be of any of the supported models
pub enum TokenizerOption {
    /// Bert Tokenizer
    Bert(BertTokenizer),
    /// Roberta Tokenizer
    Roberta(RobertaTokenizer),
    /// Roberta Tokenizer
    XLMRoberta(XLMRobertaTokenizer),
    /// Marian Tokenizer
    Marian(MarianTokenizer),
    /// T5 Tokenizer
    T5(T5Tokenizer),
    /// Albert Tokenizer
    Albert(AlbertTokenizer),
}

impl ConfigOption {
    /// Interface method to load a configuration from file
    pub fn from_file(model_type: ModelType, path: &Path) -> Self {
        match model_type {
            ModelType::Bert | ModelType::Roberta | ModelType::XLMRoberta => {
                ConfigOption::Bert(BertConfig::from_file(path))
            }
            ModelType::DistilBert => ConfigOption::DistilBert(DistilBertConfig::from_file(path)),
            ModelType::Electra => ConfigOption::Electra(ElectraConfig::from_file(path)),
            ModelType::Marian => ConfigOption::Marian(BartConfig::from_file(path)),
            ModelType::T5 => ConfigOption::T5(T5Config::from_file(path)),
            ModelType::Albert => ConfigOption::Albert(AlbertConfig::from_file(path)),
        }
    }

    pub fn get_label_mapping(self) -> HashMap<i64, String> {
        match self {
            Self::Bert(config) => config
                .id2label
                .expect("No label dictionary (id2label) provided in configuration file"),
            Self::DistilBert(config) => config
                .id2label
                .expect("No label dictionary (id2label) provided in configuration file"),
            Self::Electra(config) => config
                .id2label
                .expect("No label dictionary (id2label) provided in configuration file"),
            Self::Marian(config) => config
                .id2label
                .expect("No label dictionary (id2label) provided in configuration file"),
            Self::Albert(config) => config
                .id2label
                .expect("No label dictionary (id2label) provided in configuration file"),
            Self::T5(_) => panic!("T5 does not use a label mapping"),
        }
    }
}

impl TokenizerOption {
    /// Interface method to load a tokenizer from file
    pub fn from_file(
        model_type: ModelType,
        vocab_path: &str,
        merges_path: Option<&str>,
        lower_case: bool,
        strip_accents: Option<bool>,
        add_prefix_space: Option<bool>,
    ) -> Result<Self, RustBertError> {
        let tokenizer = match model_type {
            ModelType::Bert | ModelType::DistilBert | ModelType::Electra => {
                if add_prefix_space.is_some() {
                    return Err(RustBertError::InvalidConfigurationError(
                        format!("Optional input `add_prefix_space` set to value {} but cannot be used by {:?}",
                                add_prefix_space.unwrap(),
                                model_type)));
                }
                TokenizerOption::Bert(BertTokenizer::from_file(
                    vocab_path,
                    lower_case,
                    strip_accents.unwrap_or(lower_case),
                )?)
            }
            ModelType::Roberta => {
                if strip_accents.is_some() {
                    return Err(RustBertError::InvalidConfigurationError(format!(
                        "Optional input `strip_accents` set to value {} but cannot be used by {:?}",
                        strip_accents.unwrap(),
                        model_type
                    )));
                }
                TokenizerOption::Roberta(RobertaTokenizer::from_file(
                    vocab_path,
                    merges_path.expect("No merges specified!"),
                    lower_case,
                    add_prefix_space.unwrap_or(false),
                )?)
            }
            ModelType::Marian => {
                if strip_accents.is_some() {
                    return Err(RustBertError::InvalidConfigurationError(format!(
                        "Optional input `strip_accents` set to value {} but cannot be used by {:?}",
                        strip_accents.unwrap(),
                        model_type
                    )));
                }
                if add_prefix_space.is_some() {
                    return Err(RustBertError::InvalidConfigurationError(
                        format!("Optional input `add_prefix_space` set to value {} but cannot be used by {:?}",
                                add_prefix_space.unwrap(),
                                model_type)));
                }
                TokenizerOption::Marian(MarianTokenizer::from_files(
                    vocab_path,
                    merges_path.expect("No merges specified!"),
                    lower_case,
                )?)
            }
            ModelType::T5 => {
                if strip_accents.is_some() {
                    return Err(RustBertError::InvalidConfigurationError(format!(
                        "Optional input `strip_accents` set to value {} but cannot be used by {:?}",
                        strip_accents.unwrap(),
                        model_type
                    )));
                }
                if add_prefix_space.is_some() {
                    return Err(RustBertError::InvalidConfigurationError(
                        format!("Optional input `add_prefix_space` set to value {} but cannot be used by {:?}",
                                add_prefix_space.unwrap(),
                                model_type)));
                }
                TokenizerOption::T5(T5Tokenizer::from_file(vocab_path, lower_case)?)
            }
            ModelType::XLMRoberta => {
                if strip_accents.is_some() {
                    return Err(RustBertError::InvalidConfigurationError(format!(
                        "Optional input `strip_accents` set to value {} but cannot be used by {:?}",
                        strip_accents.unwrap(),
                        model_type
                    )));
                }
                if add_prefix_space.is_some() {
                    return Err(RustBertError::InvalidConfigurationError(
                        format!("Optional input `add_prefix_space` set to value {} but cannot be used by {:?}",
                                add_prefix_space.unwrap(),
                                model_type)));
                }
                TokenizerOption::XLMRoberta(XLMRobertaTokenizer::from_file(vocab_path, lower_case)?)
            }
            ModelType::Albert => {
                if strip_accents.is_some() {
                    return Err(RustBertError::InvalidConfigurationError(format!(
                        "Optional input `strip_accents` set to value {} but cannot be used by {:?}",
                        strip_accents.unwrap(),
                        model_type
                    )));
                }
                TokenizerOption::Albert(AlbertTokenizer::from_file(
                    vocab_path,
                    lower_case,
                    strip_accents.unwrap_or(lower_case),
                )?)
            }
        };
        Ok(tokenizer)
    }

    /// Returns the model type
    pub fn model_type(&self) -> ModelType {
        match *self {
            Self::Bert(_) => ModelType::Bert,
            Self::Roberta(_) => ModelType::Roberta,
            Self::XLMRoberta(_) => ModelType::XLMRoberta,
            Self::Marian(_) => ModelType::Marian,
            Self::T5(_) => ModelType::T5,
            Self::Albert(_) => ModelType::Albert,
        }
    }

    /// Interface method
    pub fn encode_list(
        &self,
        text_list: Vec<&str>,
        max_len: usize,
        truncation_strategy: &TruncationStrategy,
        stride: usize,
    ) -> Vec<TokenizedInput> {
        match *self {
            Self::Bert(ref tokenizer) => {
                tokenizer.encode_list(text_list, max_len, truncation_strategy, stride)
            }
            Self::Roberta(ref tokenizer) => {
                tokenizer.encode_list(text_list, max_len, truncation_strategy, stride)
            }
            Self::Marian(ref tokenizer) => {
                tokenizer.encode_list(text_list, max_len, truncation_strategy, stride)
            }
            Self::T5(ref tokenizer) => {
                tokenizer.encode_list(text_list, max_len, truncation_strategy, stride)
            }
            Self::XLMRoberta(ref tokenizer) => {
                tokenizer.encode_list(text_list, max_len, truncation_strategy, stride)
            }
            Self::Albert(ref tokenizer) => {
                tokenizer.encode_list(text_list, max_len, truncation_strategy, stride)
            }
        }
    }

    /// Interface method to tokenization
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        match *self {
            Self::Bert(ref tokenizer) => tokenizer.tokenize(text),
            Self::Roberta(ref tokenizer) => tokenizer.tokenize(text),
            Self::Marian(ref tokenizer) => tokenizer.tokenize(text),
            Self::T5(ref tokenizer) => tokenizer.tokenize(text),
            Self::XLMRoberta(ref tokenizer) => tokenizer.tokenize(text),
            Self::Albert(ref tokenizer) => tokenizer.tokenize(text),
        }
    }

    /// Interface method to build input with special tokens
    pub fn build_input_with_special_tokens(
        &self,
        tokens_1: Vec<i64>,
        tokens_2: Option<Vec<i64>>,
        offsets_1: Vec<Option<Offset>>,
        offsets_2: Option<Vec<Option<Offset>>>,
        original_offsets_1: Vec<Vec<OffsetSize>>,
        original_offsets_2: Option<Vec<Vec<OffsetSize>>>,
        mask_1: Vec<Mask>,
        mask_2: Option<Vec<Mask>>,
    ) -> (
        Vec<i64>,
        Vec<i8>,
        Vec<i8>,
        Vec<Option<Offset>>,
        Vec<Vec<OffsetSize>>,
        Vec<Mask>,
    ) {
        match *self {
            Self::Bert(ref tokenizer) => tokenizer.build_input_with_special_tokens(
                tokens_1,
                tokens_2,
                offsets_1,
                offsets_2,
                original_offsets_1,
                original_offsets_2,
                mask_1,
                mask_2,
            ),
            Self::Roberta(ref tokenizer) => tokenizer.build_input_with_special_tokens(
                tokens_1,
                tokens_2,
                offsets_1,
                offsets_2,
                original_offsets_1,
                original_offsets_2,
                mask_1,
                mask_2,
            ),
            Self::XLMRoberta(ref tokenizer) => tokenizer.build_input_with_special_tokens(
                tokens_1,
                tokens_2,
                offsets_1,
                offsets_2,
                original_offsets_1,
                original_offsets_2,
                mask_1,
                mask_2,
            ),
            Self::Marian(ref tokenizer) => tokenizer.build_input_with_special_tokens(
                tokens_1,
                tokens_2,
                offsets_1,
                offsets_2,
                original_offsets_1,
                original_offsets_2,
                mask_1,
                mask_2,
            ),
            Self::T5(ref tokenizer) => tokenizer.build_input_with_special_tokens(
                tokens_1,
                tokens_2,
                offsets_1,
                offsets_2,
                original_offsets_1,
                original_offsets_2,
                mask_1,
                mask_2,
            ),
            Self::Albert(ref tokenizer) => tokenizer.build_input_with_special_tokens(
                tokens_1,
                tokens_2,
                offsets_1,
                offsets_2,
                original_offsets_1,
                original_offsets_2,
                mask_1,
                mask_2,
            ),
        }
    }

    /// Interface method to convert tokens to ids
    pub fn convert_tokens_to_ids(&self, tokens: &Vec<String>) -> Vec<i64> {
        match *self {
            Self::Bert(ref tokenizer) => tokenizer.convert_tokens_to_ids(tokens),
            Self::Roberta(ref tokenizer) => tokenizer.convert_tokens_to_ids(tokens),
            Self::Marian(ref tokenizer) => tokenizer.convert_tokens_to_ids(tokens),
            Self::T5(ref tokenizer) => tokenizer.convert_tokens_to_ids(tokens),
            Self::XLMRoberta(ref tokenizer) => tokenizer.convert_tokens_to_ids(tokens),
            Self::Albert(ref tokenizer) => tokenizer.convert_tokens_to_ids(tokens),
        }
    }

    /// Interface method
    pub fn get_pad_id(&self) -> Option<i64> {
        match *self {
            Self::Bert(ref tokenizer) => Some(
                *tokenizer
                    .vocab()
                    .special_values
                    .get(BertVocab::pad_value())
                    .expect("PAD token not found in vocabulary"),
            ),
            Self::Roberta(ref tokenizer) => Some(
                *tokenizer
                    .vocab()
                    .special_values
                    .get(RobertaVocab::pad_value())
                    .expect("PAD token not found in vocabulary"),
            ),
            Self::XLMRoberta(ref tokenizer) => Some(
                *tokenizer
                    .vocab()
                    .special_values
                    .get(XLMRobertaVocab::pad_value())
                    .expect("PAD token not found in vocabulary"),
            ),
            Self::Marian(ref tokenizer) => Some(
                *tokenizer
                    .vocab()
                    .special_values
                    .get(MarianVocab::pad_value())
                    .expect("PAD token not found in vocabulary"),
            ),
            Self::T5(ref tokenizer) => Some(
                *tokenizer
                    .vocab()
                    .special_values
                    .get(T5Vocab::pad_value())
                    .expect("PAD token not found in vocabulary"),
            ),
            Self::Albert(ref tokenizer) => Some(
                *tokenizer
                    .vocab()
                    .special_values
                    .get(T5Vocab::pad_value())
                    .expect("PAD token not found in vocabulary"),
            ),
        }
    }

    /// Interface method
    pub fn get_sep_id(&self) -> Option<i64> {
        match *self {
            Self::Bert(ref tokenizer) => Some(
                *tokenizer
                    .vocab()
                    .special_values
                    .get(BertVocab::sep_value())
                    .expect("SEP token not found in vocabulary"),
            ),
            Self::Roberta(ref tokenizer) => Some(
                *tokenizer
                    .vocab()
                    .special_values
                    .get(RobertaVocab::sep_value())
                    .expect("SEP token not found in vocabulary"),
            ),
            Self::XLMRoberta(ref tokenizer) => Some(
                *tokenizer
                    .vocab()
                    .special_values
                    .get(XLMRobertaVocab::sep_value())
                    .expect("SEP token not found in vocabulary"),
            ),
            Self::Albert(ref tokenizer) => Some(
                *tokenizer
                    .vocab()
                    .special_values
                    .get(AlbertVocab::sep_value())
                    .expect("SEP token not found in vocabulary"),
            ),
            Self::Marian(_) => None,
            Self::T5(_) => None,
        }
    }
}
