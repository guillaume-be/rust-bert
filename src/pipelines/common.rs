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
use crate::albert::AlbertConfig;
use crate::bart::BartConfig;
use crate::bert::BertConfig;
use crate::common::error::RustBertError;
use crate::distilbert::DistilBertConfig;
use crate::electra::ElectraConfig;
use crate::t5::T5Config;
use crate::xlnet::XLNetConfig;
use crate::Config;
use rust_tokenizers::tokenizer::{
    AlbertTokenizer, BertTokenizer, MarianTokenizer, MultiThreadedTokenizer, RobertaTokenizer,
    T5Tokenizer, Tokenizer, TruncationStrategy, XLMRobertaTokenizer, XLNetTokenizer,
};
use rust_tokenizers::vocab::{
    AlbertVocab, BertVocab, MarianVocab, RobertaVocab, T5Vocab, XLMRobertaVocab, XLNetVocab,
};
use rust_tokenizers::{TokenIdsWithOffsets, TokenizedInput};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
/// # Identifies the type of model
pub enum ModelType {
    Bart,
    Bert,
    DistilBert,
    Roberta,
    XLMRoberta,
    Electra,
    Marian,
    T5,
    Albert,
    XLNet,
}

/// # Abstraction that holds a model configuration, can be of any of the supported models
pub enum ConfigOption {
    /// Bart configuration
    Bart(BartConfig),
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
    /// XLNet configuration
    XLNet(XLNetConfig),
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
    /// Albert Tokenizer
    XLNet(XLNetTokenizer),
}

impl ConfigOption {
    /// Interface method to load a configuration from file
    pub fn from_file<P: AsRef<Path>>(model_type: ModelType, path: P) -> Self {
        match model_type {
            ModelType::Bart => ConfigOption::Bart(BartConfig::from_file(path)),
            ModelType::Bert | ModelType::Roberta | ModelType::XLMRoberta => {
                ConfigOption::Bert(BertConfig::from_file(path))
            }
            ModelType::DistilBert => ConfigOption::DistilBert(DistilBertConfig::from_file(path)),
            ModelType::Electra => ConfigOption::Electra(ElectraConfig::from_file(path)),
            ModelType::Marian => ConfigOption::Marian(BartConfig::from_file(path)),
            ModelType::T5 => ConfigOption::T5(T5Config::from_file(path)),
            ModelType::Albert => ConfigOption::Albert(AlbertConfig::from_file(path)),
            ModelType::XLNet => ConfigOption::XLNet(XLNetConfig::from_file(path)),
        }
    }

    pub fn get_label_mapping(self) -> HashMap<i64, String> {
        match self {
            Self::Bart(config) => config
                .id2label
                .expect("No label dictionary (id2label) provided in configuration file"),
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
            Self::XLNet(config) => config
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
            ModelType::Roberta | ModelType::Bart => {
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
            ModelType::XLNet => {
                if add_prefix_space.is_some() {
                    return Err(RustBertError::InvalidConfigurationError(
                        format!("Optional input `add_prefix_space` set to value {} but cannot be used by {:?}",
                                add_prefix_space.unwrap(),
                                model_type)));
                }
                TokenizerOption::XLNet(XLNetTokenizer::from_file(
                    vocab_path,
                    lower_case,
                    strip_accents.unwrap(),
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
            Self::XLNet(_) => ModelType::XLNet,
        }
    }

    /// Interface method
    pub fn encode_list(
        &self,
        text_list: &[&str],
        max_len: usize,
        truncation_strategy: &TruncationStrategy,
        stride: usize,
    ) -> Vec<TokenizedInput> {
        match *self {
            Self::Bert(ref tokenizer) => MultiThreadedTokenizer::encode_list(
                tokenizer,
                text_list,
                max_len,
                truncation_strategy,
                stride,
            ),
            Self::Roberta(ref tokenizer) => MultiThreadedTokenizer::encode_list(
                tokenizer,
                text_list,
                max_len,
                truncation_strategy,
                stride,
            ),
            Self::Marian(ref tokenizer) => MultiThreadedTokenizer::encode_list(
                tokenizer,
                text_list,
                max_len,
                truncation_strategy,
                stride,
            ),
            Self::T5(ref tokenizer) => MultiThreadedTokenizer::encode_list(
                tokenizer,
                text_list,
                max_len,
                truncation_strategy,
                stride,
            ),
            Self::XLMRoberta(ref tokenizer) => MultiThreadedTokenizer::encode_list(
                tokenizer,
                text_list,
                max_len,
                truncation_strategy,
                stride,
            ),
            Self::Albert(ref tokenizer) => MultiThreadedTokenizer::encode_list(
                tokenizer,
                text_list,
                max_len,
                truncation_strategy,
                stride,
            ),
            Self::XLNet(ref tokenizer) => MultiThreadedTokenizer::encode_list(
                tokenizer,
                text_list,
                max_len,
                truncation_strategy,
                stride,
            ),
        }
    }

    /// Interface method for pair encoding
    pub fn encode_pair_list(
        &self,
        text_pair_list: &[(&str, &str)],
        max_len: usize,
        truncation_strategy: &TruncationStrategy,
        stride: usize,
    ) -> Vec<TokenizedInput> {
        match *self {
            Self::Bert(ref tokenizer) => MultiThreadedTokenizer::encode_pair_list(
                tokenizer,
                text_pair_list,
                max_len,
                truncation_strategy,
                stride,
            ),
            Self::Roberta(ref tokenizer) => MultiThreadedTokenizer::encode_pair_list(
                tokenizer,
                text_pair_list,
                max_len,
                truncation_strategy,
                stride,
            ),
            Self::Marian(ref tokenizer) => MultiThreadedTokenizer::encode_pair_list(
                tokenizer,
                text_pair_list,
                max_len,
                truncation_strategy,
                stride,
            ),
            Self::T5(ref tokenizer) => MultiThreadedTokenizer::encode_pair_list(
                tokenizer,
                text_pair_list,
                max_len,
                truncation_strategy,
                stride,
            ),
            Self::XLMRoberta(ref tokenizer) => MultiThreadedTokenizer::encode_pair_list(
                tokenizer,
                text_pair_list,
                max_len,
                truncation_strategy,
                stride,
            ),
            Self::Albert(ref tokenizer) => MultiThreadedTokenizer::encode_pair_list(
                tokenizer,
                text_pair_list,
                max_len,
                truncation_strategy,
                stride,
            ),
            Self::XLNet(ref tokenizer) => MultiThreadedTokenizer::encode_pair_list(
                tokenizer,
                text_pair_list,
                max_len,
                truncation_strategy,
                stride,
            ),
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
            Self::XLNet(ref tokenizer) => tokenizer.tokenize(text),
        }
    }

    /// Interface method to build input with special tokens
    pub fn build_input_with_special_tokens(
        &self,
        token_ids_with_offsets_1: TokenIdsWithOffsets,
        token_ids_with_offsets_2: Option<TokenIdsWithOffsets>,
    ) -> TokenizedInput {
        let token_ids_with_special_tokens = match *self {
            Self::Bert(ref tokenizer) => tokenizer.build_input_with_special_tokens(
                token_ids_with_offsets_1,
                token_ids_with_offsets_2,
            ),
            Self::Roberta(ref tokenizer) => tokenizer.build_input_with_special_tokens(
                token_ids_with_offsets_1,
                token_ids_with_offsets_2,
            ),
            Self::XLMRoberta(ref tokenizer) => tokenizer.build_input_with_special_tokens(
                token_ids_with_offsets_1,
                token_ids_with_offsets_2,
            ),
            Self::Marian(ref tokenizer) => tokenizer.build_input_with_special_tokens(
                token_ids_with_offsets_1,
                token_ids_with_offsets_2,
            ),
            Self::T5(ref tokenizer) => tokenizer.build_input_with_special_tokens(
                token_ids_with_offsets_1,
                token_ids_with_offsets_2,
            ),
            Self::Albert(ref tokenizer) => tokenizer.build_input_with_special_tokens(
                token_ids_with_offsets_1,
                token_ids_with_offsets_2,
            ),
            Self::XLNet(ref tokenizer) => tokenizer.build_input_with_special_tokens(
                token_ids_with_offsets_1,
                token_ids_with_offsets_2,
            ),
        };
        TokenizedInput {
            token_ids: token_ids_with_special_tokens.token_ids,
            segment_ids: token_ids_with_special_tokens.segment_ids,
            special_tokens_mask: token_ids_with_special_tokens.special_tokens_mask,
            overflowing_tokens: vec![],
            num_truncated_tokens: 0,
            token_offsets: token_ids_with_special_tokens.token_offsets,
            reference_offsets: token_ids_with_special_tokens.reference_offsets,
            mask: token_ids_with_special_tokens.mask,
        }
    }

    /// Interface method to convert tokens to ids
    pub fn convert_tokens_to_ids(&self, tokens: &[String]) -> Vec<i64> {
        match *self {
            Self::Bert(ref tokenizer) => tokenizer.convert_tokens_to_ids(tokens),
            Self::Roberta(ref tokenizer) => tokenizer.convert_tokens_to_ids(tokens),
            Self::Marian(ref tokenizer) => tokenizer.convert_tokens_to_ids(tokens),
            Self::T5(ref tokenizer) => tokenizer.convert_tokens_to_ids(tokens),
            Self::XLMRoberta(ref tokenizer) => tokenizer.convert_tokens_to_ids(tokens),
            Self::Albert(ref tokenizer) => tokenizer.convert_tokens_to_ids(tokens),
            Self::XLNet(ref tokenizer) => tokenizer.convert_tokens_to_ids(tokens),
        }
    }

    /// Interface method
    pub fn get_pad_id(&self) -> Option<i64> {
        match *self {
            Self::Bert(ref tokenizer) => Some(
                *MultiThreadedTokenizer::vocab(tokenizer)
                    .special_values
                    .get(BertVocab::pad_value())
                    .expect("PAD token not found in vocabulary"),
            ),
            Self::Roberta(ref tokenizer) => Some(
                *MultiThreadedTokenizer::vocab(tokenizer)
                    .special_values
                    .get(RobertaVocab::pad_value())
                    .expect("PAD token not found in vocabulary"),
            ),
            Self::XLMRoberta(ref tokenizer) => Some(
                *MultiThreadedTokenizer::vocab(tokenizer)
                    .special_values
                    .get(XLMRobertaVocab::pad_value())
                    .expect("PAD token not found in vocabulary"),
            ),
            Self::Marian(ref tokenizer) => Some(
                *MultiThreadedTokenizer::vocab(tokenizer)
                    .special_values
                    .get(MarianVocab::pad_value())
                    .expect("PAD token not found in vocabulary"),
            ),
            Self::T5(ref tokenizer) => Some(
                *MultiThreadedTokenizer::vocab(tokenizer)
                    .special_values
                    .get(T5Vocab::pad_value())
                    .expect("PAD token not found in vocabulary"),
            ),
            Self::Albert(ref tokenizer) => Some(
                *MultiThreadedTokenizer::vocab(tokenizer)
                    .special_values
                    .get(AlbertVocab::pad_value())
                    .expect("PAD token not found in vocabulary"),
            ),
            Self::XLNet(ref tokenizer) => Some(
                *MultiThreadedTokenizer::vocab(tokenizer)
                    .special_values
                    .get(XLNetVocab::pad_value())
                    .expect("PAD token not found in vocabulary"),
            ),
        }
    }

    /// Interface method
    pub fn get_sep_id(&self) -> Option<i64> {
        match *self {
            Self::Bert(ref tokenizer) => Some(
                *MultiThreadedTokenizer::vocab(tokenizer)
                    .special_values
                    .get(BertVocab::sep_value())
                    .expect("SEP token not found in vocabulary"),
            ),
            Self::Roberta(ref tokenizer) => Some(
                *MultiThreadedTokenizer::vocab(tokenizer)
                    .special_values
                    .get(RobertaVocab::sep_value())
                    .expect("SEP token not found in vocabulary"),
            ),
            Self::XLMRoberta(ref tokenizer) => Some(
                *MultiThreadedTokenizer::vocab(tokenizer)
                    .special_values
                    .get(XLMRobertaVocab::sep_value())
                    .expect("SEP token not found in vocabulary"),
            ),
            Self::Albert(ref tokenizer) => Some(
                *MultiThreadedTokenizer::vocab(tokenizer)
                    .special_values
                    .get(AlbertVocab::sep_value())
                    .expect("SEP token not found in vocabulary"),
            ),
            Self::XLNet(ref tokenizer) => Some(
                *MultiThreadedTokenizer::vocab(tokenizer)
                    .special_values
                    .get(XLNetVocab::sep_value())
                    .expect("SEP token not found in vocabulary"),
            ),
            Self::Marian(_) => None,
            Self::T5(_) => None,
        }
    }
}
