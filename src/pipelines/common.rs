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
use crate::deberta::DebertaConfig;
use crate::deberta_v2::DebertaV2Config;
use crate::distilbert::DistilBertConfig;
use crate::electra::ElectraConfig;
use crate::fnet::FNetConfig;
use crate::gpt2::Gpt2Config;
use crate::gpt_j::GptJConfig;
use crate::gpt_neo::GptNeoConfig;
use crate::longformer::LongformerConfig;
use crate::longt5::LongT5Config;
use crate::m2m_100::M2M100Config;
use crate::marian::MarianConfig;
use crate::mbart::MBartConfig;
use crate::mobilebert::MobileBertConfig;
use crate::openai_gpt::OpenAiGptConfig;
use crate::pegasus::PegasusConfig;
use crate::prophetnet::ProphetNetConfig;
use crate::reformer::ReformerConfig;
use crate::roberta::RobertaConfig;
use crate::t5::T5Config;
use crate::xlnet::XLNetConfig;
use crate::Config;
use rust_tokenizers::tokenizer::{
    AlbertTokenizer, BertTokenizer, DeBERTaTokenizer, DeBERTaV2Tokenizer, FNetTokenizer,
    Gpt2Tokenizer, M2M100Tokenizer, MBart50Tokenizer, MarianTokenizer, MultiThreadedTokenizer,
    NLLBTokenizer, OpenAiGptTokenizer, PegasusTokenizer, ProphetNetTokenizer, ReformerTokenizer,
    RobertaTokenizer, T5Tokenizer, Tokenizer, TruncationStrategy, XLMRobertaTokenizer,
    XLNetTokenizer,
};
use rust_tokenizers::vocab::Vocab;
use rust_tokenizers::{TokenIdsWithOffsets, TokenizedInput, TokensWithOffsets};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::convert::TryFrom;
use std::path::Path;

#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq, Eq)]
/// # Identifies the type of model
pub enum ModelType {
    Bart,
    #[serde(alias = "bert")]
    Bert,
    #[serde(alias = "distilbert")]
    DistilBert,
    Deberta,
    DebertaV2,
    #[serde(alias = "roberta")]
    Roberta,
    XLMRoberta,
    Electra,
    Marian,
    MobileBert,
    #[serde(alias = "t5")]
    T5,
    #[serde(alias = "longt5")]
    LongT5,
    #[serde(alias = "albert")]
    Albert,
    XLNet,
    GPT2,
    GPTJ,
    OpenAiGpt,
    Reformer,
    ProphetNet,
    Longformer,
    Pegasus,
    GPTNeo,
    MBart,
    M2M100,
    #[serde(alias = "m2m100")]
    NLLB,
    FNet,
}

/// # Abstraction that holds a model configuration, can be of any of the supported models
pub enum ConfigOption {
    /// Bart configuration
    Bart(BartConfig),
    /// Bert configuration
    Bert(BertConfig),
    /// DistilBert configuration
    DistilBert(DistilBertConfig),
    /// DeBERTa configuration
    Deberta(DebertaConfig),
    /// DeBERTa V2 configuration
    DebertaV2(DebertaV2Config),
    /// Electra configuration
    Electra(ElectraConfig),
    /// Marian configuration
    Marian(MarianConfig),
    /// MobileBert configuration
    MobileBert(MobileBertConfig),
    /// OpenAI GPT configuration
    OpenAiGpt(OpenAiGptConfig),
    /// T5 configuration
    T5(T5Config),
    /// LongT5 configuration
    LongT5(LongT5Config),
    /// Albert configuration
    Albert(AlbertConfig),
    /// XLNet configuration
    XLNet(XLNetConfig),
    /// GPT2 configuration
    GPT2(Gpt2Config),
    /// GPT-J configuration
    GPTJ(GptJConfig),
    /// Reformer configuration
    Reformer(ReformerConfig),
    /// RoBERTa configuration
    Roberta(RobertaConfig),
    /// ProphetNet configuration
    ProphetNet(ProphetNetConfig),
    /// Longformer configuration
    Longformer(LongformerConfig),
    /// Pegasus configuration
    Pegasus(PegasusConfig),
    /// GPT-Neo configuration
    GPTNeo(GptNeoConfig),
    /// MBart configuration
    MBart(MBartConfig),
    /// M2M100 configuration
    M2M100(M2M100Config),
    /// FNet configuration
    FNet(FNetConfig),
}

/// # Abstraction that holds a particular tokenizer, can be of any of the supported models
pub enum TokenizerOption {
    /// Bert Tokenizer
    Bert(BertTokenizer),
    /// DeBERTa Tokenizer
    Deberta(DeBERTaTokenizer),
    /// DeBERTa V2 Tokenizer
    DebertaV2(DeBERTaV2Tokenizer),
    /// Roberta Tokenizer
    Roberta(RobertaTokenizer),
    /// XLMRoberta Tokenizer
    XLMRoberta(XLMRobertaTokenizer),
    /// Marian Tokenizer
    Marian(MarianTokenizer),
    /// T5 Tokenizer
    T5(T5Tokenizer),
    /// Albert Tokenizer
    Albert(AlbertTokenizer),
    /// XLNet Tokenizer
    XLNet(XLNetTokenizer),
    /// GPT2 Tokenizer
    GPT2(Gpt2Tokenizer),
    /// GPT Tokenizer
    OpenAiGpt(OpenAiGptTokenizer),
    /// Reformer Tokenizer
    Reformer(ReformerTokenizer),
    /// ProphetNet Tokenizer
    ProphetNet(ProphetNetTokenizer),
    /// Pegasus Tokenizer
    Pegasus(PegasusTokenizer),
    /// MBart50 Tokenizer
    MBart50(MBart50Tokenizer),
    /// M2M100 Tokenizer
    M2M100(M2M100Tokenizer),
    /// NLLB tokenizer.
    NLLB(NLLBTokenizer),
    /// FNet Tokenizer
    FNet(FNetTokenizer),
    /// Bart Tokenizer
    Bart(RobertaTokenizer),
}

impl ConfigOption {
    /// Interface method to load a configuration from file
    pub fn from_file<P: AsRef<Path>>(model_type: ModelType, path: P) -> Self {
        match model_type {
            ModelType::Bart => ConfigOption::Bart(BartConfig::from_file(path)),
            ModelType::Bert => ConfigOption::Bert(BertConfig::from_file(path)),
            ModelType::Deberta => ConfigOption::Deberta(DebertaConfig::from_file(path)),
            ModelType::DebertaV2 => ConfigOption::DebertaV2(DebertaV2Config::from_file(path)),
            ModelType::DistilBert => ConfigOption::DistilBert(DistilBertConfig::from_file(path)),
            ModelType::Electra => ConfigOption::Electra(ElectraConfig::from_file(path)),
            ModelType::Marian => ConfigOption::Marian(MarianConfig::from_file(path)),
            ModelType::MobileBert => ConfigOption::MobileBert(MobileBertConfig::from_file(path)),
            ModelType::T5 => ConfigOption::T5(T5Config::from_file(path)),
            ModelType::LongT5 => ConfigOption::LongT5(LongT5Config::from_file(path)),
            ModelType::Albert => ConfigOption::Albert(AlbertConfig::from_file(path)),
            ModelType::XLNet => ConfigOption::XLNet(XLNetConfig::from_file(path)),
            ModelType::GPT2 => ConfigOption::GPT2(Gpt2Config::from_file(path)),
            ModelType::GPTJ => ConfigOption::GPTJ(GptJConfig::from_file(path)),
            ModelType::GPTNeo => ConfigOption::GPTNeo(GptNeoConfig::from_file(path)),
            ModelType::OpenAiGpt => ConfigOption::OpenAiGpt(OpenAiGptConfig::from_file(path)),
            ModelType::Reformer => ConfigOption::Reformer(ReformerConfig::from_file(path)),
            ModelType::ProphetNet => ConfigOption::ProphetNet(ProphetNetConfig::from_file(path)),
            ModelType::Longformer => ConfigOption::Longformer(LongformerConfig::from_file(path)),
            ModelType::Pegasus => ConfigOption::Pegasus(PegasusConfig::from_file(path)),
            ModelType::Roberta | ModelType::XLMRoberta => {
                ConfigOption::Roberta(RobertaConfig::from_file(path))
            }
            ModelType::MBart => ConfigOption::MBart(MBartConfig::from_file(path)),
            ModelType::M2M100 | ModelType::NLLB => {
                ConfigOption::M2M100(M2M100Config::from_file(path))
            }
            ModelType::FNet => ConfigOption::FNet(FNetConfig::from_file(path)),
        }
    }

    pub fn get_label_mapping(&self) -> &HashMap<i64, String> {
        match self {
            Self::Bart(config) => config
                .id2label
                .as_ref()
                .expect("No label dictionary (id2label) provided in configuration file"),
            Self::Bert(config) => config
                .id2label
                .as_ref()
                .expect("No label dictionary (id2label) provided in configuration file"),
            Self::Deberta(config) => config
                .id2label
                .as_ref()
                .expect("No label dictionary (id2label) provided in configuration file"),
            Self::DebertaV2(config) => config
                .id2label
                .as_ref()
                .expect("No label dictionary (id2label) provided in configuration file"),
            Self::DistilBert(config) => config
                .id2label
                .as_ref()
                .expect("No label dictionary (id2label) provided in configuration file"),
            Self::Electra(config) => config
                .id2label
                .as_ref()
                .expect("No label dictionary (id2label) provided in configuration file"),
            Self::Marian(config) => config
                .id2label
                .as_ref()
                .expect("No label dictionary (id2label) provided in configuration file"),
            Self::MobileBert(config) => config
                .id2label
                .as_ref()
                .expect("No label dictionary (id2label) provided in configuration file"),
            Self::Albert(config) => config
                .id2label
                .as_ref()
                .expect("No label dictionary (id2label) provided in configuration file"),
            Self::XLNet(config) => config
                .id2label
                .as_ref()
                .expect("No label dictionary (id2label) provided in configuration file"),
            Self::Reformer(config) => config
                .id2label
                .as_ref()
                .expect("No label dictionary (id2label) provided in configuration file"),
            Self::ProphetNet(config) => config
                .id2label
                .as_ref()
                .expect("No label dictionary (id2label) provided in configuration file"),
            Self::Longformer(config) => config
                .id2label
                .as_ref()
                .expect("No label dictionary (id2label) provided in configuration file"),
            Self::MBart(config) => config
                .id2label
                .as_ref()
                .expect("No label dictionary (id2label) provided in configuration file"),
            Self::M2M100(config) => config
                .id2label
                .as_ref()
                .expect("No label dictionary (id2label) provided in configuration file"),
            Self::FNet(config) => config
                .id2label
                .as_ref()
                .expect("No label dictionary (id2label) provided in configuration file"),
            Self::Roberta(config) => config
                .id2label
                .as_ref()
                .expect("No label dictionary (id2label) provided in configuration file"),
            Self::T5(_) => panic!("T5 does not use a label mapping"),
            Self::LongT5(_) => panic!("LongT5 does not use a label mapping"),
            Self::OpenAiGpt(_) => panic!("OpenAI GPT does not use a label mapping"),
            Self::GPT2(_) => panic!("GPT2 does not use a label mapping"),
            Self::GPTJ(_) => panic!("GPT-J does not use a label mapping"),
            Self::GPTNeo(_) => panic!("GPT-Neo does not use a label mapping"),
            Self::Pegasus(_) => panic!("Pegasus does not use a label mapping"),
        }
    }

    pub fn get_max_len(&self) -> Option<i64> {
        match self {
            Self::Bart(config) => Some(config.max_position_embeddings),
            Self::Bert(config) => Some(config.max_position_embeddings),
            Self::Deberta(config) => Some(config.max_position_embeddings),
            Self::DebertaV2(config) => Some(config.max_position_embeddings),
            Self::DistilBert(config) => Some(config.max_position_embeddings),
            Self::Electra(config) => Some(config.max_position_embeddings),
            Self::Marian(config) => Some(config.max_position_embeddings),
            Self::MobileBert(config) => Some(config.max_position_embeddings),
            Self::T5(_) => None,
            Self::LongT5(_) => None,
            Self::Albert(config) => Some(config.max_position_embeddings),
            Self::XLNet(_) => None,
            Self::GPT2(config) => Some(config.n_positions),
            Self::GPTJ(config) => Some(config.n_positions),
            Self::Reformer(config) => Some(config.max_position_embeddings),
            Self::ProphetNet(config) => Some(config.max_position_embeddings),
            Self::Longformer(config) => Some(config.max_position_embeddings),
            Self::Pegasus(config) => Some(config.max_position_embeddings),
            Self::OpenAiGpt(config) => Some(config.n_positions),
            Self::GPTNeo(config) => Some(config.max_position_embeddings),
            Self::MBart(config) => Some(config.max_position_embeddings),
            Self::M2M100(config) => Some(config.max_position_embeddings),
            Self::FNet(config) => Some(config.max_position_embeddings),
            Self::Roberta(config) => Some(config.max_position_embeddings),
        }
    }
}

impl TryFrom<&ConfigOption> for BertConfig {
    type Error = RustBertError;

    fn try_from(config: &ConfigOption) -> Result<Self, Self::Error> {
        match config {
            ConfigOption::Bert(config) | ConfigOption::Roberta(config) => Ok(config.clone()),
            _ => Err(RustBertError::InvalidConfigurationError(
                "You can only supply a BertConfig for Bert or a RobertaConfig for Roberta!"
                    .to_string(),
            )),
        }
    }
}

impl TryFrom<&ConfigOption> for DistilBertConfig {
    type Error = RustBertError;

    fn try_from(config: &ConfigOption) -> Result<Self, Self::Error> {
        if let ConfigOption::DistilBert(config) = config {
            Ok(config.clone())
        } else {
            Err(RustBertError::InvalidConfigurationError(
                "You can only supply a DistilBertConfig for DistilBert!".to_string(),
            ))
        }
    }
}

impl TryFrom<&ConfigOption> for AlbertConfig {
    type Error = RustBertError;

    fn try_from(config: &ConfigOption) -> Result<Self, Self::Error> {
        if let ConfigOption::Albert(config) = config {
            Ok(config.clone())
        } else {
            Err(RustBertError::InvalidConfigurationError(
                "You can only supply an AlbertConfig for Albert!".to_string(),
            ))
        }
    }
}

impl TryFrom<&ConfigOption> for T5Config {
    type Error = RustBertError;

    fn try_from(config: &ConfigOption) -> Result<Self, Self::Error> {
        if let ConfigOption::T5(config) = config {
            Ok(config.clone())
        } else {
            Err(RustBertError::InvalidConfigurationError(
                "You can only supply a T5Config for T5!".to_string(),
            ))
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
        strip_accents: impl Into<Option<bool>>,
        add_prefix_space: impl Into<Option<bool>>,
    ) -> Result<Self, RustBertError> {
        let strip_accents = strip_accents.into();
        let add_prefix_space = add_prefix_space.into();

        let tokenizer = match model_type {
            ModelType::Bert
            | ModelType::DistilBert
            | ModelType::Electra
            | ModelType::MobileBert => {
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
            ModelType::Deberta => {
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
                TokenizerOption::Deberta(DeBERTaTokenizer::from_file(
                    vocab_path,
                    merges_path.expect("No merges specified!"),
                    lower_case,
                )?)
            }
            ModelType::DebertaV2 => TokenizerOption::DebertaV2(DeBERTaV2Tokenizer::from_file(
                vocab_path,
                lower_case,
                strip_accents.unwrap_or(false),
                add_prefix_space.unwrap_or(false),
            )?),
            ModelType::Roberta | ModelType::Longformer => {
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
            ModelType::Bart => {
                if strip_accents.is_some() {
                    return Err(RustBertError::InvalidConfigurationError(format!(
                        "Optional input `strip_accents` set to value {} but cannot be used by {:?}",
                        strip_accents.unwrap(),
                        model_type
                    )));
                }
                TokenizerOption::Bart(RobertaTokenizer::from_file(
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
            ModelType::T5 | ModelType::LongT5 => {
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
                    strip_accents.unwrap_or(false),
                )?)
            }
            ModelType::Reformer => {
                if add_prefix_space.is_some() {
                    return Err(RustBertError::InvalidConfigurationError(
                        format!("Optional input `add_prefix_space` set to value {} but cannot be used by {:?}",
                                add_prefix_space.unwrap(),
                                model_type)));
                }
                if strip_accents.is_some() {
                    return Err(RustBertError::InvalidConfigurationError(
                        format!("Optional input `add_prefix_space` set to value {} but cannot be used by {:?}",
                                add_prefix_space.unwrap(),
                                model_type)));
                }
                TokenizerOption::Reformer(ReformerTokenizer::from_file(vocab_path, lower_case)?)
            }
            ModelType::GPT2 | ModelType::GPTNeo | ModelType::GPTJ => {
                TokenizerOption::GPT2(Gpt2Tokenizer::from_file(
                    vocab_path,
                    merges_path.expect("No merges specified!"),
                    lower_case,
                )?)
            }
            ModelType::OpenAiGpt => TokenizerOption::OpenAiGpt(OpenAiGptTokenizer::from_file(
                vocab_path,
                merges_path.expect("No merges specified!"),
                lower_case,
            )?),
            ModelType::ProphetNet => {
                if add_prefix_space.is_some() {
                    return Err(RustBertError::InvalidConfigurationError(
                        format!("Optional input `add_prefix_space` set to value {} but cannot be used by {:?}",
                                add_prefix_space.unwrap(),
                                model_type)));
                }
                TokenizerOption::ProphetNet(ProphetNetTokenizer::from_file(
                    vocab_path,
                    lower_case,
                    strip_accents.unwrap_or(lower_case),
                )?)
            }
            ModelType::Pegasus => {
                if add_prefix_space.is_some() {
                    return Err(RustBertError::InvalidConfigurationError(
                        format!("Optional input `add_prefix_space` set to value {} but cannot be used by {:?}",
                                add_prefix_space.unwrap(),
                                model_type)));
                }
                if strip_accents.is_some() {
                    return Err(RustBertError::InvalidConfigurationError(format!(
                        "Optional input `strip_accents` set to value {} but cannot be used by {:?}",
                        strip_accents.unwrap(),
                        model_type
                    )));
                }
                TokenizerOption::Pegasus(PegasusTokenizer::from_file(vocab_path, lower_case)?)
            }
            ModelType::MBart => {
                if add_prefix_space.is_some() {
                    return Err(RustBertError::InvalidConfigurationError(
                        format!("Optional input `add_prefix_space` set to value {} but cannot be used by {:?}",
                                add_prefix_space.unwrap(),
                                model_type)));
                }
                if strip_accents.is_some() {
                    return Err(RustBertError::InvalidConfigurationError(format!(
                        "Optional input `strip_accents` set to value {} but cannot be used by {:?}",
                        strip_accents.unwrap(),
                        model_type
                    )));
                }
                TokenizerOption::MBart50(MBart50Tokenizer::from_file(vocab_path, lower_case)?)
            }
            ModelType::M2M100 => {
                if add_prefix_space.is_some() {
                    return Err(RustBertError::InvalidConfigurationError(
                        format!("Optional input `add_prefix_space` set to value {} but cannot be used by {:?}",
                                add_prefix_space.unwrap(),
                                model_type)));
                }
                if strip_accents.is_some() {
                    return Err(RustBertError::InvalidConfigurationError(format!(
                        "Optional input `strip_accents` set to value {} but cannot be used by {:?}",
                        strip_accents.unwrap(),
                        model_type
                    )));
                }
                TokenizerOption::M2M100(M2M100Tokenizer::from_files(
                    vocab_path,
                    merges_path.expect("No merges specified!"),
                    lower_case,
                )?)
            }
            ModelType::NLLB => {
                if add_prefix_space.is_some() {
                    return Err(RustBertError::InvalidConfigurationError(
                        format!("Optional input `add_prefix_space` set to value {} but cannot be used by {:?}",
                                add_prefix_space.unwrap(),
                                model_type)));
                }
                if strip_accents.is_some() {
                    return Err(RustBertError::InvalidConfigurationError(format!(
                        "Optional input `strip_accents` set to value {} but cannot be used by {:?}",
                        strip_accents.unwrap(),
                        model_type
                    )));
                }
                TokenizerOption::NLLB(NLLBTokenizer::from_files(
                    vocab_path,
                    merges_path.expect("No merges specified."),
                )?)
            }
            ModelType::FNet => TokenizerOption::FNet(FNetTokenizer::from_file(
                vocab_path,
                lower_case,
                strip_accents.unwrap_or(false),
            )?),
        };
        Ok(tokenizer)
    }

    /// Returns the model type
    pub fn model_type(&self) -> ModelType {
        match *self {
            Self::Bert(_) => ModelType::Bert,
            Self::Deberta(_) => ModelType::Deberta,
            Self::DebertaV2(_) => ModelType::DebertaV2,
            Self::Roberta(_) => ModelType::Roberta,
            Self::Bart(_) => ModelType::Bart,
            Self::XLMRoberta(_) => ModelType::XLMRoberta,
            Self::Marian(_) => ModelType::Marian,
            Self::T5(_) => ModelType::T5,
            Self::Albert(_) => ModelType::Albert,
            Self::XLNet(_) => ModelType::XLNet,
            Self::GPT2(_) => ModelType::GPT2,
            Self::OpenAiGpt(_) => ModelType::OpenAiGpt,
            Self::Reformer(_) => ModelType::Reformer,
            Self::ProphetNet(_) => ModelType::ProphetNet,
            Self::Pegasus(_) => ModelType::Pegasus,
            Self::MBart50(_) => ModelType::MBart,
            Self::M2M100(_) | Self::NLLB(_) => ModelType::M2M100,
            Self::FNet(_) => ModelType::FNet,
        }
    }

    /// Interface method
    pub fn encode_list<S>(
        &self,
        text_list: &[S],
        max_len: usize,
        truncation_strategy: &TruncationStrategy,
        stride: usize,
    ) -> Vec<TokenizedInput>
    where
        S: AsRef<str> + Sync,
    {
        match *self {
            Self::Bert(ref tokenizer) => MultiThreadedTokenizer::encode_list(
                tokenizer,
                text_list,
                max_len,
                truncation_strategy,
                stride,
            ),
            Self::Deberta(ref tokenizer) => MultiThreadedTokenizer::encode_list(
                tokenizer,
                text_list,
                max_len,
                truncation_strategy,
                stride,
            ),
            Self::DebertaV2(ref tokenizer) => MultiThreadedTokenizer::encode_list(
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
            Self::Bart(ref tokenizer) => MultiThreadedTokenizer::encode_list(
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
            Self::GPT2(ref tokenizer) => MultiThreadedTokenizer::encode_list(
                tokenizer,
                text_list,
                max_len,
                truncation_strategy,
                stride,
            ),
            Self::OpenAiGpt(ref tokenizer) => MultiThreadedTokenizer::encode_list(
                tokenizer,
                text_list,
                max_len,
                truncation_strategy,
                stride,
            ),
            Self::Reformer(ref tokenizer) => MultiThreadedTokenizer::encode_list(
                tokenizer,
                text_list,
                max_len,
                truncation_strategy,
                stride,
            ),
            Self::ProphetNet(ref tokenizer) => MultiThreadedTokenizer::encode_list(
                tokenizer,
                text_list,
                max_len,
                truncation_strategy,
                stride,
            ),
            Self::Pegasus(ref tokenizer) => MultiThreadedTokenizer::encode_list(
                tokenizer,
                text_list,
                max_len,
                truncation_strategy,
                stride,
            ),
            Self::MBart50(ref tokenizer) => MultiThreadedTokenizer::encode_list(
                tokenizer,
                text_list,
                max_len,
                truncation_strategy,
                stride,
            ),
            Self::M2M100(ref tokenizer) => MultiThreadedTokenizer::encode_list(
                tokenizer,
                text_list,
                max_len,
                truncation_strategy,
                stride,
            ),
            Self::FNet(ref tokenizer) => MultiThreadedTokenizer::encode_list(
                tokenizer,
                text_list,
                max_len,
                truncation_strategy,
                stride,
            ),
            Self::NLLB(ref tokenizer) => MultiThreadedTokenizer::encode_list(
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
            Self::Deberta(ref tokenizer) => MultiThreadedTokenizer::encode_pair_list(
                tokenizer,
                text_pair_list,
                max_len,
                truncation_strategy,
                stride,
            ),
            Self::DebertaV2(ref tokenizer) => MultiThreadedTokenizer::encode_pair_list(
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
            Self::Bart(ref tokenizer) => MultiThreadedTokenizer::encode_pair_list(
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
            Self::GPT2(ref tokenizer) => MultiThreadedTokenizer::encode_pair_list(
                tokenizer,
                text_pair_list,
                max_len,
                truncation_strategy,
                stride,
            ),
            Self::OpenAiGpt(ref tokenizer) => MultiThreadedTokenizer::encode_pair_list(
                tokenizer,
                text_pair_list,
                max_len,
                truncation_strategy,
                stride,
            ),
            Self::Reformer(ref tokenizer) => MultiThreadedTokenizer::encode_pair_list(
                tokenizer,
                text_pair_list,
                max_len,
                truncation_strategy,
                stride,
            ),
            Self::ProphetNet(ref tokenizer) => MultiThreadedTokenizer::encode_pair_list(
                tokenizer,
                text_pair_list,
                max_len,
                truncation_strategy,
                stride,
            ),
            Self::Pegasus(ref tokenizer) => MultiThreadedTokenizer::encode_pair_list(
                tokenizer,
                text_pair_list,
                max_len,
                truncation_strategy,
                stride,
            ),
            Self::MBart50(ref tokenizer) => MultiThreadedTokenizer::encode_pair_list(
                tokenizer,
                text_pair_list,
                max_len,
                truncation_strategy,
                stride,
            ),
            Self::M2M100(ref tokenizer) => MultiThreadedTokenizer::encode_pair_list(
                tokenizer,
                text_pair_list,
                max_len,
                truncation_strategy,
                stride,
            ),
            Self::NLLB(ref tokenizer) => MultiThreadedTokenizer::encode_pair_list(
                tokenizer,
                text_pair_list,
                max_len,
                truncation_strategy,
                stride,
            ),
            Self::FNet(ref tokenizer) => MultiThreadedTokenizer::encode_pair_list(
                tokenizer,
                text_pair_list,
                max_len,
                truncation_strategy,
                stride,
            ),
        }
    }

    /// Interface method for pair encoding (single input)
    pub fn encode_pair(
        &self,
        text_1: &str,
        text_2: Option<&str>,
        max_len: usize,
        truncation_strategy: &TruncationStrategy,
        stride: usize,
    ) -> TokenizedInput {
        match *self {
            Self::Bert(ref tokenizer) => {
                tokenizer.encode(text_1, text_2, max_len, truncation_strategy, stride)
            }
            Self::Deberta(ref tokenizer) => {
                tokenizer.encode(text_1, text_2, max_len, truncation_strategy, stride)
            }
            Self::DebertaV2(ref tokenizer) => {
                tokenizer.encode(text_1, text_2, max_len, truncation_strategy, stride)
            }
            Self::Roberta(ref tokenizer) => {
                tokenizer.encode(text_1, text_2, max_len, truncation_strategy, stride)
            }
            Self::Bart(ref tokenizer) => {
                tokenizer.encode(text_1, text_2, max_len, truncation_strategy, stride)
            }
            Self::Marian(ref tokenizer) => {
                tokenizer.encode(text_1, text_2, max_len, truncation_strategy, stride)
            }
            Self::T5(ref tokenizer) => {
                tokenizer.encode(text_1, text_2, max_len, truncation_strategy, stride)
            }
            Self::XLMRoberta(ref tokenizer) => {
                tokenizer.encode(text_1, text_2, max_len, truncation_strategy, stride)
            }
            Self::Albert(ref tokenizer) => {
                tokenizer.encode(text_1, text_2, max_len, truncation_strategy, stride)
            }
            Self::XLNet(ref tokenizer) => {
                tokenizer.encode(text_1, text_2, max_len, truncation_strategy, stride)
            }
            Self::GPT2(ref tokenizer) => {
                tokenizer.encode(text_1, text_2, max_len, truncation_strategy, stride)
            }
            Self::OpenAiGpt(ref tokenizer) => {
                tokenizer.encode(text_1, text_2, max_len, truncation_strategy, stride)
            }
            Self::Reformer(ref tokenizer) => {
                tokenizer.encode(text_1, text_2, max_len, truncation_strategy, stride)
            }
            Self::ProphetNet(ref tokenizer) => {
                tokenizer.encode(text_1, text_2, max_len, truncation_strategy, stride)
            }
            Self::Pegasus(ref tokenizer) => {
                tokenizer.encode(text_1, text_2, max_len, truncation_strategy, stride)
            }
            Self::MBart50(ref tokenizer) => {
                tokenizer.encode(text_1, text_2, max_len, truncation_strategy, stride)
            }
            Self::M2M100(ref tokenizer) => {
                tokenizer.encode(text_1, text_2, max_len, truncation_strategy, stride)
            }
            Self::NLLB(ref tokenizer) => {
                tokenizer.encode(text_1, text_2, max_len, truncation_strategy, stride)
            }
            Self::FNet(ref tokenizer) => {
                tokenizer.encode(text_1, text_2, max_len, truncation_strategy, stride)
            }
        }
    }

    /// Interface method to tokenization
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        match *self {
            Self::Bert(ref tokenizer) => tokenizer.tokenize(text),
            Self::Deberta(ref tokenizer) => tokenizer.tokenize(text),
            Self::DebertaV2(ref tokenizer) => tokenizer.tokenize(text),
            Self::Roberta(ref tokenizer) => tokenizer.tokenize(text),
            Self::Bart(ref tokenizer) => tokenizer.tokenize(text),
            Self::Marian(ref tokenizer) => tokenizer.tokenize(text),
            Self::T5(ref tokenizer) => tokenizer.tokenize(text),
            Self::XLMRoberta(ref tokenizer) => tokenizer.tokenize(text),
            Self::Albert(ref tokenizer) => tokenizer.tokenize(text),
            Self::XLNet(ref tokenizer) => tokenizer.tokenize(text),
            Self::GPT2(ref tokenizer) => tokenizer.tokenize(text),
            Self::OpenAiGpt(ref tokenizer) => tokenizer.tokenize(text),
            Self::Reformer(ref tokenizer) => tokenizer.tokenize(text),
            Self::ProphetNet(ref tokenizer) => tokenizer.tokenize(text),
            Self::Pegasus(ref tokenizer) => tokenizer.tokenize(text),
            Self::MBart50(ref tokenizer) => tokenizer.tokenize(text),
            Self::M2M100(ref tokenizer) => tokenizer.tokenize(text),
            Self::NLLB(ref tokenizer) => tokenizer.tokenize(text),
            Self::FNet(ref tokenizer) => tokenizer.tokenize(text),
        }
    }

    /// Interface method to tokenization
    pub fn tokenize_with_offsets(&self, text: &str) -> TokensWithOffsets {
        match *self {
            Self::Bert(ref tokenizer) => tokenizer.tokenize_with_offsets(text),
            Self::Deberta(ref tokenizer) => tokenizer.tokenize_with_offsets(text),
            Self::DebertaV2(ref tokenizer) => tokenizer.tokenize_with_offsets(text),
            Self::Roberta(ref tokenizer) => tokenizer.tokenize_with_offsets(text),
            Self::Bart(ref tokenizer) => tokenizer.tokenize_with_offsets(text),
            Self::Marian(ref tokenizer) => tokenizer.tokenize_with_offsets(text),
            Self::T5(ref tokenizer) => tokenizer.tokenize_with_offsets(text),
            Self::XLMRoberta(ref tokenizer) => tokenizer.tokenize_with_offsets(text),
            Self::Albert(ref tokenizer) => tokenizer.tokenize_with_offsets(text),
            Self::XLNet(ref tokenizer) => tokenizer.tokenize_with_offsets(text),
            Self::GPT2(ref tokenizer) => tokenizer.tokenize_with_offsets(text),
            Self::OpenAiGpt(ref tokenizer) => tokenizer.tokenize_with_offsets(text),
            Self::Reformer(ref tokenizer) => tokenizer.tokenize_with_offsets(text),
            Self::ProphetNet(ref tokenizer) => tokenizer.tokenize_with_offsets(text),
            Self::Pegasus(ref tokenizer) => tokenizer.tokenize_with_offsets(text),
            Self::MBart50(ref tokenizer) => tokenizer.tokenize_with_offsets(text),
            Self::M2M100(ref tokenizer) => tokenizer.tokenize_with_offsets(text),
            Self::NLLB(ref tokenizer) => tokenizer.tokenize_with_offsets(text),
            Self::FNet(ref tokenizer) => tokenizer.tokenize_with_offsets(text),
        }
    }

    /// Interface method to tokenization
    pub fn tokenize_list<S>(&self, text: &[S]) -> Vec<Vec<String>>
    where
        S: AsRef<str> + Sync,
    {
        match *self {
            Self::Bert(ref tokenizer) => MultiThreadedTokenizer::tokenize_list(tokenizer, text),
            Self::Deberta(ref tokenizer) => MultiThreadedTokenizer::tokenize_list(tokenizer, text),
            Self::DebertaV2(ref tokenizer) => {
                MultiThreadedTokenizer::tokenize_list(tokenizer, text)
            }
            Self::Roberta(ref tokenizer) => MultiThreadedTokenizer::tokenize_list(tokenizer, text),
            Self::Bart(ref tokenizer) => MultiThreadedTokenizer::tokenize_list(tokenizer, text),
            Self::Marian(ref tokenizer) => MultiThreadedTokenizer::tokenize_list(tokenizer, text),
            Self::T5(ref tokenizer) => MultiThreadedTokenizer::tokenize_list(tokenizer, text),
            Self::XLMRoberta(ref tokenizer) => {
                MultiThreadedTokenizer::tokenize_list(tokenizer, text)
            }
            Self::Albert(ref tokenizer) => MultiThreadedTokenizer::tokenize_list(tokenizer, text),
            Self::XLNet(ref tokenizer) => MultiThreadedTokenizer::tokenize_list(tokenizer, text),
            Self::GPT2(ref tokenizer) => MultiThreadedTokenizer::tokenize_list(tokenizer, text),
            Self::OpenAiGpt(ref tokenizer) => {
                MultiThreadedTokenizer::tokenize_list(tokenizer, text)
            }
            Self::Reformer(ref tokenizer) => MultiThreadedTokenizer::tokenize_list(tokenizer, text),
            Self::ProphetNet(ref tokenizer) => {
                MultiThreadedTokenizer::tokenize_list(tokenizer, text)
            }
            Self::Pegasus(ref tokenizer) => MultiThreadedTokenizer::tokenize_list(tokenizer, text),
            Self::MBart50(ref tokenizer) => MultiThreadedTokenizer::tokenize_list(tokenizer, text),
            Self::M2M100(ref tokenizer) => MultiThreadedTokenizer::tokenize_list(tokenizer, text),
            Self::NLLB(ref tokenizer) => MultiThreadedTokenizer::tokenize_list(tokenizer, text),
            Self::FNet(ref tokenizer) => MultiThreadedTokenizer::tokenize_list(tokenizer, text),
        }
    }

    /// Interface method to decoding
    pub fn decode(
        &self,
        token_ids: &[i64],
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> String {
        match *self {
            Self::Bert(ref tokenizer) => {
                tokenizer.decode(token_ids, skip_special_tokens, clean_up_tokenization_spaces)
            }
            Self::Deberta(ref tokenizer) => {
                tokenizer.decode(token_ids, skip_special_tokens, clean_up_tokenization_spaces)
            }
            Self::DebertaV2(ref tokenizer) => {
                tokenizer.decode(token_ids, skip_special_tokens, clean_up_tokenization_spaces)
            }
            Self::Roberta(ref tokenizer) => {
                tokenizer.decode(token_ids, skip_special_tokens, clean_up_tokenization_spaces)
            }
            Self::Bart(ref tokenizer) => {
                tokenizer.decode(token_ids, skip_special_tokens, clean_up_tokenization_spaces)
            }
            Self::Marian(ref tokenizer) => {
                tokenizer.decode(token_ids, skip_special_tokens, clean_up_tokenization_spaces)
            }
            Self::T5(ref tokenizer) => {
                tokenizer.decode(token_ids, skip_special_tokens, clean_up_tokenization_spaces)
            }
            Self::XLMRoberta(ref tokenizer) => {
                tokenizer.decode(token_ids, skip_special_tokens, clean_up_tokenization_spaces)
            }
            Self::Albert(ref tokenizer) => {
                tokenizer.decode(token_ids, skip_special_tokens, clean_up_tokenization_spaces)
            }
            Self::XLNet(ref tokenizer) => {
                tokenizer.decode(token_ids, skip_special_tokens, clean_up_tokenization_spaces)
            }
            Self::GPT2(ref tokenizer) => {
                tokenizer.decode(token_ids, skip_special_tokens, clean_up_tokenization_spaces)
            }
            Self::OpenAiGpt(ref tokenizer) => {
                tokenizer.decode(token_ids, skip_special_tokens, clean_up_tokenization_spaces)
            }
            Self::Reformer(ref tokenizer) => {
                tokenizer.decode(token_ids, skip_special_tokens, clean_up_tokenization_spaces)
            }
            Self::ProphetNet(ref tokenizer) => {
                tokenizer.decode(token_ids, skip_special_tokens, clean_up_tokenization_spaces)
            }
            Self::Pegasus(ref tokenizer) => {
                tokenizer.decode(token_ids, skip_special_tokens, clean_up_tokenization_spaces)
            }
            Self::MBart50(ref tokenizer) => {
                tokenizer.decode(token_ids, skip_special_tokens, clean_up_tokenization_spaces)
            }
            Self::M2M100(ref tokenizer) => {
                tokenizer.decode(token_ids, skip_special_tokens, clean_up_tokenization_spaces)
            }
            Self::NLLB(ref tokenizer) => {
                tokenizer.decode(token_ids, skip_special_tokens, clean_up_tokenization_spaces)
            }
            Self::FNet(ref tokenizer) => {
                tokenizer.decode(token_ids, skip_special_tokens, clean_up_tokenization_spaces)
            }
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
            Self::Deberta(ref tokenizer) => tokenizer.build_input_with_special_tokens(
                token_ids_with_offsets_1,
                token_ids_with_offsets_2,
            ),
            Self::DebertaV2(ref tokenizer) => tokenizer.build_input_with_special_tokens(
                token_ids_with_offsets_1,
                token_ids_with_offsets_2,
            ),
            Self::Roberta(ref tokenizer) => tokenizer.build_input_with_special_tokens(
                token_ids_with_offsets_1,
                token_ids_with_offsets_2,
            ),
            Self::Bart(ref tokenizer) => tokenizer.build_input_with_special_tokens(
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
            Self::GPT2(ref tokenizer) => tokenizer.build_input_with_special_tokens(
                token_ids_with_offsets_1,
                token_ids_with_offsets_2,
            ),
            Self::OpenAiGpt(ref tokenizer) => tokenizer.build_input_with_special_tokens(
                token_ids_with_offsets_1,
                token_ids_with_offsets_2,
            ),
            Self::Reformer(ref tokenizer) => tokenizer.build_input_with_special_tokens(
                token_ids_with_offsets_1,
                token_ids_with_offsets_2,
            ),
            Self::ProphetNet(ref tokenizer) => tokenizer.build_input_with_special_tokens(
                token_ids_with_offsets_1,
                token_ids_with_offsets_2,
            ),
            Self::Pegasus(ref tokenizer) => tokenizer.build_input_with_special_tokens(
                token_ids_with_offsets_1,
                token_ids_with_offsets_2,
            ),
            Self::MBart50(ref tokenizer) => tokenizer.build_input_with_special_tokens(
                token_ids_with_offsets_1,
                token_ids_with_offsets_2,
            ),
            Self::M2M100(ref tokenizer) => tokenizer.build_input_with_special_tokens(
                token_ids_with_offsets_1,
                token_ids_with_offsets_2,
            ),
            Self::NLLB(ref tokenizer) => tokenizer.build_input_with_special_tokens(
                token_ids_with_offsets_1,
                token_ids_with_offsets_2,
            ),
            Self::FNet(ref tokenizer) => tokenizer.build_input_with_special_tokens(
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
    pub fn convert_tokens_to_ids<S>(&self, tokens: &[S]) -> Vec<i64>
    where
        S: AsRef<str>,
    {
        match *self {
            Self::Bert(ref tokenizer) => tokenizer.convert_tokens_to_ids(tokens),
            Self::Deberta(ref tokenizer) => tokenizer.convert_tokens_to_ids(tokens),
            Self::DebertaV2(ref tokenizer) => tokenizer.convert_tokens_to_ids(tokens),
            Self::Roberta(ref tokenizer) => tokenizer.convert_tokens_to_ids(tokens),
            Self::Bart(ref tokenizer) => tokenizer.convert_tokens_to_ids(tokens),
            Self::Marian(ref tokenizer) => tokenizer.convert_tokens_to_ids(tokens),
            Self::T5(ref tokenizer) => tokenizer.convert_tokens_to_ids(tokens),
            Self::XLMRoberta(ref tokenizer) => tokenizer.convert_tokens_to_ids(tokens),
            Self::Albert(ref tokenizer) => tokenizer.convert_tokens_to_ids(tokens),
            Self::XLNet(ref tokenizer) => tokenizer.convert_tokens_to_ids(tokens),
            Self::GPT2(ref tokenizer) => tokenizer.convert_tokens_to_ids(tokens),
            Self::OpenAiGpt(ref tokenizer) => tokenizer.convert_tokens_to_ids(tokens),
            Self::Reformer(ref tokenizer) => tokenizer.convert_tokens_to_ids(tokens),
            Self::ProphetNet(ref tokenizer) => tokenizer.convert_tokens_to_ids(tokens),
            Self::Pegasus(ref tokenizer) => tokenizer.convert_tokens_to_ids(tokens),
            Self::MBart50(ref tokenizer) => tokenizer.convert_tokens_to_ids(tokens),
            Self::M2M100(ref tokenizer) => tokenizer.convert_tokens_to_ids(tokens),
            Self::NLLB(ref tokenizer) => tokenizer.convert_tokens_to_ids(tokens),
            Self::FNet(ref tokenizer) => tokenizer.convert_tokens_to_ids(tokens),
        }
    }

    /// Interface method
    pub fn get_unk_id(&self) -> i64 {
        match *self {
            Self::Bert(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                vocab.token_to_id(vocab.get_unknown_value())
            }
            Self::Deberta(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                vocab.token_to_id(vocab.get_unknown_value())
            }
            Self::DebertaV2(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                vocab.token_to_id(vocab.get_unknown_value())
            }
            Self::Roberta(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                vocab.token_to_id(vocab.get_unknown_value())
            }
            Self::Bart(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                vocab.token_to_id(vocab.get_unknown_value())
            }
            Self::XLMRoberta(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                vocab.token_to_id(vocab.get_unknown_value())
            }
            Self::Marian(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                vocab.token_to_id(vocab.get_unknown_value())
            }
            Self::T5(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                vocab.token_to_id(vocab.get_unknown_value())
            }
            Self::Albert(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                vocab.token_to_id(vocab.get_unknown_value())
            }
            Self::XLNet(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                vocab.token_to_id(vocab.get_unknown_value())
            }
            Self::GPT2(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                vocab.token_to_id(vocab.get_unknown_value())
            }
            Self::OpenAiGpt(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                vocab.token_to_id(vocab.get_unknown_value())
            }
            Self::Reformer(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                vocab.token_to_id(vocab.get_unknown_value())
            }
            Self::ProphetNet(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                vocab.token_to_id(vocab.get_unknown_value())
            }
            Self::Pegasus(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                vocab.token_to_id(vocab.get_unknown_value())
            }
            Self::MBart50(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                vocab.token_to_id(vocab.get_unknown_value())
            }
            Self::M2M100(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                vocab.token_to_id(vocab.get_unknown_value())
            }
            Self::NLLB(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                vocab.token_to_id(vocab.get_unknown_value())
            }
            Self::FNet(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                vocab.token_to_id(vocab.get_unknown_value())
            }
        }
    }

    /// Interface method
    pub fn get_pad_id(&self) -> Option<i64> {
        match *self {
            Self::Bert(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_pad_value()))
            }
            Self::Deberta(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_pad_value()))
            }
            Self::DebertaV2(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_pad_value()))
            }
            Self::Roberta(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_pad_value()))
            }
            Self::Bart(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_pad_value()))
            }
            Self::XLMRoberta(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_pad_value()))
            }
            Self::Marian(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_pad_value()))
            }
            Self::T5(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_pad_value()))
            }
            Self::Albert(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_pad_value()))
            }
            Self::XLNet(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_pad_value()))
            }
            Self::ProphetNet(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_pad_value()))
            }
            Self::Pegasus(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_pad_value()))
            }
            Self::MBart50(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_pad_value()))
            }
            Self::M2M100(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_pad_value()))
            }
            Self::NLLB(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_pad_value()))
            }
            Self::FNet(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_pad_value()))
            }
            Self::Reformer(_) => None,
            Self::GPT2(_) => None,
            Self::OpenAiGpt(_) => None,
        }
    }

    /// Interface method
    pub fn get_sep_id(&self) -> Option<i64> {
        match *self {
            Self::Bert(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_sep_value()))
            }
            Self::Deberta(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_sep_value()))
            }
            Self::DebertaV2(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_sep_value()))
            }
            Self::Roberta(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_sep_value()))
            }
            Self::Bart(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_sep_value()))
            }
            Self::XLMRoberta(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_sep_value()))
            }
            Self::Albert(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_sep_value()))
            }
            Self::XLNet(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_sep_value()))
            }
            Self::ProphetNet(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_sep_value()))
            }
            Self::MBart50(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_sep_value()))
            }
            Self::M2M100(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_sep_value()))
            }
            Self::NLLB(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_sep_value()))
            }
            Self::FNet(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_sep_value()))
            }
            Self::Marian(_) => None,
            Self::T5(_) => None,
            Self::GPT2(_) => None,
            Self::OpenAiGpt(_) => None,
            Self::Reformer(_) => None,
            Self::Pegasus(_) => None,
        }
    }

    /// Interface method
    pub fn get_mask_id(&self) -> Option<i64> {
        match *self {
            Self::Bert(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_mask_value()))
            }
            Self::Deberta(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_mask_value()))
            }
            Self::DebertaV2(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_mask_value()))
            }
            Self::Roberta(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_mask_value()))
            }
            Self::Bart(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_mask_value()))
            }
            Self::XLMRoberta(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_mask_value()))
            }
            Self::Albert(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_mask_value()))
            }
            Self::XLNet(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_mask_value()))
            }
            Self::ProphetNet(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_mask_value()))
            }
            Self::MBart50(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_mask_value()))
            }
            Self::FNet(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_mask_value()))
            }
            Self::Pegasus(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_mask_value()))
            }
            Self::Marian(_) => None,
            Self::M2M100(_) => None,
            Self::NLLB(_) => None,
            Self::T5(_) => None,
            Self::GPT2(_) => None,
            Self::OpenAiGpt(_) => None,
            Self::Reformer(_) => None,
        }
    }

    /// Interface method
    pub fn get_mask_value(&self) -> Option<&str> {
        match self {
            Self::Bert(ref tokenizer) => {
                Some(MultiThreadedTokenizer::vocab(tokenizer).get_mask_value())
            }
            Self::Deberta(ref tokenizer) => {
                Some(MultiThreadedTokenizer::vocab(tokenizer).get_mask_value())
            }
            Self::DebertaV2(ref tokenizer) => {
                Some(MultiThreadedTokenizer::vocab(tokenizer).get_mask_value())
            }
            Self::Roberta(ref tokenizer) => {
                Some(MultiThreadedTokenizer::vocab(tokenizer).get_mask_value())
            }
            Self::Bart(ref tokenizer) => {
                Some(MultiThreadedTokenizer::vocab(tokenizer).get_mask_value())
            }
            Self::XLMRoberta(ref tokenizer) => {
                Some(MultiThreadedTokenizer::vocab(tokenizer).get_mask_value())
            }
            Self::Albert(ref tokenizer) => {
                Some(MultiThreadedTokenizer::vocab(tokenizer).get_mask_value())
            }
            Self::XLNet(ref tokenizer) => {
                Some(MultiThreadedTokenizer::vocab(tokenizer).get_mask_value())
            }
            Self::ProphetNet(ref tokenizer) => {
                Some(MultiThreadedTokenizer::vocab(tokenizer).get_mask_value())
            }
            Self::MBart50(ref tokenizer) => {
                Some(MultiThreadedTokenizer::vocab(tokenizer).get_mask_value())
            }
            Self::FNet(ref tokenizer) => {
                Some(MultiThreadedTokenizer::vocab(tokenizer).get_mask_value())
            }
            Self::Pegasus(ref tokenizer) => {
                Some(MultiThreadedTokenizer::vocab(tokenizer).get_mask_value())
            }
            Self::M2M100(_) => None,
            Self::NLLB(_) => None,
            Self::Marian(_) => None,
            Self::T5(_) => None,
            Self::GPT2(_) => None,
            Self::OpenAiGpt(_) => None,
            Self::Reformer(_) => None,
        }
    }

    /// Interface method
    pub fn get_bos_id(&self) -> Option<i64> {
        match *self {
            Self::Roberta(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_bos_value()))
            }
            Self::Bart(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_bos_value()))
            }
            Self::DebertaV2(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_bos_value()))
            }
            Self::XLMRoberta(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_bos_value()))
            }
            Self::Albert(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_bos_value()))
            }
            Self::XLNet(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_bos_value()))
            }
            Self::M2M100(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_bos_value()))
            }
            Self::NLLB(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_bos_value()))
            }
            Self::GPT2(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_bos_value()))
            }
            Self::Deberta(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_bos_value()))
            }
            Self::MBart50(_) => Some(0),
            Self::FNet(_) => None,
            Self::Bert(_) => None,
            Self::Marian(_) => Some(0),
            Self::T5(_) => None,
            Self::ProphetNet(_) => None,
            Self::OpenAiGpt(_) => None,
            Self::Reformer(_) => None,
            Self::Pegasus(_) => Some(0),
        }
    }

    /// Interface method
    pub fn get_eos_id(&self) -> Option<i64> {
        match *self {
            Self::Roberta(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_eos_value()))
            }
            Self::Bart(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_eos_value()))
            }
            Self::DebertaV2(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_eos_value()))
            }
            Self::XLMRoberta(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_eos_value()))
            }
            Self::Albert(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_eos_value()))
            }
            Self::XLNet(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_eos_value()))
            }
            Self::MBart50(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_eos_value()))
            }
            Self::M2M100(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_eos_value()))
            }
            Self::NLLB(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_eos_value()))
            }
            Self::GPT2(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_eos_value()))
            }
            Self::Deberta(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_eos_value()))
            }
            Self::Marian(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_eos_value()))
            }
            Self::T5(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_eos_value()))
            }
            Self::Reformer(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_eos_value()))
            }
            Self::Pegasus(ref tokenizer) => {
                let vocab = MultiThreadedTokenizer::vocab(tokenizer);
                Some(vocab.token_to_id(vocab.get_eos_value()))
            }
            Self::FNet(_) => None,
            Self::Bert(_) => None,
            Self::ProphetNet(_) => None,
            Self::OpenAiGpt(_) => None,
        }
    }

    /// Interface method
    pub fn add_extra_ids(&mut self, num_extra_ids: i64) {
        match *self {
            Self::Bert(ref mut tokenizer) => tokenizer.add_extra_ids(num_extra_ids),
            Self::Deberta(ref mut tokenizer) => tokenizer.add_extra_ids(num_extra_ids),
            Self::DebertaV2(ref mut tokenizer) => tokenizer.add_extra_ids(num_extra_ids),
            Self::Roberta(ref mut tokenizer) => tokenizer.add_extra_ids(num_extra_ids),
            Self::Bart(ref mut tokenizer) => tokenizer.add_extra_ids(num_extra_ids),
            Self::Marian(ref mut tokenizer) => tokenizer.add_extra_ids(num_extra_ids),
            Self::T5(ref mut tokenizer) => tokenizer.add_extra_ids(num_extra_ids),
            Self::XLMRoberta(ref mut tokenizer) => tokenizer.add_extra_ids(num_extra_ids),
            Self::Albert(ref mut tokenizer) => tokenizer.add_extra_ids(num_extra_ids),
            Self::XLNet(ref mut tokenizer) => tokenizer.add_extra_ids(num_extra_ids),
            Self::GPT2(ref mut tokenizer) => tokenizer.add_extra_ids(num_extra_ids),
            Self::OpenAiGpt(ref mut tokenizer) => tokenizer.add_extra_ids(num_extra_ids),
            Self::Reformer(ref mut tokenizer) => tokenizer.add_extra_ids(num_extra_ids),
            Self::ProphetNet(ref mut tokenizer) => tokenizer.add_extra_ids(num_extra_ids),
            Self::Pegasus(ref mut tokenizer) => tokenizer.add_extra_ids(num_extra_ids),
            Self::MBart50(ref mut tokenizer) => tokenizer.add_extra_ids(num_extra_ids),
            Self::M2M100(ref mut tokenizer) => tokenizer.add_extra_ids(num_extra_ids),
            Self::NLLB(ref mut tokenizer) => tokenizer.add_extra_ids(num_extra_ids),
            Self::FNet(ref mut tokenizer) => tokenizer.add_extra_ids(num_extra_ids),
        }
    }

    /// Interface method
    pub fn add_tokens(&mut self, tokens: &[&str]) {
        match *self {
            Self::Bert(ref mut tokenizer) => tokenizer.add_tokens(tokens),
            Self::Deberta(ref mut tokenizer) => tokenizer.add_tokens(tokens),
            Self::DebertaV2(ref mut tokenizer) => tokenizer.add_tokens(tokens),
            Self::Roberta(ref mut tokenizer) => tokenizer.add_tokens(tokens),
            Self::Bart(ref mut tokenizer) => tokenizer.add_tokens(tokens),
            Self::Marian(ref mut tokenizer) => tokenizer.add_tokens(tokens),
            Self::T5(ref mut tokenizer) => tokenizer.add_tokens(tokens),
            Self::XLMRoberta(ref mut tokenizer) => tokenizer.add_tokens(tokens),
            Self::Albert(ref mut tokenizer) => tokenizer.add_tokens(tokens),
            Self::XLNet(ref mut tokenizer) => tokenizer.add_tokens(tokens),
            Self::GPT2(ref mut tokenizer) => tokenizer.add_tokens(tokens),
            Self::OpenAiGpt(ref mut tokenizer) => tokenizer.add_tokens(tokens),
            Self::Reformer(ref mut tokenizer) => tokenizer.add_tokens(tokens),
            Self::ProphetNet(ref mut tokenizer) => tokenizer.add_tokens(tokens),
            Self::Pegasus(ref mut tokenizer) => tokenizer.add_tokens(tokens),
            Self::MBart50(ref mut tokenizer) => tokenizer.add_tokens(tokens),
            Self::M2M100(ref mut tokenizer) => tokenizer.add_tokens(tokens),
            Self::NLLB(ref mut tokenizer) => tokenizer.add_tokens(tokens),
            Self::FNet(ref mut tokenizer) => tokenizer.add_tokens(tokens),
        }
    }
}
