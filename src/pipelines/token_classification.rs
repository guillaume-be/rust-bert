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

//! # Token classification pipeline (Named Entity Recognition, Part-of-Speech tagging)
//! More generic token classification pipeline, works with multiple models (Bert, Roberta)
//!
//! ```no_run
//! use rust_bert::pipelines::token_classification::{TokenClassificationModel,TokenClassificationConfig};
//! use rust_bert::resources::RemoteResource;
//! use rust_bert::bert::{BertModelResources, BertVocabResources, BertConfigResources};
//! use rust_bert::pipelines::common::ModelType;
//! # fn main() -> anyhow::Result<()> {
//!
//! //Load a configuration
//! use rust_bert::pipelines::token_classification::LabelAggregationOption;
//! let config = TokenClassificationConfig::new(
//!    ModelType::Bert,
//!    RemoteResource::from_pretrained(BertModelResources::BERT_NER),
//!    RemoteResource::from_pretrained(BertVocabResources::BERT_NER),
//!    RemoteResource::from_pretrained(BertConfigResources::BERT_NER),
//!    None, //merges resource only relevant with ModelType::Roberta
//!    false, //lowercase
//!    None, //strip_accents
//!    None, //add_prefix_space
//!    LabelAggregationOption::Mode
//! );
//!
//! //Create the model
//! let token_classification_model = TokenClassificationModel::new(config)?;
//!
//! let input = [
//!     "My name is Amy. I live in Paris.",
//!     "Paris is a city in France."
//! ];
//! let output = token_classification_model.predict(&input, true, true); //ignore_first_label = true (only returns the NER parts, ignoring first label O)
//! # Ok(())
//! # }
//! ```
//! Output: \
//! ```no_run
//! # use rust_bert::pipelines::token_classification::Token;
//! use rust_tokenizers::{Mask, Offset};
//! # let output =
//! [
//!     Token {
//!         text: String::from("[CLS]"),
//!         score: 0.9995001554489136,
//!         label: String::from("O"),
//!         label_index: 0,
//!         sentence: 0,
//!         index: 0,
//!         word_index: 0,
//!         offset: None,
//!         mask: Mask::Special,
//!     },
//!     Token {
//!         text: String::from("My"),
//!         score: 0.9980450868606567,
//!         label: String::from("O"),
//!         label_index: 0,
//!         sentence: 0,
//!         index: 1,
//!         word_index: 1,
//!         offset: Some(Offset { begin: 0, end: 2 }),
//!         mask: Mask::None,
//!     },
//!     Token {
//!         text: String::from("name"),
//!         score: 0.9995062351226807,
//!         label: String::from("O"),
//!         label_index: 0,
//!         sentence: 0,
//!         index: 2,
//!         word_index: 2,
//!         offset: Some(Offset { begin: 3, end: 7 }),
//!         mask: Mask::None,
//!     },
//!     Token {
//!         text: String::from("is"),
//!         score: 0.9997343420982361,
//!         label: String::from("O"),
//!         label_index: 0,
//!         sentence: 0,
//!         index: 3,
//!         word_index: 3,
//!         offset: Some(Offset { begin: 8, end: 10 }),
//!         mask: Mask::None,
//!     },
//!     Token {
//!         text: String::from("Amélie"),
//!         score: 0.9913727683112525,
//!         label: String::from("I-PER"),
//!         label_index: 4,
//!         sentence: 0,
//!         index: 4,
//!         word_index: 4,
//!         offset: Some(Offset { begin: 11, end: 17 }),
//!         mask: Mask::None,
//!     }, // ...
//! ]
//! # ;
//! ```

use crate::albert::AlbertForTokenClassification;
use crate::bert::BertForTokenClassification;
use crate::common::error::RustBertError;
use crate::deberta::DebertaForTokenClassification;
use crate::distilbert::DistilBertForTokenClassification;
use crate::electra::ElectraForTokenClassification;
use crate::fnet::FNetForTokenClassification;
use crate::longformer::LongformerForTokenClassification;
use crate::mobilebert::MobileBertForTokenClassification;
use crate::pipelines::common::{ConfigOption, ModelType, TokenizerOption};
use crate::resources::ResourceProvider;
use crate::roberta::RobertaForTokenClassification;
use crate::xlnet::XLNetForTokenClassification;
use ordered_float::OrderedFloat;
use rust_tokenizers::tokenizer::Tokenizer;
use rust_tokenizers::{
    ConsolidatableTokens, ConsolidatedTokenIterator, Mask, Offset, TokenIdsWithOffsets, TokenTrait,
    TokenizedInput,
};
use serde::{Deserialize, Serialize};
use std::borrow::Borrow;
use std::cmp::min;
use std::collections::HashMap;
use tch::nn::VarStore;
use tch::{nn, no_grad, Device, Kind, Tensor};

use crate::deberta_v2::DebertaV2ForTokenClassification;
#[cfg(feature = "remote")]
use crate::{
    bert::{BertConfigResources, BertModelResources, BertVocabResources},
    resources::RemoteResource,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
/// # Token generated by a `TokenClassificationModel`
pub struct Token {
    /// String representation of the Token
    pub text: String,
    /// Confidence score
    pub score: f64,
    /// Token label (e.g. ORG, LOC in case of NER)
    pub label: String,
    /// Label index
    pub label_index: i64,
    /// Sentence index
    pub sentence: usize,
    /// Token position index
    pub index: u16,
    /// Token word position index
    pub word_index: u16,
    /// Token offsets
    pub offset: Option<Offset>,
    /// Token mask
    pub mask: Mask,
}

impl TokenTrait for Token {
    fn offset(&self) -> Option<Offset> {
        self.offset
    }

    fn mask(&self) -> Mask {
        self.mask
    }

    fn as_str(&self) -> &str {
        self.text.as_str()
    }
}

impl ConsolidatableTokens<Token> for Vec<Token> {
    fn iter_consolidate_tokens(&self) -> ConsolidatedTokenIterator<Token> {
        ConsolidatedTokenIterator::new(self)
    }
}

#[derive(Debug)]
struct InputFeature {
    /// Encoded input ids
    input_ids: Vec<i64>,
    /// Offsets reference to the original string
    offsets: Vec<Option<Offset>>,
    /// Token category (mask)
    mask: Vec<Mask>,
    /// per-token flag indicating if this feature carries the output label for this token
    reference_feature: Vec<bool>,
    /// Reference example index (long inputs may be broken into multiple input features)
    example_index: usize,
}

type LabelAggregationFunction = Box<fn(&[Token]) -> (i64, String)>;

/// # Enum defining the label aggregation method for sub tokens
/// Defines the behaviour for labels aggregation if the consolidation of sub-tokens is enabled.
pub enum LabelAggregationOption {
    /// The label of the first sub token is assigned to the entire token
    First,
    /// The label of the last sub token is assigned to the entire token
    Last,
    /// The most frequent sub- token is  assigned to the entire token
    Mode,
    /// The user can provide a function mapping a `&Vec<Token>` to a `(i64, String)` tuple corresponding to the label index, label String to return
    Custom(LabelAggregationFunction),
}

/// # Configuration for TokenClassificationModel
/// Contains information regarding the model to load and device to place the model on.
pub struct TokenClassificationConfig {
    /// Model type
    pub model_type: ModelType,
    /// Model weights resource (default: pretrained BERT model on CoNLL)
    pub model_resource: Box<dyn ResourceProvider + Send>,
    /// Config resource (default: pretrained BERT model on CoNLL)
    pub config_resource: Box<dyn ResourceProvider + Send>,
    /// Vocab resource (default: pretrained BERT model on CoNLL)
    pub vocab_resource: Box<dyn ResourceProvider + Send>,
    /// Merges resource (default: pretrained BERT model on CoNLL)
    pub merges_resource: Option<Box<dyn ResourceProvider + Send>>,
    /// Automatically lower case all input upon tokenization (assumes a lower-cased model)
    pub lower_case: bool,
    /// Flag indicating if the tokenizer should strip accents (normalization). Only used for BERT / ALBERT models
    pub strip_accents: Option<bool>,
    /// Flag indicating if the tokenizer should add a white space before each tokenized input (needed for some Roberta models)
    pub add_prefix_space: Option<bool>,
    /// Device to place the model on (default: CUDA/GPU when available)
    pub device: Device,
    /// Sub-tokens aggregation method (default: `LabelAggregationOption::First`)
    pub label_aggregation_function: LabelAggregationOption,
    /// Batch size for predictions
    pub batch_size: usize,
}

impl TokenClassificationConfig {
    /// Instantiate a new token classification configuration of the supplied type.
    ///
    /// # Arguments
    ///
    /// * `model_type` - `ModelType` indicating the model type to load (must match with the actual data to be loaded!)
    /// * model - The `ResourceProvider` pointing to the model to load (e.g.  model.ot)
    /// * config - The `ResourceProvider` pointing to the model configuration to load (e.g. config.json)
    /// * vocab - The `ResourceProvider` pointing to the tokenizers' vocabulary to load (e.g.  vocab.txt/vocab.json)
    /// * vocab - An optional `ResourceProvider` pointing to the tokenizers' merge file to load (e.g.  merges.txt), needed only for Roberta.
    /// * lower_case - A `bool` indicating whether the tokenizer should lower case all input (in case of a lower-cased model)
    pub fn new<RM, RC, RV>(
        model_type: ModelType,
        model_resource: RM,
        config_resource: RC,
        vocab_resource: RV,
        merges_resource: Option<RV>,
        lower_case: bool,
        strip_accents: impl Into<Option<bool>>,
        add_prefix_space: impl Into<Option<bool>>,
        label_aggregation_function: LabelAggregationOption,
    ) -> TokenClassificationConfig
    where
        RM: ResourceProvider + Send + 'static,
        RC: ResourceProvider + Send + 'static,
        RV: ResourceProvider + Send + 'static,
    {
        TokenClassificationConfig {
            model_type,
            model_resource: Box::new(model_resource),
            config_resource: Box::new(config_resource),
            vocab_resource: Box::new(vocab_resource),
            merges_resource: merges_resource.map(|r| Box::new(r) as Box<_>),
            lower_case,
            strip_accents: strip_accents.into(),
            add_prefix_space: add_prefix_space.into(),
            device: Device::cuda_if_available(),
            label_aggregation_function,
            batch_size: 64,
        }
    }
}

#[cfg(feature = "remote")]
impl Default for TokenClassificationConfig {
    /// Provides a default CoNLL-2003 NER model (English)
    fn default() -> TokenClassificationConfig {
        TokenClassificationConfig::new(
            ModelType::Bert,
            RemoteResource::from_pretrained(BertModelResources::BERT_NER),
            RemoteResource::from_pretrained(BertConfigResources::BERT_NER),
            RemoteResource::from_pretrained(BertVocabResources::BERT_NER),
            None,
            false,
            None,
            None,
            LabelAggregationOption::First,
        )
    }
}

#[allow(clippy::large_enum_variant)]
/// # Abstraction that holds one particular token sequence classifier model, for any of the supported models
pub enum TokenClassificationOption {
    /// Bert for Token Classification
    Bert(BertForTokenClassification),
    /// DeBERTa for Token Classification
    Deberta(DebertaForTokenClassification),
    /// DeBERTa V2 for Token Classification
    DebertaV2(DebertaV2ForTokenClassification),
    /// DistilBert for Token Classification
    DistilBert(DistilBertForTokenClassification),
    /// MobileBert for Token Classification
    MobileBert(MobileBertForTokenClassification),
    /// Roberta for Token Classification
    Roberta(RobertaForTokenClassification),
    /// XLM Roberta for Token Classification
    XLMRoberta(RobertaForTokenClassification),
    /// Electra for Token Classification
    Electra(ElectraForTokenClassification),
    /// Albert for Token Classification
    Albert(AlbertForTokenClassification),
    /// XLNet for Token Classification
    XLNet(XLNetForTokenClassification),
    /// Longformer for Token Classification
    Longformer(LongformerForTokenClassification),
    /// FNet for Token Classification
    FNet(FNetForTokenClassification),
}

impl TokenClassificationOption {
    /// Instantiate a new token sequence classification model of the supplied type.
    ///
    /// # Arguments
    ///
    /// * `model_type` - `ModelType` indicating the model type to load (must match with the actual data to be loaded)
    /// * `p` - `tch::nn::Path` path to the model file to load (e.g. model.ot)
    /// * `config` - A configuration (the model type of the configuration must be compatible with the value for
    /// `model_type`)
    pub fn new<'p, P>(
        model_type: ModelType,
        p: P,
        config: &ConfigOption,
    ) -> Result<Self, RustBertError>
    where
        P: Borrow<nn::Path<'p>>,
    {
        match model_type {
            ModelType::Bert => {
                if let ConfigOption::Bert(config) = config {
                    Ok(TokenClassificationOption::Bert(
                        BertForTokenClassification::new(p, config)?,
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a BertConfig for Bert!".to_string(),
                    ))
                }
            }
            ModelType::Deberta => {
                if let ConfigOption::Deberta(config) = config {
                    Ok(TokenClassificationOption::Deberta(
                        DebertaForTokenClassification::new(p, config)?,
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a DebertaConfig for DeBERTa!".to_string(),
                    ))
                }
            }
            ModelType::DebertaV2 => {
                if let ConfigOption::DebertaV2(config) = config {
                    Ok(TokenClassificationOption::DebertaV2(
                        DebertaV2ForTokenClassification::new(p, config)?,
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a DebertaConfig for DeBERTa V2!".to_string(),
                    ))
                }
            }
            ModelType::DistilBert => {
                if let ConfigOption::DistilBert(config) = config {
                    Ok(TokenClassificationOption::DistilBert(
                        DistilBertForTokenClassification::new(p, config)?,
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a DistilBertConfig for DistilBert!".to_string(),
                    ))
                }
            }
            ModelType::MobileBert => {
                if let ConfigOption::MobileBert(config) = config {
                    Ok(TokenClassificationOption::MobileBert(
                        MobileBertForTokenClassification::new(p, config)?,
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a MobileBertConfig for MobileBert!".to_string(),
                    ))
                }
            }
            ModelType::Roberta => {
                if let ConfigOption::Roberta(config) = config {
                    Ok(TokenClassificationOption::Roberta(
                        RobertaForTokenClassification::new(p, config)?,
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a RobertaConfig for Roberta!".to_string(),
                    ))
                }
            }
            ModelType::XLMRoberta => {
                if let ConfigOption::Roberta(config) = config {
                    Ok(TokenClassificationOption::XLMRoberta(
                        RobertaForTokenClassification::new(p, config)?,
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a RobertaConfig for XLMRoberta!".to_string(),
                    ))
                }
            }
            ModelType::Electra => {
                if let ConfigOption::Electra(config) = config {
                    Ok(TokenClassificationOption::Electra(
                        ElectraForTokenClassification::new(p, config)?,
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a BertConfig for Roberta!".to_string(),
                    ))
                }
            }
            ModelType::Albert => {
                if let ConfigOption::Albert(config) = config {
                    Ok(TokenClassificationOption::Albert(
                        AlbertForTokenClassification::new(p, config)?,
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply an AlbertConfig for Albert!".to_string(),
                    ))
                }
            }
            ModelType::XLNet => {
                if let ConfigOption::XLNet(config) = config {
                    Ok(TokenClassificationOption::XLNet(
                        XLNetForTokenClassification::new(p, config)?,
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply an AlbertConfig for Albert!".to_string(),
                    ))
                }
            }
            ModelType::Longformer => {
                if let ConfigOption::Longformer(config) = config {
                    Ok(TokenClassificationOption::Longformer(
                        LongformerForTokenClassification::new(p, config)?,
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a LongformerConfig for Longformer!".to_string(),
                    ))
                }
            }
            ModelType::FNet => {
                if let ConfigOption::FNet(config) = config {
                    Ok(TokenClassificationOption::FNet(
                        FNetForTokenClassification::new(p, config)?,
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply an FNetConfig for FNet!".to_string(),
                    ))
                }
            }
            _ => Err(RustBertError::InvalidConfigurationError(format!(
                "Token classification not implemented for {model_type:?}!"
            ))),
        }
    }

    /// Returns the `ModelType` for this TokenClassificationOption
    pub fn model_type(&self) -> ModelType {
        match *self {
            Self::Bert(_) => ModelType::Bert,
            Self::Deberta(_) => ModelType::Deberta,
            Self::DebertaV2(_) => ModelType::DebertaV2,
            Self::Roberta(_) => ModelType::Roberta,
            Self::XLMRoberta(_) => ModelType::XLMRoberta,
            Self::DistilBert(_) => ModelType::DistilBert,
            Self::MobileBert(_) => ModelType::MobileBert,
            Self::Electra(_) => ModelType::Electra,
            Self::Albert(_) => ModelType::Albert,
            Self::XLNet(_) => ModelType::XLNet,
            Self::Longformer(_) => ModelType::Longformer,
            Self::FNet(_) => ModelType::FNet,
        }
    }

    fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        mask: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        train: bool,
    ) -> Tensor {
        match *self {
            Self::Bert(ref model) => {
                model
                    .forward_t(
                        input_ids,
                        mask,
                        token_type_ids,
                        position_ids,
                        input_embeds,
                        train,
                    )
                    .logits
            }
            Self::Deberta(ref model) => {
                model
                    .forward_t(
                        input_ids,
                        mask,
                        token_type_ids,
                        position_ids,
                        input_embeds,
                        train,
                    )
                    .expect("Error in DeBERTa forward_t")
                    .logits
            }
            Self::DebertaV2(ref model) => {
                model
                    .forward_t(
                        input_ids,
                        mask,
                        token_type_ids,
                        position_ids,
                        input_embeds,
                        train,
                    )
                    .expect("Error in DeBERTa V2 forward_t")
                    .logits
            }
            Self::DistilBert(ref model) => {
                model
                    .forward_t(input_ids, mask, input_embeds, train)
                    .expect("Error in distilbert forward_t")
                    .logits
            }
            Self::MobileBert(ref model) => {
                model
                    .forward_t(input_ids, None, None, input_embeds, mask, train)
                    .expect("Error in mobilebert forward_t")
                    .logits
            }
            Self::Roberta(ref model) | Self::XLMRoberta(ref model) => {
                model
                    .forward_t(
                        input_ids,
                        mask,
                        token_type_ids,
                        position_ids,
                        input_embeds,
                        train,
                    )
                    .logits
            }
            Self::Electra(ref model) => {
                model
                    .forward_t(
                        input_ids,
                        mask,
                        token_type_ids,
                        position_ids,
                        input_embeds,
                        train,
                    )
                    .logits
            }
            Self::Albert(ref model) => {
                model
                    .forward_t(
                        input_ids,
                        mask,
                        token_type_ids,
                        position_ids,
                        input_embeds,
                        train,
                    )
                    .logits
            }
            Self::XLNet(ref model) => {
                model
                    .forward_t(
                        input_ids,
                        mask,
                        None,
                        None,
                        None,
                        token_type_ids,
                        input_embeds,
                        train,
                    )
                    .logits
            }
            Self::Longformer(ref model) => {
                model
                    .forward_t(
                        input_ids,
                        mask,
                        None,
                        token_type_ids,
                        position_ids,
                        input_embeds,
                        train,
                    )
                    .expect("Error in longformer forward_t")
                    .logits
            }
            Self::FNet(ref model) => {
                model
                    .forward_t(input_ids, token_type_ids, position_ids, input_embeds, train)
                    .expect("Error in fnet forward_t")
                    .logits
            }
        }
    }
}

/// # TokenClassificationModel for Named Entity Recognition or Part-of-Speech tagging
pub struct TokenClassificationModel {
    tokenizer: TokenizerOption,
    token_sequence_classifier: TokenClassificationOption,
    label_mapping: HashMap<i64, String>,
    var_store: VarStore,
    label_aggregation_function: LabelAggregationOption,
    max_length: usize,
    batch_size: usize,
}

impl TokenClassificationModel {
    /// Build a new `TokenClassificationModel`
    ///
    /// # Arguments
    ///
    /// * `config` - `TokenClassificationConfig` object containing the resource references (model, vocabulary, configuration) and device placement (CPU/GPU)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> anyhow::Result<()> {
    /// use rust_bert::pipelines::token_classification::TokenClassificationModel;
    ///
    /// let model = TokenClassificationModel::new(Default::default())?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(
        config: TokenClassificationConfig,
    ) -> Result<TokenClassificationModel, RustBertError> {
        let vocab_path = config.vocab_resource.get_local_path()?;
        let merges_path = config
            .merges_resource
            .as_ref()
            .map(|resource| resource.get_local_path())
            .transpose()?;

        let tokenizer = TokenizerOption::from_file(
            config.model_type,
            vocab_path.to_str().unwrap(),
            merges_path.as_deref().map(|path| path.to_str().unwrap()),
            config.lower_case,
            config.strip_accents,
            config.add_prefix_space,
        )?;
        Self::new_with_tokenizer(config, tokenizer)
    }

    /// Build a new `TokenClassificationModel` with a provided tokenizer.
    ///
    /// # Arguments
    ///
    /// * `config` - `TokenClassificationConfig` object containing the resource references (model, vocabulary, configuration) and device placement (CPU/GPU)
    /// * `tokenizer` - `TokenizerOption` tokenizer to use for token classification
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> anyhow::Result<()> {
    /// use rust_bert::pipelines::common::{ModelType, TokenizerOption};
    /// use rust_bert::pipelines::token_classification::TokenClassificationModel;
    /// let tokenizer = TokenizerOption::from_file(
    ///     ModelType::Bert,
    ///     "path/to/vocab.txt",
    ///     None,
    ///     false,
    ///     None,
    ///     None,
    /// )?;
    /// let model = TokenClassificationModel::new_with_tokenizer(Default::default(), tokenizer)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new_with_tokenizer(
        config: TokenClassificationConfig,
        tokenizer: TokenizerOption,
    ) -> Result<TokenClassificationModel, RustBertError> {
        let config_path = config.config_resource.get_local_path()?;
        let weights_path = config.model_resource.get_local_path()?;
        let device = config.device;
        let label_aggregation_function = config.label_aggregation_function;

        let mut var_store = VarStore::new(device);
        let model_config = ConfigOption::from_file(config.model_type, config_path);
        let max_length = model_config
            .get_max_len()
            .map(|v| v as usize)
            .unwrap_or(usize::MAX);
        let token_sequence_classifier =
            TokenClassificationOption::new(config.model_type, var_store.root(), &model_config)?;
        let label_mapping = model_config.get_label_mapping().clone();
        let batch_size = config.batch_size;
        var_store.load(weights_path)?;
        Ok(TokenClassificationModel {
            tokenizer,
            token_sequence_classifier,
            label_mapping,
            var_store,
            label_aggregation_function,
            max_length,
            batch_size,
        })
    }

    /// Get a reference to the model tokenizer.
    pub fn get_tokenizer(&self) -> &TokenizerOption {
        &self.tokenizer
    }

    /// Get a mutable reference to the model tokenizer.
    pub fn get_tokenizer_mut(&mut self) -> &mut TokenizerOption {
        &mut self.tokenizer
    }

    fn generate_features<S>(&self, input: S, example_index: usize) -> Vec<InputFeature>
    where
        S: AsRef<str>,
    {
        let tokenized_input = self.tokenizer.tokenize_with_offsets(input.as_ref());
        let encoded_input = TokenIdsWithOffsets {
            ids: self
                .tokenizer
                .convert_tokens_to_ids(&tokenized_input.tokens),
            offsets: tokenized_input.offsets,
            reference_offsets: tokenized_input.reference_offsets,
            masks: tokenized_input.masks,
        };

        let sequence_added_tokens = self
            .tokenizer
            .build_input_with_special_tokens(
                TokenIdsWithOffsets {
                    ids: vec![],
                    offsets: vec![],
                    reference_offsets: vec![],
                    masks: vec![],
                },
                None,
            )
            .token_ids
            .len();

        let max_content_length = self.max_length - sequence_added_tokens;
        let doc_stride = self.max_length / 4;

        let mut spans: Vec<InputFeature> = vec![];
        let mut start_token = 0_usize;
        let total_length = encoded_input.ids.len();

        while (spans.len() * doc_stride) < encoded_input.ids.len() {
            let end_token = min(start_token + max_content_length, total_length);
            let sub_encoded_input = TokenIdsWithOffsets {
                ids: encoded_input.ids[start_token..end_token].to_vec(),
                offsets: encoded_input.offsets[start_token..end_token].to_vec(),
                reference_offsets: encoded_input.reference_offsets[start_token..end_token].to_vec(),
                masks: encoded_input.masks[start_token..end_token].to_vec(),
            };

            let encoded_span = self
                .tokenizer
                .build_input_with_special_tokens(sub_encoded_input, None);

            let reference_feature = self.get_reference_feature_flag(
                start_token,
                end_token,
                total_length,
                doc_stride,
                &encoded_span,
            );

            let feature = InputFeature {
                input_ids: encoded_span.token_ids,
                offsets: encoded_span.token_offsets,
                mask: encoded_span.mask,
                reference_feature,
                example_index,
            };
            spans.push(feature);
            if end_token == encoded_input.ids.len() {
                break;
            }
            start_token = end_token - doc_stride;
        }
        spans
    }

    fn get_reference_feature_flag(
        &self,
        start_token: usize,
        end_token: usize,
        total_length: usize,
        doc_stride: usize,
        encoded_span: &TokenizedInput,
    ) -> Vec<bool> {
        // set halfway through the doc_stride to be false if the feature is not the first/last
        let start_cutoff = if start_token > 0 {
            let leading_special_tokens = {
                let mut counter = 0;
                let mut masks = encoded_span.mask.iter();
                while masks.next().unwrap_or(&Mask::None) == &Mask::Special {
                    counter += 1;
                }
                counter
            };
            doc_stride / 2 + leading_special_tokens
        } else {
            0
        };
        let end_cutoff = if end_token < total_length {
            let trailing_special_tokens = {
                let mut counter = 0;
                let mut masks = encoded_span.mask.iter().rev();
                while masks.next().unwrap_or(&Mask::None) == &Mask::Special {
                    counter += 1;
                }
                counter
            };
            encoded_span.token_ids.len() - doc_stride / 2 - trailing_special_tokens
        } else {
            encoded_span.token_ids.len()
        };
        let mut reference_feature = vec![true; encoded_span.token_ids.len()];
        reference_feature[..start_cutoff]
            .iter_mut()
            .for_each(|v| *v = false);
        reference_feature[end_cutoff..]
            .iter_mut()
            .for_each(|v| *v = false);
        reference_feature
    }

    /// Classify tokens in a text sequence
    ///
    /// # Arguments
    ///
    /// * `input` - `&[&str]` Array of texts to extract entities from.
    /// * `consolidate_subtokens` - bool flag indicating if subtokens should be consolidated at the token level
    /// * `return_special` - bool flag indicating if labels for special tokens should be returned
    ///
    /// # Returns
    ///
    /// * `Vec<Vec<Token>>` containing Tokens with associated labels (for example POS tags) for each input provided
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> anyhow::Result<()> {
    /// # use rust_bert::pipelines::token_classification::TokenClassificationModel;
    ///
    /// let ner_model = TokenClassificationModel::new(Default::default())?;
    /// let input = [
    ///     "My name is Amy. I live in Paris.",
    ///     "Paris is a city in France.",
    /// ];
    /// let output = ner_model.predict(&input, true, true);
    /// # Ok(())
    /// # }
    /// ```
    pub fn predict<S>(
        &self,
        input: &[S],
        consolidate_sub_tokens: bool,
        return_special: bool,
    ) -> Vec<Vec<Token>>
    where
        S: AsRef<str>,
    {
        let mut features: Vec<InputFeature> = input
            .iter()
            .enumerate()
            .flat_map(|(example_index, example)| self.generate_features(example, example_index))
            .collect();

        let mut example_tokens_map: Vec<Vec<Token>> = vec![Vec::new(); input.len()];
        let mut start = 0usize;
        let len_features = features.len();

        while start < len_features {
            let end = start + min(len_features - start, self.batch_size);

            no_grad(|| {
                let batch_features = &mut features[start..end];
                let (input_ids, attention_masks) = self.pad_features(batch_features);
                let output = self.token_sequence_classifier.forward_t(
                    Some(&input_ids),
                    Some(&attention_masks),
                    None,
                    None,
                    None,
                    false,
                );
                let score = output.exp()
                    / output
                        .exp()
                        .sum_dim_intlist([-1].as_slice(), true, Kind::Float);
                let label_indices = score.argmax(-1, true);
                for sentence_idx in 0..label_indices.size()[0] {
                    let labels = label_indices.get(sentence_idx);
                    let feature = &features[sentence_idx as usize];
                    let sentence_reference_flag = &feature.reference_feature;
                    let original_chars = input[feature.example_index]
                        .as_ref()
                        .chars()
                        .collect::<Vec<char>>();
                    let mut word_idx: u16 = 0;
                    for position_idx in sentence_reference_flag
                        .iter()
                        .enumerate()
                        .filter(|(_, flag)| **flag)
                        .map(|(pos, _)| pos)
                    {
                        let mask = feature.mask[position_idx];
                        if (mask == Mask::Special) & (!return_special) {
                            continue;
                        }
                        if !(mask == Mask::Continuation) {
                            word_idx += 1;
                        }
                        let token = {
                            self.decode_token(
                                &original_chars,
                                feature,
                                &input_ids,
                                &labels,
                                &score,
                                sentence_idx,
                                position_idx as i64,
                                word_idx,
                            )
                        };
                        example_tokens_map[feature.example_index].push(token);
                    }
                }
            });
            start = end;
        }
        let mut tokens = example_tokens_map;

        if consolidate_sub_tokens {
            self.consolidate_tokens(&mut tokens, &self.label_aggregation_function);
        }
        tokens
    }

    fn pad_features(&self, features: &mut [InputFeature]) -> (Tensor, Tensor) {
        let max_len = features
            .iter()
            .map(|feature| feature.input_ids.len())
            .max()
            .unwrap();

        let attention_masks = features
            .iter()
            .map(|feature| &feature.input_ids)
            .map(|input| {
                let mut attention_mask = Vec::with_capacity(max_len);
                attention_mask.resize(input.len(), 1);
                attention_mask.resize(max_len, 0);
                attention_mask
            })
            .map(|input| Tensor::from_slice(&(input)))
            .collect::<Vec<_>>();

        let padding_index = self
            .tokenizer
            .get_pad_id()
            .expect("Only tokenizers with a padding index can be used for token classification");
        for feature in features.iter_mut() {
            feature.input_ids.resize(max_len, padding_index);
            feature.offsets.resize(max_len, None);
            feature.reference_feature.resize(max_len, false);
        }

        let padded_input_ids = features
            .iter()
            .map(|input| Tensor::from_slice(input.input_ids.as_slice()))
            .collect::<Vec<_>>();

        let input_ids = Tensor::stack(&padded_input_ids, 0).to(self.var_store.device());
        let attention_masks = Tensor::stack(&attention_masks, 0).to(self.var_store.device());
        (input_ids, attention_masks)
    }

    fn decode_token(
        &self,
        original_sentence_chars: &[char],
        sentence_tokens: &InputFeature,
        input_tensor: &Tensor,
        labels: &Tensor,
        score: &Tensor,
        sentence_idx: i64,
        position_idx: i64,
        word_index: u16,
    ) -> Token {
        let label_id = labels.int64_value(&[position_idx]);
        let token_id = input_tensor.int64_value(&[sentence_idx, position_idx]);

        let offsets = &sentence_tokens.offsets[position_idx as usize];

        let text = match offsets {
            None => match self.tokenizer {
                TokenizerOption::Bert(ref tokenizer) => {
                    Tokenizer::decode(tokenizer, &[token_id], false, false)
                }
                TokenizerOption::Roberta(ref tokenizer) => {
                    Tokenizer::decode(tokenizer, &[token_id], false, false)
                }
                TokenizerOption::XLMRoberta(ref tokenizer) => {
                    Tokenizer::decode(tokenizer, &[token_id], false, false)
                }
                TokenizerOption::Albert(ref tokenizer) => {
                    Tokenizer::decode(tokenizer, &[token_id], false, false)
                }
                TokenizerOption::XLNet(ref tokenizer) => {
                    Tokenizer::decode(tokenizer, &[token_id], false, false)
                }
                _ => panic!(
                    "Token classification not implemented for {:?}!",
                    self.tokenizer.model_type()
                ),
            },
            Some(offsets) => {
                let (start_char, end_char) = (offsets.begin as usize, offsets.end as usize);
                let end_char = min(end_char, original_sentence_chars.len());
                let text = original_sentence_chars[start_char..end_char]
                    .iter()
                    .collect();
                text
            }
        };

        Token {
            text,
            score: score.double_value(&[sentence_idx, position_idx, label_id]),
            label: self
                .label_mapping
                .get(&label_id)
                .expect("Index out of vocabulary bounds.")
                .to_owned(),
            label_index: label_id,
            sentence: sentence_idx as usize,
            index: position_idx as u16,
            word_index,
            offset: offsets.to_owned(),
            mask: sentence_tokens.mask[position_idx as usize],
        }
    }

    fn consolidate_tokens(
        &self,
        tokens: &mut Vec<Vec<Token>>,
        label_aggregation_function: &LabelAggregationOption,
    ) {
        for sequence_tokens in tokens {
            let mut tokens_to_replace = vec![];
            let token_iter = sequence_tokens.iter_consolidate_tokens();
            let mut cursor = 0;

            for sub_tokens in token_iter {
                if sub_tokens.len() > 1 {
                    let (label_index, label) =
                        self.consolidate_labels(sub_tokens, label_aggregation_function);
                    let sentence = (sub_tokens[0]).sentence;
                    let index = (sub_tokens[0]).index;
                    let word_index = (sub_tokens[0]).word_index;
                    let offset_start = sub_tokens
                        .first()
                        .unwrap()
                        .offset
                        .as_ref()
                        .map(|offset| offset.begin);
                    let offset_end = sub_tokens
                        .last()
                        .unwrap()
                        .offset
                        .as_ref()
                        .map(|offset| offset.end);
                    let offset = if let (Some(offset_start), Some(offset_end)) =
                        (offset_start, offset_end)
                    {
                        Some(Offset::new(offset_start, offset_end))
                    } else {
                        None
                    };
                    let mut text = String::new();
                    let mut score = 1f64;
                    for current_sub_token in sub_tokens.iter() {
                        text.push_str(current_sub_token.text.as_str());
                        score *= if current_sub_token.label_index == label_index {
                            current_sub_token.score
                        } else {
                            1.0 - current_sub_token.score
                        };
                    }
                    let token = Token {
                        text,
                        score,
                        label,
                        label_index,
                        sentence,
                        index,
                        word_index,
                        offset,
                        mask: Default::default(),
                    };
                    tokens_to_replace.push(((cursor, cursor + sub_tokens.len()), token));
                }
                cursor += sub_tokens.len();
            }
            for ((start, end), token) in tokens_to_replace.into_iter().rev() {
                sequence_tokens.splice(start..end, [token].iter().cloned());
            }
        }
    }

    fn consolidate_labels(
        &self,
        tokens: &[Token],
        aggregation: &LabelAggregationOption,
    ) -> (i64, String) {
        match aggregation {
            LabelAggregationOption::First => {
                let token = tokens.first().unwrap();
                (token.label_index, token.label.clone())
            }
            LabelAggregationOption::Last => {
                let token = tokens.last().unwrap();
                (token.label_index, token.label.clone())
            }
            LabelAggregationOption::Mode => {
                let counts = tokens.iter().fold(HashMap::new(), |mut m, c| {
                    let (ref mut count, ref mut score) = m
                        .entry((c.label_index, c.label.as_str()))
                        .or_insert((0, 0.0_f64));
                    *count += 1;
                    *score = score.max(c.score);
                    m
                });
                counts
                    .into_iter()
                    .max_by_key(|&(_, (count, score))| (count, OrderedFloat(score)))
                    .map(|((label_index, label), _)| (label_index, label.to_owned()))
                    .unwrap()
            }
            LabelAggregationOption::Custom(function) => function(tokens),
        }
    }
}
