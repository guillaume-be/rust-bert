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
//! use rust_bert::resources::{Resource,RemoteResource};
//! use rust_bert::bert::{BertModelResources, BertVocabResources, BertConfigResources};
//! use rust_bert::pipelines::common::ModelType;
//! # fn main() -> anyhow::Result<()> {
//!
//! //Load a configuration
//! use rust_bert::pipelines::token_classification::LabelAggregationOption;
//! let config = TokenClassificationConfig::new(ModelType::Bert,
//!    Resource::Remote(RemoteResource::from_pretrained(BertModelResources::BERT_NER)),
//!    Resource::Remote(RemoteResource::from_pretrained(BertVocabResources::BERT_NER)),
//!    Resource::Remote(RemoteResource::from_pretrained(BertConfigResources::BERT_NER)),
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
use crate::bert::{
    BertConfigResources, BertForTokenClassification, BertModelResources, BertVocabResources,
};
use crate::common::error::RustBertError;
use crate::common::resources::{RemoteResource, Resource};
use crate::distilbert::DistilBertForTokenClassification;
use crate::electra::ElectraForTokenClassification;
use crate::longformer::LongformerForTokenClassification;
use crate::mobilebert::MobileBertForTokenClassification;
use crate::pipelines::common::{ConfigOption, ModelType, TokenizerOption};
use crate::roberta::RobertaForTokenClassification;
use crate::xlnet::XLNetForTokenClassification;
use itertools::Itertools;
use rust_tokenizers::tokenizer::{Tokenizer, TruncationStrategy};
use rust_tokenizers::{
    ConsolidatableTokens, ConsolidatedTokenIterator, Mask, Offset, TokenTrait, TokenizedInput,
};
use serde::{Deserialize, Serialize};
use std::borrow::Borrow;
use std::cmp::min;
use std::collections::HashMap;
use tch::kind::Kind::Float;
use tch::nn::VarStore;
use tch::{nn, no_grad, Device, Tensor};

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
    pub model_resource: Resource,
    /// Config resource (default: pretrained BERT model on CoNLL)
    pub config_resource: Resource,
    /// Vocab resource (default: pretrained BERT model on CoNLL)
    pub vocab_resource: Resource,
    /// Merges resource (default: pretrained BERT model on CoNLL)
    pub merges_resource: Option<Resource>,
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
}

impl TokenClassificationConfig {
    /// Instantiate a new token classification configuration of the supplied type.
    ///
    /// # Arguments
    ///
    /// * `model_type` - `ModelType` indicating the model type to load (must match with the actual data to be loaded!)
    /// * model - The `Resource` pointing to the model to load (e.g.  model.ot)
    /// * config - The `Resource' pointing to the model configuration to load (e.g. config.json)
    /// * vocab - The `Resource' pointing to the tokenizers' vocabulary to load (e.g.  vocab.txt/vocab.json)
    /// * vocab - An optional `Resource` tuple (`Option<Resource>`) pointing to the tokenizers' merge file to load (e.g.  merges.txt), needed only for Roberta.
    /// * lower_case - A `bool' indicating whether the tokenizer should lower case all input (in case of a lower-cased model)
    pub fn new(
        model_type: ModelType,
        model_resource: Resource,
        config_resource: Resource,
        vocab_resource: Resource,
        merges_resource: Option<Resource>,
        lower_case: bool,
        strip_accents: impl Into<Option<bool>>,
        add_prefix_space: impl Into<Option<bool>>,
        label_aggregation_function: LabelAggregationOption,
    ) -> TokenClassificationConfig {
        TokenClassificationConfig {
            model_type,
            model_resource,
            config_resource,
            vocab_resource,
            merges_resource,
            lower_case,
            strip_accents: strip_accents.into(),
            add_prefix_space: add_prefix_space.into(),
            device: Device::cuda_if_available(),
            label_aggregation_function,
        }
    }
}

impl Default for TokenClassificationConfig {
    /// Provides a default CoNLL-2003 NER model (English)
    fn default() -> TokenClassificationConfig {
        TokenClassificationConfig {
            model_type: ModelType::Bert,
            model_resource: Resource::Remote(RemoteResource::from_pretrained(
                BertModelResources::BERT_NER,
            )),
            config_resource: Resource::Remote(RemoteResource::from_pretrained(
                BertConfigResources::BERT_NER,
            )),
            vocab_resource: Resource::Remote(RemoteResource::from_pretrained(
                BertVocabResources::BERT_NER,
            )),
            merges_resource: None,
            lower_case: false,
            strip_accents: None,
            add_prefix_space: None,
            device: Device::cuda_if_available(),
            label_aggregation_function: LabelAggregationOption::First,
        }
    }
}

/// # Abstraction that holds one particular token sequence classifier model, for any of the supported models
pub enum TokenClassificationOption {
    /// Bert for Token Classification
    Bert(BertForTokenClassification),
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
                        BertForTokenClassification::new(p, config),
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a BertConfig for Bert!".to_string(),
                    ))
                }
            }
            ModelType::DistilBert => {
                if let ConfigOption::DistilBert(config) = config {
                    Ok(TokenClassificationOption::DistilBert(
                        DistilBertForTokenClassification::new(p, config),
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
                        MobileBertForTokenClassification::new(p, config),
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a MobileBertConfig for MobileBert!".to_string(),
                    ))
                }
            }
            ModelType::Roberta => {
                if let ConfigOption::Bert(config) = config {
                    Ok(TokenClassificationOption::Roberta(
                        RobertaForTokenClassification::new(p, config),
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a BertConfig for Roberta!".to_string(),
                    ))
                }
            }
            ModelType::XLMRoberta => {
                if let ConfigOption::Bert(config) = config {
                    Ok(TokenClassificationOption::XLMRoberta(
                        RobertaForTokenClassification::new(p, config),
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a BertConfig for XLMRoberta!".to_string(),
                    ))
                }
            }
            ModelType::Electra => {
                if let ConfigOption::Electra(config) = config {
                    Ok(TokenClassificationOption::Electra(
                        ElectraForTokenClassification::new(p, config),
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
                        AlbertForTokenClassification::new(p, config),
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
                        XLNetForTokenClassification::new(p, config).unwrap(),
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
                        LongformerForTokenClassification::new(p, config),
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a LongformerConfig for Longformer!".to_string(),
                    ))
                }
            }
            _ => Err(RustBertError::InvalidConfigurationError(format!(
                "Token classification not implemented for {:?}!",
                model_type
            ))),
        }
    }

    /// Returns the `ModelType` for this TokenClassificationOption
    pub fn model_type(&self) -> ModelType {
        match *self {
            Self::Bert(_) => ModelType::Bert,
            Self::Roberta(_) => ModelType::Roberta,
            Self::XLMRoberta(_) => ModelType::XLMRoberta,
            Self::DistilBert(_) => ModelType::DistilBert,
            Self::MobileBert(_) => ModelType::MobileBert,
            Self::Electra(_) => ModelType::Electra,
            Self::Albert(_) => ModelType::Albert,
            Self::XLNet(_) => ModelType::XLNet,
            Self::Longformer(_) => ModelType::Longformer,
        }
    }

    fn forward_t(
        &self,
        input_ids: Option<Tensor>,
        mask: Option<Tensor>,
        token_type_ids: Option<Tensor>,
        position_ids: Option<Tensor>,
        input_embeds: Option<Tensor>,
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
            Self::DistilBert(ref model) => {
                model
                    .forward_t(input_ids, mask, input_embeds, train)
                    .expect("Error in distilbert forward_t")
                    .logits
            }
            Self::MobileBert(ref model) => {
                model
                    .forward_t(
                        input_ids.as_ref(),
                        None,
                        None,
                        input_embeds,
                        mask.as_ref(),
                        train,
                    )
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
                        input_ids.as_ref(),
                        mask.as_ref(),
                        None,
                        None,
                        None,
                        token_type_ids.as_ref(),
                        input_embeds,
                        train,
                    )
                    .logits
            }
            Self::Longformer(ref model) => {
                model
                    .forward_t(
                        input_ids.as_ref(),
                        mask.as_ref(),
                        None,
                        token_type_ids.as_ref(),
                        position_ids.as_ref(),
                        input_embeds.as_ref(),
                        train,
                    )
                    .expect("Error in longformer forward_t")
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
        let config_path = config.config_resource.get_local_path()?;
        let vocab_path = config.vocab_resource.get_local_path()?;
        let weights_path = config.model_resource.get_local_path()?;
        let merges_path = if let Some(merges_resource) = &config.merges_resource {
            Some(merges_resource.get_local_path()?)
        } else {
            None
        };
        let device = config.device;
        let label_aggregation_function = config.label_aggregation_function;

        let tokenizer = TokenizerOption::from_file(
            config.model_type,
            vocab_path.to_str().unwrap(),
            merges_path.as_deref().map(|path| path.to_str().unwrap()),
            config.lower_case,
            config.strip_accents,
            config.add_prefix_space,
        )?;
        let mut var_store = VarStore::new(device);
        let model_config = ConfigOption::from_file(config.model_type, config_path);
        let token_sequence_classifier =
            TokenClassificationOption::new(config.model_type, &var_store.root(), &model_config)?;
        let label_mapping = model_config.get_label_mapping();
        var_store.load(weights_path)?;
        Ok(TokenClassificationModel {
            tokenizer,
            token_sequence_classifier,
            label_mapping,
            var_store,
            label_aggregation_function,
        })
    }

    fn prepare_for_model<'a, S>(&self, input: S) -> (Vec<TokenizedInput>, Tensor)
    where
        S: AsRef<[&'a str]>,
    {
        let tokenized_input: Vec<TokenizedInput> =
            self.tokenizer
                .encode_list(input.as_ref(), 128, &TruncationStrategy::LongestFirst, 0);
        let max_len = tokenized_input
            .iter()
            .map(|input| input.token_ids.len())
            .max()
            .unwrap();
        let tokenized_input_tensors: Vec<tch::Tensor> = tokenized_input
            .iter()
            .map(|input| input.token_ids.clone())
            .map(|mut input| {
                input.extend(vec![
                    self.tokenizer.get_pad_id().expect(
                        "The Tokenizer used for token classification should contain a PAD id"
                    );
                    max_len - input.len()
                ]);
                input
            })
            .map(|input| Tensor::of_slice(&(input)))
            .collect::<Vec<_>>();
        (
            tokenized_input,
            Tensor::stack(tokenized_input_tensors.as_slice(), 0).to(self.var_store.device()),
        )
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
    /// * `Vec<Token>` containing Tokens with associated labels (for example POS tags)
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
    pub fn predict<'a, S>(
        &self,
        input: S,
        consolidate_sub_tokens: bool,
        return_special: bool,
    ) -> Vec<Token>
    where
        S: AsRef<[&'a str]>,
    {
        let (tokenized_input, input_tensor) = self.prepare_for_model(input.as_ref());
        let output = no_grad(|| {
            self.token_sequence_classifier.forward_t(
                Some(input_tensor.copy()),
                None,
                None,
                None,
                None,
                false,
            )
        });
        let output = output.detach().to(Device::Cpu);
        let score: Tensor = output.exp() / output.exp().sum1(&[-1], true, Float);
        let labels_idx = &score.argmax(-1, true);
        let mut tokens: Vec<Token> = vec![];
        for sentence_idx in 0..labels_idx.size()[0] {
            let labels = labels_idx.get(sentence_idx);
            let sentence_tokens = &tokenized_input[sentence_idx as usize];
            let original_chars = input.as_ref()[sentence_idx as usize].chars().collect_vec();
            let mut word_idx: u16 = 0;
            for position_idx in 0..sentence_tokens.token_ids.len() {
                let mask = sentence_tokens.mask[position_idx];
                if (mask == Mask::Special) & (!return_special) {
                    continue;
                }
                if !(mask == Mask::Continuation) {
                    word_idx += 1;
                }
                let token = {
                    self.decode_token(
                        &original_chars,
                        sentence_tokens,
                        &input_tensor,
                        &labels,
                        &score,
                        sentence_idx,
                        position_idx as i64,
                        word_idx - 1,
                    )
                };
                tokens.push(token);
            }
        }
        if consolidate_sub_tokens {
            self.consolidate_tokens(&mut tokens, &self.label_aggregation_function);
        }
        tokens
    }

    fn decode_token(
        &self,
        original_sentence_chars: &[char],
        sentence_tokens: &TokenizedInput,
        input_tensor: &Tensor,
        labels: &Tensor,
        score: &Tensor,
        sentence_idx: i64,
        position_idx: i64,
        word_index: u16,
    ) -> Token {
        let label_id = labels.int64_value(&[position_idx as i64]);
        let token_id = input_tensor.int64_value(&[sentence_idx, position_idx as i64]);

        let offsets = &sentence_tokens.token_offsets[position_idx as usize];

        let text = match offsets {
            None => match self.tokenizer {
                TokenizerOption::Bert(ref tokenizer) => {
                    Tokenizer::decode(tokenizer, vec![token_id], false, false)
                }
                TokenizerOption::Roberta(ref tokenizer) => {
                    Tokenizer::decode(tokenizer, vec![token_id], false, false)
                }
                TokenizerOption::XLMRoberta(ref tokenizer) => {
                    Tokenizer::decode(tokenizer, vec![token_id], false, false)
                }
                TokenizerOption::Albert(ref tokenizer) => {
                    Tokenizer::decode(tokenizer, vec![token_id], false, false)
                }
                TokenizerOption::XLNet(ref tokenizer) => {
                    Tokenizer::decode(tokenizer, vec![token_id], false, false)
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
        tokens: &mut Vec<Token>,
        label_aggregation_function: &LabelAggregationOption,
    ) {
        let mut tokens_to_replace = vec![];
        let token_iter = tokens.iter_consolidate_tokens();
        let mut cursor = 0;

        for sub_tokens in token_iter {
            if sub_tokens.len() > 1 {
                let (label_index, label) =
                    self.consolidate_labels(sub_tokens, label_aggregation_function);
                let sentence = (sub_tokens[0]).sentence;
                let index = (sub_tokens[0]).index;
                let word_index = (sub_tokens[0]).word_index;
                let offset_start = match &sub_tokens.first().unwrap().offset {
                    Some(offset) => Some(offset.begin),
                    None => None,
                };
                let offset_end = match &sub_tokens.last().unwrap().offset {
                    Some(offset) => Some(offset.end),
                    None => None,
                };
                let offset =
                    if let (Some(offset_start), Some(offset_end)) = (offset_start, offset_end) {
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
            tokens.splice(start..end, [token].iter().cloned());
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
                    *m.entry((c.label_index, c.label.as_str())).or_insert(0) += 1;
                    m
                });
                counts
                    .into_iter()
                    .max_by(|a, b| a.1.cmp(&b.1))
                    .map(|((label_index, label), _)| (label_index, label.to_owned()))
                    .unwrap()
            }
            LabelAggregationOption::Custom(function) => function(tokens),
        }
    }
}
