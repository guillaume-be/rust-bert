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

//! # Question Answering pipeline
//! Extractive question answering from a given question and context. By default, the dependencies for this
//! model will be downloaded for a DistilBERT model finetuned on SQuAD (Stanford Question Answering Dataset).
//! Customized DistilBERT models can be loaded by overwriting the resources in the configuration.
//! The dependencies will be downloaded to the user's home directory, under ~/.cache/.rustbert/distilbert-qa
//!
//! ```no_run
//! use rust_bert::pipelines::question_answering::{QaInput, QuestionAnsweringModel};
//!
//! # fn main() -> anyhow::Result<()> {
//! let qa_model = QuestionAnsweringModel::new(Default::default())?;
//!
//! let question = String::from("Where does Amy live ?");
//! let context = String::from("Amy lives in Amsterdam");
//!
//! let answers = qa_model.predict(&vec![QaInput { question, context }], 1, 32);
//! # Ok(())
//! # }
//! ```
//!
//! Output: \
//! ```no_run
//! # use rust_bert::pipelines::question_answering::Answer;
//! # let output =
//! [Answer {
//!     score: 0.9976,
//!     start: 13,
//!     end: 21,
//!     answer: String::from("Amsterdam"),
//! }]
//! # ;
//! ```

use crate::albert::AlbertForQuestionAnswering;
use crate::bert::BertForQuestionAnswering;
use crate::common::error::RustBertError;
use crate::deberta::DebertaForQuestionAnswering;
use crate::distilbert::DistilBertForQuestionAnswering;
use crate::fnet::FNetForQuestionAnswering;
use crate::longformer::LongformerForQuestionAnswering;
use crate::mobilebert::MobileBertForQuestionAnswering;
use crate::pipelines::common::{ConfigOption, ModelType, TokenizerOption};
use crate::reformer::ReformerForQuestionAnswering;
use crate::resources::ResourceProvider;
use crate::roberta::RobertaForQuestionAnswering;
use crate::xlnet::XLNetForQuestionAnswering;
use rust_tokenizers::{Offset, TokenIdsWithOffsets, TokenizedInput};
use serde::{Deserialize, Serialize};
use std::borrow::Borrow;
use std::cmp::min;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use tch::kind::Kind::Float;
use tch::nn::VarStore;
use tch::{nn, no_grad, Device, Tensor};

use crate::deberta_v2::DebertaV2ForQuestionAnswering;
#[cfg(feature = "remote")]
use crate::{
    distilbert::{DistilBertConfigResources, DistilBertModelResources, DistilBertVocabResources},
    resources::RemoteResource,
};

#[derive(Serialize, Deserialize)]
/// # Input for Question Answering
/// Includes a context (containing the answer) and question strings
pub struct QaInput {
    /// Question string
    pub question: String,
    /// Context or query
    pub context: String,
}

#[derive(Debug)]
struct QaFeature {
    pub input_ids: Vec<i64>,
    pub offsets: Vec<Option<Offset>>,
    pub p_mask: Vec<i8>,
    pub example_index: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// # Output for Question Answering
pub struct Answer {
    /// Confidence score
    pub score: f64,
    /// Start position of answer span
    pub start: usize,
    /// End position of answer span
    pub end: usize,
    /// Answer span
    pub answer: String,
}

impl PartialEq for Answer {
    fn eq(&self, other: &Self) -> bool {
        (self.start == other.start) && (self.end == other.end) && (self.answer == other.answer)
    }
}

fn remove_duplicates<T: PartialEq + Clone>(vector: &mut Vec<T>) -> &mut Vec<T> {
    let mut potential_duplicates = vec![];
    vector.retain(|item| {
        if potential_duplicates.contains(item) {
            false
        } else {
            potential_duplicates.push(item.clone());
            true
        }
    });
    vector
}

/// # Configuration for question answering
/// Contains information regarding the model to load and device to place the model on.
pub struct QuestionAnsweringConfig {
    /// Model weights resource (default: pretrained DistilBERT model on SQuAD)
    pub model_resource: Box<dyn ResourceProvider + Send>,
    /// Config resource (default: pretrained DistilBERT model on SQuAD)
    pub config_resource: Box<dyn ResourceProvider + Send>,
    /// Vocab resource (default: pretrained DistilBERT model on SQuAD)
    pub vocab_resource: Box<dyn ResourceProvider + Send>,
    /// Merges resource (default: None)
    pub merges_resource: Option<Box<dyn ResourceProvider + Send>>,
    /// Device to place the model on (default: CUDA/GPU when available)
    pub device: Device,
    /// Model type
    pub model_type: ModelType,
    /// Flag indicating if the model expects a lower casing of the input
    pub lower_case: bool,
    /// Flag indicating if the tokenizer should strip accents (normalization). Only used for BERT / ALBERT models
    pub strip_accents: Option<bool>,
    /// Flag indicating if the tokenizer should add a white space before each tokenized input (needed for some Roberta models)
    pub add_prefix_space: Option<bool>,
    /// Maximum sequence length for the combined query and context
    pub max_seq_length: usize,
    /// Stride to apply if the context needs to be broken down due to a large length. Represents the number of overlapping tokens between sliding windows.
    pub doc_stride: usize,
    /// Maximum length for the query
    pub max_query_length: usize,
    /// Maximum length for the answer
    pub max_answer_length: usize,
}

impl QuestionAnsweringConfig {
    /// Instantiate a new question answering configuration of the supplied type.
    ///
    /// # Arguments
    ///
    /// * `model_type` - `ModelType` indicating the model type to load (must match with the actual data to be loaded!)
    /// * model_resource - The `ResourceProvider` pointing to the model to load (e.g.  model.ot)
    /// * config_resource - The `ResourceProvider` pointing to the model configuration to load (e.g. config.json)
    /// * vocab_resource - The `ResourceProvider` pointing to the tokenizer's vocabulary to load (e.g.  vocab.txt/vocab.json)
    /// * merges_resource - An optional `ResourceProvider` pointing to the tokenizer's merge file to load (e.g.  merges.txt), needed only for Roberta.
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
    ) -> QuestionAnsweringConfig
    where
        RM: ResourceProvider + Send + 'static,
        RC: ResourceProvider + Send + 'static,
        RV: ResourceProvider + Send + 'static,
    {
        QuestionAnsweringConfig {
            model_type,
            model_resource: Box::new(model_resource),
            config_resource: Box::new(config_resource),
            vocab_resource: Box::new(vocab_resource),
            merges_resource: merges_resource.map(|r| Box::new(r) as Box<_>),
            lower_case,
            strip_accents: strip_accents.into(),
            add_prefix_space: add_prefix_space.into(),
            device: Device::cuda_if_available(),
            max_seq_length: 384,
            doc_stride: 128,
            max_query_length: 64,
            max_answer_length: 15,
        }
    }

    /// Instantiate a new question answering configuration of the supplied type.
    ///
    /// # Arguments
    ///
    /// * `model_type` - `ModelType` indicating the model type to load (must match with the actual data to be loaded!)
    /// * model_resource - The `ResourceProvider` pointing to the model to load (e.g.  model.ot)
    /// * config_resource - The `ResourceProvider` pointing to the model configuration to load (e.g. config.json)
    /// * vocab_resource - The `ResourceProvider` pointing to the tokenizer's vocabulary to load (e.g.  vocab.txt/vocab.json)
    /// * merges_resource - An optional `ResourceProvider` pointing to the tokenizer's merge file to load (e.g.  merges.txt), needed only for Roberta.
    /// * lower_case - A `bool` indicating whether the tokenizer should lower case all input (in case of a lower-cased model)
    /// * max_seq_length - Optional maximum sequence token length to limit memory footprint. If the context is too long, it will be processed with sliding windows. Defaults to 384.
    /// * max_query_length - Optional maximum question token length. Defaults to 64.
    /// * doc_stride - Optional stride to apply if a sliding window is required to process the input context. Represents the number of overlapping tokens between sliding windows. This should be lower than the max_seq_length minus max_query_length (otherwise there is a risk for the sliding window not to progress). Defaults to 128.
    /// * max_answer_length - Optional maximum token length for the extracted answer. Defaults to 15.
    pub fn custom_new<RM, RC, RV>(
        model_type: ModelType,
        model_resource: RM,
        config_resource: RC,
        vocab_resource: RV,
        merges_resource: Option<RV>,
        lower_case: bool,
        strip_accents: impl Into<Option<bool>>,
        add_prefix_space: impl Into<Option<bool>>,
        max_seq_length: impl Into<Option<usize>>,
        doc_stride: impl Into<Option<usize>>,
        max_query_length: impl Into<Option<usize>>,
        max_answer_length: impl Into<Option<usize>>,
    ) -> QuestionAnsweringConfig
    where
        RM: ResourceProvider + Send + 'static,
        RC: ResourceProvider + Send + 'static,
        RV: ResourceProvider + Send + 'static,
    {
        QuestionAnsweringConfig {
            model_type,
            model_resource: Box::new(model_resource),
            config_resource: Box::new(config_resource),
            vocab_resource: Box::new(vocab_resource),
            merges_resource: merges_resource.map(|r| Box::new(r) as Box<_>),
            lower_case,
            strip_accents: strip_accents.into(),
            add_prefix_space: add_prefix_space.into(),
            device: Device::cuda_if_available(),
            max_seq_length: max_seq_length.into().unwrap_or(384),
            doc_stride: doc_stride.into().unwrap_or(128),
            max_query_length: max_query_length.into().unwrap_or(64),
            max_answer_length: max_answer_length.into().unwrap_or(15),
        }
    }
}

#[cfg(feature = "remote")]
impl Default for QuestionAnsweringConfig {
    fn default() -> QuestionAnsweringConfig {
        QuestionAnsweringConfig {
            model_resource: Box::new(RemoteResource::from_pretrained(
                DistilBertModelResources::DISTIL_BERT_SQUAD,
            )),
            config_resource: Box::new(RemoteResource::from_pretrained(
                DistilBertConfigResources::DISTIL_BERT_SQUAD,
            )),
            vocab_resource: Box::new(RemoteResource::from_pretrained(
                DistilBertVocabResources::DISTIL_BERT_SQUAD,
            )),
            merges_resource: None,
            device: Device::cuda_if_available(),
            model_type: ModelType::DistilBert,
            lower_case: false,
            add_prefix_space: None,
            strip_accents: None,
            max_seq_length: 384,
            doc_stride: 128,
            max_query_length: 64,
            max_answer_length: 15,
        }
    }
}

#[allow(clippy::large_enum_variant)]
/// # Abstraction that holds one particular question answering model, for any of the supported models
pub enum QuestionAnsweringOption {
    /// Bert for Question Answering
    Bert(BertForQuestionAnswering),
    /// DeBERTa for Question Answering
    Deberta(DebertaForQuestionAnswering),
    /// DeBERTa V2 for Question Answering
    DebertaV2(DebertaV2ForQuestionAnswering),
    /// DistilBert for Question Answering
    DistilBert(DistilBertForQuestionAnswering),
    /// MobileBert for Question Answering
    MobileBert(MobileBertForQuestionAnswering),
    /// Roberta for Question Answering
    Roberta(RobertaForQuestionAnswering),
    /// XLMRoberta for Question Answering
    XLMRoberta(RobertaForQuestionAnswering),
    /// Albert for Question Answering
    Albert(AlbertForQuestionAnswering),
    /// XLNet for Question Answering
    XLNet(XLNetForQuestionAnswering),
    /// Reformer for Question Answering
    Reformer(ReformerForQuestionAnswering),
    /// Longformer for Question Answering
    Longformer(LongformerForQuestionAnswering),
    /// FNet for Question Answering
    FNet(FNetForQuestionAnswering),
}

impl QuestionAnsweringOption {
    /// Instantiate a new question answering model of the supplied type.
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
                    Ok(QuestionAnsweringOption::Bert(
                        BertForQuestionAnswering::new(p, config),
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a BertConfig for Bert!".to_string(),
                    ))
                }
            }
            ModelType::Deberta => {
                if let ConfigOption::Deberta(config) = config {
                    Ok(QuestionAnsweringOption::Deberta(
                        DebertaForQuestionAnswering::new(p, config),
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a DebertaConfig for DeBERTa!".to_string(),
                    ))
                }
            }
            ModelType::DebertaV2 => {
                if let ConfigOption::DebertaV2(config) = config {
                    Ok(QuestionAnsweringOption::DebertaV2(
                        DebertaV2ForQuestionAnswering::new(p, config),
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a DebertaV2Config for DeBERTa V2!".to_string(),
                    ))
                }
            }
            ModelType::DistilBert => {
                if let ConfigOption::DistilBert(config) = config {
                    Ok(QuestionAnsweringOption::DistilBert(
                        DistilBertForQuestionAnswering::new(p, config),
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a DistilBertConfig for DistilBert!".to_string(),
                    ))
                }
            }
            ModelType::MobileBert => {
                if let ConfigOption::MobileBert(config) = config {
                    Ok(QuestionAnsweringOption::MobileBert(
                        MobileBertForQuestionAnswering::new(p, config),
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a MobileBertConfig for MobileBert!".to_string(),
                    ))
                }
            }
            ModelType::Roberta => {
                if let ConfigOption::Roberta(config) = config {
                    Ok(QuestionAnsweringOption::Roberta(
                        RobertaForQuestionAnswering::new(p, config),
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a RobertaConfig for Roberta!".to_string(),
                    ))
                }
            }
            ModelType::XLMRoberta => {
                if let ConfigOption::Bert(config) = config {
                    Ok(QuestionAnsweringOption::XLMRoberta(
                        RobertaForQuestionAnswering::new(p, config),
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a BertConfig for Roberta!".to_string(),
                    ))
                }
            }
            ModelType::Albert => {
                if let ConfigOption::Albert(config) = config {
                    Ok(QuestionAnsweringOption::Albert(
                        AlbertForQuestionAnswering::new(p, config),
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply an AlbertConfig for Albert!".to_string(),
                    ))
                }
            }
            ModelType::XLNet => {
                if let ConfigOption::XLNet(config) = config {
                    Ok(QuestionAnsweringOption::XLNet(
                        XLNetForQuestionAnswering::new(p, config)?,
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a XLNetConfig for XLNet!".to_string(),
                    ))
                }
            }
            ModelType::Reformer => {
                if let ConfigOption::Reformer(config) = config {
                    Ok(QuestionAnsweringOption::Reformer(
                        ReformerForQuestionAnswering::new(p, config)?,
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a ReformerConfig for Reformer!".to_string(),
                    ))
                }
            }
            ModelType::Longformer => {
                if let ConfigOption::Longformer(config) = config {
                    Ok(QuestionAnsweringOption::Longformer(
                        LongformerForQuestionAnswering::new(p, config),
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a LongformerConfig for Longformer!".to_string(),
                    ))
                }
            }
            ModelType::FNet => {
                if let ConfigOption::FNet(config) = config {
                    Ok(QuestionAnsweringOption::FNet(
                        FNetForQuestionAnswering::new(p, config),
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a FNetConfig for FNet!".to_string(),
                    ))
                }
            }
            _ => Err(RustBertError::InvalidConfigurationError(format!(
                "QuestionAnswering not implemented for {model_type:?}!",
            ))),
        }
    }

    /// Returns the `ModelType` for this SequenceClassificationOption
    pub fn model_type(&self) -> ModelType {
        match *self {
            Self::Bert(_) => ModelType::Bert,
            Self::Deberta(_) => ModelType::Deberta,
            Self::DebertaV2(_) => ModelType::DebertaV2,
            Self::Roberta(_) => ModelType::Roberta,
            Self::XLMRoberta(_) => ModelType::XLMRoberta,
            Self::DistilBert(_) => ModelType::DistilBert,
            Self::MobileBert(_) => ModelType::MobileBert,
            Self::Albert(_) => ModelType::Albert,
            Self::XLNet(_) => ModelType::XLNet,
            Self::Reformer(_) => ModelType::Reformer,
            Self::Longformer(_) => ModelType::Longformer,
            Self::FNet(_) => ModelType::FNet,
        }
    }

    /// Interface method to forward_t() of the particular models.
    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        mask: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        train: bool,
    ) -> (Tensor, Tensor) {
        match *self {
            Self::Bert(ref model) => {
                let outputs = model.forward_t(input_ids, mask, None, None, input_embeds, train);
                (outputs.start_logits, outputs.end_logits)
            }
            Self::Deberta(ref model) => {
                let outputs = model
                    .forward_t(input_ids, mask, None, None, input_embeds, train)
                    .expect("Error in Deberta forward_t");
                (outputs.start_logits, outputs.end_logits)
            }
            Self::DebertaV2(ref model) => {
                let outputs = model
                    .forward_t(input_ids, mask, None, None, input_embeds, train)
                    .expect("Error in Deberta V2 forward_t");
                (outputs.start_logits, outputs.end_logits)
            }
            Self::DistilBert(ref model) => {
                let outputs = model
                    .forward_t(input_ids, mask, input_embeds, train)
                    .expect("Error in distilbert forward_t");
                (outputs.start_logits, outputs.end_logits)
            }
            Self::MobileBert(ref model) => {
                let outputs = model
                    .forward_t(input_ids, None, None, input_embeds, mask, train)
                    .expect("Error in mobilebert forward_t");
                (outputs.start_logits, outputs.end_logits)
            }
            Self::Roberta(ref model) | Self::XLMRoberta(ref model) => {
                let outputs = model.forward_t(input_ids, mask, None, None, input_embeds, train);
                (outputs.start_logits, outputs.end_logits)
            }
            Self::Albert(ref model) => {
                let outputs = model.forward_t(input_ids, mask, None, None, input_embeds, train);
                (outputs.start_logits, outputs.end_logits)
            }
            Self::XLNet(ref model) => {
                let outputs =
                    model.forward_t(input_ids, mask, None, None, None, None, input_embeds, train);
                (outputs.start_logits, outputs.end_logits)
            }
            Self::Reformer(ref model) => {
                let outputs = model
                    .forward_t(input_ids, None, None, mask, None, train)
                    .expect("Error in reformer forward pass");
                (outputs.start_logits, outputs.end_logits)
            }
            Self::Longformer(ref model) => {
                let outputs = model
                    .forward_t(input_ids, mask, None, None, None, None, train)
                    .expect("Error in reformer forward pass");
                (outputs.start_logits, outputs.end_logits)
            }
            Self::FNet(ref model) => {
                let outputs = model
                    .forward_t(input_ids, None, None, None, train)
                    .expect("Error in fnet forward pass");
                (outputs.start_logits, outputs.end_logits)
            }
        }
    }
}

/// # QuestionAnsweringModel to perform extractive question answering
pub struct QuestionAnsweringModel {
    tokenizer: TokenizerOption,
    pad_idx: i64,
    sep_idx: i64,
    max_seq_len: usize,
    doc_stride: usize,
    max_query_length: usize,
    max_answer_len: usize,
    qa_model: QuestionAnsweringOption,
    var_store: VarStore,
}

impl QuestionAnsweringModel {
    /// Build a new `QuestionAnsweringModel`
    ///
    /// # Arguments
    ///
    /// * `question_answering_config` - `QuestionAnsweringConfig` object containing the resource references (model, vocabulary, configuration) and device placement (CPU/GPU)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> anyhow::Result<()> {
    /// use rust_bert::pipelines::question_answering::QuestionAnsweringModel;
    ///
    /// let qa_model = QuestionAnsweringModel::new(Default::default())?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(
        question_answering_config: QuestionAnsweringConfig,
    ) -> Result<QuestionAnsweringModel, RustBertError> {
        let vocab_path = question_answering_config.vocab_resource.get_local_path()?;
        let merges_path = question_answering_config
            .merges_resource
            .as_ref()
            .map(|resource| resource.get_local_path())
            .transpose()?;

        let tokenizer = TokenizerOption::from_file(
            question_answering_config.model_type,
            vocab_path.to_str().unwrap(),
            merges_path.as_deref().map(|path| path.to_str().unwrap()),
            question_answering_config.lower_case,
            question_answering_config.strip_accents,
            question_answering_config.add_prefix_space,
        )?;
        Self::new_with_tokenizer(question_answering_config, tokenizer)
    }

    /// Build a new `QuestionAnsweringModel` with a provided tokenizer.
    ///
    /// # Arguments
    ///
    /// * `question_answering_config` - `QuestionAnsweringConfig` object containing the resource references (model, vocabulary, configuration) and device placement (CPU/GPU)
    /// * `tokenizer` - `TokenizerOption` tokenizer to use for question answering.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> anyhow::Result<()> {
    /// use rust_bert::pipelines::common::{ModelType, TokenizerOption};
    /// use rust_bert::pipelines::question_answering::QuestionAnsweringModel;
    /// let tokenizer = TokenizerOption::from_file(
    ///     ModelType::Bert,
    ///     "path/to/vocab.txt",
    ///     None,
    ///     false,
    ///     None,
    ///     None,
    /// )?;
    /// let qa_model = QuestionAnsweringModel::new_with_tokenizer(Default::default(), tokenizer)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new_with_tokenizer(
        question_answering_config: QuestionAnsweringConfig,
        tokenizer: TokenizerOption,
    ) -> Result<QuestionAnsweringModel, RustBertError> {
        let config_path = question_answering_config.config_resource.get_local_path()?;
        let weights_path = question_answering_config.model_resource.get_local_path()?;
        let device = question_answering_config.device;

        let pad_idx = tokenizer
            .get_pad_id()
            .expect("The Tokenizer used for Question Answering should contain a PAD id");
        let sep_idx = tokenizer
            .get_sep_id()
            .expect("The Tokenizer used for Question Answering should contain a SEP id");
        let mut var_store = VarStore::new(device);
        let mut model_config =
            ConfigOption::from_file(question_answering_config.model_type, config_path);

        if let ConfigOption::DistilBert(ref mut config) = model_config {
            config.sinusoidal_pos_embds = false;
        };

        let qa_model = QuestionAnsweringOption::new(
            question_answering_config.model_type,
            var_store.root(),
            &model_config,
        )?;

        if question_answering_config.max_seq_length
            < (question_answering_config.max_query_length
                + question_answering_config.doc_stride
                + 24)
        {
            return Err(RustBertError::InvalidConfigurationError(format!(
                "This configuration could cause an excessive number of sliding windows generated.\
                Please ensure max_seq_length > max_query_length + doc_stride + 24.\
                Got max_seq_length: {}, max_query_length: {}, doc_stride: {}",
                question_answering_config.max_seq_length,
                question_answering_config.max_query_length,
                question_answering_config.doc_stride
            )));
        }

        var_store.load(weights_path)?;
        Ok(QuestionAnsweringModel {
            tokenizer,
            pad_idx,
            sep_idx,
            max_seq_len: question_answering_config.max_seq_length,
            doc_stride: question_answering_config.doc_stride,
            max_query_length: question_answering_config.max_query_length,
            max_answer_len: question_answering_config.max_answer_length,
            qa_model,
            var_store,
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

    /// Perform extractive question answering given a list of `QaInputs`
    ///
    /// # Arguments
    ///
    /// * `qa_inputs` - `&[QaInput]` Array of Question Answering inputs (context and question pairs)
    /// * `top_k` - return the top-k answers for each QaInput. Set to 1 to return only the best answer.
    /// * `batch_size` - maximum batch size for the model forward pass.
    ///
    /// # Returns
    /// * `Vec<Vec<Answer>>` Vector (same length as `qa_inputs`) of vectors (each of length `top_k`) containing the extracted answers.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> anyhow::Result<()> {
    /// use rust_bert::pipelines::question_answering::{QaInput, QuestionAnsweringModel};
    ///
    /// let qa_model = QuestionAnsweringModel::new(Default::default())?;
    ///
    /// let question_1 = String::from("Where does Amy live ?");
    /// let context_1 = String::from("Amy lives in Amsterdam");
    /// let question_2 = String::from("Where does Eric live");
    /// let context_2 = String::from("While Amy lives in Amsterdam, Eric is in The Hague.");
    ///
    /// let qa_input_1 = QaInput {
    ///     question: question_1,
    ///     context: context_1,
    /// };
    /// let qa_input_2 = QaInput {
    ///     question: question_2,
    ///     context: context_2,
    /// };
    /// let answers = qa_model.predict(&[qa_input_1, qa_input_2], 1, 32);
    ///
    /// # Ok(())
    /// # }
    /// ```
    pub fn predict(
        &self,
        qa_inputs: &[QaInput],
        top_k: i64,
        batch_size: usize,
    ) -> Vec<Vec<Answer>> {
        let mut features: Vec<QaFeature> = qa_inputs
            .iter()
            .enumerate()
            .flat_map(|(example_index, qa_example)| {
                self.generate_features(
                    qa_example,
                    self.max_seq_len,
                    self.doc_stride,
                    self.max_query_length,
                    example_index as i64,
                )
            })
            .collect();

        let mut example_top_k_answers_map: HashMap<usize, Vec<Answer>> = HashMap::new();
        let mut start = 0usize;
        let len_features = features.len();

        while start < len_features {
            let end = start + min(len_features - start, batch_size);
            let batch_features = &mut features[start..end];
            no_grad(|| {
                let (input_ids, attention_masks) = self.pad_features(batch_features);

                let (start_logits, end_logits) =
                    self.qa_model
                        .forward_t(Some(&input_ids), Some(&attention_masks), None, false);

                let start_logits = start_logits.detach();
                let end_logits = end_logits.detach();
                let example_index_to_feature_end_position: Vec<(usize, i64)> = batch_features
                    .iter()
                    .enumerate()
                    .map(|(feature_index, feature)| {
                        (feature.example_index as usize, feature_index as i64 + 1)
                    })
                    .collect();

                let mut feature_id_start = 0;

                for (example_id, max_feature_id) in example_index_to_feature_end_position {
                    let mut answers: Vec<Answer> = vec![];
                    let example = &qa_inputs[example_id];
                    for feature_idx in feature_id_start..max_feature_id {
                        let feature = &batch_features[feature_idx as usize];
                        let p_mask = (Tensor::from_slice(&feature.p_mask) - 1)
                            .abs()
                            .to_device(start_logits.device())
                            .eq(0);

                        let start = start_logits.get(feature_idx).masked_fill(&p_mask, -10000);
                        let end = end_logits.get(feature_idx).masked_fill(&p_mask, -10000);

                        let start = start.exp() / start.exp().sum(Float);
                        let end = end.exp() / end.exp().sum(Float);

                        let (starts, ends, scores) = self.decode(&start, &end, top_k);

                        for idx in 0..starts.len() {
                            let start_pos = feature.offsets[starts[idx] as usize]
                                .unwrap_or(Offset { begin: 0, end: 0 })
                                .begin as usize;
                            let end_pos = feature.offsets[ends[idx] as usize]
                                .unwrap_or(Offset { begin: 0, end: 0 })
                                .end as usize;
                            let answer = example
                                .context
                                .chars()
                                .take(end_pos)
                                .skip(start_pos)
                                .collect::<String>();

                            answers.push(Answer {
                                score: scores[idx],
                                start: start_pos,
                                end: end_pos,
                                answer,
                            });
                        }
                    }
                    feature_id_start = max_feature_id;
                    let example_answers = example_top_k_answers_map
                        .entry(example_id)
                        .or_insert_with(Vec::new);
                    example_answers.extend(answers);
                }
            });
            start = end;
        }
        let mut all_answers = vec![];
        for example_id in 0..qa_inputs.len() {
            if let Some(answers) = example_top_k_answers_map.get_mut(&example_id) {
                remove_duplicates(answers).sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
                all_answers.push(answers[..min(answers.len(), top_k as usize)].to_vec());
            } else {
                all_answers.push(vec![]);
            }
        }
        all_answers
    }

    fn decode(&self, start: &Tensor, end: &Tensor, top_k: i64) -> (Vec<i64>, Vec<i64>, Vec<f64>) {
        let outer = start.unsqueeze(-1).matmul(&end.unsqueeze(0));
        let start_dim = start.size()[0];
        let end_dim = end.size()[0];
        let candidates = outer
            .triu(0)
            .tril(self.max_answer_len as i64 - 1)
            .flatten(0, -1);
        let idx_sort = if top_k == 1 {
            candidates.argmax(0, true)
        } else if candidates.size()[0] < top_k {
            candidates.argsort(0, true)
        } else {
            candidates.argsort(0, true).slice(0, 0, top_k, 1)
        };
        let mut start: Vec<i64> = vec![];
        let mut end: Vec<i64> = vec![];
        let mut scores: Vec<f64> = vec![];
        for flat_index_position in 0..idx_sort.size()[0] {
            let flat_index = idx_sort.int64_value(&[flat_index_position]);
            scores.push(candidates.double_value(&[flat_index]));
            start.push(flat_index / start_dim);
            end.push(flat_index % end_dim);
        }
        (start, end, scores)
    }

    fn generate_features(
        &self,
        qa_example: &QaInput,
        max_seq_length: usize,
        doc_stride: usize,
        max_query_length: usize,
        example_index: i64,
    ) -> Vec<QaFeature> {
        let mut encoded_query = self.tokenizer.tokenize_with_offsets(&qa_example.question);
        encoded_query.tokens.truncate(max_query_length);
        encoded_query.offsets.truncate(max_query_length);
        encoded_query.reference_offsets.truncate(max_query_length);
        encoded_query.masks.truncate(max_query_length);
        let encoded_query = TokenIdsWithOffsets {
            ids: self.tokenizer.convert_tokens_to_ids(&encoded_query.tokens),
            offsets: encoded_query.offsets,
            reference_offsets: encoded_query.reference_offsets,
            masks: encoded_query.masks,
        };

        let sequence_pair_added_tokens = self
            .tokenizer
            .build_input_with_special_tokens(
                TokenIdsWithOffsets {
                    ids: vec![],
                    offsets: vec![],
                    reference_offsets: vec![],
                    masks: vec![],
                },
                Some(TokenIdsWithOffsets {
                    ids: vec![],
                    offsets: vec![],
                    reference_offsets: vec![],
                    masks: vec![],
                }),
            )
            .token_ids
            .len();

        let mut spans: Vec<QaFeature> = vec![];

        let tokenized_context = self.tokenizer.tokenize_with_offsets(&qa_example.context);
        let encoded_context = TokenIdsWithOffsets {
            ids: self
                .tokenizer
                .convert_tokens_to_ids(&tokenized_context.tokens),
            offsets: tokenized_context.offsets,
            reference_offsets: tokenized_context.reference_offsets,
            masks: tokenized_context.masks,
        };
        let max_context_length =
            max_seq_length - sequence_pair_added_tokens - encoded_query.ids.len();

        let mut start_token = 0_usize;
        while (spans.len() * doc_stride) < encoded_context.ids.len() {
            let end_token = min(start_token + max_context_length, encoded_context.ids.len());
            let sub_encoded_context = TokenIdsWithOffsets {
                ids: encoded_context.ids[start_token..end_token].to_vec(),
                offsets: encoded_context.offsets[start_token..end_token].to_vec(),
                reference_offsets: encoded_context.reference_offsets[start_token..end_token]
                    .to_vec(),
                masks: encoded_context.masks[start_token..end_token].to_vec(),
            };

            let encoded_span = self
                .tokenizer
                .build_input_with_special_tokens(encoded_query.clone(), Some(sub_encoded_context));
            let p_mask = self.get_mask(&encoded_span);
            let qa_feature = QaFeature {
                input_ids: encoded_span.token_ids,
                offsets: encoded_span.token_offsets,
                p_mask,
                example_index,
            };
            spans.push(qa_feature);
            if end_token == encoded_context.ids.len() {
                break;
            }
            start_token = end_token - doc_stride;
        }
        spans
    }

    fn pad_features(&self, features: &mut [QaFeature]) -> (Tensor, Tensor) {
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

        for feature in features.iter_mut() {
            feature.offsets.resize(max_len, None);
            feature.p_mask.resize(max_len, 1);
            feature.input_ids.resize(max_len, self.pad_idx);
        }

        let padded_input_ids = features
            .iter_mut()
            .map(|input| Tensor::from_slice(input.input_ids.as_slice()))
            .collect::<Vec<_>>();

        let input_ids = Tensor::stack(&padded_input_ids, 0).to(self.var_store.device());
        let attention_masks = Tensor::stack(&attention_masks, 0).to(self.var_store.device());
        (input_ids, attention_masks)
    }

    fn get_mask(&self, encoded_span: &TokenizedInput) -> Vec<i8> {
        let sep_indices: Vec<usize> = encoded_span
            .token_ids
            .iter()
            .enumerate()
            .filter(|(_, &value)| value == self.sep_idx)
            .map(|(position, _)| position)
            .collect();

        let mut p_mask: Vec<i8> = encoded_span
            .segment_ids
            .iter()
            .map(|v| min(v, &1i8))
            .map(|&v| 1i8 - v)
            .collect();
        for sep_position in sep_indices {
            p_mask[sep_position] = 1;
        }
        p_mask
    }
}

pub fn squad_processor(file_path: PathBuf) -> Vec<QaInput> {
    let file = fs::File::open(file_path).expect("unable to open file");
    let json: serde_json::Value =
        serde_json::from_reader(file).expect("JSON not properly formatted");
    let data = json
        .get("data")
        .expect("SQuAD file does not contain data field")
        .as_array()
        .expect("Data array not properly formatted");

    let mut qa_inputs: Vec<QaInput> = Vec::with_capacity(data.len());
    for qa_input in data.iter() {
        let qa_input = qa_input.as_object().unwrap();
        let paragraphs = qa_input.get("paragraphs").unwrap().as_array().unwrap();
        for paragraph in paragraphs.iter() {
            let paragraph = paragraph.as_object().unwrap();
            let context = paragraph.get("context").unwrap().as_str().unwrap();
            let qas = paragraph.get("qas").unwrap().as_array().unwrap();
            for qa in qas.iter() {
                let question = qa
                    .as_object()
                    .unwrap()
                    .get("question")
                    .unwrap()
                    .as_str()
                    .unwrap();
                qa_inputs.push(QaInput {
                    question: question.to_owned(),
                    context: context.to_owned(),
                });
            }
        }
    }
    qa_inputs
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    #[ignore] // no need to run, compilation is enough to verify it is Send
    fn test() {
        let config = QuestionAnsweringConfig::default();
        let _: Box<dyn Send> = Box::new(QuestionAnsweringModel::new(config));
    }
}
