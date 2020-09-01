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

use crate::albert::AlbertForSequenceClassification;
use crate::bart::{
    BartConfigResources, BartForSequenceClassification, BartMergesResources, BartModelResources,
    BartVocabResources,
};
use crate::bert::BertForSequenceClassification;
use crate::distilbert::DistilBertModelClassifier;
use crate::pipelines::common::{ConfigOption, ModelType};
use crate::resources::{RemoteResource, Resource};
use crate::roberta::RobertaForSequenceClassification;
use std::borrow::Borrow;
use tch::{nn, Device, Tensor};

/// # Configuration for ZeroShotClassificationModel
/// Contains information regarding the model to load and device to place the model on.
pub struct ZeroShotClassificationConfig {
    /// Model type
    pub model_type: ModelType,
    /// Model weights resource (default: pretrained BERT model on CoNLL)
    pub model_resource: Resource,
    /// Config resource (default: pretrained BERT model on CoNLL)
    pub config_resource: Resource,
    /// Vocab resource (default: pretrained BERT model on CoNLL)
    pub vocab_resource: Resource,
    /// Merges resource (default: None)
    pub merges_resource: Option<Resource>,
    /// Automatically lower case all input upon tokenization (assumes a lower-cased model)
    pub lower_case: bool,
    /// Flag indicating if the tokenizer should strip accents (normalization). Only used for BERT / ALBERT models
    pub strip_accents: Option<bool>,
    /// Flag indicating if the tokenizer should add a white space before each tokenized input (needed for some Roberta models)
    pub add_prefix_space: Option<bool>,
    /// Device to place the model on (default: CUDA/GPU when available)
    pub device: Device,
}

impl ZeroShotClassificationConfig {
    /// Instantiate a new zero shot classification configuration of the supplied type.
    ///
    /// # Arguments
    ///
    /// * `model_type` - `ModelType` indicating the model type to load (must match with the actual data to be loaded!)
    /// * model - The `Resource` pointing to the model to load (e.g.  model.ot)
    /// * config - The `Resource' pointing to the model configuration to load (e.g. config.json)
    /// * vocab - The `Resource' pointing to the tokenizer's vocabulary to load (e.g.  vocab.txt/vocab.json)
    /// * vocab - An optional `Resource` tuple (`Option<Resource>`) pointing to the tokenizer's merge file to load (e.g.  merges.txt), needed only for Roberta.
    /// * lower_case - A `bool' indicating whether the tokeniser should lower case all input (in case of a lower-cased model)
    pub fn new(
        model_type: ModelType,
        model_resource: Resource,
        config_resource: Resource,
        vocab_resource: Resource,
        merges_resource: Option<Resource>,
        lower_case: bool,
        strip_accents: impl Into<Option<bool>>,
        add_prefix_space: impl Into<Option<bool>>,
    ) -> ZeroShotClassificationConfig {
        ZeroShotClassificationConfig {
            model_type,
            model_resource,
            config_resource,
            vocab_resource,
            merges_resource,
            lower_case,
            strip_accents: strip_accents.into(),
            add_prefix_space: add_prefix_space.into(),
            device: Device::cuda_if_available(),
        }
    }
}

impl Default for ZeroShotClassificationConfig {
    /// Provides a defaultSST-2 sentiment analysis model (English)
    fn default() -> ZeroShotClassificationConfig {
        ZeroShotClassificationConfig {
            model_type: ModelType::DistilBert,
            model_resource: Resource::Remote(RemoteResource::from_pretrained(
                BartModelResources::BART_MNLI,
            )),
            config_resource: Resource::Remote(RemoteResource::from_pretrained(
                BartConfigResources::BART_MNLI,
            )),
            vocab_resource: Resource::Remote(RemoteResource::from_pretrained(
                BartVocabResources::BART_MNLI,
            )),
            merges_resource: Some(Resource::Remote(RemoteResource::from_pretrained(
                BartMergesResources::BART_MNLI,
            ))),
            lower_case: false,
            strip_accents: None,
            add_prefix_space: None,
            device: Device::cuda_if_available(),
        }
    }
}

/// # Abstraction that holds one particular zero shot classification model, for any of the supported models
/// The models are using a classification architecture that should be trained on Natural Language Inference.
/// The models should output a Tensor of size > 2 in the label dimension, with the first logit corresponding
/// to contradiction and the last logit corresponding to entailment.
pub enum ZeroShotClassificationOption {
    /// Bart for Sequence Classification
    Bart(BartForSequenceClassification),
    /// Bert for Sequence Classification
    Bert(BertForSequenceClassification),
    /// DistilBert for Sequence Classification
    DistilBert(DistilBertModelClassifier),
    /// Roberta for Sequence Classification
    Roberta(RobertaForSequenceClassification),
    /// XLMRoberta for Sequence Classification
    XLMRoberta(RobertaForSequenceClassification),
    /// Albert for Sequence Classification
    Albert(AlbertForSequenceClassification),
}

impl ZeroShotClassificationOption {
    /// Instantiate a new zero shot classification model of the supplied type.
    ///
    /// # Arguments
    ///
    /// * `model_type` - `ModelType` indicating the model type to load (must match with the actual data to be loaded)
    /// * `p` - `tch::nn::Path` path to the model file to load (e.g. model.ot)
    /// * `config` - A configuration (the model type of the configuration must be compatible with the value for
    /// `model_type`)
    pub fn new<'p, P>(model_type: ModelType, p: P, config: &ConfigOption) -> Self
    where
        P: Borrow<nn::Path<'p>>,
    {
        match model_type {
            ModelType::Bart => {
                if let ConfigOption::Bart(config) = config {
                    ZeroShotClassificationOption::Bart(BartForSequenceClassification::new(
                        p, config,
                    ))
                } else {
                    panic!("You can only supply a BartConfig for Bart!");
                }
            }
            ModelType::Bert => {
                if let ConfigOption::Bert(config) = config {
                    ZeroShotClassificationOption::Bert(BertForSequenceClassification::new(
                        p, config,
                    ))
                } else {
                    panic!("You can only supply a BertConfig for Bert!");
                }
            }
            ModelType::DistilBert => {
                if let ConfigOption::DistilBert(config) = config {
                    ZeroShotClassificationOption::DistilBert(DistilBertModelClassifier::new(
                        p, config,
                    ))
                } else {
                    panic!("You can only supply a DistilBertConfig for DistilBert!");
                }
            }
            ModelType::Roberta => {
                if let ConfigOption::Bert(config) = config {
                    ZeroShotClassificationOption::Roberta(RobertaForSequenceClassification::new(
                        p, config,
                    ))
                } else {
                    panic!("You can only supply a BertConfig for Roberta!");
                }
            }
            ModelType::XLMRoberta => {
                if let ConfigOption::Bert(config) = config {
                    ZeroShotClassificationOption::XLMRoberta(RobertaForSequenceClassification::new(
                        p, config,
                    ))
                } else {
                    panic!("You can only supply a BertConfig for Roberta!");
                }
            }
            ModelType::Albert => {
                if let ConfigOption::Albert(config) = config {
                    ZeroShotClassificationOption::Albert(AlbertForSequenceClassification::new(
                        p, config,
                    ))
                } else {
                    panic!("You can only supply an AlbertConfig for Albert!");
                }
            }
            ModelType::Electra => {
                panic!("SequenceClassification not implemented for Electra!");
            }
            ModelType::Marian => {
                panic!("SequenceClassification not implemented for Marian!");
            }
            ModelType::T5 => {
                panic!("SequenceClassification not implemented for T5!");
            }
        }
    }

    /// Returns the `ModelType` for this SequenceClassificationOption
    pub fn model_type(&self) -> ModelType {
        match *self {
            Self::Bart(_) => ModelType::Bart,
            Self::Bert(_) => ModelType::Bert,
            Self::Roberta(_) => ModelType::Roberta,
            Self::XLMRoberta(_) => ModelType::Roberta,
            Self::DistilBert(_) => ModelType::DistilBert,
            Self::Albert(_) => ModelType::Albert,
        }
    }

    /// Interface method to forward_t() of the particular models.
    pub fn forward_t(
        &self,
        input_ids: Option<Tensor>,
        mask: Option<Tensor>,
        token_type_ids: Option<Tensor>,
        position_ids: Option<Tensor>,
        input_embeds: Option<Tensor>,
        train: bool,
    ) -> Tensor {
        match *self {
            Self::Bart(ref model) => {
                model
                    .forward_t(
                        &input_ids.expect("`input_ids` must be provided for BART models"),
                        mask.as_ref(),
                        None,
                        None,
                        None,
                        train,
                    )
                    .0
            }
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
                    .0
            }
            Self::DistilBert(ref model) => {
                model
                    .forward_t(input_ids, mask, input_embeds, train)
                    .expect("Error in distilbert forward_t")
                    .0
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
                    .0
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
                    .0
            }
        }
    }
}
