// Copyright (c) 2020  The Google AI Language Team Authors, The HuggingFace Inc. team and github/lonePatient
// Copyright 2020 Guillaume Becquin
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::common::activations::{Activation, TensorFunction};
use crate::mobilebert::embeddings::MobileBertEmbeddings;
use crate::mobilebert::encoder::{MobileBertEncoder, MobileBertPooler};
use crate::{Config, RustBertError};
use serde::{Deserialize, Serialize};
use std::borrow::Borrow;
use std::collections::HashMap;
use tch::nn::{Init, LayerNormConfig, Module};
use tch::{nn, Kind, Tensor};

/// # MobileBERT Pretrained model weight files
pub struct MobileBertModelResources;

/// # MobileBERT Pretrained model config files
pub struct MobileBertConfigResources;

/// # MobileBERT Pretrained model vocab files
pub struct MobileBertVocabResources;

impl MobileBertModelResources {
    /// Shared under Apache 2.0 license by the Google team at https://huggingface.co/google/mobilebert-uncased. Modified with conversion to C-array format.
    pub const MOBILEBERT_UNCASED: (&'static str, &'static str) = (
        "mobilebert-uncased/model",
        "https://huggingface.co/google/mobilebert-uncased/resolve/main/rust_model.ot",
    );
}

impl MobileBertConfigResources {
    /// Shared under Apache 2.0 license by the Google team at https://huggingface.co/google/mobilebert-uncased. Modified with conversion to C-array format.
    pub const MOBILEBERT_UNCASED: (&'static str, &'static str) = (
        "mobilebert-uncased/config",
        "https://huggingface.co/google/mobilebert-uncased/resolve/main/config.json",
    );
}

impl MobileBertVocabResources {
    /// Shared under Apache 2.0 license by the Google team at https://huggingface.co/google/mobilebert-uncased. Modified with conversion to C-array format.
    pub const MOBILEBERT_UNCASED: (&'static str, &'static str) = (
        "mobilebert-uncased/vocab",
        "https://huggingface.co/google/mobilebert-uncased/resolve/main/vocab.txt",
    );
}

#[allow(non_camel_case_types)]
#[derive(Clone, Debug, Serialize, Deserialize, Copy)]
pub enum NormalizationType {
    layer_norm,
    no_norm,
}

#[derive(Debug)]
pub struct NoNorm {
    weight: Tensor,
    bias: Tensor,
}

impl NoNorm {
    pub fn new<'p, P>(p: P, hidden_size: i64) -> NoNorm
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let weight = p.var("weight", &[hidden_size], Init::Const(1.0));
        let bias = p.var("bias", &[hidden_size], Init::Const(0.0));
        NoNorm { weight, bias }
    }
}

impl Module for NoNorm {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs * &self.weight + &self.bias
    }
}

pub enum NormalizationLayer {
    LayerNorm(nn::LayerNorm),
    NoNorm(NoNorm),
}

impl NormalizationLayer {
    pub fn new<'p, P>(
        p: P,
        normalization_type: NormalizationType,
        hidden_size: i64,
        eps: Option<f64>,
    ) -> NormalizationLayer
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        match normalization_type {
            NormalizationType::layer_norm => {
                let layer_norm_config = LayerNormConfig {
                    eps: eps.unwrap_or(1e-12),
                    ..Default::default()
                };
                let layer_norm = nn::layer_norm(p, vec![hidden_size], layer_norm_config);
                NormalizationLayer::LayerNorm(layer_norm)
            }
            NormalizationType::no_norm => {
                let layer_norm = NoNorm::new(p, hidden_size);
                NormalizationLayer::NoNorm(layer_norm)
            }
        }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        match self {
            NormalizationLayer::LayerNorm(ref layer_norm) => input.apply(layer_norm),
            NormalizationLayer::NoNorm(ref layer_norm) => input.apply(layer_norm),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
/// # MobileBERT model configuration
/// Defines the MobileBERT model architecture (e.g. number of layers, hidden layer size, label mapping...)
pub struct MobileBertConfig {
    pub hidden_act: Activation,
    pub attention_probs_dropout_prob: f64,
    pub hidden_dropout_prob: f64,
    pub hidden_size: i64,
    pub initializer_range: f64,
    pub intermediate_size: i64,
    pub max_position_embeddings: i64,
    pub num_attention_heads: i64,
    pub num_hidden_layers: i64,
    pub type_vocab_size: i64,
    pub vocab_size: i64,
    pub embedding_size: i64,
    pub layer_norm_eps: Option<f64>,
    pub pad_token_idx: Option<i64>,
    pub trigram_input: Option<bool>,
    pub use_bottleneck: Option<bool>,
    pub use_bottleneck_attention: Option<bool>,
    pub intra_bottleneck_size: Option<i64>,
    pub key_query_shared_bottleneck: Option<bool>,
    pub num_feedforward_networks: Option<i64>,
    pub normalization_type: Option<NormalizationType>,
    pub output_attentions: Option<bool>,
    pub output_hidden_states: Option<bool>,
    pub classifier_activation: Option<bool>,
    pub is_decoder: Option<bool>,
    pub id2label: Option<HashMap<i64, String>>,
    pub label2id: Option<HashMap<String, i64>>,
}

impl Config<MobileBertConfig> for MobileBertConfig {}

pub struct MobileBertPredictionHeadTransform {
    dense: nn::Linear,
    activation_function: TensorFunction,
    layer_norm: NormalizationLayer,
}

impl MobileBertPredictionHeadTransform {
    pub fn new<'p, P>(p: P, config: &MobileBertConfig) -> MobileBertPredictionHeadTransform
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let dense = nn::linear(
            p / "dense",
            config.hidden_size,
            config.hidden_size,
            Default::default(),
        );
        let activation_function = config.hidden_act.get_function();
        let layer_norm = NormalizationLayer::new(
            p / "LayerNorm",
            config
                .normalization_type
                .unwrap_or(NormalizationType::no_norm),
            config.hidden_size,
            config.layer_norm_eps,
        );
        MobileBertPredictionHeadTransform {
            dense,
            activation_function,
            layer_norm,
        }
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Tensor {
        let hidden_states = hidden_states.apply(&self.dense);
        let hidden_states = self.activation_function.get_fn()(&hidden_states);
        self.layer_norm.forward(&hidden_states)
    }
}

pub struct MobileBertLMPredictionHead {
    transform: MobileBertPredictionHeadTransform,
    dense_weight: Tensor,
    bias: Tensor,
}

impl MobileBertLMPredictionHead {
    pub fn new<'p, P>(p: P, config: &MobileBertConfig) -> MobileBertLMPredictionHead
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let transform = MobileBertPredictionHeadTransform::new(p / "transform", config);

        let dense_p = p / "dense";
        let dense_weight = dense_p.var(
            "weight",
            &[
                config.hidden_size - config.embedding_size,
                config.vocab_size,
            ],
            Init::KaimingUniform,
        );
        let bias = p.var("bias", &[config.vocab_size], Init::Const(0.0));

        MobileBertLMPredictionHead {
            transform,
            dense_weight,
            bias,
        }
    }

    pub fn forward(&self, hidden_states: &Tensor, embeddings: &Tensor) -> Tensor {
        let hidden_states = self.transform.forward(hidden_states);
        let hidden_states = hidden_states.matmul(&Tensor::cat(
            &[&embeddings.transpose(0, 1), &self.dense_weight],
            0,
        ));
        hidden_states + &self.bias
    }
}

pub struct MobileBertOnlyMLMHead {
    predictions: MobileBertLMPredictionHead,
}

impl MobileBertOnlyMLMHead {
    pub fn new<'p, P>(p: P, config: &MobileBertConfig) -> MobileBertOnlyMLMHead
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let predictions = MobileBertLMPredictionHead::new(p / "predictions", config);
        MobileBertOnlyMLMHead { predictions }
    }

    pub fn forward(&self, hidden_states: &Tensor, embeddings: &Tensor) -> Tensor {
        self.predictions.forward(hidden_states, embeddings)
    }
}

pub struct MobileBertModel {
    embeddings: MobileBertEmbeddings,
    encoder: MobileBertEncoder,
    pooler: Option<MobileBertPooler>,
    position_ids: Tensor,
}

impl MobileBertModel {
    pub fn new<'p, P>(p: P, config: &MobileBertConfig, add_pooling_layer: bool) -> MobileBertModel
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let embeddings = MobileBertEmbeddings::new(p / "embeddings", config);
        let encoder = MobileBertEncoder::new(p / "encoder", config);
        let pooler = if add_pooling_layer {
            Some(MobileBertPooler::new(p / "pooler", config))
        } else {
            None
        };
        let position_ids =
            Tensor::arange(config.max_position_embeddings, (Kind::Int64, p.device()))
                .expand(&[1, -1], true);
        MobileBertModel {
            embeddings,
            encoder,
            pooler,
            position_ids,
        }
    }

    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        input_embeds: Option<Tensor>,
        attention_mask: Option<&Tensor>,
        train: bool,
    ) -> Result<MobileBertOutput, RustBertError> {
        let (input_shape, device) = match input_ids {
            Some(input_value) => match &input_embeds {
                Some(_) => {
                    return Err(RustBertError::ValueError(
                        "Only one of input ids or input embeddings may be set".into(),
                    ));
                }
                None => (input_value.size(), input_value.device()),
            },
            None => match &input_embeds {
                Some(embeds) => (vec![embeds.size()[0], embeds.size()[1]], embeds.device()),
                None => {
                    return Err(RustBertError::ValueError(
                        "At least one of input ids or input embeddings must be set".into(),
                    ));
                }
            },
        };

        let calc_attention_mask = if attention_mask.is_none() {
            Some(Tensor::ones(input_shape.as_slice(), (Kind::Int64, device)))
        } else {
            None
        };

        let calc_token_type_ids = if token_type_ids.is_none() {
            Some(Tensor::zeros(input_shape.as_slice(), (Kind::Int64, device)))
        } else {
            None
        };

        let calc_position_ids = if position_ids.is_none() {
            Some(self.position_ids.slice(1, 0, input_shape[1], 1))
        } else {
            None
        };

        let position_ids = position_ids.unwrap_or(calc_position_ids.as_ref().unwrap());

        let attention_mask = attention_mask.unwrap_or(calc_attention_mask.as_ref().unwrap());
        let attention_mask = match attention_mask.dim() {
            3 => attention_mask.unsqueeze(1),
            2 => attention_mask.unsqueeze(1).unsqueeze(1),
            _ => {
                return Err(RustBertError::ValueError(
                    "Invalid attention mask dimension, must be 2 or 3".into(),
                ));
            }
        };
        let attention_mask: Tensor = (attention_mask.ones_like() - attention_mask) * -10000.0;

        let token_type_ids = token_type_ids.unwrap_or(calc_token_type_ids.as_ref().unwrap());

        let embedding_output = self.embeddings.forward_t(
            input_ids,
            token_type_ids.into(),
            position_ids,
            input_embeds,
            train,
        )?;

        let encoder_output =
            self.encoder
                .forward_t(&embedding_output, Some(&attention_mask), train);

        let pooled_output = if let Some(pooler) = &self.pooler {
            Some(pooler.forward(&encoder_output.hidden_state))
        } else {
            None
        };

        Ok(MobileBertOutput {
            hidden_state: encoder_output.hidden_state,
            pooled_output,
            all_hidden_states: encoder_output.all_hidden_states,
            all_attentions: encoder_output.all_attentions,
        })
    }

    fn get_embeddings(&self) -> &Tensor {
        &self.embeddings.word_embeddings.ws
    }
}

pub struct MobileBertForMaskedLM {
    mobilebert: MobileBertModel,
    classifier: MobileBertOnlyMLMHead,
}

impl MobileBertForMaskedLM {
    pub fn new<'p, P>(p: P, config: &MobileBertConfig) -> MobileBertForMaskedLM
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let mobilebert = MobileBertModel::new(p / "mobilebert", config, false);
        let classifier = MobileBertOnlyMLMHead::new(p / "cls", config);
        MobileBertForMaskedLM {
            mobilebert,
            classifier,
        }
    }

    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        input_embeds: Option<Tensor>,
        attention_mask: Option<&Tensor>,
        train: bool,
    ) -> Result<MobileBertMaskedLMOutput, RustBertError> {
        let mobilebert_output = self.mobilebert.forward_t(
            input_ids,
            token_type_ids,
            position_ids,
            input_embeds,
            attention_mask,
            train,
        )?;

        let logits = self.classifier.forward(
            &mobilebert_output.hidden_state,
            self.mobilebert.get_embeddings(),
        );

        Ok(MobileBertMaskedLMOutput {
            logits,
            all_hidden_states: mobilebert_output.all_hidden_states,
            all_attentions: mobilebert_output.all_attentions,
        })
    }
}

/// Container for the MobileBert output.
pub struct MobileBertOutput {
    /// Last hidden states from the model
    pub hidden_state: Tensor,
    /// Pooled output
    pub pooled_output: Option<Tensor>,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all intermediate layers
    pub all_attentions: Option<Vec<Tensor>>,
}

/// Container for the MobileBert masked LM model output.
pub struct MobileBertMaskedLMOutput {
    /// Logits for the vocabulary items at each sequence position
    pub logits: Tensor,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all intermediate layers
    pub all_attentions: Option<Vec<Tensor>>,
}
