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
use crate::common::dropout::Dropout;
use crate::common::embeddings::get_shape_and_device_from_ids_embeddings_pair;
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
    /// Shared under Apache 2.0 license by the Google team at <https://huggingface.co/google/mobilebert-uncased>. Modified with conversion to C-array format.
    pub const MOBILEBERT_UNCASED: (&'static str, &'static str) = (
        "mobilebert-uncased/model",
        "https://huggingface.co/google/mobilebert-uncased/resolve/main/rust_model.ot",
    );
    /// Shared under MIT license at <https://huggingface.co/mrm8488/mobilebert-finetuned-pos>. Modified with conversion to C-array format.
    pub const MOBILEBERT_ENGLISH_POS: (&'static str, &'static str) = (
        "mobilebert-finetuned-pos/model",
        "https://huggingface.co/mrm8488/mobilebert-finetuned-pos/resolve/main/rust_model.ot",
    );
}

impl MobileBertConfigResources {
    /// Shared under Apache 2.0 license by the Google team at <https://huggingface.co/google/mobilebert-uncased>. Modified with conversion to C-array format.
    pub const MOBILEBERT_UNCASED: (&'static str, &'static str) = (
        "mobilebert-uncased/config",
        "https://huggingface.co/google/mobilebert-uncased/resolve/main/config.json",
    );
    /// Shared under MIT license at <https://huggingface.co/mrm8488/mobilebert-finetuned-pos>. Modified with conversion to C-array format.
    pub const MOBILEBERT_ENGLISH_POS: (&'static str, &'static str) = (
        "mobilebert-finetuned-pos/config",
        "https://huggingface.co/mrm8488/mobilebert-finetuned-pos/resolve/main/config.json",
    );
}

impl MobileBertVocabResources {
    /// Shared under Apache 2.0 license by the Google team at <https://huggingface.co/google/mobilebert-uncased>. Modified with conversion to C-array format.
    pub const MOBILEBERT_UNCASED: (&'static str, &'static str) = (
        "mobilebert-uncased/vocab",
        "https://huggingface.co/google/mobilebert-uncased/resolve/main/vocab.txt",
    );
    /// Shared under MIT license at <https://huggingface.co/mrm8488/mobilebert-finetuned-pos>. Modified with conversion to C-array format.
    pub const MOBILEBERT_ENGLISH_POS: (&'static str, &'static str) = (
        "mobilebert-finetuned-pos/vocab",
        "https://huggingface.co/mrm8488/mobilebert-finetuned-pos/resolve/main/vocab.txt",
    );
}

#[allow(non_camel_case_types)]
#[derive(Clone, Debug, Serialize, Deserialize, Copy)]
/// # Normalization type to use for the MobileBERT model.
/// `no_norm` uses a matrix multiplication with a set of learned weights, while `layer_norm` uses a
/// build-in layer normalization module.
pub enum NormalizationType {
    layer_norm,
    no_norm,
}

#[derive(Debug)]
/// # No-normalization option for MobileBERT
/// Basic module performing a linear multiplication using trained coefficients and bias
pub struct NoNorm {
    weight: Tensor,
    bias: Tensor,
}

impl NoNorm {
    /// Creates a new `NoNorm` layer of given hidden size.
    ///
    /// # Arguments:
    ///
    /// * hidden_size - input tensor's hidden size
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::mobilebert::NoNorm;
    /// use tch::{nn, Device};
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let hidden_size = 512;
    /// let no_norm = NoNorm::new(&p.root(), hidden_size);
    /// ```
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

impl Config for MobileBertConfig {}

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
            NormalizationType::layer_norm,
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

/// # MobileBertModel Base model
/// Base architecture for MobileBERT models. Task-specific models will be built from this common base model
/// It is made of the following blocks:
/// - `embeddings`: Word, token type and position embeddings
/// - `encoder`: `MobileBertEncoder` made of a stack of `MobileBertLayer`
/// - `pooler`: Optional `MobileBertPooler` taking the first sequence element hidden state for sequence-level tasks
/// - `position_ids` preset position ids tensor used in case they are not provided by the user
pub struct MobileBertModel {
    embeddings: MobileBertEmbeddings,
    encoder: MobileBertEncoder,
    pooler: Option<MobileBertPooler>,
    position_ids: Tensor,
}

impl MobileBertModel {
    /// Build a new `MobileBertModel`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the MobileBERT model
    /// * `config` - `MobileBertConfig` object defining the model architecture and decoder status
    /// * `add_pooling_layer` - boolean flag indicating if a pooling layer should be added after the encoder
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::mobilebert::{MobileBertConfig, MobileBertModel};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = MobileBertConfig::from_file(config_path);
    /// let add_pooling_layer = true;
    /// let mobilebert = MobileBertModel::new(&p.root() / "mobilebert", &config, add_pooling_layer);
    /// ```
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

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see `input_embeds`)
    /// * `token_type_ids` - Optional segment id of shape (*batch size*, *sequence_length*). Convention is value of 0 for the first sentence (incl. *SEP*) and 1 for the second sentence. If None set to 0.
    /// * `position_ids` - Optional position ids of shape (*batch size*, *sequence_length*). If None, will be incremented from 0.
    /// * `input_embeds` - Optional pre-computed input embeddings of shape (*batch size*, *sequence_length*, *hidden_size*). If None, input ids must be provided (see `input_ids`)
    /// * `attention_mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `MobileBertOutput` containing:
    ///   - `hidden_state` - `Tensor` of shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `pooled_output` - Optional `Tensor` of shape (*batch size*, *hidden_size*) if the model was created with an optional pooling layer
    ///   - `all_hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `all_attentions` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use tch::{nn, Device, Tensor, no_grad};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::Int64;
    /// use rust_bert::mobilebert::{MobileBertConfig, MobileBertModel};
    /// # let config_path = Path::new("path/to/config.json");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = MobileBertConfig::from_file(config_path);
    /// let add_pooling_layer = true;
    /// let model = MobileBertModel::new(&vs.root(), &config, add_pooling_layer);
    /// let (batch_size, sequence_length) = (64, 128);
    /// let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    /// let attention_mask = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    /// let token_type_ids = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    /// let position_ids = Tensor::arange(sequence_length, (Int64, device))
    ///     .expand(&[batch_size, sequence_length], true);
    ///
    /// let model_output = no_grad(|| {
    ///     model
    ///         .forward_t(
    ///             Some(&input_tensor),
    ///             Some(&token_type_ids),
    ///             Some(&position_ids),
    ///             None,
    ///             Some(&attention_mask),
    ///             false,
    ///         )
    ///         .unwrap()
    /// });
    /// ```
    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        train: bool,
    ) -> Result<MobileBertOutput, RustBertError> {
        let (input_shape, device) =
            get_shape_and_device_from_ids_embeddings_pair(input_ids, input_embeds)?;

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

        let position_ids = position_ids.unwrap_or_else(|| calc_position_ids.as_ref().unwrap());

        let attention_mask =
            attention_mask.unwrap_or_else(|| calc_attention_mask.as_ref().unwrap());
        let attention_mask = match attention_mask.dim() {
            3 => attention_mask.unsqueeze(1),
            2 => attention_mask.unsqueeze(1).unsqueeze(1),
            _ => {
                return Err(RustBertError::ValueError(
                    "Invalid attention mask dimension, must be 2 or 3".into(),
                ));
            }
        };

        let token_type_ids =
            token_type_ids.unwrap_or_else(|| calc_token_type_ids.as_ref().unwrap());

        let embedding_output = self.embeddings.forward_t(
            input_ids,
            token_type_ids,
            position_ids,
            input_embeds,
            train,
        )?;

        let attention_mask: Tensor = ((attention_mask.ones_like() - attention_mask) * -10000.0)
            .to_kind(embedding_output.kind());

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

/// # MobileBERT for masked language model
/// Base MobileBERT model with a masked language model head to predict missing tokens, for example `"Looks like one [MASK] is missing" -> "person"`
/// It is made of the following blocks:
/// - `mobilebert`: Base MobileBertModel
/// - `classifier`: MobileBERT LM prediction head
pub struct MobileBertForMaskedLM {
    mobilebert: MobileBertModel,
    classifier: MobileBertOnlyMLMHead,
}

impl MobileBertForMaskedLM {
    /// Build a new `MobileBertForMaskedLM`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the MobileBERT model
    /// * `config` - `MobileBertConfig` object defining the model architecture and decoder status
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::mobilebert::{MobileBertConfig, MobileBertForMaskedLM};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = MobileBertConfig::from_file(config_path);
    /// let mobilebert = MobileBertForMaskedLM::new(&p.root(), &config);
    /// ```
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

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see `input_embeds`)
    /// * `token_type_ids` - Optional segment id of shape (*batch size*, *sequence_length*). Convention is value of 0 for the first sentence (incl. *SEP*) and 1 for the second sentence. If None set to 0.
    /// * `position_ids` - Optional position ids of shape (*batch size*, *sequence_length*). If None, will be incremented from 0.
    /// * `input_embeds` - Optional pre-computed input embeddings of shape (*batch size*, *sequence_length*, *hidden_size*). If None, input ids must be provided (see `input_ids`)
    /// * `attention_mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `MobileBertMaskedLMOutput` containing:
    ///   - `logits` - `Tensor` of shape (*batch size*, *sequence_length*, *vocab_size*)
    ///   - `all_hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `all_attentions` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use tch::{nn, Device, Tensor, no_grad};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::Int64;
    /// use rust_bert::mobilebert::{MobileBertConfig, MobileBertForMaskedLM};
    /// # let config_path = Path::new("path/to/config.json");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = MobileBertConfig::from_file(config_path);
    /// let model = MobileBertForMaskedLM::new(&vs.root(), &config);
    /// let (batch_size, sequence_length) = (64, 128);
    /// let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    /// let attention_mask = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    /// let token_type_ids = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    /// let position_ids = Tensor::arange(sequence_length, (Int64, device))
    ///     .expand(&[batch_size, sequence_length], true);
    ///
    /// let model_output = no_grad(|| {
    ///     model
    ///         .forward_t(
    ///             Some(&input_tensor),
    ///             Some(&token_type_ids),
    ///             Some(&position_ids),
    ///             None,
    ///             Some(&attention_mask),
    ///             false,
    ///         )
    ///         .unwrap()
    /// });
    /// ```
    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
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

/// # MobileBERT for sequence classification
/// Base MobileBERT model with a classifier head to perform sentence or document-level classification
/// It is made of the following blocks:
/// - `mobilebert`: Base MobileBertModel
/// - `dropout`: Dropout layer before the last linear layer
/// - `classifier`: linear layer mapping from hidden to the number of classes to predict
pub struct MobileBertForSequenceClassification {
    mobilebert: MobileBertModel,
    dropout: Dropout,
    classifier: nn::Linear,
}

impl MobileBertForSequenceClassification {
    /// Build a new `MobileBertForSequenceClassification`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the MobileBERT model
    /// * `config` - `MobileBertConfig` object defining the model architecture and decoder status
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::mobilebert::{MobileBertConfig, MobileBertForSequenceClassification};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = MobileBertConfig::from_file(config_path);
    /// let mobilebert = MobileBertForSequenceClassification::new(&p.root(), &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &MobileBertConfig) -> MobileBertForSequenceClassification
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let mobilebert = MobileBertModel::new(p / "mobilebert", config, true);
        let dropout = Dropout::new(config.hidden_dropout_prob);
        let num_labels = config
            .id2label
            .as_ref()
            .expect("num_labels not provided in configuration")
            .len() as i64;
        let classifier = nn::linear(
            p / "classifier",
            config.hidden_size,
            num_labels,
            Default::default(),
        );
        MobileBertForSequenceClassification {
            mobilebert,
            dropout,
            classifier,
        }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see `input_embeds`)
    /// * `token_type_ids` - Optional segment id of shape (*batch size*, *sequence_length*). Convention is value of 0 for the first sentence (incl. *SEP*) and 1 for the second sentence. If None set to 0.
    /// * `position_ids` - Optional position ids of shape (*batch size*, *sequence_length*). If None, will be incremented from 0.
    /// * `input_embeds` - Optional pre-computed input embeddings of shape (*batch size*, *sequence_length*, *hidden_size*). If None, input ids must be provided (see `input_ids`)
    /// * `attention_mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `MobileBertSequenceClassificationOutput` containing:
    ///   - `logits` - `Tensor` of shape (*batch size*, *num_classes*)
    ///   - `all_hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `all_attentions` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use tch::{nn, Device, Tensor, no_grad};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::Int64;
    /// use rust_bert::mobilebert::{MobileBertConfig, MobileBertForSequenceClassification};
    /// # let config_path = Path::new("path/to/config.json");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = MobileBertConfig::from_file(config_path);
    /// let model = MobileBertForSequenceClassification::new(&vs.root(), &config);
    /// let (batch_size, sequence_length) = (64, 128);
    /// let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    /// let attention_mask = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    /// let token_type_ids = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    /// let position_ids = Tensor::arange(sequence_length, (Int64, device))
    ///     .expand(&[batch_size, sequence_length], true);
    ///
    /// let model_output = no_grad(|| {
    ///     model
    ///         .forward_t(
    ///             Some(&input_tensor),
    ///             Some(&token_type_ids),
    ///             Some(&position_ids),
    ///             None,
    ///             Some(&attention_mask),
    ///             false,
    ///         )
    ///         .unwrap()
    /// });
    /// ```
    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        train: bool,
    ) -> Result<MobileBertSequenceClassificationOutput, RustBertError> {
        let mobilebert_output = self.mobilebert.forward_t(
            input_ids,
            token_type_ids,
            position_ids,
            input_embeds,
            attention_mask,
            train,
        )?;

        let logits = mobilebert_output
            .pooled_output
            .unwrap()
            .apply_t(&self.dropout, train)
            .apply(&self.classifier);

        Ok(MobileBertSequenceClassificationOutput {
            logits,
            all_hidden_states: mobilebert_output.all_hidden_states,
            all_attentions: mobilebert_output.all_attentions,
        })
    }
}

/// # MobileBERT for question answering
/// Extractive question-answering model based on a MobileBERT language model. Identifies the segment of a context that answers a provided question.
/// Please note that a significant amount of pre- and post-processing is required to perform end-to-end question answering.
/// See the question answering pipeline (also provided in this crate) for more details.
/// It is made of the following blocks:
/// - `mobilebert`: Base MobileBertModel
/// - `qa_outputs`: Linear layer for question answering
pub struct MobileBertForQuestionAnswering {
    mobilebert: MobileBertModel,
    qa_outputs: nn::Linear,
}

impl MobileBertForQuestionAnswering {
    /// Build a new `MobileBertForQuestionAnswering`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the MobileBERT model
    /// * `config` - `MobileBertConfig` object defining the model architecture and decoder status
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::mobilebert::{MobileBertConfig, MobileBertForQuestionAnswering};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = MobileBertConfig::from_file(config_path);
    /// let mobilebert = MobileBertForQuestionAnswering::new(&p.root(), &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &MobileBertConfig) -> MobileBertForQuestionAnswering
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let mobilebert = MobileBertModel::new(p / "mobilebert", config, false);
        let qa_outputs = nn::linear(p / "qa_outputs", config.hidden_size, 2, Default::default());

        MobileBertForQuestionAnswering {
            mobilebert,
            qa_outputs,
        }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see `input_embeds`)
    /// * `token_type_ids` - Optional segment id of shape (*batch size*, *sequence_length*). Convention is value of 0 for the first sentence (incl. *SEP*) and 1 for the second sentence. If None set to 0.
    /// * `position_ids` - Optional position ids of shape (*batch size*, *sequence_length*). If None, will be incremented from 0.
    /// * `input_embeds` - Optional pre-computed input embeddings of shape (*batch size*, *sequence_length*, *hidden_size*). If None, input ids must be provided (see `input_ids`)
    /// * `attention_mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `MobileBertQuestionAnsweringOutput` containing:
    ///   - `start_logits` - `Tensor` of shape (*batch size*, *sequence_length*) containing the logits for start of the answer
    ///   - `end_logits` - `Tensor` of shape (*batch size*, *sequence_length*) containing the logits for end of the answer
    ///   - `all_hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `all_attentions` - `Option<Vec<Vec<Tensor>>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use tch::{nn, Device, Tensor, no_grad};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::Int64;
    /// use rust_bert::mobilebert::{MobileBertConfig, MobileBertForQuestionAnswering};
    /// # let config_path = Path::new("path/to/config.json");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = MobileBertConfig::from_file(config_path);
    /// let model = MobileBertForQuestionAnswering::new(&vs.root(), &config);
    /// let (batch_size, sequence_length) = (64, 128);
    /// let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    /// let attention_mask = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    /// let token_type_ids = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    /// let position_ids = Tensor::arange(sequence_length, (Int64, device))
    ///     .expand(&[batch_size, sequence_length], true);
    ///
    /// let model_output = no_grad(|| {
    ///     model
    ///         .forward_t(
    ///             Some(&input_tensor),
    ///             Some(&token_type_ids),
    ///             Some(&position_ids),
    ///             None,
    ///             Some(&attention_mask),
    ///             false,
    ///         )
    ///         .unwrap()
    /// });
    /// ```
    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        train: bool,
    ) -> Result<MobileBertQuestionAnsweringOutput, RustBertError> {
        let mobilebert_output = self.mobilebert.forward_t(
            input_ids,
            token_type_ids,
            position_ids,
            input_embeds,
            attention_mask,
            train,
        )?;

        let sequence_output = mobilebert_output.hidden_state.apply(&self.qa_outputs);
        let logits = sequence_output.split(1, -1);
        let (start_logits, end_logits) = (&logits[0], &logits[1]);
        let start_logits = start_logits.squeeze_dim(-1);
        let end_logits = end_logits.squeeze_dim(-1);

        Ok(MobileBertQuestionAnsweringOutput {
            start_logits,
            end_logits,
            all_hidden_states: mobilebert_output.all_hidden_states,
            all_attentions: mobilebert_output.all_attentions,
        })
    }
}

/// # MobileBERT for multiple choices
/// Multiple choices model using a MobileBERT base model and a linear classifier.
/// Input should be in the form `[CLS] Context [SEP] Possible choice [SEP]`. The choice is made along the batch axis,
/// assuming all elements of the batch are alternatives to be chosen from for a given context.
/// It is made of the following blocks:
/// - `mobilebert`: Base MobileBertModel
/// - `dropout`: Dropout layer before the last start/end logits prediction
/// - `classifier`: Linear layer for multiple choices
pub struct MobileBertForMultipleChoice {
    mobilebert: MobileBertModel,
    dropout: Dropout,
    classifier: nn::Linear,
}

impl MobileBertForMultipleChoice {
    /// Build a new `MobileBertForMultipleChoice`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the MobileBERT model
    /// * `config` - `MobileBertConfig` object defining the model architecture and decoder status
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::mobilebert::{MobileBertConfig, MobileBertForMultipleChoice};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = MobileBertConfig::from_file(config_path);
    /// let mobilebert = MobileBertForMultipleChoice::new(&p.root(), &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &MobileBertConfig) -> MobileBertForMultipleChoice
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let mobilebert = MobileBertModel::new(p / "mobilebert", config, true);
        let dropout = Dropout::new(config.hidden_dropout_prob);
        let classifier = nn::linear(p / "classifier", config.hidden_size, 1, Default::default());
        MobileBertForMultipleChoice {
            mobilebert,
            dropout,
            classifier,
        }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see `input_embeds`)
    /// * `token_type_ids` - Optional segment id of shape (*batch size*, *sequence_length*). Convention is value of 0 for the first sentence (incl. *SEP*) and 1 for the second sentence. If None set to 0.
    /// * `position_ids` - Optional position ids of shape (*batch size*, *sequence_length*). If None, will be incremented from 0.
    /// * `input_embeds` - Optional pre-computed input embeddings of shape (*batch size*, *sequence_length*, *hidden_size*). If None, input ids must be provided (see `input_ids`)
    /// * `attention_mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `MobileBertSequenceClassificationOutput` containing:
    ///   - `logits` - `Tensor` of shape (*1*, *batch_size*) containing the logits for each of the alternatives given
    ///   - `all_hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `all_attentions` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use tch::{nn, Device, Tensor, no_grad};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::Int64;
    /// use rust_bert::mobilebert::{MobileBertConfig, MobileBertForMultipleChoice};
    /// # let config_path = Path::new("path/to/config.json");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = MobileBertConfig::from_file(config_path);
    /// let model = MobileBertForMultipleChoice::new(&vs.root(), &config);
    /// let (batch_size, sequence_length) = (64, 128);
    /// let (num_choices, sequence_length) = (3, 128);
    /// let input_tensor = Tensor::rand(&[num_choices, sequence_length], (Int64, device));
    /// let attention_mask = Tensor::zeros(&[num_choices, sequence_length], (Int64, device));
    /// let token_type_ids = Tensor::zeros(&[num_choices, sequence_length], (Int64, device));
    /// let position_ids = Tensor::arange(sequence_length, (Int64, device))
    ///     .expand(&[num_choices, sequence_length], true);
    ///
    /// let model_output = no_grad(|| {
    ///     model
    ///         .forward_t(
    ///             Some(&input_tensor),
    ///             Some(&token_type_ids),
    ///             Some(&position_ids),
    ///             None,
    ///             Some(&attention_mask),
    ///             false,
    ///         )
    ///         .unwrap()
    /// });
    /// ```
    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        train: bool,
    ) -> Result<MobileBertSequenceClassificationOutput, RustBertError> {
        let (input_ids, num_choices) = match input_ids {
            Some(value) => (
                Some(value.view((-1, *value.size().last().unwrap()))),
                value.size()[1],
            ),
            None => (
                None,
                input_embeds
                    .as_ref()
                    .expect("At least one of input ids or input_embeds must be provided")
                    .size()[1],
            ),
        };
        let attention_mask =
            attention_mask.map(|tensor| tensor.view((-1, *tensor.size().last().unwrap())));
        let token_type_ids =
            token_type_ids.map(|tensor| tensor.view((-1, *tensor.size().last().unwrap())));
        let input_embeds =
            input_embeds.map(|tensor| tensor.view((-1, *tensor.size().last().unwrap())));
        let position_ids =
            position_ids.map(|tensor| tensor.view((-1, *tensor.size().last().unwrap())));

        let mobilebert_output = self.mobilebert.forward_t(
            input_ids.as_ref(),
            token_type_ids.as_ref(),
            position_ids.as_ref(),
            input_embeds.as_ref(),
            attention_mask.as_ref(),
            train,
        )?;

        let logits = mobilebert_output
            .pooled_output
            .unwrap()
            .apply_t(&self.dropout, train)
            .apply(&self.classifier)
            .view([-1, num_choices]);

        Ok(MobileBertSequenceClassificationOutput {
            logits,
            all_hidden_states: mobilebert_output.all_hidden_states,
            all_attentions: mobilebert_output.all_attentions,
        })
    }
}

/// # MobileBERT for token classification (e.g. NER, POS)
/// Token-level classifier predicting a label for each token provided. Note that because of wordpiece tokenization, the labels predicted are
/// not necessarily aligned with words in the sentence.
/// It is made of the following blocks:
/// - `mobilebert`: Base MobileBertModel
/// - `dropout`: Dropout layer before the last token-level predictions layer
/// - `classifier`: Linear layer for token classification
pub struct MobileBertForTokenClassification {
    mobilebert: MobileBertModel,
    dropout: Dropout,
    classifier: nn::Linear,
}

impl MobileBertForTokenClassification {
    /// Build a new `MobileBertForMultipleChoice`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the MobileBERT model
    /// * `config` - `MobileBertConfig` object defining the model architecture and decoder status
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::mobilebert::{MobileBertConfig, MobileBertForTokenClassification};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = MobileBertConfig::from_file(config_path);
    /// let mobilebert = MobileBertForTokenClassification::new(&p.root(), &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &MobileBertConfig) -> MobileBertForTokenClassification
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let mobilebert = MobileBertModel::new(p / "mobilebert", config, false);
        let dropout = Dropout::new(config.hidden_dropout_prob);
        let num_labels = config
            .id2label
            .as_ref()
            .expect("num_labels not provided in configuration")
            .len() as i64;
        let classifier = nn::linear(
            p / "classifier",
            config.hidden_size,
            num_labels,
            Default::default(),
        );
        MobileBertForTokenClassification {
            mobilebert,
            dropout,
            classifier,
        }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see `input_embeds`)
    /// * `token_type_ids` - Optional segment id of shape (*batch size*, *sequence_length*). Convention is value of 0 for the first sentence (incl. *SEP*) and 1 for the second sentence. If None set to 0.
    /// * `position_ids` - Optional position ids of shape (*batch size*, *sequence_length*). If None, will be incremented from 0.
    /// * `input_embeds` - Optional pre-computed input embeddings of shape (*batch size*, *sequence_length*, *hidden_size*). If None, input ids must be provided (see `input_ids`)
    /// * `attention_mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `MobileBertTokenClassificationOutput` containing:
    ///   - `logits` - `Tensor` of shape (*batch size*, *sequence_length*, *num_labels*) containing the logits for each of the input tokens and classes
    ///   - `all_hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `all_attentions` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use tch::{nn, Device, Tensor, no_grad};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::Int64;
    /// use rust_bert::mobilebert::{MobileBertConfig, MobileBertForTokenClassification};
    /// # let config_path = Path::new("path/to/config.json");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = MobileBertConfig::from_file(config_path);
    /// let model = MobileBertForTokenClassification::new(&vs.root(), &config);
    /// let (batch_size, sequence_length) = (64, 128);
    /// let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    /// let attention_mask = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    /// let token_type_ids = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    /// let position_ids = Tensor::arange(sequence_length, (Int64, device))
    ///     .expand(&[batch_size, sequence_length], true);
    ///
    /// let model_output = no_grad(|| {
    ///     model
    ///         .forward_t(
    ///             Some(&input_tensor),
    ///             Some(&token_type_ids),
    ///             Some(&position_ids),
    ///             None,
    ///             Some(&attention_mask),
    ///             false,
    ///         )
    ///         .unwrap()
    /// });
    /// ```
    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        train: bool,
    ) -> Result<MobileBertTokenClassificationOutput, RustBertError> {
        let mobilebert_output = self.mobilebert.forward_t(
            input_ids,
            token_type_ids,
            position_ids,
            input_embeds,
            attention_mask,
            train,
        )?;

        let logits = mobilebert_output
            .hidden_state
            .apply_t(&self.dropout, train)
            .apply(&self.classifier);

        Ok(MobileBertTokenClassificationOutput {
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

/// Container for the MobileBert sequence classification model output.
pub struct MobileBertSequenceClassificationOutput {
    /// Logits for each input (sequence) for each target class
    pub logits: Tensor,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all intermediate layers
    pub all_attentions: Option<Vec<Tensor>>,
}

/// Container for the MobileBert token classification model output.
pub struct MobileBertTokenClassificationOutput {
    /// Logits for each sequence item (token) for each target class
    pub logits: Tensor,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all intermediate layers
    pub all_attentions: Option<Vec<Tensor>>,
}

/// Container for the MobileBert question answering model output.
pub struct MobileBertQuestionAnsweringOutput {
    /// Logits for the start position for token of each input sequence
    pub start_logits: Tensor,
    /// Logits for the end position for token of each input sequence
    pub end_logits: Tensor,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all intermediate layers
    pub all_attentions: Option<Vec<Tensor>>,
}
