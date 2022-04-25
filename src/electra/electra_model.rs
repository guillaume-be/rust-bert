// Copyright 2020 The Google Research Authors.
// Copyright 2019-present, the HuggingFace Inc. team
// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

use crate::bert::BertConfig;
use crate::common::activations::Activation;
use crate::common::dropout::Dropout;
use crate::common::embeddings::get_shape_and_device_from_ids_embeddings_pair;
use crate::electra::embeddings::ElectraEmbeddings;
use crate::{bert::encoder::BertEncoder, common::activations::TensorFunction};
use crate::{Config, RustBertError};
use serde::{Deserialize, Serialize};
use std::{borrow::Borrow, collections::HashMap};
use tch::{nn, Kind, Tensor};

/// # Electra Pretrained model weight files
pub struct ElectraModelResources;

/// # Electra Pretrained model config files
pub struct ElectraConfigResources;

/// # Electra Pretrained model vocab files
pub struct ElectraVocabResources;

impl ElectraModelResources {
    /// Shared under Apache 2.0 license by the Google team at <https://github.com/google-research/electra>. Modified with conversion to C-array format.
    pub const BASE_GENERATOR: (&'static str, &'static str) = (
        "electra-base-generator/model",
        "https://huggingface.co/google/electra-base-generator/resolve/main/rust_model.ot",
    );
    /// Shared under Apache 2.0 license by the Google team at <https://github.com/google-research/electra>. Modified with conversion to C-array format.
    pub const BASE_DISCRIMINATOR: (&'static str, &'static str) = (
        "electra-base-discriminator/model",
        "https://huggingface.co/google/electra-base-discriminator/resolve/main/rust_model.ot",
    );
}

impl ElectraConfigResources {
    /// Shared under Apache 2.0 license by the Google team at <https://github.com/google-research/electra>. Modified with conversion to C-array format.
    pub const BASE_GENERATOR: (&'static str, &'static str) = (
        "electra-base-generator/config",
        "https://huggingface.co/google/electra-base-generator/resolve/main/config.json",
    );
    /// Shared under Apache 2.0 license by the Google team at <https://github.com/google-research/electra>. Modified with conversion to C-array format.
    pub const BASE_DISCRIMINATOR: (&'static str, &'static str) = (
        "electra-base-discriminator/config",
        "https://huggingface.co/google/electra-base-discriminator/resolve/main/config.json",
    );
}

impl ElectraVocabResources {
    /// Shared under Apache 2.0 license by the Google team at <https://github.com/google-research/electra>. Modified with conversion to C-array format.
    pub const BASE_GENERATOR: (&'static str, &'static str) = (
        "electra-base-generator/vocab",
        "https://huggingface.co/google/electra-base-generator/resolve/main/vocab.txt",
    );
    /// Shared under Apache 2.0 license by the Google team at <https://github.com/google-research/electra>. Modified with conversion to C-array format.
    pub const BASE_DISCRIMINATOR: (&'static str, &'static str) = (
        "electra-base-discriminator/vocab",
        "https://huggingface.co/google/electra-base-discriminator/resolve/main/vocab.txt",
    );
}

#[derive(Debug, Serialize, Deserialize, Clone)]
/// # Electra model configuration
/// Defines the Electra model architecture (e.g. number of layers, hidden layer size, label mapping...)
pub struct ElectraConfig {
    pub hidden_act: Activation,
    pub attention_probs_dropout_prob: f64,
    pub embedding_size: i64,
    pub hidden_dropout_prob: f64,
    pub hidden_size: i64,
    pub initializer_range: f32,
    pub layer_norm_eps: Option<f64>,
    pub intermediate_size: i64,
    pub max_position_embeddings: i64,
    pub num_attention_heads: i64,
    pub num_hidden_layers: i64,
    pub type_vocab_size: i64,
    pub vocab_size: i64,
    pub pad_token_id: i64,
    pub output_past: Option<bool>,
    pub output_attentions: Option<bool>,
    pub output_hidden_states: Option<bool>,
    pub id2label: Option<HashMap<i64, String>>,
    pub label2id: Option<HashMap<String, i64>>,
}

impl Config for ElectraConfig {}

impl Default for ElectraConfig {
    fn default() -> Self {
        ElectraConfig {
            hidden_act: Activation::gelu,
            attention_probs_dropout_prob: 0.1,
            embedding_size: 128,
            hidden_dropout_prob: 0.1,
            hidden_size: 256,
            initializer_range: 0.02,
            layer_norm_eps: Some(1e-12),
            intermediate_size: 1024,
            max_position_embeddings: 512,
            num_attention_heads: 4,
            num_hidden_layers: 12,
            type_vocab_size: 2,
            vocab_size: 30522,
            pad_token_id: 0,
            output_past: None,
            output_attentions: None,
            output_hidden_states: None,
            id2label: None,
            label2id: None,
        }
    }
}

/// # Electra Base model
/// Base architecture for Electra models.
/// It is made of the following blocks:
/// - `embeddings`: `token`, `position` and `segment_id` embeddings. Note that in contrast to BERT, the embeddings dimension is not necessarily equal to the hidden layer dimensions
/// - `encoder`: BertEncoder (transformer) made of a vector of layers. Each layer is made of a self-attention layer, an intermediate (linear) and output (linear + layer norm) layers
/// - `embeddings_project`: (optional) linear layer applied to project the embeddings space to the hidden layer dimension
pub struct ElectraModel {
    embeddings: ElectraEmbeddings,
    embeddings_project: Option<nn::Linear>,
    encoder: BertEncoder,
}

/// Defines the implementation of the ElectraModel.
impl ElectraModel {
    /// Build a new `ElectraModel`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the Electra model
    /// * `config` - `ElectraConfig` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::electra::{ElectraConfig, ElectraModel};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = ElectraConfig::from_file(config_path);
    /// let electra_model: ElectraModel = ElectraModel::new(&p.root() / "electra", &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &ElectraConfig) -> ElectraModel
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let embeddings = ElectraEmbeddings::new(p / "embeddings", config);
        let embeddings_project = if config.embedding_size != config.hidden_size {
            Some(nn::linear(
                p / "embeddings_project",
                config.embedding_size,
                config.hidden_size,
                Default::default(),
            ))
        } else {
            None
        };
        let bert_config = BertConfig {
            hidden_act: config.hidden_act,
            attention_probs_dropout_prob: config.attention_probs_dropout_prob,
            hidden_dropout_prob: config.hidden_dropout_prob,
            hidden_size: config.hidden_size,
            initializer_range: config.initializer_range,
            intermediate_size: config.intermediate_size,
            max_position_embeddings: config.max_position_embeddings,
            num_attention_heads: config.num_attention_heads,
            num_hidden_layers: config.num_hidden_layers,
            type_vocab_size: config.type_vocab_size,
            vocab_size: config.vocab_size,
            output_attentions: config.output_attentions,
            output_hidden_states: config.output_hidden_states,
            is_decoder: None,
            id2label: config.id2label.clone(),
            label2id: config.label2id.clone(),
        };
        let encoder = BertEncoder::new(p / "encoder", &bert_config);
        ElectraModel {
            embeddings,
            embeddings_project,
            encoder,
        }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see `input_embeds`)
    /// * `mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `token_type_ids` - Optional segment id of shape (*batch size*, *sequence_length*). Convention is value of 0 for the first sentence (incl. *SEP*) and 1 for the second sentence. If None set to 0.
    /// * `position_ids` - Optional position ids of shape (*batch size*, *sequence_length*). If None, will be incremented from 0.
    /// * `input_embeds` - Optional pre-computed input embeddings of shape (*batch size*, *sequence_length*, *hidden_size*). If None, input ids must be provided (see `input_ids`)
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `ElectraModelOutput` containing:
    ///   - `hidden_state` - `Tensor` of shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `all_hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `all_attentions` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rust_bert::electra::{ElectraModel, ElectraConfig};
    /// # use tch::{nn, Device, Tensor, no_grad};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::Int64;
    /// # let config_path = Path::new("path/to/config.json");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = ElectraConfig::from_file(config_path);
    /// # let electra_model: ElectraModel = ElectraModel::new(&vs.root(), &config);
    /// let (batch_size, sequence_length) = (64, 128);
    /// let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    /// let mask = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    /// let token_type_ids = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    /// let position_ids = Tensor::arange(sequence_length, (Int64, device))
    ///     .expand(&[batch_size, sequence_length], true);
    ///
    /// let model_output = no_grad(|| {
    ///     electra_model
    ///         .forward_t(
    ///             Some(&input_tensor),
    ///             Some(&mask),
    ///             Some(&token_type_ids),
    ///             Some(&position_ids),
    ///             None,
    ///             false,
    ///         )
    ///         .unwrap()
    /// });
    /// ```
    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        mask: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        train: bool,
    ) -> Result<ElectraModelOutput, RustBertError> {
        let (input_shape, device) =
            get_shape_and_device_from_ids_embeddings_pair(input_ids, input_embeds)?;

        let calc_mask = if mask.is_none() {
            Some(Tensor::ones(&input_shape, (Kind::Int64, device)))
        } else {
            None
        };
        let mask = mask.unwrap_or_else(|| calc_mask.as_ref().unwrap());

        let extended_attention_mask = match mask.dim() {
            3 => mask.unsqueeze(1),
            2 => mask.unsqueeze(1).unsqueeze(1),
            _ => {
                return Err(RustBertError::ValueError(
                    "Invalid attention mask dimension, must be 2 or 3".into(),
                ));
            }
        };

        let hidden_states = self.embeddings.forward_t(
            input_ids,
            token_type_ids,
            position_ids,
            input_embeds,
            train,
        )?;

        let hidden_states = match &self.embeddings_project {
            Some(layer) => hidden_states.apply(layer),
            None => hidden_states,
        };

        let encoder_output = self.encoder.forward_t(
            &hidden_states,
            Some(&extended_attention_mask),
            None,
            None,
            train,
        );

        Ok(ElectraModelOutput {
            hidden_state: encoder_output.hidden_state,
            all_hidden_states: encoder_output.all_hidden_states,
            all_attentions: encoder_output.all_attentions,
        })
    }
}

/// # Electra Discriminator head
/// Discriminator head for Electra models
/// It is made of the following blocks:
/// - `dense`: linear layer of dimension (*hidden_size*, *hidden_size*)
/// - `dense_prediction`: linear layer of dimension (*hidden_size*, *1*) mapping the model output to a 1-dimension space to identify original and generated tokens
/// - `activation`: activation layer (one of GeLU, ReLU or Mish)
pub struct ElectraDiscriminatorHead {
    dense: nn::Linear,
    dense_prediction: nn::Linear,
    activation: TensorFunction,
}

/// Defines the implementation of the ElectraDiscriminatorHead.
impl ElectraDiscriminatorHead {
    /// Build a new `ElectraDiscriminatorHead`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the Electra model
    /// * `config` - `ElectraConfig` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::electra::{ElectraConfig, ElectraDiscriminatorHead};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = ElectraConfig::from_file(config_path);
    /// let discriminator_head = ElectraDiscriminatorHead::new(&p.root() / "electra", &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &ElectraConfig) -> ElectraDiscriminatorHead
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
        let dense_prediction = nn::linear(
            p / "dense_prediction",
            config.hidden_size,
            1,
            Default::default(),
        );
        let activation = config.hidden_act.get_function();
        ElectraDiscriminatorHead {
            dense,
            dense_prediction,
            activation,
        }
    }

    /// Forward pass through the discriminator head
    ///
    /// # Arguments
    ///
    /// * `encoder_hidden_states` - Reference to input tensor of shape (*batch size*, *sequence_length*, *hidden_size*).
    ///
    /// # Returns
    ///
    /// * `output` - `Tensor` of shape (*batch size*, *sequence_length*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rust_bert::electra::{ElectraConfig, ElectraDiscriminatorHead};
    /// # use tch::{nn, Device, Tensor, no_grad};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::Float;
    /// # let config_path = Path::new("path/to/config.json");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = ElectraConfig::from_file(config_path);
    /// # let discriminator_head = ElectraDiscriminatorHead::new(&vs.root(), &config);
    /// let (batch_size, sequence_length) = (64, 128);
    /// let input_tensor = Tensor::rand(
    ///     &[batch_size, sequence_length, config.hidden_size],
    ///     (Float, device),
    /// );
    ///
    /// let output = no_grad(|| discriminator_head.forward(&input_tensor));
    /// ```
    pub fn forward(&self, encoder_hidden_states: &Tensor) -> Tensor {
        let output = encoder_hidden_states.apply(&self.dense);
        let output = (self.activation.get_fn())(&output);
        output.apply(&self.dense_prediction).squeeze()
    }
}

/// # Electra Generator head
/// Generator head for Electra models
/// It is made of the following blocks:
/// - `dense`: linear layer of dimension (*hidden_size*, *embeddings_size*) to project the model output dimension  to the embeddings size
/// - `layer_norm`: Layer normalization
/// - `activation`: GeLU activation
pub struct ElectraGeneratorHead {
    dense: nn::Linear,
    layer_norm: nn::LayerNorm,
    activation: TensorFunction,
}

/// Defines the implementation of the ElectraGeneratorHead.
impl ElectraGeneratorHead {
    /// Build a new `ElectraGeneratorHead`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the Electra model
    /// * `config` - `ElectraConfig` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::electra::{ElectraConfig, ElectraGeneratorHead};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = ElectraConfig::from_file(config_path);
    /// let generator_head = ElectraGeneratorHead::new(&p.root() / "electra", &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &ElectraConfig) -> ElectraGeneratorHead
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let layer_norm = nn::layer_norm(
            p / "LayerNorm",
            vec![config.embedding_size],
            Default::default(),
        );
        let dense = nn::linear(
            p / "dense",
            config.hidden_size,
            config.embedding_size,
            Default::default(),
        );
        let activation = Activation::gelu.get_function();

        ElectraGeneratorHead {
            dense,
            layer_norm,
            activation,
        }
    }

    /// Forward pass through the generator head
    ///
    /// # Arguments
    ///
    /// * `encoder_hidden_states` - Reference to input tensor of shape (*batch size*, *sequence_length*, *hidden_size*).
    ///
    /// # Returns
    ///
    /// * `output` - `Tensor` of shape (*batch size*, *sequence_length*, *embeddings_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rust_bert::electra::{ElectraConfig, ElectraGeneratorHead};
    /// # use tch::{nn, Device, Tensor, no_grad};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::Float;
    /// # let config_path = Path::new("path/to/config.json");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = ElectraConfig::from_file(config_path);
    /// # let generator_head = ElectraGeneratorHead::new(&vs.root(), &config);
    /// let (batch_size, sequence_length) = (64, 128);
    /// let input_tensor = Tensor::rand(
    ///     &[batch_size, sequence_length, config.hidden_size],
    ///     (Float, device),
    /// );
    ///
    /// let output = no_grad(|| generator_head.forward(&input_tensor));
    /// ```
    pub fn forward(&self, encoder_hidden_states: &Tensor) -> Tensor {
        let output = encoder_hidden_states.apply(&self.dense);
        let output = (self.activation.get_fn())(&output);
        output.apply(&self.layer_norm)
    }
}

/// # Electra for Masked Language Modeling
/// Masked Language modeling Electra model
/// It is made of the following blocks:
/// - `electra`: `ElectraModel` (based on a `BertEncoder` and custom embeddings)
/// - `generator_head`: `ElectraGeneratorHead` to generate token predictions of dimension *embedding_size*
/// - `lm_head`: linear layer of dimension (*embeddings_size*, *vocab_size*) to project the output to the vocab size
pub struct ElectraForMaskedLM {
    electra: ElectraModel,
    generator_head: ElectraGeneratorHead,
    lm_head: nn::Linear,
}

/// Defines the implementation of the ElectraForMaskedLM.
impl ElectraForMaskedLM {
    /// Build a new `ElectraForMaskedLM`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the Electra model
    /// * `config` - `ElectraConfig` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::electra::{ElectraConfig, ElectraForMaskedLM};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = ElectraConfig::from_file(config_path);
    /// let electra_model: ElectraForMaskedLM = ElectraForMaskedLM::new(&p.root(), &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &ElectraConfig) -> ElectraForMaskedLM
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let electra = ElectraModel::new(p / "electra", config);
        let generator_head = ElectraGeneratorHead::new(p / "generator_predictions", config);
        let lm_head = nn::linear(
            p / "generator_lm_head",
            config.embedding_size,
            config.vocab_size,
            Default::default(),
        );

        ElectraForMaskedLM {
            electra,
            generator_head,
            lm_head,
        }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see `input_embeds`)
    /// * `mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `token_type_ids` - Optional segment id of shape (*batch size*, *sequence_length*). Convention is value of 0 for the first sentence (incl. *SEP*) and 1 for the second sentence. If None set to 0.
    /// * `position_ids` - Optional position ids of shape (*batch size*, *sequence_length*). If None, will be incremented from 0.
    /// * `input_embeds` - Optional pre-computed input embeddings of shape (*batch size*, *sequence_length*, *hidden_size*). If None, input ids must be provided (see `input_ids`)
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `ElectraMaskedLMOutput` containing:
    ///   - `prediction_scores` - `Tensor` of shape (*batch size*, *sequence_length*, *vocab_size*)
    ///   - `all_hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `all_attentions` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rust_bert::electra::{ElectraForMaskedLM, ElectraConfig};
    /// # use tch::{nn, Device, Tensor, no_grad};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::Int64;
    /// # let config_path = Path::new("path/to/config.json");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = ElectraConfig::from_file(config_path);
    /// # let electra_model: ElectraForMaskedLM = ElectraForMaskedLM::new(&vs.root(), &config);
    /// let (batch_size, sequence_length) = (64, 128);
    /// let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    /// let mask = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    /// let token_type_ids = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    /// let position_ids = Tensor::arange(sequence_length, (Int64, device))
    ///     .expand(&[batch_size, sequence_length], true);
    ///
    /// let model_output = no_grad(|| {
    ///     electra_model.forward_t(
    ///         Some(&input_tensor),
    ///         Some(&mask),
    ///         Some(&token_type_ids),
    ///         Some(&position_ids),
    ///         None,
    ///         false,
    ///     )
    /// });
    /// ```
    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        mask: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        train: bool,
    ) -> ElectraMaskedLMOutput {
        let base_model_output = self
            .electra
            .forward_t(
                input_ids,
                mask,
                token_type_ids,
                position_ids,
                input_embeds,
                train,
            )
            .unwrap();
        let hidden_states = self.generator_head.forward(&base_model_output.hidden_state);
        let prediction_scores = hidden_states.apply(&self.lm_head);
        ElectraMaskedLMOutput {
            prediction_scores,
            all_hidden_states: base_model_output.all_hidden_states,
            all_attentions: base_model_output.all_attentions,
        }
    }
}

/// # Electra Discriminator
/// Electra discriminator model
/// It is made of the following blocks:
/// - `electra`: `ElectraModel` (based on a `BertEncoder` and custom embeddings)
/// - `discriminator_head`: `ElectraDiscriminatorHead` to classify each token into either `original` or `generated`
pub struct ElectraDiscriminator {
    electra: ElectraModel,
    discriminator_head: ElectraDiscriminatorHead,
}

/// Defines the implementation of the ElectraDiscriminator.
impl ElectraDiscriminator {
    /// Build a new `ElectraDiscriminator`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the Electra model
    /// * `config` - `ElectraConfig` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::electra::{ElectraConfig, ElectraDiscriminator};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = ElectraConfig::from_file(config_path);
    /// let electra_model: ElectraDiscriminator = ElectraDiscriminator::new(&p.root(), &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &ElectraConfig) -> ElectraDiscriminator
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let electra = ElectraModel::new(p / "electra", config);
        let discriminator_head =
            ElectraDiscriminatorHead::new(p / "discriminator_predictions", config);

        ElectraDiscriminator {
            electra,
            discriminator_head,
        }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see `input_embeds`)
    /// * `mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `token_type_ids` - Optional segment id of shape (*batch size*, *sequence_length*). Convention is value of 0 for the first sentence (incl. *SEP*) and 1 for the second sentence. If None set to 0.
    /// * `position_ids` - Optional position ids of shape (*batch size*, *sequence_length*). If None, will be incremented from 0.
    /// * `input_embeds` - Optional pre-computed input embeddings of shape (*batch size*, *sequence_length*, *hidden_size*). If None, input ids must be provided (see `input_ids`)
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `ElectraDiscriminatorOutput` containing:
    ///   - `logits` - `Tensor` of shape (*batch size*, *sequence_length*) containing the probability of each token to be generated by a language model
    ///   - `all_hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `all_attentions` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rust_bert::electra::{ElectraDiscriminator, ElectraConfig};
    /// # use tch::{nn, Device, Tensor, no_grad};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::Int64;
    /// # let config_path = Path::new("path/to/config.json");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = ElectraConfig::from_file(config_path);
    /// # let electra_model: ElectraDiscriminator = ElectraDiscriminator::new(&vs.root(), &config);
    ///  let (batch_size, sequence_length) = (64, 128);
    ///  let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    ///  let mask = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    ///  let token_type_ids = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    ///  let position_ids = Tensor::arange(sequence_length, (Int64, device)).expand(&[batch_size, sequence_length], true);
    ///
    ///  let model_output = no_grad(|| {
    ///    electra_model
    ///         .forward_t(Some(&input_tensor),
    ///                    Some(&mask),
    ///                    Some(&token_type_ids),
    ///                    Some(&position_ids),
    ///                    None,
    ///                    false)
    ///    });
    /// ```
    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        mask: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        train: bool,
    ) -> ElectraDiscriminatorOutput {
        let base_model_output = self
            .electra
            .forward_t(
                input_ids,
                mask,
                token_type_ids,
                position_ids,
                input_embeds,
                train,
            )
            .unwrap();
        let probabilities = self
            .discriminator_head
            .forward(&base_model_output.hidden_state)
            .sigmoid();
        ElectraDiscriminatorOutput {
            probabilities,
            all_hidden_states: base_model_output.all_hidden_states,
            all_attentions: base_model_output.all_attentions,
        }
    }
}

/// # Electra for token classification (e.g. POS, NER)
/// Electra model with a token tagging head
/// It is made of the following blocks:
/// - `electra`: `ElectraModel` (based on a `BertEncoder` and custom embeddings)
/// - `dropout`: Dropout layer
/// - `classifier`: linear layer of dimension (*hidden_size*, *num_classes*) to project the output to the target label space
pub struct ElectraForTokenClassification {
    electra: ElectraModel,
    dropout: Dropout,
    classifier: nn::Linear,
}

/// Defines the implementation of the ElectraForTokenClassification.
impl ElectraForTokenClassification {
    /// Build a new `ElectraForTokenClassification`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the Electra model
    /// * `config` - `ElectraConfig` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::electra::{ElectraConfig, ElectraForTokenClassification};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = ElectraConfig::from_file(config_path);
    /// let electra_model: ElectraForTokenClassification =
    ///     ElectraForTokenClassification::new(&p.root(), &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &ElectraConfig) -> ElectraForTokenClassification
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let electra = ElectraModel::new(p / "electra", config);
        let dropout = Dropout::new(config.hidden_dropout_prob);
        let num_labels = config
            .id2label
            .as_ref()
            .expect("id2label must be provided for classifiers")
            .len() as i64;
        let classifier = nn::linear(
            p / "classifier",
            config.hidden_size,
            num_labels,
            Default::default(),
        );

        ElectraForTokenClassification {
            electra,
            dropout,
            classifier,
        }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see `input_embeds`)
    /// * `mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `token_type_ids` - Optional segment id of shape (*batch size*, *sequence_length*). Convention is value of 0 for the first sentence (incl. *SEP*) and 1 for the second sentence. If None set to 0.
    /// * `position_ids` - Optional position ids of shape (*batch size*, *sequence_length*). If None, will be incremented from 0.
    /// * `input_embeds` - Optional pre-computed input embeddings of shape (*batch size*, *sequence_length*, *hidden_size*). If None, input ids must be provided (see `input_ids`)
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `ElectraTokenClassificationOutput` containing:
    ///   - `logits` - `Tensor` of shape (*batch size*, *sequence_length*, *num_labels*) containing the logits for each of the input tokens and classes
    ///   - `all_hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `all_attentions` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rust_bert::electra::{ElectraForTokenClassification, ElectraConfig};
    /// # use tch::{nn, Device, Tensor, no_grad};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::Int64;
    /// # let config_path = Path::new("path/to/config.json");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = ElectraConfig::from_file(config_path);
    /// # let electra_model: ElectraForTokenClassification = ElectraForTokenClassification::new(&vs.root(), &config);
    ///  let (batch_size, sequence_length) = (64, 128);
    ///  let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    ///  let mask = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    ///  let token_type_ids = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    ///  let position_ids = Tensor::arange(sequence_length, (Int64, device)).expand(&[batch_size, sequence_length], true);
    ///
    ///  let model_output = no_grad(|| {
    ///    electra_model
    ///         .forward_t(Some(&input_tensor),
    ///                    Some(&mask),
    ///                    Some(&token_type_ids),
    ///                    Some(&position_ids),
    ///                    None,
    ///                    false)
    ///    });
    /// ```
    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        mask: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        train: bool,
    ) -> ElectraTokenClassificationOutput {
        let base_model_output = self
            .electra
            .forward_t(
                input_ids,
                mask,
                token_type_ids,
                position_ids,
                input_embeds,
                train,
            )
            .unwrap();
        let logits = base_model_output
            .hidden_state
            .apply_t(&self.dropout, train)
            .apply(&self.classifier);
        ElectraTokenClassificationOutput {
            logits,
            all_hidden_states: base_model_output.all_hidden_states,
            all_attentions: base_model_output.all_attentions,
        }
    }
}

/// Container for the Electra model output.
pub struct ElectraModelOutput {
    /// Last hidden states from the model
    pub hidden_state: Tensor,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all intermediate layers
    pub all_attentions: Option<Vec<Tensor>>,
}

/// Container for the Electra discriminator model output.
pub struct ElectraDiscriminatorOutput {
    /// Probabilities for each sequence item (token) to be generated by a language model
    pub probabilities: Tensor,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all intermediate layers
    pub all_attentions: Option<Vec<Tensor>>,
}

/// Container for the Electra masked LM model output.
pub struct ElectraMaskedLMOutput {
    /// Logits for the vocabulary items at each sequence position
    pub prediction_scores: Tensor,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all intermediate layers
    pub all_attentions: Option<Vec<Tensor>>,
}

/// Container for the Electra token classification model output.
pub struct ElectraTokenClassificationOutput {
    /// Logits for each sequence item (token) for each target class
    pub logits: Tensor,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all intermediate layers
    pub all_attentions: Option<Vec<Tensor>>,
}
