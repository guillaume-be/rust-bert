// Copyright 2018 Google AI and Google Brain team.
// Copyright 2020-present, the HuggingFace Inc. team.
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

use crate::albert::encoder::AlbertTransformer;
use crate::common::activations::Activation;
use crate::common::dropout::Dropout;
use crate::common::embeddings::get_shape_and_device_from_ids_embeddings_pair;
use crate::{albert::embeddings::AlbertEmbeddings, common::activations::TensorFunction};
use crate::{Config, RustBertError};
use serde::{Deserialize, Serialize};
use std::{borrow::Borrow, collections::HashMap};
use tch::nn::Module;
use tch::{nn, Kind, Tensor};

/// # ALBERT Pretrained model weight files
pub struct AlbertModelResources;

/// # ALBERT Pretrained model config files
pub struct AlbertConfigResources;

/// # ALBERT Pretrained model vocab files
pub struct AlbertVocabResources;

impl AlbertModelResources {
    /// Shared under Apache 2.0 license by the Google team at <https://github.com/google-research/ALBERT>. Modified with conversion to C-array format.
    pub const ALBERT_BASE_V2: (&'static str, &'static str) = (
        "albert-base-v2/model",
        "https://huggingface.co/albert-base-v2/resolve/main/rust_model.ot",
    );
}

impl AlbertConfigResources {
    /// Shared under Apache 2.0 license by the Google team at <https://github.com/google-research/ALBERT>. Modified with conversion to C-array format.
    pub const ALBERT_BASE_V2: (&'static str, &'static str) = (
        "albert-base-v2/config",
        "https://huggingface.co/albert-base-v2/resolve/main/config.json",
    );
}

impl AlbertVocabResources {
    /// Shared under Apache 2.0 license by the Google team at <https://github.com/google-research/ALBERT>. Modified with conversion to C-array format.
    pub const ALBERT_BASE_V2: (&'static str, &'static str) = (
        "albert-base-v2/spiece",
        "https://huggingface.co/albert-base-v2/resolve/main/spiece.model",
    );
}

#[derive(Debug, Serialize, Deserialize, Clone)]
/// # ALBERT model configuration
/// Defines the ALBERT model architecture (e.g. number of layers, hidden layer size, label mapping...)
pub struct AlbertConfig {
    pub hidden_act: Activation,
    pub attention_probs_dropout_prob: f64,
    pub classifier_dropout_prob: Option<f64>,
    pub bos_token_id: i64,
    pub eos_token_id: i64,
    pub down_scale_factor: i64,
    pub embedding_size: i64,
    pub gap_size: i64,
    pub hidden_dropout_prob: f64,
    pub hidden_size: i64,
    pub initializer_range: f32,
    pub inner_group_num: i64,
    pub intermediate_size: i64,
    pub layer_norm_eps: Option<f64>,
    pub max_position_embeddings: i64,
    pub net_structure_type: i64,
    pub num_attention_heads: i64,
    pub num_hidden_groups: i64,
    pub num_hidden_layers: i64,
    pub num_memory_blocks: i64,
    pub pad_token_id: i64,
    pub type_vocab_size: i64,
    pub vocab_size: i64,
    pub output_attentions: Option<bool>,
    pub output_hidden_states: Option<bool>,
    pub is_decoder: Option<bool>,
    pub id2label: Option<HashMap<i64, String>>,
    pub label2id: Option<HashMap<String, i64>>,
}

impl Config for AlbertConfig {}

/// # ALBERT Base model
/// Base architecture for ALBERT models. Task-specific models will be built from this common base model
/// It is made of the following blocks:
/// - `embeddings`: `token`, `position` and `segment_id` embeddings
/// - `encoder`: Encoder (transformer) made of a vector of layers. Each layer is made of a self-attention layer, an intermediate (linear) and output (linear + layer norm) layers. Note that the weights are shared across layers, allowing for a reduction in the model memory footprint.
/// - `pooler`: linear layer applied to the first element of the sequence (*MASK* token)
/// - `pooler_activation`: Tanh activation function for the pooling layer
pub struct AlbertModel {
    embeddings: AlbertEmbeddings,
    encoder: AlbertTransformer,
    pooler: nn::Linear,
    pooler_activation: TensorFunction,
}

impl AlbertModel {
    /// Build a new `AlbertModel`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the ALBERT model
    /// * `config` - `AlbertConfig` object defining the model architecture and decoder status
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::albert::{AlbertConfig, AlbertModel};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = AlbertConfig::from_file(config_path);
    /// let albert: AlbertModel = AlbertModel::new(&p.root() / "albert", &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &AlbertConfig) -> AlbertModel
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let embeddings = AlbertEmbeddings::new(p / "embeddings", config);
        let encoder = AlbertTransformer::new(p / "encoder", config);
        let pooler = nn::linear(
            p / "pooler",
            config.hidden_size,
            config.hidden_size,
            Default::default(),
        );
        let pooler_activation = Activation::tanh.get_function();

        AlbertModel {
            embeddings,
            encoder,
            pooler,
            pooler_activation,
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
    /// * `AlbertOutput` containing:
    ///   - `hidden_state` - `Tensor` of shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `pooled_output` - `Tensor` of shape (*batch size*, *hidden_size*)
    ///   - `all_hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `all_attentions` - `Option<Vec<Vec<Tensor>>>` of length *num_hidden_layers* of nested length *inner_group_num* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use tch::{nn, Device, Tensor, no_grad};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::Int64;
    /// use rust_bert::albert::{AlbertConfig, AlbertModel};
    /// # let config_path = Path::new("path/to/config.json");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = AlbertConfig::from_file(config_path);
    /// # let albert_model: AlbertModel = AlbertModel::new(&vs.root(), &config);
    /// let (batch_size, sequence_length) = (64, 128);
    /// let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    /// let mask = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    /// let token_type_ids = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    /// let position_ids = Tensor::arange(sequence_length, (Int64, device))
    ///     .expand(&[batch_size, sequence_length], true);
    ///
    /// let model_output = no_grad(|| {
    ///     albert_model
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
    ) -> Result<AlbertOutput, RustBertError> {
        let (input_shape, device) =
            get_shape_and_device_from_ids_embeddings_pair(input_ids, input_embeds)?;

        let calc_mask = if mask.is_none() {
            Some(Tensor::ones(&input_shape, (Kind::Int64, device)))
        } else {
            None
        };
        let mask = mask.unwrap_or_else(|| calc_mask.as_ref().unwrap());

        let embedding_output = self.embeddings.forward_t(
            input_ids,
            token_type_ids,
            position_ids,
            input_embeds,
            train,
        )?;

        let extended_attention_mask = mask.unsqueeze(1).unsqueeze(2);
        let extended_attention_mask: Tensor =
            ((extended_attention_mask.ones_like() - extended_attention_mask) * -10000.0)
                .to_kind(embedding_output.kind());

        let transformer_output =
            self.encoder
                .forward_t(&embedding_output, Some(extended_attention_mask), train);

        let pooled_output = self
            .pooler
            .forward(&transformer_output.hidden_state.select(1, 0));
        let pooled_output = (self.pooler_activation.get_fn())(&pooled_output);

        Ok(AlbertOutput {
            hidden_state: transformer_output.hidden_state,
            pooled_output,
            all_hidden_states: transformer_output.all_hidden_states,
            all_attentions: transformer_output.all_attentions,
        })
    }
}

pub struct AlbertMLMHead {
    layer_norm: nn::LayerNorm,
    dense: nn::Linear,
    decoder: nn::Linear,
    activation: TensorFunction,
}

impl AlbertMLMHead {
    pub fn new<'p, P>(p: P, config: &AlbertConfig) -> AlbertMLMHead
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let layer_norm_eps = config.layer_norm_eps.unwrap_or(1e-12);
        let layer_norm_config = nn::LayerNormConfig {
            eps: layer_norm_eps,
            ..Default::default()
        };
        let layer_norm = nn::layer_norm(
            p / "LayerNorm",
            vec![config.embedding_size],
            layer_norm_config,
        );
        let dense = nn::linear(
            p / "dense",
            config.hidden_size,
            config.embedding_size,
            Default::default(),
        );
        let decoder = nn::linear(
            p / "decoder",
            config.embedding_size,
            config.vocab_size,
            Default::default(),
        );

        let activation = config.hidden_act.get_function();

        AlbertMLMHead {
            layer_norm,
            dense,
            decoder,
            activation,
        }
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Tensor {
        let output: Tensor = (self.activation.get_fn())(&hidden_states.apply(&self.dense));
        output.apply(&self.layer_norm).apply(&self.decoder)
    }
}

/// # ALBERT for masked language model
/// Base ALBERT model with a masked language model head to predict missing tokens, for example `"Looks like one [MASK] is missing" -> "person"`
/// It is made of the following blocks:
/// - `albert`: Base AlbertModel
/// - `predictions`: ALBERT MLM prediction head
pub struct AlbertForMaskedLM {
    albert: AlbertModel,
    predictions: AlbertMLMHead,
}

impl AlbertForMaskedLM {
    /// Build a new `AlbertForMaskedLM`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the ALBERT model
    /// * `config` - `AlbertConfig` object defining the model architecture and decoder status
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::albert::{AlbertConfig, AlbertForMaskedLM};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = AlbertConfig::from_file(config_path);
    /// let albert: AlbertForMaskedLM = AlbertForMaskedLM::new(&p.root(), &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &AlbertConfig) -> AlbertForMaskedLM
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let albert = AlbertModel::new(p / "albert", config);
        let predictions = AlbertMLMHead::new(p / "predictions", config);

        AlbertForMaskedLM {
            albert,
            predictions,
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
    /// * `AlbertMaskedLMOutput` containing:
    ///   - `prediction_scores` - `Tensor` of shape (*batch size*, *sequence_length*, *vocab_size*)
    ///   - `all_hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `all_attentions` - `Option<Vec<Vec<Tensor>>>` of length *num_hidden_layers* of nested length *inner_group_num* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use tch::{nn, Device, Tensor, no_grad};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::Int64;
    /// use rust_bert::albert::{AlbertConfig, AlbertForMaskedLM};
    /// # let config_path = Path::new("path/to/config.json");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = AlbertConfig::from_file(config_path);
    /// # let albert_model: AlbertForMaskedLM = AlbertForMaskedLM::new(&vs.root(), &config);
    /// let (batch_size, sequence_length) = (64, 128);
    /// let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    /// let mask = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    /// let token_type_ids = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    /// let position_ids = Tensor::arange(sequence_length, (Int64, device))
    ///     .expand(&[batch_size, sequence_length], true);
    ///
    /// let masked_lm_output = no_grad(|| {
    ///     albert_model.forward_t(
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
    ) -> AlbertMaskedLMOutput {
        let base_model_output = self
            .albert
            .forward_t(
                input_ids,
                mask,
                token_type_ids,
                position_ids,
                input_embeds,
                train,
            )
            .unwrap();
        let prediction_scores = self.predictions.forward(&base_model_output.hidden_state);
        AlbertMaskedLMOutput {
            prediction_scores,
            all_hidden_states: base_model_output.all_hidden_states,
            all_attentions: base_model_output.all_attentions,
        }
    }
}

/// # ALBERT for sequence classification
/// Base ALBERT model with a classifier head to perform sentence or document-level classification
/// It is made of the following blocks:
/// - `albert`: Base AlbertModel
/// - `dropout`: Dropout layer
/// - `classifier`: linear layer for classification
pub struct AlbertForSequenceClassification {
    albert: AlbertModel,
    dropout: Dropout,
    classifier: nn::Linear,
}

impl AlbertForSequenceClassification {
    /// Build a new `AlbertForSequenceClassification`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the ALBERT model
    /// * `config` - `AlbertConfig` object defining the model architecture and decoder status
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::albert::{AlbertConfig, AlbertForSequenceClassification};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = AlbertConfig::from_file(config_path);
    /// let albert: AlbertForSequenceClassification =
    ///     AlbertForSequenceClassification::new(&p.root(), &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &AlbertConfig) -> AlbertForSequenceClassification
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let albert = AlbertModel::new(p / "albert", config);
        let classifier_dropout_prob = config.classifier_dropout_prob.unwrap_or(0.1);
        let dropout = Dropout::new(classifier_dropout_prob);
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

        AlbertForSequenceClassification {
            albert,
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
    /// * `AlbertSequenceClassificationOutput` containing:
    ///   - `logits` - `Tensor` of shape (*batch size*, *num_labels*)
    ///   - `all_hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `all_attentions` - `Option<Vec<Vec<Tensor>>>` of length *num_hidden_layers* of nested length *inner_group_num* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use tch::{nn, Device, Tensor, no_grad};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::Int64;
    /// use rust_bert::albert::{AlbertConfig, AlbertForSequenceClassification};
    /// # let config_path = Path::new("path/to/config.json");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = AlbertConfig::from_file(config_path);
    /// # let albert_model: AlbertForSequenceClassification = AlbertForSequenceClassification::new(&vs.root(), &config);
    ///  let (batch_size, sequence_length) = (64, 128);
    ///  let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    ///  let mask = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    ///  let token_type_ids = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    ///  let position_ids = Tensor::arange(sequence_length, (Int64, device)).expand(&[batch_size, sequence_length], true);
    ///
    ///  let classification_output = no_grad(|| {
    ///    albert_model
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
    ) -> AlbertSequenceClassificationOutput {
        let base_model_output = self
            .albert
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
            .pooled_output
            .apply_t(&self.dropout, train)
            .apply(&self.classifier);
        AlbertSequenceClassificationOutput {
            logits,
            all_hidden_states: base_model_output.all_hidden_states,
            all_attentions: base_model_output.all_attentions,
        }
    }
}

/// # ALBERT for token classification (e.g. NER, POS)
/// Token-level classifier predicting a label for each token provided. Note that because of SentencePiece tokenization, the labels predicted are
/// not necessarily aligned with words in the sentence.
/// It is made of the following blocks:
/// - `albert`: Base AlbertModel
/// - `dropout`: Dropout to apply on the encoder last hidden states
/// - `classifier`: Linear layer for token classification
pub struct AlbertForTokenClassification {
    albert: AlbertModel,
    dropout: Dropout,
    classifier: nn::Linear,
}

impl AlbertForTokenClassification {
    /// Build a new `AlbertForTokenClassification`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the ALBERT model
    /// * `config` - `AlbertConfig` object defining the model architecture and decoder status
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::albert::{AlbertConfig, AlbertForTokenClassification};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = AlbertConfig::from_file(config_path);
    /// let albert: AlbertForTokenClassification =
    ///     AlbertForTokenClassification::new(&p.root(), &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &AlbertConfig) -> AlbertForTokenClassification
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let albert = AlbertModel::new(p / "albert", config);
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

        AlbertForTokenClassification {
            albert,
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
    /// * `AlbertTokenClassificationOutput` containing:
    ///   - `logits` - `Tensor` of shape (*batch size*, *sequence_length*, *num_labels*) containing the logits for each of the input tokens and classes
    ///   - `all_hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `all_attentions` - `Option<Vec<Vec<Tensor>>>` of length *num_hidden_layers* of nested length *inner_group_num* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use tch::{nn, Device, Tensor, no_grad};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::Int64;
    /// use rust_bert::albert::{AlbertConfig, AlbertForTokenClassification};
    /// # let config_path = Path::new("path/to/config.json");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = AlbertConfig::from_file(config_path);
    /// # let albert_model: AlbertForTokenClassification = AlbertForTokenClassification::new(&vs.root(), &config);
    ///  let (batch_size, sequence_length) = (64, 128);
    ///  let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    ///  let mask = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    ///  let token_type_ids = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    ///  let position_ids = Tensor::arange(sequence_length, (Int64, device)).expand(&[batch_size, sequence_length], true);
    ///
    ///  let model_output = no_grad(|| {
    ///    albert_model
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
    ) -> AlbertTokenClassificationOutput {
        let base_model_output = self
            .albert
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
        AlbertTokenClassificationOutput {
            logits,
            all_hidden_states: base_model_output.all_hidden_states,
            all_attentions: base_model_output.all_attentions,
        }
    }
}

/// # ALBERT for question answering
/// Extractive question-answering model based on a ALBERT language model. Identifies the segment of a context that answers a provided question.
/// Please note that a significant amount of pre- and post-processing is required to perform end-to-end question answering.
/// See the question answering pipeline (also provided in this crate) for more details.
/// It is made of the following blocks:
/// - `albert`: Base AlbertModel
/// - `qa_outputs`: Linear layer for question answering
pub struct AlbertForQuestionAnswering {
    albert: AlbertModel,
    qa_outputs: nn::Linear,
}

impl AlbertForQuestionAnswering {
    /// Build a new `AlbertForQuestionAnswering`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the ALBERT model
    /// * `config` - `AlbertConfig` object defining the model architecture and decoder status
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::albert::{AlbertConfig, AlbertForQuestionAnswering};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = AlbertConfig::from_file(config_path);
    /// let albert: AlbertForQuestionAnswering = AlbertForQuestionAnswering::new(&p.root(), &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &AlbertConfig) -> AlbertForQuestionAnswering
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let albert = AlbertModel::new(p / "albert", config);
        let num_labels = 2;
        let qa_outputs = nn::linear(
            p / "qa_outputs",
            config.hidden_size,
            num_labels,
            Default::default(),
        );

        AlbertForQuestionAnswering { albert, qa_outputs }
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
    /// * `AlbertQuestionAnsweringOutput` containing:
    ///   - `start_logits` - `Tensor` of shape (*batch size*, *sequence_length*) containing the logits for start of the answer
    ///   - `end_logits` - `Tensor` of shape (*batch size*, *sequence_length*) containing the logits for end of the answer
    ///   - `all_hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `all_attentions` - `Option<Vec<Vec<Tensor>>>` of length *num_hidden_layers* of nested length *inner_group_num* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use tch::{nn, Device, Tensor, no_grad};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::Int64;
    /// use rust_bert::albert::{AlbertConfig, AlbertForQuestionAnswering};
    /// # let config_path = Path::new("path/to/config.json");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = AlbertConfig::from_file(config_path);
    /// # let albert_model: AlbertForQuestionAnswering = AlbertForQuestionAnswering::new(&vs.root(), &config);
    ///  let (batch_size, sequence_length) = (64, 128);
    ///  let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    ///  let mask = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    ///  let token_type_ids = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    ///  let position_ids = Tensor::arange(sequence_length, (Int64, device)).expand(&[batch_size, sequence_length], true);
    ///
    ///  let model_output = no_grad(|| {
    ///    albert_model
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
    ) -> AlbertQuestionAnsweringOutput {
        let base_model_output = self
            .albert
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
            .apply(&self.qa_outputs)
            .split(1, -1);
        let (start_logits, end_logits) = (&logits[0], &logits[1]);
        let start_logits = start_logits.squeeze_dim(-1);
        let end_logits = end_logits.squeeze_dim(-1);

        AlbertQuestionAnsweringOutput {
            start_logits,
            end_logits,
            all_hidden_states: base_model_output.all_hidden_states,
            all_attentions: base_model_output.all_attentions,
        }
    }
}

/// # ALBERT for multiple choices
/// Multiple choices model using a ALBERT base model and a linear classifier.
/// Input should be in the form `[CLS] Context [SEP] Possible choice [SEP]`. The choice is made along the batch axis,
/// assuming all elements of the batch are alternatives to be chosen from for a given context.
/// It is made of the following blocks:
/// - `albert`: Base AlbertModel
/// - `dropout`: Dropout for hidden states output
/// - `classifier`: Linear layer for multiple choices
pub struct AlbertForMultipleChoice {
    albert: AlbertModel,
    dropout: Dropout,
    classifier: nn::Linear,
}

impl AlbertForMultipleChoice {
    /// Build a new `AlbertForMultipleChoice`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the ALBERT model
    /// * `config` - `AlbertConfig` object defining the model architecture and decoder status
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::albert::{AlbertConfig, AlbertForMultipleChoice};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = AlbertConfig::from_file(config_path);
    /// let albert: AlbertForMultipleChoice = AlbertForMultipleChoice::new(&p.root(), &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &AlbertConfig) -> AlbertForMultipleChoice
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let albert = AlbertModel::new(p / "albert", config);
        let dropout = Dropout::new(config.hidden_dropout_prob);
        let num_labels = 1;
        let classifier = nn::linear(
            p / "classifier",
            config.hidden_size,
            num_labels,
            Default::default(),
        );

        AlbertForMultipleChoice {
            albert,
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
    /// * `AlbertSequenceClassificationOutput` containing:
    ///   - `logits` - `Tensor` of shape (*1*, *batch size*) containing the logits for each of the alternatives given
    ///   - `all_hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `all_attentions` - `Option<Vec<Vec<Tensor>>>` of length *num_hidden_layers* of nested length *inner_group_num* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use tch::{nn, Device, Tensor, no_grad};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::Int64;
    /// use rust_bert::albert::{AlbertConfig, AlbertForMultipleChoice};
    /// # let config_path = Path::new("path/to/config.json");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = AlbertConfig::from_file(config_path);
    /// # let albert_model: AlbertForMultipleChoice = AlbertForMultipleChoice::new(&vs.root(), &config);
    ///  let (batch_size, sequence_length) = (64, 128);
    ///  let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    ///  let mask = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    ///  let token_type_ids = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    ///  let position_ids = Tensor::arange(sequence_length, (Int64, device)).expand(&[batch_size, sequence_length], true);
    ///
    ///  let model_output = no_grad(|| {
    ///    albert_model
    ///         .forward_t(Some(&input_tensor),
    ///                    Some(&mask),
    ///                    Some(&token_type_ids),
    ///                    Some(&position_ids),
    ///                    None,
    ///                    false).unwrap()
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
    ) -> Result<AlbertSequenceClassificationOutput, RustBertError> {
        let (input_ids, input_embeds, num_choices) = match &input_ids {
            Some(input_value) => match &input_embeds {
                Some(_) => {
                    return Err(RustBertError::ValueError(
                        "Only one of input ids or input embeddings may be set".into(),
                    ));
                }
                None => (
                    Some(input_value.view((-1, *input_value.size().last().unwrap()))),
                    None,
                    input_value.size()[1],
                ),
            },
            None => match &input_embeds {
                Some(embeds) => (
                    None,
                    Some(embeds.view((-1, embeds.size()[1], embeds.size()[2]))),
                    embeds.size()[1],
                ),
                None => {
                    return Err(RustBertError::ValueError(
                        "At least one of input ids or input embeddings must be set".into(),
                    ));
                }
            },
        };

        let mask = mask.map(|tensor| tensor.view((-1, *tensor.size().last().unwrap())));
        let token_type_ids =
            token_type_ids.map(|tensor| tensor.view((-1, *tensor.size().last().unwrap())));
        let position_ids =
            position_ids.map(|tensor| tensor.view((-1, *tensor.size().last().unwrap())));

        let base_model_output = self.albert.forward_t(
            input_ids.as_ref(),
            mask.as_ref(),
            token_type_ids.as_ref(),
            position_ids.as_ref(),
            input_embeds.as_ref(),
            train,
        )?;
        let logits = base_model_output
            .pooled_output
            .apply_t(&self.dropout, train)
            .apply(&self.classifier)
            .view((-1, num_choices));

        Ok(AlbertSequenceClassificationOutput {
            logits,
            all_hidden_states: base_model_output.all_hidden_states,
            all_attentions: base_model_output.all_attentions,
        })
    }
}

/// Container for the ALBERT model output.
pub struct AlbertOutput {
    /// Last hidden states from the model
    pub hidden_state: Tensor,
    /// Pooled output (hidden state for the first token)
    pub pooled_output: Tensor,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all intermediate layers
    pub all_attentions: Option<Vec<Vec<Tensor>>>,
}

/// Container for the ALBERT masked LM model output.
pub struct AlbertMaskedLMOutput {
    /// Logits for the vocabulary items at each sequence position
    pub prediction_scores: Tensor,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all intermediate layers
    pub all_attentions: Option<Vec<Vec<Tensor>>>,
}

/// Container for the ALBERT sequence classification model
pub struct AlbertSequenceClassificationOutput {
    /// Logits for each input (sequence) for each target class
    pub logits: Tensor,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all intermediate layers
    pub all_attentions: Option<Vec<Vec<Tensor>>>,
}

/// Container for the ALBERT token classification model
pub struct AlbertTokenClassificationOutput {
    /// Logits for each sequence item (token) for each target class
    pub logits: Tensor,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all intermediate layers
    pub all_attentions: Option<Vec<Vec<Tensor>>>,
}

/// Container for the ALBERT question answering model
pub struct AlbertQuestionAnsweringOutput {
    /// Logits for the start position for token of each input sequence
    pub start_logits: Tensor,
    /// Logits for the end position for token of each input sequence
    pub end_logits: Tensor,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all intermediate layers
    pub all_attentions: Option<Vec<Vec<Tensor>>>,
}
