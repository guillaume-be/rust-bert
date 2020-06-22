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


use std::collections::HashMap;
use crate::Config;
use serde::{Deserialize, Serialize};
use crate::albert::embeddings::AlbertEmbeddings;
use crate::albert::encoder::AlbertTransformer;
use tch::{nn, Tensor, Kind};
use crate::common::activations::{_tanh, _gelu_new, _gelu, _relu, _mish};
use tch::nn::Module;
use crate::common::dropout::Dropout;

/// # ALBERT Pretrained model weight files
pub struct AlbertModelResources;

/// # ALBERT Pretrained model config files
pub struct AlbertConfigResources;

/// # ALBERT Pretrained model vocab files
pub struct AlbertVocabResources;

impl AlbertModelResources {
    /// Shared under Apache 2.0 license by the Google team at https://github.com/google-research/ALBERT. Modified with conversion to C-array format.
    pub const ALBERT_BASE_V2: (&'static str, &'static str) = ("albert-base-v2/model.ot", "https://cdn.huggingface.co/albert-base-v2/rust_model.ot");
}

impl AlbertConfigResources {
    /// Shared under Apache 2.0 license by the Google team at https://github.com/google-research/ALBERT. Modified with conversion to C-array format.
    pub const ALBERT_BASE_V2: (&'static str, &'static str) = ("albert-base-v2/config.json", "https://cdn.huggingface.co/albert-base-v2-config.json");
}

impl AlbertVocabResources {
    /// Shared under Apache 2.0 license by the Google team at https://github.com/google-research/ALBERT. Modified with conversion to C-array format.
    pub const ALBERT_BASE_V2: (&'static str, &'static str) = ("albert-base-v2/spiece.model", "https://cdn.huggingface.co/albert-base-v2-spiece.model");
}


#[allow(non_camel_case_types)]
#[derive(Clone, Debug, Serialize, Deserialize)]
/// # Activation function used in the attention layer and masked language model head
pub enum Activation {
    /// Gaussian Error Linear Unit ([Hendrycks et al., 2016,](https://arxiv.org/abs/1606.08415))
    gelu_new,
    /// Gaussian Error Linear Unit ([Hendrycks et al., 2016,](https://arxiv.org/abs/1606.08415))
    gelu,
    /// Rectified Linear Unit
    relu,
    /// Mish ([Misra, 2019](https://arxiv.org/abs/1908.08681))
    mish,
}


#[derive(Debug, Serialize, Deserialize)]
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

impl Config<AlbertConfig> for AlbertConfig {}

/// # ALBERT Base model
/// Base architecture for ALBERT models. Task-specific models will be built from this common base model
/// It is made of the following blocks:
/// - `embeddings`: `token`, `position` and `segment_id` embeddings
/// - `encoder`: Encoder (transformer) made of a vector of layers. Each layer is made of a self-attention layer, an intermediate (linear) and output (linear + layer norm) layers. Note that the weights are shared across layers, allowing for a reduction in the model memory footprint.
/// - `pooler`: linear layer applied to the first element of the sequence (*[MASK]* token)
/// - `pooler_activation`: Tanh activation function for the pooling layer
pub struct AlbertModel {
    embeddings: AlbertEmbeddings,
    encoder: AlbertTransformer,
    pooler: nn::Linear,
    pooler_activation: Box<dyn Fn(&Tensor) -> Tensor>,
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
    /// use tch::{nn, Device};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use rust_bert::albert::{AlbertConfig, AlbertModel};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = AlbertConfig::from_file(config_path);
    /// let albert: AlbertModel = AlbertModel::new(&(&p.root() / "albert"), &config);
    /// ```
    ///
    pub fn new(p: &nn::Path, config: &AlbertConfig) -> AlbertModel {
        let embeddings = AlbertEmbeddings::new(&(p / "embeddings"), config);
        let encoder = AlbertTransformer::new(&(p / "encoder"), config);
        let pooler = nn::linear(&(p / "pooler"), config.hidden_size, config.hidden_size, Default::default());
        let pooler_activation = Box::new(_tanh);

        AlbertModel { embeddings, encoder, pooler, pooler_activation }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see `input_embeds`)
    /// * `mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `token_type_ids` - Optional segment id of shape (*batch size*, *sequence_length*). Convention is value of 0 for the first sentence (incl. *[SEP]*) and 1 for the second sentence. If None set to 0.
    /// * `position_ids` - Optional position ids of shape (*batch size*, *sequence_length*). If None, will be incremented from 0.
    /// * `input_embeds` - Optional pre-computed input embeddings of shape (*batch size*, *sequence_length*, *hidden_size*). If None, input ids must be provided (see `input_ids`)
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `output` - `Tensor` of shape (*batch size*, *sequence_length*, *hidden_size*)
    /// * `pooled_output` - `Tensor` of shape (*batch size*, *hidden_size*)
    /// * `hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    /// * `attentions` - `Option<Vec<Vec<Tensor>>>` of length *num_hidden_layers* of nested length *inner_group_num* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    ///# use tch::{nn, Device, Tensor, no_grad};
    ///# use rust_bert::Config;
    ///# use std::path::Path;
    ///# use tch::kind::Kind::Int64;
    /// use rust_bert::albert::{AlbertConfig, AlbertModel};
    ///# let config_path = Path::new("path/to/config.json");
    ///# let device = Device::Cpu;
    ///# let vs = nn::VarStore::new(device);
    ///# let config = AlbertConfig::from_file(config_path);
    ///# let albert_model: AlbertModel = AlbertModel::new(&vs.root(), &config);
    ///  let (batch_size, sequence_length) = (64, 128);
    ///  let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    ///  let mask = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    ///  let token_type_ids = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    ///  let position_ids = Tensor::arange(sequence_length, (Int64, device)).expand(&[batch_size, sequence_length], true);
    ///
    ///  let (output, pooled_output, all_hidden_states, all_attentions) = no_grad(|| {
    ///    albert_model
    ///         .forward_t(Some(input_tensor),
    ///                    Some(mask),
    ///                    Some(token_type_ids),
    ///                    Some(position_ids),
    ///                    None,
    ///                    false).unwrap()
    ///    });
    ///
    /// ```
    ///
    pub fn forward_t(&self,
                     input_ids: Option<Tensor>,
                     mask: Option<Tensor>,
                     token_type_ids: Option<Tensor>,
                     position_ids: Option<Tensor>,
                     input_embeds: Option<Tensor>,
                     train: bool)
                     -> Result<(Tensor, Tensor, Option<Vec<Tensor>>, Option<Vec<Vec<Tensor>>>), &'static str> {
        let (input_shape, device) = match &input_ids {
            Some(input_value) => match &input_embeds {
                Some(_) => { return Err("Only one of input ids or input embeddings may be set"); }
                None => (input_value.size(), input_value.device())
            }
            None => match &input_embeds {
                Some(embeds) => (vec!(embeds.size()[0], embeds.size()[1]), embeds.device()),
                None => { return Err("At least one of input ids or input embeddings must be set"); }
            }
        };

        let mask = match mask {
            Some(value) => value,
            None => Tensor::ones(&input_shape, (Kind::Int64, device))
        };

        let extended_attention_mask = mask.unsqueeze(1).unsqueeze(2);
        let extended_attention_mask: Tensor = (extended_attention_mask.ones_like() - extended_attention_mask) * -10000.0;

        let embedding_output = match self.embeddings.forward_t(input_ids, token_type_ids, position_ids, input_embeds, train) {
            Ok(value) => value,
            Err(e) => { return Err(e); }
        };

        let (hidden_state, all_hidden_states, all_attentions) =
            self.encoder.forward_t(&embedding_output,
                                   Some(extended_attention_mask),
                                   train);

        let pooled_output = self.pooler.forward(&hidden_state.select(1, 0));
        let pooled_output = (self.pooler_activation)(&pooled_output);

        Ok((hidden_state, pooled_output, all_hidden_states, all_attentions))
    }
}

pub struct AlbertMLMHead {
    layer_norm: nn::LayerNorm,
    dense: nn::Linear,
    decoder: nn::Linear,
    activation: Box<dyn Fn(&Tensor) -> Tensor>,
}

impl AlbertMLMHead {
    pub fn new(p: &nn::Path, config: &AlbertConfig) -> AlbertMLMHead {
        let layer_norm_eps = match config.layer_norm_eps {
            Some(value) => value,
            None => 1e-12
        };
        let layer_norm_config = nn::LayerNormConfig { eps: layer_norm_eps, ..Default::default() };
        let layer_norm = nn::layer_norm(&(p / "LayerNorm"), vec![config.embedding_size], layer_norm_config);
        let dense = nn::linear(&(p / "dense"), config.hidden_size, config.embedding_size, Default::default());
        let decoder = nn::linear(&(p / "decoder"), config.embedding_size, config.vocab_size, Default::default());

        let activation = Box::new(match &config.hidden_act {
            Activation::gelu_new => _gelu_new,
            Activation::gelu => _gelu,
            Activation::relu => _relu,
            Activation::mish => _mish
        });

        AlbertMLMHead { layer_norm, dense, decoder, activation }
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Tensor {
        let output: Tensor = (self.activation)(&hidden_states.apply(&self.dense));
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
    /// use tch::{nn, Device};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use rust_bert::albert::{AlbertConfig, AlbertForMaskedLM};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = AlbertConfig::from_file(config_path);
    /// let albert: AlbertForMaskedLM = AlbertForMaskedLM::new(&p.root(), &config);
    /// ```
    ///
    pub fn new(p: &nn::Path, config: &AlbertConfig) -> AlbertForMaskedLM {
        let albert = AlbertModel::new(&(p / "albert"), config);
        let predictions = AlbertMLMHead::new(&(p / "predictions"), config);

        AlbertForMaskedLM { albert, predictions }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see `input_embeds`)
    /// * `mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `token_type_ids` - Optional segment id of shape (*batch size*, *sequence_length*). Convention is value of 0 for the first sentence (incl. *[SEP]*) and 1 for the second sentence. If None set to 0.
    /// * `position_ids` - Optional position ids of shape (*batch size*, *sequence_length*). If None, will be incremented from 0.
    /// * `input_embeds` - Optional pre-computed input embeddings of shape (*batch size*, *sequence_length*, *hidden_size*). If None, input ids must be provided (see `input_ids`)
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `output` - `Tensor` of shape (*batch size*, *sequence_length*, *vocab_size*)
    /// * `hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    /// * `attentions` - `Option<Vec<Vec<Tensor>>>` of length *num_hidden_layers* of nested length *inner_group_num* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    ///# use tch::{nn, Device, Tensor, no_grad};
    ///# use rust_bert::Config;
    ///# use std::path::Path;
    ///# use tch::kind::Kind::Int64;
    /// use rust_bert::albert::{AlbertConfig, AlbertForMaskedLM};
    ///# let config_path = Path::new("path/to/config.json");
    ///# let device = Device::Cpu;
    ///# let vs = nn::VarStore::new(device);
    ///# let config = AlbertConfig::from_file(config_path);
    ///# let albert_model: AlbertForMaskedLM = AlbertForMaskedLM::new(&vs.root(), &config);
    ///  let (batch_size, sequence_length) = (64, 128);
    ///  let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    ///  let mask = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    ///  let token_type_ids = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    ///  let position_ids = Tensor::arange(sequence_length, (Int64, device)).expand(&[batch_size, sequence_length], true);
    ///
    ///  let (output, all_hidden_states, all_attentions) = no_grad(|| {
    ///    albert_model
    ///         .forward_t(Some(input_tensor),
    ///                    Some(mask),
    ///                    Some(token_type_ids),
    ///                    Some(position_ids),
    ///                    None,
    ///                    false)
    ///    });
    ///
    /// ```
    ///
    pub fn forward_t(&self,
                     input_ids: Option<Tensor>,
                     mask: Option<Tensor>,
                     token_type_ids: Option<Tensor>,
                     position_ids: Option<Tensor>,
                     input_embeds: Option<Tensor>,
                     train: bool) -> (Tensor, Option<Vec<Tensor>>, Option<Vec<Vec<Tensor>>>) {
        let (hidden_state, _, all_hidden_states, all_attentions) = self.albert.forward_t(input_ids, mask, token_type_ids, position_ids, input_embeds, train).unwrap();
        let prediction_scores = self.predictions.forward(&hidden_state);
        (prediction_scores, all_hidden_states, all_attentions)
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
    /// use tch::{nn, Device};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use rust_bert::albert::{AlbertConfig, AlbertForSequenceClassification};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = AlbertConfig::from_file(config_path);
    /// let albert: AlbertForSequenceClassification = AlbertForSequenceClassification::new(&p.root(), &config);
    /// ```
    ///
    pub fn new(p: &nn::Path, config: &AlbertConfig) -> AlbertForSequenceClassification {
        let albert = AlbertModel::new(&(p / "albert"), config);
        let classifier_dropout_prob = match config.classifier_dropout_prob {
            Some(value) => value,
            None => 0.1
        };
        let dropout = Dropout::new(classifier_dropout_prob);
        let num_labels = config.id2label.as_ref().expect("num_labels not provided in configuration").len() as i64;
        let classifier = nn::linear(&(p / "classifier"), config.hidden_size, num_labels, Default::default());

        AlbertForSequenceClassification { albert, dropout, classifier }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see `input_embeds`)
    /// * `mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `token_type_ids` - Optional segment id of shape (*batch size*, *sequence_length*). Convention is value of 0 for the first sentence (incl. *[SEP]*) and 1 for the second sentence. If None set to 0.
    /// * `position_ids` - Optional position ids of shape (*batch size*, *sequence_length*). If None, will be incremented from 0.
    /// * `input_embeds` - Optional pre-computed input embeddings of shape (*batch size*, *sequence_length*, *hidden_size*). If None, input ids must be provided (see `input_ids`)
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `output` - `Tensor` of shape (*batch size*, *num_labels*)
    /// * `hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    /// * `attentions` - `Option<Vec<Vec<Tensor>>>` of length *num_hidden_layers* of nested length *inner_group_num* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    ///# use tch::{nn, Device, Tensor, no_grad};
    ///# use rust_bert::Config;
    ///# use std::path::Path;
    ///# use tch::kind::Kind::Int64;
    /// use rust_bert::albert::{AlbertConfig, AlbertForSequenceClassification};
    ///# let config_path = Path::new("path/to/config.json");
    ///# let device = Device::Cpu;
    ///# let vs = nn::VarStore::new(device);
    ///# let config = AlbertConfig::from_file(config_path);
    ///# let albert_model: AlbertForSequenceClassification = AlbertForSequenceClassification::new(&vs.root(), &config);
    ///  let (batch_size, sequence_length) = (64, 128);
    ///  let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    ///  let mask = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    ///  let token_type_ids = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    ///  let position_ids = Tensor::arange(sequence_length, (Int64, device)).expand(&[batch_size, sequence_length], true);
    ///
    ///  let (output, all_hidden_states, all_attentions) = no_grad(|| {
    ///    albert_model
    ///         .forward_t(Some(input_tensor),
    ///                    Some(mask),
    ///                    Some(token_type_ids),
    ///                    Some(position_ids),
    ///                    None,
    ///                    false)
    ///    });
    ///
    /// ```
    ///
    pub fn forward_t(&self,
                     input_ids: Option<Tensor>,
                     mask: Option<Tensor>,
                     token_type_ids: Option<Tensor>,
                     position_ids: Option<Tensor>,
                     input_embeds: Option<Tensor>,
                     train: bool) -> (Tensor, Option<Vec<Tensor>>, Option<Vec<Vec<Tensor>>>) {
        let (_, pooled_output, all_hidden_states, all_attentions) = self.albert.forward_t(input_ids, mask, token_type_ids, position_ids, input_embeds, train).unwrap();
        let logits = pooled_output.apply_t(&self.dropout, train).apply(&self.classifier);
        (logits, all_hidden_states, all_attentions)
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
    /// use tch::{nn, Device};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use rust_bert::albert::{AlbertConfig, AlbertForTokenClassification};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = AlbertConfig::from_file(config_path);
    /// let albert: AlbertForTokenClassification = AlbertForTokenClassification::new(&p.root(), &config);
    /// ```
    ///
    pub fn new(p: &nn::Path, config: &AlbertConfig) -> AlbertForTokenClassification {
        let albert = AlbertModel::new(&(p / "albert"), config);
        let dropout = Dropout::new(config.hidden_dropout_prob);
        let num_labels = config.id2label.as_ref().expect("num_labels not provided in configuration").len() as i64;
        let classifier = nn::linear(&(p / "classifier"), config.hidden_size, num_labels, Default::default());

        AlbertForTokenClassification { albert, dropout, classifier }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see `input_embeds`)
    /// * `mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `token_type_ids` - Optional segment id of shape (*batch size*, *sequence_length*). Convention is value of 0 for the first sentence (incl. *[SEP]*) and 1 for the second sentence. If None set to 0.
    /// * `position_ids` - Optional position ids of shape (*batch size*, *sequence_length*). If None, will be incremented from 0.
    /// * `input_embeds` - Optional pre-computed input embeddings of shape (*batch size*, *sequence_length*, *hidden_size*). If None, input ids must be provided (see `input_ids`)
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `output` - `Tensor` of shape (*batch size*, *sequence_length*, *num_labels*) containing the logits for each of the input tokens and classes
    /// * `hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    /// * `attentions` - `Option<Vec<Vec<Tensor>>>` of length *num_hidden_layers* of nested length *inner_group_num* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    ///# use tch::{nn, Device, Tensor, no_grad};
    ///# use rust_bert::Config;
    ///# use std::path::Path;
    ///# use tch::kind::Kind::Int64;
    /// use rust_bert::albert::{AlbertConfig, AlbertForTokenClassification};
    ///# let config_path = Path::new("path/to/config.json");
    ///# let device = Device::Cpu;
    ///# let vs = nn::VarStore::new(device);
    ///# let config = AlbertConfig::from_file(config_path);
    ///# let albert_model: AlbertForTokenClassification = AlbertForTokenClassification::new(&vs.root(), &config);
    ///  let (batch_size, sequence_length) = (64, 128);
    ///  let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    ///  let mask = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    ///  let token_type_ids = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    ///  let position_ids = Tensor::arange(sequence_length, (Int64, device)).expand(&[batch_size, sequence_length], true);
    ///
    ///  let (output, all_hidden_states, all_attentions) = no_grad(|| {
    ///    albert_model
    ///         .forward_t(Some(input_tensor),
    ///                    Some(mask),
    ///                    Some(token_type_ids),
    ///                    Some(position_ids),
    ///                    None,
    ///                    false)
    ///    });
    ///
    /// ```
    ///
    pub fn forward_t(&self,
                     input_ids: Option<Tensor>,
                     mask: Option<Tensor>,
                     token_type_ids: Option<Tensor>,
                     position_ids: Option<Tensor>,
                     input_embeds: Option<Tensor>,
                     train: bool) -> (Tensor, Option<Vec<Tensor>>, Option<Vec<Vec<Tensor>>>) {
        let (sequence_output, _, all_hidden_states, all_attentions) = self.albert.forward_t(input_ids, mask, token_type_ids, position_ids, input_embeds, train).unwrap();
        let logits = sequence_output.apply_t(&self.dropout, train).apply(&self.classifier);
        (logits, all_hidden_states, all_attentions)
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
    /// use tch::{nn, Device};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use rust_bert::albert::{AlbertConfig, AlbertForQuestionAnswering};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = AlbertConfig::from_file(config_path);
    /// let albert: AlbertForQuestionAnswering = AlbertForQuestionAnswering::new(&p.root(), &config);
    /// ```
    ///
    pub fn new(p: &nn::Path, config: &AlbertConfig) -> AlbertForQuestionAnswering {
        let albert = AlbertModel::new(&(p / "albert"), config);
        let num_labels = 2;
        let qa_outputs = nn::linear(&(p / "qa_outputs"), config.hidden_size, num_labels, Default::default());

        AlbertForQuestionAnswering { albert, qa_outputs }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see `input_embeds`)
    /// * `mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `token_type_ids` - Optional segment id of shape (*batch size*, *sequence_length*). Convention is value of 0 for the first sentence (incl. *[SEP]*) and 1 for the second sentence. If None set to 0.
    /// * `position_ids` - Optional position ids of shape (*batch size*, *sequence_length*). If None, will be incremented from 0.
    /// * `input_embeds` - Optional pre-computed input embeddings of shape (*batch size*, *sequence_length*, *hidden_size*). If None, input ids must be provided (see `input_ids`)
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `start_scores` - `Tensor` of shape (*batch size*, *sequence_length*) containing the logits for start of the answer
    /// * `end_scores` - `Tensor` of shape (*batch size*, *sequence_length*) containing the logits for end of the answer
    /// * `hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    /// * `attentions` - `Option<Vec<Vec<Tensor>>>` of length *num_hidden_layers* of nested length *inner_group_num* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    ///# use tch::{nn, Device, Tensor, no_grad};
    ///# use rust_bert::Config;
    ///# use std::path::Path;
    ///# use tch::kind::Kind::Int64;
    /// use rust_bert::albert::{AlbertConfig, AlbertForQuestionAnswering};
    ///# let config_path = Path::new("path/to/config.json");
    ///# let device = Device::Cpu;
    ///# let vs = nn::VarStore::new(device);
    ///# let config = AlbertConfig::from_file(config_path);
    ///# let albert_model: AlbertForQuestionAnswering = AlbertForQuestionAnswering::new(&vs.root(), &config);
    ///  let (batch_size, sequence_length) = (64, 128);
    ///  let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    ///  let mask = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    ///  let token_type_ids = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    ///  let position_ids = Tensor::arange(sequence_length, (Int64, device)).expand(&[batch_size, sequence_length], true);
    ///
    ///  let (start_logits, end_logits, all_hidden_states, all_attentions) = no_grad(|| {
    ///    albert_model
    ///         .forward_t(Some(input_tensor),
    ///                    Some(mask),
    ///                    Some(token_type_ids),
    ///                    Some(position_ids),
    ///                    None,
    ///                    false)
    ///    });
    ///
    /// ```
    ///
    pub fn forward_t(&self,
                     input_ids: Option<Tensor>,
                     mask: Option<Tensor>,
                     token_type_ids: Option<Tensor>,
                     position_ids: Option<Tensor>,
                     input_embeds: Option<Tensor>,
                     train: bool) -> (Tensor, Tensor, Option<Vec<Tensor>>, Option<Vec<Vec<Tensor>>>) {
        let (sequence_output, _, all_hidden_states, all_attentions) = self.albert.forward_t(input_ids, mask, token_type_ids, position_ids, input_embeds, train).unwrap();
        let logits = sequence_output.apply(&self.qa_outputs).split(1, -1);
        let (start_logits, end_logits) = (&logits[0], &logits[1]);
        let start_logits = start_logits.squeeze1(-1);
        let end_logits = end_logits.squeeze1(-1);

        (start_logits, end_logits, all_hidden_states, all_attentions)
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
    /// use tch::{nn, Device};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use rust_bert::albert::{AlbertConfig, AlbertForMultipleChoice};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = AlbertConfig::from_file(config_path);
    /// let albert: AlbertForMultipleChoice = AlbertForMultipleChoice::new(&p.root(), &config);
    /// ```
    ///
    pub fn new(p: &nn::Path, config: &AlbertConfig) -> AlbertForMultipleChoice {
        let albert = AlbertModel::new(&(p / "albert"), config);
        let dropout = Dropout::new(config.hidden_dropout_prob);
        let num_labels = 1;
        let classifier = nn::linear(&(p / "classifier"), config.hidden_size, num_labels, Default::default());

        AlbertForMultipleChoice { albert, dropout, classifier }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see `input_embeds`)
    /// * `mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `token_type_ids` - Optional segment id of shape (*batch size*, *sequence_length*). Convention is value of 0 for the first sentence (incl. *[SEP]*) and 1 for the second sentence. If None set to 0.
    /// * `position_ids` - Optional position ids of shape (*batch size*, *sequence_length*). If None, will be incremented from 0.
    /// * `input_embeds` - Optional pre-computed input embeddings of shape (*batch size*, *sequence_length*, *hidden_size*). If None, input ids must be provided (see `input_ids`)
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `output` - `Tensor` of shape (*1*, *batch size*) containing the logits for each of the alternatives given
    /// * `hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    /// * `attentions` - `Option<Vec<Vec<Tensor>>>` of length *num_hidden_layers* of nested length *inner_group_num* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    ///# use tch::{nn, Device, Tensor, no_grad};
    ///# use rust_bert::Config;
    ///# use std::path::Path;
    ///# use tch::kind::Kind::Int64;
    /// use rust_bert::albert::{AlbertConfig, AlbertForMultipleChoice};
    ///# let config_path = Path::new("path/to/config.json");
    ///# let device = Device::Cpu;
    ///# let vs = nn::VarStore::new(device);
    ///# let config = AlbertConfig::from_file(config_path);
    ///# let albert_model: AlbertForMultipleChoice = AlbertForMultipleChoice::new(&vs.root(), &config);
    ///  let (batch_size, sequence_length) = (64, 128);
    ///  let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    ///  let mask = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    ///  let token_type_ids = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    ///  let position_ids = Tensor::arange(sequence_length, (Int64, device)).expand(&[batch_size, sequence_length], true);
    ///
    ///  let (output, all_hidden_states, all_attentions) = no_grad(|| {
    ///    albert_model
    ///         .forward_t(Some(input_tensor),
    ///                    Some(mask),
    ///                    Some(token_type_ids),
    ///                    Some(position_ids),
    ///                    None,
    ///                    false).unwrap()
    ///    });
    ///
    /// ```
    ///
    pub fn forward_t(&self,
                     input_ids: Option<Tensor>,
                     mask: Option<Tensor>,
                     token_type_ids: Option<Tensor>,
                     position_ids: Option<Tensor>,
                     input_embeds: Option<Tensor>,
                     train: bool) -> Result<(Tensor, Option<Vec<Tensor>>, Option<Vec<Vec<Tensor>>>), &'static str> {
        let (input_ids, input_embeds, num_choices) = match &input_ids {
            Some(input_value) => match &input_embeds {
                Some(_) => { return Err("Only one of input ids or input embeddings may be set"); }
                None => (Some(input_value.view((-1, *input_value.size().last().unwrap()))), None, input_value.size()[1])
            }
            None => match &input_embeds {
                Some(embeds) => (None, Some(embeds.view((-1, embeds.size()[1], embeds.size()[2]))), embeds.size()[1]),
                None => { return Err("At least one of input ids or input embeddings must be set"); }
            }
        };

        let mask = match mask {
            Some(value) => Some(value.view((-1, *value.size().last().unwrap()))),
            None => None
        };
        let token_type_ids = match token_type_ids {
            Some(value) => Some(value.view((-1, *value.size().last().unwrap()))),
            None => None
        };
        let position_ids = match position_ids {
            Some(value) => Some(value.view((-1, *value.size().last().unwrap()))),
            None => None
        };


        let (_, pooled_output, all_hidden_states, all_attentions) = self.albert.forward_t(input_ids, mask, token_type_ids, position_ids, input_embeds, train).unwrap();
        let logits = pooled_output.apply_t(&self.dropout, train).apply(&self.classifier).view((-1, num_choices));

        Ok((logits, all_hidden_states, all_attentions))
    }
}