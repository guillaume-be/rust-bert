// Copyright 2021 Google Research
// Copyright 2020-present, the HuggingFace Inc. team.
// Copyright 2021 Guillaume Becquin
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::common::activations::{TensorFunction, _tanh};
use crate::common::dropout::Dropout;
use crate::common::embeddings::get_shape_and_device_from_ids_embeddings_pair;
use crate::fnet::embeddings::FNetEmbeddings;
use crate::fnet::encoder::FNetEncoder;
use crate::{Activation, Config, RustBertError};
use serde::{Deserialize, Serialize};
use std::borrow::Borrow;
use std::collections::HashMap;
use tch::nn::LayerNormConfig;
use tch::{nn, Tensor};

/// # FNet Pretrained model weight files
pub struct FNetModelResources;

/// # FNet Pretrained model config files
pub struct FNetConfigResources;

/// # FNet Pretrained model vocab files
pub struct FNetVocabResources;

impl FNetModelResources {
    /// Shared under Apache 2.0 license by the Google team at <https://github.com/google-research/google-research/tree/master/f_net>. Modified with conversion to C-array format.
    pub const BASE: (&'static str, &'static str) = (
        "fnet-base/model",
        "https://huggingface.co/google/fnet-base/resolve/main/rust_model.ot",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/gchhablani/fnet-base-finetuned-sst2>. Modified with conversion to C-array format.
    pub const BASE_SST2: (&'static str, &'static str) = (
        "fnet-base-sst2/model",
        "https://huggingface.co/gchhablani/fnet-base-finetuned-sst2/resolve/main/rust_model.ot",
    );
}

impl FNetConfigResources {
    /// Shared under Apache 2.0 license by the Google team at <https://github.com/google-research/google-research/tree/master/f_net>. Modified with conversion to C-array format.
    pub const BASE: (&'static str, &'static str) = (
        "fnet-base/config",
        "https://huggingface.co/google/fnet-base/resolve/main/config.json",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/gchhablani/fnet-base-finetuned-sst2>. Modified with conversion to C-array format.
    pub const BASE_SST2: (&'static str, &'static str) = (
        "fnet-base-sst2/config",
        "https://huggingface.co/gchhablani/fnet-base-finetuned-sst2/resolve/main/config.json",
    );
}

impl FNetVocabResources {
    /// Shared under Apache 2.0 license by the Google team at <https://github.com/google-research/google-research/tree/master/f_net>. Modified with conversion to C-array format.
    pub const BASE: (&'static str, &'static str) = (
        "fnet-base/spiece",
        "https://huggingface.co/google/fnet-base/resolve/main/spiece.model",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/gchhablani/fnet-base-finetuned-sst2>. Modified with conversion to C-array format.
    pub const BASE_SST2: (&'static str, &'static str) = (
        "fnet-base-sst2/spiece",
        "https://huggingface.co/google/fnet-base/resolve/main/spiece.model",
    );
}

#[derive(Debug, Serialize, Deserialize)]
/// # FNet model configuration
/// Defines the FNet model architecture (e.g. number of layers, hidden layer size, label mapping...)
pub struct FNetConfig {
    pub vocab_size: i64,
    pub hidden_size: i64,
    pub num_hidden_layers: i64,
    pub intermediate_size: i64,
    pub hidden_act: Activation,
    pub hidden_dropout_prob: f64,
    pub max_position_embeddings: i64,
    pub type_vocab_size: i64,
    pub initializer_range: f64,
    pub layer_norm_eps: Option<f64>,
    pub pad_token_id: Option<i64>,
    pub bos_token_id: Option<i64>,
    pub eos_token_id: Option<i64>,
    pub id2label: Option<HashMap<i64, String>>,
    pub label2id: Option<HashMap<String, i64>>,
    pub output_attentions: Option<bool>,
    pub output_hidden_states: Option<bool>,
}

impl Config for FNetConfig {}

struct FNetPooler {
    dense: nn::Linear,
    activation: TensorFunction,
}

impl FNetPooler {
    pub fn new<'p, P>(p: P, config: &FNetConfig) -> FNetPooler
    where
        P: Borrow<nn::Path<'p>>,
    {
        let dense = nn::linear(
            p.borrow() / "dense",
            config.hidden_size,
            config.hidden_size,
            Default::default(),
        );
        let activation = TensorFunction::new(Box::new(_tanh));

        FNetPooler { dense, activation }
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Tensor {
        self.activation.get_fn()(&hidden_states.select(1, 0).apply(&self.dense))
    }
}

struct FNetPredictionHeadTransform {
    dense: nn::Linear,
    activation: TensorFunction,
    layer_norm: nn::LayerNorm,
}

impl FNetPredictionHeadTransform {
    pub fn new<'p, P>(p: P, config: &FNetConfig) -> FNetPredictionHeadTransform
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
        let activation = config.hidden_act.get_function();
        let layer_norm_config = LayerNormConfig {
            eps: config.layer_norm_eps.unwrap_or(1e-12),
            ..Default::default()
        };
        let layer_norm =
            nn::layer_norm(p / "LayerNorm", vec![config.hidden_size], layer_norm_config);

        FNetPredictionHeadTransform {
            dense,
            activation,
            layer_norm,
        }
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Tensor {
        let hidden_states = hidden_states.apply(&self.dense);
        let hidden_states: Tensor = self.activation.get_fn()(&hidden_states);
        hidden_states.apply(&self.layer_norm)
    }
}

struct FNetLMPredictionHead {
    transform: FNetPredictionHeadTransform,
    decoder: nn::Linear,
}

impl FNetLMPredictionHead {
    pub fn new<'p, P>(p: P, config: &FNetConfig) -> FNetLMPredictionHead
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let transform = FNetPredictionHeadTransform::new(p / "transform", config);
        let decoder = nn::linear(
            p / "decoder",
            config.hidden_size,
            config.vocab_size,
            Default::default(),
        );

        FNetLMPredictionHead { transform, decoder }
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Tensor {
        self.transform.forward(hidden_states).apply(&self.decoder)
    }
}

/// # FNet Base model
/// Base architecture for FNet models. Task-specific models will be built from this common base model
/// It is made of the following blocks:
/// - `embeddings`: FNetEmbeddings combining word, position and segment embeddings
/// - `encoder`: `FNetEncoder` made of a stack of `FNetLayer`
/// - `pooler`: Optional `FNetPooler` taking the first sequence element hidden state for sequence-level tasks
pub struct FNetModel {
    embeddings: FNetEmbeddings,
    encoder: FNetEncoder,
    pooler: Option<FNetPooler>,
}

impl FNetModel {
    /// Build a new `FNetModel`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the FNet model
    /// * `config` - `FNetConfig` object defining the model architecture
    /// * `add_pooling_layer` - boolean flag indicating if a pooling layer should be added after the encoder
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::fnet::{FNetConfig, FNetModel};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = FNetConfig::from_file(config_path);
    /// let add_pooling_layer = true;
    /// let fnet = FNetModel::new(&p.root() / "fnet", &config, add_pooling_layer);
    /// ```
    pub fn new<'p, P>(p: P, config: &FNetConfig, add_pooling_layer: bool) -> FNetModel
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let embeddings = FNetEmbeddings::new(p / "embeddings", config);
        let encoder = FNetEncoder::new(p / "encoder", config);
        let pooler = if add_pooling_layer {
            Some(FNetPooler::new(p / "pooler", config))
        } else {
            None
        };

        FNetModel {
            embeddings,
            encoder,
            pooler,
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
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `FNetModelOutput` containing:
    ///   - `hidden_state` - `Tensor` of shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `pooled_output` - Optional `Tensor` of shape (*batch size*, *hidden_size*) if the model was created with an optional pooling layer
    ///   - `all_hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use tch::{nn, Device, Tensor, no_grad};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::Int64;
    /// use rust_bert::fnet::{FNetConfig, FNetModel};
    /// # let config_path = Path::new("path/to/config.json");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = FNetConfig::from_file(config_path);
    /// let add_pooling_layer = true;
    /// let model = FNetModel::new(&vs.root(), &config, add_pooling_layer);
    /// let (batch_size, sequence_length) = (64, 128);
    /// let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
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
        input_embeddings: Option<&Tensor>,
        train: bool,
    ) -> Result<FNetModelOutput, RustBertError> {
        let hidden_states = self.embeddings.forward_t(
            input_ids,
            token_type_ids,
            position_ids,
            input_embeddings,
            train,
        )?;

        let encoder_output = self.encoder.forward_t(&hidden_states, train);
        let pooled_output = if let Some(pooler) = &self.pooler {
            Some(pooler.forward(&encoder_output.hidden_states))
        } else {
            None
        };
        Ok(FNetModelOutput {
            hidden_states: encoder_output.hidden_states,
            pooled_output,
            all_hidden_states: encoder_output.all_hidden_states,
        })
    }
}

/// # FNet for masked language model
/// Base FNet model with a masked language model head to predict missing tokens, for example `"Looks like one [MASK] is missing" -> "person"`
/// It is made of the following blocks:
/// - `fnet`: Base FNet model
/// - `lm_head`: FNet LM prediction head
pub struct FNetForMaskedLM {
    fnet: FNetModel,
    lm_head: FNetLMPredictionHead,
}

impl FNetForMaskedLM {
    /// Build a new `FNetForMaskedLM`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the FNet model
    /// * `config` - `FNetConfig` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::fnet::{FNetConfig, FNetForMaskedLM};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = FNetConfig::from_file(config_path);
    /// let fnet = FNetForMaskedLM::new(&p.root() / "fnet", &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &FNetConfig) -> FNetForMaskedLM
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let fnet = FNetModel::new(p / "fnet", config, false);
        let lm_head = FNetLMPredictionHead::new(p.sub("cls").sub("predictions"), config);

        FNetForMaskedLM { fnet, lm_head }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see `input_embeds`)
    /// * `token_type_ids` - Optional segment id of shape (*batch size*, *sequence_length*). Convention is value of 0 for the first sentence (incl. *SEP*) and 1 for the second sentence. If None set to 0.
    /// * `position_ids` - Optional position ids of shape (*batch size*, *sequence_length*). If None, will be incremented from 0.
    /// * `input_embeds` - Optional pre-computed input embeddings of shape (*batch size*, *sequence_length*, *hidden_size*). If None, input ids must be provided (see `input_ids`)
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `FNetMaskedLMOutput` containing:
    ///   - `prediction_scores` - `Tensor` of shape (*batch size*, *sequence_length*, *vocab_size*)
    ///   - `all_hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use tch::{nn, Device, Tensor, no_grad};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::Int64;
    /// use rust_bert::fnet::{FNetConfig, FNetForMaskedLM};
    /// # let config_path = Path::new("path/to/config.json");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = FNetConfig::from_file(config_path);
    /// let model = FNetForMaskedLM::new(&vs.root(), &config);
    /// let (batch_size, sequence_length) = (64, 128);
    /// let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
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
        input_embeddings: Option<&Tensor>,
        train: bool,
    ) -> Result<FNetMaskedLMOutput, RustBertError> {
        let model_output = self.fnet.forward_t(
            input_ids,
            token_type_ids,
            position_ids,
            input_embeddings,
            train,
        )?;

        let prediction_scores = self.lm_head.forward(&model_output.hidden_states);

        Ok(FNetMaskedLMOutput {
            prediction_scores,
            all_hidden_states: model_output.all_hidden_states,
        })
    }
}

/// # FNet for sequence classification
/// Base FNet model with a classifier head to perform sentence or document-level classification
/// It is made of the following blocks:
/// - `fnet`: Base FNet model
/// - `dropout`: Dropout layer before the last linear layer
/// - `classifier`: linear layer mapping from hidden to the number of classes to predict
pub struct FNetForSequenceClassification {
    fnet: FNetModel,
    dropout: Dropout,
    classifier: nn::Linear,
}

impl FNetForSequenceClassification {
    /// Build a new `FNetForSequenceClassification`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the FNet model
    /// * `config` - `FNetConfig` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::fnet::{FNetConfig, FNetForSequenceClassification};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = FNetConfig::from_file(config_path);
    /// let fnet = FNetForSequenceClassification::new(&p.root() / "fnet", &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &FNetConfig) -> FNetForSequenceClassification
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let fnet = FNetModel::new(p / "fnet", config, true);
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

        FNetForSequenceClassification {
            fnet,
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
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `FNetSequenceClassificationOutput` containing:
    ///   - `logits` - `Tensor` of shape (*batch size*, *num_classes*)
    ///   - `all_hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use tch::{nn, Device, Tensor, no_grad};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::Int64;
    /// use rust_bert::fnet::{FNetConfig, FNetForSequenceClassification};
    /// # let config_path = Path::new("path/to/config.json");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = FNetConfig::from_file(config_path);
    /// let model = FNetForSequenceClassification::new(&vs.root(), &config);
    /// let (batch_size, sequence_length) = (64, 128);
    /// let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
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
        input_embeddings: Option<&Tensor>,
        train: bool,
    ) -> Result<FNetSequenceClassificationOutput, RustBertError> {
        let base_model_output = self.fnet.forward_t(
            input_ids,
            token_type_ids,
            position_ids,
            input_embeddings,
            train,
        )?;

        let logits = base_model_output
            .pooled_output
            .unwrap()
            .apply_t(&self.dropout, train)
            .apply(&self.classifier);

        Ok(FNetSequenceClassificationOutput {
            logits,
            all_hidden_states: base_model_output.all_hidden_states,
        })
    }
}

/// # FNet for multiple choices
/// Multiple choices model using a FNet base model and a linear classifier.
/// Input should be in the form `[CLS] Context [SEP] Possible choice [SEP]`. The choice is made along the batch axis,
/// assuming all elements of the batch are alternatives to be chosen from for a given context.
/// It is made of the following blocks:
/// - `fnet`: Base FNet model
/// - `dropout`: Dropout layer before the last start/end logits prediction
/// - `classifier`: Linear layer for multiple choices
pub struct FNetForMultipleChoice {
    fnet: FNetModel,
    dropout: Dropout,
    classifier: nn::Linear,
}

impl FNetForMultipleChoice {
    /// Build a new `FNetForMultipleChoice`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the FNet model
    /// * `config` - `FNetConfig` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::fnet::{FNetConfig, FNetForMultipleChoice};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = FNetConfig::from_file(config_path);
    /// let fnet = FNetForMultipleChoice::new(&p.root() / "fnet", &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &FNetConfig) -> FNetForMultipleChoice
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let fnet = FNetModel::new(p / "fnet", config, true);
        let dropout = Dropout::new(config.hidden_dropout_prob);
        let classifier = nn::linear(p / "classifier", config.hidden_size, 1, Default::default());

        FNetForMultipleChoice {
            fnet,
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
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `FNetSequenceClassificationOutput` containing:
    ///   - `logits` - `Tensor` of shape (*1*, *batch_size*) containing the logits for each of the alternatives given
    ///   - `all_hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use tch::{nn, Device, Tensor, no_grad};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::Int64;
    /// use rust_bert::fnet::{FNetConfig, FNetForMultipleChoice};
    /// # let config_path = Path::new("path/to/config.json");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = FNetConfig::from_file(config_path);
    /// let model = FNetForMultipleChoice::new(&vs.root(), &config);
    /// let (batch_size, sequence_length) = (64, 128);
    /// let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
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
        input_embeddings: Option<&Tensor>,
        train: bool,
    ) -> Result<FNetSequenceClassificationOutput, RustBertError> {
        let (input_shape, _) =
            get_shape_and_device_from_ids_embeddings_pair(input_ids, input_embeddings)?;
        let num_choices = input_shape[1];

        let input_ids = input_ids.map(|tensor| tensor.view((-1, *tensor.size().last().unwrap())));
        let token_type_ids =
            token_type_ids.map(|tensor| tensor.view((-1, *tensor.size().last().unwrap())));
        let position_ids =
            position_ids.map(|tensor| tensor.view((-1, *tensor.size().last().unwrap())));
        let input_embeddings =
            input_embeddings.map(|tensor| tensor.view((-1, tensor.size()[2], tensor.size()[3])));

        let base_model_output = self.fnet.forward_t(
            input_ids.as_ref(),
            token_type_ids.as_ref(),
            position_ids.as_ref(),
            input_embeddings.as_ref(),
            train,
        )?;

        let logits = base_model_output
            .pooled_output
            .unwrap()
            .apply_t(&self.dropout, train)
            .apply(&self.classifier)
            .view((-1, num_choices));

        Ok(FNetSequenceClassificationOutput {
            logits,
            all_hidden_states: base_model_output.all_hidden_states,
        })
    }
}

/// # FNet for token classification (e.g. NER, POS)
/// Token-level classifier predicting a label for each token provided. Note that because of wordpiece tokenization, the labels predicted are
/// not necessarily aligned with words in the sentence.
/// It is made of the following blocks:
/// - `fnet`: Base FNet
/// - `dropout`: Dropout layer before the last token-level predictions layer
/// - `classifier`: Linear layer for token classification
pub struct FNetForTokenClassification {
    fnet: FNetModel,
    dropout: Dropout,
    classifier: nn::Linear,
}

impl FNetForTokenClassification {
    /// Build a new `FNetForTokenClassification`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the FNet model
    /// * `config` - `FNetConfig` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::fnet::{FNetConfig, FNetForTokenClassification};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = FNetConfig::from_file(config_path);
    /// let fnet = FNetForTokenClassification::new(&p.root() / "fnet", &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &FNetConfig) -> FNetForTokenClassification
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let fnet = FNetModel::new(p / "fnet", config, false);
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

        FNetForTokenClassification {
            fnet,
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
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `FNetTokenClassificationOutput` containing:
    ///   - `logits` - `Tensor` of shape (*batch size*, *sequence_length*, *num_labels*) containing the logits for each of the input tokens and classes
    ///   - `all_hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use tch::{nn, Device, Tensor, no_grad};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::Int64;
    /// use rust_bert::fnet::{FNetConfig, FNetForTokenClassification};
    /// # let config_path = Path::new("path/to/config.json");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = FNetConfig::from_file(config_path);
    /// let model = FNetForTokenClassification::new(&vs.root(), &config);
    /// let (batch_size, sequence_length) = (64, 128);
    /// let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
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
        input_embeddings: Option<&Tensor>,
        train: bool,
    ) -> Result<FNetTokenClassificationOutput, RustBertError> {
        let base_model_output = self.fnet.forward_t(
            input_ids,
            token_type_ids,
            position_ids,
            input_embeddings,
            train,
        )?;

        let logits = base_model_output
            .hidden_states
            .apply_t(&self.dropout, train)
            .apply(&self.classifier);

        Ok(FNetTokenClassificationOutput {
            logits,
            all_hidden_states: base_model_output.all_hidden_states,
        })
    }
}

/// # FNet for question answering
/// Extractive question-answering model based on a FNet language model. Identifies the segment of a context that answers a provided question.
/// Please note that a significant amount of pre- and post-processing is required to perform end-to-end question answering.
/// See the question answering pipeline (also provided in this crate) for more details.
/// It is made of the following blocks:
/// - `fnet`: Base FNet
/// - `qa_outputs`: Linear layer for question answering
pub struct FNetForQuestionAnswering {
    fnet: FNetModel,
    qa_outputs: nn::Linear,
}

impl FNetForQuestionAnswering {
    /// Build a new `FNetForQuestionAnswering`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the FNet model
    /// * `config` - `FNetConfig` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::fnet::{FNetConfig, FNetForQuestionAnswering};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = FNetConfig::from_file(config_path);
    /// let fnet = FNetForQuestionAnswering::new(&p.root() / "fnet", &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &FNetConfig) -> FNetForQuestionAnswering
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let fnet = FNetModel::new(p / "fnet", config, false);
        let qa_outputs = nn::linear(p / "classifier", config.hidden_size, 2, Default::default());

        FNetForQuestionAnswering { fnet, qa_outputs }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see `input_embeds`)
    /// * `token_type_ids` - Optional segment id of shape (*batch size*, *sequence_length*). Convention is value of 0 for the first sentence (incl. *SEP*) and 1 for the second sentence. If None set to 0.
    /// * `position_ids` - Optional position ids of shape (*batch size*, *sequence_length*). If None, will be incremented from 0.
    /// * `input_embeds` - Optional pre-computed input embeddings of shape (*batch size*, *sequence_length*, *hidden_size*). If None, input ids must be provided (see `input_ids`)
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `FNetQuestionAnsweringOutput` containing:
    ///   - `start_logits` - `Tensor` of shape (*batch size*, *sequence_length*) containing the logits for start of the answer
    ///   - `end_logits` - `Tensor` of shape (*batch size*, *sequence_length*) containing the logits for end of the answer
    ///   - `all_hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use tch::{nn, Device, Tensor, no_grad};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::Int64;
    /// use rust_bert::fnet::{FNetConfig, FNetForTokenClassification};
    /// # let config_path = Path::new("path/to/config.json");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = FNetConfig::from_file(config_path);
    /// let model = FNetForTokenClassification::new(&vs.root(), &config);
    /// let (batch_size, sequence_length) = (64, 128);
    /// let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
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
        input_embeddings: Option<&Tensor>,
        train: bool,
    ) -> Result<FNetQuestionAnsweringOutput, RustBertError> {
        let base_model_output = self.fnet.forward_t(
            input_ids,
            token_type_ids,
            position_ids,
            input_embeddings,
            train,
        )?;

        let logits = base_model_output
            .hidden_states
            .apply(&self.qa_outputs)
            .split(1, -1);
        let (start_logits, end_logits) = (&logits[0], &logits[1]);
        let start_logits = start_logits.squeeze_dim(-1);
        let end_logits = end_logits.squeeze_dim(-1);

        Ok(FNetQuestionAnsweringOutput {
            start_logits,
            end_logits,
            all_hidden_states: base_model_output.all_hidden_states,
        })
    }
}

/// Container for the FNet model output.
pub struct FNetModelOutput {
    /// Last hidden states from the model
    pub hidden_states: Tensor,
    /// Pooled output (hidden state for the first token)
    pub pooled_output: Option<Tensor>,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<Tensor>>,
}

/// Container for the FNet masked LM model output.
pub struct FNetMaskedLMOutput {
    /// Logits for the vocabulary items at each sequence position
    pub prediction_scores: Tensor,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<Tensor>>,
}

/// Container for the FNet sequence classification model output.
pub struct FNetSequenceClassificationOutput {
    /// Logits for each input (sequence) for each target class
    pub logits: Tensor,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<Tensor>>,
}

/// Container for the FNet token classification model output.
pub type FNetTokenClassificationOutput = FNetSequenceClassificationOutput;

/// Container for the FNet question answering model output.
pub struct FNetQuestionAnsweringOutput {
    /// Logits for the start position for token of each input sequence
    pub start_logits: Tensor,
    /// Logits for the end position for token of each input sequence
    pub end_logits: Tensor,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<Tensor>>,
}

#[cfg(test)]
mod test {
    use tch::Device;

    use crate::{
        resources::{RemoteResource, ResourceProvider},
        Config,
    };

    use super::*;

    #[test]
    #[ignore] // compilation is enough, no need to run
    fn fnet_model_send() {
        let config_resource = Box::new(RemoteResource::from_pretrained(FNetConfigResources::BASE));
        let config_path = config_resource.get_local_path().expect("");

        //    Set-up masked LM model
        let device = Device::cuda_if_available();
        let vs = tch::nn::VarStore::new(device);
        let config = FNetConfig::from_file(config_path);

        let _: Box<dyn Send> = Box::new(FNetModel::new(&vs.root(), &config, true));
    }
}
