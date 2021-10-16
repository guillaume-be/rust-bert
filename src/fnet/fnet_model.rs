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
use crate::{Activation, Config};
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
}

impl FNetConfigResources {
    /// Shared under Apache 2.0 license by the Google team at <https://github.com/google-research/google-research/tree/master/f_net>. Modified with conversion to C-array format.
    pub const BASE: (&'static str, &'static str) = (
        "fnet-base/config",
        "https://huggingface.co/google/fnet-base/resolve/main/config.json",
    );
}

impl FNetVocabResources {
    /// Shared under Apache 2.0 license by the Google team at <https://github.com/google-research/google-research/tree/master/f_net>. Modified with conversion to C-array format.
    pub const BASE: (&'static str, &'static str) = (
        "fnet-base/spiece",
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
