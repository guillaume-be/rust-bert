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

use crate::albert::attention::AlbertSelfAttention;
use tch::{nn, Tensor};
use crate::albert::AlbertConfig;
use crate::albert::albert::Activation;
use crate::common::activations::{_gelu_new, _gelu, _relu, _mish};

pub struct AlbertLayer {
    attention: AlbertSelfAttention,
    full_layer_layer_norm: nn::LayerNorm,
    ffn: nn::Linear,
    ffn_output: nn::Linear,
    activation: Box<dyn Fn(&Tensor) -> Tensor>,
}

impl AlbertLayer {
    pub fn new(p: &nn::Path, config: &AlbertConfig) -> AlbertLayer {
        let attention = AlbertSelfAttention::new(p / "attention", &config);

        let layer_norm_eps = match config.layer_norm_eps {
            Some(value) => value,
            None => 1e-12
        };
        let layer_norm_config = nn::LayerNormConfig { eps: layer_norm_eps, ..Default::default() };
        let full_layer_layer_norm = nn::layer_norm(&(p / "full_layer_layer_norm"), vec![config.hidden_size], layer_norm_config);

        let ffn = nn::linear(&(p / "ffn"), config.hidden_size, config.intermediate_size, Default::default());
        let ffn_output = nn::linear(&(p / "ffn_output"), config.intermediate_size, config.hidden_size, Default::default());

        let activation = Box::new(match &config.hidden_act {
            Activation::gelu_new => _gelu_new,
            Activation::gelu => _gelu,
            Activation::relu => _relu,
            Activation::mish => _mish
        });

        AlbertLayer { attention, full_layer_layer_norm, ffn, ffn_output, activation }
    }

    pub fn forward_t(&self,
                     hidden_states: &Tensor,
                     mask: &Option<Tensor>,
                     train: bool) -> (Tensor, Option<Tensor>) {
        let (attention_output, attention_weights) = self.attention.forward_t(hidden_states, mask, train);
        let ffn_output = attention_output.apply(&self.ffn);
        let ffn_output: Tensor = (self.activation)(&ffn_output);
        let ffn_output = ffn_output.apply(&self.ffn_output).apply(&self.full_layer_layer_norm);

        (ffn_output, attention_weights)
    }
}