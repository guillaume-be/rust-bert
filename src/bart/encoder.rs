// Copyright 2020 The Facebook AI Research Team Authors
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

use crate::bart::attention::SelfAttention;
use tch::{nn, Tensor};
use crate::common::dropout::Dropout;
use crate::bart::BartConfig;
use crate::bart::bart::Activation;
use crate::common::activations::{_gelu, _relu, _swish, _gelu_new, _tanh};

pub struct EncoderLayer {
    self_attention: SelfAttention,
    self_attention_layer_norm: nn::LayerNorm,
    dropout: Dropout,
    activation_dropout: Dropout,
    activation: Box<dyn Fn(&Tensor) -> Tensor>,
    fc1: nn::Linear,
    fc2: nn::Linear,
    final_layer_norm: nn::LayerNorm,
}

impl EncoderLayer {
    pub fn new(p: nn::Path, config: &BartConfig) -> EncoderLayer {
        let layer_norm_config = nn::LayerNormConfig { eps: 1e-5, ..Default::default() };
        let output_attention = match config.output_attentions {
            Some(value) => value,
            None => false
        };
        let self_attention = SelfAttention::new(&p / "self_attn ",
                                                config.d_model,
                                                config.encoder_attention_heads,
                                                config.attention_dropout,
                                                false,
                                                false,
                                                output_attention);
        let self_attention_layer_norm = nn::layer_norm(&p / "self_attn_layer_norm",
                                                       vec![config.d_model],
                                                       layer_norm_config);
        let dropout = Dropout::new(config.dropout);
        let activation_dropout = Dropout::new(config.activation_dropout);
        let activation_function = match &config.activation_function {
            Some(act_function) => act_function,
            None => &Activation::gelu
        };
        let activation = Box::new(match activation_function {
            Activation::gelu => _gelu,
            Activation::relu => _relu,
            Activation::swish => _swish,
            Activation::gelu_new => _gelu_new,
            Activation::tanh => _tanh
        });
        let fc1 = nn::linear(&p / "fc1", config.d_model, config.encoder_ffn_dim, Default::default());
        let fc2 = nn::linear(&p / "fc2", config.encoder_ffn_dim, config.d_model, Default::default());

        let final_layer_norm = nn::layer_norm(&p / "final_layer_norm",
                                              vec![config.d_model],
                                              layer_norm_config);

        EncoderLayer { self_attention, self_attention_layer_norm, dropout, activation_dropout, activation, fc1, fc2, final_layer_norm }
    }
}