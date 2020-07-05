// Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
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

use crate::common::dropout::Dropout;
use crate::t5::attention::{LayerState, T5LayerCrossAttention, T5LayerSelfAttention};
use crate::t5::T5Config;
use std::borrow::Borrow;
use tch::nn::LinearConfig;
use tch::{nn, Tensor};

pub struct T5DenseReluDense {
    wi: nn::Linear,
    wo: nn::Linear,
    dropout: Dropout,
}

impl T5DenseReluDense {
    pub fn new<'p, P>(p: P, config: &T5Config) -> T5DenseReluDense
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let linear_config = LinearConfig {
            bias: false,
            ..Default::default()
        };
        let wi = nn::linear(p / "wi", config.d_model, config.d_ff, linear_config);
        let wo = nn::linear(p / "wi", config.d_ff, config.d_model, linear_config);
        let dropout = Dropout::new(config.dropout_rate);

        T5DenseReluDense { wi, wo, dropout }
    }

    pub fn forward_t(&self, hidden_states: &Tensor, train: bool) -> Tensor {
        hidden_states
            .apply(&self.wi)
            .relu()
            .apply_t(&self.dropout, train)
            .apply(&self.wo)
    }
}

pub struct T5LayerFF {
    dense_relu_dense: T5DenseReluDense,
    layer_norm: nn::LayerNorm,
    dropout: Dropout,
}

impl T5LayerFF {
    pub fn new<'p, P>(p: P, config: &T5Config) -> T5LayerFF
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let dense_relu_dense = T5DenseReluDense::new(p / "DenseReluDense", config);
        let layer_norm_config = nn::LayerNormConfig {
            eps: config.layer_norm_epsilon,
            ..Default::default()
        };
        let layer_norm = nn::layer_norm(p / "layer_norm", vec![config.d_model], layer_norm_config);
        let dropout = Dropout::new(config.dropout_rate);

        T5LayerFF {
            dense_relu_dense,
            layer_norm,
            dropout,
        }
    }

    pub fn forward_t(&self, hidden_states: &Tensor, train: bool) -> Tensor {
        let y = &self
            .dense_relu_dense
            .forward_t(&hidden_states.apply(&self.layer_norm), train);

        hidden_states + y.apply_t(&self.dropout, train)
    }
}

pub struct T5Block {
    self_attention: T5LayerSelfAttention,
    cross_attention: Option<T5LayerCrossAttention>,
    ff_layer: T5LayerFF,
}

impl T5Block {
    pub fn new<'p, P>(
        p: P,
        config: &T5Config,
        has_relative_attention_bias: bool,
        is_decoder: bool,
        store_cache: bool,
        output_attentions: bool,
    ) -> T5Block
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow() / "layer";
        let mut module_index = 0;

        let self_attention = T5LayerSelfAttention::new(
            &p / module_index,
            config,
            is_decoder,
            store_cache,
            output_attentions,
            has_relative_attention_bias,
        );

        let cross_attention = if is_decoder {
            module_index += 1;
            Some(T5LayerCrossAttention::new(
                &p / module_index,
                config,
                is_decoder,
                store_cache,
                output_attentions,
                has_relative_attention_bias,
            ))
        } else {
            None
        };
        module_index += 1;

        let ff_layer = T5LayerFF::new(&p / module_index, config);

        T5Block {
            self_attention,
            cross_attention,
            ff_layer,
        }
    }

    pub fn forward_t(
        &self,
        input: &Tensor,
        position_bias: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
        encoder_attention_mask: Option<&Tensor>,
        encoder_decoder_position_bias: Option<&Tensor>,
        mut layer_states: (Option<LayerState>, Option<LayerState>),
        train: bool,
    ) -> (
        Tensor,
        (Option<Tensor>, Option<Tensor>),
        (Option<LayerState>, Option<LayerState>),
    ) {
        let (hidden_states, self_attention_weights, self_attention_layer_past) = self
            .self_attention
            .forward_t(input, position_bias, attention_mask, layer_states.0, train);

        let (hidden_states, cross_attention_weights, cross_attention_layer_past) =
            if self.cross_attention.is_some() & encoder_hidden_states.is_some() {
                let query_length = match &self_attention_layer_past {
                    Some(value) => Some(value.prev_key.size()[2]),
                    None => None,
                };
                self.cross_attention.as_ref().unwrap().forward_t(
                    &hidden_states,
                    encoder_hidden_states,
                    encoder_decoder_position_bias,
                    encoder_attention_mask,
                    layer_states.1,
                    query_length,
                    train,
                )
            } else {
                (hidden_states, None, None)
            };

        let attention_weights = (self_attention_weights, cross_attention_weights);

        layer_states = (self_attention_layer_past, cross_attention_layer_past);
        let hidden_states = self.ff_layer.forward_t(&hidden_states, train);

        (hidden_states, attention_weights, layer_states)
    }
}
