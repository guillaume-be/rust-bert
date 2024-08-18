// Copyright 2021 The Eleuther AI and HuggingFace Inc. team. All rights reserved.
// Copyright 2022 Guillaume Becquin
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
use crate::gpt_j::attention::{GptJAttention, LayerState};
use crate::gpt_j::gpt_j_model::GptJConfig;
use std::borrow::Borrow;
use tch::nn::Linear;
use tch::{nn, Tensor};

pub struct GptJMLP {
    fc_in: Linear,
    fc_out: Linear,
    activation: TensorFunction,
    dropout: Dropout,
}

impl GptJMLP {
    pub fn new<'p, P>(p: P, config: &GptJConfig) -> GptJMLP
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let intermediate_size = if let Some(n_inner) = config.n_inner {
            n_inner
        } else {
            4 * config.n_embd
        };
        let fc_in = nn::linear(
            p / "fc_in",
            config.n_embd,
            intermediate_size,
            Default::default(),
        );
        let fc_out = nn::linear(
            p / "fc_out",
            intermediate_size,
            config.n_embd,
            Default::default(),
        );

        let activation = match &config.afn {
            Some(activation_enum) => match activation_enum {
                Activation::gelu => &Activation::gelu_new,
                default => default,
            },
            None => &Activation::gelu_new,
        }
        .get_function();

        let resid_pdrop = config.resid_pdrop.unwrap_or(0.1);
        let dropout = Dropout::new(resid_pdrop);

        GptJMLP {
            fc_in,
            fc_out,
            activation,
            dropout,
        }
    }

    pub fn forward_t(&self, hidden_states: &Tensor, train: bool) -> Tensor {
        let h = (self.activation.get_fn())(&hidden_states.apply(&self.fc_in));
        h.apply(&self.fc_out).apply_t(&self.dropout, train)
    }
}

pub struct GptJBlock {
    ln_1: nn::LayerNorm,
    attn: GptJAttention,
    mlp: GptJMLP,
}

impl GptJBlock {
    pub fn new<'p, P>(p: P, config: &GptJConfig) -> GptJBlock
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let layer_norm_config = nn::LayerNormConfig {
            eps: config.layer_norm_epsilon,
            ..Default::default()
        };
        let ln_1 = nn::layer_norm(p / "ln_1", vec![config.n_embd], layer_norm_config);
        let attn = GptJAttention::new(p / "attn", config);
        let mlp = GptJMLP::new(p / "mlp", config);

        GptJBlock { ln_1, attn, mlp }
    }

    pub fn forward_t(
        &self,
        hidden_states: &Tensor,
        layer_past: Option<&LayerState>,
        attention_mask: Option<&Tensor>,
        train: bool,
    ) -> (Tensor, Option<LayerState>, Option<Tensor>) {
        let residual = hidden_states;
        let hidden_states = hidden_states.apply(&self.ln_1);

        let (attn_output, present, attn_weights) =
            self.attn
                .forward_t(&hidden_states, attention_mask, layer_past, train);

        let feed_forward_hidden_states = self.mlp.forward_t(&hidden_states, train);
        let hidden_states = attn_output + feed_forward_hidden_states + residual;

        (hidden_states, present, attn_weights)
    }
}
