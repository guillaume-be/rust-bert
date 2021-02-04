// Copyright 2020 The Allen Institute for AI team and The HuggingFace Inc. team.
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

use crate::common::dropout::Dropout;
use crate::longformer::attention::LongformerSelfAttention;
use crate::longformer::LongformerConfig;
use std::borrow::Borrow;
use tch::{nn, Tensor};

pub struct LongformerSelfOutput {
    dense: nn::Linear,
    layer_norm: nn::LayerNorm,
    dropout: Dropout,
}

impl LongformerSelfOutput {
    pub fn new<'p, P>(p: P, config: &LongformerConfig) -> LongformerSelfOutput
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
        let layer_norm_config = nn::LayerNormConfig {
            eps: config.layer_norm_eps.unwrap_or(1e-12),
            ..Default::default()
        };

        let layer_norm =
            nn::layer_norm(p / "LayerNorm", vec![config.hidden_size], layer_norm_config);

        let dropout = Dropout::new(config.hidden_dropout_prob);
        LongformerSelfOutput {
            dense,
            layer_norm,
            dropout,
        }
    }

    pub fn forward_t(&self, hidden_states: &Tensor, input_tensor: &Tensor, train: bool) -> Tensor {
        let hidden_states = hidden_states
            .apply(&self.dense)
            .apply_t(&self.dropout, train);

        (hidden_states + input_tensor).apply(&self.layer_norm)
    }
}

pub struct LongformerAttention {
    self_attention: LongformerSelfAttention,
    output: LongformerSelfOutput,
}

impl LongformerAttention {
    pub fn new<'p, P>(p: P, config: &LongformerConfig, layer_id: i64) -> LongformerAttention
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let self_attention = LongformerSelfAttention::new(p / "self", config, layer_id);
        let output = LongformerSelfOutput::new(p / "output", config);

        LongformerAttention {
            self_attention,
            output,
        }
    }

    pub fn forward_t(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
        is_index_masked: &Tensor,
        is_index_global_attention: &Tensor,
        is_global_attention: bool,
        train: bool,
    ) -> (Tensor, Option<Tensor>) {
        let (attention_outputs, attention_scores) = self.self_attention.forward_t(
            hidden_states,
            attention_mask,
            is_index_masked,
            is_index_global_attention,
            is_global_attention,
            train,
        );

        let attention_outputs = self
            .output
            .forward_t(&attention_outputs, hidden_states, train);

        (attention_outputs, attention_scores)
    }
}
