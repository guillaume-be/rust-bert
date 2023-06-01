// Copyright 2021 The Eleuther AI and HuggingFace Inc. team. All rights reserved.
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
use crate::gpt_neo::gpt_neo_model::AttentionLayerType;
use crate::gpt_neo::GptNeoConfig;
use std::borrow::Borrow;
use tch::{nn, Kind, Tensor};

#[derive(Debug)]
/// # Cache for GPT-Neo attention layers
/// Stores the cached value of key and value
pub struct LayerState {
    /// Cached keys
    pub prev_key: Tensor,
    /// Cached values
    pub prev_value: Option<Tensor>,
}

impl Clone for LayerState {
    fn clone(&self) -> Self {
        LayerState {
            prev_key: self.prev_key.copy(),
            prev_value: self.prev_value.as_ref().map(|value| value.copy()),
        }
    }
}

impl LayerState {
    pub(crate) fn reorder_cache(&mut self, new_indices: &Tensor) {
        self.prev_key = self.prev_key.index_select(0, new_indices);
        self.prev_value = self
            .prev_value
            .as_ref()
            .map(|value| value.index_select(0, new_indices));
    }
}

pub struct GptNeoSelfAttention {
    k_proj: nn::Linear,
    v_proj: nn::Linear,
    q_proj: nn::Linear,
    out_proj: nn::Linear,
    attention_dropout: Dropout,
    resid_dropout: Dropout,
    bias: Tensor,
    num_heads: i64,
    head_dim: i64,
    output_attentions: bool,
}

impl GptNeoSelfAttention {
    pub fn new<'p, P>(
        p: P,
        config: &GptNeoConfig,
        attention_type: &AttentionLayerType,
    ) -> GptNeoSelfAttention
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let max_positions = config.max_position_embeddings;

        let mut bias = Tensor::ones([max_positions, max_positions], (Kind::Uint8, p.device()))
            .tril(0)
            .view([1, 1, max_positions, max_positions])
            .requires_grad_(false);

        if attention_type == &AttentionLayerType::Local {
            let _ = bias.bitwise_or_tensor_(&bias.tril(-config.window_size));
        }

        let attention_dropout = Dropout::new(config.attention_dropout);
        let resid_dropout = Dropout::new(config.resid_dropout);

        let num_heads = config.num_heads;
        let head_dim = config.hidden_size / config.num_heads;

        let linear_config = nn::LinearConfig {
            bias: false,
            ..Default::default()
        };
        let k_proj = nn::linear(
            p / "k_proj",
            config.hidden_size,
            config.hidden_size,
            linear_config,
        );
        let v_proj = nn::linear(
            p / "v_proj",
            config.hidden_size,
            config.hidden_size,
            linear_config,
        );
        let q_proj = nn::linear(
            p / "q_proj",
            config.hidden_size,
            config.hidden_size,
            linear_config,
        );
        let out_proj = nn::linear(
            p / "out_proj",
            config.hidden_size,
            config.hidden_size,
            Default::default(),
        );

        let output_attentions = config.output_attentions.unwrap_or(false);

        GptNeoSelfAttention {
            k_proj,
            v_proj,
            q_proj,
            out_proj,
            attention_dropout,
            resid_dropout,
            bias,
            num_heads,
            head_dim,
            output_attentions,
        }
    }

    fn split_heads(input_tensor: &Tensor, num_heads: i64, attention_head_size: i64) -> Tensor {
        let mut new_shape = input_tensor.size();
        let _ = new_shape.pop();
        new_shape.extend_from_slice(&[num_heads, attention_head_size]);
        let reshaped_tensor = input_tensor.view(new_shape.as_slice());
        reshaped_tensor.permute([0, 2, 1, 3])
    }

    fn merge_heads(input_tensor: &Tensor, num_heads: i64, attention_head_size: i64) -> Tensor {
        let output_tensor = input_tensor.permute([0, 2, 1, 3]).contiguous();
        let mut new_shape = output_tensor.size();
        new_shape.truncate(new_shape.len() - 2);
        new_shape.push(num_heads * attention_head_size);
        output_tensor.view(new_shape.as_slice())
    }

    fn attend(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attention_mask: Option<&Tensor>,
        train: bool,
    ) -> (Tensor, Tensor) {
        let query = query.to_kind(Kind::Float);
        let key = key.to_kind(Kind::Float);

        let attention_weights = query.matmul(&key.transpose(-1, -2));

        let query_dims = query.size();
        let key_dims = key.size();
        let query_length = query_dims[query_dims.len() - 2];
        let key_length = key_dims[key_dims.len() - 2];

        let causal_mask = &self
            .bias
            .slice(2, key_length - query_length, key_length, 1)
            .slice(3, 0, key_length, 1)
            .to_kind(Kind::Bool)
            .to_device(attention_weights.device());

        let mut attention_weights = attention_weights.where_self(
            causal_mask,
            &Tensor::from_slice(&[-1e9f32]).to_device(attention_weights.device()),
        );
        if let Some(attention_mask_value) = attention_mask {
            attention_weights = attention_weights + attention_mask_value;
        };

        let attention_weights = attention_weights.softmax(-1, attention_weights.kind());
        let attention_weights = attention_weights
            .to_kind(value.kind())
            .apply_t(&self.attention_dropout, train);
        let attention_output = attention_weights.matmul(value);
        (attention_output, attention_weights)
    }

    pub fn forward_t(
        &self,
        hidden_states: &Tensor,
        layer_state: Option<&LayerState>,
        attention_mask: Option<&Tensor>,
        train: bool,
    ) -> (Tensor, Option<Tensor>, Option<LayerState>) {
        let query = hidden_states.apply(&self.q_proj);
        let key = hidden_states.apply(&self.k_proj);
        let value = hidden_states.apply(&self.v_proj);

        let query = Self::split_heads(&query, self.num_heads, self.head_dim);
        let mut key = Self::split_heads(&key, self.num_heads, self.head_dim);
        let mut value = Self::split_heads(&value, self.num_heads, self.head_dim);

        if let Some(layer_state_value) = &layer_state {
            key = Tensor::cat(&[&layer_state_value.prev_key, &key], -2);
            value = Tensor::cat(
                &[layer_state_value.prev_value.as_ref().unwrap(), &value],
                -2,
            );
        };

        let layer_state = Some(LayerState {
            prev_key: key.copy(),
            prev_value: Some(value.copy()),
        });

        let (attention_output, attention_weights) =
            self.attend(&query, &key, &value, attention_mask, train);

        let attention_output = Self::merge_heads(&attention_output, self.num_heads, self.head_dim)
            .apply(&self.out_proj)
            .apply_t(&self.resid_dropout, train);

        let attention_weights = if self.output_attentions {
            Some(attention_weights)
        } else {
            None
        };

        (attention_output, attention_weights, layer_state)
    }
}
