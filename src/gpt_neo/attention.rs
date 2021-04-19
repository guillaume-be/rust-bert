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
use crate::gpt_neo::GptNeoConfig;
use crate::RustBertError;
use std::borrow::Borrow;
use tch::nn::Init;
use tch::{nn, Kind, Tensor};

#[derive(Debug)]
/// # Cache for GPTNeo attention layers
/// Stores the cached value of key and value
pub struct LayerState {
    /// Cached keys
    pub prev_key: Tensor,
    /// Cached values
    pub prev_value: Tensor,
}

impl Clone for LayerState {
    fn clone(&self) -> Self {
        LayerState {
            prev_key: self.prev_key.copy(),
            prev_value: self.prev_value.copy(),
        }
    }
}

impl LayerState {
    pub(crate) fn reorder_cache(&mut self, new_indices: &Tensor) {
        self.prev_key = self.prev_key.index_select(0, new_indices);
        self.prev_value = self.prev_value.index_select(0, new_indices);
    }
}

trait GptNeoAttention {
    fn get_block_length_and_num_blocks(sequence_length: i64, window_size: i64) -> (i64, i64) {
        let mut block_length = window_size;
        while sequence_length % block_length != 0 {
            block_length -= 1;
        }
        let num_blocks = sequence_length / block_length;
        (block_length, num_blocks)
    }

    fn look_back(
        input_tensor: &Tensor,
        block_length: i64,
        window_size: i64,
        pad_value: Option<i64>,
        is_key_value: bool,
    ) -> Result<Tensor, RustBertError> {
        let padding_size = match input_tensor.size().len() {
            3 => Vec::from([0, 0, window_size, 0]),
            2 => Vec::from([window_size, 0]),
            _ => {
                return Err(RustBertError::ValueError(format!(
                    "Invalid tensor rank, expected 2 or 3, got {}",
                    input_tensor.size().len()
                )));
            }
        };

        let mut padded_tensor = match pad_value {
            None => input_tensor.constant_pad_nd(padding_size.as_slice()),
            Some(value) => {
                if value == 0 {
                    input_tensor.constant_pad_nd(padding_size.as_slice())
                } else {
                    (input_tensor - value).constant_pad_nd(padding_size.as_slice()) + value
                }
            }
        };

        padded_tensor = padded_tensor.unfold(1, window_size + block_length, block_length);
        if is_key_value {
            padded_tensor = padded_tensor.transpose(-2, -1);
        }

        Ok(padded_tensor)
    }

    fn split_heads(
        input_tensor: &Tensor,
        num_heads: i64,
        attention_head_size: i64,
    ) -> Result<Tensor, RustBertError> {
        let mut new_shape = input_tensor.size();
        let _ = new_shape.pop();
        new_shape.extend_from_slice(&[num_heads, attention_head_size]);

        let reshaped_tensor = input_tensor.view(new_shape.as_slice());

        Ok(match reshaped_tensor.size().len() {
            5 => reshaped_tensor.permute(&[0, 1, 3, 2, 4]),
            4 => reshaped_tensor.permute(&[0, 2, 1, 3]),
            _ => {
                return Err(RustBertError::ValueError(format!(
                    "Invalid tensor rank, expected 4 or 5, got {}",
                    input_tensor.size().len()
                )));
            }
        })
    }

    fn merge_heads(
        input_tensor: &Tensor,
        num_heads: i64,
        attention_head_size: i64,
    ) -> Result<Tensor, RustBertError> {
        let output_tensor = match input_tensor.size().len() {
            5 => input_tensor.permute(&[0, 1, 3, 2, 4]).contiguous(),
            4 => input_tensor.permute(&[0, 2, 1, 3]).contiguous(),
            _ => {
                return Err(RustBertError::ValueError(format!(
                    "Invalid tensor rank, expected 4 or 5, got {}",
                    input_tensor.size().len()
                )));
            }
        };
        let mut new_shape = input_tensor.size();
        new_shape.truncate(new_shape.len() - 2);
        new_shape.push(num_heads * attention_head_size);
        Ok(output_tensor.view(new_shape.as_slice()))
    }

    fn split_sequence_length_dim_to(
        input_tensor: &Tensor,
        dim_factor_1: i64,
        dim_factor_2: i64,
        hidden_size: i64,
    ) -> Result<Tensor, RustBertError> {
        let batch_size = input_tensor.size()[0];
        let mut split_dim_shape = Vec::from([batch_size, dim_factor_1, dim_factor_2]);

        Ok(match input_tensor.size().len() {
            3 => {
                split_dim_shape.push(hidden_size);
                input_tensor.reshape(split_dim_shape.as_slice())
            }
            2 => input_tensor.reshape(split_dim_shape.as_slice()),
            _ => {
                return Err(RustBertError::ValueError(format!(
                    "Invalid tensor rank, expected 2 or 3, got {}",
                    input_tensor.size().len()
                )));
            }
        })
    }

    fn attend(
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        causal_mask: &Tensor,
        masked_bias: &Tensor,
        attention_dropout: &Dropout,
        attention_mask: Option<&Tensor>,
        train: bool,
    ) -> (Tensor, Tensor) {
        let mut attention_weights = query
            .matmul(&key.transpose(-1, -2))
            .where1(causal_mask, masked_bias);

        if let Some(attention_mask_value) = attention_mask {
            attention_weights = attention_weights + attention_mask_value;
        };

        attention_weights = attention_weights
            .softmax(-1, Kind::Float)
            .apply_t(attention_dropout, train);

        let attention_output = attention_weights.matmul(value);
        (attention_output, attention_weights)
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
    masked_bias: Tensor,
    num_heads: i64,
    head_dim: i64,
    output_attentions: bool,
}

impl GptNeoAttention for GptNeoSelfAttention {}

impl GptNeoSelfAttention {
    pub fn new<'p, P>(p: P, config: &GptNeoConfig) -> GptNeoSelfAttention
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let max_positions = config.max_position_embeddings;

        let mut bias = p.var(
            "bias",
            &[1, 1, max_positions, max_positions],
            Init::Const(0.),
        );
        bias.copy_(
            &Tensor::ones(&[max_positions, max_positions], (Kind::Int8, p.device()))
                .tril(0)
                .view([1, 1, max_positions, max_positions]),
        );
        let masked_bias = p.var("masked_bias", &[0], Init::Const(-1e9));

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
            p / "k_proj",
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
            masked_bias,
            num_heads,
            head_dim,
            output_attentions,
        }
    }

    pub fn forward_t(
        &self,
        hidden_states: &Tensor,
        mut layer_state: Option<LayerState>,
        attention_mask: Option<&Tensor>,
        train: bool,
    ) -> Result<(Tensor, Option<Tensor>, Option<LayerState>), RustBertError> {
        let query = hidden_states.apply(&self.q_proj);
        let key = hidden_states.apply(&self.k_proj);
        let value = hidden_states.apply(&self.v_proj);

        let query = Self::split_heads(&query, self.num_heads, self.head_dim)?;
        let mut key = Self::split_heads(&key, self.num_heads, self.head_dim)?;
        let mut value = Self::split_heads(&value, self.num_heads, self.head_dim)?;

        if let Some(layer_state_value) = &layer_state {
            key = Tensor::cat(&[&layer_state_value.prev_key, &key], -2);
            value = Tensor::cat(&[&layer_state_value.prev_value, &key], -2);
            layer_state.as_mut().unwrap().prev_key = key.copy();
            layer_state.as_mut().unwrap().prev_value = value.copy();
        } else {
            layer_state = Some(LayerState {
                prev_key: key.copy(),
                prev_value: value.copy(),
            });
        };

        let query_dims = query.size();
        let key_dims = key.size();
        let query_length = query_dims[query_dims.len() - 2];
        let key_length = key_dims[key_dims.len() - 2];

        let causal_mask = self
            .bias
            .narrow(0, key_length - query_length, key_length)
            .slice(1, 0, key_length, 1)
            .unsqueeze(0)
            .unsqueeze(0);

        let (attention_output, attention_weights) = Self::attend(
            &query,
            &key,
            &value,
            &causal_mask,
            &self.masked_bias,
            &self.attention_dropout,
            attention_mask,
            train,
        );

        let attention_output = Self::merge_heads(&attention_output, self.num_heads, self.head_dim)?
            .apply(&self.out_proj)
            .apply_t(&self.resid_dropout, train);

        let attention_weights = if self.output_attentions {
            Some(attention_weights)
        } else {
            None
        };

        Ok((attention_output, attention_weights, layer_state))
    }
}
