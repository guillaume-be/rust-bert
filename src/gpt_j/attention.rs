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

use crate::common::dropout::Dropout;
use crate::common::kind::get_min;
use crate::gpt_j::gpt_j_model::GptJConfig;
use std::borrow::Borrow;
use tch::nn::Linear;
use tch::{nn, IndexOp, Kind, NewAxis, Tensor};

#[derive(Debug)]
/// # Cache for GPT-J attention layers
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

pub struct GptJAttention {
    bias: Tensor,
    attn_dropout: Dropout,
    resid_dropout: Dropout,
    scale_attn: f32,
    k_proj: Linear,
    v_proj: Linear,
    q_proj: Linear,
    out_proj: Linear,
    output_attentions: bool,
    dim_per_head: i64,
    n_head: i64,
    rotary_dim: Option<i64>,
    scale: bool,
    use_cache: bool,
}

impl GptJAttention {
    pub fn new<'p, P>(p: P, config: &GptJConfig) -> GptJAttention
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let max_positions = config.n_positions;
        let bias = Tensor::ones([max_positions, max_positions], (Kind::Uint8, p.device()))
            .tril(0)
            .view([1, 1, max_positions, max_positions])
            .requires_grad_(false);
        let bias = p.var_copy("bias", &bias);

        let attn_pdrop = config.attn_pdrop.unwrap_or(0.1);
        let resid_pdrop = config.resid_pdrop.unwrap_or(0.1);
        let output_attentions = config.output_attentions.unwrap_or(false);

        let attn_dropout = Dropout::new(attn_pdrop);
        let resid_dropout = Dropout::new(resid_pdrop);

        assert_eq!(
            config.n_embd % config.n_head,
            0,
            "Attention hidden states not a multiple of the number of heads"
        );
        let dim_per_head = config.n_embd / config.n_head;

        let scale_attn = (dim_per_head as f32).sqrt();

        let linear_config = nn::LinearConfig {
            bias: false,
            ..Default::default()
        };
        let k_proj = nn::linear(p / "k_proj", config.n_embd, config.n_embd, linear_config);
        if config.use_float16 {
            (p / "k_proj").half();
        }
        let v_proj = nn::linear(p / "v_proj", config.n_embd, config.n_embd, linear_config);
        if config.use_float16 {
            (p / "v_proj").half();
        }
        let q_proj = nn::linear(p / "q_proj", config.n_embd, config.n_embd, linear_config);
        if config.use_float16 {
            (p / "q_proj").half();
        }
        let out_proj = nn::linear(p / "out_proj", config.n_embd, config.n_embd, linear_config);
        if config.use_float16 {
            (p / "out_proj").half();
        }

        GptJAttention {
            bias,
            attn_dropout,
            resid_dropout,
            output_attentions,
            scale_attn,
            k_proj,
            v_proj,
            q_proj,
            out_proj,
            dim_per_head,
            n_head: config.n_head,
            rotary_dim: config.rotary_dim,
            scale: config.scale_attn_weights.unwrap_or(true),
            use_cache: config.use_cache.unwrap_or(true),
        }
    }

    fn split_heads(
        tensor: &Tensor,
        num_heads: i64,
        attention_head_size: i64,
        rotary: bool,
    ) -> Tensor {
        let mut new_shape = tensor.size();
        let _ = new_shape.pop();
        new_shape.extend_from_slice(&[num_heads, attention_head_size]);
        let tensor = tensor.view(new_shape.as_slice());
        if rotary {
            tensor
        } else if tensor.size().len() == 5 {
            tensor.permute([0, 1, 3, 2, 4]) // (batch, blocks, head, block_length, head_features)
        } else if tensor.size().len() == 4 {
            tensor.permute([0, 2, 1, 3]) // (batch, head, seq_length, head_features)
        } else {
            panic!(
                "Input tensor should either be a rotary head, or its rank be one of [4, 5] but is: {}",
                tensor.size().len()
            )
        }
    }

    fn merge_heads(tensor: &Tensor, num_heads: i64, attention_head_size: i64) -> Tensor {
        let tensor = if tensor.size().len() == 5 {
            tensor.permute([0, 1, 3, 2, 4]).contiguous()
        } else if tensor.size().len() == 4 {
            tensor.permute([0, 2, 1, 3]).contiguous()
        } else {
            panic!(
                "Input tensor rank should be one of [4, 5], but is: {}",
                tensor.size().len()
            )
        };
        let mut new_shape = tensor.size();
        new_shape.truncate(new_shape.len() - 2);
        new_shape.push(num_heads * attention_head_size);
        tensor.view(new_shape.as_slice())
    }

    fn attention(
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

        let mask_value = get_min(attention_weights.kind()).unwrap();
        let mask_value = Tensor::full(
            attention_weights.size(),
            mask_value,
            (attention_weights.kind(), attention_weights.device()),
        );

        let mut attention_weights = attention_weights.where_self(causal_mask, &mask_value);
        if self.scale {
            attention_weights /= self.scale_attn;
        }
        if let Some(attention_mask_value) = attention_mask {
            attention_weights += attention_mask_value;
        };
        let attention_weights = attention_weights.softmax(-1, attention_weights.kind());
        let attention_weights = attention_weights
            .to_kind(value.kind())
            .apply_t(&self.attn_dropout, train);

        let attention_output = attention_weights.matmul(value);

        (attention_output, attention_weights)
    }

    pub fn forward_t(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        layer_past: Option<&LayerState>,
        train: bool,
    ) -> (Tensor, Option<LayerState>, Option<Tensor>) {
        let query = hidden_states.apply(&self.q_proj);
        let key = hidden_states.apply(&self.k_proj);
        let value = hidden_states.apply(&self.v_proj);

        let mut query = Self::split_heads(&query, self.n_head, self.dim_per_head, true);
        let mut key = Self::split_heads(&key, self.n_head, self.dim_per_head, true);
        let mut value = Self::split_heads(&value, self.n_head, self.dim_per_head, false);

        let mut seq_len = key.size()[1];
        let mut offset = 0;

        if let Some(layer_past) = layer_past {
            offset = layer_past.prev_key.size()[layer_past.prev_key.size().len() - 2];
            seq_len += offset
        };

        if let Some(rotary_dim) = self.rotary_dim {
            let k_rot = key.slice(3, 0, rotary_dim, 1);
            let k_pass = key.slice(3, rotary_dim, key.size()[3], 1);

            let q_rot = query.slice(3, 0, rotary_dim, 1);
            let q_pass = query.slice(3, rotary_dim, query.size()[3], 1);

            let sincos = fixed_pos_embedding(&k_rot, seq_len);
            let k_rot = apply_rotary_pos_emb(&k_rot, &sincos, offset);
            let q_rot = apply_rotary_pos_emb(&q_rot, &sincos, offset);

            key = Tensor::cat(&[k_rot, k_pass], -1);
            query = Tensor::cat(&[q_rot, q_pass], -1);
        } else {
            let sincos = fixed_pos_embedding(&key, seq_len);
            key = apply_rotary_pos_emb(&key, &sincos, offset);
            query = apply_rotary_pos_emb(&query, &sincos, offset);
        }

        key = key.permute([0, 2, 1, 3]);
        query = query.permute([0, 2, 1, 3]);

        if let Some(layer_past) = layer_past {
            key = Tensor::cat(&[&layer_past.prev_key, &key], -2);
            value = Tensor::cat(&[&layer_past.prev_value, &value], -2);
        }

        let present = self.use_cache.then(|| LayerState {
            prev_key: key.copy(),
            prev_value: value.copy(),
        });

        let (attn_output, attn_weights) =
            self.attention(&query, &key, &value, attention_mask, train);

        let attn_output = Self::merge_heads(&attn_output, self.n_head, self.dim_per_head)
            .apply(&self.out_proj)
            .apply_t(&self.resid_dropout, train);

        let attn_weights = self.output_attentions.then_some(attn_weights);

        (attn_output, present, attn_weights)
    }
}

fn fixed_pos_embedding(x: &Tensor, seq_len: i64) -> (Tensor, Tensor) {
    let dim = x.size()[x.size().len() - 1];
    let inv_freq = 1.0
        / Tensor::pow_scalar(
            10_000,
            &(Tensor::arange_start_step(0, dim, 2, (x.kind(), x.device())) / dim),
        );
    let sinusoid_inp = Tensor::einsum(
        "i , j -> i j",
        &[Tensor::arange(seq_len, (x.kind(), x.device())), inv_freq],
        None::<i64>,
    );
    (sinusoid_inp.sin(), sinusoid_inp.cos())
}

fn apply_rotary_pos_emb(x: &Tensor, (sin, cos): &(Tensor, Tensor), offset: i64) -> Tensor {
    let sin = duplicate_interleave(sin).i((NewAxis, offset..x.size()[1] + offset, NewAxis, ..));
    let cos = duplicate_interleave(cos).i((NewAxis, offset..x.size()[1] + offset, NewAxis, ..));
    (x * cos) + (rotate_every_two(x) * sin)
}

/// A simple version of `torch.repeat_interleave` for duplicating a matrix while interleaving the copy.
fn duplicate_interleave(m: &Tensor) -> Tensor {
    let dim0 = m.size()[0];
    m.view([-1, 1]) // flatten the matrix
        .repeat([1, 2]) // repeat all elements into the 2nd dimension
        .view([dim0, -1]) // reshape into a matrix, interleaving the copy
}

fn rotate_every_two(x: &Tensor) -> Tensor {
    let x1 = x.slice(3, 0, x.size()[3], 2);
    let x2 = x.slice(3, 1, x.size()[3], 2);
    Tensor::stack(&[-x2, x1], -1).flatten(-2, -1)
}
