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
use crate::t5::layer_norm::T5LayerNorm;
use crate::t5::T5Config;
use std::borrow::Borrow;
use tch::nn::LinearConfig;
use tch::{nn, Device, Kind, Tensor};

#[derive(Debug)]
/// # Cache for T5 attention layers
/// Stores the cached value of key, value and key to avoid recalculation (e.g. at each generation step)
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

#[derive(Debug)]
pub struct T5Attention {
    is_decoder: bool,
    is_bidirectional: bool,
    has_relative_attention_bias: bool,
    relative_attention_num_buckets: i64,
    d_kv: i64,
    n_heads: i64,
    dropout: Dropout,
    inner_dim: i64,
    output_attentions: bool,
    store_cache: bool,
    query: nn::Linear,
    key: nn::Linear,
    value: nn::Linear,
    output: nn::Linear,
    relative_attention_bias: Option<nn::Embedding>,
}

impl T5Attention {
    pub fn new<'p, P>(
        p: P,
        config: &T5Config,
        is_decoder: bool,
        is_bidirectional: bool,
        store_cache: bool,
        output_attentions: bool,
        has_relative_attention_bias: bool,
    ) -> T5Attention
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let linear_config = LinearConfig {
            bias: false,
            ..Default::default()
        };

        let inner_dim = config.num_heads * config.d_kv;
        let key = nn::linear(p / "k", config.d_model, inner_dim, linear_config);
        let value = nn::linear(p / "v", config.d_model, inner_dim, linear_config);
        let query = nn::linear(p / "q", config.d_model, inner_dim, linear_config);
        let output = nn::linear(p / "o", inner_dim, config.d_model, linear_config);

        let dropout = Dropout::new(config.dropout_rate);
        let relative_attention_bias = if has_relative_attention_bias {
            Some(nn::embedding(
                p / "relative_attention_bias",
                config.relative_attention_num_buckets,
                config.num_heads,
                Default::default(),
            ))
        } else {
            None
        };

        T5Attention {
            is_decoder,
            is_bidirectional,
            has_relative_attention_bias,
            relative_attention_num_buckets: config.relative_attention_num_buckets,
            d_kv: config.d_kv,
            n_heads: config.num_heads,
            dropout,
            inner_dim,
            output_attentions,
            store_cache,
            query,
            key,
            value,
            output,
            relative_attention_bias,
        }
    }

    fn unshape(&self, x: Tensor, bs: i64) -> Tensor {
        x.transpose(1, 2)
            .contiguous()
            .view((bs, -1, self.inner_dim))
    }

    fn shape(&self, x: Tensor, bs: i64) -> Tensor {
        x.view((bs, -1, self.n_heads, self.d_kv)).transpose(1, 2)
    }

    pub fn forward_t(
        &self,
        hidden_states: &Tensor,
        key_value_states: Option<&Tensor>,
        position_bias: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        mut layer_state: Option<LayerState>,
        query_length: Option<i64>,
        train: bool,
    ) -> (Tensor, Option<Tensor>, Option<Tensor>, Option<LayerState>) {
        let input_size = hidden_states.size();
        let (bs, seq_length, _) = (input_size[0], input_size[1], input_size[2]);

        let real_seq_length = if layer_state.is_some() {
            match query_length {
                Some(value) => value,
                None => seq_length + layer_state.as_ref().unwrap().prev_key.size()[2],
            }
        } else {
            seq_length
        };

        let key_length = match key_value_states {
            Some(value) => value.size()[1],
            None => real_seq_length,
        };

        let q: Tensor = self.shape(hidden_states.as_ref().apply(&self.query), bs);

        let (mut k, mut v) = if key_value_states.is_none() {
            (
                self.shape(hidden_states.apply(&self.key), bs),
                self.shape(hidden_states.apply(&self.value), bs),
            )
        } else {
            (
                self.shape(key_value_states.as_ref().unwrap().apply(&self.key), bs),
                self.shape(key_value_states.as_ref().unwrap().apply(&self.value), bs),
            )
        };

        if layer_state.is_some() {
            let layer_state = layer_state.as_ref().unwrap();
            if key_value_states.is_none() {
                k = Tensor::cat(&[&layer_state.prev_key, &k], 2);
                v = Tensor::cat(&[&layer_state.prev_value, &v], 2);
            } else {
                k = layer_state.prev_key.copy();
                v = layer_state.prev_value.copy();
            }
        };

        layer_state = if self.is_decoder & self.store_cache {
            Some(LayerState {
                prev_key: k.copy(),
                prev_value: v.copy(),
            })
        } else {
            None
        };

        let mut scores = Tensor::einsum("bnqd,bnkd->bnqk", &[q, k]);

        let calculated_position_bias = if position_bias.is_none() {
            let mut temp_value = if self.has_relative_attention_bias {
                self.compute_bias(real_seq_length, key_length, hidden_states.device())
            } else {
                Tensor::zeros(
                    &[1, self.n_heads, real_seq_length, key_length],
                    (scores.kind(), scores.device()),
                )
            };
            if layer_state.is_some() {
                let length = temp_value.size()[2];
                temp_value = temp_value.slice(2, length - seq_length, length, 1);
            };
            if let Some(attention_mask) = attention_mask {
                temp_value = temp_value + attention_mask
            };
            Some(temp_value)
        } else {
            None
        };

        let position_bias = if let Some(position_bias) = position_bias {
            position_bias
        } else {
            calculated_position_bias.as_ref().unwrap()
        };

        scores += position_bias;

        let attention_weights = scores
            .softmax(-1, scores.kind())
            .apply_t(&self.dropout, train);
        let context = self
            .unshape(attention_weights.matmul(&v), bs)
            .apply(&self.output);

        let attention_weights = if self.output_attentions {
            Some(attention_weights)
        } else {
            None
        };

        let position_bias = if self.has_relative_attention_bias {
            calculated_position_bias
        } else {
            None
        };

        (context, attention_weights, position_bias, layer_state)
    }

    fn get_relative_position_bucket(
        &self,
        relative_position: &Tensor,
        bidirectional: bool,
        num_buckets: i64,
        max_distance: i64,
    ) -> Tensor {
        let n = -relative_position;
        let mut num_buckets = num_buckets;
        let mut ret = n.zeros_like();
        let n = if bidirectional {
            num_buckets /= 2;
            ret += n.lt(0).to_kind(Kind::Int64) * num_buckets;
            n.abs()
        } else {
            n.max_other(&n.zeros_like())
        };

        let max_exact = num_buckets / 2;
        let is_small = n.lt(max_exact);

        let value_if_large: Tensor = ((n.to_kind(Kind::Float) / max_exact as f64).log2()
            / (max_distance as f64 / max_exact as f64).log2()
            * (num_buckets - max_exact) as f64)
            .to_kind(Kind::Int64)
            + max_exact;

        let value_if_large = value_if_large.min_other(&value_if_large.full_like(num_buckets - 1));
        ret += n.where_self(&is_small, &value_if_large);
        ret
    }

    fn compute_bias(&self, q_len: i64, k_len: i64, device: Device) -> Tensor {
        let context_position = Tensor::arange(q_len, (Kind::Int64, device)).unsqueeze(1);
        let memory_position = Tensor::arange(k_len, (Kind::Int64, device)).unsqueeze(0);
        let relative_position = memory_position - context_position;

        let rp_bucket = self.get_relative_position_bucket(
            &relative_position,
            self.is_bidirectional,
            self.relative_attention_num_buckets,
            128,
        );
        rp_bucket
            .apply(self.relative_attention_bias.as_ref().unwrap())
            .permute(&[2, 0, 1])
            .unsqueeze(0)
    }
}

pub struct T5LayerSelfAttention {
    self_attention: T5Attention,
    layer_norm: T5LayerNorm,
    dropout: Dropout,
}

impl T5LayerSelfAttention {
    pub fn new<'p, P>(
        p: P,
        config: &T5Config,
        has_relative_attention_bias: bool,
        is_decoder: bool,
        store_cache: bool,
        output_attentions: bool,
    ) -> T5LayerSelfAttention
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let self_attention = T5Attention::new(
            p / "SelfAttention",
            config,
            is_decoder,
            !is_decoder,
            store_cache,
            output_attentions,
            has_relative_attention_bias,
        );

        let layer_norm =
            T5LayerNorm::new(p / "layer_norm", config.d_model, config.layer_norm_epsilon);
        let dropout = Dropout::new(config.dropout_rate);

        T5LayerSelfAttention {
            self_attention,
            layer_norm,
            dropout,
        }
    }

    pub fn forward_t(
        &self,
        hidden_states: &Tensor,
        position_bias: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        layer_state: Option<LayerState>,
        train: bool,
    ) -> (Tensor, Option<Tensor>, Option<Tensor>, Option<LayerState>) {
        let norm_x = hidden_states.apply(&self.layer_norm);
        let (y, attention_weights, position_bias, layer_state) = self.self_attention.forward_t(
            &norm_x,
            None,
            position_bias,
            attention_mask,
            layer_state,
            None,
            train,
        );

        let output = hidden_states + y.apply_t(&self.dropout, train);

        (output, attention_weights, position_bias, layer_state)
    }
}

pub struct T5LayerCrossAttention {
    encoder_decoder_attention: T5Attention,
    layer_norm: T5LayerNorm,
    dropout: Dropout,
}

impl T5LayerCrossAttention {
    pub fn new<'p, P>(
        p: P,
        config: &T5Config,
        has_relative_attention_bias: bool,
        is_decoder: bool,
        store_cache: bool,
        output_attentions: bool,
    ) -> T5LayerCrossAttention
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let encoder_decoder_attention = T5Attention::new(
            p / "EncDecAttention",
            config,
            is_decoder,
            true,
            store_cache,
            output_attentions,
            has_relative_attention_bias,
        );

        let layer_norm =
            T5LayerNorm::new(p / "layer_norm", config.d_model, config.layer_norm_epsilon);
        let dropout = Dropout::new(config.dropout_rate);

        T5LayerCrossAttention {
            encoder_decoder_attention,
            layer_norm,
            dropout,
        }
    }

    pub fn forward_t(
        &self,
        hidden_states: &Tensor,
        kv: Option<&Tensor>,
        position_bias: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        layer_state: Option<LayerState>,
        query_length: Option<i64>,
        train: bool,
    ) -> (Tensor, Option<Tensor>, Option<Tensor>, Option<LayerState>) {
        let norm_x = hidden_states.apply(&self.layer_norm);

        let (y, attention_weights, position_bias, layer_state) =
            self.encoder_decoder_attention.forward_t(
                &norm_x,
                kv,
                position_bias,
                attention_mask,
                layer_state,
                query_length,
                train,
            );

        let output = hidden_states + y.apply_t(&self.dropout, train);

        (output, attention_weights, position_bias, layer_state)
    }
}
