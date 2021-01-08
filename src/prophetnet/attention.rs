// Copyright 2020 The Microsoft Authors and The HuggingFace Inc. team.
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
use crate::prophetnet::ProphetNetConfig;
use crate::RustBertError;
use std::borrow::Borrow;
use tch::{nn, Kind, Tensor};

#[derive(Debug)]
/// # Cache for ProphetNet attention layers
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

pub struct ProphetNetAttention {
    key_proj: nn::Linear,
    value_proj: nn::Linear,
    query_proj: nn::Linear,
    out_proj: nn::Linear,
    dropout: Dropout,
    attention_dropout: Dropout,
    num_attention_heads: i64,
    head_dim: i64,
    output_attentions: bool,
}

impl ProphetNetAttention {
    pub fn new<'p, P>(
        p: P,
        config: ProphetNetConfig,
        num_attention_heads: i64,
    ) -> Result<ProphetNetAttention, RustBertError>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let dropout = Dropout::new(config.dropout);
        let attention_dropout = Dropout::new(config.attention_dropout);

        if config.hidden_size % num_attention_heads != 0 {
            return Err(RustBertError::InvalidConfigurationError(format!(
                "Invalid number of heads for self attention, {} not a multiple of {}",
                config.hidden_size, num_attention_heads
            )));
        }

        let head_dim = config.hidden_size / num_attention_heads;

        let key_proj = nn::linear(
            p / "key_proj",
            config.hidden_size,
            config.hidden_size,
            Default::default(),
        );

        let value_proj = nn::linear(
            p / "value_proj",
            config.hidden_size,
            config.hidden_size,
            Default::default(),
        );

        let query_proj = nn::linear(
            p / "query_proj",
            config.hidden_size,
            config.hidden_size,
            Default::default(),
        );

        let out_proj = nn::linear(
            p / "out_proj",
            config.hidden_size,
            config.hidden_size,
            Default::default(),
        );

        let output_attentions = config.output_attentions.unwrap_or(false);

        Ok(ProphetNetAttention {
            key_proj,
            value_proj,
            query_proj,
            out_proj,
            dropout,
            attention_dropout,
            num_attention_heads,
            head_dim,
            output_attentions,
        })
    }

    fn flatten(&self, x: Tensor, dim_0: i64, bs: i64) -> Tensor {
        x.contiguous()
            .view((dim_0, bs * self.num_attention_heads, self.head_dim))
            .transpose(0, 1)
    }

    pub fn forward_t(
        &self,
        hidden_states: &Tensor,
        key_value_states: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        mut layer_state: Option<LayerState>,
        train: bool,
    ) -> (Tensor, Option<Tensor>, Option<LayerState>) {
        let hidden_states_size = hidden_states.size();
        let (sequence_length, batch_size, hidden_size) = (
            hidden_states_size[0],
            hidden_states_size[1],
            hidden_states_size[2],
        );
        let is_cross_attention = key_value_states.is_some();
        let query_states = hidden_states.apply(&self.query_proj) / (self.head_dim as f64).sqrt();
        let query_states = self.flatten(query_states, sequence_length, batch_size);

        let (key_states, value_states) = if !is_cross_attention {
            let key_states = self.flatten(hidden_states.apply(&self.key_proj), -1, batch_size);
            let value_states = self.flatten(hidden_states.apply(&self.value_proj), -1, batch_size);
            (key_states, value_states)
        } else if layer_state.is_none() {
            let key_states = self.flatten(
                key_value_states.unwrap().apply(&self.key_proj),
                -1,
                batch_size,
            );
            let value_states = self.flatten(
                key_value_states.unwrap().apply(&self.value_proj),
                -1,
                batch_size,
            );
            (key_states, value_states)
        } else {
            let past_state = layer_state.as_ref().unwrap();
            (
                past_state.prev_key.view([
                    batch_size * self.num_attention_heads,
                    -1,
                    self.head_dim,
                ]),
                past_state.prev_value.view([
                    batch_size * self.num_attention_heads,
                    -1,
                    self.head_dim,
                ]),
            )
        };

        if is_cross_attention {
            layer_state.as_mut().unwrap().prev_key =
                key_states.view([batch_size, self.num_attention_heads, -1, self.head_dim]);
            layer_state.as_mut().unwrap().prev_value =
                value_states.view([batch_size, self.num_attention_heads, -1, self.head_dim]);
        };

        let key_sequence_key = key_states.size()[1];
        let mut attention_weights = query_states.bmm(&key_states.transpose(1, 2));

        if let Some(attention_mask) = attention_mask {
            attention_weights = attention_weights + attention_mask;
        };

        let attention_weights_reshaped = attention_weights.view([
            batch_size,
            self.num_attention_heads,
            sequence_length,
            key_sequence_key,
        ]);

        let attention_probs = attention_weights_reshaped
            .view([
                batch_size * self.num_attention_heads,
                sequence_length,
                key_sequence_key,
            ])
            .softmax(-1, Kind::Float)
            .apply_t(&self.attention_dropout, train);

        let attention_output = attention_probs
            .bmm(&value_states)
            .transpose(0, 1)
            .contiguous()
            .view([sequence_length, batch_size, hidden_size])
            .apply(&self.out_proj)
            .apply_t(&self.dropout, train);

        let attention_weights = if self.output_attentions {
            Some(attention_weights_reshaped)
        } else {
            None
        };
        (attention_output, attention_weights, layer_state)
    }
}

fn compute_relative_buckets(
    num_buckets: i64,
    max_distance: i64,
    relative_positions: &Tensor,
    bidirectional: bool,
) -> Tensor {
    let inverse_relative_positions = -relative_positions;

    let (num_buckets, relative_positions_bucket, inverse_relative_positions) = if bidirectional {
        let num_buckets = num_buckets / 2;
        let relative_position_bucket =
            inverse_relative_positions.lt(0).totype(Kind::Int) * num_buckets;
        let inverse_relative_position = inverse_relative_positions.abs();
        (
            num_buckets,
            relative_position_bucket,
            inverse_relative_position,
        )
    } else {
        (
            num_buckets,
            relative_positions.zeros_like(),
            inverse_relative_positions.max1(&inverse_relative_positions.zeros_like()),
        )
    };
    let max_exact = num_buckets / 2;
    let is_small = inverse_relative_positions.lt(max_exact);
    let max_exact_f64 = max_exact as f64;
    let val_if_large = (inverse_relative_positions.totype(Kind::Float) / max_exact_f64).log()
        / (max_distance as f64 / max_exact_f64).log2()
        * (num_buckets as f64 - max_exact_f64)
        + max_exact_f64;

    let val_if_large = val_if_large
        .min1(&(val_if_large.ones_like() * (num_buckets as f64 - 1.0)))
        .totype(Kind::Int);

    let relative_positions_bucket = relative_positions_bucket
        + is_small.where1(&inverse_relative_positions.totype(Kind::Int), &val_if_large);

    relative_positions_bucket
}

fn compute_all_stream_relative_bucket(
    num_buckets: i64,
    max_distance: i64,
    position_ids: &Tensor,
) -> (Tensor, Tensor) {
    let main_stream_relative_positions =
        position_ids
            .unsqueeze(1)
            .repeat(&[1, *position_ids.size().last().unwrap(), 1])
            - position_ids.unsqueeze(-1);

    let predicting_stream_relative_positions = Tensor::cat(&[&(position_ids - 1), position_ids], 1)
        .unsqueeze(1)
        .repeat(&[1, *position_ids.size().last().unwrap(), 1])
        - position_ids.unsqueeze(-1);

    let main_relative_position_buckets = compute_relative_buckets(
        num_buckets,
        max_distance,
        &main_stream_relative_positions,
        false,
    );

    let predict_relative_position_buckets = compute_relative_buckets(
        num_buckets,
        max_distance,
        &predicting_stream_relative_positions,
        false,
    );

    (
        main_relative_position_buckets,
        predict_relative_position_buckets,
    )
}
