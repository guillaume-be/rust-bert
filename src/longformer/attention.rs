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
use crate::longformer::LongformerConfig;
use std::borrow::Borrow;
use tch::{nn, Kind, Tensor};

pub struct LongformerSelfAttention {
    query: nn::Linear,
    key: nn::Linear,
    value: nn::Linear,
    query_global: nn::Linear,
    key_global: nn::Linear,
    value_global: nn::Linear,
    dropout: Dropout,
    attention_window: i64,
    one_sided_attention_window_size: i64,
    num_heads: i64,
    head_dim: i64,
    embed_dim: i64,
}

impl LongformerSelfAttention {
    pub fn new<'p, P>(p: P, config: &LongformerConfig, layer_id: i64) -> LongformerSelfAttention
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let num_heads = config.num_attention_heads;
        let head_dim = config.hidden_size / num_heads;
        let embed_dim = config.hidden_size;

        let query = nn::linear(
            p / "query",
            config.hidden_size,
            config.hidden_size,
            Default::default(),
        );
        let key = nn::linear(
            p / "key",
            config.hidden_size,
            config.hidden_size,
            Default::default(),
        );
        let value = nn::linear(
            p / "value",
            config.hidden_size,
            config.hidden_size,
            Default::default(),
        );
        let query_global = nn::linear(
            p / "query_global",
            config.hidden_size,
            config.hidden_size,
            Default::default(),
        );
        let key_global = nn::linear(
            p / "key_global",
            config.hidden_size,
            config.hidden_size,
            Default::default(),
        );
        let value_global = nn::linear(
            p / "value_global",
            config.hidden_size,
            config.hidden_size,
            Default::default(),
        );

        let dropout = Dropout::new(config.attention_probs_dropout_prob);
        let attention_window = config.attention_window[layer_id as usize];
        let one_sided_attention_window_size = attention_window / 2;

        LongformerSelfAttention {
            query,
            key,
            value,
            query_global,
            key_global,
            value_global,
            dropout,
            attention_window,
            one_sided_attention_window_size,
            num_heads,
            head_dim,
            embed_dim,
        }
    }

    fn pad_and_transpose_last_two_dims(&self, hidden_states: &Tensor, padding: &[i64]) -> Tensor {
        hidden_states.constant_pad_nd(padding).transpose(-1, -2)
    }

    fn pad_and_diagonalize(&self, chunked_hidden_states: &Tensor) -> Tensor {
        let chunked_hidden_states_shape = chunked_hidden_states.size();
        let (total_num_heads, num_chunks, window_overlap, hidden_dim) = (
            chunked_hidden_states_shape[0],
            chunked_hidden_states_shape[1],
            chunked_hidden_states_shape[2],
            chunked_hidden_states_shape[3],
        );

        chunked_hidden_states
            .constant_pad_nd(&[0, window_overlap + 1])
            .view([total_num_heads, num_chunks, -1])
            .slice(2, 0, -window_overlap, 1)
            .view([
                total_num_heads,
                num_chunks,
                window_overlap,
                window_overlap + hidden_dim,
            ])
            .slice(3, 0, -1, 1)
    }

    fn chunk(&self, hidden_states: &Tensor, window_overlap: i64) -> Tensor {
        let hidden_states_shape = hidden_states.size();
        let hidden_states = hidden_states.view([
            hidden_states_shape[0],
            hidden_states_shape[1] / (window_overlap * 2),
            window_overlap * 2,
            hidden_states_shape[2],
        ]);

        let mut chunk_size = hidden_states.size();
        chunk_size[1] = chunk_size[1] * 2 - 1;

        let mut chunk_stride = hidden_states.stride();
        chunk_stride[1] = chunk_stride[1] / 2;

        hidden_states.as_strided(chunk_size.as_slice(), chunk_stride.as_slice(), None)
    }

    fn mask_invalid_locations(&self, input_tensor: &mut Tensor, affected_sequence_length: i64) {
        let input_size = input_tensor.size();
        let beginning_input_size = vec![
            input_size[0],
            affected_sequence_length,
            input_size[2],
            affected_sequence_length + 1,
        ];
        let ending_input_size = vec![
            input_size[0],
            affected_sequence_length,
            input_size[2],
            affected_sequence_length + 1,
        ];

        let beginning_mask = Tensor::ones(
            &[affected_sequence_length, affected_sequence_length + 1],
            (Kind::Int, input_tensor.device()),
        )
        .tril(0)
        .flip(&[0])
        .unsqueeze(2)
        .unsqueeze(0);

        let ending_mask = beginning_mask
            .flip(&[1, 3])
            .expand(ending_input_size.as_slice(), true)
            .eq(1);

        let beginning_mask = beginning_mask
            .expand(beginning_input_size.as_slice(), true)
            .eq(1);

        let _ = input_tensor
            .slice(1, 0, affected_sequence_length, 1)
            .slice(3, 0, affected_sequence_length + 1, 1)
            .masked_fill_(&beginning_mask, std::f64::NEG_INFINITY);

        let _ = input_tensor
            .narrow(1, -affected_sequence_length, affected_sequence_length)
            .narrow(
                3,
                -(affected_sequence_length + 1),
                affected_sequence_length + 1,
            )
            .masked_fill_(&ending_mask, std::f64::NEG_INFINITY);
    }
}
