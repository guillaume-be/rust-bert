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

    fn sliding_chunks_query_key_matmul(
        &self,
        query: &Tensor,
        key: &Tensor,
        window_overlap: i64,
    ) -> Tensor {
        let (batch_size, sequence_length, num_heads, head_dim) = query.size4().unwrap();
        let chunks_count = sequence_length / window_overlap - 1;

        let query =
            query
                .transpose(1, 2)
                .reshape(&[batch_size * num_heads, sequence_length, head_dim]);
        let key = key
            .transpose(1, 2)
            .reshape(&[batch_size * num_heads, sequence_length, head_dim]);

        let query = self.chunk(&query, window_overlap);
        let key = self.chunk(&key, window_overlap);

        let diagonal_chunked_attention_scores = self.pad_and_transpose_last_two_dims(
            &Tensor::einsum("bcxd,bcyd->bcxy", &[query, key]),
            &[0, 0, 0, 1],
        );

        let mut diagonal_attention_scores = Tensor::empty(
            &[
                batch_size * num_heads,
                chunks_count + 1,
                window_overlap,
                window_overlap * 2 + 1,
            ],
            (Kind::Float, diagonal_chunked_attention_scores.device()),
        );

        let diagonal_attention_scores_size = diagonal_attention_scores.size();
        let diagonal_chunked_attention_scores_size = diagonal_chunked_attention_scores.size();

        diagonal_attention_scores
            .slice(1, 0, -1, 1)
            .slice(3, window_overlap, diagonal_attention_scores_size[3], 1)
            .copy_(
                &diagonal_chunked_attention_scores
                    .slice(2, 0, window_overlap, 1)
                    .slice(3, 0, window_overlap + 1, 1),
            );

        diagonal_attention_scores
            .select(1, -1)
            .slice(3, window_overlap, diagonal_attention_scores_size[3], 1)
            .copy_(
                &diagonal_chunked_attention_scores
                    .select(1, -1)
                    .slice(
                        2,
                        window_overlap,
                        diagonal_chunked_attention_scores_size[2],
                        1,
                    )
                    .slice(3, 0, window_overlap + 1, 1),
            );

        diagonal_attention_scores
            .slice(1, 1, diagonal_attention_scores_size[1], 1)
            .slice(3, 0, window_overlap, 1)
            .copy_(
                &diagonal_chunked_attention_scores
                    .slice(2, -(window_overlap + 1), -1, 1)
                    .slice(
                        3,
                        window_overlap + 1,
                        diagonal_chunked_attention_scores_size[3],
                        1,
                    ),
            );

        diagonal_attention_scores
            .select(1, 0)
            .slice(2, 1, window_overlap, 1)
            .slice(3, 1, window_overlap, 1)
            .copy_(
                &diagonal_chunked_attention_scores
                    .select(1, 0)
                    .slice(2, 0, window_overlap - 1, 1)
                    .slice(
                        3,
                        1 - window_overlap,
                        diagonal_chunked_attention_scores_size[3],
                        1,
                    ),
            );

        let _ = diagonal_attention_scores
            .view_(&[
                batch_size,
                num_heads,
                sequence_length,
                2 * window_overlap + 1,
            ])
            .transpose_(2, 1);

        self.mask_invalid_locations(&mut diagonal_attention_scores, window_overlap);

        diagonal_attention_scores
    }

    fn sliding_chunks_matmul_attention_probas_value(
        &self,
        attention_probas: &Tensor,
        value: &Tensor,
        window_overlap: i64,
    ) -> Tensor {
        let (batch_size, sequence_length, num_heads, head_dim) = value.size4().unwrap();
        let chunk_counts = sequence_length / window_overlap - 1;

        let chunked_attention_probas = attention_probas.transpose(1, 2).reshape(&[
            batch_size * num_heads,
            sequence_length / window_overlap,
            window_overlap,
            2 * window_overlap + 1,
        ]);

        let value =
            value
                .transpose(1, 2)
                .reshape(&[batch_size * num_heads, sequence_length, head_dim]);

        let padded_value = (value + 1).constant_pad_nd(&[0, 0, window_overlap, window_overlap]) - 1;
        let chunked_value_size = &[
            batch_size * num_heads,
            chunk_counts + 1,
            3 * window_overlap,
            head_dim,
        ];
        let chunked_value_stride = padded_value.stride();
        let chunked_value_stride = &[
            chunked_value_stride[0],
            window_overlap * chunked_value_stride[1],
            chunked_value_stride[1],
            chunked_value_stride[2],
        ];

        let chunked_value = padded_value.as_strided(chunked_value_size, chunked_value_stride, None);
        let chunked_attention_probas = self.pad_and_diagonalize(&chunked_attention_probas);

        Tensor::einsum(
            "bcwd,bcdh->bcwh",
            &[chunked_attention_probas, chunked_value],
        )
        .view([batch_size, num_heads, sequence_length, head_dim])
        .transpose(1, 2)
    }

    fn get_global_attention_indices(
        &self,
        is_index_global_attn: &Tensor,
    ) -> (i64, Vec<Tensor>, Vec<Tensor>, Vec<Tensor>) {
        let num_global_attention_indices = is_index_global_attn.sum1(&[1], false, Kind::Int64);
        let max_num_global_attention_indices = i64::from(num_global_attention_indices.max());
        let is_index_global_attn_nonzero = is_index_global_attn.nonzero_numpy();

        let is_local_index_global_attention = Tensor::arange(
            max_num_global_attention_indices,
            (Kind::Int64, is_index_global_attn.device()),
        )
        .lt1(&num_global_attention_indices.unsqueeze(-1));

        let is_local_index_global_attention_nonzero =
            is_local_index_global_attention.nonzero_numpy();

        let is_local_index_no_global_attention_nonzero =
            is_local_index_global_attention.eq(0).nonzero_numpy();

        (
            max_num_global_attention_indices,
            is_index_global_attn_nonzero,
            is_local_index_global_attention_nonzero,
            is_local_index_no_global_attention_nonzero,
        )
    }

    fn concat_with_global_key_attention_probas(
        &self,
        key_vectors: &Tensor,
        query_vectors: &Tensor,
        max_num_global_attention_indices: i64,
        is_index_global_attn_nonzero: Vec<Tensor>,
        is_local_index_global_attention_nonzero: Vec<Tensor>,
        is_local_index_no_global_attention_nonzero: Vec<Tensor>,
    ) -> Tensor {
        let batch_size = key_vectors.size()[0];

        let key_vectors_only_global = Tensor::zeros(
            &[
                batch_size,
                max_num_global_attention_indices,
                self.num_heads,
                self.head_dim,
            ],
            (Kind::Float, key_vectors.device()),
        );

        key_vectors_only_global
            .index(is_local_index_global_attention_nonzero.as_slice())
            .copy_(&key_vectors.index(is_index_global_attn_nonzero.as_slice()));

        let attention_probas_from_global_key = Tensor::einsum(
            "blhd,bshd->blhs",
            &[query_vectors, &key_vectors_only_global],
        );

        let _ = attention_probas_from_global_key
            .index_select(0, &is_local_index_no_global_attention_nonzero[0])
            .index_select(3, &is_local_index_no_global_attention_nonzero[1])
            .fill_(-10000f64);

        attention_probas_from_global_key
    }

    fn compute_attention_output_with_global_indices(
        &self,
        value_vectors: &Tensor,
        attention_probas: &Tensor,
        max_num_global_attention_indices: i64,
        is_index_global_attn_nonzero: Vec<Tensor>,
        is_local_index_global_attention_nonzero: Vec<Tensor>,
    ) -> Tensor {
        let batch_size = attention_probas.size()[0];

        let attention_probas_only_global =
            attention_probas.narrow(-1, 0, max_num_global_attention_indices);
        let value_vectors_only_global = Tensor::zeros(
            &[
                batch_size,
                max_num_global_attention_indices,
                self.num_heads,
                self.head_dim,
            ],
            (Kind::Float, value_vectors.device()),
        );

        value_vectors_only_global
            .index(is_local_index_global_attention_nonzero.as_slice())
            .copy_(&value_vectors.index(is_index_global_attn_nonzero.as_slice()));

        let attention_output_only_global = attention_probas_only_global
            .transpose(1, 2)
            .matmul(&value_vectors_only_global.transpose(1, 2))
            .transpose(1, 2);

        let attention_probas_without_global = attention_probas
            .narrow(
                -1,
                max_num_global_attention_indices,
                *attention_probas.size().last().unwrap() - max_num_global_attention_indices,
            )
            .contiguous();

        let attn_output_without_global = self.sliding_chunks_matmul_attention_probas_value(
            &attention_probas_without_global,
            &value_vectors,
            self.one_sided_attention_window_size,
        );
        attention_output_only_global + attn_output_without_global
    }

    fn compute_global_attention_output_from_hidden(
        &self,
        hidden_states: &Tensor,
        max_num_global_attention_indices: i64,
        is_index_global_attn_nonzero: Vec<Tensor>,
        is_local_index_global_attention_nonzero: Vec<Tensor>,
        is_local_index_no_global_attention_nonzero: Vec<Tensor>,
        is_index_masked: &Tensor,
        train: bool,
    ) -> (Tensor, Tensor) {
        let hidden_states_shape = hidden_states.size();
        let (sequence_length, batch_size) = (hidden_states_shape[0], hidden_states_shape[1]);

        let global_attention_hidden_states = Tensor::zeros(
            &[max_num_global_attention_indices, batch_size, self.embed_dim],
            (Kind::Float, hidden_states.device()),
        );

        global_attention_hidden_states
            .index(
                is_local_index_global_attention_nonzero
                    .iter()
                    .rev()
                    .collect::<Vec<&Tensor>>()
                    .as_slice(),
            )
            .copy_(
                &hidden_states.index(
                    is_index_global_attn_nonzero
                        .iter()
                        .rev()
                        .collect::<Vec<&Tensor>>()
                        .as_slice(),
                ),
            );

        let global_query_vectors_only_global = (global_attention_hidden_states
            .apply(&self.query_global)
            / (self.head_dim as f64).sqrt())
        .contiguous()
        .view([
            max_num_global_attention_indices,
            batch_size * self.num_heads,
            self.head_dim,
        ])
        .transpose(0, 1);
        let global_key_vectors = hidden_states
            .apply(&self.key_global)
            .contiguous()
            .view([-1, batch_size * self.num_heads, self.head_dim])
            .transpose(0, 1);
        let global_value_vectors = hidden_states
            .apply(&self.value_global)
            .contiguous()
            .view([-1, batch_size * self.num_heads, self.head_dim])
            .transpose(0, 1);

        let mut global_attention_scores = global_query_vectors_only_global
            .bmm(&global_key_vectors.transpose(1, 2))
            .view([
                batch_size,
                self.num_heads,
                max_num_global_attention_indices,
                sequence_length,
            ]);

        let _ = global_attention_scores
            .index_select(0, &is_local_index_no_global_attention_nonzero[0])
            .index_select(2, &is_local_index_no_global_attention_nonzero[1])
            .fill_(-10000f64);

        let _ = global_attention_scores
            .masked_fill_(&is_index_masked.unsqueeze(1).unsqueeze(1), -10000f64);

        let _ = global_attention_scores.view_(&[
            batch_size * self.num_heads,
            max_num_global_attention_indices,
            sequence_length,
        ]);

        let global_attention_probas = global_attention_scores
            .softmax(-1, Kind::Float)
            .apply_t(&self.dropout, train);

        let global_attention_output = global_attention_probas.bmm(&global_value_vectors);

        let _ = global_attention_probas.view_(&[
            batch_size,
            self.num_heads,
            max_num_global_attention_indices,
            sequence_length,
        ]);
        let _ = global_attention_output.view_(&[
            batch_size,
            self.num_heads,
            max_num_global_attention_indices,
            self.head_dim,
        ]);

        (global_attention_output, global_attention_probas)
    }
}
