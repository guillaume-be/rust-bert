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
use crate::common::kind::get_negative_infinity;
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
    one_sided_attention_window_size: i64,
    num_heads: i64,
    head_dim: i64,
    embed_dim: i64,
    output_attentions: bool,
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
        let one_sided_attention_window_size = config.attention_window[layer_id as usize] / 2;
        let output_attentions = config.output_attentions.unwrap_or(false);

        LongformerSelfAttention {
            query,
            key,
            value,
            query_global,
            key_global,
            value_global,
            dropout,
            one_sided_attention_window_size,
            num_heads,
            head_dim,
            embed_dim,
            output_attentions,
        }
    }

    fn pad_and_transpose_last_two_dims(&self, hidden_states: &Tensor, padding: &[i64]) -> Tensor {
        let output = hidden_states.constant_pad_nd(padding);
        let mut output_shape = output.size();
        let last_dim = output_shape.pop().unwrap();
        let second_last_dim = output_shape.pop().unwrap();
        output_shape.push(last_dim);
        output_shape.push(second_last_dim);
        output.view(output_shape.as_slice())
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
        .unsqueeze(0)
        .unsqueeze(2);

        let ending_mask = beginning_mask.flip(&[1, 3]);

        let beginning_mask = beginning_mask
            .expand(beginning_input_size.as_slice(), true)
            .eq(1);

        let ending_mask = ending_mask.expand(ending_input_size.as_slice(), true).eq(1);

        let _ = input_tensor
            .slice(1, 0, affected_sequence_length, 1)
            .slice(3, 0, affected_sequence_length + 1, 1)
            .masked_fill_(
                &beginning_mask,
                get_negative_infinity(input_tensor.kind()).unwrap(),
            );

        let _ = input_tensor
            .narrow(1, -affected_sequence_length, affected_sequence_length)
            .narrow(
                3,
                -(affected_sequence_length + 1),
                affected_sequence_length + 1,
            )
            .masked_fill_(
                &ending_mask,
                get_negative_infinity(input_tensor.kind()).unwrap(),
            );
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

        let diagonal_attention_scores = Tensor::empty(
            &[
                batch_size * num_heads,
                chunks_count + 1,
                window_overlap,
                window_overlap * 2 + 1,
            ],
            (
                diagonal_chunked_attention_scores.kind(),
                diagonal_chunked_attention_scores.device(),
            ),
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
            .slice(2, window_overlap, diagonal_attention_scores_size[3], 1)
            .copy_(
                &diagonal_chunked_attention_scores
                    .select(1, -1)
                    .slice(
                        1,
                        window_overlap,
                        diagonal_chunked_attention_scores_size[2],
                        1,
                    )
                    .slice(2, 0, window_overlap + 1, 1),
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
            .slice(1, 1, window_overlap, 1)
            .slice(2, 1, window_overlap, 1)
            .copy_(
                &diagonal_chunked_attention_scores
                    .select(1, 0)
                    .slice(1, 0, window_overlap - 1, 1)
                    .slice(
                        2,
                        1 - window_overlap,
                        diagonal_chunked_attention_scores_size[3],
                        1,
                    ),
            );

        let mut diagonal_attention_scores = diagonal_attention_scores
            .view([
                batch_size,
                num_heads,
                sequence_length,
                2 * window_overlap + 1,
            ])
            .transpose(2, 1);

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
    ) -> GlobalAttentionIndices {
        let num_global_attention_indices =
            is_index_global_attn.sum_dim_intlist(&[1], false, Kind::Int64);
        let max_num_global_attention_indices = i64::from(num_global_attention_indices.max());
        let is_index_global_attn_nonzero = is_index_global_attn
            .nonzero_numpy()
            .into_iter()
            .map(Some)
            .collect();

        let is_local_index_global_attention = Tensor::arange(
            max_num_global_attention_indices,
            (Kind::Int64, is_index_global_attn.device()),
        )
        .lt_tensor(&num_global_attention_indices.unsqueeze(-1));

        let is_local_index_global_attention_nonzero = is_local_index_global_attention
            .nonzero_numpy()
            .into_iter()
            .map(Some)
            .collect();

        let is_local_index_no_global_attention_nonzero = is_local_index_global_attention
            .eq(0)
            .nonzero_numpy()
            .into_iter()
            .map(Some)
            .collect();

        GlobalAttentionIndices {
            max_num_global_attention_indices,
            is_index_global_attn_nonzero,
            is_local_index_global_attention_nonzero,
            is_local_index_no_global_attention_nonzero,
        }
    }

    fn concat_with_global_key_attention_probas(
        &self,
        key_vectors: &Tensor,
        query_vectors: &Tensor,
        max_num_global_attention_indices: i64,
        is_index_global_attn_nonzero: &[Option<Tensor>],
        is_local_index_global_attention_nonzero: &[Option<Tensor>],
        is_local_index_no_global_attention_nonzero: &[Option<Tensor>],
    ) -> Tensor {
        let batch_size = key_vectors.size()[0];

        let mut key_vectors_only_global = Tensor::zeros(
            &[
                batch_size,
                max_num_global_attention_indices,
                self.num_heads,
                self.head_dim,
            ],
            (key_vectors.kind(), key_vectors.device()),
        );

        let _ = key_vectors_only_global.index_put_(
            is_local_index_global_attention_nonzero,
            &key_vectors.index(is_index_global_attn_nonzero),
            false,
        );

        let attention_probas_from_global_key = Tensor::einsum(
            "blhd,bshd->blhs",
            &[query_vectors, &key_vectors_only_global],
        );

        let _ = attention_probas_from_global_key
            .index_select(
                0,
                is_local_index_no_global_attention_nonzero[0]
                    .as_ref()
                    .unwrap(),
            )
            .index_select(
                3,
                is_local_index_no_global_attention_nonzero[1]
                    .as_ref()
                    .unwrap(),
            )
            .fill_(-10000f64);

        attention_probas_from_global_key
    }

    fn compute_attention_output_with_global_indices(
        &self,
        value_vectors: &Tensor,
        attention_probas: &Tensor,
        max_num_global_attention_indices: i64,
        is_index_global_attn_nonzero: &[Option<Tensor>],
        is_local_index_global_attention_nonzero: &[Option<Tensor>],
    ) -> Tensor {
        let batch_size = attention_probas.size()[0];

        let attention_probas_only_global =
            attention_probas.narrow(-1, 0, max_num_global_attention_indices);
        let mut value_vectors_only_global = Tensor::zeros(
            &[
                batch_size,
                max_num_global_attention_indices,
                self.num_heads,
                self.head_dim,
            ],
            (value_vectors.kind(), value_vectors.device()),
        );

        let _ = value_vectors_only_global.index_put_(
            is_local_index_global_attention_nonzero,
            &value_vectors.index(is_index_global_attn_nonzero),
            false,
        );

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
            value_vectors,
            self.one_sided_attention_window_size,
        );
        attention_output_only_global + attn_output_without_global
    }

    fn compute_global_attention_output_from_hidden(
        &self,
        hidden_states: &Tensor,
        max_num_global_attention_indices: i64,
        is_index_global_attn_nonzero: &[Option<Tensor>],
        is_local_index_global_attention_nonzero: &[Option<Tensor>],
        is_local_index_no_global_attention_nonzero: &[Option<Tensor>],
        is_index_masked: &Tensor,
        train: bool,
    ) -> (Tensor, Tensor) {
        let hidden_states_shape = hidden_states.size();
        let (sequence_length, batch_size) = (hidden_states_shape[0], hidden_states_shape[1]);

        let mut global_attention_hidden_states = Tensor::zeros(
            &[max_num_global_attention_indices, batch_size, self.embed_dim],
            (hidden_states.kind(), hidden_states.device()),
        );

        let _ = global_attention_hidden_states.index_put_(
            is_local_index_global_attention_nonzero
                .iter()
                .rev()
                .map(|o| o.as_ref())
                .collect::<Vec<Option<&Tensor>>>()
                .as_slice(),
            &hidden_states.index(
                is_index_global_attn_nonzero
                    .iter()
                    .rev()
                    .map(|o| o.as_ref())
                    .collect::<Vec<Option<&Tensor>>>()
                    .as_slice(),
            ),
            false,
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

        let global_attention_scores = global_query_vectors_only_global
            .bmm(&global_key_vectors.transpose(1, 2))
            .view([
                batch_size,
                self.num_heads,
                max_num_global_attention_indices,
                sequence_length,
            ]);

        let _ = global_attention_scores
            .index_select(
                0,
                is_local_index_no_global_attention_nonzero[0]
                    .as_ref()
                    .unwrap(),
            )
            .index_select(
                2,
                is_local_index_no_global_attention_nonzero[1]
                    .as_ref()
                    .unwrap(),
            )
            .fill_(-10000_f64);

        let global_attention_scores = global_attention_scores
            .masked_fill(&is_index_masked.unsqueeze(1).unsqueeze(1), -10000_f64)
            .view([
                batch_size * self.num_heads,
                max_num_global_attention_indices,
                sequence_length,
            ]);

        let global_attention_probas = global_attention_scores
            .softmax(-1, global_attention_scores.kind())
            .apply_t(&self.dropout, train);

        let global_attention_output = global_attention_probas.bmm(&global_value_vectors);

        let global_attention_probas = global_attention_probas.view([
            batch_size,
            self.num_heads,
            max_num_global_attention_indices,
            sequence_length,
        ]);
        let global_attention_output = global_attention_output.view([
            batch_size,
            self.num_heads,
            max_num_global_attention_indices,
            self.head_dim,
        ]);

        (global_attention_output, global_attention_probas)
    }

    pub fn forward_t(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
        is_index_masked: &Tensor,
        is_index_global_attention: &Tensor,
        is_global_attention: bool,
        train: bool,
    ) -> (Tensor, Option<Tensor>, Option<Tensor>) {
        let hidden_states = hidden_states.transpose(0, 1);
        let query_vectors = hidden_states.apply(&self.query) / (self.head_dim as f64).sqrt();
        let key_vectors = hidden_states.apply(&self.key);
        let value_vectors = hidden_states.apply(&self.value);

        let (sequence_length, batch_size, embed_dim) = hidden_states.size3().unwrap();

        let query_vectors = query_vectors
            .view([sequence_length, batch_size, self.num_heads, self.head_dim])
            .transpose(0, 1);
        let key_vectors = key_vectors
            .view([sequence_length, batch_size, self.num_heads, self.head_dim])
            .transpose(0, 1);

        let mut attention_scores = self.sliding_chunks_query_key_matmul(
            &query_vectors,
            &key_vectors,
            self.one_sided_attention_window_size,
        );

        let remove_from_windowed_attention_mask = attention_mask.ne(0).unsqueeze(-1).unsqueeze(-1);
        let float_mask = remove_from_windowed_attention_mask
            .totype(attention_scores.kind())
            .masked_fill(&remove_from_windowed_attention_mask, -10000.0);

        let diagonal_mask = self.sliding_chunks_query_key_matmul(
            &Tensor::ones(
                float_mask.size().as_slice(),
                (float_mask.kind(), float_mask.device()),
            ),
            &float_mask,
            self.one_sided_attention_window_size,
        );

        attention_scores = attention_scores + &diagonal_mask;

        let (
            max_num_global_attention_indices,
            is_index_global_attn_nonzero,
            is_local_index_global_attention_nonzero,
            is_local_index_no_global_attention_nonzero,
        ) = if is_global_attention {
            let global_attention_indices =
                self.get_global_attention_indices(is_index_global_attention);

            let global_key_attention_scores = self.concat_with_global_key_attention_probas(
                &key_vectors,
                &query_vectors,
                global_attention_indices.max_num_global_attention_indices,
                global_attention_indices
                    .is_index_global_attn_nonzero
                    .as_slice(),
                global_attention_indices
                    .is_local_index_global_attention_nonzero
                    .as_slice(),
                global_attention_indices
                    .is_local_index_no_global_attention_nonzero
                    .as_slice(),
            );

            attention_scores = Tensor::cat(&[&global_key_attention_scores, &attention_scores], -1);
            (
                Some(global_attention_indices.max_num_global_attention_indices),
                Some(global_attention_indices.is_index_global_attn_nonzero),
                Some(global_attention_indices.is_local_index_global_attention_nonzero),
                Some(global_attention_indices.is_local_index_no_global_attention_nonzero),
            )
        } else {
            (None, None, None, None)
        };

        let mut attention_probas = attention_scores
            .softmax(-1, attention_scores.kind())
            .masked_fill(&is_index_masked.unsqueeze(-1).unsqueeze(-1), 0.0)
            .apply_t(&self.dropout, train);

        let value_vectors = value_vectors
            .view([sequence_length, batch_size, self.num_heads, self.head_dim])
            .transpose(0, 1);

        let attention_output = if is_global_attention {
            self.compute_attention_output_with_global_indices(
                &value_vectors,
                &attention_probas,
                max_num_global_attention_indices.unwrap(),
                is_index_global_attn_nonzero.as_ref().unwrap(),
                is_local_index_global_attention_nonzero.as_ref().unwrap(),
            )
        } else {
            self.sliding_chunks_matmul_attention_probas_value(
                &attention_probas,
                &value_vectors,
                self.one_sided_attention_window_size,
            )
        };

        let mut attention_output =
            attention_output
                .transpose(0, 1)
                .reshape(&[sequence_length, batch_size, embed_dim]);

        let global_attention_probas = if is_global_attention {
            let (global_attention_output, global_attention_probas) = self
                .compute_global_attention_output_from_hidden(
                    &hidden_states,
                    max_num_global_attention_indices.unwrap(),
                    is_index_global_attn_nonzero.as_ref().unwrap(),
                    is_local_index_global_attention_nonzero.as_ref().unwrap(),
                    is_local_index_no_global_attention_nonzero.as_ref().unwrap(),
                    is_index_masked,
                    train,
                );

            let nonzero_global_attention_output = global_attention_output.transpose(1, 2).index(&[
                Some(
                    is_local_index_global_attention_nonzero.as_ref().unwrap()[0]
                        .as_ref()
                        .unwrap(),
                ),
                Some(
                    is_local_index_global_attention_nonzero.as_ref().unwrap()[1]
                        .as_ref()
                        .unwrap(),
                ),
            ]);
            let _ = attention_output.index_put_(
                is_index_global_attn_nonzero
                    .as_ref()
                    .unwrap()
                    .iter()
                    .rev()
                    .map(|o| o.as_ref())
                    .collect::<Vec<Option<&Tensor>>>()
                    .as_slice(),
                &nonzero_global_attention_output.view([
                    is_local_index_global_attention_nonzero.as_ref().unwrap()[0]
                        .as_ref()
                        .unwrap()
                        .size()[0],
                    -1,
                ]),
                false,
            );

            let _ = attention_probas.index_put_(
                is_index_global_attn_nonzero.as_ref().unwrap(),
                &Tensor::zeros(
                    attention_probas
                        .index(is_index_global_attn_nonzero.as_ref().unwrap())
                        .size()
                        .as_slice(),
                    (attention_output.kind(), attention_output.device()),
                ),
                false,
            );

            Some(global_attention_probas)
        } else {
            None
        };

        let attention_probas = if self.output_attentions {
            Some(attention_probas)
        } else {
            None
        };

        let global_attention_probas = if self.output_attentions {
            global_attention_probas
        } else {
            None
        };

        (
            attention_output.transpose(0, 1),
            attention_probas,
            global_attention_probas,
        )
    }
}

struct GlobalAttentionIndices {
    max_num_global_attention_indices: i64,
    is_index_global_attn_nonzero: Vec<Option<Tensor>>,
    is_local_index_global_attention_nonzero: Vec<Option<Tensor>>,
    is_local_index_no_global_attention_nonzero: Vec<Option<Tensor>>,
}
