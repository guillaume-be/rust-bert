// Copyright 2020 The Trax Authors and The HuggingFace Inc. team.
// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
use crate::reformer::attention_utils::{look_adjacent, split_seq_length_dim_to, stable_argsort};
use crate::reformer::ReformerConfig;
use crate::RustBertError;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::borrow::Borrow;
use std::convert::{TryFrom, TryInto};
use tch::nn::LinearConfig;
use tch::{nn, Kind, Tensor};

#[derive(Debug)]
/// # Cache for Reformer attention layers
/// Stores the cached value of buckets and states to avoid recalculation (e.g. at each generation step)
pub struct LayerState {
    /// Cached buckets
    pub prev_buckets: Tensor,
    /// Cached states
    pub prev_states: Tensor,
}

#[allow(non_camel_case_types)]
#[derive(Clone, Debug, Serialize, Deserialize, Copy, PartialEq, Eq, Hash)]
/// # Attention type for the model (local or LSH)
pub enum AttentionType {
    /// Local attention
    local,
    /// LSH attention
    lsh,
}

#[derive(Debug)]
pub enum NumBuckets {
    Array(Vec<i64>),
    Integer(i64),
}

impl TryFrom<&Value> for NumBuckets {
    type Error = RustBertError;

    fn try_from(value: &Value) -> Result<NumBuckets, RustBertError> {
        match value {
            Value::Number(value) => {
                if let Some(integer_value) = value.as_i64() {
                    Ok(NumBuckets::Integer(integer_value))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "Expected an integer or list of integers for num_buckets".to_string(),
                    ))
                }
            }
            Value::Array(array) => {
                let mut num_bucket = vec![];
                for value in array {
                    if let Some(integer_value) = value.as_i64() {
                        num_bucket.push(integer_value);
                    } else {
                        return Err(RustBertError::InvalidConfigurationError(
                            "Expected an integer or list of integers for num_buckets".to_string(),
                        ));
                    }
                }
                Ok(NumBuckets::Array(num_bucket))
            }
            _ => Err(RustBertError::InvalidConfigurationError(
                "Expected an integer or list of integers for num_buckets".to_string(),
            )),
        }
    }
}

impl NumBuckets {
    pub fn max_bucket(&self) -> i64 {
        match self {
            NumBuckets::Integer(int_value) => *int_value,
            NumBuckets::Array(array_value) => {
                let mut product = 1;
                for value in array_value {
                    product *= value;
                }
                product
            }
        }
    }
}

#[derive(Debug)]
/// # LSH Self Attention for Reformer model
pub struct LSHSelfAttention {
    chunk_length: i64,
    num_hashes: i64,
    num_buckets: NumBuckets,
    num_chunks_before: i64,
    num_chunks_after: i64,
    hash_seed: Option<i64>,
    is_decoder: bool,
    max_position_embeddings: i64,
    dropout: Dropout,
    num_attention_heads: i64,
    attention_head_size: i64,
    all_head_size: i64,
    hidden_size: i64,
    query_key: nn::Linear,
    value: nn::Linear,
    self_mask_value: Tensor,
    mask_value: Tensor,
}

impl LSHSelfAttention {
    pub fn new<'p, P>(p: P, config: &ReformerConfig) -> Result<LSHSelfAttention, RustBertError>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let chunk_length = config.lsh_attn_chunk_length.unwrap_or(64);
        let num_hashes = config.num_hashes;
        let num_buckets = (&config.num_buckets).try_into()?;
        let num_chunks_before = config.lsh_num_chunks_before.unwrap_or(1);
        let num_chunks_after = config.lsh_num_chunks_after.unwrap_or(0);
        let hash_seed = config.hash_seed;
        let is_decoder = config.is_decoder;
        let max_position_embeddings = config.max_position_embeddings;

        let dropout = Dropout::new(config.hidden_dropout_prob);

        let num_attention_heads = config.num_attention_heads;
        let attention_head_size = config.attention_head_size;
        let all_head_size = num_attention_heads * attention_head_size;
        let hidden_size = config.hidden_size;

        let linear_config = LinearConfig {
            bias: false,
            ..Default::default()
        };
        let query_key = nn::linear(p / "query_key", hidden_size, all_head_size, linear_config);
        let value = nn::linear(p / "value", hidden_size, all_head_size, linear_config);

        let self_mask_value = Tensor::of_slice(&[-1e5]);
        let mask_value = Tensor::of_slice(&[1e9]);

        Ok(LSHSelfAttention {
            chunk_length,
            num_hashes,
            num_buckets,
            num_chunks_before,
            num_chunks_after,
            hash_seed,
            is_decoder,
            max_position_embeddings,
            dropout,
            num_attention_heads,
            attention_head_size,
            all_head_size,
            hidden_size,
            query_key,
            value,
            self_mask_value,
            mask_value,
        })
    }

    fn query_per_attention_head(&self, hidden_states: &Tensor) -> Tensor {
        let per_head_query_key = self
            .query_key
            .ws
            .reshape(&[
                self.num_attention_heads,
                self.attention_head_size,
                self.hidden_size,
            ])
            .transpose(-2, -1);
        Tensor::einsum("balh,ahr->balr", &[hidden_states, &per_head_query_key])
    }

    fn value_per_attention_head(&self, hidden_states: &Tensor) -> Tensor {
        let per_head_value = self
            .value
            .ws
            .reshape(&[
                self.num_attention_heads,
                self.attention_head_size,
                self.hidden_size,
            ])
            .transpose(-2, -1);
        Tensor::einsum("balh,ahr->balr", &[hidden_states, &per_head_value])
    }

    fn hash_vectors(
        &self,
        vectors: &Tensor,
        num_hashes: i64,
        attention_mask: Option<&Tensor>,
        increase_num_buckets: bool,
    ) -> Tensor {
        let input_shape = vectors.size();
        let batch_size = input_shape[0];

        let (rotation_size, mut num_buckets) = match &self.num_buckets {
            NumBuckets::Integer(num_buckets) => (*num_buckets, *num_buckets),
            NumBuckets::Array(buckets_array) => {
                let mut rotation_size = 0;
                let mut num_buckets = 1;
                for bucket_factor in buckets_array {
                    rotation_size += bucket_factor;
                    num_buckets *= bucket_factor;
                }
                (rotation_size, num_buckets)
            }
        };

        let vectors = vectors.detach();
        if let Some(seed) = self.hash_seed {
            tch::manual_seed(seed);
        };

        let rotations_shape = [
            self.num_attention_heads,
            *input_shape.last().unwrap(),
            num_hashes,
            rotation_size / 2,
        ];
        let random_rotations = Tensor::randn(&rotations_shape, (vectors.kind(), vectors.device()));
        let rotated_vectors = Tensor::einsum("bmtd,mdhr->bmhtr", &[vectors, random_rotations]);

        let mut buckets = match &self.num_buckets {
            NumBuckets::Integer(_) => {
                Tensor::cat(&[&rotated_vectors, &(-1 * &rotated_vectors)], -1).argmax(-1, false)
            }
            NumBuckets::Array(buckets_array) if buckets_array.len() == 1 => {
                Tensor::cat(&[&rotated_vectors, &(-1 * &rotated_vectors)], -1).argmax(-1, false)
            }
            NumBuckets::Array(buckets_array) => {
                let (mut buckets, mut cur_sum, mut cur_product) = (
                    Tensor::zeros(&[1], (rotated_vectors.kind(), rotated_vectors.device())),
                    0,
                    1,
                );
                for bucket_factor in buckets_array {
                    let rotated_vector_factor =
                        rotated_vectors.slice(-1, cur_sum, cur_sum + bucket_factor / 2, 1);
                    let rotated_vector_factor = Tensor::cat(
                        &[&rotated_vector_factor, &(-1 * &rotated_vector_factor)],
                        -1,
                    )
                    .argmax(-1, false);
                    cur_sum += bucket_factor / 2;
                    buckets = buckets + cur_product * rotated_vector_factor;
                    cur_product *= bucket_factor;
                }
                buckets
            }
        };

        if let Some(attention_mask_value) = attention_mask {
            if i64::from(attention_mask_value.sum(Kind::Int))
                < batch_size * *attention_mask_value.size().last().unwrap()
            {
                num_buckets += 1;
                let buckets_mask = attention_mask_value
                    .unsqueeze(1)
                    .unsqueeze(1)
                    .expand(&buckets.size(), true)
                    .to_kind(Kind::Int8);
                buckets = buckets.where1(&buckets_mask, &Tensor::of_slice(&[num_buckets - 1]))
            } else if increase_num_buckets {
                num_buckets += 1;
            }
        } else if increase_num_buckets {
            num_buckets += 1;
        }

        let offsets = (num_buckets * Tensor::arange(num_hashes, (Kind::Int64, buckets.device())))
            .view([1, 1, -1, 1]);
        let mut offset_shape = vec![batch_size, self.num_attention_heads];
        offset_shape.extend_from_slice(&offsets.size()[offsets.size().len() - 2..]);
        let offsets = offsets.expand(&offset_shape, true);

        (buckets + offsets).flatten(2, 3)
    }

    fn get_sorted_bucket_indices_undo_sorted_bucket_indices(
        &self,
        buckets: &Tensor,
    ) -> (Tensor, Tensor) {
        tch::no_grad(|| {
            let sorted_bucket_indices = stable_argsort(buckets, -1);
            let indices = Tensor::arange(
                *sorted_bucket_indices.size().last().unwrap(),
                (Kind::Int64, buckets.device()),
            )
            .view([1, 1, -1])
            .expand(sorted_bucket_indices.size().as_slice(), true);

            let mut undo_sorted_bucket_indices = sorted_bucket_indices.new_empty(
                sorted_bucket_indices.size().as_slice(),
                (Kind::Int64, buckets.device()),
            );
            let _ = undo_sorted_bucket_indices.scatter_(-1, &sorted_bucket_indices, &indices);
            (sorted_bucket_indices, undo_sorted_bucket_indices)
        })
    }

    fn attend(
        &self,
        query_vectors: Tensor,
        mut key_vectors: Tensor,
        mut value_vectors: Tensor,
        sorted_bucket_indices_per_hash: Tensor,
        attention_mask: Option<&Tensor>,
        do_standard_self_attention: bool,
        do_cached_attention: bool,
        train: bool,
    ) -> Result<(Tensor, Tensor, Tensor), RustBertError> {
        if !do_standard_self_attention {
            key_vectors = look_adjacent(key_vectors, self.num_chunks_before, self.num_chunks_after);
            value_vectors =
                look_adjacent(value_vectors, self.num_chunks_before, self.num_chunks_after);
        }

        let mut query_key_dots = query_vectors.matmul(&key_vectors.transpose(-1, -2));

        let (query_bucket_idx, key_value_bucket_idx) = if !do_standard_self_attention {
            let query_bucket_idx = split_seq_length_dim_to(
                &sorted_bucket_indices_per_hash,
                -1,
                self.chunk_length,
                self.num_attention_heads,
                None,
            )?;
            let key_value_bucket_idx = look_adjacent(
                query_bucket_idx.copy(),
                self.num_chunks_before,
                self.num_chunks_after,
            );
            (query_bucket_idx, key_value_bucket_idx)
        } else if do_standard_self_attention & (query_key_dots.dim() > 4) {
            let mut query_shape = sorted_bucket_indices_per_hash.size();
            query_shape[sorted_bucket_indices_per_hash.dim() - 1] = 1;
            let query_bucket_idx = sorted_bucket_indices_per_hash.new_full(
                query_shape.as_slice(),
                i64::from(sorted_bucket_indices_per_hash.max()),
                (Kind::Int64, sorted_bucket_indices_per_hash.device()),
            );
            (query_bucket_idx, sorted_bucket_indices_per_hash)
        } else if do_standard_self_attention & (query_key_dots.dim() <= 4) {
            let query_bucket_idx = query_key_dots.select(3, -1).ones_like()
                * (query_key_dots.size().last().unwrap() - 1);
            let mut query_shape = query_bucket_idx.size();
            query_shape[query_bucket_idx.dim() - 1] = -1;
            let key_value_bucket_idx = Tensor::arange(
                *query_key_dots.size().last().unwrap(),
                (Kind::Int64, query_key_dots.device()),
            )
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(query_shape.as_slice(), true);
            (query_bucket_idx, key_value_bucket_idx)
        } else {
            (
                sorted_bucket_indices_per_hash.copy(),
                sorted_bucket_indices_per_hash,
            )
        };

        if !do_cached_attention {
            let mask = self.compute_attention_mask(
                &query_bucket_idx,
                &key_value_bucket_idx,
                attention_mask,
                query_key_dots.size().as_slice(),
                do_standard_self_attention,
            );

            if let Some(mask) = mask {
                query_key_dots = query_key_dots.where1(&mask, &self.mask_value);
            }
        }
        {
            let self_mask = query_bucket_idx
                .unsqueeze(-1)
                .ne1(&key_value_bucket_idx.unsqueeze(-2));
            query_key_dots = query_key_dots.where1(&self_mask, &self.self_mask_value);
        }

        let mut logits = query_key_dots.logsumexp(&[-1], true);
        let attention_probs = (query_key_dots - &logits)
            .exp()
            .apply_t(&self.dropout, train);

        let mut out_vectors = attention_probs.matmul(&value_vectors);
        if out_vectors.dim() > 4 {
            logits = logits.flatten(2, 3).squeeze1(-1);
            out_vectors = out_vectors.flatten(2, 3)
        }

        Ok((out_vectors, logits, attention_probs))
    }

    fn compute_attention_mask(
        &self,
        query_indices: &Tensor,
        key_indices: &Tensor,
        attention_mask: Option<&Tensor>,
        query_key_dot_shape: &[i64],
        do_standard_self_attention: bool,
    ) -> Option<Tensor> {
        let attention_mask = if let Some(attention_mask_value) = attention_mask {
            let mut attention_mask = attention_mask_value.unsqueeze(1);
            if !do_standard_self_attention {
                let mut query_shape = query_indices.size();
                query_shape[query_indices.dim() - 1] = -1;
                attention_mask = attention_mask
                    .unsqueeze(1)
                    .expand(query_shape.as_slice(), true);
                attention_mask = attention_mask.gather(-1, key_indices, false);
            }
            Some(
                attention_mask
                    .unsqueeze(-2)
                    .expand(query_key_dot_shape, true),
            )
        } else {
            None
        };

        if self.is_decoder {
            let causal_mask = query_indices.unsqueeze(-1).ge1(&key_indices.unsqueeze(-2));
            let attention_mask = if let Some(attention_mask) = attention_mask {
                causal_mask * attention_mask
            } else {
                causal_mask
            };
            Some(attention_mask)
        } else {
            None
        }
    }

    fn get_relevant_hidden_states_and_buckets(
        &self,
        query_vectors: &Tensor,
        attention_mask: Option<&Tensor>,
        num_hashes: i64,
        hidden_states: &Tensor,
        past_states: &Tensor,
        past_buckets: &Tensor,
    ) -> (Tensor, Tensor, Tensor) {
        let hidden_states = Tensor::cat(&[past_states, hidden_states], 1);
        let hidden_states_shape = hidden_states.size();
        let (batch_size, sequence_length) = (hidden_states_shape[0], hidden_states_shape[1]);
        let max_bucket = self.num_buckets.max_bucket();
        let increase_num_buckets = i64::from(past_buckets.max()) > num_hashes * max_bucket - 1;

        let query_buckets = self.hash_vectors(
            query_vectors,
            num_hashes,
            attention_mask,
            increase_num_buckets,
        );

        let concat_buckets = Tensor::cat(&[past_buckets, &query_buckets.unsqueeze(-1)], -1);
        let bucket_indices = stable_argsort(&concat_buckets, -1);

        let relevant_bucket_indices = bucket_indices.eq(bucket_indices.size().last().unwrap() - 1);
        let relevant_bucket_indices_chunk =
            self.expand_to_indices_in_relevant_chunk(&relevant_bucket_indices, sequence_length);
        let relevant_bucket_indices_chunk = bucket_indices.index(&[
            relevant_bucket_indices_chunk.get(0),
            relevant_bucket_indices_chunk.get(1),
            relevant_bucket_indices_chunk.get(2),
            relevant_bucket_indices_chunk.get(3),
        ]);

        let bucket_indices_batch_offset = sequence_length
            * (batch_size
                * Tensor::arange(
                    *relevant_bucket_indices_chunk.size().last().unwrap(),
                    (Kind::Int64, hidden_states.device()),
                )
                / *relevant_bucket_indices_chunk.size().last().unwrap());

        let relevant_bucket_indices_chunk_all_batch =
            &relevant_bucket_indices_chunk + bucket_indices_batch_offset;

        let relevant_hidden_states = hidden_states
            .reshape(&[-1, self.hidden_size])
            .index_select(0, &relevant_bucket_indices_chunk_all_batch)
            .reshape(&[batch_size, self.num_attention_heads, -1, self.hidden_size]);

        let relevant_bucket_indices_chunk = relevant_bucket_indices_chunk.reshape(&[
            batch_size,
            self.num_attention_heads,
            num_hashes,
            -1,
        ]);

        (
            relevant_hidden_states,
            relevant_bucket_indices_chunk,
            query_buckets,
        )
    }

    fn expand_to_indices_in_relevant_chunk(
        &self,
        indices: &Tensor,
        sequence_length: i64,
    ) -> Tensor {
        let start_indices_chunk = ((indices.select(1, -1) / self.chunk_length)
            - self.num_chunks_before)
            * self.chunk_length;
        let total_chunk_size =
            self.chunk_length * (1 + self.num_chunks_before + self.num_chunks_after);

        let expanded_start_indices = start_indices_chunk
            .unsqueeze(-1)
            .expand(&[indices.size()[0], total_chunk_size], true);
        let chunk_sequence_indices = expanded_start_indices
            + Tensor::arange(total_chunk_size, (Kind::Int64, indices.device()))
                .unsqueeze(0)
                .expand(&[indices.size()[0], total_chunk_size], true);

        let chunk_sequence_indices = chunk_sequence_indices.flatten(0, 1) / sequence_length;
        let mut indices = indices
            .unsqueeze(1)
            .expand(&[indices.size()[0], total_chunk_size, -1], true)
            .flatten(0, 1);

        indices.select(1, -1).copy_(&chunk_sequence_indices);
        indices
    }
}
