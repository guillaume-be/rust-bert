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
use crate::reformer::attention_utils::{
    look_adjacent, merge_hidden_size_dim, retrieve_relevant_hidden_states, reverse_sort,
    split_hidden_size_dim, split_seq_length_dim_to, stable_argsort,
};
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
    pub prev_buckets: Option<Tensor>,
    /// Cached states
    pub prev_states: Tensor,
}

impl Clone for LayerState {
    fn clone(&self) -> Self {
        let prev_buckets = self
            .prev_buckets
            .as_ref()
            .map(|prev_buckets| prev_buckets.copy());
        LayerState {
            prev_buckets,
            prev_states: self.prev_states.copy(),
        }
    }
}

impl LayerState {
    pub(crate) fn reorder_cache(&mut self, new_indices: &Tensor) {
        self.prev_states = self.prev_states.index_select(0, new_indices);
        if let Some(prev_buckets_value) = &self.prev_buckets {
            self.prev_buckets = Some(prev_buckets_value.index_select(0, new_indices));
        }
    }
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

fn apply_mask_with_fp16_compatibility(
    input_tensor: &Tensor,
    mask: &Tensor,
    fp32_value: &Tensor,
    fp16_value: &Tensor,
) -> Result<Tensor, RustBertError> {
    Ok(match input_tensor.kind() {
        Kind::Float => input_tensor.where_self(
            &mask.to_kind(Kind::Bool),
            &fp32_value.to_device(input_tensor.device()),
        ),
        Kind::Half => input_tensor.where_self(
            &mask.to_kind(Kind::Bool),
            &fp16_value
                .to_device(input_tensor.device()),
        ),
        Kind::BFloat16 => input_tensor.where_self(
            &mask.to_kind(Kind::Bool),
            &fp16_value
                .to_kind(input_tensor.kind())
                .to_device(input_tensor.device()),
        ),
        _ => {
            return Err(RustBertError::ValueError(format!(
                "Type not supported: {:?}, supported types are Float (single precision), Half and BFloat16 (half precision)",
                input_tensor.kind()
            )))
        }
    })
}

/// # LSH Self Attention for Reformer model
pub struct LSHSelfAttention {
    chunk_length: i64,
    num_hashes: i64,
    num_buckets: NumBuckets,
    num_chunks_before: i64,
    num_chunks_after: i64,
    hash_seed: Option<i64>,
    is_decoder: bool,
    dropout: Dropout,
    num_attention_heads: i64,
    attention_head_size: i64,
    hidden_size: i64,
    query_key: nn::Linear,
    value: nn::Linear,
    self_mask_value_fp32: Tensor,
    mask_value_fp32: Tensor,
    self_mask_value_fp16: Tensor,
    mask_value_fp16: Tensor,
    use_cache: bool,
    output_attentions: bool,
}

impl LSHSelfAttention {
    pub fn new<'p, P>(
        p: P,
        config: &ReformerConfig,
        output_attentions: bool,
        use_cache: bool,
    ) -> Result<LSHSelfAttention, RustBertError>
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

        let self_mask_value_fp32 = Tensor::from_slice(&[-1e5])
            .to_kind(Kind::Float)
            .to(p.device());
        let mask_value_fp32 = Tensor::from_slice(&[-1e9])
            .to_kind(Kind::Float)
            .to(p.device());

        let self_mask_value_fp16 = Tensor::from_slice(&[-1e3])
            .to_kind(Kind::Half)
            .to(p.device());
        let mask_value_fp16 = Tensor::from_slice(&[-1e4])
            .to_kind(Kind::Half)
            .to(p.device());

        Ok(LSHSelfAttention {
            chunk_length,
            num_hashes,
            num_buckets,
            num_chunks_before,
            num_chunks_after,
            hash_seed,
            is_decoder,
            dropout,
            num_attention_heads,
            attention_head_size,
            hidden_size,
            query_key,
            value,
            self_mask_value_fp32,
            mask_value_fp32,
            self_mask_value_fp16,
            mask_value_fp16,
            use_cache,
            output_attentions,
        })
    }

    fn query_per_attention_head(&self, hidden_states: &Tensor) -> Tensor {
        let per_head_query_key = self
            .query_key
            .ws
            .reshape([
                self.num_attention_heads,
                self.attention_head_size,
                self.hidden_size,
            ])
            .transpose(-2, -1);
        Tensor::einsum(
            "balh,ahr->balr",
            &[hidden_states, &per_head_query_key],
            None::<i64>,
        )
    }

    fn value_per_attention_head(&self, hidden_states: &Tensor) -> Tensor {
        let per_head_value = self
            .value
            .ws
            .reshape([
                self.num_attention_heads,
                self.attention_head_size,
                self.hidden_size,
            ])
            .transpose(-2, -1);
        Tensor::einsum(
            "balh,ahr->balr",
            &[hidden_states, &per_head_value],
            None::<i64>,
        )
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
        let random_rotations = Tensor::randn(rotations_shape, (vectors.kind(), vectors.device()));
        let rotated_vectors = Tensor::einsum(
            "bmtd,mdhr->bmhtr",
            &[vectors, random_rotations],
            None::<i64>,
        );

        let mut buckets = match &self.num_buckets {
            NumBuckets::Integer(_) => {
                Tensor::cat(&[&rotated_vectors, &(-1 * &rotated_vectors)], -1).argmax(-1, false)
            }
            NumBuckets::Array(buckets_array) if buckets_array.len() == 1 => {
                Tensor::cat(&[&rotated_vectors, &(-1 * &rotated_vectors)], -1).argmax(-1, false)
            }
            NumBuckets::Array(buckets_array) => {
                let (mut buckets, mut cur_sum, mut cur_product) = (
                    Tensor::zeros([1], (rotated_vectors.kind(), rotated_vectors.device())),
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
            if i64::try_from(attention_mask_value.sum(Kind::Int)).unwrap()
                < batch_size * *attention_mask_value.size().last().unwrap()
            {
                num_buckets += 1;
                let buckets_mask = attention_mask_value
                    .unsqueeze(1)
                    .unsqueeze(1)
                    .expand(buckets.size(), true)
                    .to_kind(Kind::Bool);
                buckets = buckets.where_self(
                    &buckets_mask,
                    &Tensor::from_slice(&[num_buckets - 1])
                        .to_kind(buckets.kind())
                        .to(buckets_mask.device()),
                )
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
            let sorted_bucket_indices = stable_argsort(buckets, buckets.dim() as i64 - 1);
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
        } else if do_cached_attention & (query_key_dots.dim() > 4) {
            let mut query_shape = sorted_bucket_indices_per_hash.size();
            query_shape[sorted_bucket_indices_per_hash.dim() - 1] = 1;
            let query_bucket_idx = sorted_bucket_indices_per_hash.new_full(
                query_shape.as_slice(),
                i64::try_from(sorted_bucket_indices_per_hash.max()).unwrap(),
                (Kind::Int64, sorted_bucket_indices_per_hash.device()),
            );
            (query_bucket_idx, sorted_bucket_indices_per_hash)
        } else if do_cached_attention & (query_key_dots.dim() <= 4) {
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
                query_key_dots = apply_mask_with_fp16_compatibility(
                    &query_key_dots,
                    &mask,
                    &self.mask_value_fp32,
                    &self.mask_value_fp16,
                )?;
            }
        }
        {
            let self_mask = query_bucket_idx
                .unsqueeze(-1)
                .ne_tensor(&key_value_bucket_idx.unsqueeze(-2));
            query_key_dots = apply_mask_with_fp16_compatibility(
                &query_key_dots,
                &self_mask,
                &self.self_mask_value_fp32,
                &self.self_mask_value_fp16,
            )?;
        }

        let mut logits = query_key_dots.logsumexp([-1], true);
        let attention_probs = (query_key_dots - &logits)
            .exp()
            .apply_t(&self.dropout, train);

        let mut out_vectors = attention_probs.matmul(&value_vectors);
        if out_vectors.dim() > 4 {
            logits = logits.flatten(2, 3).squeeze_dim(-1);
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
            let causal_mask = query_indices
                .unsqueeze(-1)
                .ge_tensor(&key_indices.unsqueeze(-2));
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
        let increase_num_buckets =
            i64::try_from(past_buckets.max()).unwrap() > num_hashes * max_bucket - 1;

        let query_buckets = self.hash_vectors(
            query_vectors,
            num_hashes,
            attention_mask,
            increase_num_buckets,
        );

        let concat_buckets = Tensor::cat(&[past_buckets, &query_buckets.unsqueeze(-1)], -1);
        let bucket_indices = stable_argsort(&concat_buckets, concat_buckets.dim() as i64 - 1);

        let relevant_bucket_indices = bucket_indices
            .eq(bucket_indices.size().last().unwrap() - 1)
            .nonzero();

        let relevant_bucket_indices_chunk = self
            .expand_to_indices_in_relevant_chunk(&relevant_bucket_indices, sequence_length)
            .transpose(0, 1);

        let relevant_bucket_indices_chunk = bucket_indices.index(&[
            Some(relevant_bucket_indices_chunk.get(0)),
            Some(relevant_bucket_indices_chunk.get(1)),
            Some(relevant_bucket_indices_chunk.get(2)),
            Some(relevant_bucket_indices_chunk.get(3)),
        ]);

        let bucket_indices_batch_offset = sequence_length
            * (batch_size
                * Tensor::arange(
                    *relevant_bucket_indices_chunk.size().last().unwrap(),
                    (Kind::Int64, hidden_states.device()),
                )
                .divide_scalar_mode(
                    *relevant_bucket_indices_chunk.size().last().unwrap(),
                    "floor",
                ));

        let relevant_bucket_indices_chunk_all_batch =
            &relevant_bucket_indices_chunk + bucket_indices_batch_offset;

        let relevant_hidden_states = hidden_states
            .reshape([-1, self.hidden_size])
            .index_select(
                0,
                &relevant_bucket_indices_chunk_all_batch.to_kind(Kind::Int64),
            )
            .reshape([batch_size, self.num_attention_heads, -1, self.hidden_size]);

        let relevant_bucket_indices_chunk = relevant_bucket_indices_chunk.reshape([
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
        let start_indices_chunk = (indices
            .select(1, -1)
            .divide_scalar_mode(self.chunk_length, "floor")
            - self.num_chunks_before)
            * self.chunk_length;
        let total_chunk_size =
            self.chunk_length * (1 + self.num_chunks_before + self.num_chunks_after);

        let expanded_start_indices = start_indices_chunk
            .unsqueeze(-1)
            .expand([indices.size()[0], total_chunk_size], true);
        let chunk_sequence_indices = expanded_start_indices
            + Tensor::arange(total_chunk_size, (Kind::Int64, indices.device()))
                .unsqueeze(0)
                .expand([indices.size()[0], total_chunk_size], true);

        let chunk_sequence_indices = chunk_sequence_indices
            .flatten(0, 1)
            .remainder(sequence_length);
        let indices = indices
            .unsqueeze(1)
            .expand([indices.size()[0], total_chunk_size, -1], true)
            .flatten(0, 1);

        indices.select(1, -1).copy_(&chunk_sequence_indices);
        indices
    }

    fn len_norm(&self, input_tensor: &Tensor, epsilon: f64) -> Tensor {
        let variance =
            (input_tensor * input_tensor).mean_dim([-1].as_slice(), true, input_tensor.kind());
        input_tensor * (variance + epsilon).rsqrt()
    }

    fn len_and_dim_norm(&self, input_tensor: &Tensor) -> Tensor {
        self.len_norm(input_tensor, 1e-6)
            * Tensor::from_slice(&[self.attention_head_size])
                .to_kind(input_tensor.kind())
                .to_device(input_tensor.device())
                .rsqrt()
    }

    fn gather_by_expansion(&self, vectors: &Tensor, indices: &Tensor, num_hashes: i64) -> Tensor {
        let expanded_indices = indices
            .unsqueeze(-1)
            .expand([-1, -1, -1, self.attention_head_size], true);
        vectors
            .repeat([1, 1, num_hashes, 1])
            .gather(2, &expanded_indices, false)
    }

    pub fn forward_t(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        num_hashes: Option<i64>,
        buckets: Option<Tensor>,
        layer_state: Option<&LayerState>,
        train: bool,
    ) -> Result<(Tensor, Option<Tensor>, Option<Tensor>), RustBertError> {
        let input_size = hidden_states.size();
        let (batch_size, sequence_length) = (input_size[0], input_size[1]);
        let num_hashes = num_hashes.unwrap_or(self.num_hashes);

        let do_cached_attention = self.use_cache & layer_state.is_some();

        let (
            key_value_hidden_states,
            mut query_key_vectors,
            mut value_vectors,
            query_vectors,
            mut sorted_bucket_idx,
            mut buckets,
            query_key_split,
        ) = if do_cached_attention {
            let layer_state = layer_state.as_ref().unwrap();

            let mut query_vectors = split_hidden_size_dim(
                &hidden_states.apply(&self.query_key),
                self.num_attention_heads,
                self.attention_head_size,
            );

            let (
                key_value_hidden_states,
                query_key_vectors,
                value_vectors,
                sorted_bucket_idx,
                buckets,
                query_key_split,
            ) = if let Some(prev_buckets) = &layer_state.prev_buckets {
                let (key_value_hidden_states, sorted_bucket_idx, buckets) = self
                    .get_relevant_hidden_states_and_buckets(
                        &query_vectors,
                        attention_mask,
                        num_hashes,
                        hidden_states,
                        &layer_state.prev_states,
                        prev_buckets,
                    );
                let query_key_vectors = self.query_per_attention_head(&key_value_hidden_states);
                let value_vectors = self.value_per_attention_head(&key_value_hidden_states);

                let query_key_vectors = split_seq_length_dim_to(
                    &query_key_vectors,
                    num_hashes,
                    -1,
                    self.num_attention_heads,
                    Some(self.attention_head_size),
                )?;

                let value_vectors = split_seq_length_dim_to(
                    &value_vectors,
                    num_hashes,
                    -1,
                    self.num_attention_heads,
                    Some(self.attention_head_size),
                )?;

                query_vectors = query_vectors.unsqueeze(2).repeat([1, 1, num_hashes, 1, 1]);
                (
                    key_value_hidden_states,
                    query_key_vectors,
                    value_vectors,
                    Some(sorted_bucket_idx),
                    Some(buckets),
                    true,
                )
            } else {
                let key_value_hidden_states =
                    Tensor::cat(&[&layer_state.prev_states, hidden_states], 1);
                let query_key_vectors = key_value_hidden_states.apply(&self.query_key);
                let value_vectors = key_value_hidden_states.apply(&self.value);
                (
                    key_value_hidden_states,
                    query_key_vectors,
                    value_vectors,
                    None,
                    buckets,
                    false,
                )
            };
            (
                Some(key_value_hidden_states),
                query_key_vectors,
                value_vectors,
                Some(query_vectors),
                sorted_bucket_idx,
                buckets,
                query_key_split,
            )
        } else {
            (
                None,
                hidden_states.apply(&self.query_key),
                hidden_states.apply(&self.value),
                None,
                None,
                buckets,
                false,
            )
        };

        if !query_key_split {
            query_key_vectors = split_hidden_size_dim(
                &query_key_vectors,
                self.num_attention_heads,
                self.attention_head_size,
            );
            value_vectors = split_hidden_size_dim(
                &value_vectors,
                self.num_attention_heads,
                self.attention_head_size,
            );
        }

        if do_cached_attention & layer_state.is_some()
            && layer_state.as_ref().unwrap().prev_buckets.is_none()
                & (key_value_hidden_states.unwrap().size()[1] >= self.chunk_length)
        {
            buckets =
                Some(self.hash_vectors(&query_key_vectors, num_hashes, attention_mask, false));
        }

        let do_standard_attention =
            (sequence_length <= self.chunk_length) | (self.use_cache & layer_state.is_some());

        let (sorted_bucket_idx_per_hash, undo_sorted_bucket_idx) = if !do_standard_attention {
            buckets = if let Some(bucket_value) = buckets {
                Some(bucket_value.view([
                    batch_size,
                    self.num_attention_heads,
                    num_hashes * sequence_length,
                ]))
            } else {
                Some(self.hash_vectors(&query_key_vectors, num_hashes, attention_mask, false))
            };
            let (sorted_bucket_idx_local, undo_sorted_bucket_idx) = self
                .get_sorted_bucket_indices_undo_sorted_bucket_indices(buckets.as_ref().unwrap());
            sorted_bucket_idx = Some(sorted_bucket_idx_local);
            let sorted_bucket_idx_per_hash = sorted_bucket_idx.unwrap().remainder(sequence_length);

            query_key_vectors = self.gather_by_expansion(
                &query_key_vectors,
                &sorted_bucket_idx_per_hash,
                num_hashes,
            );
            value_vectors =
                self.gather_by_expansion(&value_vectors, &sorted_bucket_idx_per_hash, num_hashes);

            query_key_vectors = split_seq_length_dim_to(
                &query_key_vectors,
                -1,
                self.chunk_length,
                self.num_attention_heads,
                Some(self.attention_head_size),
            )?;
            value_vectors = split_seq_length_dim_to(
                &value_vectors,
                -1,
                self.chunk_length,
                self.num_attention_heads,
                Some(self.attention_head_size),
            )?;
            (sorted_bucket_idx_per_hash, Some(undo_sorted_bucket_idx))
        } else if do_cached_attention & {
            if let Some(layer_state_value) = layer_state {
                layer_state_value.prev_buckets.is_some()
            } else {
                false
            }
        } {
            (sorted_bucket_idx.unwrap().copy(), None)
        } else {
            (
                Tensor::arange(sequence_length, (Kind::Int64, query_key_vectors.device()))
                    .repeat([batch_size, self.num_attention_heads, 1]),
                None,
            )
        };

        let key_vectors = self.len_and_dim_norm(&query_key_vectors);
        let query_vectors = query_vectors.unwrap_or(query_key_vectors);

        let (mut out_vectors, mut logits, attention_probs) = self.attend(
            query_vectors,
            key_vectors,
            value_vectors,
            sorted_bucket_idx_per_hash,
            attention_mask,
            do_standard_attention,
            do_cached_attention,
            train,
        )?;

        if !do_standard_attention {
            let temp = reverse_sort(&out_vectors, &logits, &undo_sorted_bucket_idx.unwrap());
            out_vectors = temp.0;
            logits = temp.1;
        }

        if (!do_standard_attention
            | (do_cached_attention & {
                if let Some(layer_state_value) = layer_state {
                    layer_state_value.prev_buckets.is_some()
                } else {
                    false
                }
            }))
            & (num_hashes > 1)
        {
            out_vectors = split_seq_length_dim_to(
                &out_vectors,
                num_hashes,
                sequence_length,
                self.num_attention_heads,
                Some(self.attention_head_size),
            )?;
            logits = split_seq_length_dim_to(
                &logits,
                num_hashes,
                sequence_length,
                self.num_attention_heads,
                Some(self.attention_head_size),
            )?
            .unsqueeze(-1);
            let probs_vectors = (&logits - &logits.logsumexp([2].as_slice(), true)).exp();
            let out_kind = out_vectors.kind();
            out_vectors =
                (out_vectors * probs_vectors).sum_dim_intlist([2].as_slice(), false, out_kind);
        }

        out_vectors = merge_hidden_size_dim(
            &out_vectors,
            self.num_attention_heads,
            self.attention_head_size,
        );

        let attention_probs = if self.output_attentions {
            Some(attention_probs)
        } else {
            None
        };

        buckets = buckets.map(|buckets_value| {
            buckets_value.view([batch_size, self.num_attention_heads, num_hashes, -1])
        });

        Ok((out_vectors, attention_probs, buckets))
    }
}

#[derive(Debug)]
/// # Local Self Attention for Reformer model
pub struct LocalSelfAttention {
    chunk_length: i64,
    num_chunks_before: i64,
    num_chunks_after: i64,
    is_decoder: bool,
    dropout: Dropout,
    num_attention_heads: i64,
    attention_head_size: i64,
    query: nn::Linear,
    key: nn::Linear,
    value: nn::Linear,
    mask_value_fp32: Tensor,
    mask_value_fp16: Tensor,
    use_cache: bool,
    output_attentions: bool,
}

impl LocalSelfAttention {
    pub fn new<'p, P>(
        p: P,
        config: &ReformerConfig,
        output_attentions: bool,
        use_cache: bool,
    ) -> LocalSelfAttention
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let chunk_length = config.local_attn_chunk_length.unwrap_or(64);
        let num_chunks_before = config.local_num_chunks_before.unwrap_or(1);
        let num_chunks_after = config.local_num_chunks_after.unwrap_or(0);
        let is_decoder = config.is_decoder;

        let dropout = Dropout::new(config.hidden_dropout_prob);

        let num_attention_heads = config.num_attention_heads;
        let attention_head_size = config.attention_head_size;
        let all_head_size = num_attention_heads * attention_head_size;
        let hidden_size = config.hidden_size;

        let linear_config = LinearConfig {
            bias: false,
            ..Default::default()
        };
        let query = nn::linear(p / "query", hidden_size, all_head_size, linear_config);
        let key = nn::linear(p / "key", hidden_size, all_head_size, linear_config);
        let value = nn::linear(p / "value", hidden_size, all_head_size, linear_config);

        let mask_value_fp32 = Tensor::from_slice(&[-1e9])
            .to_kind(Kind::Float)
            .to(p.device());

        let mask_value_fp16 = Tensor::from_slice(&[-1e4])
            .to_kind(Kind::Half)
            .to(p.device());

        LocalSelfAttention {
            chunk_length,
            num_chunks_before,
            num_chunks_after,
            is_decoder,
            dropout,
            num_attention_heads,
            attention_head_size,
            query,
            key,
            value,
            mask_value_fp32,
            mask_value_fp16,
            use_cache,
            output_attentions,
        }
    }

    fn compute_attention_mask(
        &self,
        query_indices: &Tensor,
        key_indices: &Tensor,
        attention_mask: Option<&Tensor>,
        query_key_dots_shape: &[i64],
        do_standard_attention: bool,
    ) -> Option<Tensor> {
        let mut attention_mask = attention_mask.map(|mask| {
            let mut mask = mask.to_kind(Kind::Int8).unsqueeze(1);
            if !do_standard_attention {
                mask = split_seq_length_dim_to(&mask, -1, self.chunk_length, 1, None).unwrap();
                mask = look_adjacent(mask, self.num_chunks_before, self.num_chunks_after);
            }
            mask.unsqueeze(-2).expand(query_key_dots_shape, true)
        });

        if self.is_decoder {
            let causal_mask = query_indices
                .unsqueeze(-1)
                .ge_tensor(&key_indices.unsqueeze(-2));
            attention_mask = Some(if let Some(mask) = attention_mask {
                causal_mask * mask
            } else {
                causal_mask
            });
        };
        attention_mask
    }

    pub fn forward_t(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        layer_state: Option<&LayerState>,
        train: bool,
    ) -> Result<(Tensor, Option<Tensor>), RustBertError> {
        let input_size = hidden_states.size();
        let (batch_size, sequence_length) = (input_size[0], input_size[1]);

        let (query_vectors, key_vectors, value_vectors) = if layer_state.is_some() & self.use_cache
        {
            let layer_state_value = layer_state.as_ref().unwrap();
            let key_value_hidden_states = retrieve_relevant_hidden_states(
                &layer_state_value.prev_states,
                self.chunk_length,
                self.num_chunks_before,
            );
            let key_value_hidden_states =
                Tensor::cat(&[&key_value_hidden_states, hidden_states], 1);
            let query_vectors = hidden_states.apply(&self.query);
            let key_vectors = key_value_hidden_states.apply(&self.key);
            let value_vectors = key_value_hidden_states.apply(&self.value);
            (query_vectors, key_vectors, value_vectors)
        } else {
            let query_vectors = hidden_states.apply(&self.query);
            let key_vectors = hidden_states.apply(&self.key);
            let value_vectors = hidden_states.apply(&self.value);
            (query_vectors, key_vectors, value_vectors)
        };
        let mut query_vectors = split_hidden_size_dim(
            &query_vectors,
            self.num_attention_heads,
            self.attention_head_size,
        );
        let key_vectors = split_hidden_size_dim(
            &key_vectors,
            self.num_attention_heads,
            self.attention_head_size,
        );
        let mut value_vectors = split_hidden_size_dim(
            &value_vectors,
            self.num_attention_heads,
            self.attention_head_size,
        );

        let key_kind_device = (key_vectors.kind(), key_vectors.device());
        let mut key_vectors = key_vectors
            / Tensor::from_slice(&[self.attention_head_size])
                .to_kind(key_kind_device.0)
                .to(key_kind_device.1)
                .sqrt();

        let indices = Tensor::arange(sequence_length, (Kind::Int64, query_vectors.device()))
            .repeat([batch_size, self.num_attention_heads, 1]);

        let do_standard_attention = sequence_length <= self.chunk_length;

        let (query_indices, key_indices) = if !do_standard_attention {
            query_vectors = split_seq_length_dim_to(
                &query_vectors,
                -1,
                self.chunk_length,
                self.num_attention_heads,
                Some(self.attention_head_size),
            )?;
            key_vectors = split_seq_length_dim_to(
                &key_vectors,
                -1,
                self.chunk_length,
                self.num_attention_heads,
                Some(self.attention_head_size),
            )?;
            value_vectors = split_seq_length_dim_to(
                &value_vectors,
                -1,
                self.chunk_length,
                self.num_attention_heads,
                Some(self.attention_head_size),
            )?;

            let query_indices = split_seq_length_dim_to(
                &indices,
                -1,
                self.chunk_length,
                self.num_attention_heads,
                None,
            )?;
            let key_indices = query_indices.copy();

            key_vectors = look_adjacent(key_vectors, self.num_chunks_before, self.num_chunks_after);
            value_vectors =
                look_adjacent(value_vectors, self.num_chunks_before, self.num_chunks_after);
            let key_indices =
                look_adjacent(key_indices, self.num_chunks_before, self.num_chunks_after);
            (query_indices, key_indices)
        } else {
            (indices.copy(), indices.copy())
        };

        let mut query_key_dots = query_vectors.matmul(&key_vectors.transpose(-1, -2));
        let attention_mask = self.compute_attention_mask(
            &query_indices,
            &key_indices,
            attention_mask,
            query_key_dots.size().as_slice(),
            do_standard_attention,
        );

        if let Some(mask) = attention_mask {
            query_key_dots = apply_mask_with_fp16_compatibility(
                &query_key_dots,
                &mask,
                &self.mask_value_fp32,
                &self.mask_value_fp16,
            )?;
        }

        let logits = query_key_dots.logsumexp([-1], true);
        let attention_probs = (query_key_dots - logits)
            .exp()
            .apply_t(&self.dropout, train);

        let mut out_vectors = attention_probs.matmul(&value_vectors);
        if !do_standard_attention {
            out_vectors = out_vectors.flatten(2, 3);
        }
        out_vectors = merge_hidden_size_dim(
            &out_vectors,
            self.num_attention_heads,
            self.attention_head_size,
        );
        let attention_probs = if self.output_attentions {
            Some(attention_probs)
        } else {
            None
        };
        Ok((out_vectors, attention_probs))
    }
}

pub enum AttentionModule {
    LSHSelfAttention(LSHSelfAttention),
    LocalSelfAttention(LocalSelfAttention),
}

impl AttentionModule {
    pub fn new<'p, P>(
        p: P,
        attention_type: &AttentionType,
        config: &ReformerConfig,
        output_attentions: bool,
        use_past: bool,
    ) -> Result<Self, RustBertError>
    where
        P: Borrow<nn::Path<'p>>,
    {
        Ok(match attention_type {
            AttentionType::lsh => AttentionModule::LSHSelfAttention(LSHSelfAttention::new(
                p,
                config,
                output_attentions,
                use_past,
            )?),
            AttentionType::local => AttentionModule::LocalSelfAttention(LocalSelfAttention::new(
                p,
                config,
                output_attentions,
                use_past,
            )),
        })
    }

    pub fn forward_t(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        num_hashes: Option<i64>,
        buckets: Option<Tensor>,
        layer_state: Option<&LayerState>,
        train: bool,
    ) -> Result<(Tensor, Option<Tensor>, Option<Tensor>), RustBertError> {
        match self {
            AttentionModule::LSHSelfAttention(ref attention) => attention.forward_t(
                hidden_states,
                attention_mask,
                num_hashes,
                buckets,
                layer_state,
                train,
            ),
            AttentionModule::LocalSelfAttention(ref attention) => {
                let output =
                    attention.forward_t(hidden_states, attention_mask, layer_state, train)?;
                Ok((output.0, output.1, None))
            }
        }
    }
}

#[derive(Debug)]
/// # Reformer attention dense layer
pub struct ReformerSelfOutput {
    dense: nn::Linear,
    dropout: Dropout,
}

impl ReformerSelfOutput {
    pub fn new<'p, P>(p: P, config: &ReformerConfig) -> ReformerSelfOutput
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let linear_config = LinearConfig {
            bias: false,
            ..Default::default()
        };
        let dense = nn::linear(
            p / "dense",
            config.num_attention_heads * config.attention_head_size,
            config.hidden_size,
            linear_config,
        );
        let dropout = Dropout::new(config.hidden_dropout_prob);

        ReformerSelfOutput { dense, dropout }
    }

    pub fn forward_t(&self, hidden_states: &Tensor, train: bool) -> Tensor {
        hidden_states
            .apply(&self.dense)
            .apply_t(&self.dropout, train)
    }
}

pub struct ReformerAttentionOutput {
    pub attention_output: Tensor,
    pub attention_probs: Option<Tensor>,
    pub buckets: Option<Tensor>,
    pub new_layer_state: Option<LayerState>,
}

/// # Reformer attention layer
pub struct ReformerAttention {
    self_attention: AttentionModule,
    layer_norm: nn::LayerNorm,
    self_output: ReformerSelfOutput,
    use_past: bool,
}

impl ReformerAttention {
    pub fn new<'p, P>(
        p: P,
        config: &ReformerConfig,
        attention_type: &AttentionType,
        output_attentions: bool,
        use_past: bool,
    ) -> Result<ReformerAttention, RustBertError>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let layer_norm_config = nn::LayerNormConfig {
            eps: config.layer_norm_eps.unwrap_or(1e-12),
            ..Default::default()
        };
        let layer_norm = nn::layer_norm(
            p / "layer_norm",
            vec![config.hidden_size],
            layer_norm_config,
        );

        let self_attention = AttentionModule::new(
            p / "self_attention",
            attention_type,
            config,
            output_attentions,
            use_past,
        )?;

        let self_output = ReformerSelfOutput::new(p / "output", config);

        Ok(ReformerAttention {
            self_attention,
            layer_norm,
            self_output,
            use_past,
        })
    }

    pub fn forward_t(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        num_hashes: Option<i64>,
        buckets: Option<Tensor>,
        layer_state: Option<LayerState>,
        original_sequence_length: i64,
        train: bool,
    ) -> Result<ReformerAttentionOutput, RustBertError> {
        let hidden_states = hidden_states.apply(&self.layer_norm);

        let (attention_hidden_state, attention_probs, buckets) = self.self_attention.forward_t(
            &hidden_states,
            attention_mask,
            num_hashes,
            buckets,
            layer_state.as_ref(),
            train,
        )?;
        let new_layer_state = if self.use_past {
            let prev_buckets = if let Some(buckets_value) = &buckets {
                if layer_state.is_none() | {
                    if layer_state.is_some() {
                        layer_state.as_ref().unwrap().prev_buckets.is_none()
                    } else {
                        false
                    }
                } {
                    if original_sequence_length > 1 {
                        Some(buckets_value.slice(3, 0, original_sequence_length, 1))
                    } else {
                        Some(buckets_value.copy())
                    }
                } else {
                    Some(Tensor::cat(
                        &[
                            buckets_value,
                            layer_state.as_ref().unwrap().prev_buckets.as_ref().unwrap(),
                        ],
                        -1,
                    ))
                }
            } else {
                None
            };

            let prev_states = if let Some(layer_state_value) = &layer_state {
                Tensor::cat(&[&layer_state_value.prev_states, &hidden_states], 1)
            } else {
                hidden_states.slice(1, 0, original_sequence_length, 1)
            };
            Some(LayerState {
                prev_buckets,
                prev_states,
            })
        } else {
            None
        };

        let attention_output = self.self_output.forward_t(&attention_hidden_state, train);

        Ok(ReformerAttentionOutput {
            attention_output,
            attention_probs,
            buckets,
            new_layer_state,
        })
    }
}
