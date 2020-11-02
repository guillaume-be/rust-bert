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
use crate::reformer::ReformerConfig;
use crate::RustBertError;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::borrow::Borrow;
use std::convert::{TryFrom, TryInto};
use tch::nn::LinearConfig;
use tch::{nn, Kind, Tensor};

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
    self_mask_value: f64,
    mask_value: f64,
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

        let self_mask_value = -1e5;
        let mask_value = -1e9;

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
}
