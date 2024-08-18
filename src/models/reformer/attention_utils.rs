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

use crate::reformer::attention::AttentionType;
use crate::RustBertError;
use std::cmp::min;
use std::collections::HashSet;
use tch::{Kind, Tensor};

pub fn stable_argsort(input_tensor: &Tensor, dim: i64) -> Tensor {
    let scaling_dim = input_tensor.size()[dim as usize];
    let scaled_offset = Tensor::arange(scaling_dim, (Kind::Int, input_tensor.device()))
        .view([1, 1, -1])
        .expand(input_tensor.size(), true);
    let scaled_tensor = scaling_dim * input_tensor + (scaled_offset / scaling_dim);
    scaled_tensor.argsort(dim, false)
}

fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

pub fn lcm(a: i64, b: i64) -> i64 {
    a * (b / gcd(a, b))
}

pub fn get_least_common_mult_chunk_len(
    attention_types: &[AttentionType],
    lsh_attn_chunk_length: Option<i64>,
    local_attn_chunk_length: Option<i64>,
) -> i64 {
    let num_unique_attention_type = attention_types
        .iter()
        .collect::<HashSet<&AttentionType>>()
        .len();
    match num_unique_attention_type {
        1 => {
            if attention_types[0] == AttentionType::lsh {
                lsh_attn_chunk_length.unwrap_or(64)
            } else {
                local_attn_chunk_length.unwrap_or(64)
            }
        }
        2 => lcm(
            lsh_attn_chunk_length.unwrap_or(64),
            local_attn_chunk_length.unwrap_or(64),
        ),
        _ => panic!("Impossible scenario - only 2 attention types supported"),
    }
}

pub fn get_min_chunk_len(
    attention_types: &[AttentionType],
    lsh_attn_chunk_length: Option<i64>,
    local_attn_chunk_length: Option<i64>,
) -> i64 {
    let num_unique_attention_type = attention_types
        .iter()
        .collect::<HashSet<&AttentionType>>()
        .len();
    match num_unique_attention_type {
        1 => {
            if attention_types[0] == AttentionType::lsh {
                lsh_attn_chunk_length.unwrap_or(64)
            } else {
                local_attn_chunk_length.unwrap_or(64)
            }
        }
        2 => min(
            lsh_attn_chunk_length.unwrap_or(64),
            local_attn_chunk_length.unwrap_or(64),
        ),
        _ => panic!("Impossible scenario - only 2 attention types supported"),
    }
}

pub fn look_adjacent(vectors: Tensor, num_chunks_before: i64, num_chunks_after: i64) -> Tensor {
    if (num_chunks_before == 0) & (num_chunks_after == 0) {
        vectors
    } else {
        let mut calc_slices =
            Vec::with_capacity((num_chunks_before + num_chunks_after + 1) as usize);
        let mut ref_slices =
            Vec::with_capacity((num_chunks_before + num_chunks_after + 1) as usize);
        for i in -num_chunks_before..num_chunks_after + 1 {
            calc_slices.push(Tensor::cat(
                &[
                    &vectors.slice(2, i, vectors.size()[2], 1),
                    &vectors.slice(2, 0, i, 1),
                ],
                2,
            ))
        }
        for i in -num_chunks_before..num_chunks_after + 1 {
            if i == 0 {
                ref_slices.push(&vectors)
            } else {
                ref_slices.push(&calc_slices[(i + num_chunks_before) as usize])
            }
        }
        Tensor::cat(ref_slices.as_slice(), 3)
    }
}

pub fn split_hidden_size_dim(
    input: &Tensor,
    num_attention_heads: i64,
    attention_head_size: i64,
) -> Tensor {
    let mut new_x_shape = input.size();
    let _ = new_x_shape.pop();
    new_x_shape.push(num_attention_heads);
    new_x_shape.push(attention_head_size);
    input.view(new_x_shape.as_slice()).transpose(2, 1)
}

pub fn merge_hidden_size_dim(
    input: &Tensor,
    num_attention_heads: i64,
    attention_head_size: i64,
) -> Tensor {
    let new_shape = [
        input.size()[0],
        -1,
        num_attention_heads * attention_head_size,
    ];
    input.permute([0, 2, 1, 3]).reshape(new_shape)
}

pub fn split_seq_length_dim_to(
    vectors: &Tensor,
    dim_factor_1: i64,
    dim_factor_2: i64,
    num_attention_heads: i64,
    attention_head_size: Option<i64>,
) -> Result<Tensor, RustBertError> {
    let input_size = vectors.size();
    let batch_size = input_size[0];
    let mut split_dim_shape = vec![batch_size, num_attention_heads, dim_factor_1, dim_factor_2];

    if input_size.len() == 4 {
        let attention_head_size = if let Some(attention_head_size_value) = attention_head_size {
            attention_head_size_value
        } else {
            return Err(RustBertError::ValueError(
                "attention_head_size must be provided for inputs of dimension 4".to_string(),
            ));
        };
        split_dim_shape.push(attention_head_size);
    };
    Ok(vectors.reshape(split_dim_shape.as_slice()))
}

pub fn reverse_sort(
    out_vectors: &Tensor,
    logits: &Tensor,
    undo_sorted_bucket_idx: &Tensor,
) -> (Tensor, Tensor) {
    let expanded_undo_sort_indices = undo_sorted_bucket_idx
        .unsqueeze(-1)
        .expand(out_vectors.size().as_slice(), true);
    let out_vectors = out_vectors.gather(2, &expanded_undo_sort_indices, false);
    let logits = logits.gather(2, undo_sorted_bucket_idx, false);
    (out_vectors, logits)
}

pub fn retrieve_relevant_hidden_states(
    previous_hidden_states: &Tensor,
    chunk_length: i64,
    num_chunks_before: i64,
) -> Tensor {
    let end_position = previous_hidden_states.size()[1];
    let start_position = ((end_position / chunk_length) - num_chunks_before) * chunk_length;
    previous_hidden_states.slice(1, start_position, end_position, 1)
}

#[cfg(test)]
mod test {
    use crate::reformer::attention::AttentionType;
    use crate::reformer::attention_utils::{
        get_least_common_mult_chunk_len, get_min_chunk_len, lcm,
    };

    #[test]
    fn test_lcm_calculation() {
        let test_pairs = [(7, 3), (1, 1), (8, 9), (48, 32), (-1, -1), (1, 0), (0, 1)];
        let expected_results = [21, 1, 72, 96, -1, 0, 0];

        for (test_pair, expected_result) in test_pairs.iter().zip(expected_results.iter()) {
            assert_eq!(lcm(test_pair.0, test_pair.1), *expected_result);
        }
    }

    #[test]
    fn test_get_least_common_mult_chunk_len() {
        // Given
        let lsh_attn_chunk_length = Some(48);
        let local_attn_chunk_length = Some(32);

        // When
        let attention_types = [
            vec![
                AttentionType::lsh,
                AttentionType::local,
                AttentionType::lsh,
                AttentionType::local,
                AttentionType::lsh,
                AttentionType::local,
            ],
            vec![AttentionType::lsh, AttentionType::lsh, AttentionType::lsh],
            vec![
                AttentionType::local,
                AttentionType::local,
                AttentionType::local,
            ],
        ];
        let expected_results = [96, 48, 32];

        // Then
        for (test_types, expected_result) in attention_types.iter().zip(expected_results.iter()) {
            assert_eq!(
                get_least_common_mult_chunk_len(
                    test_types,
                    lsh_attn_chunk_length,
                    local_attn_chunk_length
                ),
                *expected_result
            );
        }
    }

    #[test]
    fn test_get_min_chunk_len() {
        // Given
        let lsh_attn_chunk_length = Some(48);
        let local_attn_chunk_length = Some(32);

        // When
        let attention_types = [
            vec![
                AttentionType::lsh,
                AttentionType::local,
                AttentionType::lsh,
                AttentionType::local,
                AttentionType::lsh,
                AttentionType::local,
            ],
            vec![AttentionType::lsh, AttentionType::lsh, AttentionType::lsh],
            vec![
                AttentionType::local,
                AttentionType::local,
                AttentionType::local,
            ],
        ];
        let expected_results = [32, 48, 32];

        // Then
        for (test_types, expected_result) in attention_types.iter().zip(expected_results.iter()) {
            assert_eq!(
                get_min_chunk_len(test_types, lsh_attn_chunk_length, local_attn_chunk_length),
                *expected_result
            );
        }
    }
}
