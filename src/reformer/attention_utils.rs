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
use std::cmp::min;
use std::collections::HashSet;
use std::iter::FromIterator;
use tch::{Kind, Tensor};

pub fn stable_argsort(input_tensor: &Tensor, dim: i64) -> Tensor {
    let scaling_dim = input_tensor.size()[dim as usize];
    let scaled_offset = Tensor::arange(scaling_dim, (Kind::Int, input_tensor.device()))
        .view([1, 1, -1])
        .expand(&input_tensor.size(), true);
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
    let num_unique_attention_type =
        HashSet::<&AttentionType>::from_iter(attention_types.iter()).len();
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
    let num_unique_attention_type =
        HashSet::<&AttentionType>::from_iter(attention_types.iter()).len();
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

#[cfg(test)]
mod test {
    use crate::reformer::attention::AttentionType;
    use crate::reformer::attention_utils::{get_least_common_mult_chunk_len, get_min_chunk_len};
    use crate::reformer::lcm;

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
