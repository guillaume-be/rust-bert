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

use crate::common::activations::Activation;
use crate::reformer::attention::AttentionType;
use crate::Config;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// # Reformer Pretrained model weight files
pub struct ReformerModelResources;

/// # Reformer Pretrained model config files
pub struct ReformerConfigResources;

/// # Reformer Pretrained model vocab files
pub struct ReformerVocabResources;

impl ReformerModelResources {
    /// Shared under Apache 2.0 license by the Trax Authors at https://github.com/google/trax/tree/master/trax/models/reformer. Modified with conversion to C-array format.
    pub const CRIME_AND_PUNISHMENT: (&'static str, &'static str) = (
        "xlnet-base-cased/model",
        "https://cdn.huggingface.co/google/reformer-crime-and-punishment/rust_model.ot",
    );
}

impl ReformerConfigResources {
    /// Shared under Apache 2.0 license by the Trax Authors at https://github.com/google/trax/tree/master/trax/models/reformer. Modified with conversion to C-array format.
    pub const CRIME_AND_PUNISHMENT: (&'static str, &'static str) = (
        "xlnet-base-cased/config",
        "https://cdn.huggingface.co/google/reformer-crime-and-punishment/config.json",
    );
}

impl ReformerVocabResources {
    /// Shared under Apache 2.0 license by the Trax Authors at https://github.com/google/trax/tree/master/trax/models/reformer. Modified with conversion to C-array format.
    pub const CRIME_AND_PUNISHMENT: (&'static str, &'static str) = (
        "xlnet-base-cased/spiece",
        "https://cdn.huggingface.co/google/reformer-crime-and-punishment/spiece.model",
    );
}

#[derive(Debug, Serialize, Deserialize)]
/// # Reformer model configuration
/// Defines the Reformer model architecture (e.g. number of layers, hidden layer size, label mapping...)
pub struct ReformerConfig {
    pub attention_head_size: i64,
    pub attention_probs_dropout_prob: f64,
    pub attn_layers: Vec<AttentionType>,
    pub axial_norm_std: f64,
    pub axial_pos_embds: bool,
    pub axial_pos_embds_dim: Vec<i64>,
    pub axial_pos_shape: Vec<i64>,
    pub chunk_size_lm_head: i64,
    pub chunk_size_feed_forward: Option<i64>,
    pub eos_token_id: i64,
    pub pad_token_id: i64,
    pub feed_forward_size: i64,
    pub hash_seed: Option<i64>,
    pub hidden_act: Activation,
    pub hidden_dropout_prob: f64,
    pub hidden_size: i64,
    pub initializer_range: Option<f64>,
    pub intermediate_size: i64,
    pub is_decoder: bool,
    pub layer_norm_eps: Option<f64>,
    pub local_attn_chunk_length: i64,
    pub lsh_attn_chunk_length: i64,
    pub max_position_embeddings: i64,
    pub vocab_size: i64,
    pub num_attention_heads: i64,
    pub num_buckets: Vec<i64>,
    pub local_num_chunks_after: Option<i64>,
    pub local_num_chunks_before: Option<i64>,
    pub local_attention_probs_dropout_prob: Option<f64>,
    pub lsh_num_chunks_after: Option<i64>,
    pub lsh_num_chunks_before: Option<i64>,
    pub lsh_attention_probs_dropout_prob: Option<f64>,
    pub num_hashes: i64,
    pub num_hidden_layers: i64,
    pub use_cache: Option<bool>,
    pub id2label: Option<HashMap<i64, String>>,
    pub label2id: Option<HashMap<String, i64>>,
    pub output_attentions: Option<bool>,
    pub output_hidden_states: Option<bool>,
}

impl Config<ReformerConfig> for ReformerConfig {}
