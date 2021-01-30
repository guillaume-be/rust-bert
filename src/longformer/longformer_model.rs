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

use crate::{Activation, Config};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// # Longformer Pretrained model weight files
pub struct LongformerModelResources;

/// # Longformer Pretrained model config files
pub struct LongformerConfigResources;

/// # Longformer Pretrained model vocab files
pub struct LongformerVocabResources;

/// # Longformer Pretrained model merges files
pub struct LongformerMergesResources;

impl LongformerModelResources {
    /// Shared under Apache 2.0 license by the AllenAI team at https://github.com/allenai/longformer. Modified with conversion to C-array format.
    pub const LONGFORMER_BASE_4096: (&'static str, &'static str) = (
        "longformer-base-4096/model",
        "https://huggingface.co/allenai/longformer-base-4096/resolve/main/rust_model.ot",
    );
}

impl LongformerConfigResources {
    /// Shared under Apache 2.0 license by the AllenAI team at https://github.com/allenai/longformer. Modified with conversion to C-array format.
    pub const LONGFORMER_BASE_4096: (&'static str, &'static str) = (
        "longformer-base-4096/config",
        "https://huggingface.co/allenai/longformer-base-4096/resolve/main/config.json",
    );
}

impl LongformerVocabResources {
    /// Shared under Apache 2.0 license by the AllenAI team at https://github.com/allenai/longformer. Modified with conversion to C-array format.
    pub const LONGFORMER_BASE_4096: (&'static str, &'static str) = (
        "longformer-base-4096/vocab",
        "https://huggingface.co/allenai/longformer-base-4096/resolve/main/vocab.json",
    );
}

impl LongformerMergesResources {
    /// Shared under Apache 2.0 license by the AllenAI team at https://github.com/allenai/longformer. Modified with conversion to C-array format.
    pub const LONGFORMER_BASE_4096: (&'static str, &'static str) = (
        "longformer-base-4096/merges",
        "https://huggingface.co/allenai/longformer-base-4096/resolve/main/merges.txt",
    );
}

#[derive(Debug, Serialize, Deserialize, Clone)]
/// # Longformer model configuration
/// Defines the Longformer model architecture (e.g. number of layers, hidden layer size, label mapping...)
pub struct LongformerConfig {
    pub hidden_act: Activation,
    pub attention_window: Vec<i64>,
    pub attention_probs_dropout_prob: f64,
    pub hidden_dropout_prob: f64,
    pub hidden_size: i64,
    pub initializer_range: f32,
    pub intermediate_size: i64,
    pub max_position_embeddings: i64,
    pub num_attention_heads: i64,
    pub num_hidden_layers: i64,
    pub type_vocab_size: i64,
    pub vocab_size: i64,
    pub output_attentions: Option<bool>,
    pub output_hidden_states: Option<bool>,
    pub is_decoder: Option<bool>,
    pub id2label: Option<HashMap<i64, String>>,
    pub label2id: Option<HashMap<String, i64>>,
}

impl Config<LongformerConfig> for LongformerConfig {}
