// Copyright 2021, Google and The HuggingFace Inc. team. All rights reserved.
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

/// # Pegasus Pretrained model weight files
pub struct PegasusModelResources;

/// # Pegasus Pretrained model config files
pub struct PegasusConfigResources;

/// # Pegasus Pretrained model vocab files
pub struct PegasusVocabResources;

impl PegasusModelResources {
    /// Shared under Apache 2.0 license by the Pegasus team at https://huggingface.co/google/pegasus-cnn_dailymail. Modified with conversion to C-array format.
    pub const CNN_DAILYMAIL: (&'static str, &'static str) = (
        "pegasus-cnn_dailymail/model",
        "https://huggingface.co/google/pegasus-cnn_dailymail/resolve/main/rust_model.ot",
    );
}

impl PegasusConfigResources {
    /// Shared under Apache 2.0 license by the Pegasus team at https://huggingface.co/google/pegasus-cnn_dailymail.
    pub const CNN_DAILYMAIL: (&'static str, &'static str) = (
        "pegasus-cnn_dailymail/config",
        "https://huggingface.co/google/pegasus-cnn_dailymail/resolve/main/config.json",
    );
}

impl PegasusVocabResources {
    /// Shared under Apache 2.0 license by the Pegasus team at https://huggingface.co/google/pegasus-cnn_dailymail.
    pub const CNN_DAILYMAIL: (&'static str, &'static str) = (
        "pegasus-cnn_dailymail/spiece",
        "https://huggingface.co/google/pegasus-cnn_dailymail/resolve/main/spiece.model",
    );
}

#[derive(Debug, Serialize, Deserialize, Clone)]
/// # Pegasus model configuration
/// Defines the Pegasus model architecture (e.g. number of layers, hidden layer size, label mapping...)
pub struct PegasusConfig {
    pub num_labels: Option<i64>,
    pub activation_function: Option<Activation>,
    pub activation_dropout: f64,
    pub attention_dropout: f64,
    pub classifier_dropout: Option<f64>,
    pub d_model: i64,
    pub decoder_attention_heads: i64,
    pub decoder_ffn_dim: i64,
    pub decoder_layerdrop: f64,
    pub decoder_layers: i64,
    pub decoder_start_token_id: Option<i64>,
    pub dropout: f64,
    pub encoder_attention_heads: i64,
    pub encoder_ffn_dim: i64,
    pub encoder_layerdrop: f64,
    pub encoder_layers: i64,
    pub bos_token_id: Option<i64>,
    pub eos_token_id: Option<i64>,
    pub pad_token_id: Option<i64>,
    pub id2label: Option<HashMap<i64, String>>,
    pub label2id: Option<HashMap<String, i64>>,
    pub init_std: f64,
    pub is_decoder: Option<bool>,
    pub is_encoder_decoder: Option<bool>,
    pub max_position_embeddings: i64,
    pub min_length: Option<i64>,
    pub normalize_embedding: Option<bool>,
    pub num_hidden_layers: i64,
    pub output_attentions: Option<bool>,
    pub output_hidden_states: Option<bool>,
    pub output_past: Option<bool>,
    pub static_position_embeddings: Option<bool>,
    pub scale_embedding: Option<bool>,
    pub vocab_size: i64,
}

impl Config<PegasusConfig> for PegasusConfig {}
