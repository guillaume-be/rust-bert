// Copyright 2020, Microsoft and the HuggingFace Inc. team.
// Copyright 2022 Guillaume Becquin
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::deberta::{deserialize_attention_type, PositionAttentionTypes};
use crate::Activation;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// # DeBERTaV2 Pretrained model weight files
pub struct DebertaV2ModelResources;

/// # DeBERTaV2 Pretrained model config files
pub struct DebertaV2ConfigResources;

/// # DeBERTaV2 Pretrained model vocab files
pub struct DebertaV2VocabResources;

impl DebertaV2ModelResources {
    /// Shared under MIT license by the Microsoft team at <https://huggingface.co/microsoft/deberta-v3-base>. Modified with conversion to C-array format.
    pub const DEBERTA_V3_BASE: (&'static str, &'static str) = (
        "deberta-v3-base/model",
        "https://huggingface.co/microsoft/deberta-v3-base/resolve/main/rust_model.ot",
    );
}

impl DebertaV2ConfigResources {
    /// Shared under MIT license by the Microsoft team at <https://huggingface.co/microsoft/deberta-v3-base>. Modified with conversion to C-array format.
    pub const DEBERTA_V3_BASE: (&'static str, &'static str) = (
        "deberta-base/config",
        "https://huggingface.co/microsoft/deberta-v3-base/resolve/main/config.json",
    );
}

impl DebertaV2VocabResources {
    /// Shared under MIT license by the Microsoft team at <https://huggingface.co/microsoft/deberta-v3-base>. Modified with conversion to C-array format.
    pub const DEBERTA_V3_BASE: (&'static str, &'static str) = (
        "deberta-base/vocab",
        "https://huggingface.co/microsoft/deberta-v3-base/resolve/main/vocab.json",
    );
}

#[derive(Debug, Serialize, Deserialize)]
/// # DeBERTa (v2) model configuration
/// Defines the DeBERTa (v2) model architecture (e.g. number of layers, hidden layer size, label mapping...)
pub struct DebertaV2Config {
    pub hidden_act: Activation,
    pub attention_probs_dropout_prob: f64,
    pub hidden_dropout_prob: f64,
    pub hidden_size: i64,
    pub initializer_range: f64,
    pub intermediate_size: i64,
    pub max_position_embeddings: i64,
    pub num_attention_heads: i64,
    pub num_hidden_layers: i64,
    pub type_vocab_size: i64,
    pub vocab_size: i64,
    pub position_biased_input: Option<bool>,
    #[serde(default, deserialize_with = "deserialize_attention_type")]
    pub pos_att_type: Option<PositionAttentionTypes>,
    pub pooler_dropout: Option<f64>,
    pub pooler_hidden_act: Option<Activation>,
    pub pooler_hidden_size: Option<i64>,
    pub layer_norm_eps: Option<f64>,
    pub pad_token_id: Option<i64>,
    pub relative_attention: Option<bool>,
    pub max_relative_positions: Option<i64>,
    pub embedding_size: Option<i64>,
    pub talking_head: Option<bool>,
    pub output_hidden_states: Option<bool>,
    pub output_attentions: Option<bool>,
    pub classifier_activation: Option<bool>,
    pub classifier_dropout: Option<f64>,
    pub is_decoder: Option<bool>,
    pub id2label: Option<HashMap<i64, String>>,
    pub label2id: Option<HashMap<String, i64>>,
}
