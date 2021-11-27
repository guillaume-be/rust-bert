// Copyright 2020, Microsoft and the HuggingFace Inc. team.
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

use crate::{Activation, Config};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// # DeBERTa Pretrained model weight files
pub struct DebertaModelResources;

/// # DeBERTa Pretrained model config files
pub struct DebertaConfigResources;

/// # DeBERTa Pretrained model vocab files
pub struct DebertaVocabResources;

/// # DeBERTa Pretrained model merges files
pub struct DebertaMergesResources;

impl DebertaModelResources {
    /// Shared under MIT license by the Microsoft team at <https://huggingface.co/microsoft/deberta-base>. Modified with conversion to C-array format.
    pub const DEBERTA_BASE: (&'static str, &'static str) = (
        "deberta-base/model",
        "https://huggingface.co/microsoft/deberta-base/resolve/main/rust_model.ot",
    );
}

impl DebertaConfigResources {
    /// Shared under MIT license by the Microsoft team at <https://huggingface.co/microsoft/deberta-base>. Modified with conversion to C-array format.
    pub const DEBERTA_BASE: (&'static str, &'static str) = (
        "deberta-base/config",
        "https://huggingface.co/microsoft/deberta-base/resolve/main/config.json",
    );
}

impl DebertaVocabResources {
    /// Shared under MIT license by the Microsoft team at <https://huggingface.co/microsoft/deberta-base>. Modified with conversion to C-array format.
    pub const DEBERTA_BASE: (&'static str, &'static str) = (
        "deberta-base/vocab",
        "https://huggingface.co/microsoft/deberta-base/resolve/main/vocab.json",
    );
}

impl DebertaMergesResources {
    /// Shared under MIT license by the Microsoft team at <https://huggingface.co/microsoft/deberta-base>. Modified with conversion to C-array format.
    pub const DEBERTA_BASE: (&'static str, &'static str) = (
        "deberta-base/merges",
        "https://huggingface.co/microsoft/deberta-base/resolve/main/merges.txt",
    );
}

#[allow(non_camel_case_types)]
#[derive(Clone, Debug, Serialize, Deserialize, Copy)]
/// # Position attention type to use for the DeBERTa model.
pub enum PositionAttentionType {
    p2c,
    c2p,
    p2p,
}

#[derive(Debug, Serialize, Deserialize)]
/// # DeBERTa model configuration
/// Defines the DeBERTa model architecture (e.g. number of layers, hidden layer size, label mapping...)
pub struct DebertaConfig {
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
    pub embedding_size: i64,
    pub relative_attention: bool,
    pub position_biased_input: bool,
    pub pos_att_type: Option<Vec<PositionAttentionType>>,
    pub pooler_dropout: Option<f64>,
    pub pooler_hidden: Option<Activation>,
    pub pooler_hidden_size: Option<i64>,
    pub layer_norm_eps: Option<f64>,
    pub pad_token_id: Option<i64>,
    pub output_hidden_states: Option<bool>,
    pub classifier_activation: Option<bool>,
    pub is_decoder: Option<bool>,
    pub id2label: Option<HashMap<i64, String>>,
    pub label2id: Option<HashMap<String, i64>>,
}

impl Config for DebertaConfig {}
