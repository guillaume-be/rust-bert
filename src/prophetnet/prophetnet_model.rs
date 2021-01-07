// Copyright 2020 The Microsoft Authors and The HuggingFace Inc. team.
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

/// # ProphetNet Pretrained model weight files
pub struct ProphetNetModelResources;

/// # ProphetNet Pretrained model config files
pub struct ProphetNetConfigResources;

/// # ProphetNet Pretrained model vocab files
pub struct ProphetNetVocabResources;

impl ProphetNetModelResources {
    /// Shared under MIT license by the Microsoft team at https://github.com/microsoft/ProphetNet. Modified with conversion to C-array format.
    pub const PROPHETNET_LARGE_UNCASED: (&'static str, &'static str) = (
        "prophetnet-large-uncased/model",
        "https://huggingface.co/microsoft/prophetnet-large-uncased/resolve/main/rust_model.ot",
    );
}

impl ProphetNetConfigResources {
    /// Shared under MIT license by the Microsoft team at https://github.com/microsoft/ProphetNet. Modified with conversion to C-array format.
    pub const PROPHETNET_LARGE_UNCASED: (&'static str, &'static str) = (
        "prophetnet-large-uncased/config",
        "https://huggingface.co/microsoft/prophetnet-large-uncased/resolve/main/config.json",
    );
}

impl ProphetNetVocabResources {
    /// Shared under MIT license by the Microsoft team at https://github.com/microsoft/ProphetNet. Modified with conversion to C-array format.
    pub const PROPHETNET_LARGE_UNCASED: (&'static str, &'static str) = (
        "prophetnet-large-uncased/vocab",
        "https://huggingface.co/microsoft/prophetnet-large-uncased/resolve/main/prophetnet.tokenizer",
    );
}

#[derive(Debug, Serialize, Deserialize)]
/// # ProphetNet model configuration
/// Defines the ProphetNet model architecture (e.g. number of layers, hidden layer size, label mapping...)
pub struct ProphetNetConfig {
    pub activation_function: Activation,
    pub activation_dropout: f64,
    pub attention_dropout: f64,
    pub decoder_ffn_dim: i64,
    pub decoder_layerdrop: f64,
    pub decoder_max_position_embeddings: i64,
    pub decoder_start_token_id: i64,
    pub disable_ngram_loss: bool,
    pub dropout: f64,
    pub encoder_ffn_dim: i64,
    pub encoder_layerdrop: f64,
    pub encoder_max_position_embeddings: i64,
    pub eps: f64,
    pub hidden_size: i64,
    pub init_std: f64,
    pub is_encoder_decoder: bool,
    pub max_position_embeddings: i64,
    pub bos_token_id: i64,
    pub eos_token_id: i64,
    pub ngram: i64,
    pub id2label: Option<HashMap<i64, String>>,
    pub label2id: Option<HashMap<String, i64>>,
    pub num_buckets: i64,
    pub num_decoder_attention_heads: i64,
    pub num_decoder_layers: i64,
    pub num_encoder_attention_heads: i64,
    pub num_encoder_layers: i64,
    pub output_past: Option<bool>,
    pub pad_token_id: i64,
    pub relative_max_distance: i64,
    pub vocab_size: i64,
    pub output_attentions: Option<bool>,
    pub output_hidden_states: Option<bool>,
}

impl Config<ProphetNetConfig> for ProphetNetConfig {}
