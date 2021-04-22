// Copyright 2021 The Eleuther AI and HuggingFace Inc. team. All rights reserved.
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

/// # GPT-Neo Pretrained model weight files
pub struct GptNeoModelResources;

/// # GPT-Neo Pretrained model config files
pub struct GptNeoConfigResources;

/// # GPT-Neo Pretrained model vocab files
pub struct GptNeoVocabResources;

/// # GPT-Neo Pretrained model merges files
pub struct GptNeoMergesResources;

impl GptNeoModelResources {
    /// Shared under Apache 2.0 license by the EleutherAI contributors at https://www.eleuther.ai. Modified with conversion to C-array format.
    pub const GPT_NEO_125M: (&'static str, &'static str) = (
        "gpt-neo-125M/model",
        "https://huggingface.co/EleutherAI/gpt-neo-125M/resolve/main/rust_model.ot",
    );
}

impl GptNeoConfigResources {
    /// Shared under Apache 2.0 license by the EleutherAI contributors at https://www.eleuther.ai. Modified with conversion to C-array format.
    pub const GPT_NEO_125M: (&'static str, &'static str) = (
        "gpt-neo-125M/config",
        "https://huggingface.co/EleutherAI/gpt-neo-125M/resolve/main/config.json",
    );
}

impl GptNeoVocabResources {
    /// Shared under Modified MIT license by the OpenAI team at https://github.com/openai/gpt-2/blob/master/LICENSE. Modified with conversion to C-array format.
    pub const GPT_NEO_125M: (&'static str, &'static str) = (
        "gpt-neo-125M/vocab",
        "https://huggingface.co/gpt2/resolve/main/vocab.json",
    );
}

impl GptNeoMergesResources {
    /// Shared under Modified MIT license by the OpenAI team at https://github.com/openai/gpt-2/blob/master/LICENSE. Modified with conversion to C-array format.
    pub const GPT_NEO_125M: (&'static str, &'static str) = (
        "gpt-neo-125M/merges",
        "https://huggingface.co/gpt2/resolve/main/merges.txt",
    );
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
#[serde(rename_all = "camelCase")]
/// #GPT-Neo attention layer type
pub enum AttentionLayerType {
    Global,
    Local,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
/// # GPT-Neo model configuration
/// Defines the GPT-Neo model architecture (e.g. number of layers, hidden layer size, vocab size...).
pub struct GptNeoConfig {
    pub activation_function: Activation,
    pub attention_dropout: f64,
    pub attention_layers: Vec<AttentionLayerType>,
    pub attention_types: Vec<(Vec<AttentionLayerType>, i64)>,
    pub intermediate_size: Option<i64>,
    pub bos_token_id: i64,
    pub eos_token_id: i64,
    pub vocab_size: i64,
    pub num_layers: i64,
    pub num_heads: i64,
    pub hidden_size: i64,
    pub window_size: i64,
    pub embed_dropout: f64,
    pub initializer_range: f64,
    pub layer_norm_epsilon: f64,
    pub max_position_embeddings: i64,
    pub output_past: Option<bool>,
    pub output_attentions: Option<bool>,
    pub output_hidden_states: Option<bool>,
    pub resid_dropout: f64,
}

impl Config<GptNeoConfig> for GptNeoConfig {}
