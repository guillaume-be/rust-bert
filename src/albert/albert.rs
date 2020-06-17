// Copyright 2018 Google AI and Google Brain team.
// Copyright 2020-present, the HuggingFace Inc. team.
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


use std::collections::HashMap;
use crate::Config;
use serde::{Deserialize, Serialize};

#[allow(non_camel_case_types)]
#[derive(Clone, Debug, Serialize, Deserialize)]
/// # Activation function used in the attention layer and masked language model head
pub enum Activation {
    /// Gaussian Error Linear Unit ([Hendrycks et al., 2016,](https://arxiv.org/abs/1606.08415))
    gelu_new,
    /// Gaussian Error Linear Unit ([Hendrycks et al., 2016,](https://arxiv.org/abs/1606.08415))
    gelu,
    /// Rectified Linear Unit
    relu,
    /// Mish ([Misra, 2019](https://arxiv.org/abs/1908.08681))
    mish,
}


#[derive(Debug, Serialize, Deserialize)]
/// # ALBERT model configuration
/// Defines the ALBERT model architecture (e.g. number of layers, hidden layer size, label mapping...)
pub struct AlbertConfig {
    pub hidden_act: Activation,
    pub attention_probs_dropout_prob: f64,
    pub bos_token_id: i64,
    pub eos_token_id: i64,
    pub down_scale_factor: i64,
    pub embedding_size: i64,
    pub gap_size: i64,
    pub hidden_dropout_prob: f64,
    pub hidden_size: i64,
    pub initializer_range: f32,
    pub inner_group_num: i64,
    pub intermediate_size: i64,
    pub layer_norm_eps: Option<f64>,
    pub max_position_embeddings: i64,
    pub net_structure_type: i64,
    pub num_attention_heads: i64,
    pub num_hidden_groups: i64,
    pub num_hidden_layers: i64,
    pub num_memory_blocks: i64,
    pub pad_token_id: i64,
    pub type_vocab_size: i64,
    pub vocab_size: i64,
    pub output_attentions: Option<bool>,
    pub output_hidden_states: Option<bool>,
    pub is_decoder: Option<bool>,
    pub id2label: Option<HashMap<i64, String>>,
    pub label2id: Option<HashMap<String, i64>>,
}

impl Config<AlbertConfig> for AlbertConfig {}