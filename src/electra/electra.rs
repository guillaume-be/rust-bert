// Copyright 2020 The Google Research Authors.
// Copyright 2019-present, the HuggingFace Inc. team
// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
// Copyright 2019 Guillaume Becquin
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::bert::Activation;
use crate::Config;

#[derive(Debug, Serialize, Deserialize)]
/// # Electra model configuration
/// Defines the Electra model architecture (e.g. number of layers, hidden layer size, label mapping...)
pub struct ElectraConfig {
    pub hidden_act: Activation,
    pub attention_probs_dropout_prob: f64,
    pub embedding_size: i64,
    pub hidden_dropout_prob: f64,
    pub hidden_size: i64,
    pub initializer_range: f32,
    pub layer_norm_eps: Option<f64>,
    pub intermediate_size: i64,
    pub max_position_embeddings: i64,
    pub num_attention_heads: i64,
    pub num_hidden_layers: i64,
    pub type_vocab_size: i64,
    pub vocab_size: i64,
    pub pad_token_id: i64,
    pub output_past: Option<bool>,
    pub output_attentions: Option<bool>,
    pub output_hidden_states: Option<bool>,
    pub id2label: Option<HashMap<i64, String>>,
    pub label2id: Option<HashMap<String, i64>>,
}

impl Config<ElectraConfig> for ElectraConfig {}