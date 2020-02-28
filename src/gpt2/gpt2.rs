// Copyright 2018-present, the HuggingFace Inc. team
// Copyright 2018-present, The OpenAI Team Authors
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

use crate::common::config::Config;
use serde::{Deserialize, Serialize};
use tch::nn;
use crate::common::dropout::Dropout;
use tch::nn::embedding;

#[derive(Debug, Serialize, Deserialize)]
pub struct Gpt2Config {
    pub attn_pdrop: Option<f64>,
    pub embd_pdrop: Option<f64>,
    pub hidden_dropout_prob: Option<f64>,
    pub initializer_range: f64,
    pub layer_norm_epsilon: f64,
    pub n_ctx: i64,
    pub n_embd: i64,
    pub n_head: i64,
    pub n_layer: i64,
    pub n_positions: i64,
    pub num_labels: Option<i64>,
    pub output_attentions: Option<bool>,
    pub output_hidden_states: Option<bool>,
    pub resid_pdrop: Option<f64>,
    pub vocab_size: i64,
}

impl Config<Gpt2Config> for Gpt2Config {}

pub struct Gpt2Model {
    _wte: nn::Embedding,
    _wpe: nn::Embedding,
    _drop: Dropout,
    _ln_f: nn::LayerNorm,
}

impl Gpt2Model {
    pub fn new(p: &nn::Path, config: &Gpt2Config) -> Gpt2Model {
        let wte = embedding(&(p / "wte"), config.vocab_size, config.n_embd, Default::default());
        let wpe = embedding(&(p / "wpe"), config.n_positions, config.n_embd, Default::default());

        let embd_pdrop = match config.embd_pdrop {
            Some(value) => value,
            None => 0.1
        };
        let drop = Dropout::new(embd_pdrop);
        let layer_norm_config = nn::LayerNormConfig { eps: config.layer_norm_epsilon, ..Default::default() };
        let ln_f = nn::layer_norm(p / "ln_f ", vec![config.n_embd], layer_norm_config);

        Gpt2Model { _wte: wte, _wpe: wpe, _drop: drop, _ln_f: ln_f }
    }

}

