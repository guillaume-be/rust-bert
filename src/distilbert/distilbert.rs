// Copyright 2019-present, the HuggingFace Inc. team, The Google AI Language Team and Facebook, Inc.
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

extern crate tch;

use std::path::Path;
use tch::Tensor;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub enum Activation {
    Gelu,
    Relu,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DistilBertConfig {
    pub activation: Activation,
    pub attention_dropout: f32,
    pub dim: i64,
    pub dropout: f64,
    pub hidden_dim: i64,
    pub id2label: HashMap<i32, String>,
    pub initializer_range: f32,
    pub is_decoder: bool,
    pub label2id: HashMap<String, i32>,
    pub max_position_embeddings: i64,
    pub n_heads: i64,
    pub n_layers: i64,
    pub num_labels: i64,
    pub output_attentions: bool,
    pub output_hidden_states: bool,
    pub output_past: bool,
    pub qa_dropout: f32,
    pub seq_classifier_dropout: f32,
    pub sinusoidal_pos_embds: bool,
    pub tie_weights: bool,
    pub torchscript: bool,
    pub use_bfloat16: bool,
    pub vocab_size: i64,
}

impl DistilBertConfig {
    pub fn from_file(path: &Path) -> DistilBertConfig {
        let f = File::open(path).expect("Could not open configuration file.");
        let br = BufReader::new(f);
        let config: DistilBertConfig = serde_json::from_reader(br).expect("could not parse configuration");
        config
    }
}

fn _gelu(x: Tensor) -> Tensor {
    &x * 0.5 * (1.0 + (x / ((2.0 as f64).sqrt())).erf())
}