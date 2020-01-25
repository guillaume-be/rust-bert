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
    activation: Activation,
    attention_dropout: f32,
    dim: usize,
    dropout: f32,
    hidden_dim: usize,
    id2label: HashMap<i32, String>,
    initializer_range: f32,
    is_decoder: bool,
    label2id: HashMap<String, i32>,
    max_position_embeddings: usize,
    n_heads: usize,
    n_layers: usize,
    num_labels: usize,
    output_attentions: bool,
    output_hidden_states: bool,
    output_past: bool,
    qa_dropout: f32,
    seq_classifier_dropout: f32,
    sinusoidal_pos_embds: bool,
    tie_weights: bool,
    torchscript: bool,
    use_bfloat16: bool,
    vocab_size: usize,
}

impl DistilBertConfig {
    pub fn from_file(path: &Path) -> DistilBertConfig {
        let f = File::open(path).expect("Could not open configuration file.");
        let br = BufReader::new(f);
        let config: DistilBertConfig = serde_json::from_reader(br).expect("could not parse configuration");
        config
    }
}

fn gelu(x: Tensor) -> Tensor {
    &x * 0.5 * (1.0 + (x / ((2.0 as f64).sqrt())).erf())
}

fn create_sinusoidal_embeddings() {}