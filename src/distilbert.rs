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
use tch::{Tensor, nn};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use serde::{Deserialize, Serialize};
use self::tch::nn::{ModuleT, embedding};
use tch::Kind;

#[derive(Debug, Serialize, Deserialize)]
pub enum Activation {
    Gelu,
    Relu,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DistilBertConfig {
    activation: Activation,
    attention_dropout: f32,
    dim: i64,
    dropout: f32,
    hidden_dim: i64,
    id2label: HashMap<i32, String>,
    initializer_range: f32,
    is_decoder: bool,
    label2id: HashMap<String, i32>,
    max_position_embeddings: i64,
    n_heads: i64,
    n_layers: i64,
    num_labels: i64,
    output_attentions: bool,
    output_hidden_states: bool,
    output_past: bool,
    qa_dropout: f32,
    seq_classifier_dropout: f32,
    sinusoidal_pos_embds: bool,
    tie_weights: bool,
    torchscript: bool,
    use_bfloat16: bool,
    vocab_size: i64,
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

//ToDo: add sinusoidal embeddings creation, layernorm and dropout to embeddings
//ToDo: refactor embeddings layer as a struct with Impl block
//fn create_sinusoidal_embeddings() {}


pub fn embeddings(p: nn::Path, config: DistilBertConfig) -> impl ModuleT {
    let word_embeddings: nn::Embedding = embedding(&p / "word_embeddings",
                                                   config.vocab_size,
                                                   config.dim,
                                                   Default::default());
    let position_embeddings: nn::Embedding = embedding(&p / "position_embeddings",
                                                       config.max_position_embeddings,
                                                       config.dim,
                                                       Default::default());

    nn::func_t(move |input_ids, _train| {
        let seq_length = (&input_ids).size().last().unwrap().to_owned();
        let position_ids = Tensor::arange(seq_length, (Kind::Int64, input_ids.device()));
        let position_ids = position_ids.unsqueeze(0).expand_as(input_ids);

        let word_embed = input_ids.apply(&word_embeddings);
        let position_embed = position_ids.apply(&position_embeddings);

        let embeddings = word_embed + position_embed;

        embeddings
    })
}