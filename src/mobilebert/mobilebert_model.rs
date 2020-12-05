// Copyright (c) 2020  The Google AI Language Team Authors, The HuggingFace Inc. team and github/lonePatient
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
use crate::common::activations::Activation;
use crate::Config;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// # BERT Pretrained model weight files
pub struct MobileBertModelResources;

/// # BERT Pretrained model config files
pub struct MobileBertConfigResources;

/// # BERT Pretrained model vocab files
pub struct MobileBertVocabResources;

impl MobileBertModelResources {
    /// Shared under Apache 2.0 license by the Google team at https://huggingface.co/google/mobilebert-uncased. Modified with conversion to C-array format.
    pub const MOBILEBERT_UNCASED: (&'static str, &'static str) = (
        "mobilebert-uncased/model",
        "https://huggingface.co/google/mobilebert-uncased/resolve/main/rust_model.ot",
    );
}

impl MobileBertConfigResources {
    /// Shared under Apache 2.0 license by the Google team at https://huggingface.co/google/mobilebert-uncased. Modified with conversion to C-array format.
    pub const MOBILEBERT_UNCASED: (&'static str, &'static str) = (
        "mobilebert-uncased/config",
        "https://huggingface.co/google/mobilebert-uncased/resolve/main/config.json",
    );
}

impl MobileBertVocabResources {
    /// Shared under Apache 2.0 license by the Google team at https://huggingface.co/google/mobilebert-uncased. Modified with conversion to C-array format.
    pub const MOBILEBERT_UNCASED: (&'static str, &'static str) = (
        "mobilebert-uncased/vocab",
        "https://huggingface.co/google/mobilebert-uncased/resolve/main/vocab.txt",
    );
}

#[allow(non_camel_case_types)]
#[derive(Clone, Debug, Serialize, Deserialize, Copy)]
pub enum NormalizationType {
    layer_norm,
    no_norm,
}

#[derive(Debug, Serialize, Deserialize)]
/// # MobileBERT model configuration
/// Defines the MobileBERT model architecture (e.g. number of layers, hidden layer size, label mapping...)
pub struct MobileBertConfig {
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
    pub trigram_input: Option<bool>,
    pub use_bottleneck: Option<bool>,
    pub use_bottleneck_attention: Option<bool>,
    pub intra_bottleneck_size: Option<i64>,
    pub key_query_shared_bottleneck: Option<bool>,
    pub num_feedforward_networks: Option<i64>,
    pub normalization_type: Option<NormalizationType>,
    pub output_attentions: Option<bool>,
    pub output_hidden_states: Option<bool>,
    pub is_decoder: Option<bool>,
    pub id2label: Option<HashMap<i64, String>>,
    pub label2id: Option<HashMap<String, i64>>,
}

impl Config<MobileBertConfig> for MobileBertConfig {}
