// Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
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
use crate::Config;
use serde::{Deserialize, Serialize};

/// # T5 Pretrained model weight files
pub struct T5ModelResources;

/// # T5 Pretrained model config files
pub struct T5ConfigResources;

/// # T5 Pretrained model vocab files
pub struct T5VocabResources;

impl T5ModelResources {
    /// Shared under Apache 2.0 license by the T5 Authors at https://github.com/google-research/text-to-text-transfer-transformer. Modified with conversion to C-array format.
    pub const T5_SMALL: (&'static str, &'static str) = (
        "t5-small/model.ot",
        "https://cdn.huggingface.co/t5-small/rust_model.ot",
    );
}

impl T5ConfigResources {
    /// Shared under Apache 2.0 license by the Google team at https://github.com/google-research/text-to-text-transfer-transformer.
    pub const T5_SMALL: (&'static str, &'static str) = (
        "t5-small/config.json",
        "https://cdn.huggingface.co/t5-small/config.json",
    );
}

impl T5VocabResources {
    /// Shared under Apache 2.0 license by the Google team at https://github.com/google-research/text-to-text-transfer-transformer.
    pub const T5_SMALL: (&'static str, &'static str) = (
        "t5-small/spiece.model",
        "https://s3.amazonaws.com/models.huggingface.co/bert/t5-spiece.model",
    );
}

#[derive(Debug, Serialize, Deserialize)]
/// # T5 model configuration
/// Defines the T5 model architecture (e.g. number of layers, hidden layer size, label mapping...)
pub struct T5Config {
    pub dropout_rate: f64,
    pub d_model: i64,
    pub d_ff: i64,
    pub d_kv: i64,
    pub decoder_start_token_id: Option<i64>,
    pub eos_token_id: Option<i64>,
    pub initializer_factor: f64,
    pub is_encoder_decoder: Option<bool>,
    pub layer_norm_epsilon: f64,
    pub n_positions: i64,
    pub num_heads: i64,
    pub num_layers: i64,
    pub output_past: Option<bool>,
    pub pad_token_id: Option<i64>,
    pub relative_attention_num_buckets: i64,
    pub vocab_size: i64,
    task_specific_params: TaskSpecificParams,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TaskSpecificParams {
    summarization: Summarization,
    translation_en_to_de: TranslationEnToDe,
    translation_en_to_fr: TranslationEnToFr,
    translation_en_to_ro: TranslationEnToRo,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Summarization {
    early_stopping: bool,
    length_penalty: f64,
    max_length: i64,
    min_length: i64,
    no_repeat_ngram_size: i64,
    num_beams: i64,
    prefix: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TranslationEnToDe {
    early_stopping: bool,
    max_length: i64,
    num_beams: i64,
    prefix: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TranslationEnToFr {
    early_stopping: bool,
    max_length: i64,
    num_beams: i64,
    prefix: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TranslationEnToRo {
    early_stopping: bool,
    max_length: i64,
    num_beams: i64,
    prefix: String,
}

impl Config<T5Config> for T5Config {}
