// Copyright 2022 Google LLC., LongT5 Authors and HuggingFace Inc. team.
// Copyright 2022 Guillaume Becquin
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::t5::{FeedForwardProj, T5Attention, T5Config, TaskSpecificParams};
use crate::Config;
use serde::{Deserialize, Serialize};

/// # LongT5 Pretrained model weight files
pub struct LongT5ModelResources;

/// # LongT5 Pretrained model config files
pub struct LongT5ConfigResources;

/// # LongT5 Pretrained model vocab files
pub struct LongT5VocabResources;

impl LongT5ModelResources {
    /// Shared under Apache 2.0 license at <https://huggingface.co/pszemraj/long-t5-tglobal-base-16384-book-summary>. Modified with conversion to C-array format.
    pub const TGLOBAL_BASE_BOOK_SUMMARY: (&'static str, &'static str) = (
        "longt5-tglobal-base-book-summary/model",
        "https://huggingface.co/pszemraj/long-t5-tglobal-base-16384-book-summary/resolve/main/rust_model.ot",
    );
}

impl LongT5ConfigResources {
    /// Shared under Apache 2.0 license at <https://huggingface.co/pszemraj/long-t5-tglobal-base-16384-book-summary>. Modified with conversion to C-array format.
    pub const TGLOBAL_BASE_BOOK_SUMMARY: (&'static str, &'static str) = (
        "longt5-tglobal-base-book-summary/config",
        "https://huggingface.co/pszemraj/long-t5-tglobal-base-16384-book-summary/resolve/main/config.json",
    );
}

impl LongT5VocabResources {
    /// Shared under Apache 2.0 license at <https://huggingface.co/pszemraj/long-t5-tglobal-base-16384-book-summary>. Modified with conversion to C-array format.
    pub const TGLOBAL_BASE_BOOK_SUMMARY: (&'static str, &'static str) = (
        "longt5-tglobal-base-book-summary/spiece",
        "https://huggingface.co/pszemraj/long-t5-tglobal-base-16384-book-summary/resolve/main/spiece.model",
    );
}

#[derive(Clone, Debug, Serialize, Deserialize, Copy)]
#[serde(rename_all = "kebab-case")]
/// # Options for LongT5 encoder attention type
pub enum EncoderAttentionType {
    /// Local
    Local,
    /// Transient Global
    TransientGlobal,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
/// # LongT5 model configuration
/// Defines the LongT5 model architecture (e.g. number of layers, hidden layer size, label mapping...)
pub struct LongT5Config {
    pub dropout_rate: f64,
    pub d_model: i64,
    pub d_ff: i64,
    pub d_kv: i64,
    pub decoder_start_token_id: Option<i64>,
    pub eos_token_id: Option<i64>,
    pub initializer_factor: f64,
    pub is_encoder_decoder: Option<bool>,
    pub layer_norm_epsilon: f64,
    pub num_heads: i64,
    pub num_layers: i64,
    pub num_decoder_layers: Option<i64>,
    pub local_radius: i64,
    pub global_block_size: i64,
    pub output_past: Option<bool>,
    pub pad_token_id: Option<i64>,
    pub relative_attention_num_buckets: i64,
    pub relative_attention_max_distance: Option<i64>,
    pub encoder_attention_type: Option<EncoderAttentionType>,
    pub vocab_size: i64,
    pub feed_forward_proj: Option<FeedForwardProj>,
    pub tie_word_embeddings: Option<bool>,
    pub task_specific_params: Option<TaskSpecificParams>,
    pub output_attentions: Option<bool>,
    pub output_hidden_states: Option<bool>,
}

impl Config for LongT5Config {}

impl Default for LongT5Config {
    fn default() -> Self {
        LongT5Config {
            dropout_rate: 0.1,
            d_model: 512,
            d_ff: 2048,
            d_kv: 64,
            decoder_start_token_id: None,
            eos_token_id: Some(1),
            initializer_factor: 1.0,
            is_encoder_decoder: None,
            layer_norm_epsilon: 1e-6,
            num_heads: 8,
            num_layers: 6,
            num_decoder_layers: None,
            local_radius: 127,
            global_block_size: 16,
            output_past: None,
            pad_token_id: Some(0),
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: Some(128),
            encoder_attention_type: Some(EncoderAttentionType::Local),
            vocab_size: 32128,
            feed_forward_proj: Some(FeedForwardProj::Relu),
            tie_word_embeddings: None,
            task_specific_params: None,
            output_attentions: None,
            output_hidden_states: None,
        }
    }
}

impl Into<T5Config> for &LongT5Config {
    fn into(self) -> T5Config {
        T5Config {
            dropout_rate: self.dropout_rate,
            d_model: self.d_model,
            d_ff: self.d_ff,
            d_kv: self.d_kv,
            decoder_start_token_id: self.decoder_start_token_id,
            bos_token_id: None,
            eos_token_id: self.eos_token_id,
            initializer_factor: self.initializer_factor,
            is_encoder_decoder: self.is_encoder_decoder,
            layer_norm_epsilon: self.layer_norm_epsilon,
            num_heads: self.num_heads,
            num_layers: self.num_layers,
            output_past: self.output_past,
            pad_token_id: self.pad_token_id,
            relative_attention_num_buckets: self.relative_attention_num_buckets,
            relative_attention_max_distance: self.relative_attention_max_distance,
            vocab_size: self.vocab_size,
            feed_forward_proj: self.feed_forward_proj,
            tie_word_embeddings: self.tie_word_embeddings,
            task_specific_params: self.task_specific_params.clone(),
            output_attentions: self.output_attentions,
            output_hidden_states: self.output_hidden_states,
        }
    }
}
