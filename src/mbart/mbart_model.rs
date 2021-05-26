// Copyright 2021, The Facebook AI Research Team and The HuggingFace Inc. team. All rights reserved.
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

use crate::common::dropout::Dropout;
use crate::{Activation, Config};
use serde::{Deserialize, Serialize};
use std::borrow::Borrow;
use std::collections::HashMap;
use tch::kind::Kind::Int64;
use tch::{nn, Tensor};

/// # MBART Pretrained model weight files
pub struct MBartModelResources;

/// # MBART Pretrained model config files
pub struct MBartConfigResources;

/// # MBART Pretrained model vocab files
pub struct MBartVocabResources;

impl MBartModelResources {
    /// Shared under MIT license by the Facebook AI Research Fairseq team at https://github.com/pytorch/fairseq. Modified with conversion to C-array format.
    pub const MBART50_MANY_TO_MANY: (&'static str, &'static str) = (
        "mbart/model",
        "https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt/resolve/main/rust_model.ot",
    );
}

impl MBartConfigResources {
    /// Shared under MIT license by the Facebook AI Research Fairseq team at https://github.com/pytorch/fairseq. Modified with conversion to C-array format.
    pub const MBART50_MANY_TO_MANY: (&'static str, &'static str) = (
        "mbart/config",
        "https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt/resolve/main/config.json",
    );
}

impl MBartVocabResources {
    /// Shared under MIT license by the Facebook AI Research Fairseq team at https://github.com/pytorch/fairseq. Modified with conversion to C-array format.
    pub const MBART50_MANY_TO_MANY: (&'static str, &'static str) = (
        "mbart/vocab",
        "https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt/resolve/main/sentencepiece.bpe.model",
    );
}

#[derive(Debug, Serialize, Deserialize, Clone)]
/// # MBART model configuration
/// Defines the MBART model architecture (e.g. number of layers, hidden layer size, label mapping...)
pub struct MBartConfig {
    pub vocab_size: i64,
    pub max_position_embeddings: i64,
    pub encoder_layers: i64,
    pub encoder_attention_heads: i64,
    pub encoder_ffn_dim: i64,
    pub encoder_layerdrop: f64,
    pub decoder_layers: i64,
    pub decoder_ffn_dim: i64,
    pub decoder_attention_heads: i64,
    pub decoder_layerdrop: f64,
    pub is_encoder_decoder: Option<bool>,
    pub activation_function: Option<Activation>,
    pub d_model: i64,
    pub dropout: f64,
    pub activation_dropout: f64,
    pub attention_dropout: f64,
    pub classifier_dropout: Option<f64>,
    pub scale_embedding: Option<bool>,
    pub bos_token_id: Option<i64>,
    pub eos_token_id: Option<i64>,
    pub pad_token_id: Option<i64>,
    pub forced_eos_token_id: Option<i64>,
    pub decoder_start_token_id: Option<i64>,
    pub id2label: Option<HashMap<i64, String>>,
    pub label2id: Option<HashMap<String, i64>>,
    pub init_std: f64,
    pub min_length: Option<i64>,
    pub no_repeat_ngram_size: Option<i64>,
    pub normalize_embedding: Option<bool>,
    pub num_hidden_layers: i64,
    pub output_attentions: Option<bool>,
    pub output_hidden_states: Option<bool>,
    pub output_past: Option<bool>,
    pub static_position_embeddings: Option<bool>,
}

impl Config<MBartConfig> for MBartConfig {}

fn _shift_tokens_right(input_ids: &Tensor, pad_token_id: i64) -> Tensor {
    let output = input_ids.masked_fill(&input_ids.eq(-100), pad_token_id);
    let index_eos: Tensor = input_ids.ne(pad_token_id).sum1(&[1], true, Int64) - 1;
    output
        .select(1, 0)
        .copy_(&input_ids.gather(1, &index_eos, true).squeeze());
    output
        .slice(1, 1, *output.size().last().unwrap(), 1)
        .copy_(&input_ids.slice(1, 0, *output.size().last().unwrap() - 1, 1));
    output
}

pub struct MBartClassificationHead {
    dense: nn::Linear,
    dropout: Dropout,
    out_proj: nn::Linear,
}

impl MBartClassificationHead {
    pub fn new<'p, P>(p: P, config: &MBartConfig) -> MBartClassificationHead
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let dense = nn::linear(
            p / "dense",
            config.d_model,
            config.d_model,
            Default::default(),
        );

        let num_labels = config
            .id2label
            .as_ref()
            .expect("id2label not provided in configuration")
            .len() as i64;

        let out_proj = nn::linear(
            p / "out_proj",
            config.d_model,
            num_labels,
            Default::default(),
        );

        let dropout = Dropout::new(config.classifier_dropout.unwrap_or(0.0));

        MBartClassificationHead {
            dense,
            dropout,
            out_proj,
        }
    }

    pub fn forward_t(&self, hidden_states: &Tensor, train: bool) -> Tensor {
        hidden_states
            .apply_t(&self.dropout, train)
            .apply(&self.dense)
            .tanh()
            .apply_t(&self.dropout, train)
            .apply(&self.out_proj)
    }
}
