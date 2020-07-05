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
use crate::t5::attention::LayerState;
use crate::t5::encoder::T5Stack;
use crate::Config;
use serde::{Deserialize, Serialize};
use std::borrow::Borrow;
use tch::nn::embedding;
use tch::{nn, Tensor};

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

pub struct T5Model {
    pub(crate) encoder: T5Stack,
    decoder: T5Stack,
    pub(crate) embeddings: nn::Embedding,
}

impl T5Model {
    pub fn new<'p, P>(
        p: P,
        config: &T5Config,
        output_attentions: bool,
        output_hidden_states: bool,
    ) -> T5Model
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let embeddings: nn::Embedding = embedding(
            p / "shared",
            config.vocab_size,
            config.d_model,
            Default::default(),
        );

        let encoder = T5Stack::new(
            p / "encoder",
            config,
            false,
            false,
            output_attentions,
            output_hidden_states,
        );
        let decoder = T5Stack::new(
            p / "decoder",
            config,
            true,
            true,
            output_attentions,
            output_hidden_states,
        );

        T5Model {
            encoder,
            decoder,
            embeddings,
        }
    }

    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        encoder_outputs: Option<(Tensor, Option<Vec<Tensor>>, Option<Vec<Tensor>>)>,
        decoder_input_ids: Option<&Tensor>,
        decoder_attention_mask: Option<&Tensor>,
        input_embeds: Option<Tensor>,
        decoder_input_embeds: Option<Tensor>,
        old_layer_states: Option<Vec<(Option<LayerState>, Option<LayerState>)>>,
        train: bool,
    ) -> (
        Tensor,
        Tensor,
        Option<Vec<(Option<LayerState>, Option<LayerState>)>>,
        Option<Vec<Tensor>>,
        Option<Vec<Tensor>>,
        Option<Vec<Tensor>>,
        Option<Vec<Tensor>>,
    ) {
        let (encoder_hidden_states, all_encoder_hidden_states, all_encoder_attentions) =
            match encoder_outputs {
                Some(value) => value,
                None => {
                    let (
                        encoder_hidden_states,
                        all_encoder_hidden_states,
                        all_encoder_attentions,
                        _,
                    ) = self
                        .encoder
                        .forward_t(
                            input_ids,
                            attention_mask,
                            None,
                            None,
                            input_embeds,
                            &self.embeddings,
                            None,
                            train,
                        )
                        .unwrap();
                    (
                        encoder_hidden_states,
                        all_encoder_hidden_states,
                        all_encoder_attentions,
                    )
                }
            };

        let (calculated_decoder_input_ids, calculated_decoder_input_embeds) =
            if old_layer_states.is_some() {
                let decoder_input_ids = match decoder_input_ids {
                    Some(value) => Some(value.select(1, -1)),
                    None => None,
                };
                let decoder_input_embeds = match &decoder_input_embeds {
                    Some(value) => Some(value.select(1, -1)),
                    None => None,
                };
                (decoder_input_ids, decoder_input_embeds)
            } else {
                (None, None)
            };

        let (decoder_input_ids, decoder_input_embeds) = if old_layer_states.is_some() {
            (
                calculated_decoder_input_ids.as_ref(),
                calculated_decoder_input_embeds,
            )
        } else {
            (decoder_input_ids, decoder_input_embeds)
        };

        let (decoder_outputs, all_decoder_hidden_states, all_decoder_attentions, decoder_cache) =
            self.decoder
                .forward_t(
                    decoder_input_ids,
                    decoder_attention_mask,
                    Some(&encoder_hidden_states),
                    attention_mask,
                    decoder_input_embeds,
                    &self.embeddings,
                    old_layer_states,
                    train,
                )
                .unwrap();
        (
            decoder_outputs,
            encoder_hidden_states,
            decoder_cache,
            all_decoder_hidden_states,
            all_decoder_attentions,
            all_encoder_hidden_states,
            all_encoder_attentions,
        )
    }
}
