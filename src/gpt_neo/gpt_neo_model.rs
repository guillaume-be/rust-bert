// Copyright 2021 The Eleuther AI and HuggingFace Inc. team. All rights reserved.
// Copyright 2021 Guillaume Becquin
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
use crate::gpt_neo::decoder::GptNeoBlock;
use crate::gpt_neo::LayerState;
use crate::{Activation, Config, RustBertError};
use serde::{Deserialize, Serialize};
use std::borrow::Borrow;
use tch::{nn, Kind, Tensor};

/// # GPT-Neo Pretrained model weight files
pub struct GptNeoModelResources;

/// # GPT-Neo Pretrained model config files
pub struct GptNeoConfigResources;

/// # GPT-Neo Pretrained model vocab files
pub struct GptNeoVocabResources;

/// # GPT-Neo Pretrained model merges files
pub struct GptNeoMergesResources;

impl GptNeoModelResources {
    /// Shared under Apache 2.0 license by the EleutherAI contributors at https://www.eleuther.ai. Modified with conversion to C-array format.
    pub const GPT_NEO_125M: (&'static str, &'static str) = (
        "gpt-neo-125M/model",
        "https://huggingface.co/EleutherAI/gpt-neo-125M/resolve/main/rust_model.ot",
    );
}

impl GptNeoConfigResources {
    /// Shared under Apache 2.0 license by the EleutherAI contributors at https://www.eleuther.ai. Modified with conversion to C-array format.
    pub const GPT_NEO_125M: (&'static str, &'static str) = (
        "gpt-neo-125M/config",
        "https://huggingface.co/EleutherAI/gpt-neo-125M/resolve/main/config.json",
    );
}

impl GptNeoVocabResources {
    /// Shared under Modified MIT license by the OpenAI team at https://github.com/openai/gpt-2/blob/master/LICENSE. Modified with conversion to C-array format.
    pub const GPT_NEO_125M: (&'static str, &'static str) = (
        "gpt-neo-125M/vocab",
        "https://huggingface.co/gpt2/resolve/main/vocab.json",
    );
}

impl GptNeoMergesResources {
    /// Shared under Modified MIT license by the OpenAI team at https://github.com/openai/gpt-2/blob/master/LICENSE. Modified with conversion to C-array format.
    pub const GPT_NEO_125M: (&'static str, &'static str) = (
        "gpt-neo-125M/merges",
        "https://huggingface.co/gpt2/resolve/main/merges.txt",
    );
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
#[serde(rename_all = "camelCase")]
/// #GPT-Neo attention layer type
pub enum AttentionLayerType {
    Global,
    Local,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
/// # GPT-Neo model configuration
/// Defines the GPT-Neo model architecture (e.g. number of layers, hidden layer size, vocab size...).
pub struct GptNeoConfig {
    pub activation_function: Activation,
    pub attention_dropout: f64,
    pub attention_layers: Vec<AttentionLayerType>,
    pub attention_types: Vec<(Vec<AttentionLayerType>, i64)>,
    pub intermediate_size: Option<i64>,
    pub bos_token_id: i64,
    pub eos_token_id: i64,
    pub vocab_size: i64,
    pub num_layers: i64,
    pub num_heads: i64,
    pub hidden_size: i64,
    pub window_size: i64,
    pub embed_dropout: f64,
    pub initializer_range: f64,
    pub layer_norm_epsilon: f64,
    pub max_position_embeddings: i64,
    pub output_past: Option<bool>,
    pub output_attentions: Option<bool>,
    pub output_hidden_states: Option<bool>,
    pub resid_dropout: f64,
}

impl Config<GptNeoConfig> for GptNeoConfig {}

pub struct GptNeoModel {
    word_embeddings: nn::Embedding,
    position_embeddings: nn::Embedding,
    layers: Vec<GptNeoBlock>,
    dropout: Dropout,
    layer_norm: nn::LayerNorm,
    output_attentions: bool,
    output_hidden_states: bool,
}

impl GptNeoModel {
    pub fn new<'p, P>(p: P, config: &GptNeoConfig) -> Result<GptNeoModel, RustBertError>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let word_embeddings = nn::embedding(
            p / "wte",
            config.vocab_size,
            config.hidden_size,
            Default::default(),
        );

        let position_embeddings = nn::embedding(
            p / "wpe",
            config.max_position_embeddings,
            config.hidden_size,
            Default::default(),
        );

        let dropout = Dropout::new(config.embed_dropout);

        let layer_norm_config = nn::LayerNormConfig {
            eps: config.layer_norm_epsilon,
            ..Default::default()
        };

        let layer_norm = nn::layer_norm(p / "ln_f", vec![config.hidden_size], layer_norm_config);

        let mut layers: Vec<GptNeoBlock> = Vec::with_capacity(config.num_layers as usize);
        let p_layers = p / "h";
        for layer_index in 0..config.num_layers {
            layers.push(GptNeoBlock::new(
                &p_layers / layer_index,
                layer_index as usize,
                config,
            )?);
        }

        let output_attentions = config.output_attentions.unwrap_or(false);
        let output_hidden_states = config.output_hidden_states.unwrap_or(false);

        Ok(GptNeoModel {
            word_embeddings,
            position_embeddings,
            layers,
            dropout,
            layer_norm,
            output_attentions,
            output_hidden_states,
        })
    }

    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        layer_states: Option<Vec<Option<LayerState>>>,
        attention_mask: Option<&Tensor>,
        train: bool,
    ) -> Result<GptNeoOutput, RustBertError> {
        let (calc_input_embeddings, input_shape, device) = if let Some(input_ids) = input_ids {
            if input_embeds.is_none() {
                (
                    Some(input_ids.apply(&self.word_embeddings)),
                    input_ids.size(),
                    input_ids.device(),
                )
            } else {
                return Err(RustBertError::ValueError(
                    "Only one of input ids or input embeddings may be set".into(),
                ));
            }
        } else if let Some(input_embeds) = input_embeds {
            let mut input_shape = input_embeds.size();
            let _ = input_shape.pop();
            (None, input_shape, input_embeds.device())
        } else {
            return Err(RustBertError::ValueError(
                "At least one of input ids or input embeddings must be set".into(),
            ));
        };

        let (batch_size, current_sequence_length) = (input_shape[0], input_shape[1]);

        let past_length = if let Some(past_state_value) = &layer_states {
            if let Some(first_layer_state) = &past_state_value[0] {
                let mut size_iter = first_layer_state.prev_key.size().into_iter().rev();
                size_iter.next();
                size_iter.next().unwrap()
            } else {
                0
            }
        } else {
            0
        };

        let calc_position_ids = if position_ids.is_none() {
            let position_ids = Tensor::arange1(
                past_length,
                current_sequence_length + past_length,
                (Kind::Int64, device),
            );
            Some(
                position_ids
                    .unsqueeze(0)
                    .view([-1, current_sequence_length]),
            )
        } else {
            None
        };

        let position_ids = position_ids.unwrap_or_else(|| calc_position_ids.as_ref().unwrap());
    }
}

/// Container for the GPT-Neo model output.
pub struct GptNeoOutput {
    /// Last hidden states from the model
    pub hidden_states: Tensor,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all intermediate layers
    pub all_attentions: Option<Vec<Tensor>>,
}
