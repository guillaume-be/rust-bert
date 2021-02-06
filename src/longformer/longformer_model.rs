// Copyright 2020 The Allen Institute for AI team and The HuggingFace Inc. team.
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

use crate::common::activations::{TensorFunction, _tanh};
use crate::longformer::embeddings::LongformerEmbeddings;
use crate::longformer::encoder::LongformerEncoder;
use crate::{Activation, Config, RustBertError};
use serde::{Deserialize, Serialize};
use std::borrow::Borrow;
use std::collections::HashMap;
use tch::nn::Module;
use tch::{nn, Kind, Tensor};

/// # Longformer Pretrained model weight files
pub struct LongformerModelResources;

/// # Longformer Pretrained model config files
pub struct LongformerConfigResources;

/// # Longformer Pretrained model vocab files
pub struct LongformerVocabResources;

/// # Longformer Pretrained model merges files
pub struct LongformerMergesResources;

impl LongformerModelResources {
    /// Shared under Apache 2.0 license by the AllenAI team at https://github.com/allenai/longformer. Modified with conversion to C-array format.
    pub const LONGFORMER_BASE_4096: (&'static str, &'static str) = (
        "longformer-base-4096/model",
        "https://huggingface.co/allenai/longformer-base-4096/resolve/main/rust_model.ot",
    );
}

impl LongformerConfigResources {
    /// Shared under Apache 2.0 license by the AllenAI team at https://github.com/allenai/longformer. Modified with conversion to C-array format.
    pub const LONGFORMER_BASE_4096: (&'static str, &'static str) = (
        "longformer-base-4096/config",
        "https://huggingface.co/allenai/longformer-base-4096/resolve/main/config.json",
    );
}

impl LongformerVocabResources {
    /// Shared under Apache 2.0 license by the AllenAI team at https://github.com/allenai/longformer. Modified with conversion to C-array format.
    pub const LONGFORMER_BASE_4096: (&'static str, &'static str) = (
        "longformer-base-4096/vocab",
        "https://huggingface.co/allenai/longformer-base-4096/resolve/main/vocab.json",
    );
}

impl LongformerMergesResources {
    /// Shared under Apache 2.0 license by the AllenAI team at https://github.com/allenai/longformer. Modified with conversion to C-array format.
    pub const LONGFORMER_BASE_4096: (&'static str, &'static str) = (
        "longformer-base-4096/merges",
        "https://huggingface.co/allenai/longformer-base-4096/resolve/main/merges.txt",
    );
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
#[serde(rename_all = "camelCase")]
pub enum PositionEmbeddingType {
    Absolute,
    RelativeKey,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
/// # Longformer model configuration
/// Defines the Longformer model architecture (e.g. number of layers, hidden layer size, label mapping...)
pub struct LongformerConfig {
    pub hidden_act: Activation,
    pub attention_window: Vec<i64>,
    pub attention_probs_dropout_prob: f64,
    pub hidden_dropout_prob: f64,
    pub hidden_size: i64,
    pub initializer_range: f32,
    pub intermediate_size: i64,
    pub max_position_embeddings: i64,
    pub num_attention_heads: i64,
    pub num_hidden_layers: i64,
    pub type_vocab_size: i64,
    pub vocab_size: i64,
    pub sep_token_id: i64,
    pub pad_token_id: Option<i64>,
    pub layer_norm_eps: Option<f64>,
    pub output_attentions: Option<bool>,
    pub output_hidden_states: Option<bool>,
    pub position_embedding_type: Option<PositionEmbeddingType>,
    pub is_decoder: Option<bool>,
    pub id2label: Option<HashMap<i64, String>>,
    pub label2id: Option<HashMap<String, i64>>,
}

impl Config<LongformerConfig> for LongformerConfig {}

fn get_question_end_index(input_ids: &Tensor, sep_token_id: i64) -> Tensor {
    input_ids
        .eq(sep_token_id)
        .nonzero()
        .view([input_ids.size()[0], 3, 2])
        .select(2, 1)
        .select(1, 0)
}

fn compute_global_attention_mask(
    input_ids: &Tensor,
    sep_token_id: i64,
    before_sep_token: bool,
) -> Tensor {
    let question_end_index = get_question_end_index(input_ids, sep_token_id).unsqueeze(1);
    let attention_mask = Tensor::arange(input_ids.size()[1], (Kind::Int8, input_ids.device()));

    if before_sep_token {
        attention_mask.expand_as(input_ids).lt1(&question_end_index)
    } else {
        attention_mask
            .expand_as(input_ids)
            .gt1(&(question_end_index + 1))
            * attention_mask
                .expand_as(input_ids)
                .lt(*input_ids.size().last().unwrap())
    }
}

#[derive(Debug)]
pub struct LongformerPooler {
    dense: nn::Linear,
    activation: TensorFunction,
}

impl LongformerPooler {
    pub fn new<'p, P>(p: P, config: &LongformerConfig) -> LongformerPooler
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let dense = nn::linear(
            p / "dense",
            config.hidden_size,
            config.hidden_size,
            Default::default(),
        );

        let activation = TensorFunction::new(Box::new(_tanh));

        LongformerPooler { dense, activation }
    }
}

impl Module for LongformerPooler {
    fn forward(&self, hidden_states: &Tensor) -> Tensor {
        self.activation.get_fn()(&hidden_states.select(1, 0).apply(&self.dense))
    }
}

#[derive(Debug)]
pub struct LongformerLMHead {
    dense: nn::Linear,
    layer_norm: nn::LayerNorm,
    decoder: nn::Linear,
}

impl LongformerLMHead {
    pub fn new<'p, P>(p: P, config: &LongformerConfig) -> LongformerLMHead
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let dense = nn::linear(
            p / "dense",
            config.hidden_size,
            config.hidden_size,
            Default::default(),
        );

        let layer_norm_config = nn::LayerNormConfig {
            eps: config.layer_norm_eps.unwrap_or(1e-12),
            ..Default::default()
        };

        let layer_norm = nn::layer_norm(
            p / "layer_norm",
            vec![config.hidden_size],
            layer_norm_config,
        );

        let decoder = nn::linear(
            p / "dense",
            config.hidden_size,
            config.vocab_size,
            Default::default(),
        );

        LongformerLMHead {
            dense,
            layer_norm,
            decoder,
        }
    }
}

impl Module for LongformerLMHead {
    fn forward(&self, hidden_states: &Tensor) -> Tensor {
        hidden_states
            .apply(&self.dense)
            .gelu()
            .apply(&self.layer_norm)
            .apply(&self.decoder)
    }
}

pub struct LongformerModel {
    embeddings: LongformerEmbeddings,
    encoder: LongformerEncoder,
    pooler: Option<LongformerPooler>,
    attention_window: Vec<i64>,
    max_attention_window: i64,
}

impl LongformerModel {
    pub fn new<'p, P>(p: P, config: &LongformerConfig, add_pooling_layer: bool) -> LongformerModel
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let embeddings = LongformerEmbeddings::new(p / "embeddings", config);
        let encoder = LongformerEncoder::new(p / "encoder", config);
        let pooler = if add_pooling_layer {
            Some(LongformerPooler::new(p / "pooler", config))
        } else {
            None
        };

        let attention_window = config.attention_window.clone();
        let max_attention_window = *attention_window.iter().max().unwrap();

        LongformerModel {
            embeddings,
            encoder,
            pooler,
            attention_window,
            max_attention_window,
        }
    }

    fn pad_with_nonzero_value(
        &self,
        tensor: &Tensor,
        padding: &[i64],
        padding_value: i64,
    ) -> Tensor {
        (tensor - padding_value).constant_pad_nd(padding) + padding_value
    }

    fn pad_with_boolean(&self, tensor: &Tensor, padding: &[i64], padding_value: bool) -> Tensor {
        if !padding_value {
            tensor.constant_pad_nd(padding)
        } else {
            ((tensor.logical_not()).constant_pad_nd(padding)).logical_not()
        }
    }

    fn pad_to_window_size(
        &self,
        input_ids: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        pad_token_id: i64,
        padding_length: i64,
        train: bool,
    ) -> Result<
        (
            Option<Tensor>,
            Option<Tensor>,
            Option<Tensor>,
            Option<Tensor>,
            Option<Tensor>,
        ),
        RustBertError,
    > {
        let input_shape = if let Some(input_ids) = input_ids {
            if input_embeds.is_none() {
                input_ids.size()
            } else {
                return Err(RustBertError::ValueError(
                    "Only one of input ids or input embeddings may be set".into(),
                ));
            }
        } else if let Some(input_embeds) = input_embeds {
            input_embeds.size()[..2].to_vec()
        } else {
            return Err(RustBertError::ValueError(
                "At least one of input ids or input embeddings must be set".into(),
            ));
        };

        let (batch_size, sequence_length) = (input_shape[0], input_shape[1]);
        // ToDo: move check to the method calling pad_to_window_size
        // let padding_length = (self.max_attention_window
        //     - sequence_length % self.max_attention_window)
        //     % self.max_attention_window;
        //
        // let (input_ids, position_ids,inputs_embeds,attention_mask,token_type_ids) = if padding_length > 0 {
        let input_ids = input_ids
            .map(|value| self.pad_with_nonzero_value(value, &[0, padding_length], pad_token_id));
        let position_ids = position_ids
            .map(|value| self.pad_with_nonzero_value(value, &[0, padding_length], pad_token_id));
        let inputs_embeds = input_embeds.map(|value| {
            let input_ids_padding = Tensor::full(
                &[batch_size, padding_length],
                pad_token_id,
                (Kind::Int64, value.device()),
            );
            let input_embeds_padding = self
                .embeddings
                .forward_t(Some(&input_ids_padding), None, None, None, train)
                .unwrap();

            Tensor::cat(&[value, &input_embeds_padding], -2)
        });

        let attention_mask =
            attention_mask.map(|value| self.pad_with_boolean(&value, &[0, padding_length], false));
        let token_type_ids =
            token_type_ids.map(|value| value.constant_pad_nd(&[0, padding_length]));
        Ok((
            input_ids,
            position_ids,
            inputs_embeds,
            attention_mask,
            token_type_ids,
        ))
    }
}
