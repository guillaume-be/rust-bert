// Copyright 2020 The Trax Authors and The HuggingFace Inc. team.
// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
use crate::reformer::attention::{AttentionType, LayerState};
use crate::reformer::attention_utils::{get_least_common_mult_chunk_len, get_min_chunk_len};
use crate::reformer::embeddings::ReformerEmbeddings;
use crate::reformer::encoder::{ReformerEncoder, ReformerModelOutput};
use crate::{Config, RustBertError};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::borrow::Borrow;
use std::collections::HashMap;
use tch::{nn, Device, Kind, Tensor};

/// # Reformer Pretrained model weight files
pub struct ReformerModelResources;

/// # Reformer Pretrained model config files
pub struct ReformerConfigResources;

/// # Reformer Pretrained model vocab files
pub struct ReformerVocabResources;

impl ReformerModelResources {
    /// Shared under Apache 2.0 license by the Trax Authors at https://github.com/google/trax/tree/master/trax/models/reformer. Modified with conversion to C-array format.
    pub const CRIME_AND_PUNISHMENT: (&'static str, &'static str) = (
        "reformer-crime-punishment/model",
        "https://cdn.huggingface.co/google/reformer-crime-and-punishment/rust_model.ot",
    );
}

impl ReformerConfigResources {
    /// Shared under Apache 2.0 license by the Trax Authors at https://github.com/google/trax/tree/master/trax/models/reformer. Modified with conversion to C-array format.
    pub const CRIME_AND_PUNISHMENT: (&'static str, &'static str) = (
        "reformer-crime-punishment/config",
        "https://cdn.huggingface.co/google/reformer-crime-and-punishment/config.json",
    );
}

impl ReformerVocabResources {
    /// Shared under Apache 2.0 license by the Trax Authors at https://github.com/google/trax/tree/master/trax/models/reformer. Modified with conversion to C-array format.
    pub const CRIME_AND_PUNISHMENT: (&'static str, &'static str) = (
        "reformer-crime-punishment/spiece",
        "https://cdn.huggingface.co/google/reformer-crime-and-punishment/spiece.model",
    );
}

#[derive(Debug, Serialize, Deserialize)]
/// # Reformer model configuration
/// Defines the Reformer model architecture (e.g. number of layers, hidden layer size, label mapping...)
pub struct ReformerConfig {
    pub attention_head_size: i64,
    pub attention_probs_dropout_prob: f64,
    pub attn_layers: Vec<AttentionType>,
    pub axial_norm_std: f64,
    pub axial_pos_embds: bool,
    pub axial_pos_embds_dim: Vec<i64>,
    pub axial_pos_shape: Vec<i64>,
    pub chunk_size_lm_head: i64,
    pub chunk_size_feed_forward: Option<i64>,
    pub eos_token_id: i64,
    pub pad_token_id: i64,
    pub feed_forward_size: i64,
    pub hash_seed: Option<i64>,
    pub hidden_act: Activation,
    pub hidden_dropout_prob: f64,
    pub hidden_size: i64,
    pub initializer_range: Option<f64>,
    pub intermediate_size: i64,
    pub is_decoder: bool,
    pub layer_norm_eps: Option<f64>,
    pub max_position_embeddings: i64,
    pub vocab_size: i64,
    pub num_attention_heads: i64,
    pub num_buckets: Value,
    pub local_attn_chunk_length: Option<i64>,
    pub local_num_chunks_after: Option<i64>,
    pub local_num_chunks_before: Option<i64>,
    pub local_attention_probs_dropout_prob: Option<f64>,
    pub lsh_attn_chunk_length: Option<i64>,
    pub lsh_num_chunks_after: Option<i64>,
    pub lsh_num_chunks_before: Option<i64>,
    pub lsh_attention_probs_dropout_prob: Option<f64>,
    pub num_hashes: i64,
    pub num_hidden_layers: i64,
    pub use_cache: Option<bool>,
    pub id2label: Option<HashMap<i64, String>>,
    pub label2id: Option<HashMap<String, i64>>,
    pub output_attentions: Option<bool>,
    pub output_hidden_states: Option<bool>,
}

impl Config<ReformerConfig> for ReformerConfig {}

pub struct ReformerLMHead {
    decoder: nn::Linear,
    chunk_size_lm_head: i64,
}

impl ReformerLMHead {
    pub fn new<'p, P>(p: P, config: &ReformerConfig) -> ReformerLMHead
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let decoder = nn::linear(
            p / "decoder",
            2 * config.hidden_size,
            config.vocab_size,
            Default::default(),
        );

        ReformerLMHead {
            decoder,
            chunk_size_lm_head: config.chunk_size_lm_head,
        }
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Tensor {
        if self.chunk_size_lm_head > 0 {
            let num_chunks = hidden_states.size()[1] / self.chunk_size_lm_head;
            let input_tensors_chunk = hidden_states.chunk(num_chunks, 1);
            let output_chunks = input_tensors_chunk
                .iter()
                .map(|v| v.apply(&self.decoder))
                .collect::<Vec<Tensor>>();
            Tensor::cat(output_chunks.as_slice(), 1)
        } else {
            hidden_states.apply(&self.decoder)
        }
    }
}

pub struct PaddedReformerInput {
    pub input_ids: Option<Tensor>,
    pub input_embeds: Option<Tensor>,
    pub attention_mask: Option<Tensor>,
    pub position_ids: Option<Tensor>,
    pub new_input_shape: Vec<i64>,
}

pub struct ReformerModel {
    embeddings: ReformerEmbeddings,
    encoder: ReformerEncoder,
    least_common_mult_chunk_length: i64,
    min_chunk_length: i64,
    pad_token_id: i64,
}

impl ReformerModel {
    pub fn new<'p, P>(p: P, config: &ReformerConfig) -> Result<ReformerModel, RustBertError>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let embeddings = ReformerEmbeddings::new(p / "embeddings", config)?;
        let encoder = ReformerEncoder::new(p / "encoder", config)?;

        let least_common_mult_chunk_length = get_least_common_mult_chunk_len(
            config.attn_layers.as_slice(),
            config.lsh_attn_chunk_length,
            config.local_attn_chunk_length,
        );
        let min_chunk_length = get_min_chunk_len(
            config.attn_layers.as_slice(),
            config.lsh_attn_chunk_length,
            config.local_attn_chunk_length,
        );

        let pad_token_id = config.pad_token_id;

        Ok(ReformerModel {
            embeddings,
            encoder,
            least_common_mult_chunk_length,
            min_chunk_length,
            pad_token_id,
        })
    }

    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        input_embeds: Option<Tensor>,
        attention_mask: Option<&Tensor>,
        num_hashes: Option<i64>,
        old_layer_states: Option<Vec<Option<LayerState>>>,
        train: bool,
    ) -> Result<ReformerModelOutput, RustBertError> {
        let (input_shape, device) = if let Some(input_ids) = input_ids {
            (input_ids.size(), input_ids.device())
        } else if let Some(input_embeds) = &input_embeds {
            (input_embeds.size(), input_embeds.device())
        } else {
            return Err(RustBertError::ValueError(
                "At least one of input ids or input embeddings must be set".into(),
            ));
        };

        let original_sequence_length = *input_shape.last().unwrap();

        let must_pad_to_match_chunk_length =
            (input_shape.last().unwrap() & self.least_common_mult_chunk_length != 0)
                & (*input_shape.last().unwrap() as i64 > self.min_chunk_length)
                & old_layer_states.is_some();

        let start_idx_pos_encodings = if let Some(layer_states) = &old_layer_states {
            if let Some(layer_state) = &layer_states[0] {
                layer_state.prev_states.size()[1]
            } else {
                0
            }
        } else {
            0
        };

        let encoder_outputs = if must_pad_to_match_chunk_length {
            let padding_length = self.least_common_mult_chunk_length
                - input_shape.last().unwrap() % self.least_common_mult_chunk_length;
            let padded_input = self.pad_to_mult_of_chunk_length(
                input_ids,
                input_embeds,
                attention_mask,
                position_ids,
                input_shape.as_slice(),
                padding_length,
                device,
            )?;
            let embedding_output = self.embeddings.forward_t(
                padded_input.input_ids.as_ref(),
                padded_input.position_ids.as_ref(),
                padded_input.input_embeds,
                start_idx_pos_encodings,
                train,
            )?;

            let mut encoder_output = self.encoder.forward_t(
                &embedding_output,
                padded_input.attention_mask.as_ref(),
                num_hashes,
                old_layer_states,
                original_sequence_length,
                train,
            )?;
            encoder_output.hidden_states =
                encoder_output
                    .hidden_states
                    .slice(1, 0, original_sequence_length, 1);
            encoder_output
        } else {
            let embedding_output = self.embeddings.forward_t(
                input_ids,
                position_ids,
                input_embeds,
                start_idx_pos_encodings,
                train,
            )?;

            self.encoder.forward_t(
                &embedding_output,
                attention_mask,
                num_hashes,
                old_layer_states,
                original_sequence_length,
                train,
            )?
        };
        Ok(encoder_outputs)
    }

    fn pad_to_mult_of_chunk_length(
        &self,
        input_ids: Option<&Tensor>,
        input_embeds: Option<Tensor>,
        attention_mask: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        input_shape: &[i64],
        padding_length: i64,
        device: Device,
    ) -> Result<PaddedReformerInput, RustBertError> {
        let input_ids_padding = Tensor::full(
            &[input_shape[0], padding_length],
            self.pad_token_id,
            (Kind::Int64, device),
        );

        let attention_mask = Some(if let Some(attention_mask) = attention_mask {
            let attention_mask_padding = Tensor::zeros(
                &[input_shape[0], padding_length],
                (attention_mask.kind(), device),
            );
            Tensor::cat(&[attention_mask, &attention_mask_padding], -1)
        } else {
            Tensor::cat(
                &[
                    Tensor::ones(input_shape, (Kind::Int8, device)),
                    Tensor::zeros(&[input_shape[0], padding_length], (Kind::Int8, device)),
                ],
                -1,
            )
        });

        let mut new_input_shape = vec![];

        let (input_ids, position_ids) = if let Some(input_ids) = input_ids {
            let input_ids = Tensor::cat(&[input_ids, &input_ids_padding], -1);
            new_input_shape = input_ids.size();
            let position_ids = if let Some(position_ids) = position_ids {
                let position_ids_padding = Tensor::arange2(
                    *input_shape.last().unwrap(),
                    self.least_common_mult_chunk_length,
                    1,
                    (Kind::Int64, device),
                )
                .unsqueeze(0)
                .expand(&[input_shape[0], padding_length], true);
                Some(Tensor::cat(&[position_ids, &position_ids_padding], -1))
            } else {
                None
            };
            (Some(input_ids), position_ids)
        } else {
            (None, None)
        };

        let input_embeds = if let Some(input_embeds) = input_embeds {
            let input_embeds_padding = self.embeddings.forward_t(
                Some(&input_ids_padding),
                None,
                None,
                *input_shape.last().unwrap(),
                false,
            )?;
            let input_embeds = Tensor::cat(&[input_embeds, input_embeds_padding], -1);
            new_input_shape = input_embeds.size();
            Some(input_embeds)
        } else {
            None
        };
        Ok(PaddedReformerInput {
            input_ids,
            input_embeds,
            attention_mask,
            position_ids,
            new_input_shape,
        })
    }
}

pub struct ReformerModelWithLMHead {
    reformer: ReformerModel,
    lm_head: ReformerLMHead,
}

impl ReformerModelWithLMHead {
    pub fn new<'p, P>(
        p: P,
        config: &ReformerConfig,
    ) -> Result<ReformerModelWithLMHead, RustBertError>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        if !config.is_decoder {
            return Err(RustBertError::InvalidConfigurationError("Reformer must be a decoder to be used as a language model. `is_decoder` has been set to `false`.".to_string()));
        }

        if let Some(lsh_num_chunks_after) = config.lsh_num_chunks_after {
            if config.attn_layers.contains(&AttentionType::lsh) & (lsh_num_chunks_after != 0) {
                return Err(RustBertError::InvalidConfigurationError(
                    format!("For text generation using LSH attention ensure `config.lsh_num_chunks_after` is set to 0 (currently {})", lsh_num_chunks_after),
                ));
            }
        }

        if let Some(local_num_chunks_after) = config.local_num_chunks_after {
            if config.attn_layers.contains(&AttentionType::local) & (local_num_chunks_after != 0) {
                return Err(RustBertError::InvalidConfigurationError(
                    format!("For text generation using local attention ensure `config.local_num_chunks_after` is set to 0 (currently {})", local_num_chunks_after),
                ));
            }
        }

        let reformer = ReformerModel::new(p / "reformer", config)?;
        let lm_head = ReformerLMHead::new(p / "lm_head", config);

        Ok(ReformerModelWithLMHead { reformer, lm_head })
    }

    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        input_embeds: Option<Tensor>,
        attention_mask: Option<&Tensor>,
        num_hashes: Option<i64>,
        old_layer_states: Option<Vec<Option<LayerState>>>,
        train: bool,
    ) -> Result<ReformerLMModelOutput, RustBertError> {
        let reformer_output = self.reformer.forward_t(
            input_ids,
            position_ids,
            input_embeds,
            attention_mask,
            num_hashes,
            old_layer_states,
            train,
        )?;

        let logits = self.lm_head.forward(&reformer_output.hidden_states);

        Ok(ReformerLMModelOutput {
            logits,
            all_hidden_states: reformer_output.all_hidden_states,
            all_attentions: reformer_output.all_attentions,
            next_cache: reformer_output.next_cache,
        })
    }
}

///Container holding a Reformer model with LM head output
pub struct ReformerLMModelOutput {
    /// logits
    pub logits: Tensor,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all intermediate layers
    pub all_attentions: Option<Vec<Tensor>>,
    /// Cached outputs of the model (attention layers keys and values) if the model is used for generation
    pub next_cache: Option<Vec<Option<LayerState>>>,
}
