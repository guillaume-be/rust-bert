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

use std::borrow::Borrow;
use std::collections::HashMap;

use rust_tokenizers::tokenizer::ReformerTokenizer;
use rust_tokenizers::vocab::ReformerVocab;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tch::{nn, Device, Kind, Tensor};

use crate::common::activations::Activation;
use crate::common::dropout::Dropout;
use crate::common::embeddings::get_shape_and_device_from_ids_embeddings_pair;
use crate::pipelines::common::{ModelType, TokenizerOption};
use crate::pipelines::generation_utils::private_generation_utils::{
    PreparedInput, PrivateLanguageGenerator,
};
use crate::pipelines::generation_utils::{
    Cache, GenerateConfig, LMHeadModel, LMModelOutput, LanguageGenerator,
};
use crate::reformer::attention::{AttentionType, LayerState};
use crate::reformer::attention_utils::{get_least_common_mult_chunk_len, get_min_chunk_len};
use crate::reformer::embeddings::ReformerEmbeddings;
use crate::reformer::encoder::{ReformerEncoder, ReformerModelOutput};
use crate::{Config, RustBertError};

/// # Reformer Pretrained model weight files
pub struct ReformerModelResources;

/// # Reformer Pretrained model config files
pub struct ReformerConfigResources;

/// # Reformer Pretrained model vocab files
pub struct ReformerVocabResources;

impl ReformerModelResources {
    /// Shared under Apache 2.0 license by the Trax Authors at <https://github.com/google/trax/tree/master/trax/models/reformer>. Modified with conversion to C-array format.
    pub const CRIME_AND_PUNISHMENT: (&'static str, &'static str) = (
        "reformer-crime-punishment/model",
        "https://huggingface.co/google/reformer-crime-and-punishment/resolve/main/rust_model.ot",
    );
}

impl ReformerConfigResources {
    /// Shared under Apache 2.0 license by the Trax Authors at <https://github.com/google/trax/tree/master/trax/models/reformer>. Modified with conversion to C-array format.
    pub const CRIME_AND_PUNISHMENT: (&'static str, &'static str) = (
        "reformer-crime-punishment/config",
        "https://huggingface.co/google/reformer-crime-and-punishment/resolve/main/config.json",
    );
}

impl ReformerVocabResources {
    /// Shared under Apache 2.0 license by the Trax Authors at <https://github.com/google/trax/tree/master/trax/models/reformer>. Modified with conversion to C-array format.
    pub const CRIME_AND_PUNISHMENT: (&'static str, &'static str) = (
        "reformer-crime-punishment/spiece",
        "https://huggingface.co/google/reformer-crime-and-punishment/resolve/main/spiece.model",
    );
}

#[derive(Debug, Serialize, Deserialize, Clone)]
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

impl Config for ReformerConfig {}

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

/// # Reformer Base model
/// Base architecture for the Reformer model. Usually complemented with a task-specific head, such as a language model head.
/// It is made of the following blocks:
/// - `embeddings`: `ReformerEmbeddings` Reformer embeddings, combining word and position embeddings
/// - `encoder`: `ReformerEncoder` (transformer) made of a vector of Reformer layer with local or LSH attention.
/// caching is implemented for the decoder to avoid recalculating static states (encoder key/values and previously calculated decoder key/values)
/// - `least_common_mult_chunk_length`: least common chunk length for all attention layers
/// - `min_chunk_length`: minimum chunk length for all attention layers
/// - `pad_token_id`: padding token id used to pad to chunk length multiple if input is long enough to be chunked.
pub struct ReformerModel {
    embeddings: ReformerEmbeddings,
    encoder: ReformerEncoder,
    least_common_mult_chunk_length: i64,
    min_chunk_length: i64,
    pad_token_id: i64,
}

impl ReformerModel {
    /// Build a new `ReformerModel`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the BART model
    /// * `config` - `ReformerConfig` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::reformer::{ReformerConfig, ReformerModel};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = ReformerConfig::from_file(config_path);
    /// let reformer_model: ReformerModel =
    ///     ReformerModel::new(&p.root() / "reformer", &config).unwrap();
    /// ```
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

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). Must be provided when no pre-computed embeddings are given.
    /// * `position_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If not provided will be calculated on the fly starting from position 0.
    /// * `input_embeds` - Optional input tensor of shape (*batch size*, *sequence_length*, *embeddings_dim*). Must be provided when no input ids are given.
    /// * `attention_mask` - Optional attention mask of shape (*batch size*, *sequence_length*). Positions with a mask with value 0 will be masked.
    /// * `num_hashes` - Optional specification of the number of hashes to use. If not provided will use the value provided in the model configuration.
    /// * `old_layer_states` - Optional cached input (`Option<Vec<Option<LayerState>>>`) containing previous values for the cached states and buckets.
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `ReformerModelOutput` containing:
    ///   - `hidden_states` - `Tensor` of shape (*batch size*, *sequence_length*, *hidden_size*) representing the activations of the last hidden state
    ///   - `all_hidden_states` - `Option<Vec<Tensor>>` of length *n_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `all_attentions` - `Option<Vec<Tensor>>` of length *n_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `cache` - `Option<Vec<Option<LayerState>>>` of length *n_layer*  containing values for the states and buckets for future use.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use tch::{nn, Device, Tensor, no_grad, Kind};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::{Int64, Double};
    /// use rust_bert::reformer::{ReformerConfig, ReformerModel};
    /// # let config_path = Path::new("path/to/config.json");
    /// # let vocab_path = Path::new("path/to/spiece.model");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = ReformerConfig::from_file(config_path);
    /// # let reformer_model: ReformerModel = ReformerModel::new(&vs.root(), &config).unwrap();
    /// let (batch_size, sequence_length) = (64, 128);
    /// let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    /// let input_positions = Tensor::arange(sequence_length, (Kind::Int64, device))
    ///     .unsqueeze(0)
    ///     .expand(&[batch_size, sequence_length], true);
    /// let attention_mask = Tensor::ones(&[batch_size, sequence_length], (Int64, device));
    ///
    /// let model_output = no_grad(|| {
    ///     reformer_model.forward_t(
    ///         Some(&input_tensor),
    ///         Some(&input_positions),
    ///         None,
    ///         Some(&attention_mask),
    ///         Some(4),
    ///         None,
    ///         false,
    ///     )
    /// });
    /// ```
    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        num_hashes: Option<i64>,
        old_layer_states: Option<Vec<Option<LayerState>>>,
        train: bool,
    ) -> Result<ReformerModelOutput, RustBertError> {
        let (input_shape, device) =
            get_shape_and_device_from_ids_embeddings_pair(input_ids, input_embeds)?;

        let original_sequence_length = *input_shape.last().unwrap();

        let must_pad_to_match_chunk_length =
            (input_shape.last().unwrap() % self.least_common_mult_chunk_length != 0)
                & (*input_shape.last().unwrap() as i64 > self.min_chunk_length)
                & old_layer_states.is_none();

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
                padded_input.input_embeds.as_ref(),
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
        input_embeds: Option<&Tensor>,
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
                let position_ids_padding = Tensor::arange_start(
                    *input_shape.last().unwrap(),
                    self.least_common_mult_chunk_length,
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
            let input_embeds = Tensor::cat(&[input_embeds, &input_embeds_padding], -1);
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

/// # Reformer Model for text generation
/// Reformer model with a vocabulary decoding head
/// It is made of the following blocks:
/// - `reformer`: `ReformerModel` Base Reformer model
/// - `lm_head`: `ReformerLMHead` projecting hidden states to the vocabulary dimension
pub struct ReformerModelWithLMHead {
    reformer: ReformerModel,
    lm_head: ReformerLMHead,
}

impl ReformerModelWithLMHead {
    /// Build a new `ReformerModelWithLMHead`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the BART model
    /// * `config` - `ReformerConfig` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::reformer::{ReformerConfig, ReformerModelWithLMHead};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = ReformerConfig::from_file(config_path);
    /// let reformer_model: ReformerModelWithLMHead =
    ///     ReformerModelWithLMHead::new(&p.root(), &config).unwrap();
    /// ```
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

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). Must be provided when no pre-computed embeddings are given.
    /// * `position_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If not provided will be calculated on the fly starting from position 0.
    /// * `input_embeds` - Optional input tensor of shape (*batch size*, *sequence_length*, *embeddings_dim*). Must be provided when no input ids are given.
    /// * `attention_mask` - Optional attention mask of shape (*batch size*, *sequence_length*). Positions with a mask with value 0 will be masked.
    /// * `num_hashes` - Optional specification of the number of hashes to use. If not provided will use the value provided in the model configuration.
    /// * `old_layer_states` - Optional cached input (`Option<Vec<Option<LayerState>>>`) containing previous values for the cached states and buckets.
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `ReformerLMModelOutput` containing:
    ///   - `logits` - `Tensor` of shape (*batch size*, *sequence_length*, *vocab_size*) representing the logits for each vocabulary item
    ///   - `all_hidden_states` - `Option<Vec<Tensor>>` of length *n_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `all_attentions` - `Option<Vec<Tensor>>` of length *n_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `cache` - `Option<Vec<Option<LayerState>>>` of length *n_layer*  containing values for the states and buckets for future use.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use tch::{nn, Device, Tensor, no_grad, Kind};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::{Int64, Double};
    /// use rust_bert::reformer::{ReformerConfig, ReformerModelWithLMHead};
    /// # let config_path = Path::new("path/to/config.json");
    /// # let vocab_path = Path::new("path/to/spiece.model");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = ReformerConfig::from_file(config_path);
    /// # let reformer_model: ReformerModelWithLMHead = ReformerModelWithLMHead::new(&vs.root(), &config).unwrap();
    /// let (batch_size, sequence_length) = (64, 128);
    /// let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    /// let input_positions = Tensor::arange(sequence_length, (Kind::Int64, device)).unsqueeze(0).expand(&[batch_size, sequence_length], true);
    /// let attention_mask = Tensor::ones(&[batch_size, sequence_length], (Int64, device));
    ///
    /// let model_output = no_grad(|| {
    ///     reformer_model.forward_t(
    ///         Some(&input_tensor),
    ///         Some(&input_positions),
    ///         None,
    ///         Some(&attention_mask),
    ///         Some(4),
    ///         None,
    ///         false,
    ///     )
    /// });
    /// ```
    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
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

impl LMHeadModel for ReformerModelWithLMHead {
    fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        cache: Cache,
        attention_mask: Option<&Tensor>,
        _token_type_ids: Option<&Tensor>,
        _position_ids: Option<&Tensor>,
        _input_embeds: Option<&Tensor>,
        _encoder_outputs: Option<&Tensor>,
        _decoder_input_ids: Option<&Tensor>,
        train: bool,
    ) -> Result<LMModelOutput, RustBertError> {
        let output = match cache {
            Cache::ReformerCache(cached_layer_states) => self.forward_t(
                input_ids,
                None,
                None,
                attention_mask,
                None,
                cached_layer_states,
                train,
            ),
            Cache::None => self.forward_t(input_ids, None, None, attention_mask, None, None, train),
            _ => {
                return Err(RustBertError::ValueError(
                    "Cache not compatible with Reformer Model".into(),
                ));
            }
        }?;

        Ok(LMModelOutput {
            lm_logits: output.logits,
            cache: Cache::ReformerCache(output.next_cache),
        })
    }
}

pub struct ReformerClassificationHead {
    dense: nn::Linear,
    dropout: Dropout,
    out_proj: nn::Linear,
}

impl ReformerClassificationHead {
    pub fn new<'p, P>(
        p: P,
        config: &ReformerConfig,
    ) -> Result<ReformerClassificationHead, RustBertError>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let dense = nn::linear(
            p / "dense",
            2 * config.hidden_size,
            config.hidden_size,
            Default::default(),
        );
        let num_labels = match &config.id2label {
            Some(value) => value.len() as i64,
            None => {
                return Err(RustBertError::InvalidConfigurationError(
                    "an id to label mapping must be provided for classification tasks".to_string(),
                ));
            }
        };
        let out_proj = nn::linear(
            p / "out_proj",
            config.hidden_size,
            num_labels,
            Default::default(),
        );
        let dropout = Dropout::new(config.hidden_dropout_prob);

        Ok(ReformerClassificationHead {
            dense,
            dropout,
            out_proj,
        })
    }

    pub fn forward_t(&self, hidden_states: &Tensor, train: bool) -> Tensor {
        hidden_states
            .select(1, 0)
            .apply_t(&self.dropout, train)
            .apply(&self.dense)
            .tanh()
            .apply_t(&self.dropout, train)
            .apply(&self.out_proj)
    }
}

/// # Reformer Model for sequence classification
/// Reformer model with a classification head
/// It is made of the following blocks:
/// - `reformer`: `ReformerModel` Base Reformer model
/// - `classifier`: `ReformerClassificationHead` projecting hidden states to the target labels
pub struct ReformerForSequenceClassification {
    reformer: ReformerModel,
    classifier: ReformerClassificationHead,
}

impl ReformerForSequenceClassification {
    /// Build a new `ReformerForSequenceClassification`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the BART model
    /// * `config` - `ReformerConfig` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::reformer::{ReformerConfig, ReformerForSequenceClassification};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = ReformerConfig::from_file(config_path);
    /// let reformer_model: ReformerForSequenceClassification =
    ///     ReformerForSequenceClassification::new(&p.root(), &config).unwrap();
    /// ```
    pub fn new<'p, P>(
        p: P,
        config: &ReformerConfig,
    ) -> Result<ReformerForSequenceClassification, RustBertError>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let reformer = ReformerModel::new(p / "reformer", config)?;
        let classifier = ReformerClassificationHead::new(p / "classifier", config)?;

        Ok(ReformerForSequenceClassification {
            reformer,
            classifier,
        })
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). Must be provided when no pre-computed embeddings are given.
    /// * `position_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If not provided will be calculated on the fly starting from position 0.
    /// * `input_embeds` - Optional input tensor of shape (*batch size*, *sequence_length*, *embeddings_dim*). Must be provided when no input ids are given.
    /// * `attention_mask` - Optional attention mask of shape (*batch size*, *sequence_length*). Positions with a mask with value 0 will be masked.
    /// * `num_hashes` - Optional specification of the number of hashes to use. If not provided will use the value provided in the model configuration.
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `ReformerClassificationOutput` containing:
    ///   - `logits` - `Tensor` of shape (*batch size*, *sequence_length*, *num_classes*) representing the logits for each target class
    ///   - `all_hidden_states` - `Option<Vec<Tensor>>` of length *n_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `all_attentions` - `Option<Vec<Tensor>>` of length *n_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use tch::{nn, Device, Tensor, no_grad, Kind};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::{Int64, Double};
    /// use rust_bert::reformer::{ReformerConfig, ReformerForSequenceClassification};
    /// # let config_path = Path::new("path/to/config.json");
    /// # let vocab_path = Path::new("path/to/spiece.model");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = ReformerConfig::from_file(config_path);
    /// # let reformer_model: ReformerForSequenceClassification = ReformerForSequenceClassification::new(&vs.root(), &config).unwrap();
    /// let (batch_size, sequence_length) = (64, 128);
    /// let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    /// let input_positions = Tensor::arange(sequence_length, (Kind::Int64, device)).unsqueeze(0).expand(&[batch_size, sequence_length], true);
    /// let attention_mask = Tensor::ones(&[batch_size, sequence_length], (Int64, device));
    ///
    /// let model_output = no_grad(|| {
    ///     reformer_model.forward_t(
    ///         Some(&input_tensor),
    ///         Some(&input_positions),
    ///         None,
    ///         Some(&attention_mask),
    ///         Some(4),
    ///         false,
    ///     )
    /// });
    /// ```
    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        num_hashes: Option<i64>,
        train: bool,
    ) -> Result<ReformerClassificationOutput, RustBertError> {
        let reformer_output = self.reformer.forward_t(
            input_ids,
            position_ids,
            input_embeds,
            attention_mask,
            num_hashes,
            None,
            train,
        )?;

        let logits = self
            .classifier
            .forward_t(&reformer_output.hidden_states, train);

        Ok(ReformerClassificationOutput {
            logits,
            all_hidden_states: reformer_output.all_hidden_states,
            all_attentions: reformer_output.all_attentions,
        })
    }
}

/// # Reformer Model for question answering
/// Extractive question-answering model based on a Reformer language model. Identifies the segment of a context that answers a provided question.
/// Please note that a significant amount of pre- and post-processing is required to perform end-to-end question answering.
/// See the question answering pipeline (also provided in this crate) for more details.
/// It is made of the following blocks:
/// - `reformer`: `ReformerModel` Base Reformer model
/// - `qa_outputs`: Linear layer for question answering, mapping to start and end logits for the answer.
pub struct ReformerForQuestionAnswering {
    reformer: ReformerModel,
    qa_outputs: nn::Linear,
}

impl ReformerForQuestionAnswering {
    /// Build a new `ReformerForQuestionAnswering`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the BART model
    /// * `config` - `ReformerConfig` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::reformer::{ReformerConfig, ReformerForQuestionAnswering};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = ReformerConfig::from_file(config_path);
    /// let reformer_model: ReformerForQuestionAnswering =
    ///     ReformerForQuestionAnswering::new(&p.root(), &config).unwrap();
    /// ```
    pub fn new<'p, P>(
        p: P,
        config: &ReformerConfig,
    ) -> Result<ReformerForQuestionAnswering, RustBertError>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let reformer = ReformerModel::new(p / "reformer", config)?;
        let qa_outputs = nn::linear(
            p / "qa_outputs",
            2 * config.hidden_size,
            2,
            Default::default(),
        );

        Ok(ReformerForQuestionAnswering {
            reformer,
            qa_outputs,
        })
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). Must be provided when no pre-computed embeddings are given.
    /// * `position_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If not provided will be calculated on the fly starting from position 0.
    /// * `input_embeds` - Optional input tensor of shape (*batch size*, *sequence_length*, *embeddings_dim*). Must be provided when no input ids are given.
    /// * `attention_mask` - Optional attention mask of shape (*batch size*, *sequence_length*). Positions with a mask with value 0 will be masked.
    /// * `num_hashes` - Optional specification of the number of hashes to use. If not provided will use the value provided in the model configuration.
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `ReformerClassificationOutput` containing:
    ///   - `start_logits` -  `Tensor` of shape (*batch size*, *sequence_length*) containing the logits for start of the answer
    ///   - `end_logits` - `Tensor` of shape (*batch size*, *sequence_length*) containing the logits for end of the answer
    ///   - `all_hidden_states` - `Option<Vec<Tensor>>` of length *n_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `all_attentions` - `Option<Vec<Tensor>>` of length *n_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use tch::{nn, Device, Tensor, no_grad, Kind};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::{Int64, Double};
    /// use rust_bert::reformer::{ReformerConfig, ReformerForQuestionAnswering};
    /// # let config_path = Path::new("path/to/config.json");
    /// # let vocab_path = Path::new("path/to/spiece.model");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = ReformerConfig::from_file(config_path);
    /// # let reformer_model: ReformerForQuestionAnswering = ReformerForQuestionAnswering::new(&vs.root(), &config).unwrap();
    /// let (batch_size, sequence_length) = (64, 128);
    /// let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    /// let input_positions = Tensor::arange(sequence_length, (Kind::Int64, device)).unsqueeze(0).expand(&[batch_size, sequence_length], true);
    /// let attention_mask = Tensor::ones(&[batch_size, sequence_length], (Int64, device));
    ///
    /// let model_output = no_grad(|| {
    ///     reformer_model.forward_t(
    ///         Some(&input_tensor),
    ///         Some(&input_positions),
    ///         None,
    ///         Some(&attention_mask),
    ///         Some(4),
    ///         false,
    ///     )
    /// });
    /// ```
    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        num_hashes: Option<i64>,
        train: bool,
    ) -> Result<ReformerQuestionAnsweringModelOutput, RustBertError> {
        let reformer_output = self.reformer.forward_t(
            input_ids,
            position_ids,
            input_embeds,
            attention_mask,
            num_hashes,
            None,
            train,
        )?;

        let logits = reformer_output
            .hidden_states
            .apply(&self.qa_outputs)
            .split(1, -1);
        let (start_logits, end_logits) = (&logits[0], &logits[1]);
        let start_logits = start_logits.squeeze_dim(-1);
        let end_logits = end_logits.squeeze_dim(-1);

        Ok(ReformerQuestionAnsweringModelOutput {
            start_logits,
            end_logits,
            all_hidden_states: reformer_output.all_hidden_states,
            all_attentions: reformer_output.all_attentions,
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

///Container holding a Reformer model with classification head
pub struct ReformerClassificationOutput {
    /// logits
    pub logits: Tensor,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all intermediate layers
    pub all_attentions: Option<Vec<Tensor>>,
}

///Container holding a Reformer model with question answering head
pub struct ReformerQuestionAnsweringModelOutput {
    /// start logits
    pub start_logits: Tensor,
    /// end logits
    pub end_logits: Tensor,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all intermediate layers
    pub all_attentions: Option<Vec<Tensor>>,
}

pub struct ReformerGenerator {
    model: ReformerModelWithLMHead,
    tokenizer: TokenizerOption,
    var_store: nn::VarStore,
    generate_config: GenerateConfig,
    bos_token_id: Option<i64>,
    eos_token_ids: Option<Vec<i64>>,
    pad_token_id: Option<i64>,
    is_encoder_decoder: bool,
    vocab_size: i64,
    decoder_start_id: Option<i64>,
    max_position_embeddings: i64,
}

impl ReformerGenerator {
    pub fn new(generate_config: GenerateConfig) -> Result<ReformerGenerator, RustBertError> {
        let vocab_path = generate_config.vocab_resource.get_local_path()?;

        let tokenizer = TokenizerOption::from_file(
            ModelType::Reformer,
            vocab_path.to_str().unwrap(),
            None,
            false,
            None,
            None,
        )?;

        Self::new_with_tokenizer(generate_config, tokenizer)
    }

    pub fn new_with_tokenizer(
        generate_config: GenerateConfig,
        tokenizer: TokenizerOption,
    ) -> Result<ReformerGenerator, RustBertError> {
        let config_path = generate_config.config_resource.get_local_path()?;
        let weights_path = generate_config.model_resource.get_local_path()?;
        let device = generate_config.device;

        generate_config.validate();
        let mut var_store = nn::VarStore::new(device);
        let config = ReformerConfig::from_file(config_path);
        let model = ReformerModelWithLMHead::new(&var_store.root(), &config)?;
        var_store.load(weights_path)?;

        let bos_token_id = tokenizer.get_bos_id();
        let eos_token_ids = tokenizer.get_eos_id().map(|id| vec![id]);
        let pad_token_id = Some(config.pad_token_id);
        let vocab_size = config.vocab_size;
        let is_encoder_decoder = false;
        let decoder_start_id = None;
        let max_position_embeddings = config.max_position_embeddings;

        Ok(ReformerGenerator {
            model,
            tokenizer,
            var_store,
            generate_config,
            bos_token_id,
            eos_token_ids,
            pad_token_id,
            is_encoder_decoder,
            vocab_size,
            decoder_start_id,
            max_position_embeddings,
        })
    }
}

impl PrivateLanguageGenerator<ReformerModelWithLMHead, ReformerVocab, ReformerTokenizer>
    for ReformerGenerator
{
    fn get_model(&self) -> &ReformerModelWithLMHead {
        &self.model
    }
    fn _get_tokenizer(&self) -> &TokenizerOption {
        &self.tokenizer
    }
    fn get_var_store(&self) -> &nn::VarStore {
        &self.var_store
    }
    fn get_var_store_mut(&mut self) -> &mut nn::VarStore {
        &mut self.var_store
    }
    fn get_config(&self) -> &GenerateConfig {
        &self.generate_config
    }
    fn get_bos_id(&self) -> Option<i64> {
        self.bos_token_id
    }
    fn get_eos_ids(&self) -> Option<&Vec<i64>> {
        self.eos_token_ids.as_ref()
    }
    fn get_pad_id(&self) -> Option<i64> {
        self.pad_token_id
    }
    fn is_encoder_decoder(&self) -> bool {
        self.is_encoder_decoder
    }
    fn get_vocab_size(&self) -> i64 {
        self.vocab_size
    }
    fn get_decoder_start_id(&self) -> Option<i64> {
        self.decoder_start_id
    }
    fn get_max_positions_embeddings(&self) -> i64 {
        self.max_position_embeddings
    }

    fn prepare_inputs_for_generation<'a>(
        &self,
        input_ids: Tensor,
        _encoder_outputs: Option<&'a Tensor>,
        past: Cache,
        attention_mask: Tensor,
    ) -> PreparedInput<'a> {
        match past {
            Cache::ReformerCache(past) => PreparedInput {
                prepared_input: Some(input_ids.select(1, -1).unsqueeze(-1)),
                prepared_attention_mask: None,
                prepared_encoder_output: None,
                prepared_decoder_input: None,
                prepared_position_ids: None,
                prepared_past: Cache::ReformerCache(past),
            },
            Cache::None => PreparedInput {
                prepared_input: Some(input_ids),
                prepared_attention_mask: Some(attention_mask),
                prepared_encoder_output: None,
                prepared_decoder_input: None,
                prepared_position_ids: None,
                prepared_past: Cache::ReformerCache(None),
            },
            _ => panic!("Cache type incompatible with Reformer"),
        }
    }

    fn reorder_cache(
        &self,
        past: &mut Cache,
        encoder_outputs: Option<Tensor>,
        beam_indices: &Tensor,
    ) -> Option<Tensor> {
        match past {
            Cache::ReformerCache(old_cache_option) => match old_cache_option {
                Some(old_cache) => {
                    for layer_state in old_cache.iter_mut() {
                        if layer_state.is_some() {
                            layer_state.as_mut().unwrap().reorder_cache(beam_indices)
                        };
                    }
                }
                None => {}
            },
            Cache::None => {}
            _ => {
                panic!("Invalid cache for Reformer model");
            }
        };
        encoder_outputs
    }
}

impl LanguageGenerator<ReformerModelWithLMHead, ReformerVocab, ReformerTokenizer>
    for ReformerGenerator
{
}
