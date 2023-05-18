// Copyright 2020 The Microsoft Authors and The HuggingFace Inc. team.
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

use rust_tokenizers::tokenizer::TruncationStrategy;
use serde::{Deserialize, Serialize};
use tch::{nn, Kind, Tensor};

use crate::pipelines::common::{ModelType, TokenizerOption};
use crate::pipelines::generation_utils::private_generation_utils::{
    PreparedInput, PrivateLanguageGenerator,
};
use crate::pipelines::generation_utils::{Cache, GenerateConfig, LMModelOutput, LanguageGenerator};
use crate::prophetnet::attention::LayerState;
use crate::prophetnet::decoder::ProphetNetDecoder;
use crate::prophetnet::encoder::ProphetNetEncoder;
use crate::{Activation, Config, RustBertError};

/// # ProphetNet Pretrained model weight files
pub struct ProphetNetModelResources;

/// # ProphetNet Pretrained model config files
pub struct ProphetNetConfigResources;

/// # ProphetNet Pretrained model vocab files
pub struct ProphetNetVocabResources;

impl ProphetNetModelResources {
    /// Shared under MIT license by the Microsoft team at <https://github.com/microsoft/ProphetNet>. Modified with conversion to C-array format.
    pub const PROPHETNET_LARGE_UNCASED: (&'static str, &'static str) = (
        "prophetnet-large-uncased/model",
        "https://huggingface.co/microsoft/prophetnet-large-uncased/resolve/main/rust_model.ot",
    );
    /// Shared under MIT license by the Microsoft team at <https://github.com/microsoft/ProphetNet>. Modified with conversion to C-array format.
    pub const PROPHETNET_LARGE_CNN_DM: (&'static str, &'static str) = (
        "prophetnet-large-uncased-cnndm/model",
        "https://huggingface.co/microsoft/prophetnet-large-uncased-cnndm/resolve/main/rust_model.ot",
    );
}

impl ProphetNetConfigResources {
    /// Shared under MIT license by the Microsoft team at <https://github.com/microsoft/ProphetNet>. Modified with conversion to C-array format.
    pub const PROPHETNET_LARGE_UNCASED: (&'static str, &'static str) = (
        "prophetnet-large-uncased/config",
        "https://huggingface.co/microsoft/prophetnet-large-uncased/resolve/main/config.json",
    );
    /// Shared under MIT license by the Microsoft team at <https://github.com/microsoft/ProphetNet>. Modified with conversion to C-array format.
    pub const PROPHETNET_LARGE_CNN_DM: (&'static str, &'static str) = (
        "prophetnet-large-uncased-cnndm/config",
        "https://huggingface.co/microsoft/prophetnet-large-uncased-cnndm/resolve/main/config.json",
    );
}

impl ProphetNetVocabResources {
    /// Shared under MIT license by the Microsoft team at <https://github.com/microsoft/ProphetNet>. Modified with conversion to C-array format.
    pub const PROPHETNET_LARGE_UNCASED: (&'static str, &'static str) = (
        "prophetnet-large-uncased/vocab",
        "https://huggingface.co/microsoft/prophetnet-large-uncased/resolve/main/prophetnet.tokenizer",
    );
    /// Shared under MIT license by the Microsoft team at <https://github.com/microsoft/ProphetNet>. Modified with conversion to C-array format.
    pub const PROPHETNET_LARGE_CNN_DM: (&'static str, &'static str) = (
        "prophetnet-large-uncased-cnndm/vocab",
        "https://huggingface.co/microsoft/prophetnet-large-uncased-cnndm/resolve/main/prophetnet.tokenizer",
    );
}

#[derive(Debug, Serialize, Deserialize, Clone)]
/// # ProphetNet model configuration
/// Defines the ProphetNet model architecture (e.g. number of layers, hidden layer size, label mapping...)
pub struct ProphetNetConfig {
    pub activation_function: Activation,
    pub activation_dropout: f64,
    pub attention_dropout: f64,
    pub decoder_ffn_dim: i64,
    pub decoder_start_token_id: i64,
    pub disable_ngram_loss: bool,
    pub dropout: f64,
    pub encoder_ffn_dim: i64,
    pub eps: f64,
    pub hidden_size: i64,
    pub init_std: f64,
    pub is_encoder_decoder: bool,
    pub max_position_embeddings: i64,
    pub bos_token_id: i64,
    pub eos_token_id: i64,
    pub ngram: i64,
    pub id2label: Option<HashMap<i64, String>>,
    pub label2id: Option<HashMap<String, i64>>,
    pub num_buckets: i64,
    pub num_decoder_attention_heads: i64,
    pub num_decoder_layers: i64,
    pub num_encoder_attention_heads: i64,
    pub num_encoder_layers: i64,
    pub output_past: Option<bool>,
    pub pad_token_id: i64,
    pub relative_max_distance: i64,
    pub vocab_size: i64,
    pub output_attentions: Option<bool>,
    pub output_hidden_states: Option<bool>,
    pub add_cross_attention: Option<bool>,
}

impl Config for ProphetNetConfig {}

impl Default for ProphetNetConfig {
    fn default() -> Self {
        ProphetNetConfig {
            activation_function: Activation::gelu,
            activation_dropout: 0.1,
            attention_dropout: 0.1,
            decoder_ffn_dim: 4096,
            decoder_start_token_id: 0,
            disable_ngram_loss: false,
            dropout: 0.1,
            encoder_ffn_dim: 4096,
            eps: 0.0,
            hidden_size: 1024,
            init_std: 0.02,
            is_encoder_decoder: false,
            max_position_embeddings: 512,
            bos_token_id: 1,
            eos_token_id: 2,
            ngram: 2,
            id2label: None,
            label2id: None,
            num_buckets: 32,
            num_decoder_attention_heads: 16,
            num_decoder_layers: 12,
            num_encoder_attention_heads: 16,
            num_encoder_layers: 12,
            output_past: None,
            pad_token_id: 0,
            relative_max_distance: 128,
            vocab_size: 30522,
            output_attentions: None,
            output_hidden_states: None,
            add_cross_attention: Some(true),
        }
    }
}

/// # ProphetNet Base model
/// Base architecture for ProphetNet models. Task-specific models will be built from this common base model
/// It is made of the following blocks:
/// - `word_embeddings`: Word embeddings
/// - `encoder`: ProphetNetEncoder
/// - `decoder`: ProphetNetDecoder
pub struct ProphetNetModel {
    pub(crate) word_embeddings: nn::Embedding,
    pub(crate) encoder: ProphetNetEncoder,
    decoder: ProphetNetDecoder,
}

impl ProphetNetModel {
    /// Build a new `ProphetNetModel`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the ProphetNet model
    /// * `config` - `ProphetNetConfig` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::prophetnet::{ProphetNetConfig, ProphetNetModel};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = ProphetNetConfig::from_file(config_path);
    /// let prophetnet_model = ProphetNetModel::new(&p.root(), &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &ProphetNetConfig) -> Result<ProphetNetModel, RustBertError>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let word_embeddings_config = nn::EmbeddingConfig {
            padding_idx: config.pad_token_id,
            ..Default::default()
        };
        let word_embeddings = nn::embedding(
            p / "word_embeddings",
            config.vocab_size,
            config.hidden_size,
            word_embeddings_config,
        );

        let encoder = ProphetNetEncoder::new(p / "encoder", config)?;
        let decoder = ProphetNetDecoder::new(p / "decoder", config)?;

        Ok(ProphetNetModel {
            word_embeddings,
            encoder,
            decoder,
        })
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). This or `input_embeds` must be provided.
    /// * `attention_mask` - Optional attention mask of shape (*batch size*, *sequence_length*) for the encoder positions. Positions with a mask with value 0 will be masked.
    /// * `input_embeds` - Optional input tensor of shape (*batch size*, *sequence_length*, *embeddings dimension*). This or `input_ids` must be provided.
    /// * `decoder_input_ids` - Optional input tensor of shape (*batch size*, *target_sequence_length*). Must be provided when running in generation mode (e.g. initialized with a BOS token)
    /// * `decoder_attention_mask` - Optional attention mask of shape (*batch size*, *target_sequence_length*) for the decoder positions. Positions with a mask with value 0 will be masked.
    /// * `encoder_hidden_states` - Optional tensor of shape (*batch size*, *source_sequence_length*, *encoder_hidden_dim*) corresponding to pre-calculated encoder hidden states (useful for conditional generation)
    /// These correspond to the encoder last hidden state and optional hidden states/attention weights for encoder layers. When provided, the encoder hidden state will not be recalculated. Useful for generation tasks.
    /// * `old_layer_states` - Optional Vector `Option<Vec<Option<&LayerState>, Option<&LayerState>>>` of length *n_layer* containing tuples with the past keys and values for both the self attention and the encoder cross attention of each layer of the decoder.
    /// * `decoder_input_embeds` - Optional input tensor of shape (*batch size*, *target_sequence_length*, *embeddings dimension*). This or `decoder_input_ids` must be provided.
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `ProphetNetOutput` containing:
    ///   - `last_hidden_states` - `Tensor` of shape (*batch size*, *target_sequence_length*, *hidden_size*) representing the activations of the last hidden state for the decoder
    ///   - `ngram_hidden_states` - `Tensor` of shape (*ngram*, *batch size*, *target_sequence_length*, *hidden_size*) representing the activations of the last hidden state for the decoder ngram stream
    ///   - `next_decoder_cache` - `Option<Vec<Option<LayerState>>>` of length *n_layer* containing the past content for the the attention layers with shape (*past_sequence_length*, *batch size*, *hidden_size*)
    ///   - `all_decoder_hidden_states` - `Option<Vec<Tensor>>` of length *n_layer* with shape (*batch size*, *target_sequence_length*, *hidden_size*)
    ///   - `all_ngram_decoder_hidden_states` - `Option<Vec<Tensor>>` of length *n_layer* with shape (*ngram*, *batch size*, *target_sequence_length*, *hidden_size*)
    ///   - `all_attentions` - `Option<Vec<Tensor>>` of length *n_layer* with shape (*batch size*, *target_sequence_length*, *hidden_size*)
    ///   - `all_ngram_attentions` - `Option<Vec<Tensor>>` of length *n_layer* with shape (*ngram*, *batch size*, *target_sequence_length*, *hidden_size*)
    ///   - `all_cross_attentions` - `Option<Vec<Tensor>>` of length *n_layer* with shape (*batch size*, *target_sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use tch::{nn, Device, Tensor, no_grad, Kind};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::{Int64, Double};
    /// use rust_bert::prophetnet::{ProphetNetModel, ProphetNetConfig};
    /// # let config_path = Path::new("path/to/config.json");
    /// # let vocab_path = Path::new("path/to/vocab.txt");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = ProphetNetConfig::from_file(config_path);
    /// # let prophetnet_model: ProphetNetModel = ProphetNetModel::new(&vs.root(), &config).unwrap();
    /// let (batch_size, sequence_length, target_sequence_length) = (64, 128, 32);
    /// let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    /// let attention_mask = Tensor::ones(&[batch_size, sequence_length], (Int64, device));
    /// let target_tensor = Tensor::ones(&[batch_size, sequence_length], (Int64, device));
    /// let decoder_input_ids = Tensor::ones(&[batch_size, target_sequence_length], (Kind::Float, device));
    ///
    /// let model_output = no_grad(|| {
    ///     prophetnet_model.forward_t(
    ///         Some(&input_tensor),
    ///         Some(&attention_mask),
    ///         None,
    ///         Some(&decoder_input_ids),
    ///         None,
    ///         None,
    ///         None,
    ///         None,
    ///         false
    ///     )
    /// });
    /// ```
    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        decoder_input_ids: Option<&Tensor>,
        decoder_attention_mask: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
        old_layer_states: Option<Vec<(Option<LayerState>, Option<LayerState>)>>,
        decoder_input_embeds: Option<&Tensor>,
        train: bool,
    ) -> Result<ProphetNetOutput, RustBertError> {
        let calc_encoder_hidden_states = if encoder_hidden_states.is_none() {
            Some(
                self.encoder
                    .forward_t(
                        input_ids,
                        attention_mask,
                        input_embeds,
                        Some(&self.word_embeddings),
                        train,
                    )?
                    .hidden_states,
            )
        } else {
            None
        };
        let encoder_hidden_states =
            encoder_hidden_states.unwrap_or_else(|| calc_encoder_hidden_states.as_ref().unwrap());

        let decoder_output = self.decoder.forward_t(
            decoder_input_ids,
            decoder_attention_mask,
            encoder_hidden_states.into(),
            decoder_attention_mask,
            old_layer_states,
            decoder_input_embeds,
            Some(&self.word_embeddings),
            train,
        )?;

        Ok(ProphetNetOutput {
            last_hidden_states: decoder_output.hidden_states,
            ngram_hidden_states: decoder_output.ngram_hidden_states,
            all_decoder_hidden_states: decoder_output.all_hidden_states,
            all_ngram_hidden_states: decoder_output.all_ngram_hidden_states,
            all_attentions: decoder_output.all_attentions,
            all_ngram_attentions: decoder_output.all_ngram_attentions,
            all_cross_attentions: decoder_output.all_cross_attentions,
            next_decoder_cache: decoder_output.next_decoder_cache,
        })
    }
}

/// # ProphetNet Model for conditional generation
/// ProphetNet model with a vocabulary decoding head
/// It is made of the following blocks:
/// - `base_model`: `ProphetNetModel` Base ProphetNet model
/// - `lm_head`: Linear layer without bias to project the hidden states to the vocabulary
pub struct ProphetNetForConditionalGeneration {
    base_model: ProphetNetModel,
    lm_head: nn::Linear,
    decoder_start_token_id: i64,
    pad_token_id: i64,
    ngram: i64,
}

impl ProphetNetForConditionalGeneration {
    /// Build a new `ProphetNetForConditionalGeneration`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the ProphetNet model
    /// * `config` - `ProphetNetConfig` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::prophetnet::{ProphetNetConfig, ProphetNetForConditionalGeneration};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = ProphetNetConfig::from_file(config_path);
    /// let prophetnet_model = ProphetNetForConditionalGeneration::new(&p.root(), &config);
    /// ```
    pub fn new<'p, P>(
        p: P,
        config: &ProphetNetConfig,
    ) -> Result<ProphetNetForConditionalGeneration, RustBertError>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let base_model = ProphetNetModel::new(p / "prophetnet", config)?;
        let linear_config = nn::LinearConfig {
            bias: false,
            ..Default::default()
        };
        let lm_head = nn::linear(
            p / "lm_head",
            config.hidden_size,
            config.vocab_size,
            linear_config,
        );

        let decoder_start_token_id = config.decoder_start_token_id;
        let pad_token_id = config.pad_token_id;
        let ngram = config.ngram;

        Ok(ProphetNetForConditionalGeneration {
            base_model,
            lm_head,
            decoder_start_token_id,
            pad_token_id,
            ngram,
        })
    }

    fn shift_right(&self, input_ids: &Tensor) -> Tensor {
        let shifted_input_ids = Tensor::zeros(
            input_ids.size().as_slice(),
            (Kind::Int64, input_ids.device()),
        );

        shifted_input_ids
            .slice(-1, 1, *shifted_input_ids.size().last().unwrap(), 1)
            .copy_(&input_ids.slice(-1, 0, *input_ids.size().last().unwrap() - 1, 1));

        let _ = shifted_input_ids
            .get(-1)
            .get(0)
            .fill_(self.decoder_start_token_id);

        let _ = shifted_input_ids.masked_fill(&shifted_input_ids.eq(-100), self.pad_token_id);

        shifted_input_ids
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). This or `input_embeds` must be provided.
    /// * `attention_mask` - Optional attention mask of shape (*batch size*, *sequence_length*) for the encoder positions. Positions with a mask with value 0 will be masked.
    /// * `input_embeds` - Optional input tensor of shape (*batch size*, *sequence_length*, *embeddings dimension*). This or `input_ids` must be provided.
    /// * `decoder_input_ids` - Optional input tensor of shape (*batch size*, *target_sequence_length*). Must be provided when running in generation mode (e.g. initialized with a BOS token)
    /// * `decoder_attention_mask` - Optional attention mask of shape (*batch size*, *target_sequence_length*) for the decoder positions. Positions with a mask with value 0 will be masked.
    /// * `encoder_hidden_states` - Optional tensor of shape (*batch size*, *source_sequence_length*, *encoder_hidden_dim*) corresponding to pre-calculated encoder hidden states (useful for conditional generation)
    /// These correspond to the encoder last hidden state and optional hidden states/attention weights for encoder layers. When provided, the encoder hidden state will not be recalculated. Useful for generation tasks.
    /// * `old_layer_states` - Optional Vector `Option<Vec<Option<&LayerState>, Option<&LayerState>>>` of length *n_layer* containing tuples with the past keys and values for both the self attention and the encoder cross attention of each layer of the decoder.
    /// * `decoder_input_embeds` - Optional input tensor of shape (*batch size*, *target_sequence_length*, *embeddings dimension*). This or `decoder_input_ids` must be provided.
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `ProphetNetGenerationOutput` containing:
    ///   - `logits` - `Tensor` of shape (*batch size*, *target_sequence_length*, *vocabulary_size*) representing the activations of the last hidden state for the decoder
    ///   - `ngram_logits` - `Tensor` of shape (*ngram*, *batch size*, *target_sequence_length*, *vocabulary_size*) representing the activations of the last hidden state for the decoder ngram stream
    ///   - `next_decoder_cache` - `Option<Vec<Option<LayerState>>>` of length *n_layer* containing the past content for the the attention layers with shape (*past_sequence_length*, *batch size*, *hidden_size*)
    ///   - `all_decoder_hidden_states` - `Option<Vec<Tensor>>` of length *n_layer* with shape (*batch size*, *target_sequence_length*, *hidden_size*)
    ///   - `all_ngram_decoder_hidden_states` - `Option<Vec<Tensor>>` of length *n_layer* with shape (*ngram*, *batch size*, *target_sequence_length*, *hidden_size*)
    ///   - `all_attentions` - `Option<Vec<Tensor>>` of length *n_layer* with shape (*batch size*, *target_sequence_length*, *hidden_size*)
    ///   - `all_ngram_attentions` - `Option<Vec<Tensor>>` of length *n_layer* with shape (*ngram*, *batch size*, *target_sequence_length*, *hidden_size*)
    ///   - `all_cross_attentions` - `Option<Vec<Tensor>>` of length *n_layer* with shape (*batch size*, *target_sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use tch::{nn, Device, Tensor, no_grad, Kind};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::{Int64, Double};
    /// use rust_bert::prophetnet::{ProphetNetModel, ProphetNetConfig, ProphetNetForConditionalGeneration};
    /// # let config_path = Path::new("path/to/config.json");
    /// # let vocab_path = Path::new("path/to/vocab.txt");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = ProphetNetConfig::from_file(config_path);
    /// # let prophetnet_model: ProphetNetForConditionalGeneration = ProphetNetForConditionalGeneration::new(&vs.root(), &config).unwrap();
    /// let (batch_size, sequence_length, target_sequence_length) = (64, 128, 32);
    /// let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    /// let attention_mask = Tensor::ones(&[batch_size, sequence_length], (Int64, device));
    /// let target_tensor = Tensor::ones(&[batch_size, sequence_length], (Int64, device));
    /// let decoder_input_ids = Tensor::ones(&[batch_size, target_sequence_length], (Kind::Float, device));
    ///
    /// let model_output = no_grad(|| {
    ///     prophetnet_model.forward_t(
    ///         Some(&input_tensor),
    ///         Some(&attention_mask),
    ///         None,
    ///         Some(&decoder_input_ids),
    ///         None,
    ///         None,
    ///         None,
    ///         None,
    ///         false
    ///     )
    /// });
    /// ```
    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        decoder_input_ids: Option<&Tensor>,
        decoder_attention_mask: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
        old_layer_states: Option<Vec<(Option<LayerState>, Option<LayerState>)>>,
        decoder_input_embeds: Option<&Tensor>,
        train: bool,
    ) -> Result<ProphetNetGenerationOutput, RustBertError> {
        let calc_decoder_input_ids = if decoder_input_ids.is_none() & decoder_input_embeds.is_none()
        {
            if let Some(input_ids) = input_ids {
                Some(self.shift_right(input_ids))
            } else {
                return Err(RustBertError::ValueError("input_ids must be provided if decoder_input_ids and decoder_input_embeds are not given.".into()));
            }
        } else {
            None
        };

        let decoder_input_ids = if decoder_input_ids.is_some() {
            decoder_input_ids
        } else {
            Some(calc_decoder_input_ids.as_ref().unwrap())
        };

        let base_model_output = self.base_model.forward_t(
            input_ids,
            attention_mask,
            input_embeds,
            decoder_input_ids,
            decoder_attention_mask,
            encoder_hidden_states,
            old_layer_states,
            decoder_input_embeds,
            train,
        )?;

        let (batch_size, sequence_length) = if let Some(decoder_input_ids) = decoder_input_ids {
            let shape = decoder_input_ids.size();
            (shape[0], shape[1])
        } else if let Some(decoder_input_embeds) = decoder_input_embeds {
            let shape = decoder_input_embeds.size();
            (shape[0], shape[1])
        } else {
            return Err(RustBertError::ValueError(
                "At least one of decoder_input_ids or decoder_input_embeds must be set".into(),
            ));
        };

        if base_model_output.ngram_hidden_states.is_none() {
            return Err(RustBertError::InvalidConfigurationError(
                "ngram must be set > 0 in the configuration for conditional generation".into(),
            ));
        }

        let predict_logits = base_model_output
            .ngram_hidden_states
            .as_ref()
            .unwrap()
            .view([batch_size, self.ngram, sequence_length, -1])
            .apply(&self.lm_head);

        let logits = predict_logits.select(1, 0).contiguous();

        let ngram_logits = if self.ngram > 1 {
            Some(predict_logits.slice(1, 1, predict_logits.size()[1], 1))
        } else {
            None
        };

        Ok(ProphetNetGenerationOutput {
            logits,
            ngram_logits,
            ngram_hidden_states: base_model_output.ngram_hidden_states,
            all_decoder_hidden_states: base_model_output.all_decoder_hidden_states,
            all_ngram_hidden_states: base_model_output.all_ngram_hidden_states,
            all_attentions: base_model_output.all_attentions,
            all_ngram_attentions: base_model_output.all_ngram_attentions,
            all_cross_attentions: base_model_output.all_cross_attentions,
            next_decoder_cache: base_model_output.next_decoder_cache,
        })
    }

    pub fn encode(
        &self,
        input_ids: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
    ) -> Result<Tensor, RustBertError> {
        Ok(self
            .base_model
            .encoder
            .forward_t(
                input_ids,
                attention_mask,
                input_embeds,
                Some(&self.base_model.word_embeddings),
                false,
            )?
            .hidden_states)
    }
}

/// # ProphetNet Model for causal generation
/// ProphetNet decoder with a vocabulary decoding head
/// It is made of the following blocks:
/// - `base_model`: `ProphetNetDecoder` Base ProphetNet decoder
/// - `word_embeddings`: word embeddings used by the decoder
/// - `lm_head`: Linear layer without bias to project the hidden states to the vocabulary
pub struct ProphetNetForCausalGeneration {
    decoder: ProphetNetDecoder,
    word_embeddings: nn::Embedding,
    lm_head: nn::Linear,
    ngram: i64,
}

impl ProphetNetForCausalGeneration {
    /// Build a new `ProphetNetForCausalGeneration`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the ProphetNet model
    /// * `config` - `ProphetNetConfig` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::prophetnet::{ProphetNetConfig, ProphetNetForCausalGeneration};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = ProphetNetConfig::from_file(config_path);
    /// let prophetnet_model = ProphetNetForCausalGeneration::new(&p.root(), &config);
    /// ```
    pub fn new<'p, P>(
        p: P,
        config: &ProphetNetConfig,
    ) -> Result<ProphetNetForCausalGeneration, RustBertError>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let mut updated_config = config.clone();
        updated_config.is_encoder_decoder = false;

        let p_prophetnet = p / "prophetnet";
        let decoder = ProphetNetDecoder::new(&p_prophetnet / "decoder", &updated_config)?;
        let linear_config = nn::LinearConfig {
            bias: false,
            ..Default::default()
        };

        let word_embeddings_config = nn::EmbeddingConfig {
            padding_idx: config.pad_token_id,
            ..Default::default()
        };
        let p_decoder = &p_prophetnet / "decoder";
        let word_embeddings = nn::embedding(
            &p_decoder / "word_embeddings",
            config.vocab_size,
            config.hidden_size,
            word_embeddings_config,
        );

        let lm_head = nn::linear(
            p / "lm_head",
            config.hidden_size,
            config.vocab_size,
            linear_config,
        );

        let ngram = config.ngram;

        Ok(ProphetNetForCausalGeneration {
            decoder,
            word_embeddings,
            lm_head,
            ngram,
        })
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). This or `input_embeds` must be provided.
    /// * `attention_mask` - Optional attention mask of shape (*batch size*, *sequence_length*) for the encoder positions. Positions with a mask with value 0 will be masked.
    /// * `input_embeds` - Optional input tensor of shape (*batch size*, *sequence_length*, *embeddings dimension*). This or `input_ids` must be provided.
    /// * `decoder_input_ids` - Optional input tensor of shape (*batch size*, *target_sequence_length*). Must be provided when running in generation mode (e.g. initialized with a BOS token)
    /// * `old_layer_states` - Optional Vector `Option<Vec<Option<&LayerState>, Option<&LayerState>>>` of length *n_layer* containing tuples with the past keys and values for both the self attention and the encoder cross attention of each layer of the decoder.
    /// * `decoder_input_embeds` - Optional input tensor of shape (*batch size*, *target_sequence_length*, *embeddings dimension*). This or `decoder_input_ids` must be provided.
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `ProphetNetGenerationOutput` containing:
    ///   - `logits` - `Tensor` of shape (*batch size*, *target_sequence_length*, *vocabulary_size*) representing the activations of the last hidden state for the decoder
    ///   - `ngram_logits` - `Tensor` of shape (*ngram*, *batch size*, *target_sequence_length*, *vocabulary_size*) representing the activations of the last hidden state for the decoder ngram stream
    ///   - `next_decoder_cache` - `Option<Vec<Option<LayerState>>>` of length *n_layer* containing the past content for the the attention layers with shape (*past_sequence_length*, *batch size*, *hidden_size*)
    ///   - `all_decoder_hidden_states` - `Option<Vec<Tensor>>` of length *n_layer* with shape (*batch size*, *target_sequence_length*, *hidden_size*)
    ///   - `all_ngram_decoder_hidden_states` - `Option<Vec<Tensor>>` of length *n_layer* with shape (*ngram*, *batch size*, *target_sequence_length*, *hidden_size*)
    ///   - `all_attentions` - `Option<Vec<Tensor>>` of length *n_layer* with shape (*batch size*, *target_sequence_length*, *hidden_size*)
    ///   - `all_ngram_attentions` - `Option<Vec<Tensor>>` of length *n_layer* with shape (*ngram*, *batch size*, *target_sequence_length*, *hidden_size*)
    ///   - `all_cross_attentions` - `Option<Vec<Tensor>>` of length *n_layer* with shape (*batch size*, *target_sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use tch::{nn, Device, Tensor, no_grad, Kind};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::{Int64, Double};
    /// use rust_bert::prophetnet::{ProphetNetModel, ProphetNetConfig, ProphetNetForCausalGeneration};
    /// # let config_path = Path::new("path/to/config.json");
    /// # let vocab_path = Path::new("path/to/vocab.txt");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = ProphetNetConfig::from_file(config_path);
    /// # let prophetnet_model: ProphetNetForCausalGeneration = ProphetNetForCausalGeneration::new(&vs.root(), &config).unwrap();
    /// let (batch_size, sequence_length, target_sequence_length) = (64, 128, 32);
    /// let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    /// let attention_mask = Tensor::ones(&[batch_size, sequence_length], (Int64, device));
    /// let target_tensor = Tensor::ones(&[batch_size, sequence_length], (Int64, device));
    /// let decoder_input_ids = Tensor::ones(&[batch_size, target_sequence_length], (Kind::Float, device));
    ///
    /// let model_output = no_grad(|| {
    ///     prophetnet_model.forward_t(
    ///         Some(&input_tensor),
    ///         Some(&attention_mask),
    ///         None,
    ///         Some(&decoder_input_ids),
    ///         None,
    ///         None,
    ///         false
    ///     )
    /// });
    /// ```
    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
        encoder_attention_mask: Option<&Tensor>,
        old_layer_states: Option<Vec<(Option<LayerState>, Option<LayerState>)>>,
        train: bool,
    ) -> Result<ProphetNetGenerationOutput, RustBertError> {
        let base_model_output = self.decoder.forward_t(
            input_ids,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            old_layer_states,
            input_embeds,
            Some(&self.word_embeddings),
            train,
        )?;

        let (batch_size, sequence_length) = if let Some(input_ids) = input_ids {
            let shape = input_ids.size();
            (shape[0], shape[1])
        } else if let Some(input_embeds) = input_embeds {
            let shape = input_embeds.size();
            (shape[0], shape[1])
        } else {
            return Err(RustBertError::ValueError(
                "At least one of input_ids or input_embeds must be set".into(),
            ));
        };

        if base_model_output.ngram_hidden_states.is_none() {
            return Err(RustBertError::InvalidConfigurationError(
                "ngram must be set > 0 in the configuration for conditional generation".into(),
            ));
        }

        let predict_logits = base_model_output
            .ngram_hidden_states
            .as_ref()
            .unwrap()
            .view([batch_size, self.ngram, sequence_length, -1])
            .apply(&self.lm_head);

        let logits = predict_logits.select(1, 0).contiguous();

        let ngram_logits = if self.ngram > 1 {
            Some(predict_logits.slice(1, 1, predict_logits.size()[1], 1))
        } else {
            None
        };

        Ok(ProphetNetGenerationOutput {
            logits,
            ngram_logits,
            ngram_hidden_states: base_model_output.ngram_hidden_states,
            all_decoder_hidden_states: base_model_output.all_hidden_states,
            all_ngram_hidden_states: base_model_output.all_ngram_hidden_states,
            all_attentions: base_model_output.all_attentions,
            all_ngram_attentions: base_model_output.all_ngram_attentions,
            all_cross_attentions: base_model_output.all_cross_attentions,
            next_decoder_cache: base_model_output.next_decoder_cache,
        })
    }
}

///Container holding a ProphetNet model output
pub struct ProphetNetOutput {
    /// last decoder layer hidden state
    pub last_hidden_states: Tensor,
    /// last decoder layer ngram hidden state
    pub ngram_hidden_states: Option<Tensor>,
    /// Hidden states for all intermediate layers
    pub all_decoder_hidden_states: Option<Vec<Tensor>>,
    /// Hidden states (ngram) for all intermediate layers
    pub all_ngram_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all intermediate layers
    pub all_attentions: Option<Vec<Tensor>>,
    /// Ngram attention weights for all intermediate layers
    pub all_ngram_attentions: Option<Vec<Tensor>>,
    /// Cross attention weights for all intermediate layers
    pub all_cross_attentions: Option<Vec<Tensor>>,
    /// Cached outputs of the model (attention layers keys and values) if the model is used for generation
    pub next_decoder_cache: Option<Vec<(Option<LayerState>, Option<LayerState>)>>,
}

///Container holding a ProphetNet model generation output
pub struct ProphetNetGenerationOutput {
    /// Prediction logits
    pub logits: Tensor,
    /// Ngram prediction logits
    pub ngram_logits: Option<Tensor>,
    /// last decoder layer ngram hidden state
    pub ngram_hidden_states: Option<Tensor>,
    /// Hidden states for all intermediate layers
    pub all_decoder_hidden_states: Option<Vec<Tensor>>,
    /// Hidden states (ngram) for all intermediate layers
    pub all_ngram_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all intermediate layers
    pub all_attentions: Option<Vec<Tensor>>,
    /// Ngram attention weights for all intermediate layers
    pub all_ngram_attentions: Option<Vec<Tensor>>,
    /// Cross attention weights for all intermediate layers
    pub all_cross_attentions: Option<Vec<Tensor>>,
    /// Cached outputs of the model (attention layers keys and values) if the model is used for generation
    pub next_decoder_cache: Option<Vec<(Option<LayerState>, Option<LayerState>)>>,
}

/// # Language generation model based on the ProphetNet architecture
pub struct ProphetNetConditionalGenerator {
    model: ProphetNetForConditionalGeneration,
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

impl ProphetNetConditionalGenerator {
    /// Build a new `ProphetNetConditionalGenerator`
    ///
    /// # Arguments
    ///
    /// * `vocab_path` - Path to the model vocabulary, expected to have a structure following the [Transformers library](https://github.com/huggingface/transformers) convention
    /// * `merges_path` - Path to the bpe merges, expected to have a structure following the [Transformers library](https://github.com/huggingface/transformers) convention
    /// * `config_path` - Path to the model configuration, expected to have a structure following the [Transformers library](https://github.com/huggingface/transformers) convention
    /// * `weights_path` - Path to the model weight files. These need to be converted form the `.bin` to `.ot` format using the utility script provided.
    /// * `device` - Device to run the model on, e.g. `Device::Cpu` or `Device::Cuda(0)`
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use std::path::PathBuf;
    /// # use tch::Device;
    /// # fn main() -> anyhow::Result<()> {
    /// use rust_bert::pipelines::generation_utils::GenerateConfig;
    /// use rust_bert::prophetnet::ProphetNetConditionalGenerator;
    /// # let mut home: PathBuf = dirs::home_dir().unwrap();
    /// # home.push("rustbert");
    /// # home.push("prophetnet");
    /// # let config_path = &home.as_path().join("config.json");
    /// # let vocab_path = &home.as_path().join("vocab.txt");
    /// # let merges_path = &home.as_path().join("merges.txt");
    /// # let weights_path = &home.as_path().join("model.ot");
    /// let device = Device::cuda_if_available();
    /// let generate_config = GenerateConfig {
    ///     max_length: Some(30),
    ///     do_sample: true,
    ///     num_beams: 5,
    ///     temperature: 1.1,
    ///     num_return_sequences: 3,
    ///     ..Default::default()
    /// };
    /// let prophetnet_generator = ProphetNetConditionalGenerator::new(generate_config)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(
        generate_config: GenerateConfig,
    ) -> Result<ProphetNetConditionalGenerator, RustBertError> {
        let vocab_path = generate_config.vocab_resource.get_local_path()?;

        let tokenizer = TokenizerOption::from_file(
            ModelType::ProphetNet,
            vocab_path.to_str().unwrap(),
            None,
            true,
            true,
            None,
        )?;

        Self::new_with_tokenizer(generate_config, tokenizer)
    }

    pub fn new_with_tokenizer(
        generate_config: GenerateConfig,
        tokenizer: TokenizerOption,
    ) -> Result<ProphetNetConditionalGenerator, RustBertError> {
        let config_path = generate_config.config_resource.get_local_path()?;
        let weights_path = generate_config.model_resource.get_local_path()?;
        let device = generate_config.device;

        generate_config.validate();
        let mut var_store = nn::VarStore::new(device);
        let config = ProphetNetConfig::from_file(config_path);
        let model = ProphetNetForConditionalGeneration::new(var_store.root(), &config)?;
        var_store.load(weights_path)?;

        let bos_token_id = Some(config.bos_token_id);
        let eos_token_ids = Some(vec![config.eos_token_id]);
        let pad_token_id = Some(config.pad_token_id);
        let vocab_size = config.vocab_size;
        let is_encoder_decoder = true;
        let decoder_start_id = Some(config.decoder_start_token_id);
        let max_position_embeddings = config.max_position_embeddings;

        Ok(ProphetNetConditionalGenerator {
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

impl PrivateLanguageGenerator for ProphetNetConditionalGenerator {
    fn _get_tokenizer(&self) -> &TokenizerOption {
        &self.tokenizer
    }
    fn _get_tokenizer_mut(&mut self) -> &mut TokenizerOption {
        &mut self.tokenizer
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

    fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        cache: Cache,
        attention_mask: Option<&Tensor>,
        _token_type_ids: Option<&Tensor>,
        _position_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        encoder_outputs: Option<&Tensor>,
        decoder_input_ids: Option<&Tensor>,
        train: bool,
    ) -> Result<LMModelOutput, RustBertError> {
        let base_model_output = match cache {
            Cache::ProphetNetCache(cached_layer_states) => self.model.forward_t(
                input_ids,
                attention_mask,
                input_embeds,
                decoder_input_ids,
                None,
                encoder_outputs,
                cached_layer_states,
                None,
                train,
            )?,
            Cache::None => self.model.forward_t(
                input_ids,
                attention_mask,
                input_embeds,
                decoder_input_ids,
                None,
                encoder_outputs,
                None,
                None,
                train,
            )?,
            _ => {
                return Err(RustBertError::ValueError(
                    "Cache not compatible with ProphetNet Model".into(),
                ));
            }
        };

        Ok(LMModelOutput {
            lm_logits: base_model_output.logits,
            cache: Cache::ProphetNetCache(base_model_output.next_decoder_cache),
        })
    }

    fn encode(&self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> Option<Tensor> {
        Some(
            self.model
                .encode(Some(input_ids), attention_mask, None)
                .unwrap(),
        )
    }

    fn prepare_inputs_for_generation<'a>(
        &self,
        input_ids: Tensor,
        encoder_outputs: Option<&'a Tensor>,
        past: Cache,
        attention_mask: Tensor,
    ) -> PreparedInput<'a> {
        match past {
            Cache::ProphetNetCache(past) => PreparedInput {
                prepared_input: None,
                prepared_attention_mask: Some(attention_mask),
                prepared_encoder_output: encoder_outputs,
                prepared_decoder_input: Some(input_ids.narrow(1, -1, 1)),
                prepared_position_ids: None,
                prepared_past: Cache::ProphetNetCache(past),
            },
            Cache::None => PreparedInput {
                prepared_input: None,
                prepared_attention_mask: Some(attention_mask),
                prepared_encoder_output: encoder_outputs,
                prepared_decoder_input: Some(input_ids),
                prepared_position_ids: None,
                prepared_past: Cache::ProphetNetCache(None),
            },
            _ => panic!("Cache type incompatible with ProphetNet"),
        }
    }

    fn encode_prompt_text<S>(
        &self,
        prompt_text: &[S],
        max_len: Option<i64>,
        pad_token_id: Option<i64>,
    ) -> Tensor
    where
        S: AsRef<str> + Sync,
    {
        let tokens = self._get_tokenizer().encode_list(
            prompt_text,
            max_len
                .map(|max_len| max_len as usize)
                .unwrap_or(usize::MAX),
            &TruncationStrategy::LongestFirst,
            0,
        );
        let token_ids = tokens
            .into_iter()
            .map(|tokenized_input| tokenized_input.token_ids)
            .collect::<Vec<Vec<i64>>>();

        let max_len = token_ids.iter().map(|input| input.len()).max().unwrap();

        let pad_token = match pad_token_id {
            Some(value) => value,
            None => self._get_tokenizer().get_unk_id(),
        };

        let token_ids = token_ids
            .into_iter()
            .map(|mut input| {
                let temp = vec![pad_token; max_len - input.len()];
                input.extend(temp);
                input
            })
            .map(|tokens| Tensor::from_slice(&tokens).to(self.get_var_store().device()))
            .collect::<Vec<Tensor>>();

        Tensor::stack(&token_ids, 0)
    }

    fn reorder_cache(
        &self,
        past: &mut Cache,
        encoder_outputs: Option<Tensor>,
        beam_indices: &Tensor,
    ) -> Option<Tensor> {
        let encoder_outputs = encoder_outputs.map(|value| value.index_select(0, beam_indices));
        match past {
            Cache::ProphetNetCache(old_cache_option) => match old_cache_option {
                Some(old_cache) => {
                    for (self_layer_state, encoder_layer_state) in old_cache.iter_mut() {
                        if self_layer_state.is_some() {
                            self_layer_state
                                .as_mut()
                                .unwrap()
                                .reorder_cache(beam_indices)
                        };
                        if encoder_layer_state.is_some() {
                            encoder_layer_state
                                .as_mut()
                                .unwrap()
                                .reorder_cache(beam_indices)
                        };
                    }
                }
                None => {}
            },
            Cache::None => {}
            _ => {
                panic!("Invalid cache for ProphetNet model");
            }
        };
        encoder_outputs
    }
}

impl LanguageGenerator for ProphetNetConditionalGenerator {}
