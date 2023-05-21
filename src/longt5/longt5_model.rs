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

use crate::longt5::encoder::LongT5Stack;
use crate::longt5::LayerState;
use crate::pipelines::common::{ModelType, TokenizerOption};
use crate::pipelines::generation_utils::private_generation_utils::{
    PreparedInput, PrivateLanguageGenerator,
};
use crate::pipelines::generation_utils::{Cache, GenerateConfig, LMModelOutput, LanguageGenerator};
use crate::t5::{FeedForwardProj, T5Config, T5ModelOutput, TaskSpecificParams};
use crate::{Config, RustBertError};
use rust_tokenizers::tokenizer::TruncationStrategy;
use serde::{Deserialize, Serialize};
use std::borrow::Borrow;
use tch::nn::{embedding, LinearConfig};
use tch::{nn, Tensor};

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
    pub bos_token_id: Option<i64>,
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
            bos_token_id: None,
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

impl From<&LongT5Config> for T5Config {
    fn from(val: &LongT5Config) -> T5Config {
        T5Config {
            dropout_rate: val.dropout_rate,
            d_model: val.d_model,
            d_ff: val.d_ff,
            d_kv: val.d_kv,
            decoder_start_token_id: val.decoder_start_token_id,
            bos_token_id: None,
            eos_token_id: val.eos_token_id,
            initializer_factor: val.initializer_factor,
            is_encoder_decoder: val.is_encoder_decoder,
            layer_norm_epsilon: val.layer_norm_epsilon,
            num_heads: val.num_heads,
            num_layers: val.num_layers,
            output_past: val.output_past,
            pad_token_id: val.pad_token_id,
            relative_attention_num_buckets: val.relative_attention_num_buckets,
            relative_attention_max_distance: val.relative_attention_max_distance,
            vocab_size: val.vocab_size,
            feed_forward_proj: val.feed_forward_proj,
            tie_word_embeddings: val.tie_word_embeddings,
            task_specific_params: val.task_specific_params.clone(),
            output_attentions: val.output_attentions,
            output_hidden_states: val.output_hidden_states,
        }
    }
}

/// # LongT5 Base model
/// Base architecture for LongT5 model. Usually complemented with a task-specific head, such as a language model head.
/// It is made of the following blocks:
/// - `encoder`: `T5Stack` (transformer) made of a vector of encoding layers
/// - `decoder`: `T5Stack` (transformer)  made of a vector of decoding layers with self attention and encoder cross-attention.
/// caching is implemented for the decoder to avoid recalculating static states (encoder key/values and previously calculated decoder key/values)
/// - `embeddings`: `nn::Embedding` Shared embeddings for the encoder and decoder.
pub struct LongT5Model {
    pub(crate) encoder: LongT5Stack,
    decoder: LongT5Stack,
    pub(crate) embeddings: nn::Embedding,
}

impl LongT5Model {
    /// Build a new `LongT5Model`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the LongT5 model
    /// * `config` - `LongT5Config` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::longt5::{LongT5Config, LongT5Model};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = LongT5Config::from_file(config_path);
    /// let long_t5: LongT5Model = LongT5Model::new(&p.root() / "longt5", &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &LongT5Config) -> LongT5Model
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

        let encoder = LongT5Stack::new(
            p / "encoder",
            config,
            false,
            false,
            config.output_attentions.unwrap_or(false),
            config.output_hidden_states.unwrap_or(false),
        );
        let decoder = LongT5Stack::new(
            p / "decoder",
            config,
            true,
            true,
            config.output_attentions.unwrap_or(false),
            config.output_hidden_states.unwrap_or(false),
        );

        LongT5Model {
            encoder,
            decoder,
            embeddings,
        }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *source_sequence_length*). This or `input_embeds` must be provided.
    /// * `attention_mask` - Optional attention mask of shape (*batch size*, *source_sequence_length*) for the encoder positions. Positions with a mask with value 0 will be masked.
    /// * `encoder_outputs` - Optional tuple made of a tensor of shape (*batch size*, *source_sequence_length*, *encoder_hidden_dim*) and optional vectors of tensors of length *num_encoder_layers* with shape (*batch size*, *source_sequence_length*, *hidden_size*).
    /// These correspond to the encoder last hidden state and optional hidden states/attention weights for encoder layers. When provided, the encoder hidden state will not be recalculated. Useful for generation tasks.
    /// * `decoder_input_ids` - Optional input tensor of shape (*batch size*, *target_sequence_length*). This or `decoder_input_embeds` must be provided.
    /// * `decoder_attention_mask` - Optional attention mask of shape (*batch size*, *target_sequence_length*) for the decoder positions. Positions with a mask with value 0 will be masked.
    /// * `input_embeds` - Optional input tensor of shape (*batch size*, *source_sequence_length*, *embeddings dimension*). This or `input_ids` must be provided.
    /// * `decoder_input_embeds` - Optional input tensor of shape (*batch size*, *target_sequence_length*, *embeddings dimension*). This or `decoder_input_ids` must be provided.
    /// * `old_layer_states` - Optional vector of length `num_layers` containing tuples of optional `LayerStates` containing the last calculated key and value pairs for the decoder. This avoids recomputing attention weights at past positions and speeds up decoding.
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `LongT5ModelOutput` containing:
    ///   - `decoder_output` - `Tensor` of shape (*batch size*, *target_sequence_length*, *hidden_size*) representing the activations of the last decoder hidden state
    ///   - `encoder_hidden_states` - `Tensor` of shape (*batch size*, *source_sequence_length*, *hidden_size*) representing the activations of the last encoder hidden state
    ///   - `cache` - `Option<Vec<(Option<Vec<LayerState, LayerState>>)>>` of length *n_layer* containing the encoder padding mask and past keys and values for both the self attention and the encoder cross attention of each layer of the decoder.
    ///   - `all_encoder_hidden_states` - `Option<Vec<Tensor>>` of length *num_encoder_layers* with shape (*batch size*, *source_sequence_length*, *hidden_size*)
    ///   - `all_encoder_attentions` - `Option<Vec<Tensor>>` of length *num_encoder_layers* with shape (*batch size*, *source_sequence_length*, *hidden_size*)
    ///   - `all_decoder_hidden_states` - `Option<Vec<Tensor>>` of length *num_decoder_layers* with shape (*batch size*, *target_sequence_length*, *hidden_size*)
    ///   - `all_decoder_attentions` - `Option<Vec<Tensor>>` of length *num_decoder_layers* with shape (*batch size*, *target_sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use tch::{nn, Device, Tensor, no_grad};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::{Int64, Double};
    /// use rust_bert::longt5::{LongT5Config, LongT5Model};
    /// # let config_path = Path::new("path/to/config.json");
    /// # let vocab_path = Path::new("path/to/vocab.txt");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = LongT5Config::from_file(config_path);
    /// # let longt5_model: LongT5Model = LongT5Model::new(&vs.root(), &config);
    /// let (batch_size, source_sequence_length, target_sequence_length) = (64, 128, 56);
    /// let input_tensor = Tensor::rand(&[batch_size, source_sequence_length], (Int64, device));
    /// let target_tensor = Tensor::rand(&[batch_size, target_sequence_length], (Int64, device));
    /// let encoder_attention_mask =
    ///     Tensor::ones(&[batch_size, source_sequence_length], (Int64, device));
    /// let decoder_attention_mask =
    ///     Tensor::ones(&[batch_size, source_sequence_length], (Int64, device));
    ///
    /// let model_output = no_grad(|| {
    ///     longt5_model.forward_t(
    ///         Some(&input_tensor),
    ///         Some(&encoder_attention_mask),
    ///         None,
    ///         Some(&target_tensor),
    ///         Some(&decoder_attention_mask),
    ///         None,
    ///         None,
    ///         None,
    ///         false,
    ///     )
    /// });
    /// ```
    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        encoder_outputs: Option<&Tensor>,
        decoder_input_ids: Option<&Tensor>,
        decoder_attention_mask: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        decoder_input_embeds: Option<&Tensor>,
        old_layer_states: Option<Vec<(Option<LayerState>, Option<LayerState>)>>,
        train: bool,
    ) -> Result<LongT5ModelOutput, RustBertError> {
        let (calc_hidden_states, all_encoder_hidden_states, all_encoder_attentions) =
            if encoder_outputs.is_none() {
                let encoder_output = self.encoder.forward_t(
                    input_ids,
                    attention_mask,
                    None,
                    None,
                    input_embeds,
                    &self.embeddings,
                    None,
                    train,
                )?;
                (
                    Some(encoder_output.hidden_state),
                    encoder_output.all_hidden_states,
                    encoder_output.all_attentions,
                )
            } else {
                (None, None, None)
            };

        let encoder_output =
            encoder_outputs.unwrap_or_else(|| calc_hidden_states.as_ref().unwrap());

        let decoder_output = self
            .decoder
            .forward_t(
                decoder_input_ids,
                decoder_attention_mask,
                Some(encoder_output),
                attention_mask,
                decoder_input_embeds,
                &self.embeddings,
                old_layer_states,
                train,
            )
            .unwrap();
        Ok(LongT5ModelOutput {
            decoder_output: decoder_output.hidden_state,
            encoder_hidden_state: calc_hidden_states,
            next_cache: decoder_output.next_cache,
            all_decoder_hidden_states: decoder_output.all_hidden_states,
            all_decoder_attentions: decoder_output.all_attentions,
            all_encoder_hidden_states,
            all_encoder_attentions,
        })
    }
}

/// # LongT5 Model for conditional generation
/// LongT5 model with a vocabulary decoding head
/// It is made of the following blocks:
/// - `base_model`: `LongT5Model` Base LongT5 model
/// - `model_dim`: `f64` representation of the model dimension for scaling of the generated logits
pub struct LongT5ForConditionalGeneration {
    base_model: LongT5Model,
    model_dim: f64,
    tie_word_embeddings: bool,
    lm_head: Option<nn::Linear>,
}

impl LongT5ForConditionalGeneration {
    /// Build a new `LongT5ForConditionalGeneration`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the BART model
    /// * `config` - `LongT5Config` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::longt5::{LongT5Config, LongT5ForConditionalGeneration};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = LongT5Config::from_file(config_path);
    /// let longt5 = LongT5ForConditionalGeneration::new(&p.root() / "t5", &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &LongT5Config) -> LongT5ForConditionalGeneration
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let base_model = LongT5Model::new(p, config);
        let tie_word_embeddings = config.tie_word_embeddings.unwrap_or(true);

        let lm_head = if !tie_word_embeddings {
            Some(nn::linear(
                p / "lm_head",
                config.d_model,
                config.vocab_size,
                LinearConfig {
                    bias: false,
                    ..Default::default()
                },
            ))
        } else {
            None
        };

        LongT5ForConditionalGeneration {
            base_model,
            model_dim: config.d_model as f64,
            tie_word_embeddings,
            lm_head,
        }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *source_sequence_length*). This or `input_embeds` must be provided.
    /// * `attention_mask` - Optional attention mask of shape (*batch size*, *source_sequence_length*) for the encoder positions. Positions with a mask with value 0 will be masked.
    /// * `encoder_outputs` - Optional tuple made of a tensor of shape (*batch size*, *source_sequence_length*, *encoder_hidden_dim*) and optional vectors of tensors of length *num_encoder_layers* with shape (*batch size*, *source_sequence_length*, *hidden_size*).
    /// These correspond to the encoder last hidden state and optional hidden states/attention weights for encoder layers. When provided, the encoder hidden state will not be recalculated. Useful for generation tasks.
    /// * `decoder_input_ids` - Optional input tensor of shape (*batch size*, *target_sequence_length*). This or `decoder_input_embeds` must be provided.
    /// * `decoder_attention_mask` - Optional attention mask of shape (*batch size*, *target_sequence_length*) for the decoder positions. Positions with a mask with value 0 will be masked.
    /// * `input_embeds` - Optional input tensor of shape (*batch size*, *source_sequence_length*, *embeddings dimension*). This or `input_ids` must be provided.
    /// * `decoder_input_embeds` - Optional input tensor of shape (*batch size*, *target_sequence_length*, *embeddings dimension*). This or `decoder_input_ids` must be provided.
    /// * `old_layer_states` - Optional vector of length `num_layers` containing tuples of optional `LayerStates` containing the last calculated key and value pairs for the decoder. This avoids recomputing attention weights at past positions and speeds up decoding.
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `longT5ModelOutput` containing:
    ///   - `decoder_output` - `Tensor` of shape (*batch size*, *target_sequence_length*, *vocab_size*) representing the logits for each sequence position and vocabulary item
    ///   - `encoder_hidden_states` - `Tensor` of shape (*batch size*, *source_sequence_length*, *hidden_size*) representing the activations of the last encoder hidden state
    ///   - `cache` - `Option<Vec<(Option<Vec<LayerState, LayerState>>)>>` of length *n_layer* containing the encoder padding mask and past keys and values for both the self attention and the encoder cross attention of each layer of the decoder.
    ///   - `all_encoder_hidden_states` - `Option<Vec<Tensor>>` of length *num_encoder_layers* with shape (*batch size*, *source_sequence_length*, *hidden_size*)
    ///   - `all_encoder_attentions` - `Option<Vec<Tensor>>` of length *num_encoder_layers* with shape (*batch size*, *source_sequence_length*, *hidden_size*)
    ///   - `all_decoder_hidden_states` - `Option<Vec<Tensor>>` of length *num_decoder_layers* with shape (*batch size*, *target_sequence_length*, *hidden_size*)
    ///   - `all_decoder_attentions` - `Option<Vec<Tensor>>` of length *num_decoder_layers* with shape (*batch size*, *target_sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use tch::{nn, Device, Tensor, no_grad};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::{Int64, Double};
    /// use rust_bert::longt5::{LongT5Config, LongT5ForConditionalGeneration};
    /// # let config_path = Path::new("path/to/config.json");
    /// # let vocab_path = Path::new("path/to/vocab.txt");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = LongT5Config::from_file(config_path);
    /// # let longt5_model: LongT5ForConditionalGeneration = LongT5ForConditionalGeneration::new(&vs.root(), &config);
    /// let (batch_size, source_sequence_length, target_sequence_length) = (64, 128, 56);
    /// let input_tensor = Tensor::rand(&[batch_size, source_sequence_length], (Int64, device));
    /// let target_tensor = Tensor::rand(&[batch_size, target_sequence_length], (Int64, device));
    /// let encoder_attention_mask =
    ///     Tensor::ones(&[batch_size, source_sequence_length], (Int64, device));
    /// let decoder_attention_mask =
    ///     Tensor::ones(&[batch_size, source_sequence_length], (Int64, device));
    ///
    /// let model_output = no_grad(|| {
    ///     longt5_model.forward_t(
    ///         Some(&input_tensor),
    ///         Some(&encoder_attention_mask),
    ///         None,
    ///         Some(&target_tensor),
    ///         Some(&decoder_attention_mask),
    ///         None,
    ///         None,
    ///         None,
    ///         false,
    ///     )
    /// });
    /// ```
    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        encoder_outputs: Option<&Tensor>,
        decoder_input_ids: Option<&Tensor>,
        decoder_attention_mask: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        decoder_input_embeds: Option<&Tensor>,
        old_layer_states: Option<Vec<(Option<LayerState>, Option<LayerState>)>>,
        train: bool,
    ) -> Result<LongT5ModelOutput, RustBertError> {
        let base_model_output = self.base_model.forward_t(
            input_ids,
            attention_mask,
            encoder_outputs,
            decoder_input_ids,
            decoder_attention_mask,
            input_embeds,
            decoder_input_embeds,
            old_layer_states,
            train,
        )?;

        let lm_logits = if self.tie_word_embeddings {
            base_model_output
                .decoder_output
                .linear::<Tensor>(&self.base_model.embeddings.ws, None)
                * (self.model_dim.powf(-0.5))
        } else {
            base_model_output
                .decoder_output
                .apply(self.lm_head.as_ref().unwrap())
        };

        Ok(T5ModelOutput {
            decoder_output: lm_logits,
            ..base_model_output
        })
    }

    pub fn encode(&self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> Tensor {
        self.base_model
            .encoder
            .forward_t(
                Some(input_ids),
                attention_mask,
                None,
                None,
                None,
                &self.base_model.embeddings,
                None,
                false,
            )
            .unwrap()
            .hidden_state
    }
}

/// Container holding a LongT5 model output.
pub type LongT5ModelOutput = T5ModelOutput;

pub struct LongT5Generator {
    model: LongT5ForConditionalGeneration,
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

impl LongT5Generator {
    pub fn new(generate_config: GenerateConfig) -> Result<LongT5Generator, RustBertError> {
        let vocab_path = generate_config.vocab_resource.get_local_path()?;

        let tokenizer = TokenizerOption::from_file(
            ModelType::LongT5,
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
    ) -> Result<LongT5Generator, RustBertError> {
        let config_path = generate_config.config_resource.get_local_path()?;
        let weights_path = generate_config.model_resource.get_local_path()?;
        let device = generate_config.device;

        generate_config.validate();
        let mut var_store = nn::VarStore::new(device);

        let config = LongT5Config::from_file(config_path);
        let model = LongT5ForConditionalGeneration::new(var_store.root(), &config);
        var_store.load(weights_path)?;

        let bos_token_id = config.bos_token_id;
        let eos_token_ids = Some(match config.eos_token_id {
            Some(value) => vec![value],
            None => vec![1],
        });
        let pad_token_id = Some(config.pad_token_id.unwrap_or(0));
        let vocab_size = config.vocab_size;
        let is_encoder_decoder = true;
        let decoder_start_id = pad_token_id;
        // longT5 do not have an embedding matrix for position IDs and relies on relative positions instead
        let max_position_embeddings = i64::MAX;

        Ok(LongT5Generator {
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

impl PrivateLanguageGenerator for LongT5Generator {
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
        _input_embeds: Option<&Tensor>,
        encoder_outputs: Option<&Tensor>,
        decoder_input_ids: Option<&Tensor>,
        train: bool,
    ) -> Result<LMModelOutput, RustBertError> {
        let base_model_output = match cache {
            Cache::LongT5Cache(cached_layer_states) => self.model.forward_t(
                input_ids,
                attention_mask,
                encoder_outputs,
                decoder_input_ids,
                None,
                None,
                None,
                cached_layer_states,
                train,
            )?,
            Cache::None => self.model.forward_t(
                input_ids,
                attention_mask,
                encoder_outputs,
                decoder_input_ids,
                None,
                None,
                None,
                None,
                train,
            )?,
            _ => {
                return Err(RustBertError::ValueError(
                    "Cache not compatible with LongT5 Model".into(),
                ));
            }
        };

        Ok(LMModelOutput {
            lm_logits: base_model_output.decoder_output,
            cache: Cache::LongT5Cache(base_model_output.next_cache),
        })
    }

    fn encode(&self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> Option<Tensor> {
        Some(self.model.encode(input_ids, attention_mask))
    }

    fn prepare_inputs_for_generation<'a>(
        &self,
        input_ids: Tensor,
        encoder_outputs: Option<&'a Tensor>,
        past: Cache,
        attention_mask: Tensor,
    ) -> PreparedInput<'a> {
        match past {
            Cache::LongT5Cache(past) => PreparedInput {
                prepared_input: None,
                prepared_attention_mask: Some(attention_mask),
                prepared_encoder_output: encoder_outputs,
                prepared_decoder_input: Some(input_ids.narrow(1, -1, 1)),
                prepared_position_ids: None,
                prepared_past: Cache::LongT5Cache(past),
            },
            Cache::None => PreparedInput {
                prepared_input: None,
                prepared_attention_mask: Some(attention_mask),
                prepared_encoder_output: encoder_outputs,
                prepared_decoder_input: Some(input_ids),
                prepared_position_ids: None,
                prepared_past: Cache::LongT5Cache(None),
            },
            _ => panic!("Cache type incompatible with longT5"),
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
        match past {
            Cache::LongT5Cache(old_cache_option) => match old_cache_option {
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
                panic!("Invalid cache for LongT5 model");
            }
        };
        encoder_outputs
    }
}

impl LanguageGenerator for LongT5Generator {}
