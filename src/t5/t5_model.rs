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

use std::borrow::Borrow;

use rust_tokenizers::tokenizer::TruncationStrategy;
use serde::{Deserialize, Serialize};
use tch::nn::{embedding, LinearConfig};
use tch::{nn, Tensor};

use crate::pipelines::common::{ModelType, TokenizerOption};
use crate::pipelines::generation_utils::private_generation_utils::{
    PreparedInput, PrivateLanguageGenerator,
};
use crate::pipelines::generation_utils::{Cache, GenerateConfig, LMModelOutput, LanguageGenerator};
use crate::pipelines::translation::Language;
use crate::t5::attention::LayerState;
use crate::t5::encoder::T5Stack;
use crate::{Config, RustBertError};

/// # T5 Pretrained model weight files
pub struct T5ModelResources;

/// # T5 Pretrained model config files
pub struct T5ConfigResources;

/// # T5 Pretrained model vocab files
pub struct T5VocabResources;

/// # T5 optional prefixes
pub struct T5Prefix;

/// # T5 source languages pre-sets
pub struct T5SourceLanguages;

/// # T5 target languages pre-sets
pub type T5TargetLanguages = T5SourceLanguages;

impl T5ModelResources {
    /// Shared under Apache 2.0 license by the T5 Authors at <https://github.com/google-research/text-to-text-transfer-transformer>. Modified with conversion to C-array format.
    pub const T5_SMALL: (&'static str, &'static str) = (
        "t5-small/model",
        "https://huggingface.co/t5-small/resolve/main/rust_model.ot",
    );
    /// Shared under Apache 2.0 license by the T5 Authors at <https://github.com/google-research/text-to-text-transfer-transformer>. Modified with conversion to C-array format.
    pub const T5_BASE: (&'static str, &'static str) = (
        "t5-base/model",
        "https://huggingface.co/t5-base/resolve/main/rust_model.ot",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/sentence-t5-base>. Modified with conversion to C-array format.
    pub const SENTENCE_T5_BASE: (&'static str, &'static str) = (
        "sentence-t5-base/model",
        "https://huggingface.co/sentence-transformers/sentence-t5-base/resolve/main/rust_model.ot",
    );
}

impl T5ConfigResources {
    /// Shared under Apache 2.0 license by the Google team at <https://github.com/google-research/text-to-text-transfer-transformer>.
    pub const T5_SMALL: (&'static str, &'static str) = (
        "t5-small/config",
        "https://huggingface.co/t5-small/resolve/main/config.json",
    );
    /// Shared under Apache 2.0 license by the Google team at <https://github.com/google-research/text-to-text-transfer-transformer>.
    pub const T5_BASE: (&'static str, &'static str) = (
        "t5-base/config",
        "https://huggingface.co/t5-base/resolve/main/config.json",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/sentence-t5-base>. Modified with conversion to C-array format.
    pub const SENTENCE_T5_BASE: (&'static str, &'static str) = (
        "sentence-t5-base/config",
        "https://huggingface.co/sentence-transformers/sentence-t5-base/resolve/main/config.json",
    );
}

impl T5VocabResources {
    /// Shared under Apache 2.0 license by the Google team at <https://github.com/google-research/text-to-text-transfer-transformer>.
    pub const T5_SMALL: (&'static str, &'static str) = (
        "t5-small/spiece",
        "https://huggingface.co/t5-small/resolve/main/spiece.model",
    );
    /// Shared under Apache 2.0 license by the Google team at <https://github.com/google-research/text-to-text-transfer-transformer>.
    pub const T5_BASE: (&'static str, &'static str) = (
        "t5-base/spiece",
        "https://huggingface.co/t5-base/resolve/main/spiece.model",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/sentence-t5-base>. Modified with conversion to C-array format.
    pub const SENTENCE_T5_BASE: (&'static str, &'static str) = (
        "sentence-t5-base/spiece",
        "https://huggingface.co/sentence-transformers/sentence-t5-base/resolve/main/spiece.model",
    );
}

const T5LANGUAGES: [Language; 3] = [Language::English, Language::French, Language::German];

impl T5SourceLanguages {
    pub const T5_SMALL: [Language; 3] = T5LANGUAGES;
    pub const T5_BASE: [Language; 3] = T5LANGUAGES;
}

impl T5Prefix {
    pub const ENGLISH2FRENCH: Option<&'static str> = Some("translate English to French:");
    pub const ENGLISH2GERMAN: Option<&'static str> = Some("translate English to German:");
}

#[derive(Clone, Debug, Serialize, Deserialize, Copy)]
#[serde(rename_all = "kebab-case")]
/// # Options for T5 Feed-forward projection layer
pub enum FeedForwardProj {
    /// ReLU
    Relu,
    /// Gated geLU
    GatedGelu,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
/// # T5 model configuration
/// Defines the T5 model architecture (e.g. number of layers, hidden layer size, label mapping...)
pub struct T5Config {
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
    pub output_past: Option<bool>,
    pub pad_token_id: Option<i64>,
    pub relative_attention_num_buckets: i64,
    pub relative_attention_max_distance: Option<i64>,
    pub vocab_size: i64,
    pub feed_forward_proj: Option<FeedForwardProj>,
    pub tie_word_embeddings: Option<bool>,
    pub task_specific_params: Option<TaskSpecificParams>,
    pub output_attentions: Option<bool>,
    pub output_hidden_states: Option<bool>,
}

/// # T5 task-specific configurations
/// Defines the T5 configuration for summarization and translation tasks
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TaskSpecificParams {
    summarization: Summarization,
    translation_en_to_de: TranslationEnToDe,
    translation_en_to_fr: TranslationEnToFr,
    translation_en_to_ro: TranslationEnToRo,
}

/// # T5 summarization configuration
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Summarization {
    early_stopping: bool,
    length_penalty: f64,
    max_length: i64,
    min_length: i64,
    no_repeat_ngram_size: i64,
    num_beams: i64,
    prefix: String,
}

/// # T5 English to German configuration
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TranslationEnToDe {
    early_stopping: bool,
    max_length: i64,
    num_beams: i64,
    prefix: String,
}

/// # T5 English to French configuration
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TranslationEnToFr {
    early_stopping: bool,
    max_length: i64,
    num_beams: i64,
    prefix: String,
}

/// # T5 English to Romanian configuration
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TranslationEnToRo {
    early_stopping: bool,
    max_length: i64,
    num_beams: i64,
    prefix: String,
}

impl Config for T5Config {}

impl Default for T5Config {
    fn default() -> Self {
        T5Config {
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
            output_past: None,
            pad_token_id: Some(0),
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: Some(128),
            vocab_size: 32128,
            feed_forward_proj: Some(FeedForwardProj::Relu),
            tie_word_embeddings: None,
            task_specific_params: None,
            output_attentions: None,
            output_hidden_states: None,
        }
    }
}

/// # T5 Base model
/// Base architecture for T5 model. Usually complemented with a task-specific head, such as a language model head.
/// It is made of the following blocks:
/// - `encoder`: `T5Stack` (transformer) made of a vector of encoding layers
/// - `decoder`: `T5Stack` (transformer)  made of a vector of decoding layers with self attention and encoder cross-attention.
/// caching is implemented for the decoder to avoid recalculating static states (encoder key/values and previously calculated decoder key/values)
/// - `embeddings`: `nn::Embedding` Shared embeddings for the encoder and decoder.
pub struct T5Model {
    pub(crate) encoder: T5Stack,
    decoder: T5Stack,
    pub(crate) embeddings: nn::Embedding,
}

impl T5Model {
    /// Build a new `T5Model`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the T5 model
    /// * `config` - `T5Config` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::t5::{T5Config, T5Model};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = T5Config::from_file(config_path);
    /// let t5: T5Model = T5Model::new(&p.root() / "t5", &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &T5Config) -> T5Model
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
            config.output_attentions.unwrap_or(false),
            config.output_hidden_states.unwrap_or(false),
        );
        let decoder = T5Stack::new(
            p / "decoder",
            config,
            true,
            true,
            config.output_attentions.unwrap_or(false),
            config.output_hidden_states.unwrap_or(false),
        );

        T5Model {
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
    /// * `decoder_input_ids` - Optional input tensor of shape (*batch size*, *target_sequence_length*). This or `decoder_input_embeds` must be provided.
    /// * `encoder_outputs` - Optional tuple made of a tensor of shape (*batch size*, *source_sequence_length*, *encoder_hidden_dim*) and optional vectors of tensors of length *num_encoder_layers* with shape (*batch size*, *source_sequence_length*, *hidden_size*).
    /// These correspond to the encoder last hidden state and optional hidden states/attention weights for encoder layers. When provided, the encoder hidden state will not be recalculated. Useful for generation tasks.
    /// * `decoder_attention_mask` - Optional attention mask of shape (*batch size*, *target_sequence_length*) for the decoder positions. Positions with a mask with value 0 will be masked.
    /// * `input_embeds` - Optional input tensor of shape (*batch size*, *source_sequence_length*, *embeddings dimension*). This or `input_ids` must be provided.
    /// * `decoder_input_embeds` - Optional input tensor of shape (*batch size*, *target_sequence_length*, *embeddings dimension*). This or `decoder_input_ids` must be provided.
    /// * `old_layer_states` - Optional vector of length `num_layers` containing tuples of optional `LayerStates` containing the last calculated key and value pairs for the decoder. This avoids recomputing attention weights at past positions and speeds up decoding.
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `T5ModelOutput` containing:
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
    /// use rust_bert::t5::{T5Config, T5Model};
    /// # let config_path = Path::new("path/to/config.json");
    /// # let vocab_path = Path::new("path/to/vocab.txt");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = T5Config::from_file(config_path);
    /// # let t5_model: T5Model = T5Model::new(&vs.root(), &config);
    /// let (batch_size, source_sequence_length, target_sequence_length) = (64, 128, 56);
    /// let input_tensor = Tensor::rand(&[batch_size, source_sequence_length], (Int64, device));
    /// let target_tensor = Tensor::rand(&[batch_size, target_sequence_length], (Int64, device));
    /// let encoder_attention_mask =
    ///     Tensor::ones(&[batch_size, source_sequence_length], (Int64, device));
    /// let decoder_attention_mask =
    ///     Tensor::ones(&[batch_size, source_sequence_length], (Int64, device));
    ///
    /// let model_output = no_grad(|| {
    ///     t5_model.forward_t(
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
    ) -> T5ModelOutput {
        let calc_encoder_outputs = if encoder_outputs.is_none() {
            Some(
                self.encoder
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
                    .unwrap(),
            )
        } else {
            None
        };

        let (calc_hidden_states, all_encoder_hidden_states, all_encoder_attentions) =
            if let Some(calc_encoder_outputs) = calc_encoder_outputs {
                (
                    Some(calc_encoder_outputs.hidden_state),
                    calc_encoder_outputs.all_hidden_states,
                    calc_encoder_outputs.all_attentions,
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
        T5ModelOutput {
            decoder_output: decoder_output.hidden_state,
            encoder_hidden_state: calc_hidden_states,
            next_cache: decoder_output.next_cache,
            all_decoder_hidden_states: decoder_output.all_hidden_states,
            all_decoder_attentions: decoder_output.all_attentions,
            all_encoder_hidden_states,
            all_encoder_attentions,
        }
    }
}

/// # T5 Model for conditional generation
/// T5 model with a vocabulary decoding head
/// It is made of the following blocks:
/// - `base_model`: `T5Model` Base T5 model
/// - `model_dim`: `f64` representation of the model dimension for scaling of the generated logits
pub struct T5ForConditionalGeneration {
    base_model: T5Model,
    model_dim: f64,
    tie_word_embeddings: bool,
    lm_head: Option<nn::Linear>,
}

impl T5ForConditionalGeneration {
    /// Build a new `T5ForConditionalGeneration`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the BART model
    /// * `config` - `T5Config` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::t5::{T5Config, T5ForConditionalGeneration};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = T5Config::from_file(config_path);
    /// let t5 = T5ForConditionalGeneration::new(&p.root() / "t5", &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &T5Config) -> T5ForConditionalGeneration
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let base_model = T5Model::new(p, config);
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

        T5ForConditionalGeneration {
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
    /// * `decoder_input_ids` - Optional input tensor of shape (*batch size*, *target_sequence_length*). This or `decoder_input_embeds` must be provided.
    /// * `encoder_outputs` - Optional tuple made of a tensor of shape (*batch size*, *source_sequence_length*, *encoder_hidden_dim*) and optional vectors of tensors of length *num_encoder_layers* with shape (*batch size*, *source_sequence_length*, *hidden_size*).
    /// These correspond to the encoder last hidden state and optional hidden states/attention weights for encoder layers. When provided, the encoder hidden state will not be recalculated. Useful for generation tasks.
    /// * `decoder_attention_mask` - Optional attention mask of shape (*batch size*, *target_sequence_length*) for the decoder positions. Positions with a mask with value 0 will be masked.
    /// * `input_embeds` - Optional input tensor of shape (*batch size*, *source_sequence_length*, *embeddings dimension*). This or `input_ids` must be provided.
    /// * `decoder_input_embeds` - Optional input tensor of shape (*batch size*, *target_sequence_length*, *embeddings dimension*). This or `decoder_input_ids` must be provided.
    /// * `old_layer_states` - Optional vector of length `num_layers` containing tuples of optional `LayerStates` containing the last calculated key and value pairs for the decoder. This avoids recomputing attention weights at past positions and speeds up decoding.
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `T5ModelOutput` containing:
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
    /// use rust_bert::t5::{T5Config, T5ForConditionalGeneration};
    /// # let config_path = Path::new("path/to/config.json");
    /// # let vocab_path = Path::new("path/to/vocab.txt");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = T5Config::from_file(config_path);
    /// # let t5_model: T5ForConditionalGeneration = T5ForConditionalGeneration::new(&vs.root(), &config);
    /// let (batch_size, source_sequence_length, target_sequence_length) = (64, 128, 56);
    /// let input_tensor = Tensor::rand(&[batch_size, source_sequence_length], (Int64, device));
    /// let target_tensor = Tensor::rand(&[batch_size, target_sequence_length], (Int64, device));
    /// let encoder_attention_mask =
    ///     Tensor::ones(&[batch_size, source_sequence_length], (Int64, device));
    /// let decoder_attention_mask =
    ///     Tensor::ones(&[batch_size, source_sequence_length], (Int64, device));
    ///
    /// let model_output = no_grad(|| {
    ///     t5_model.forward_t(
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
    ) -> T5ModelOutput {
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
        );

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

        T5ModelOutput {
            decoder_output: lm_logits,
            ..base_model_output
        }
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

/// # T5 for sentence embeddings
/// Transformer usable in [`SentenceEmbeddingsModel`](crate::pipelines::sentence_embeddings::SentenceEmbeddingsModel).
pub struct T5ForSentenceEmbeddings {
    embeddings: nn::Embedding,
    encoder: T5Stack,
}

impl T5ForSentenceEmbeddings {
    /// Build a new `T5ForSentenceEmbeddings`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the BART model
    /// * `config` - `T5Config` object defining the model architecture
    ///
    /// It consists of only an encoder (there is no decoder).
    pub fn new<'p, P>(p: P, config: &T5Config) -> Self
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
            config.output_attentions.unwrap_or(false),
            config.output_hidden_states.unwrap_or(false),
        );

        Self {
            embeddings,
            encoder,
        }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Input of shape (*batch size*, *source_sequence_length*).
    /// * `mask` - Attention mask of shape (*batch size*, *source_sequence_length*) for the encoder positions. Positions with a mask with value 0 will be masked.
    ///
    /// # Returns
    ///
    /// * Tuple containing:
    ///   - `Tensor` of shape (*batch size*, *target_sequence_length*, *hidden_size*) representing the activations of the last encoder hidden state
    ///   - `Option<Vec<Tensor>>` of length *num_encoder_layers* of shape (*batch size*, *target_sequence_length*, *hidden_size*)  representing attention weights for all layers of the encoder
    pub fn forward(
        &self,
        input_ids: &Tensor,
        mask: &Tensor,
    ) -> Result<(Tensor, Option<Vec<Tensor>>), RustBertError> {
        let transformer_output = self.encoder.forward_t(
            Some(input_ids),
            Some(mask),
            None,
            None,
            None,
            &self.embeddings,
            None,
            false,
        )?;
        Ok((
            transformer_output.hidden_state,
            transformer_output.all_attentions,
        ))
    }
}

/// Container holding a T5 model output. The decoder output may hold the hidden state of
/// the last layer of the decoder, or may hold logits for a custom head module after the
/// decoder (e.g. for language modeling tasks)
pub struct T5ModelOutput {
    /// Hidden state of the last layer of the decoder, or logits for a custom head
    /// module after the decoder (e.g. for language modeling tasks)
    pub decoder_output: Tensor,
    /// Hidden state for the last layer of the encoder if they are calculated, otherwise None
    pub encoder_hidden_state: Option<Tensor>,
    /// Cached outputs of the model (attention layers keys and values) if the model is used for generation
    pub next_cache: Option<Vec<(Option<LayerState>, Option<LayerState>)>>,
    /// Hidden states for all layers of the decoder
    pub all_decoder_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all layers of the decoder
    pub all_decoder_attentions: Option<Vec<Tensor>>,
    /// Hidden states for all layers of the encoder
    pub all_encoder_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all layers of the encoder
    pub all_encoder_attentions: Option<Vec<Tensor>>,
}

pub struct T5Generator {
    model: T5ForConditionalGeneration,
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

impl T5Generator {
    pub fn new(generate_config: GenerateConfig) -> Result<T5Generator, RustBertError> {
        let vocab_path = generate_config.vocab_resource.get_local_path()?;

        let tokenizer = TokenizerOption::from_file(
            ModelType::T5,
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
    ) -> Result<T5Generator, RustBertError> {
        let config_path = generate_config.config_resource.get_local_path()?;
        let weights_path = generate_config.model_resource.get_local_path()?;
        let device = generate_config.device;

        generate_config.validate();
        let mut var_store = nn::VarStore::new(device);

        let config = T5Config::from_file(config_path);
        let model = T5ForConditionalGeneration::new(var_store.root(), &config);
        var_store.load(weights_path)?;

        let bos_token_id = Some(config.bos_token_id.unwrap_or(-1));
        let eos_token_ids = Some(match config.eos_token_id {
            Some(value) => vec![value],
            None => vec![1],
        });
        let pad_token_id = Some(config.pad_token_id.unwrap_or(0));
        let vocab_size = config.vocab_size;
        let is_encoder_decoder = true;
        let decoder_start_id = Some(0);
        // T5 do not have an embedding matrix for position IDs and relies on relative positions instead
        let max_position_embeddings = i64::MAX;

        Ok(T5Generator {
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

impl PrivateLanguageGenerator for T5Generator {
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
            Cache::T5Cache(cached_layer_states) => self.model.forward_t(
                input_ids,
                attention_mask,
                encoder_outputs,
                decoder_input_ids,
                None,
                None,
                None,
                cached_layer_states,
                train,
            ),
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
            ),
            _ => {
                return Err(RustBertError::ValueError(
                    "Cache not compatible with T5 Model".into(),
                ));
            }
        };

        Ok(LMModelOutput {
            lm_logits: base_model_output.decoder_output,
            cache: Cache::T5Cache(base_model_output.next_cache),
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
            Cache::T5Cache(past) => PreparedInput {
                prepared_input: None,
                prepared_attention_mask: Some(attention_mask),
                prepared_encoder_output: encoder_outputs,
                prepared_decoder_input: Some(input_ids.narrow(1, -1, 1)),
                prepared_position_ids: None,
                prepared_past: Cache::T5Cache(past),
            },
            Cache::None => PreparedInput {
                prepared_input: None,
                prepared_attention_mask: Some(attention_mask),
                prepared_encoder_output: encoder_outputs,
                prepared_decoder_input: Some(input_ids),
                prepared_position_ids: None,
                prepared_past: Cache::T5Cache(None),
            },
            _ => panic!("Cache type incompatible with T5"),
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
            Cache::T5Cache(old_cache_option) => match old_cache_option {
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
                panic!("Invalid cache for T5 model");
            }
        };
        encoder_outputs
    }
}

impl LanguageGenerator for T5Generator {}
