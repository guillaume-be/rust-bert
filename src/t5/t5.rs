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
use crate::pipelines::generation::{Cache, LMHeadModel};
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

/// # T5 optional prefixes
pub struct T5Prefix;

impl T5ModelResources {
    /// Shared under Apache 2.0 license by the T5 Authors at https://github.com/google-research/text-to-text-transfer-transformer. Modified with conversion to C-array format.
    pub const T5_SMALL: (&'static str, &'static str) = (
        "t5-small/model.ot",
        "https://cdn.huggingface.co/t5-small/rust_model.ot",
    );
    /// Shared under Apache 2.0 license by the T5 Authors at https://github.com/google-research/text-to-text-transfer-transformer. Modified with conversion to C-array format.
    pub const T5_BASE: (&'static str, &'static str) = (
        "t5-base/model.ot",
        "https://cdn.huggingface.co/t5-base/rust_model.ot",
    );
}

impl T5ConfigResources {
    /// Shared under Apache 2.0 license by the Google team at https://github.com/google-research/text-to-text-transfer-transformer.
    pub const T5_SMALL: (&'static str, &'static str) = (
        "t5-small/config.json",
        "https://s3.amazonaws.com/models.huggingface.co/bert/t5-small-config.json",
    );
    /// Shared under Apache 2.0 license by the Google team at https://github.com/google-research/text-to-text-transfer-transformer.
    pub const T5_BASE: (&'static str, &'static str) = (
        "t5-base/config.json",
        "https://s3.amazonaws.com/models.huggingface.co/bert/t5-base-config.json",
    );
}

impl T5VocabResources {
    /// Shared under Apache 2.0 license by the Google team at https://github.com/google-research/text-to-text-transfer-transformer.
    pub const T5_SMALL: (&'static str, &'static str) = (
        "t5-small/spiece.model",
        "https://s3.amazonaws.com/models.huggingface.co/bert/t5-spiece.model",
    );
    /// Shared under Apache 2.0 license by the Google team at https://github.com/google-research/text-to-text-transfer-transformer.
    pub const T5_BASE: (&'static str, &'static str) = (
        "t5-base/spiece.model",
        "https://s3.amazonaws.com/models.huggingface.co/bert/t5-spiece.model",
    );
}

impl T5Prefix {
    pub const ENGLISH2FRENCH: Option<&'static str> = Some("translate English to French: ");
    pub const ENGLISH2GERMAN: Option<&'static str> = Some("translate English to German: ");
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

/// # T5 task-specific configurations
/// Defines the T5 configuration for summarization and translation tasks
#[derive(Debug, Serialize, Deserialize)]
pub struct TaskSpecificParams {
    summarization: Summarization,
    translation_en_to_de: TranslationEnToDe,
    translation_en_to_fr: TranslationEnToFr,
    translation_en_to_ro: TranslationEnToRo,
}

/// # T5 summarization configuration
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

/// # T5 English to German configuration
#[derive(Debug, Serialize, Deserialize)]
pub struct TranslationEnToDe {
    early_stopping: bool,
    max_length: i64,
    num_beams: i64,
    prefix: String,
}

/// # T5 English to French configuration
#[derive(Debug, Serialize, Deserialize)]
pub struct TranslationEnToFr {
    early_stopping: bool,
    max_length: i64,
    num_beams: i64,
    prefix: String,
}

/// # T5 English to Romanian configuration
#[derive(Debug, Serialize, Deserialize)]
pub struct TranslationEnToRo {
    early_stopping: bool,
    max_length: i64,
    num_beams: i64,
    prefix: String,
}

impl Config<T5Config> for T5Config {}

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
    /// * `p` - Variable store path for the root of the BART model
    /// * `config` - `T5Config` object defining the model architecture
    /// * `output_attention` - flag indicating if the model should output the attention weights of intermediate layers
    /// * `output_hidden_states` - flag indicating if the model should output the hidden states weights of intermediate layers
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
    /// let output_attentions = true;
    /// let output_hidden_states = true;
    /// let t5: T5Model = T5Model::new(
    ///     &p.root() / "t5",
    ///     &config,
    ///     output_attentions,
    ///     output_hidden_states,
    /// );
    /// ```
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
    /// * `old_layer_states` - Optional vector of length `num_layers` containing tuples of optional `LayerStates` containing th elast calculated key and value pairs for the decoder. This avoids recomputing attention weights at past positions and speeds up decoding.
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `decoder_output` - `Tensor` of shape (*batch size*, *target_sequence_length*, *hidden_size*) representing the activations of the last decoder hidden state
    /// * `encoder_hidden_states` - `Tensor` of shape (*batch size*, *source_sequence_length*, *hidden_size*) representing the activations of the last encoder hidden state
    /// * `decoder_cache` - `Option<Vec<(Option<Vec<&LayerState, &LayerState>>)>>` of length *n_layer* containing the encoder past keys and values for
    /// both the self attention and the encoder cross attention of each layer of the decoder.
    /// * `all_encoder_hidden_states` - `Option<Vec<Tensor>>` of length *num_encoder_layers* with shape (*batch size*, *source_sequence_length*, *hidden_size*)
    /// * `all_encoder_attentions` - `Option<Vec<Tensor>>` of length *num_encoder_layers* with shape (*batch size*, *source_sequence_length*, *hidden_size*)
    /// * `all_decoder_hidden_states` - `Option<Vec<Tensor>>` of length *num_decoder_layers* with shape (*batch size*, *target_sequence_length*, *hidden_size*)
    /// * `all_decoder_attentions` - `Option<Vec<Tensor>>` of length *num_decoder_layers* with shape (*batch size*, *target_sequence_length*, *hidden_size*)
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
    /// # let t5_model: T5Model = T5Model::new(&vs.root(), &config, false, false);
    /// let (batch_size, source_sequence_length, target_sequence_length) = (64, 128, 56);
    /// let input_tensor = Tensor::rand(&[batch_size, source_sequence_length], (Int64, device));
    /// let target_tensor = Tensor::rand(&[batch_size, target_sequence_length], (Int64, device));
    /// let encoder_attention_mask =
    ///     Tensor::ones(&[batch_size, source_sequence_length], (Int64, device));
    /// let decoder_attention_mask =
    ///     Tensor::ones(&[batch_size, source_sequence_length], (Int64, device));
    ///
    /// let (
    ///     decoder_output,
    ///     encoder_hidden_states,
    ///     decoder_cache,
    ///     all_encoder_hidden_states,
    ///     all_encoder_attentions,
    ///     all_decoder_hidden_states,
    ///     all_decoder_attentions,
    /// ) = no_grad(|| {
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
                    Some(value) => Some(value.narrow(1, -1, 1)),
                    None => None,
                };
                let decoder_input_embeds = match &decoder_input_embeds {
                    Some(value) => Some(value.narrow(1, -1, 1)),
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

/// # T5 Model for conditional generation
/// T5 model with a vocabulary decoding head
/// It is made of the following blocks:
/// - `base_model`: `T5Model` Base T5 model
/// - `model_dim`: `f64` representation of the model dimension for scaling of the generated logits
pub struct T5ForConditionalGeneration {
    base_model: T5Model,
    model_dim: f64,
}

impl T5ForConditionalGeneration {
    /// Build a new `T5ForConditionalGeneration`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the BART model
    /// * `config` - `T5Config` object defining the model architecture
    /// * `output_attention` - flag indicating if the model should output the attention weights of intermediate layers
    /// * `output_hidden_states` - flag indicating if the model should output the hidden states weights of intermediate layers
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
    /// let output_attentions = true;
    /// let output_hidden_states = true;
    /// let t5 = T5ForConditionalGeneration::new(
    ///     &p.root() / "t5",
    ///     &config,
    ///     output_attentions,
    ///     output_hidden_states,
    /// );
    /// ```
    pub fn new<'p, P>(
        p: P,
        config: &T5Config,
        output_attentions: bool,
        output_hidden_states: bool,
    ) -> T5ForConditionalGeneration
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let base_model = T5Model::new(p, config, output_attentions, output_hidden_states);

        T5ForConditionalGeneration {
            base_model,
            model_dim: config.d_model as f64,
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
    /// * `old_layer_states` - Optional vector of length `num_layers` containing tuples of optional `LayerStates` containing th elast calculated key and value pairs for the decoder. This avoids recomputing attention weights at past positions and speeds up decoding.
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `decoder_output` - `Tensor` of shape (*batch size*, *target_sequence_length*, *hidden_size*) representing the activations of the last decoder hidden state
    /// * `encoder_hidden_states` - `Tensor` of shape (*batch size*, *source_sequence_length*, *hidden_size*) representing the activations of the last encoder hidden state
    /// * `decoder_cache` - `Option<Vec<(Option<Vec<&LayerState, &LayerState>>)>>` of length *n_layer* containing the encoder past keys and values for
    /// both the self attention and the encoder cross attention of each layer of the decoder.
    /// * `all_encoder_hidden_states` - `Option<Vec<Tensor>>` of length *num_encoder_layers* with shape (*batch size*, *source_sequence_length*, *hidden_size*)
    /// * `all_encoder_attentions` - `Option<Vec<Tensor>>` of length *num_encoder_layers* with shape (*batch size*, *source_sequence_length*, *hidden_size*)
    /// * `all_decoder_hidden_states` - `Option<Vec<Tensor>>` of length *num_decoder_layers* with shape (*batch size*, *target_sequence_length*, *hidden_size*)
    /// * `all_decoder_attentions` - `Option<Vec<Tensor>>` of length *num_decoder_layers* with shape (*batch size*, *target_sequence_length*, *hidden_size*)
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
    /// # let t5_model: T5ForConditionalGeneration = T5ForConditionalGeneration::new(&vs.root(), &config, false, false);
    /// let (batch_size, source_sequence_length, target_sequence_length) = (64, 128, 56);
    /// let input_tensor = Tensor::rand(&[batch_size, source_sequence_length], (Int64, device));
    /// let target_tensor = Tensor::rand(&[batch_size, target_sequence_length], (Int64, device));
    /// let encoder_attention_mask =
    ///     Tensor::ones(&[batch_size, source_sequence_length], (Int64, device));
    /// let decoder_attention_mask =
    ///     Tensor::ones(&[batch_size, source_sequence_length], (Int64, device));
    ///
    /// let (
    ///     decoder_output,
    ///     encoder_hidden_states,
    ///     decoder_cache,
    ///     all_encoder_hidden_states,
    ///     all_encoder_attentions,
    ///     all_decoder_hidden_states,
    ///     all_decoder_attentions,
    /// ) = no_grad(|| {
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
        let (
            decoder_outputs,
            encoder_hidden_states,
            decoder_cache,
            all_decoder_hidden_states,
            all_decoder_attentions,
            all_encoder_hidden_states,
            all_encoder_attentions,
        ) = self.base_model.forward_t(
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
        let lm_logits = decoder_outputs.linear::<Tensor>(&self.base_model.embeddings.ws, None)
            * (self.model_dim.powf(-0.5));

        (
            lm_logits,
            encoder_hidden_states,
            decoder_cache,
            all_decoder_hidden_states,
            all_decoder_attentions,
            all_encoder_hidden_states,
            all_encoder_attentions,
        )
    }

    pub fn encode(&self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> Tensor {
        let (encoder_hidden_states, _, _, _) = self
            .base_model
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
            .unwrap();
        encoder_hidden_states
    }
}

impl LMHeadModel for T5ForConditionalGeneration {
    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see `input_embeds`)
    /// * `layer_past` - Optional vector of length `num_layers` containing tuples of optional `LayerStates` containing th elast calculated key and value pairs for the decoder. This avoids recomputing attention weights at past positions and speeds up decoding.
    /// * `attention_mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `input_embeds` - Unused for T5
    /// * `token_type_ids` - Unused for T5
    /// * `position_ids` - Unused for T5
    /// * `encoder_outputs` - Optional tensor of shape (*batch size*, *source_sequence_length*, *hidden_size*). When provided, the encoder hidden state will not be recalculated. Useful for generation tasks.
    /// * `decoder_input_ids` - Optional input tensor of shape (*batch size*, *target_sequence_length*).
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `lm_logits` - `Tensor` of shape (*batch size*, *sequence_length*, *vocab_size*) representing the logits for each vocab item and position
    /// * `past` - `T5Cache` made of `Option<Vec<(Option<Vec<&LayerState, &LayerState>>)>>` of length *n_layer* containing the encoder past keys and values for
    ///     both the self attention and the encoder cross attention of each layer of the decoder.
    /// * `encoder_hidden_states` - `Option<Tensor>` Hidden states for the encoder
    /// * `hidden_states` - None
    /// * `attentions` - None

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
    /// # let t5_model: T5ForConditionalGeneration = T5ForConditionalGeneration::new(&vs.root(), &config, false, false);
    /// let (batch_size, source_sequence_length, target_sequence_length) = (64, 128, 56);
    /// let input_tensor = Tensor::rand(&[batch_size, source_sequence_length], (Int64, device));
    /// let target_tensor = Tensor::rand(&[batch_size, target_sequence_length], (Int64, device));
    /// let encoder_attention_mask =
    ///     Tensor::ones(&[batch_size, source_sequence_length], (Int64, device));
    /// let decoder_attention_mask =
    ///     Tensor::ones(&[batch_size, source_sequence_length], (Int64, device));
    ///
    /// let (
    ///     decoder_output,
    ///     encoder_hidden_states,
    ///     decoder_cache,
    ///     all_encoder_hidden_states,
    ///     all_encoder_attentions,
    ///     all_decoder_hidden_states,
    ///     all_decoder_attentions,
    /// ) = no_grad(|| {
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
    fn forward_t(
        &self,
        input_ids: &Option<Tensor>,
        cache: Cache,
        attention_mask: &Option<Tensor>,
        _token_type_ids: &Option<Tensor>,
        _position_ids: &Option<Tensor>,
        _input_embeds: &Option<Tensor>,
        encoder_outputs: Option<&Tensor>,
        decoder_input_ids: &Option<Tensor>,
        train: bool,
    ) -> Result<
        (
            Tensor,
            Option<Tensor>,
            Cache,
            Option<Vec<Tensor>>,
            Option<Vec<Tensor>>,
        ),
        &'static str,
    > {
        let (decoder_output, encoder_hidden_states, new_cache, _, _, _, _) = match cache {
            Cache::T5Cache(cached_layer_states) => self.base_model.forward_t(
                input_ids.as_ref(),
                attention_mask.as_ref(),
                Some((encoder_outputs.as_ref().unwrap().copy(), None, None)),
                Option::from(decoder_input_ids),
                None,
                None,
                None,
                cached_layer_states,
                train,
            ),
            Cache::None => self.base_model.forward_t(
                input_ids.as_ref(),
                attention_mask.as_ref(),
                Some((encoder_outputs.as_ref().unwrap().copy(), None, None)),
                Option::from(decoder_input_ids),
                None,
                None,
                None,
                None,
                train,
            ),
            _ => Err("Cache not compatible with T5 Model")?,
        };

        let lm_logits = decoder_output.linear::<Tensor>(&self.base_model.embeddings.ws, None)
            * (self.model_dim.powf(-0.5));

        Ok((
            lm_logits,
            Some(encoder_hidden_states),
            Cache::T5Cache(new_cache),
            None,
            None,
        ))
    }
}
