// Copyright 2018-2020 The HuggingFace Inc. team.
// Copyright 2020 Marian Team Authors
// Copyright 2019-2020 Guillaume Becquin
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::bart::{BartConfig, BartModel, BartModelOutput, LayerState};
use crate::pipelines::generation_utils::{Cache, LMHeadModel, LMModelOutput};
use crate::RustBertError;
use std::borrow::Borrow;
use tch::nn::Init;
use tch::{nn, Tensor};

/// # Marian Pretrained model weight files
pub struct MarianModelResources;

/// # Marian Pretrained model config files
pub struct MarianConfigResources;

/// # Marian Pretrained model vocab files
pub struct MarianVocabResources;

/// # Marian Pretrained sentence piece model files
pub struct MarianSpmResources;

/// # Marian optional prefixes
pub struct MarianPrefix;

impl MarianModelResources {
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at https://github.com/Helsinki-NLP/Opus-MT. Modified with conversion to C-array format.
    pub const ENGLISH2ROMANCE: (&'static str, &'static str) = (
        "marian-mt-en-ROMANCE/model",
        "https://huggingface.co/Helsinki-NLP/opus-mt-en-ROMANCE/resolve/main/rust_model.ot",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at https://github.com/Helsinki-NLP/Opus-MT. Modified with conversion to C-array format.
    pub const ROMANCE2ENGLISH: (&'static str, &'static str) = (
        "marian-mt-ROMANCE-en/model",
        "https://huggingface.co/Helsinki-NLP/opus-mt-ROMANCE-en/resolve/main/rust_model.ot",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at https://github.com/Helsinki-NLP/Opus-MT. Modified with conversion to C-array format.
    pub const ENGLISH2GERMAN: (&'static str, &'static str) = (
        "marian-mt-en-de/model",
        "https://huggingface.co/Helsinki-NLP/opus-mt-en-de/resolve/main/rust_model.ot",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at https://github.com/Helsinki-NLP/Opus-MT. Modified with conversion to C-array format.
    pub const GERMAN2ENGLISH: (&'static str, &'static str) = (
        "marian-mt-de-en/model",
        "https://huggingface.co/Helsinki-NLP/opus-mt-de-en/resolve/main/rust_model.ot",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at https://github.com/Helsinki-NLP/Opus-MT. Modified with conversion to C-array format.
    pub const ENGLISH2RUSSIAN: (&'static str, &'static str) = (
        "marian-mt-en-ru/model",
        "https://huggingface.co/Helsinki-NLP/opus-mt-en-ru/resolve/main/rust_model.ot",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at https://github.com/Helsinki-NLP/Opus-MT. Modified with conversion to C-array format.
    pub const RUSSIAN2ENGLISH: (&'static str, &'static str) = (
        "marian-mt-ru-en/model",
        "https://huggingface.co/Helsinki-NLP/opus-mt-ru-en/resolve/main/rust_model.ot",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at https://github.com/Helsinki-NLP/Opus-MT. Modified with conversion to C-array format.
    pub const FRENCH2GERMAN: (&'static str, &'static str) = (
        "marian-mt-fr-de/model",
        "https://huggingface.co/Helsinki-NLP/opus-mt-fr-de/resolve/main/rust_model.ot",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at https://github.com/Helsinki-NLP/Opus-MT. Modified with conversion to C-array format.
    pub const GERMAN2FRENCH: (&'static str, &'static str) = (
        "marian-mt-de-fr/model",
        "https://huggingface.co/Helsinki-NLP/opus-mt-de-fr/resolve/main/rust_model.ot",
    );
}

impl MarianConfigResources {
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at https://github.com/Helsinki-NLP/Opus-MT.
    pub const ENGLISH2ROMANCE: (&'static str, &'static str) = (
        "marian-mt-en-ROMANCE/config",
        "https://huggingface.co/Helsinki-NLP/opus-mt-en-ROMANCE/resolve/main/config.json",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at https://github.com/Helsinki-NLP/Opus-MT.
    pub const ROMANCE2ENGLISH: (&'static str, &'static str) = (
        "marian-mt-ROMANCE-en/config",
        "https://huggingface.co/Helsinki-NLP/opus-mt-ROMANCE-en/resolve/main/config.json",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at https://github.com/Helsinki-NLP/Opus-MT.
    pub const ENGLISH2GERMAN: (&'static str, &'static str) = (
        "marian-mt-en-de/config",
        "https://huggingface.co/Helsinki-NLP/opus-mt-en-de/resolve/main/config.json",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at https://github.com/Helsinki-NLP/Opus-MT.
    pub const GERMAN2ENGLISH: (&'static str, &'static str) = (
        "marian-mt-de-en/config",
        "https://huggingface.co/Helsinki-NLP/opus-mt-de-en/resolve/main/config.json",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at https://github.com/Helsinki-NLP/Opus-MT.
    pub const ENGLISH2RUSSIAN: (&'static str, &'static str) = (
        "marian-mt-en-ru/config",
        "https://huggingface.co/Helsinki-NLP/opus-mt-en-ru/resolve/main/config.json",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at https://github.com/Helsinki-NLP/Opus-MT.
    pub const RUSSIAN2ENGLISH: (&'static str, &'static str) = (
        "marian-mt-ru-en/config",
        "https://huggingface.co/Helsinki-NLP/opus-mt-ru-en/resolve/main/config.json",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at https://github.com/Helsinki-NLP/Opus-MT.
    pub const FRENCH2GERMAN: (&'static str, &'static str) = (
        "marian-mt-fr-de/config",
        "https://huggingface.co/Helsinki-NLP/opus-mt-fr-de/resolve/main/config.json",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at https://github.com/Helsinki-NLP/Opus-MT.
    pub const GERMAN2FRENCH: (&'static str, &'static str) = (
        "marian-mt-de-fr/config",
        "https://huggingface.co/Helsinki-NLP/opus-mt-de-fr/resolve/main/config.json",
    );
}

impl MarianVocabResources {
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at https://github.com/Helsinki-NLP/Opus-MT.
    pub const ENGLISH2ROMANCE: (&'static str, &'static str) = (
        "marian-mt-en-ROMANCE/vocab",
        "https://huggingface.co/Helsinki-NLP/opus-mt-en-ROMANCE/resolve/main/vocab.json",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at https://github.com/Helsinki-NLP/Opus-MT.
    pub const ROMANCE2ENGLISH: (&'static str, &'static str) = (
        "marian-mt-ROMANCE-en/vocab",
        "https://huggingface.co/Helsinki-NLP/opus-mt-ROMANCE-en/resolve/main/vocab.json",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at https://github.com/Helsinki-NLP/Opus-MT.
    pub const ENGLISH2GERMAN: (&'static str, &'static str) = (
        "marian-mt-en-de/vocab",
        "https://huggingface.co/Helsinki-NLP/opus-mt-en-de/resolve/main/vocab.json",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at https://github.com/Helsinki-NLP/Opus-MT.
    pub const GERMAN2ENGLISH: (&'static str, &'static str) = (
        "marian-mt-de-en/vocab",
        "https://huggingface.co/Helsinki-NLP/opus-mt-de-en/resolve/main/vocab.json",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at https://github.com/Helsinki-NLP/Opus-MT.
    pub const ENGLISH2RUSSIAN: (&'static str, &'static str) = (
        "marian-mt-en-ru/vocab",
        "https://huggingface.co/Helsinki-NLP/opus-mt-en-ru/resolve/main/vocab.json",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at https://github.com/Helsinki-NLP/Opus-MT.
    pub const RUSSIAN2ENGLISH: (&'static str, &'static str) = (
        "marian-mt-ru-en/vocab",
        "https://huggingface.co/Helsinki-NLP/opus-mt-ru-en/resolve/main/vocab.json",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at https://github.com/Helsinki-NLP/Opus-MT.
    pub const FRENCH2GERMAN: (&'static str, &'static str) = (
        "marian-mt-fr-de/vocab",
        "https://huggingface.co/Helsinki-NLP/opus-mt-fr-de/resolve/main/vocab.json",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at https://github.com/Helsinki-NLP/Opus-MT.
    pub const GERMAN2FRENCH: (&'static str, &'static str) = (
        "marian-mt-de-fr/vocab",
        "https://huggingface.co/Helsinki-NLP/opus-mt-de-fr/resolve/main/vocab.json",
    );
}

impl MarianSpmResources {
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at https://github.com/Helsinki-NLP/Opus-MT.
    pub const ENGLISH2ROMANCE: (&'static str, &'static str) = (
        "marian-mt-en-ROMANCE/spiece",
        "https://huggingface.co/Helsinki-NLP/opus-mt-en-ROMANCE/resolve/main/source.spm",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at https://github.com/Helsinki-NLP/Opus-MT.
    pub const ROMANCE2ENGLISH: (&'static str, &'static str) = (
        "marian-mt-ROMANCE-en/spiece",
        "https://huggingface.co/Helsinki-NLP/opus-mt-ROMANCE-en/resolve/main/source.spm",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at https://github.com/Helsinki-NLP/Opus-MT.
    pub const ENGLISH2GERMAN: (&'static str, &'static str) = (
        "marian-mt-en-de/spiece",
        "https://huggingface.co/Helsinki-NLP/opus-mt-en-de/resolve/main/source.spm",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at https://github.com/Helsinki-NLP/Opus-MT.
    pub const GERMAN2ENGLISH: (&'static str, &'static str) = (
        "marian-mt-de-en/spiece",
        "https://huggingface.co/Helsinki-NLP/opus-mt-de-en/resolve/main/source.spm",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at https://github.com/Helsinki-NLP/Opus-MT.
    pub const ENGLISH2RUSSIAN: (&'static str, &'static str) = (
        "marian-mt-en-ru/spiece",
        "https://huggingface.co/Helsinki-NLP/opus-mt-en-ru/resolve/main/source.spm",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at https://github.com/Helsinki-NLP/Opus-MT.
    pub const RUSSIAN2ENGLISH: (&'static str, &'static str) = (
        "marian-mt-ru-en/spiece",
        "https://huggingface.co/Helsinki-NLP/opus-mt-ru-en/resolve/main/source.spm",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at https://github.com/Helsinki-NLP/Opus-MT.
    pub const FRENCH2GERMAN: (&'static str, &'static str) = (
        "marian-mt-fr-de/spiece",
        "https://huggingface.co/Helsinki-NLP/opus-mt-fr-de/resolve/main/source.spm",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at https://github.com/Helsinki-NLP/Opus-MT.
    pub const GERMAN2FRENCH: (&'static str, &'static str) = (
        "marian-mt-de-fr/spiece",
        "https://huggingface.co/Helsinki-NLP/opus-mt-de-fr/resolve/main/source.spm",
    );
}

impl MarianPrefix {
    pub const ENGLISH2FRENCH: Option<&'static str> = Some(">>fr<< ");
    pub const ENGLISH2CATALAN: Option<&'static str> = Some(">>ca<< ");
    pub const ENGLISH2SPANISH: Option<&'static str> = Some(">>es<< ");
    pub const ENGLISH2PORTUGUESE: Option<&'static str> = Some(">>pt<< ");
    pub const ENGLISH2ITALIAN: Option<&'static str> = Some(">>it<< ");
    pub const ENGLISH2ROMANIAN: Option<&'static str> = Some(">>ro<< ");
    pub const ENGLISH2GERMAN: Option<&'static str> = None;
    pub const ENGLISH2RUSSIAN: Option<&'static str> = None;
    pub const FRENCH2ENGLISH: Option<&'static str> = None;
    pub const CATALAN2ENGLISH: Option<&'static str> = None;
    pub const SPANISH2ENGLISH: Option<&'static str> = None;
    pub const PORTUGUESE2ENGLISH: Option<&'static str> = None;
    pub const ITALIAN2ENGLISH: Option<&'static str> = None;
    pub const ROMANIAN2ENGLISH: Option<&'static str> = None;
    pub const GERMAN2ENGLISH: Option<&'static str> = None;
    pub const RUSSIAN2ENGLISH: Option<&'static str> = None;
    pub const FRENCH2GERMAN: Option<&'static str> = None;
    pub const GERMAN2FRENCH: Option<&'static str> = None;
}

/// # Marian Model for conditional generation
/// Marian model with a vocabulary decoding head
/// It is made of the following blocks:
/// - `base_model`: `BartModel` Base BART model
/// - `linear`: Linear layer with bias tied to the weights of the token id embeddings
pub struct MarianForConditionalGeneration {
    base_model: BartModel,
    final_logits_bias: Tensor,
}

impl MarianForConditionalGeneration {
    /// Build a new `MarianForConditionalGeneration`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the BART model
    /// * `config` - `BartConfig` object defining the model architecture
    /// * `generation_mode` - flag indicating if the model should run in generation mode (a decoder start token must then be provided)
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::bart::{BartConfig, BartForConditionalGeneration};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = BartConfig::from_file(config_path);
    /// let generation_mode = true;
    /// let bart: BartForConditionalGeneration =
    ///     BartForConditionalGeneration::new(&p.root() / "bart", &config, generation_mode);
    /// ```
    pub fn new<'p, P>(
        p: P,
        config: &BartConfig,
        generation_mode: bool,
    ) -> MarianForConditionalGeneration
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let base_model = BartModel::new(p / "model", config, generation_mode);
        let final_logits_bias = p.var(
            "final_logits_bias",
            &[1, config.vocab_size],
            Init::Const(0.),
        );
        MarianForConditionalGeneration {
            base_model,
            final_logits_bias,
        }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *source_sequence_length*). Must be provided when not running in generation mode
    /// * `attention_mask` - Optional attention mask of shape (*batch size*, *source_sequence_length*) for the encoder positions. Positions with a mask with value 0 will be masked.
    /// * `encoder_outputs` - Optional tuple made of a tensor of shape (*batch size*, *source_sequence_length*, *encoder_hidden_dim*) and optional vectors of tensors of length *num_encoder_layers* with shape (*batch size*, *source_sequence_length*, *hidden_size*).
    /// These correspond to the encoder last hidden state and optional hidden states/attention weights for encoder layers. When provided, the encoder hidden state will not be recalculated. Useful for generation tasks.
    /// * `decoder_input_ids` - Optional input tensor of shape (*batch size*, *target_sequence_length*). Must be provided when running in generation mode (e.g. initialiazed with a BOS token)
    /// * `decoder_attention_mask` - Optional attention mask of shape (*batch size*, *target_sequence_length*) for the decoder positions. Positions with a mask with value 0 will be masked.
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `BartModelOutput` containing:
    ///   - `decoder_output` - `Tensor` of shape (*batch size*, *target_sequence_length*, *vocab_size*) representing the logits for each vocabulary item and position
    ///   - `cache` - `(Option<Tensor>, Option<Vec<&LayerState, &LayerState>>)` of length *n_layer* containing the encoder padding mask and past keys and values for both the self attention and the encoder cross attention of each layer of the decoder.
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
    /// use rust_bert::bart::BartConfig;
    /// use rust_bert::marian::MarianForConditionalGeneration;
    /// # let config_path = Path::new("path/to/config.json");
    /// # let vocab_path = Path::new("path/to/vocab.txt");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = BartConfig::from_file(config_path);
    /// # let mut marian_model = MarianForConditionalGeneration::new(&vs.root(), &config, false);
    /// let (batch_size, source_sequence_length, target_sequence_length) = (64, 128, 56);
    /// let input_tensor = Tensor::rand(&[batch_size, source_sequence_length], (Int64, device));
    /// let target_tensor = Tensor::rand(&[batch_size, target_sequence_length], (Int64, device));
    /// let encoder_attention_mask =
    ///     Tensor::ones(&[batch_size, source_sequence_length], (Int64, device));
    /// let decoder_attention_mask =
    ///     Tensor::ones(&[batch_size, source_sequence_length], (Int64, device));
    ///
    /// let model_output = no_grad(|| {
    ///     marian_model.forward_t(
    ///         Some(&input_tensor),
    ///         Some(&encoder_attention_mask),
    ///         None,
    ///         Some(&target_tensor),
    ///         Some(&decoder_attention_mask),
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
        old_layer_states: Option<Vec<(Option<LayerState>, Option<LayerState>)>>,
        train: bool,
    ) -> BartModelOutput {
        let base_model_output = self.base_model.forward_t(
            input_ids,
            attention_mask,
            decoder_input_ids,
            encoder_outputs,
            decoder_attention_mask,
            old_layer_states,
            train,
        );

        let lm_logits = base_model_output
            .decoder_output
            .linear::<Tensor>(&self.base_model.embeddings.ws, None);
        BartModelOutput {
            decoder_output: lm_logits,
            ..base_model_output
        }
    }

    pub fn encode(&self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> Tensor {
        self.base_model
            .encoder
            .forward_t(
                input_ids,
                attention_mask,
                &self.base_model.embeddings,
                false,
            )
            .hidden_state
    }
}

impl LMHeadModel for MarianForConditionalGeneration {
    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see `input_embeds`)
    /// * `layer_past` - Unused for BART
    /// * `attention_mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `input_embeds` - Unused for BART
    /// * `token_type_ids` - Unused for BART
    /// * `position_ids` - Unused for BART
    /// * `encoder_outputs` - Optional tuple made of a tensor of shape (*batch size*, *source_sequence_length*, *encoder_hidden_dim*) and optional vectors of tensors of length *num_encoder_layers* with shape (*batch size*, *source_sequence_length*, *hidden_size*).
    /// These correspond to the encoder last hidden state and optional hidden states/attention weights for encoder layers. When provided, the encoder hidden state will not be recalculated. Useful for generation tasks.
    /// * `decoder_input_ids` - Optional input tensor of shape (*batch size*, *target_sequence_length*). Must be provided when running in generation mode (e.g. initialiazed with a BOS token)
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    ///
    /// # Returns
    ///
    /// * `LMModelOutput` containing:
    ///   - `lm_logits` - `Tensor` of shape (*batch size*, *sequence_length*, *vocab_size*) representing the logits for each vocab item and position
    ///   - `cache` - `BartCache` made of `Option<Vec<(Option<Vec<&LayerState, &LayerState>>)>>` of length *n_layer* containing the encoder past keys and values for
    ///     both the self attention and the encoder cross attention of each layer of the decoder.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use tch::{nn, Device, Tensor, no_grad};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::{Int64, Double};
    /// use rust_bert::bart::BartConfig;
    /// use rust_bert::marian::MarianForConditionalGeneration;
    /// # let config_path = Path::new("path/to/config.json");
    /// # let vocab_path = Path::new("path/to/vocab.txt");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = BartConfig::from_file(config_path);
    /// # let marian_model = MarianForConditionalGeneration::new(&vs.root(), &config, false);
    /// let (batch_size, source_sequence_length, target_sequence_length) = (64, 128, 56);
    /// let input_tensor = Tensor::rand(&[batch_size, source_sequence_length], (Int64, device));
    /// let target_tensor = Tensor::rand(&[batch_size, target_sequence_length], (Int64, device));
    /// let encoder_attention_mask =
    ///     Tensor::ones(&[batch_size, source_sequence_length], (Int64, device));
    /// let decoder_attention_mask =
    ///     Tensor::ones(&[batch_size, source_sequence_length], (Int64, device));
    ///
    /// let model_output = no_grad(|| {
    ///     marian_model.forward_t(
    ///         Some(&input_tensor),
    ///         Some(&encoder_attention_mask),
    ///         None,
    ///         Some(&target_tensor),
    ///         Some(&decoder_attention_mask),
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
    ) -> Result<LMModelOutput, RustBertError> {
        let base_model_output = match cache {
            Cache::BARTCache(cached_layer_states) => self.base_model.forward_t(
                input_ids.as_ref(),
                attention_mask.as_ref(),
                decoder_input_ids.as_ref(),
                encoder_outputs,
                None,
                cached_layer_states,
                train,
            ),
            Cache::None => self.base_model.forward_t(
                input_ids.as_ref(),
                attention_mask.as_ref(),
                decoder_input_ids.as_ref(),
                encoder_outputs,
                None,
                None,
                train,
            ),
            _ => {
                return Err(RustBertError::ValueError(
                    "Cache not compatible with Marian Model".into(),
                ));
            }
        };

        let lm_logits = base_model_output
            .decoder_output
            .linear::<Tensor>(&self.base_model.embeddings.ws, None)
            + &self.final_logits_bias;
        Ok(LMModelOutput {
            lm_logits,
            cache: Cache::BARTCache(base_model_output.cache),
        })
    }
}
