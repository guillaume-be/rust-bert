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

use crate::bart::{BartModel, BartConfig};
use tch::{Tensor, nn};
use std::borrow::BorrowMut;
use crate::pipelines::generation::LMHeadModel;
use tch::nn::Init;

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
    pub const ENGLISH2ROMANCE: (&'static str, &'static str) = ("marian-mt-en-ROMANCE/model.ot", "https://cdn.huggingface.co/Helsinki-NLP/opus-mt-en-ROMANCE/rust_model.ot");
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at https://github.com/Helsinki-NLP/Opus-MT. Modified with conversion to C-array format.
    pub const ROMANCE2ENGLISH: (&'static str, &'static str) = ("marian-mt-ROMANCE-en/model.ot", "https://cdn.huggingface.co/Helsinki-NLP/opus-mt-ROMANCE-en/rust_model.ot");
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at https://github.com/Helsinki-NLP/Opus-MT. Modified with conversion to C-array format.
    pub const ENGLISH2GERMAN: (&'static str, &'static str) = ("marian-mt-en-de/model.ot", "https://cdn.huggingface.co/Helsinki-NLP/opus-mt-en-de/rust_model.ot");
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at https://github.com/Helsinki-NLP/Opus-MT. Modified with conversion to C-array format.
    pub const GERMAN2ENGLISH: (&'static str, &'static str) = ("marian-mt-de-en/model.ot", "https://cdn.huggingface.co/Helsinki-NLP/opus-mt-de-en/rust_model.ot");
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at https://github.com/Helsinki-NLP/Opus-MT. Modified with conversion to C-array format.
    pub const FRENCH2GERMAN: (&'static str, &'static str) = ("marian-mt-fr-de/model.ot", "https://cdn.huggingface.co/Helsinki-NLP/opus-mt-fr-de/rust_model.ot");
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at https://github.com/Helsinki-NLP/Opus-MT. Modified with conversion to C-array format.
    pub const GERMAN2FRENCH: (&'static str, &'static str) = ("marian-mt-de-fr/model.ot", "https://cdn.huggingface.co/Helsinki-NLP/opus-mt-de-fr/rust_model.ot");
}

impl MarianConfigResources {
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at https://github.com/Helsinki-NLP/Opus-MT.
    pub const ENGLISH2ROMANCE: (&'static str, &'static str) = ("marian-mt-en-ROMANCE/config.json", "https://cdn.huggingface.co/Helsinki-NLP/opus-mt-en-ROMANCE/config.json");
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at https://github.com/Helsinki-NLP/Opus-MT.
    pub const ROMANCE2ENGLISH: (&'static str, &'static str) = ("marian-mt-ROMANCE-en/config.json", "https://cdn.huggingface.co/Helsinki-NLP/opus-mt-ROMANCE-en/config.json");
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at https://github.com/Helsinki-NLP/Opus-MT.
    pub const ENGLISH2GERMAN: (&'static str, &'static str) = ("marian-mt-en-de/config.json", "https://cdn.huggingface.co/Helsinki-NLP/opus-mt-en-de/config.json");
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at https://github.com/Helsinki-NLP/Opus-MT.
    pub const GERMAN2ENGLISH: (&'static str, &'static str) = ("marian-mt-de-en/config.json", "https://cdn.huggingface.co/Helsinki-NLP/opus-mt-de-en/config.json");
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at https://github.com/Helsinki-NLP/Opus-MT.
    pub const FRENCH2GERMAN: (&'static str, &'static str) = ("marian-mt-fr-de/config.json", "https://cdn.huggingface.co/Helsinki-NLP/opus-mt-fr-de/config.json");
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at https://github.com/Helsinki-NLP/Opus-MT.
    pub const GERMAN2FRENCH: (&'static str, &'static str) = ("marian-mt-de-fr/config.json", "https://cdn.huggingface.co/Helsinki-NLP/opus-mt-de-fr/config.json");
}

impl MarianVocabResources {
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at https://github.com/Helsinki-NLP/Opus-MT.
    pub const ENGLISH2ROMANCE: (&'static str, &'static str) = ("marian-mt-en-ROMANCE/vocab.json", "https://cdn.huggingface.co/Helsinki-NLP/opus-mt-en-ROMANCE/vocab.json");
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at https://github.com/Helsinki-NLP/Opus-MT.
    pub const ROMANCE2ENGLISH: (&'static str, &'static str) = ("marian-mt-ROMANCE-en/vocab.json", "https://cdn.huggingface.co/Helsinki-NLP/opus-mt-ROMANCE-en/vocab.json");
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at https://github.com/Helsinki-NLP/Opus-MT.
    pub const ENGLISH2GERMAN: (&'static str, &'static str) = ("marian-mt-en-de/vocab.json", "https://cdn.huggingface.co/Helsinki-NLP/opus-mt-en-de/vocab.json");
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at https://github.com/Helsinki-NLP/Opus-MT.
    pub const GERMAN2ENGLISH: (&'static str, &'static str) = ("marian-mt-de-en/vocab.json", "https://cdn.huggingface.co/Helsinki-NLP/opus-mt-de-en/vocab.json");
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at https://github.com/Helsinki-NLP/Opus-MT.
    pub const FRENCH2GERMAN: (&'static str, &'static str) = ("marian-mt-fr-de/vocab.json", "https://cdn.huggingface.co/Helsinki-NLP/opus-mt-fr-de/vocab.json");
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at https://github.com/Helsinki-NLP/Opus-MT.
    pub const GERMAN2FRENCH: (&'static str, &'static str) = ("marian-mt-de-fr/vocab.json", "https://cdn.huggingface.co/Helsinki-NLP/opus-mt-de-fr/vocab.json");
}

impl MarianSpmResources {
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at https://github.com/Helsinki-NLP/Opus-MT.
    pub const ENGLISH2ROMANCE: (&'static str, &'static str) = ("marian-mt-en-ROMANCE/spiece.model", "https://cdn.huggingface.co/Helsinki-NLP/opus-mt-en-ROMANCE/source.spm");
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at https://github.com/Helsinki-NLP/Opus-MT.
    pub const ROMANCE2ENGLISH: (&'static str, &'static str) = ("marian-mt-ROMANCE-en/spiece.model", "https://cdn.huggingface.co/Helsinki-NLP/opus-mt-ROMANCE-en/source.spm");
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at https://github.com/Helsinki-NLP/Opus-MT.
    pub const ENGLISH2GERMAN: (&'static str, &'static str) = ("marian-mt-en-de/spiece.model", "https://cdn.huggingface.co/Helsinki-NLP/opus-mt-en-de/source.spm");
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at https://github.com/Helsinki-NLP/Opus-MT.
    pub const GERMAN2ENGLISH: (&'static str, &'static str) = ("marian-mt-de-en/spiece.model", "https://cdn.huggingface.co/Helsinki-NLP/opus-mt-de-en/source.spm");
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at https://github.com/Helsinki-NLP/Opus-MT.
    pub const FRENCH2GERMAN: (&'static str, &'static str) = ("marian-mt-fr-de/spiece.model", "https://cdn.huggingface.co/Helsinki-NLP/opus-mt-fr-de/source.spm");
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at https://github.com/Helsinki-NLP/Opus-MT.
    pub const GERMAN2FRENCH: (&'static str, &'static str) = ("marian-mt-de-fr/spiece.model", "https://cdn.huggingface.co/Helsinki-NLP/opus-mt-de-fr/source.spm");
}

impl MarianPrefix {
    pub const ENGLISH2FRENCH: Option<&'static str> = Some(">>fr<<");
    pub const ENGLISH2CATALAN: Option<&'static str> = Some(">>ca<<");
    pub const ENGLISH2SPANISH: Option<&'static str> = Some(">>es<<");
    pub const ENGLISH2PORTUGUESE: Option<&'static str> = Some(">>pt<<");
    pub const ENGLISH2ITALIAN: Option<&'static str> = Some(">>it<<");
    pub const ENGLISH2ROMANIAN: Option<&'static str> = Some(">>ro<<");
    pub const ENGLISH2GERMAN: Option<&'static str> = None;
    pub const FRENCH2ENGLISH: Option<&'static str> = None;
    pub const CATALAN2ENGLISH: Option<&'static str> = None;
    pub const SPANISH2ENGLISH: Option<&'static str> = None;
    pub const PORTUGUESE2ENGLISH: Option<&'static str> = None;
    pub const ITALIAN2ENGLISH: Option<&'static str> = None;
    pub const ROMANIAN2ENGLISH: Option<&'static str> = None;
    pub const GERMAN2ENGLISH: Option<&'static str> = None;
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
    /// use tch::{nn, Device};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use rust_bert::bart::{BartConfig, BartForConditionalGeneration};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = BartConfig::from_file(config_path);
    /// let generation_mode = true;
    /// let bart: BartForConditionalGeneration = BartForConditionalGeneration::new(&(&p.root() / "bart"), &config, generation_mode);
    /// ```
    ///
    pub fn new(p: &nn::Path, config: &BartConfig, generation_mode: bool) -> MarianForConditionalGeneration {
        let base_model = BartModel::new(&(p / "model"), config, generation_mode);
        let final_logits_bias = p.var("final_logits_bias", &[1, config.vocab_size], Init::Const(0.));
        MarianForConditionalGeneration { base_model, final_logits_bias }
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
    /// * `lm_logits` - `Tensor` of shape (*batch size*, *target_sequence_length*, *vocab_size*) representing the logits for each vocab item and position
    /// * `encoder_hidden_states` - `Tensor` of shape (*batch size*, *source_sequence_length*, *hidden_size*) representing the activations of the last encoder hidden state
    /// * `all_encoder_hidden_states` - `Option<Vec<Tensor>>` of length *num_encoder_layers* with shape (*batch size*, *source_sequence_length*, *hidden_size*)
    /// * `all_encoder_attentions` - `Option<Vec<Tensor>>` of length *num_encoder_layers* with shape (*batch size*, *source_sequence_length*, *hidden_size*)
    /// * `all_decoder_hidden_states` - `Option<Vec<Tensor>>` of length *num_decoder_layers* with shape (*batch size*, *target_sequence_length*, *hidden_size*)
    /// * `all_decoder_attentions` - `Option<Vec<Tensor>>` of length *num_decoder_layers* with shape (*batch size*, *target_sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    ///# use tch::{nn, Device, Tensor, no_grad};
    ///# use rust_bert::Config;
    ///# use std::path::Path;
    ///# use tch::kind::Kind::{Int64, Double};
    /// use rust_bert::bart::{BartConfig, BartForConditionalGeneration};
    ///# let config_path = Path::new("path/to/config.json");
    ///# let vocab_path = Path::new("path/to/vocab.txt");
    ///# let device = Device::Cpu;
    ///# let vs = nn::VarStore::new(device);
    ///# let config = BartConfig::from_file(config_path);
    ///# let mut bart_model: BartForConditionalGeneration = BartForConditionalGeneration::new(&vs.root(), &config, false);
    ///  let (batch_size, source_sequence_length, target_sequence_length) = (64, 128, 56);
    ///  let input_tensor = Tensor::rand(&[batch_size, source_sequence_length], (Int64, device));
    ///  let target_tensor = Tensor::rand(&[batch_size, target_sequence_length], (Int64, device));
    ///  let encoder_attention_mask = Tensor::ones(&[batch_size, source_sequence_length], (Int64, device));
    ///  let decoder_attention_mask = Tensor::ones(&[batch_size, source_sequence_length], (Int64, device));
    ///
    ///  let (decoder_output, encoder_hidden_states,
    ///       all_encoder_hidden_states, all_encoder_attentions,
    ///       all_decoder_hidden_states, all_decoder_attentions) = no_grad(|| {
    ///    bart_model
    ///         .forward_t(Some(&input_tensor),
    ///                    Some(&encoder_attention_mask),
    ///                    None,
    ///                    Some(&target_tensor),
    ///                    Some(&decoder_attention_mask),
    ///                    false)
    ///    });
    ///
    /// ```
    ///
    pub fn forward_t(&mut self,
                     input_ids: Option<&Tensor>,
                     attention_mask: Option<&Tensor>,
                     encoder_outputs: Option<(Tensor, Option<Vec<Tensor>>, Option<Vec<Tensor>>)>,
                     decoder_input_ids: Option<&Tensor>,
                     decoder_attention_mask: Option<&Tensor>,
                     train: bool)
                     -> (Tensor, Tensor,
                         Option<Vec<Tensor>>, Option<Vec<Tensor>>,
                         Option<Vec<Tensor>>, Option<Vec<Tensor>>)
    {
        let (decoder_outputs, encoder_hidden_states, _,
            all_decoder_hidden_states, all_decoder_attentions,
            all_encoder_hidden_states, all_encoder_attentions) =
            self.borrow_mut().base_model.forward_t(input_ids, attention_mask, decoder_input_ids, encoder_outputs, decoder_attention_mask, train);

        let lm_logits = decoder_outputs.linear::<Tensor>(&self.base_model.embeddings.ws, None);
        (lm_logits, encoder_hidden_states,
         all_decoder_hidden_states, all_decoder_attentions,
         all_encoder_hidden_states, all_encoder_attentions)
    }

    pub(crate) fn get_base_model(&mut self) -> &mut BartModel { &mut self.base_model }

    pub fn encode(&mut self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> Tensor {
        let (encoder_hidden_states, _, _) = self.base_model.encoder.forward_t(input_ids, attention_mask, &self.base_model.embeddings, false);
        encoder_hidden_states
    }

    /// Resets the decoder cached keys and values. Should be run for every new generation using the model.
    pub fn reset_cache(&mut self) {
        self.get_base_model().reset_cache()
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
    /// * `lm_logits` - `Tensor` of shape (*batch size*, *sequence_length*, *vocab_size*) representing the logits for each vocab item and position
    /// * `past` - None
    /// * `encoder_hidden_states` - `Option<Tensor>` Hidden states for the encoder
    /// * `hidden_states` - None
    /// * `attentions` - None
    ///
    /// # Example
    ///
    /// ```no_run
    ///# use tch::{nn, Device, Tensor, no_grad};
    ///# use rust_bert::Config;
    ///# use std::path::Path;
    ///# use tch::kind::Kind::{Int64, Double};
    /// use rust_bert::gpt2::{Gpt2Config, GPT2LMHeadModel};
    /// use rust_bert::pipelines::generation::LMHeadModel;
    ///# let config_path = Path::new("path/to/config.json");
    ///# let vocab_path = Path::new("path/to/vocab.txt");
    ///# let device = Device::Cpu;
    ///# let vs = nn::VarStore::new(device);
    ///# let config = Gpt2Config::from_file(config_path);
    ///# let mut gpt2_model: GPT2LMHeadModel = GPT2LMHeadModel::new(&vs.root(), &config);
    ///  let (batch_size, sequence_length, past_sequence_length) = (64, 128, 56);
    ///  let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    ///  let mut past: Vec<Tensor> = Vec::with_capacity(config.n_layer as usize);
    ///  for _ in 0..config.n_layer as usize {
    ///    past.push(Tensor::rand(&[2, batch_size, config.n_head, past_sequence_length, config.n_embd / config.n_head], (Double, device)))
    /// }
    ///  let attention_mask = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    ///  let token_type_ids = Tensor::ones(&[batch_size, sequence_length], (Int64, device));
    ///  let position_ids = Tensor::arange(sequence_length, (Int64, device)).expand(&[batch_size, sequence_length], true);
    ///
    ///  let (output, encoder_hidden_states, _, hidden_states, attentions) = no_grad(|| {
    ///    gpt2_model
    ///         .forward_t(&Some(input_tensor),
    ///                    &Some(past),
    ///                    &Some(attention_mask),
    ///                    &Some(token_type_ids),
    ///                    &Some(position_ids),
    ///                    &None,
    ///                    None,
    ///                    &None,
    ///                    false).unwrap()
    ///    });
    ///
    /// ```
    ///
    fn forward_t(&mut self,
                 input_ids: &Option<Tensor>,
                 _layer_past: &Option<Vec<Tensor>>,
                 attention_mask: &Option<Tensor>,
                 _token_type_ids: &Option<Tensor>,
                 _position_ids: &Option<Tensor>,
                 _input_embeds: &Option<Tensor>,
                 encoder_outputs: Option<&Tensor>,
                 decoder_input_ids: &Option<Tensor>,
                 train: bool) -> Result<(Tensor, Option<Tensor>, Option<Vec<Tensor>>, Option<Vec<Tensor>>, Option<Vec<Tensor>>), &'static str> {
        let (decoder_output, encoder_hidden_states, _, _, _, _, _) = self.base_model.forward_t(input_ids.as_ref(),
                                                                                               attention_mask.as_ref(),
                                                                                               decoder_input_ids.as_ref(),
                                                                                               Some((encoder_outputs.as_ref().unwrap().copy(), None, None)),
                                                                                               None,
                                                                                               train);

        let lm_logits = decoder_output.linear::<Tensor>(&self.base_model.embeddings.ws, None) + &self.final_logits_bias;

        Ok((lm_logits, Some(encoder_hidden_states), None, None, None))
    }
}