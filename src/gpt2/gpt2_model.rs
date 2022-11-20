// Copyright 2018-present, the HuggingFace Inc. team
// Copyright 2018-present, The OpenAI Team Authors
// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
// Copyright 2019 Guillaume Becquin
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
use crate::common::dropout::Dropout;
use crate::common::embeddings::process_ids_embeddings_pair;
use crate::gpt2::transformer::Block;
use crate::pipelines::common::{ModelType, TokenizerOption};
use crate::pipelines::generation_utils::private_generation_utils::{
    PreparedInput, PrivateLanguageGenerator,
};
use crate::pipelines::generation_utils::{
    Cache, GenerateConfig, LMHeadModel, LMModelOutput, LanguageGenerator,
};
use crate::{Config, RustBertError};
use rust_tokenizers::tokenizer::Gpt2Tokenizer;
use rust_tokenizers::vocab::Gpt2Vocab;
use serde::{Deserialize, Serialize};
use std::borrow::{Borrow, BorrowMut};
use tch::kind::Kind::Int64;
use tch::nn::embedding;
use tch::{nn, Kind, Tensor};

/// # GPT2 Pretrained model weight files
pub struct Gpt2ModelResources;

/// # GPT2 Pretrained model config files
pub struct Gpt2ConfigResources;

/// # GPT2 Pretrained model vocab files
pub struct Gpt2VocabResources;

/// # GPT2 Pretrained model merges files
pub struct Gpt2MergesResources;

impl Gpt2ModelResources {
    /// Shared under Modified MIT license by the OpenAI team at <https://github.com/openai/gpt-2/blob/master/LICENSE>. Modified with conversion to C-array format.
    pub const GPT2: (&'static str, &'static str) = (
        "gpt2/model",
        "https://huggingface.co/gpt2/resolve/main/rust_model.ot",
    );
    /// Shared under Modified MIT license by the OpenAI team at <https://github.com/openai/gpt-2/blob/master/LICENSE>. Modified with conversion to C-array format.
    pub const GPT2_MEDIUM: (&'static str, &'static str) = (
        "gpt2-medium/model",
        "https://huggingface.co/gpt2-medium/resolve/main/rust_model.ot",
    );
    /// Shared under Modified MIT license by the OpenAI team at <https://github.com/openai/gpt-2/blob/master/LICENSE>. Modified with conversion to C-array format.
    pub const GPT2_LARGE: (&'static str, &'static str) = (
        "gpt2-large/model",
        "https://huggingface.co/gpt2-large/resolve/main/rust_model.ot",
    );
    /// Shared under Modified MIT license by the OpenAI team at <https://github.com/openai/gpt-2/blob/master/LICENSE>. Modified with conversion to C-array format.
    pub const GPT2_XL: (&'static str, &'static str) = (
        "gpt2-xl/model",
        "https://huggingface.co/gpt2-xl/resolve/main/rust_model.ot",
    );
    /// Shared under Apache 2.0 license by the HuggingFace Inc. team at <https://huggingface.co/models>. Modified with conversion to C-array format.
    pub const DISTIL_GPT2: (&'static str, &'static str) = (
        "distilgpt2/model",
        "https://huggingface.co/distilgpt2/resolve/main/rust_model.ot",
    );
    /// Shared under MIT license by the Microsoft team at <https://huggingface.co/microsoft/DialoGPT-medium>. Modified with conversion to C-array format.
    pub const DIALOGPT_MEDIUM: (&'static str, &'static str) = (
        "dialogpt-medium/model",
        "https://huggingface.co/microsoft/DialoGPT-medium/resolve/main/rust_model.ot",
    );
}

impl Gpt2ConfigResources {
    /// Shared under Modified MIT license by the OpenAI team at <https://github.com/openai/gpt-2/blob/master/LICENSE>. Modified with conversion to C-array format.
    pub const GPT2: (&'static str, &'static str) = (
        "gpt2/config",
        "https://huggingface.co/gpt2/resolve/main/config.json",
    );
    /// Shared under Modified MIT license by the OpenAI team at <https://github.com/openai/gpt-2/blob/master/LICENSE>. Modified with conversion to C-array format.
    pub const GPT2_MEDIUM: (&'static str, &'static str) = (
        "gpt2-medium/config",
        "https://huggingface.co/gpt2-medium/resolve/main/config.json",
    );
    /// Shared under Modified MIT license by the OpenAI team at <https://github.com/openai/gpt-2/blob/master/LICENSE>. Modified with conversion to C-array format.
    pub const GPT2_LARGE: (&'static str, &'static str) = (
        "gpt2-large/config",
        "https://huggingface.co/gpt2-large/resolve/main/config.json",
    );
    /// Shared under Modified MIT license by the OpenAI team at <https://github.com/openai/gpt-2/blob/master/LICENSE>. Modified with conversion to C-array format.
    pub const GPT2_XL: (&'static str, &'static str) = (
        "gpt2-xl/config",
        "https://huggingface.co/gpt2-xl/resolve/main/config.json",
    );
    /// Shared under Apache 2.0 license by the HuggingFace Inc. team at <https://huggingface.co/models>. Modified with conversion to C-array format.
    pub const DISTIL_GPT2: (&'static str, &'static str) = (
        "distilgpt2/config",
        "https://huggingface.co/distilgpt2/resolve/main/config.json",
    );
    /// Shared under MIT license by the Microsoft team at <https://huggingface.co/microsoft/DialoGPT-medium>. Modified with conversion to C-array format.
    pub const DIALOGPT_MEDIUM: (&'static str, &'static str) = (
        "dialogpt-medium/config",
        "https://huggingface.co/microsoft/DialoGPT-medium/resolve/main/config.json",
    );
}

impl Gpt2VocabResources {
    /// Shared under Modified MIT license by the OpenAI team at <https://github.com/openai/gpt-2/blob/master/LICENSE>. Modified with conversion to C-array format.
    pub const GPT2: (&'static str, &'static str) = (
        "gpt2/vocab",
        "https://huggingface.co/gpt2/resolve/main/vocab.json",
    );
    /// Shared under Modified MIT license by the OpenAI team at <https://github.com/openai/gpt-2/blob/master/LICENSE>. Modified with conversion to C-array format.
    pub const GPT2_MEDIUM: (&'static str, &'static str) = (
        "gpt2-medium/vocab",
        "https://huggingface.co/gpt2-medium/resolve/main/vocab.json",
    );
    /// Shared under Modified MIT license by the OpenAI team at <https://github.com/openai/gpt-2/blob/master/LICENSE>. Modified with conversion to C-array format.
    pub const GPT2_LARGE: (&'static str, &'static str) = (
        "gpt2-large/vocab",
        "https://huggingface.co/gpt2-large/resolve/main/vocab.json",
    );
    /// Shared under Modified MIT license by the OpenAI team at <https://github.com/openai/gpt-2/blob/master/LICENSE>. Modified with conversion to C-array format.
    pub const GPT2_XL: (&'static str, &'static str) = (
        "gpt2-xl/vocab",
        "https://huggingface.co/gpt2-xl/resolve/main/vocab.json",
    );
    /// Shared under Apache 2.0 license by the HuggingFace Inc. team at <https://huggingface.co/models>. Modified with conversion to C-array format.
    pub const DISTIL_GPT2: (&'static str, &'static str) = (
        "distilgpt2/vocab",
        "https://huggingface.co/distilgpt2/resolve/main/vocab.json",
    );
    /// Shared under MIT license by the Microsoft team at <https://huggingface.co/microsoft/DialoGPT-medium>. Modified with conversion to C-array format.
    pub const DIALOGPT_MEDIUM: (&'static str, &'static str) = (
        "dialogpt-medium/vocab",
        "https://huggingface.co/microsoft/DialoGPT-medium/resolve/main/vocab.json",
    );
}

impl Gpt2MergesResources {
    /// Shared under Modified MIT license by the OpenAI team at <https://github.com/openai/gpt-2/blob/master/LICENSE>. Modified with conversion to C-array format.
    pub const GPT2: (&'static str, &'static str) = (
        "gpt2/merges",
        "https://huggingface.co/gpt2/resolve/main/merges.txt",
    );
    /// Shared under Modified MIT license by the OpenAI team at <https://github.com/openai/gpt-2/blob/master/LICENSE>. Modified with conversion to C-array format.
    pub const GPT2_MEDIUM: (&'static str, &'static str) = (
        "gpt2-medium/merges",
        "https://huggingface.co/gpt2-medium/resolve/main/merges.txt",
    );
    /// Shared under Modified MIT license by the OpenAI team at <https://github.com/openai/gpt-2/blob/master/LICENSE>. Modified with conversion to C-array format.
    pub const GPT2_LARGE: (&'static str, &'static str) = (
        "gpt2-large/merges",
        "https://huggingface.co/gpt2-large/resolve/main/merges.txt",
    );
    /// Shared under Modified MIT license by the OpenAI team at <https://github.com/openai/gpt-2/blob/master/LICENSE>. Modified with conversion to C-array format.
    pub const GPT2_XL: (&'static str, &'static str) = (
        "gpt2-xl/merges",
        "https://huggingface.co/gpt2-xl/resolve/main/merges.txt",
    );
    /// Shared under Apache 2.0 license by the HuggingFace Inc. team at <https://huggingface.co/models>. Modified with conversion to C-array format.
    pub const DISTIL_GPT2: (&'static str, &'static str) = (
        "distilgpt2/merges",
        "https://huggingface.co/distilgpt2/resolve/main/merges.txt",
    );
    /// Shared under MIT license by the Microsoft team at <https://huggingface.co/microsoft/DialoGPT-medium>. Modified with conversion to C-array format.
    pub const DIALOGPT_MEDIUM: (&'static str, &'static str) = (
        "dialogpt-medium/merges",
        "https://huggingface.co/microsoft/DialoGPT-medium/resolve/main/merges.txt",
    );
}

#[derive(Debug, Serialize, Deserialize, Clone)]
/// # GPT2 model configuration
/// Defines the GPT2 model architecture (e.g. number of layers, hidden layer size, vocab size...).
/// Shared between GPT and GPT2 models
pub struct Gpt2Config {
    pub attn_pdrop: Option<f64>,
    pub embd_pdrop: Option<f64>,
    pub hidden_dropout_prob: Option<f64>,
    pub afn: Option<Activation>,
    pub initializer_range: f64,
    pub layer_norm_epsilon: f64,
    pub n_ctx: i64,
    pub n_embd: i64,
    pub n_head: i64,
    pub n_layer: i64,
    pub n_positions: i64,
    pub num_labels: Option<i64>,
    pub output_past: Option<bool>,
    pub output_attentions: Option<bool>,
    pub output_hidden_states: Option<bool>,
    pub resid_pdrop: Option<f64>,
    pub vocab_size: i64,
}

impl Config for Gpt2Config {}

impl Default for Gpt2Config {
    fn default() -> Self {
        Gpt2Config {
            attn_pdrop: Some(0.1),
            embd_pdrop: Some(0.1),
            hidden_dropout_prob: None,
            afn: Some(Activation::gelu_new),
            initializer_range: 0.02,
            layer_norm_epsilon: 1e-5,
            n_ctx: 1024,
            n_embd: 768,
            n_head: 12,
            n_layer: 12,
            n_positions: 0,
            num_labels: None,
            output_past: None,
            output_attentions: None,
            output_hidden_states: None,
            resid_pdrop: Some(0.1),
            vocab_size: 50257,
        }
    }
}

/// # GPT2 Base model
/// Base architecture for GPT2 model. Usually complemented with a task-specific head, such as a language model head.
/// It is made of the following blocks:
/// - `wte`: `token` embeddings
/// - `wpe`: `position` embeddings
/// - `h`: Encoder (transformer) made of a vector of layers. Each layer is made of a multi-head attention layer, layer-normalization layers and a MLP made of linear layers.
/// - `output_past`: flag indicating if the model should return a past state. This can be fed back to the model to improve the quality of text generated.
/// - `output_hidden_states`: flag indicating if the model should return all hidden states (as opposed to only the last layer)
/// - `output_attentions`: flag indicating if the model should return activation weights
pub struct Gpt2Model {
    wte: nn::Embedding,
    wpe: nn::Embedding,
    drop: Dropout,
    ln_f: nn::LayerNorm,
    h: Vec<Block>,
    output_past: bool,
    output_hidden_states: bool,
    output_attentions: bool,
}

impl Gpt2Model {
    /// Build a new `Gpt2Model`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the GPT2 model
    /// * `config` - `Gpt2Config` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::gpt2::{Gpt2Config, Gpt2Model};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = Gpt2Config::from_file(config_path);
    /// let gpt2: Gpt2Model = Gpt2Model::new(&p.root() / "gpt2", &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &Gpt2Config) -> Gpt2Model
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow() / "transformer";

        let wte = embedding(
            &p / "wte",
            config.vocab_size,
            config.n_embd,
            Default::default(),
        );
        let wpe = embedding(
            &p / "wpe",
            config.n_positions,
            config.n_embd,
            Default::default(),
        );

        let embd_pdrop = config.embd_pdrop.unwrap_or(0.1);
        let drop = Dropout::new(embd_pdrop);
        let layer_norm_config = nn::LayerNormConfig {
            eps: config.layer_norm_epsilon,
            ..Default::default()
        };
        let ln_f = nn::layer_norm(&p / "ln_f", vec![config.n_embd], layer_norm_config);
        let mut h: Vec<Block> = vec![];
        let h_path = &p / "h";
        for layer_index in 0..config.n_layer {
            h.push(Block::new(&h_path / layer_index, config, true));
        }
        let output_attentions = config.output_attentions.unwrap_or(false);
        let output_past = config.output_past.unwrap_or(true);
        let output_hidden_states = config.output_hidden_states.unwrap_or(false);

        Gpt2Model {
            wte,
            wpe,
            drop,
            ln_f,
            h,
            output_past,
            output_hidden_states,
            output_attentions,
        }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see `input_embeds`)
    /// * `layer_past` - Optional vector of length *n_layer* containing the past keys and values of each layer of shape (*2*, *batch size*, *number of heads*, *past_sequence_length*, *hidden size per head*). When provided, these are concatenated with the current input keys and values.
    /// * `attention_mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `input_embeds` - Optional pre-computed input embeddings of shape (*batch size*, *sequence_length*, *hidden_size*). If None, input ids must be provided (see `input_ids`)
    /// * `token_type_ids` - Optional token type ids used to indicate the portion of the input the token belongs to. If not None, token type embeddings will be added to the token and position embeddings.
    /// * `position_ids` - Optional position ids of shape (*batch size*, *sequence_length*). If None, will be incremented starting from the length of the past input.
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `Gpt2ModelOutput` containing:
    ///   - `output` - `Tensor` of shape (*batch size*, *sequence_length*, *vocab_size*) representing the activations of the last hidden state
    ///   - `cache` - `Option<Vec<Tensor>>` of length *n_layer* containing the past keys and values of each layer of shape (*2*, *batch size*, *number of heads*, *past_sequence_length*, *hidden size per head*)
    ///   - `all_hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `all_attentions` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use tch::{nn, Device, Tensor, no_grad};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::{Int64, Double};
    /// use rust_bert::gpt2::{Gpt2Config, Gpt2Model};
    /// # let config_path = Path::new("path/to/config.json");
    /// # let vocab_path = Path::new("path/to/vocab.txt");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = Gpt2Config::from_file(config_path);
    /// # let gpt2_model: Gpt2Model = Gpt2Model::new(&vs.root(), &config);
    /// let (batch_size, sequence_length, past_sequence_length) = (64, 128, 56);
    /// let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    /// let mut past: Vec<Tensor> = Vec::with_capacity(config.n_layer as usize);
    /// for _ in 0..config.n_layer as usize {
    ///     past.push(Tensor::rand(
    ///         &[
    ///             2,
    ///             batch_size,
    ///             config.n_head,
    ///             past_sequence_length,
    ///             config.n_embd / config.n_head,
    ///         ],
    ///         (Double, device),
    ///     ))
    /// }
    /// let attention_mask = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    /// let token_type_ids = Tensor::ones(&[batch_size, sequence_length], (Int64, device));
    /// let position_ids = Tensor::arange(sequence_length, (Int64, device))
    ///     .expand(&[batch_size, sequence_length], true);
    ///
    /// let model_output = no_grad(|| {
    ///     gpt2_model
    ///         .forward_t(
    ///             Some(&input_tensor),
    ///             Some(&past),
    ///             Some(&attention_mask),
    ///             Some(&token_type_ids),
    ///             Some(&position_ids),
    ///             None,
    ///             false,
    ///         )
    ///         .unwrap()
    /// });
    /// ```
    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        layer_past: Option<&Vec<Tensor>>,
        attention_mask: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        train: bool,
    ) -> Result<Gpt2ModelOutput, RustBertError> {
        let (calc_input_embeddings, input_size, _) =
            process_ids_embeddings_pair(input_ids, input_embeds, &self.wte)?;
        let input_embeddings =
            input_embeds.unwrap_or_else(|| calc_input_embeddings.as_ref().unwrap());

        let seq_length = input_size[1];

        let (layer_past, layer_past_length) = match layer_past {
            Some(value) => {
                assert_eq!(
                    value.len(),
                    self.h.len(),
                    "Past activations vector must be of length equal to the number of layers"
                );
                (
                    value
                        .iter()
                        .map(|v| Some(v.copy()))
                        .collect::<Vec<Option<Tensor>>>(),
                    value[0].size()[3],
                )
            }
            None => {
                let mut out = Vec::with_capacity(self.h.len());
                out.resize_with(self.h.len(), || None::<Tensor>);
                (out, 0)
            }
        };

        let position_ids = match position_ids {
            Some(value) => value.copy(),
            None => Tensor::arange_start(
                layer_past_length,
                seq_length + layer_past_length,
                (Int64, input_embeddings.device()),
            )
            .unsqueeze(0),
        };

        let attention_mask: Option<Tensor> = attention_mask.map(|value| {
            let attention_mask = value
                .view((input_embeddings.size()[0], -1))
                .unsqueeze(1)
                .unsqueeze(2)
                .to_kind(input_embeddings.kind());

            let attention_mask: Tensor = (1.0 - attention_mask) * (-10000.0);
            attention_mask.to_kind(input_embeddings.kind())
        });

        let position_embeds = position_ids.apply(&self.wpe);
        let token_type_embeds = match token_type_ids {
            Some(value) => value.apply(&self.wte),
            None => Tensor::zeros_like(&position_embeds),
        };
        let mut hidden_state: Tensor =
            (input_embeddings + position_embeds + token_type_embeds).apply_t(&self.drop, train);
        let mut all_presents: Option<Vec<Tensor>> =
            if self.output_past { Some(vec![]) } else { None };
        let mut all_hidden_states: Option<Vec<Tensor>> = if self.output_hidden_states {
            Some(vec![])
        } else {
            None
        };
        let mut all_attentions: Option<Vec<Tensor>> = if self.output_attentions {
            Some(vec![])
        } else {
            None
        };

        let layer_iter = self.h.iter().zip(layer_past);
        for layer_values in layer_iter {
            let (layer, past) = layer_values;
            let temp =
                layer.forward_t(&hidden_state, past.as_ref(), attention_mask.as_ref(), train);
            hidden_state = temp.0;
            if let Some(presents) = all_presents.borrow_mut() {
                presents.push(temp.1);
            };
            if let Some(attentions) = all_attentions.borrow_mut() {
                attentions.push(temp.2.unwrap());
            };
            if let Some(hidden_states) = all_hidden_states.borrow_mut() {
                hidden_states.push(hidden_state.as_ref().copy());
            };
        }

        Ok(Gpt2ModelOutput {
            output: hidden_state.apply(&self.ln_f),
            cache: all_presents,
            all_hidden_states,
            all_attentions,
        })
    }
}

/// # GPT2 Language Modeling head
/// GPT2 model with a decoding head (linear layer without bias). The weights of the linear layer are tied to the word embeddings
/// It is made of the following blocks:
/// - `transformer`: Base Gpt2Model
pub struct GPT2LMHeadModel {
    transformer: Gpt2Model,
}

impl GPT2LMHeadModel {
    /// Build a new `GPT2LMHeadModel`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the GPT2 model
    /// * `config` - `Gpt2Config` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::gpt2::{GPT2LMHeadModel, Gpt2Config};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = Gpt2Config::from_file(config_path);
    /// let gpt2: GPT2LMHeadModel = GPT2LMHeadModel::new(&p.root() / "gpt2", &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &Gpt2Config) -> GPT2LMHeadModel
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let transformer = Gpt2Model::new(p, config);

        GPT2LMHeadModel { transformer }
    }
}

impl LMHeadModel for GPT2LMHeadModel {
    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see `input_embeds`)
    /// * `layer_past` - Optional vector of size *n_layer* containing the past keys and values of each layer of shape (*2*, *batch size*, *number of heads*, *past_sequence_length*, *hidden size per head*). When provided, these are concatenated with the current input keys and values.
    /// * `attention_mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `input_embeds` - Optional pre-computed input embeddings of shape (*batch size*, *sequence_length*, *hidden_size*). If None, input ids must be provided (see `input_ids`)
    /// * `token_type_ids` - Optional token type ids used to indicate the portion of the input the token belongs to. If not None, token type embeddings will be added to the token and position embeddings.
    /// * `position_ids` - Optional position ids of shape (*batch size*, *sequence_length*). If None, will be incremented starting from the length of the past input.
    /// * `_encoder_outputs` - Optional tensor of shape (*batch size*, *source_sequence_length*, *encoder_hidden_dim*). Unused for GPT2
    /// * `_decoder_input_ids` - Optional tensor of shape (*batch size*, *target_sequence_length*). Unused for GPT2
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    ///
    /// # Returns
    ///
    /// * `LMModelOutput` containing:
    ///   - `lm_logits` - `Tensor` of shape (*batch size*, *sequence_length*, *vocab_size*) representing the logits for each vocab item and position
    ///   - `cache` - `Gpt2Cache` made of `Option<Vec<Tensor>>` of length *n_layer* containing the past keys and values of each layer of shape (*2*, *batch size*, *number of heads*, *past_sequence_length*, *hidden size per head*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use tch::{nn, Device, Tensor, no_grad};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::{Int64, Double};
    /// use rust_bert::gpt2::{GPT2LMHeadModel, Gpt2Config};
    /// use rust_bert::pipelines::generation_utils::{Cache, LMHeadModel};
    /// # let config_path = Path::new("path/to/config.json");
    /// # let vocab_path = Path::new("path/to/vocab.txt");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = Gpt2Config::from_file(config_path);
    /// # let mut gpt2_model: GPT2LMHeadModel = GPT2LMHeadModel::new(&vs.root(), &config);
    /// let (batch_size, sequence_length, past_sequence_length) = (64, 128, 56);
    /// let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    /// let mut past: Vec<Tensor> = Vec::with_capacity(config.n_layer as usize);
    /// for _ in 0..config.n_layer as usize {
    ///     past.push(Tensor::rand(
    ///         &[
    ///             2,
    ///             batch_size,
    ///             config.n_head,
    ///             past_sequence_length,
    ///             config.n_embd / config.n_head,
    ///         ],
    ///         (Double, device),
    ///     ))
    /// }
    /// let attention_mask = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    /// let token_type_ids = Tensor::ones(&[batch_size, sequence_length], (Int64, device));
    /// let position_ids = Tensor::arange(sequence_length, (Int64, device))
    ///     .expand(&[batch_size, sequence_length], true);
    ///
    /// let model_output = no_grad(|| {
    ///     gpt2_model
    ///         .forward_t(
    ///             Some(&input_tensor),
    ///             Cache::GPT2Cache(Some(past)),
    ///             Some(&attention_mask),
    ///             Some(&token_type_ids),
    ///             Some(&position_ids),
    ///             None,
    ///             None,
    ///             None,
    ///             false,
    ///         )
    ///         .unwrap()
    /// });
    /// ```
    fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        layer_past: Cache,
        attention_mask: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        _encoder_outputs: Option<&Tensor>,
        _decoder_input_ids: Option<&Tensor>,
        train: bool,
    ) -> Result<LMModelOutput, RustBertError> {
        let base_model_output = match layer_past {
            Cache::GPT2Cache(layer_past) => self.transformer.forward_t(
                input_ids,
                layer_past.as_ref(),
                attention_mask,
                token_type_ids,
                position_ids,
                input_embeds,
                train,
            ),
            Cache::None => self.transformer.forward_t(
                input_ids,
                None,
                attention_mask,
                token_type_ids,
                position_ids,
                input_embeds,
                train,
            ),
            _ => {
                return Err(RustBertError::ValueError(
                    "Cache not compatible with GPT2 Model".into(),
                ));
            }
        }?;

        let lm_logits = base_model_output
            .output
            .linear::<Tensor>(&self.transformer.wte.ws, None);
        Ok(LMModelOutput {
            lm_logits,
            cache: Cache::GPT2Cache(base_model_output.cache),
        })
    }
}

/// Container for the GPT2 model output.
pub struct Gpt2ModelOutput {
    /// Hidden state of the last layer of the decoder, or logits for a custom head
    /// module after the decoder (e.g. vocabulary logits for language modeling tasks)
    pub output: Tensor,
    /// Cached attention layers keys and values if the model is used for generation
    pub cache: Option<Vec<Tensor>>,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all intermediate layers
    pub all_attentions: Option<Vec<Tensor>>,
}

/// # Language generation model based on the GPT2 architecture
pub struct GPT2Generator {
    model: GPT2LMHeadModel,
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

impl GPT2Generator {
    /// Build a new `GPT2Generator`
    ///
    /// # Arguments
    ///
    /// * `generate_config` - `GenerateConfig` object containing the resource references (model, vocabulary, configuration), generation options and device placement (CPU/GPU)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> anyhow::Result<()> {
    /// use rust_bert::gpt2::GPT2Generator;
    /// use rust_bert::pipelines::generation_utils::GenerateConfig;
    ///
    /// let generate_config = GenerateConfig {
    ///     max_length: Some(30),
    ///     do_sample: true,
    ///     num_beams: 5,
    ///     temperature: 1.1,
    ///     num_return_sequences: 3,
    ///     ..Default::default()
    /// };
    /// let gpt2_generator = GPT2Generator::new(generate_config)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(generate_config: GenerateConfig) -> Result<GPT2Generator, RustBertError> {
        let vocab_path = generate_config.vocab_resource.get_local_path()?;
        let merges_path = generate_config
            .merges_resource
            .as_ref()
            .ok_or_else(|| {
                RustBertError::InvalidConfigurationError(
                    "GPT2 expects a merges resources to be provided".to_string(),
                )
            })?
            .get_local_path()?;

        let tokenizer = TokenizerOption::from_file(
            ModelType::GPT2,
            vocab_path.to_str().unwrap(),
            Some(merges_path.to_str().unwrap()),
            false,
            None,
            None,
        )?;

        Self::new_with_tokenizer(generate_config, tokenizer)
    }

    pub fn new_with_tokenizer(
        generate_config: GenerateConfig,
        tokenizer: TokenizerOption,
    ) -> Result<GPT2Generator, RustBertError> {
        let config_path = generate_config.config_resource.get_local_path()?;
        let weights_path = generate_config.model_resource.get_local_path()?;
        let device = generate_config.device;

        generate_config.validate();
        let mut var_store = nn::VarStore::new(device);

        let config = Gpt2Config::from_file(config_path);
        let model = GPT2LMHeadModel::new(&var_store.root(), &config);
        var_store.load(weights_path)?;

        let bos_token_id = tokenizer.get_bos_id();
        let eos_token_ids = tokenizer.get_eos_id().map(|id| vec![id]);
        let pad_token_id = tokenizer.get_pad_id();
        let max_position_embeddings = config.n_positions;
        let is_encoder_decoder = false;
        let vocab_size = config.vocab_size;
        let decoder_start_id = None;

        Ok(GPT2Generator {
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

impl PrivateLanguageGenerator<GPT2LMHeadModel, Gpt2Vocab, Gpt2Tokenizer> for GPT2Generator {
    fn get_model(&self) -> &GPT2LMHeadModel {
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
        let position_ids = (attention_mask.totype(Kind::Int64).cumsum(-1, Kind::Int64) - 1)
            .masked_fill(&attention_mask.eq(0), 1);

        match past {
            Cache::GPT2Cache(past) => {
                if past.is_some() {
                    PreparedInput {
                        prepared_input: Some(input_ids.select(1, -1).unsqueeze(-1)),
                        prepared_attention_mask: Some(attention_mask),
                        prepared_encoder_output: None,
                        prepared_decoder_input: None,
                        prepared_position_ids: Some(position_ids.select(1, -1).unsqueeze(-1)),
                        prepared_past: Cache::GPT2Cache(past),
                    }
                } else {
                    PreparedInput {
                        prepared_input: Some(input_ids),
                        prepared_attention_mask: Some(attention_mask),
                        prepared_encoder_output: None,
                        prepared_decoder_input: None,
                        prepared_position_ids: Some(position_ids),
                        prepared_past: Cache::GPT2Cache(None),
                    }
                }
            }
            Cache::None => PreparedInput {
                prepared_input: Some(input_ids),
                prepared_attention_mask: Some(attention_mask),
                prepared_encoder_output: None,
                prepared_decoder_input: None,
                prepared_position_ids: Some(position_ids),
                prepared_past: Cache::GPT2Cache(None),
            },
            _ => panic!("Cache type incompatible with GPT2"),
        }
    }

    fn reorder_cache(
        &self,
        past: &mut Cache,
        _encoder_outputs: Option<Tensor>,
        beam_indices: &Tensor,
    ) -> Option<Tensor> {
        match past {
            Cache::GPT2Cache(cached_decoder_state) => match cached_decoder_state {
                Some(value) => {
                    for layer_past in value.iter_mut() {
                        *layer_past = layer_past.index_select(1, beam_indices);
                    }
                    None
                }
                None => None,
            },
            Cache::None => None,
            _ => {
                panic!("Invalid cache for GPT2 model");
            }
        }
    }
}

impl LanguageGenerator<GPT2LMHeadModel, Gpt2Vocab, Gpt2Tokenizer> for GPT2Generator {}
