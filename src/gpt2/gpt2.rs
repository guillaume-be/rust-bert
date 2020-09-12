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

use crate::common::dropout::Dropout;
use crate::common::linear::{linear_no_bias, LinearNoBias};
use crate::gpt2::transformer::Block;
use crate::pipelines::generation::{Cache, LMHeadModel};
use crate::Config;
use serde::{Deserialize, Serialize};
use std::borrow::{Borrow, BorrowMut};
use tch::kind::Kind::Int64;
use tch::nn::embedding;
use tch::{nn, Tensor};

/// # GPT2 Pretrained model weight files
pub struct Gpt2ModelResources;

/// # GPT2 Pretrained model config files
pub struct Gpt2ConfigResources;

/// # GPT2 Pretrained model vocab files
pub struct Gpt2VocabResources;

/// # GPT2 Pretrained model merges files
pub struct Gpt2MergesResources;

impl Gpt2ModelResources {
    /// Shared under Modified MIT license by the OpenAI team at https://github.com/openai/gpt-2/blob/master/LICENSE. Modified with conversion to C-array format.
    pub const GPT2: (&'static str, &'static str) = (
        "gpt2/model",
        "https://cdn.huggingface.co/gpt2-rust_model.ot",
    );
    /// Shared under Modified MIT license by the OpenAI team at https://github.com/openai/gpt-2/blob/master/LICENSE. Modified with conversion to C-array format.
    pub const GPT2_MEDIUM: (&'static str, &'static str) = (
        "gpt2-medium/model",
        "https://cdn.huggingface.co/gpt2-medium-rust_model.ot",
    );
    /// Shared under Modified MIT license by the OpenAI team at https://github.com/openai/gpt-2/blob/master/LICENSE. Modified with conversion to C-array format.
    pub const GPT2_LARGE: (&'static str, &'static str) = (
        "gpt2-large/model",
        "https://cdn.huggingface.co/gpt2-large-rust_model.ot",
    );
    /// Shared under Modified MIT license by the OpenAI team at https://github.com/openai/gpt-2/blob/master/LICENSE. Modified with conversion to C-array format.
    pub const GPT2_XL: (&'static str, &'static str) = (
        "gpt2-xl/model",
        "https://cdn.huggingface.co/gpt2-xl-rust_model.ot",
    );
    /// Shared under Apache 2.0 license by the HuggingFace Inc. team at https://huggingface.co/models. Modified with conversion to C-array format.
    pub const DISTIL_GPT2: (&'static str, &'static str) = (
        "distilgpt2/model",
        "https://cdn.huggingface.co/distilgpt2-rust_model.ot",
    );
    /// Shared under MIT license by the Microsoft team at https://huggingface.co/microsoft/DialoGPT-medium. Modified with conversion to C-array format.
    pub const DIALOGPT_MEDIUM: (&'static str, &'static str) = (
        "dialogpt-medium/model",
        "https://cdn.huggingface.co/microsoft/DialoGPT-medium/rust_model.ot",
    );
}

impl Gpt2ConfigResources {
    /// Shared under Modified MIT license by the OpenAI team at https://github.com/openai/gpt-2/blob/master/LICENSE. Modified with conversion to C-array format.
    pub const GPT2: (&'static str, &'static str) =
        ("gpt2/config", "https://cdn.huggingface.co/gpt2-config.json");
    /// Shared under Modified MIT license by the OpenAI team at https://github.com/openai/gpt-2/blob/master/LICENSE. Modified with conversion to C-array format.
    pub const GPT2_MEDIUM: (&'static str, &'static str) = (
        "gpt2-medium/config",
        "https://cdn.huggingface.co/gpt2-medium-config.json",
    );
    /// Shared under Modified MIT license by the OpenAI team at https://github.com/openai/gpt-2/blob/master/LICENSE. Modified with conversion to C-array format.
    pub const GPT2_LARGE: (&'static str, &'static str) = (
        "gpt2-large/config",
        "https://cdn.huggingface.co/gpt2-large-config.json",
    );
    /// Shared under Modified MIT license by the OpenAI team at https://github.com/openai/gpt-2/blob/master/LICENSE. Modified with conversion to C-array format.
    pub const GPT2_XL: (&'static str, &'static str) = (
        "gpt2-xl/config",
        "https://cdn.huggingface.co/gpt2-xl-config.json",
    );
    /// Shared under Apache 2.0 license by the HuggingFace Inc. team at https://huggingface.co/models. Modified with conversion to C-array format.
    pub const DISTIL_GPT2: (&'static str, &'static str) = (
        "distilgpt2/config",
        "https://cdn.huggingface.co/distilgpt2-config.json",
    );
    /// Shared under MIT license by the Microsoft team at https://huggingface.co/microsoft/DialoGPT-medium. Modified with conversion to C-array format.
    pub const DIALOGPT_MEDIUM: (&'static str, &'static str) = (
        "dialogpt-medium/config",
        "https://cdn.huggingface.co/microsoft/DialoGPT-medium/config.json",
    );
}

impl Gpt2VocabResources {
    /// Shared under Modified MIT license by the OpenAI team at https://github.com/openai/gpt-2/blob/master/LICENSE. Modified with conversion to C-array format.
    pub const GPT2: (&'static str, &'static str) =
        ("gpt2/vocab", "https://cdn.huggingface.co/gpt2-vocab.json");
    /// Shared under Modified MIT license by the OpenAI team at https://github.com/openai/gpt-2/blob/master/LICENSE. Modified with conversion to C-array format.
    pub const GPT2_MEDIUM: (&'static str, &'static str) = (
        "gpt2-medium/vocab",
        "https://cdn.huggingface.co/gpt2-medium-vocab.json",
    );
    /// Shared under Modified MIT license by the OpenAI team at https://github.com/openai/gpt-2/blob/master/LICENSE. Modified with conversion to C-array format.
    pub const GPT2_LARGE: (&'static str, &'static str) = (
        "gpt2-large/vocab",
        "https://cdn.huggingface.co/gpt2-large-vocab.json",
    );
    /// Shared under Modified MIT license by the OpenAI team at https://github.com/openai/gpt-2/blob/master/LICENSE. Modified with conversion to C-array format.
    pub const GPT2_XL: (&'static str, &'static str) = (
        "gpt2-xl/vocab",
        "https://cdn.huggingface.co/gpt2-xl-vocab.json",
    );
    /// Shared under Apache 2.0 license by the HuggingFace Inc. team at https://huggingface.co/models. Modified with conversion to C-array format.
    pub const DISTIL_GPT2: (&'static str, &'static str) = (
        "distilgpt2/vocab",
        "https://cdn.huggingface.co/distilgpt2-vocab.json",
    );
    /// Shared under MIT license by the Microsoft team at https://huggingface.co/microsoft/DialoGPT-medium. Modified with conversion to C-array format.
    pub const DIALOGPT_MEDIUM: (&'static str, &'static str) = (
        "dialogpt-medium/vocab",
        "https://cdn.huggingface.co/microsoft/DialoGPT-medium/vocab.json",
    );
}

impl Gpt2MergesResources {
    /// Shared under Modified MIT license by the OpenAI team at https://github.com/openai/gpt-2/blob/master/LICENSE. Modified with conversion to C-array format.
    pub const GPT2: (&'static str, &'static str) =
        ("gpt2/merges", "https://cdn.huggingface.co/gpt2-merges.txt");
    /// Shared under Modified MIT license by the OpenAI team at https://github.com/openai/gpt-2/blob/master/LICENSE. Modified with conversion to C-array format.
    pub const GPT2_MEDIUM: (&'static str, &'static str) = (
        "gpt2-medium/merges",
        "https://cdn.huggingface.co/gpt2-medium-merges.txt",
    );
    /// Shared under Modified MIT license by the OpenAI team at https://github.com/openai/gpt-2/blob/master/LICENSE. Modified with conversion to C-array format.
    pub const GPT2_LARGE: (&'static str, &'static str) = (
        "gpt2-large/merges",
        "https://cdn.huggingface.co/gpt2-large-merges.txt",
    );
    /// Shared under Modified MIT license by the OpenAI team at https://github.com/openai/gpt-2/blob/master/LICENSE. Modified with conversion to C-array format.
    pub const GPT2_XL: (&'static str, &'static str) = (
        "gpt2-xl/merges",
        "https://cdn.huggingface.co/gpt2-xl-merges.txt",
    );
    /// Shared under Apache 2.0 license by the HuggingFace Inc. team at https://huggingface.co/models. Modified with conversion to C-array format.
    pub const DISTIL_GPT2: (&'static str, &'static str) = (
        "distilgpt2/merges",
        "https://cdn.huggingface.co/distilgpt2-merges.txt",
    );
    /// Shared under MIT license by the Microsoft team at https://huggingface.co/microsoft/DialoGPT-medium. Modified with conversion to C-array format.
    pub const DIALOGPT_MEDIUM: (&'static str, &'static str) = (
        "dialogpt-medium/merges",
        "https://cdn.huggingface.co/microsoft/DialoGPT-medium/merges.txt",
    );
}

#[allow(non_camel_case_types)]
#[derive(Debug, Serialize, Deserialize)]
/// # Activation function used in the fully connected layers of the transformer block
pub enum GptActivation {
    /// Gaussian Error Linear Unit ([Hendrycks et al., 2016,](https://arxiv.org/abs/1606.08415))
    gelu,
    /// Rectified Linear Unit
    relu,
    /// Swish: a Self-Gated Activation Function ([Ramachandran et al., 2017](https://arxiv.org/pdf/1710.05941v1.pdf))
    swish,
}

#[derive(Debug, Serialize, Deserialize)]
/// # GPT2 model configuration
/// Defines the GPT2 model architecture (e.g. number of layers, hidden layer size, vocab size...).
/// Shared between GPT and GPT2 models
pub struct Gpt2Config {
    pub attn_pdrop: Option<f64>,
    pub embd_pdrop: Option<f64>,
    pub hidden_dropout_prob: Option<f64>,
    pub afn: Option<GptActivation>,
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

impl Config<Gpt2Config> for Gpt2Config {}

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

        let embd_pdrop = match config.embd_pdrop {
            Some(value) => value,
            None => 0.1,
        };
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
        let output_attentions = match config.output_attentions {
            Some(value) => value,
            None => false,
        };
        let output_past = match config.output_past {
            Some(value) => value,
            None => true,
        };
        let output_hidden_states = match config.output_hidden_states {
            Some(value) => value,
            None => false,
        };
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
    /// * `output` - `Tensor` of shape (*batch size*, *sequence_length*, *hidden_size*) representing the activations of the last hidden state
    /// * `past` - `Option<Vec<Tensor>>` of length *n_layer* containing the past keys and values of each layer of shape (*2*, *batch size*, *number of heads*, *past_sequence_length*, *hidden size per head*)
    /// * `hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    /// * `attentions` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
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
    /// let (output, past, hidden_states, attentions) = no_grad(|| {
    ///     gpt2_model
    ///         .forward_t(
    ///             &Some(input_tensor),
    ///             &Some(past),
    ///             &Some(attention_mask),
    ///             &Some(token_type_ids),
    ///             &Some(position_ids),
    ///             &None,
    ///             false,
    ///         )
    ///         .unwrap()
    /// });
    /// ```
    pub fn forward_t(
        &self,
        input_ids: &Option<Tensor>,
        layer_past: &Option<Vec<Tensor>>,
        attention_mask: &Option<Tensor>,
        token_type_ids: &Option<Tensor>,
        position_ids: &Option<Tensor>,
        input_embeds: &Option<Tensor>,
        train: bool,
    ) -> Result<
        (
            Tensor,
            Option<Vec<Tensor>>,
            Option<Vec<Tensor>>,
            Option<Vec<Tensor>>,
        ),
        &'static str,
    > {
        let (input_embeddings, seq_length) = match input_ids {
            Some(input_value) => match input_embeds {
                Some(_) => {
                    return Err("Only one of input ids or input embeddings may be set");
                }
                None => (
                    input_value.apply(&self.wte),
                    *input_value.size().last().unwrap(),
                ),
            },
            None => match input_embeds {
                Some(embeds) => (embeds.copy(), embeds.size()[1]),
                None => {
                    return Err("At least one of input ids or input embeddings must be set");
                }
            },
        };

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
            None => Tensor::arange1(
                layer_past_length,
                seq_length + layer_past_length,
                (Int64, input_embeddings.device()),
            )
            .unsqueeze(0),
        };

        let attention_mask: Option<Tensor> = match attention_mask {
            Some(value) => Some(
                (value
                    .view((input_embeddings.size()[0], -1))
                    .unsqueeze(1)
                    .unsqueeze(2)
                    - 1.0)
                    * 10000.0,
            ),
            None => None,
        };

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

        let mut layer_iter = self.h.iter().zip(layer_past);
        loop {
            match layer_iter.next() {
                Some(layer_values) => {
                    let (layer, past) = layer_values;
                    if let Some(hidden_states) = all_hidden_states.borrow_mut() {
                        hidden_states.push(hidden_state.as_ref().copy());
                    };

                    let temp = layer.forward_t(&hidden_state, &past, &attention_mask, train);
                    hidden_state = temp.0;
                    if let Some(presents) = all_presents.borrow_mut() {
                        presents.push(temp.1.as_ref().copy());
                    };
                    if let Some(attentions) = all_attentions.borrow_mut() {
                        attentions.push(temp.2.as_ref().unwrap().copy());
                    };
                }
                None => break,
            };
        }

        Ok((
            hidden_state.apply(&self.ln_f),
            all_presents,
            all_hidden_states,
            all_attentions,
        ))
    }
}

/// # GPT2 Language Modeling head
/// GPT2 model with a decoding head (linear layer without bias). The weights of the linear layer are tied to the word embeddings
/// It is made of the following blocks:
/// - `transformer`: Base Gpt2Model
/// - `lm_head`: Linear layer without bias tied to the weights of the token id embeddings
pub struct GPT2LMHeadModel {
    transformer: Gpt2Model,
    lm_head: LinearNoBias,
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
        let lm_head = linear_no_bias(
            p / "lm_head",
            config.n_embd,
            config.vocab_size,
            Default::default(),
        );
        GPT2LMHeadModel {
            transformer,
            lm_head,
        }
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
    /// * `output` - `Tensor` of shape (*batch size*, *sequence_length*, *vocab_size*) representing the logits for each vocab item and position
    /// * `encoder_hidden_states` - None
    /// * `past` - `Option<Vec<Tensor>>` of length *n_layer* containing the past keys and values of each layer of shape (*2*, *batch size*, *number of heads*, *past_sequence_length*, *hidden size per head*)
    /// * `hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    /// * `attentions` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use tch::{nn, Device, Tensor, no_grad};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::{Int64, Double};
    /// use rust_bert::gpt2::{GPT2LMHeadModel, Gpt2Config};
    /// use rust_bert::pipelines::generation::{Cache, LMHeadModel};
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
    /// let (output, _, past, hidden_states, attentions) = no_grad(|| {
    ///     gpt2_model
    ///         .forward_t(
    ///             &Some(input_tensor),
    ///             Cache::GPT2Cache(Some(past)),
    ///             &Some(attention_mask),
    ///             &Some(token_type_ids),
    ///             &Some(position_ids),
    ///             &None,
    ///             None,
    ///             &None,
    ///             false,
    ///         )
    ///         .unwrap()
    /// });
    /// ```
    fn forward_t(
        &self,
        input_ids: &Option<Tensor>,
        layer_past: Cache,
        attention_mask: &Option<Tensor>,
        token_type_ids: &Option<Tensor>,
        position_ids: &Option<Tensor>,
        input_embeds: &Option<Tensor>,
        _encoder_outputs: Option<&Tensor>,
        _decoder_input_ids: &Option<Tensor>,
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
        let (output, past, all_hidden_states, all_attentions) = match layer_past {
            Cache::GPT2Cache(layer_past) => Ok(self.transformer.forward_t(
                input_ids,
                &layer_past,
                attention_mask,
                token_type_ids,
                position_ids,
                input_embeds,
                train,
            )?),
            Cache::None => Ok(self.transformer.forward_t(
                input_ids,
                &None,
                attention_mask,
                token_type_ids,
                position_ids,
                input_embeds,
                train,
            )?),
            _ => Err("Cache not compatible with GPT2 model"),
        }?;

        let lm_logits = output.apply(&self.lm_head);
        Ok((
            lm_logits,
            None,
            Cache::GPT2Cache(past),
            all_hidden_states,
            all_attentions,
        ))
    }
}
