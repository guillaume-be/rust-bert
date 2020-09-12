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
use crate::gpt2::Gpt2Config;
use crate::openai_gpt::transformer::Block;
use crate::pipelines::generation::{Cache, LMHeadModel};
use std::borrow::{Borrow, BorrowMut};
use tch::kind::Kind::Int64;
use tch::nn::embedding;
use tch::{nn, Tensor};

/// # GPT Pretrained model weight files
pub struct OpenAiGptModelResources;

/// # GPT Pretrained model config files
pub struct OpenAiGptConfigResources;

/// # GPT Pretrained model vocab files
pub struct OpenAiGptVocabResources;

/// # GPT Pretrained model merges files
pub struct OpenAiGptMergesResources;

impl OpenAiGptModelResources {
    /// Shared under MIT license by the OpenAI team at https://github.com/openai/finetune-transformer-lm. Modified with conversion to C-array format.
    pub const GPT: (&'static str, &'static str) = (
        "openai-gpt/model",
        "https://cdn.huggingface.co/openai-gpt-rust_model.ot",
    );
}

impl OpenAiGptConfigResources {
    /// Shared under MIT license by the OpenAI team at https://github.com/openai/finetune-transformer-lm. Modified with conversion to C-array format.
    pub const GPT: (&'static str, &'static str) = (
        "openai-gpt/config",
        "https://cdn.huggingface.co/openai-gpt-config.json",
    );
}

impl OpenAiGptVocabResources {
    /// Shared under MIT license by the OpenAI team at https://github.com/openai/finetune-transformer-lm. Modified with conversion to C-array format.
    pub const GPT: (&'static str, &'static str) = (
        "openai-gpt/vocab",
        "https://cdn.huggingface.co/openai-gpt-vocab.json",
    );
}

impl OpenAiGptMergesResources {
    /// Shared under MIT license by the OpenAI team at https://github.com/openai/finetune-transformer-lm. Modified with conversion to C-array format.
    pub const GPT: (&'static str, &'static str) = (
        "openai-gpt/merges",
        "https://cdn.huggingface.co/openai-gpt-merges.txt",
    );
}

/// # GPT Base model
/// Base architecture for GPT model. Usually complemented with a task-specific head, such as a language model head. As opposed to GPT2, GPT does not give the possibility to re-use past activations as an input.
/// It is made of the following blocks:
/// - `tokens_embed`: `token` embeddings
/// - `positions_embed`: `position` embeddings
/// - `h`: Encoder (transformer) made of a vector of layers. Each layer is made of a multi-head attention layer, layer-normalization layers and a MLP made of linear layers.
/// - `output_hidden_states`: flag indicating if the model should return all hidden states (as opposed to only the last layer)
/// - `output_attentions`: flag indicating if the model should return activation weights
pub struct OpenAiGptModel {
    tokens_embed: nn::Embedding,
    positions_embed: nn::Embedding,
    drop: Dropout,
    h: Vec<Block>,
    output_hidden_states: bool,
    output_attentions: bool,
}

impl OpenAiGptModel {
    /// Build a new `OpenAiGptModel`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the GPT model
    /// * `config` - `Gpt2Config` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::gpt2::Gpt2Config;
    /// use rust_bert::openai_gpt::OpenAiGptModel;
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = Gpt2Config::from_file(config_path);
    /// let gpt2: OpenAiGptModel = OpenAiGptModel::new(&p.root() / "gpt", &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &Gpt2Config) -> OpenAiGptModel
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let tokens_embed = embedding(
            p / "tokens_embed",
            config.vocab_size,
            config.n_embd,
            Default::default(),
        );
        let positions_embed = embedding(
            p / "positions_embed",
            config.n_positions,
            config.n_embd,
            Default::default(),
        );

        let embd_pdrop = match config.embd_pdrop {
            Some(value) => value,
            None => 0.1,
        };
        let drop = Dropout::new(embd_pdrop);
        let mut h: Vec<Block> = vec![];
        let h_path = p / "h";
        for layer_index in 0..config.n_layer {
            h.push(Block::new(&h_path / layer_index, config, true));
        }
        let output_attentions = match config.output_attentions {
            Some(value) => value,
            None => false,
        };
        let output_hidden_states = match config.output_hidden_states {
            Some(value) => value,
            None => false,
        };
        OpenAiGptModel {
            tokens_embed,
            positions_embed,
            drop,
            h,
            output_hidden_states,
            output_attentions,
        }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see `input_embeds`)
    /// * `attention_mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `input_embeds` - Optional pre-computed input embeddings of shape (*batch size*, *sequence_length*, *hidden_size*). If None, input ids must be provided (see `input_ids`)
    /// * `token_type_ids` - Optional token type ids used to indicate the portion of the input the token belongs to. If not None, token type embeddings will be added to the token and position embeddings.
    /// * `position_ids` - Optional position ids of shape (*batch size*, *sequence_length*). If None, will be incremented starting from the length of the past input.
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `output` - `Tensor` of shape (*batch size*, *sequence_length*, *hidden_size*) representing the activations of the last hidden state
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
    /// use rust_bert::gpt2::Gpt2Config;
    /// use rust_bert::openai_gpt::OpenAiGptModel;
    /// # let config_path = Path::new("path/to/config.json");
    /// # let vocab_path = Path::new("path/to/vocab.txt");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = Gpt2Config::from_file(config_path);
    /// # let gpt_model: OpenAiGptModel = OpenAiGptModel::new(&vs.root(), &config);
    /// let (batch_size, sequence_length, past_sequence_length) = (64, 128, 56);
    /// let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    /// let attention_mask = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    /// let token_type_ids = Tensor::ones(&[batch_size, sequence_length], (Int64, device));
    /// let position_ids = Tensor::arange(sequence_length, (Int64, device))
    ///     .expand(&[batch_size, sequence_length], true);
    ///
    /// let (output, hidden_states, attentions) = no_grad(|| {
    ///     gpt_model
    ///         .forward_t(
    ///             &Some(input_tensor),
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
        attention_mask: &Option<Tensor>,
        token_type_ids: &Option<Tensor>,
        position_ids: &Option<Tensor>,
        input_embeds: &Option<Tensor>,
        train: bool,
    ) -> Result<(Tensor, Option<Vec<Tensor>>, Option<Vec<Tensor>>), &'static str> {
        let (input_embeddings, seq_length) = match input_ids {
            Some(input_value) => match input_embeds {
                Some(_) => {
                    return Err("Only one of input ids or input embeddings may be set");
                }
                None => (
                    input_value.apply(&self.tokens_embed),
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

        let position_ids = match position_ids {
            Some(value) => value.copy(),
            None => Tensor::arange(seq_length, (Int64, input_embeddings.device())).unsqueeze(0),
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

        let position_embeds = position_ids.apply(&self.positions_embed);
        let token_type_embeds = match token_type_ids {
            Some(value) => value.apply(&self.tokens_embed),
            None => Tensor::zeros_like(&position_embeds),
        };
        let mut hidden_state: Tensor =
            (input_embeddings + position_embeds + token_type_embeds).apply_t(&self.drop, train);
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

        let mut layers = self.h.iter();
        loop {
            match layers.next() {
                Some(layer) => {
                    if let Some(hidden_states) = all_hidden_states.borrow_mut() {
                        hidden_states.push(hidden_state.as_ref().copy());
                    };

                    let temp = layer.forward_t(&hidden_state, &attention_mask, train);
                    hidden_state = temp.0;
                    if let Some(attentions) = all_attentions.borrow_mut() {
                        attentions.push(temp.1.as_ref().unwrap().copy());
                    };
                }
                None => break,
            };
        }

        Ok((hidden_state, all_hidden_states, all_attentions))
    }
}

/// # GPT Language Modeling head
/// GPT model with a decoding head (linear layer without bias). The weights of the linear layer are tied to the word embeddings
/// It is made of the following blocks:
/// - `transformer`: Base Gpt2Model
/// - `lm_head`: Linear layer without bias tied to the weights of the token id embeddings
pub struct OpenAIGPTLMHeadModel {
    transformer: OpenAiGptModel,
    lm_head: LinearNoBias,
}

impl OpenAIGPTLMHeadModel {
    /// Build a new `OpenAIGPTLMHeadModel`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the GPT model
    /// * `config` - `Gpt2Config` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::gpt2::Gpt2Config;
    /// use rust_bert::openai_gpt::OpenAIGPTLMHeadModel;
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = Gpt2Config::from_file(config_path);
    /// let gpt2: OpenAIGPTLMHeadModel = OpenAIGPTLMHeadModel::new(&p.root() / "gpt", &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &Gpt2Config) -> OpenAIGPTLMHeadModel
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let transformer = OpenAiGptModel::new(p, config);
        let lm_head = linear_no_bias(
            p / "lm_head",
            config.n_embd,
            config.vocab_size,
            Default::default(),
        );
        OpenAIGPTLMHeadModel {
            transformer,
            lm_head,
        }
    }
}

impl LMHeadModel for OpenAIGPTLMHeadModel {
    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see `input_embeds`)
    /// * `_layer_past` - Unused for GPT
    /// * `attention_mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `input_embeds` - Optional pre-computed input embeddings of shape (*batch size*, *sequence_length*, *hidden_size*). If None, input ids must be provided (see `input_ids`)
    /// * `token_type_ids` - Optional token type ids used to indicate the portion of the input the token belongs to. If not None, token type embeddings will be added to the token and position embeddings.
    /// * `position_ids` - Optional position ids of shape (*batch size*, *sequence_length*). If None, will be incremented starting from the length of the past input.
    /// * `_encoder_outputs` - Unused for GPT
    /// * `_decoder_input_ids` - Unused for GPT
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    ///
    /// # Returns
    ///
    /// * `output` - `Tensor` of shape (*batch size*, *sequence_length*, *vocab_size*) representing the logits for each vocab item and position
    /// * `encoder_hidden_states` - None
    /// * `past` - None
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
    /// use rust_bert::gpt2::Gpt2Config;
    /// use rust_bert::openai_gpt::OpenAIGPTLMHeadModel;
    /// use rust_bert::pipelines::generation::{LMHeadModel, Cache};
    /// # let config_path = Path::new("path/to/config.json");
    /// # let vocab_path = Path::new("path/to/vocab.txt");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = Gpt2Config::from_file(config_path);
    /// # let mut gpt_model: OpenAIGPTLMHeadModel = OpenAIGPTLMHeadModel::new(&vs.root(), &config);
    ///  let (batch_size, sequence_length, past_sequence_length) = (64, 128, 56);
    ///  let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    ///  let attention_mask = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    ///  let token_type_ids = Tensor::ones(&[batch_size, sequence_length], (Int64, device));
    ///  let position_ids = Tensor::arange(sequence_length, (Int64, device)).expand(&[batch_size, sequence_length], true);
    ///
    ///  let (output, _, _, hidden_states, attentions) = no_grad(|| {
    ///    gpt_model
    ///         .forward_t(&Some(input_tensor),
    ///                    Cache::None,
    ///                    &Some(attention_mask),
    ///                    &Some(token_type_ids),
    ///                    &Some(position_ids),
    ///                    &None,
    ///                    None,
    ///                    &None,
    ///                    false).unwrap()
    ///    });
    /// ```
    fn forward_t(
        &self,
        input_ids: &Option<Tensor>,
        _layer_past: Cache,
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
        let (output, all_hidden_states, all_attentions) = self.transformer.forward_t(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            input_embeds,
            train,
        )?;

        let lm_logits = output.apply(&self.lm_head);
        Ok((
            lm_logits,
            None,
            Cache::None,
            all_hidden_states,
            all_attentions,
        ))
    }
}
