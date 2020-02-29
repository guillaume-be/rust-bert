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

use crate::common::config::Config;
use serde::{Deserialize, Serialize};
use tch::{nn, Tensor};
use crate::common::dropout::Dropout;
use tch::nn::embedding;
use crate::gpt2::transformer::Block;
use tch::kind::Kind::Int64;
use std::borrow::BorrowMut;
use crate::common::linear::{LinearNoBias, linear_no_bias};

#[derive(Debug, Serialize, Deserialize)]
pub struct Gpt2Config {
    pub attn_pdrop: Option<f64>,
    pub embd_pdrop: Option<f64>,
    pub hidden_dropout_prob: Option<f64>,
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
    pub fn new(p: &nn::Path, config: &Gpt2Config) -> Gpt2Model {
        let p = &(p / "transformer");
        let wte = embedding(&(p / "wte"), config.vocab_size, config.n_embd, Default::default());
        let wpe = embedding(&(p / "wpe"), config.n_positions, config.n_embd, Default::default());

        let embd_pdrop = match config.embd_pdrop {
            Some(value) => value,
            None => 0.1
        };
        let drop = Dropout::new(embd_pdrop);
        let layer_norm_config = nn::LayerNormConfig { eps: config.layer_norm_epsilon, ..Default::default() };
        let ln_f = nn::layer_norm(p / "ln_f", vec![config.n_embd], layer_norm_config);
        let mut h: Vec<Block> = vec!();
        let h_path = &(p / "h");
        for layer_index in 0..config.n_layer {
            h.push(Block::new(&(h_path / layer_index), config, true));
        };
        let output_attentions = match config.output_attentions {
            Some(value) => value,
            None => false
        };
        let output_past = match config.output_past {
            Some(value) => value,
            None => true
        };
        let output_hidden_states = match config.output_hidden_states {
            Some(value) => value,
            None => false
        };
        Gpt2Model { wte, wpe, drop, ln_f, h, output_past, output_hidden_states, output_attentions }
    }

    pub fn forward_t(&self,
                     input_ids: &Option<Tensor>,
                     layer_past: &Option<Vec<Tensor>>,
                     attention_mask: &Option<Tensor>,
                     token_type_ids: &Option<Tensor>,
                     position_ids: &Option<Tensor>,
                     input_embeds: &Option<Tensor>,
                     train: bool) -> Result<(Tensor, Option<Vec<Tensor>>, Option<Vec<Tensor>>, Option<Vec<Tensor>>), &'static str> {
        let (input_embeddings, seq_length) = match input_ids {
            Some(input_value) => match input_embeds {
                Some(_) => { return Err("Only one of input ids or input embeddings may be set"); }
                None => (input_value.apply(&self.wte), *input_value.size().last().unwrap())
            }
            None => match input_embeds {
                Some(embeds) => (embeds.copy(), embeds.size()[1]),
                None => { return Err("At least one of input ids or input embeddings must be set"); }
            }
        };

        let (layer_past, layer_past_length) = match layer_past {
            Some(value) => {
                assert_eq!(value.len(), self.h.len(), "Past activations vector must be of length equal to the number of layers");
                (value.iter().map(|v| Some(v.copy())).collect::<Vec<Option<Tensor>>>(), value[0].size()[3])
            }
            None => {
                let mut out = Vec::with_capacity(self.h.len());
                out.resize_with(self.h.len(), || None::<Tensor>);
                (out, 0)
            }
        };

        let position_ids = match position_ids {
            Some(value) => value.copy(),
            None => Tensor::arange1(layer_past_length, seq_length + layer_past_length, (Int64, input_embeddings.device())).unsqueeze(0)
        };

        let attention_mask: Option<Tensor> = match attention_mask {
            Some(value) => {
                Some(
                    (value
                        .view((input_embeddings.size()[0], -1))
                        .unsqueeze(1)
                        .unsqueeze(2)
                        - 1.0
                    ) * 10000.0)
            }
            None => None
        };

        let position_embeds = position_ids.apply(&self.wpe);
        let token_type_embeds = match token_type_ids {
            Some(value) => value.apply(&self.wte),
            None => Tensor::zeros_like(&position_embeds)
        };
        let mut hidden_state: Tensor = (input_embeddings + position_embeds + token_type_embeds).apply_t(&self.drop, train);
        let mut all_presents: Option<Vec<Tensor>> = if self.output_past { Some(vec!()) } else { None };
        let mut all_hidden_states: Option<Vec<Tensor>> = if self.output_hidden_states { Some(vec!()) } else { None };
        let mut all_attentions: Option<Vec<Tensor>> = if self.output_attentions { Some(vec!()) } else { None };

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
                None => break
            };
        };

        Ok((hidden_state.apply(&self.ln_f), all_presents, all_hidden_states, all_attentions))
    }
}


pub struct GPT2LMHeadModel {
    transformer: Gpt2Model,
    lm_head: LinearNoBias,
}

impl GPT2LMHeadModel {
    pub fn new(p: &nn::Path, config: &Gpt2Config) -> GPT2LMHeadModel {
        let transformer = Gpt2Model::new(&p, config);
        let lm_head = linear_no_bias(&(p / "lm_head"), config.n_embd, config.vocab_size, Default::default());
        GPT2LMHeadModel { transformer, lm_head }
    }

    pub fn forward_t(&self,
                     input_ids: &Option<Tensor>,
                     layer_past: &Option<Vec<Tensor>>,
                     attention_mask: &Option<Tensor>,
                     token_type_ids: &Option<Tensor>,
                     position_ids: &Option<Tensor>,
                     input_embeds: &Option<Tensor>,
                     train: bool) -> Result<(Tensor, Option<Vec<Tensor>>, Option<Vec<Tensor>>, Option<Vec<Tensor>>), &'static str> {
        let (output,
            past,
            all_hidden_states,
            all_attentions) = self.transformer.forward_t(input_ids,
                                                         layer_past,
                                                         attention_mask,
                                                         token_type_ids,
                                                         position_ids,
                                                         input_embeds,
                                                         train)?;

        let lm_logits = output.apply(&self.lm_head);
        Ok((lm_logits, past, all_hidden_states, all_attentions))
    }
}
