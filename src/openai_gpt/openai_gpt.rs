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


use tch::{nn, Tensor};
use crate::common::dropout::Dropout;
use crate::Gpt2Config;
use tch::nn::embedding;
use tch::kind::Kind::Int64;
use std::borrow::BorrowMut;
use crate::common::linear::{LinearNoBias, linear_no_bias};
use crate::openai_gpt::transformer::Block;


pub struct OpenAiGptModel {
    tokens_embed: nn::Embedding,
    positions_embed: nn::Embedding,
    drop: Dropout,
    h: Vec<Block>,
    output_hidden_states: bool,
    output_attentions: bool,
}


impl OpenAiGptModel {
    pub fn new(p: &nn::Path, config: &Gpt2Config) -> OpenAiGptModel {
        let tokens_embed = embedding(&(p / "tokens_embed"), config.vocab_size, config.n_embd, Default::default());
        let positions_embed = embedding(&(p / "positions_embed"), config.n_positions, config.n_embd, Default::default());

        let embd_pdrop = match config.embd_pdrop {
            Some(value) => value,
            None => 0.1
        };
        let drop = Dropout::new(embd_pdrop);
        let mut h: Vec<Block> = vec!();
        let h_path = &(p / "h");
        for layer_index in 0..config.n_layer {
            h.push(Block::new(&(h_path / layer_index), config, true));
        };
        let output_attentions = match config.output_attentions {
            Some(value) => value,
            None => false
        };
        let output_hidden_states = match config.output_hidden_states {
            Some(value) => value,
            None => false
        };
        OpenAiGptModel { tokens_embed, positions_embed, drop, h, output_hidden_states, output_attentions }
    }

    pub fn forward_t(&self,
                     input_ids: &Option<Tensor>,
                     attention_mask: &Option<Tensor>,
                     token_type_ids: &Option<Tensor>,
                     position_ids: &Option<Tensor>,
                     input_embeds: &Option<Tensor>,
                     train: bool) -> Result<(Tensor, Option<Vec<Tensor>>, Option<Vec<Tensor>>), &'static str> {
        let (input_embeddings, seq_length) = match input_ids {
            Some(input_value) => match input_embeds {
                Some(_) => { return Err("Only one of input ids or input embeddings may be set"); }
                None => (input_value.apply(&self.tokens_embed), *input_value.size().last().unwrap())
            }
            None => match input_embeds {
                Some(embeds) => (embeds.copy(), embeds.size()[1]),
                None => { return Err("At least one of input ids or input embeddings must be set"); }
            }
        };

        let position_ids = match position_ids {
            Some(value) => value.copy(),
            None => Tensor::arange(seq_length, (Int64, input_embeddings.device())).unsqueeze(0)
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

        let position_embeds = position_ids.apply(&self.positions_embed);
        let token_type_embeds = match token_type_ids {
            Some(value) => value.apply(&self.tokens_embed),
            None => Tensor::zeros_like(&position_embeds)
        };
        let mut hidden_state: Tensor = (input_embeddings + position_embeds + token_type_embeds).apply_t(&self.drop, train);
        let mut all_hidden_states: Option<Vec<Tensor>> = if self.output_hidden_states { Some(vec!()) } else { None };
        let mut all_attentions: Option<Vec<Tensor>> = if self.output_attentions { Some(vec!()) } else { None };

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
                None => break
            };
        };

        Ok((hidden_state, all_hidden_states, all_attentions))
    }
}


pub struct OpenAIGPTLMHeadModel {
    transformer: OpenAiGptModel,
    lm_head: LinearNoBias,
}

impl OpenAIGPTLMHeadModel {
    pub fn new(p: &nn::Path, config: &Gpt2Config) -> OpenAIGPTLMHeadModel {
        let transformer = OpenAiGptModel::new(&p, config);
        let lm_head = linear_no_bias(&(p / "lm_head"), config.n_embd, config.vocab_size, Default::default());
        OpenAIGPTLMHeadModel { transformer, lm_head }
    }

    pub fn forward_t(&self,
                     input_ids: &Option<Tensor>,
                     attention_mask: &Option<Tensor>,
                     token_type_ids: &Option<Tensor>,
                     position_ids: &Option<Tensor>,
                     input_embeds: &Option<Tensor>,
                     train: bool) -> Result<(Tensor, Option<Vec<Tensor>>, Option<Vec<Tensor>>), &'static str> {
        let (output,
            all_hidden_states,
            all_attentions) = self.transformer.forward_t(input_ids,
                                                         attention_mask,
                                                         token_type_ids,
                                                         position_ids,
                                                         input_embeds,
                                                         train)?;

        let lm_logits = output.apply(&self.lm_head);
        Ok((lm_logits, all_hidden_states, all_attentions))
    }
}