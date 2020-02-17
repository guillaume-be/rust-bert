// Copyright 2019-present, the HuggingFace Inc. team, The Google AI Language Team and Facebook, Inc.
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

use serde::{Deserialize, Serialize};
use crate::common::config::Config;
use crate::bert::embeddings::BertEmbeddings;
use crate::bert::encoder::{BertEncoder, BertPooler};
use tch::{nn, Tensor, Kind};

#[allow(non_camel_case_types)]
#[derive(Debug, Serialize, Deserialize)]
pub enum Activation {
    gelu,
    relu,
    mish,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BertConfig {
    pub hidden_act: Activation,
    pub attention_probs_dropout_prob: f64,
    pub hidden_dropout_prob: f64,
    pub hidden_size: i64,
    pub initializer_range: f32,
    pub intermediate_size: i64,
    pub max_position_embeddings: i64,
    pub num_attention_heads: i64,
    pub num_hidden_layers: i64,
    pub type_vocab_size: i64,
    pub vocab_size: i64,
    pub output_attentions: Option<bool>,
    pub output_hidden_states: Option<bool>,
    pub is_decoder: Option<bool>,
}

impl Config<BertConfig> for BertConfig {}

pub struct BertModel {
    embeddings: BertEmbeddings,
    encoder: BertEncoder,
    pooler: BertPooler,
}

impl BertModel {
    pub fn new(p: &nn::Path, config: &BertConfig) -> BertModel {
        let p = &(p / "bert");
        let embeddings = BertEmbeddings::new(&(p / "embeddings"), config);
        let encoder = BertEncoder::new(&(p / "encoder"), config);
        let pooler = BertPooler::new(&(p / "pooler"), config);

        BertModel { embeddings, encoder, pooler }
    }

    pub fn forward_t(&self,
                     input_ids: Option<Tensor>,
                     mask: Option<Tensor>,
                     token_type_ids: Option<Tensor>,
                     position_ids: Option<Tensor>,
                     input_embeds: Option<Tensor>,
                     encoder_hidden_states: &Option<Tensor>,
                     _encoder_mask: &Option<Tensor>,
                     train: bool)
                     -> Result<(Tensor, Tensor, Option<Vec<Tensor>>, Option<Vec<Tensor>>), &'static str> {
        let (input_shape, device) = match &input_ids {
            Some(input_value) => match &input_embeds {
                Some(_) => { return Err("Only one of input ids or input embeddings may be set"); }
                None => (input_value.size(), input_value.device())
            }
            None => match &input_embeds {
                Some(embeds) => (vec!(embeds.size()[0], embeds.size()[1]), embeds.device()),
                None => { return Err("Only one of input ids or input embeddings may be set"); }
            }
        };

        let mask = match mask {
            Some(value) => value,
            None => Tensor::ones(&input_shape, (Kind::Int64, device))
        };


//        ToDo: handle decoder case
        let extended_attention_mask = match mask.dim() {
            3 => mask.unsqueeze(1),
            2 => mask.unsqueeze(1).unsqueeze(1),
            _ => { return Err("Invalid attention mask dimension, must be 2 or 3"); }
        };

        let extended_attention_mask: Tensor = (extended_attention_mask.ones_like() - extended_attention_mask) * -10000.0;

//        ToDo: handle decoder case
        let encoder_extended_attention_mask: Option<Tensor> = None;

        let embedding_output = match self.embeddings.forward_t(input_ids, token_type_ids, position_ids, input_embeds, train) {
            Ok(value) => value,
            Err(e) => { return Err(e); }
        };

        let (hidden_state, all_hidden_states, all_attentions) =
            self.encoder.forward_t(&embedding_output,
                                   &Some(extended_attention_mask),
                                   encoder_hidden_states,
                                   &encoder_extended_attention_mask,
                                   train);

        let pooled_output = self.pooler.forward(&hidden_state);

        Ok((hidden_state, pooled_output, all_hidden_states, all_attentions))

    }
}




