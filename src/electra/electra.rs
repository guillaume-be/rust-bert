// Copyright 2020 The Google Research Authors.
// Copyright 2019-present, the HuggingFace Inc. team
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
use std::collections::HashMap;
use crate::bert::{Activation, BertConfig};
use crate::Config;
use crate::electra::embeddings::ElectraEmbeddings;
use tch::{nn, Tensor, Kind};
use crate::bert::encoder::BertEncoder;
use crate::common::activations::{_gelu, _relu, _mish};

#[derive(Debug, Serialize, Deserialize)]
/// # Electra model configuration
/// Defines the Electra model architecture (e.g. number of layers, hidden layer size, label mapping...)
pub struct ElectraConfig {
    pub hidden_act: Activation,
    pub attention_probs_dropout_prob: f64,
    pub embedding_size: i64,
    pub hidden_dropout_prob: f64,
    pub hidden_size: i64,
    pub initializer_range: f32,
    pub layer_norm_eps: Option<f64>,
    pub intermediate_size: i64,
    pub max_position_embeddings: i64,
    pub num_attention_heads: i64,
    pub num_hidden_layers: i64,
    pub type_vocab_size: i64,
    pub vocab_size: i64,
    pub pad_token_id: i64,
    pub output_past: Option<bool>,
    pub output_attentions: Option<bool>,
    pub output_hidden_states: Option<bool>,
    pub id2label: Option<HashMap<i64, String>>,
    pub label2id: Option<HashMap<String, i64>>,
}

impl Config<ElectraConfig> for ElectraConfig {}

pub struct ElectraModel {
    embeddings: ElectraEmbeddings,
    embeddings_project: Option<nn::Linear>,
    encoder: BertEncoder,
}

impl ElectraModel {
    pub fn new(p: &nn::Path, config: &ElectraConfig) -> ElectraModel {
        let embeddings = ElectraEmbeddings::new(&(p / "embeddings"), config);
        let embeddings_project = if config.embedding_size != config.hidden_size {
            Some(nn::linear(&(p / "embeddings_project"), config.embedding_size, config.hidden_size, Default::default()))
        } else {
            None
        };
        let bert_config = BertConfig {
            hidden_act: config.hidden_act.clone(),
            attention_probs_dropout_prob: config.attention_probs_dropout_prob,
            hidden_dropout_prob: config.hidden_dropout_prob,
            hidden_size: config.hidden_size,
            initializer_range: config.initializer_range,
            intermediate_size: config.intermediate_size,
            max_position_embeddings: config.max_position_embeddings,
            num_attention_heads: config.num_attention_heads,
            num_hidden_layers: config.num_hidden_layers,
            type_vocab_size: config.type_vocab_size,
            vocab_size: config.vocab_size,
            output_attentions: config.output_attentions,
            output_hidden_states: config.output_hidden_states,
            is_decoder: None,
            id2label: config.id2label.clone(),
            label2id: config.label2id.clone(),
        };
        let encoder = BertEncoder::new(&(p / "encoder"), &bert_config);
        ElectraModel { embeddings, embeddings_project, encoder }
    }

    pub fn forward_t(&self,
                     input_ids: Option<Tensor>,
                     mask: Option<Tensor>,
                     token_type_ids: Option<Tensor>,
                     position_ids: Option<Tensor>,
                     input_embeds: Option<Tensor>,
                     train: bool)
                     -> Result<(Tensor, Option<Vec<Tensor>>, Option<Vec<Tensor>>), &'static str> {
        let (input_shape, device) = match &input_ids {
            Some(input_value) => match &input_embeds {
                Some(_) => { return Err("Only one of input ids or input embeddings may be set"); }
                None => (input_value.size(), input_value.device())
            }
            None => match &input_embeds {
                Some(embeds) => (vec!(embeds.size()[0], embeds.size()[1]), embeds.device()),
                None => { return Err("At least one of input ids or input embeddings must be set"); }
            }
        };

        let mask = match mask {
            Some(value) => value,
            None => Tensor::ones(&input_shape, (Kind::Int64, device))
        };

        let extended_attention_mask = match mask.dim() {
            3 => mask.unsqueeze(1),
            2 => mask.unsqueeze(1).unsqueeze(1),
            _ => { return Err("Invalid attention mask dimension, must be 2 or 3"); }
        };

        let hidden_states = match self.embeddings.forward_t(input_ids, token_type_ids, position_ids, input_embeds, train) {
            Ok(value) => value,
            Err(e) => { return Err(e); }
        };

        let hidden_states = match &self.embeddings_project {
            Some(layer) => hidden_states.apply(layer),
            None => hidden_states
        };

        let (hidden_state, all_hidden_states, all_attentions) =
            self.encoder.forward_t(&hidden_states,
                                   &Some(extended_attention_mask),
                                   &None,
                                   &None,
                                   train);

        Ok((hidden_state, all_hidden_states, all_attentions))
    }
}

pub struct ElectraDiscriminatorHead {
    dense: nn::Linear,
    dense_prediction: nn::Linear,
    activation: Box<dyn Fn(&Tensor) -> Tensor>,
}

impl ElectraDiscriminatorHead {
    pub fn new(p: &nn::Path, config: &ElectraConfig) -> ElectraDiscriminatorHead {
        let dense = nn::linear(&(p / "dense"), config.hidden_size, config.hidden_size, Default::default());
        let dense_prediction = nn::linear(&(p / "dense_prediction"), config.hidden_size, 1, Default::default());
        let activation = Box::new(match &config.hidden_act {
            Activation::gelu => _gelu,
            Activation::relu => _relu,
            Activation::mish => _mish
        });
        ElectraDiscriminatorHead { dense, dense_prediction, activation }
    }

    pub fn forward(&self, encoder_hidden_states: &Tensor) -> Tensor {
        let output = encoder_hidden_states.apply(&self.dense);
        let output = (self.activation)(&output);
        output.apply(&self.dense_prediction).squeeze()
    }
}

pub struct ElectraGeneratorHead {
    dense: nn::Linear,
    layer_norm: nn::LayerNorm,
    activation: Box<dyn Fn(&Tensor) -> Tensor>,
}

impl ElectraGeneratorHead {
    pub fn new(p: &nn::Path, config: &ElectraConfig) -> ElectraGeneratorHead {
        let layer_norm = nn::layer_norm(p / "LayerNorm", vec![config.embedding_size], Default::default());
        let dense = nn::linear(&(p / "dense"), config.hidden_size, config.embedding_size, Default::default());
        let activation = Box::new(_gelu);

        ElectraGeneratorHead { layer_norm, dense, activation }
    }

    pub fn forward(&self, encoder_hidden_states: &Tensor) -> Tensor {
        let output = encoder_hidden_states.apply(&self.dense);
        let output = (self.activation)(&output);
        output.apply(&self.layer_norm)
    }
}