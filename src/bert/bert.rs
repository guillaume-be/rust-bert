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
use crate::bert::embeddings::{BertEmbeddings, BertEmbedding};
use crate::bert::encoder::{BertEncoder, BertPooler};
use tch::{nn, Tensor, Kind};
use tch::kind::Kind::Float;
use crate::common::activations::{_gelu, _relu, _mish};
use crate::common::linear::{LinearNoBias, linear_no_bias};
use tch::nn::Init;
use crate::common::dropout::Dropout;
use std::collections::HashMap;

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
    pub id2label: Option<HashMap<i64, String>>,
    pub label2id: Option<HashMap<String, i64>>,
    pub num_labels: Option<i64>,
}

impl Config<BertConfig> for BertConfig {}

pub struct BertModel<T: BertEmbedding> {
    embeddings: T,
    encoder: BertEncoder,
    pooler: BertPooler,
    is_decoder: bool,
}

impl <T: BertEmbedding> BertModel<T> {
    pub fn new(p: &nn::Path, config: &BertConfig) -> BertModel<T> {
        let is_decoder = match config.is_decoder {
            Some(value) => value,
            None => false
        };
        let embeddings = T::new(&(p / "embeddings"), config);
        let encoder = BertEncoder::new(&(p / "encoder"), config);
        let pooler = BertPooler::new(&(p / "pooler"), config);

        BertModel { embeddings, encoder, pooler, is_decoder }
    }

    pub fn forward_t(&self,
                     input_ids: Option<Tensor>,
                     mask: Option<Tensor>,
                     token_type_ids: Option<Tensor>,
                     position_ids: Option<Tensor>,
                     input_embeds: Option<Tensor>,
                     encoder_hidden_states: &Option<Tensor>,
                     encoder_mask: &Option<Tensor>,
                     train: bool)
                     -> Result<(Tensor, Tensor, Option<Vec<Tensor>>, Option<Vec<Tensor>>), &'static str> {
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
            2 => if self.is_decoder {
                let seq_ids = Tensor::arange(input_shape[1], (Float, device));
                let causal_mask = seq_ids.unsqueeze(0).unsqueeze(0).repeat(&vec!(input_shape[0], input_shape[1], 1));
                let causal_mask = causal_mask.le1(&seq_ids.unsqueeze(0).unsqueeze(-1));
                causal_mask * mask.unsqueeze(1).unsqueeze(1)
            } else {
                mask.unsqueeze(1).unsqueeze(1)
            },
            _ => { return Err("Invalid attention mask dimension, must be 2 or 3"); }
        };

        let extended_attention_mask: Tensor = (extended_attention_mask.ones_like() - extended_attention_mask) * -10000.0;

        let encoder_extended_attention_mask: Option<Tensor> = if self.is_decoder & encoder_hidden_states.is_some() {
            let encoder_hidden_states = encoder_hidden_states.as_ref().unwrap();
            let encoder_hidden_states_shape = encoder_hidden_states.size();
            let encoder_mask = match encoder_mask {
                Some(value) => value.copy(),
                None => Tensor::ones(&[encoder_hidden_states_shape[0], encoder_hidden_states_shape[1]], (Kind::Int64, device))
            };
            match encoder_mask.dim() {
                2 => Some(encoder_mask.unsqueeze(1).unsqueeze(1)),
                3 => Some(encoder_mask.unsqueeze(1)),
                _ => { return Err("Invalid encoder attention mask dimension, must be 2 or 3"); }
            }
        } else {
            None
        };

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


pub struct BertPredictionHeadTransform {
    dense: nn::Linear,
    activation: Box<dyn Fn(&Tensor) -> Tensor>,
    layer_norm: nn::LayerNorm,
}

impl BertPredictionHeadTransform {
    pub fn new(p: &nn::Path, config: &BertConfig) -> BertPredictionHeadTransform {
        let dense = nn::linear(p / "dense", config.hidden_size, config.hidden_size, Default::default());
        let activation = Box::new(match &config.hidden_act {
            Activation::gelu => _gelu,
            Activation::relu => _relu,
            Activation::mish => _mish
        });
        let layer_norm_config = nn::LayerNormConfig { eps: 1e-12, ..Default::default() };
        let layer_norm = nn::layer_norm(p / "LayerNorm", vec![config.hidden_size], layer_norm_config);

        BertPredictionHeadTransform { dense, activation, layer_norm }
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Tensor {
        ((&self.activation)(&hidden_states.apply(&self.dense))).apply(&self.layer_norm)
    }
}

pub struct BertLMPredictionHead {
    transform: BertPredictionHeadTransform,
    decoder: LinearNoBias,
    bias: Tensor,
}

impl BertLMPredictionHead {
    pub fn new(p: &nn::Path, config: &BertConfig) -> BertLMPredictionHead {
        let p = &(p / "predictions");
        let transform = BertPredictionHeadTransform::new(&(p / "transform"), config);
        let decoder = linear_no_bias(&(p / "decoder"), config.hidden_size, config.vocab_size, Default::default());
        let bias = p.var("bias", &[config.vocab_size], Init::KaimingUniform);

        BertLMPredictionHead { transform, decoder, bias }
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Tensor {
        self.transform.forward(&hidden_states).apply(&self.decoder) + &self.bias
    }
}

pub struct BertForMaskedLM {
    bert: BertModel<BertEmbeddings>,
    cls: BertLMPredictionHead,
}

impl BertForMaskedLM {
    pub fn new(p: &nn::Path, config: &BertConfig) -> BertForMaskedLM {
        let bert = BertModel::new(&(p / "bert"), config);
        let cls = BertLMPredictionHead::new(&(p / "cls"), config);

        BertForMaskedLM { bert, cls }
    }

    pub fn forward_t(&self,
                     input_ids: Option<Tensor>,
                     mask: Option<Tensor>,
                     token_type_ids: Option<Tensor>,
                     position_ids: Option<Tensor>,
                     input_embeds: Option<Tensor>,
                     encoder_hidden_states: &Option<Tensor>,
                     encoder_mask: &Option<Tensor>,
                     train: bool) -> (Tensor, Option<Vec<Tensor>>, Option<Vec<Tensor>>) {
        let (hidden_state, _, all_hidden_states, all_attentions) = self.bert.forward_t(input_ids, mask, token_type_ids, position_ids,
                                                                                       input_embeds, encoder_hidden_states, encoder_mask, train).unwrap();

        let prediction_scores = self.cls.forward(&hidden_state);
        (prediction_scores, all_hidden_states, all_attentions)
    }
}

pub struct BertForSequenceClassification {
    bert: BertModel<BertEmbeddings>,
    dropout: Dropout,
    classifier: nn::Linear,
}

impl BertForSequenceClassification {
    pub fn new(p: &nn::Path, config: &BertConfig) -> BertForSequenceClassification {
        let bert = BertModel::new(&(p / "bert"), config);
        let dropout = Dropout::new(config.hidden_dropout_prob);
        let num_labels = config.num_labels.expect("num_labels not provided in configuration");
        let classifier = nn::linear(p / "classifier", config.hidden_size, num_labels, Default::default());

        BertForSequenceClassification { bert, dropout, classifier }
    }

    pub fn forward_t(&self,
                     input_ids: Option<Tensor>,
                     mask: Option<Tensor>,
                     token_type_ids: Option<Tensor>,
                     position_ids: Option<Tensor>,
                     input_embeds: Option<Tensor>,
                     train: bool) -> (Tensor, Option<Vec<Tensor>>, Option<Vec<Tensor>>) {
        let (_, pooled_output, all_hidden_states, all_attentions) = self.bert.forward_t(input_ids, mask, token_type_ids, position_ids,
                                                                                        input_embeds, &None, &None, train).unwrap();

        let output = pooled_output.apply_t(&self.dropout, train).apply(&self.classifier);
        (output, all_hidden_states, all_attentions)
    }
}

pub struct BertForMultipleChoice {
    bert: BertModel<BertEmbeddings>,
    dropout: Dropout,
    classifier: nn::Linear,
}

impl BertForMultipleChoice {
    pub fn new(p: &nn::Path, config: &BertConfig) -> BertForMultipleChoice {
        let bert = BertModel::new(&(p / "bert"), config);
        let dropout = Dropout::new(config.hidden_dropout_prob);
        let classifier = nn::linear(p / "classifier", config.hidden_size, 1, Default::default());

        BertForMultipleChoice { bert, dropout, classifier }
    }

    pub fn forward_t(&self,
                     input_ids: Tensor,
                     mask: Option<Tensor>,
                     token_type_ids: Option<Tensor>,
                     position_ids: Option<Tensor>,
                     input_embeds: Option<Tensor>,
                     train: bool) -> (Tensor, Option<Vec<Tensor>>, Option<Vec<Tensor>>) {
        let num_choices = input_ids.size()[1];

        let input_ids = input_ids.view((-1, *input_ids.size().last().unwrap()));
        let mask = match mask {
            Some(value) => Some(value.view((-1, *value.size().last().unwrap()))),
            None => None
        };
        let token_type_ids = match token_type_ids {
            Some(value) => Some(value.view((-1, *value.size().last().unwrap()))),
            None => None
        };
        let position_ids = match position_ids {
            Some(value) => Some(value.view((-1, *value.size().last().unwrap()))),
            None => None
        };


        let (_, pooled_output, all_hidden_states, all_attentions) = self.bert.forward_t(Some(input_ids), mask, token_type_ids, position_ids,
                                                                                        input_embeds, &None, &None, train).unwrap();

        let output = pooled_output.apply_t(&self.dropout, train).apply(&self.classifier).view((-1, num_choices));
        (output, all_hidden_states, all_attentions)
    }
}

pub struct BertForTokenClassification {
    bert: BertModel<BertEmbeddings>,
    dropout: Dropout,
    classifier: nn::Linear,
}

impl BertForTokenClassification {
    pub fn new(p: &nn::Path, config: &BertConfig) -> BertForTokenClassification {
        let bert = BertModel::new(&(p / "bert"), config);
        let dropout = Dropout::new(config.hidden_dropout_prob);
        let num_labels = config.num_labels.expect("num_labels not provided in configuration");
        let classifier = nn::linear(p / "classifier", config.hidden_size, num_labels, Default::default());

        BertForTokenClassification { bert, dropout, classifier }
    }

    pub fn forward_t(&self,
                     input_ids: Option<Tensor>,
                     mask: Option<Tensor>,
                     token_type_ids: Option<Tensor>,
                     position_ids: Option<Tensor>,
                     input_embeds: Option<Tensor>,
                     train: bool) -> (Tensor, Option<Vec<Tensor>>, Option<Vec<Tensor>>) {
        let (hidden_state, _, all_hidden_states, all_attentions) = self.bert.forward_t(input_ids, mask, token_type_ids, position_ids,
                                                                                       input_embeds, &None, &None, train).unwrap();

        let sequence_output = hidden_state.apply_t(&self.dropout, train).apply(&self.classifier);
        (sequence_output, all_hidden_states, all_attentions)
    }
}

pub struct BertForQuestionAnswering {
    bert: BertModel<BertEmbeddings>,
    qa_outputs: nn::Linear,
}

impl BertForQuestionAnswering {
    pub fn new(p: &nn::Path, config: &BertConfig) -> BertForQuestionAnswering {
        let bert = BertModel::new(&(p / "bert"), config);
        let num_labels = config.num_labels.expect("num_labels not provided in configuration");
        let qa_outputs = nn::linear(p / "qa_outputs", config.hidden_size, num_labels, Default::default());

        BertForQuestionAnswering { bert, qa_outputs }
    }

    pub fn forward_t(&self,
                     input_ids: Option<Tensor>,
                     mask: Option<Tensor>,
                     token_type_ids: Option<Tensor>,
                     position_ids: Option<Tensor>,
                     input_embeds: Option<Tensor>,
                     train: bool) -> (Tensor, Tensor, Option<Vec<Tensor>>, Option<Vec<Tensor>>) {
        let (hidden_state, _, all_hidden_states, all_attentions) = self.bert.forward_t(input_ids, mask, token_type_ids, position_ids,
                                                                                       input_embeds, &None, &None, train).unwrap();

        let sequence_output = hidden_state.apply(&self.qa_outputs);
        let logits = sequence_output.split(1, -1);
        let (start_logits, end_logits) = (&logits[0], &logits[1]);
        let start_logits = start_logits.squeeze1(-1);
        let end_logits = end_logits.squeeze1(-1);

        (start_logits, end_logits, all_hidden_states, all_attentions)
    }
}