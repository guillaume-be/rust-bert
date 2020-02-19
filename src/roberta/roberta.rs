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

use tch::{nn, Tensor};
use crate::{BertConfig, BertModel};
use crate::common::linear::{linear_no_bias, LinearNoBias};
use tch::nn::Init;
use crate::common::activations::_gelu;
use crate::roberta::embeddings::RobertaEmbeddings;
use crate::common::dropout::Dropout;

pub struct RobertaLMHead {
    dense: nn::Linear,
    decoder: LinearNoBias,
    layer_norm: nn::LayerNorm,
    bias: Tensor,
}

impl RobertaLMHead {
    pub fn new(p: &nn::Path, config: &BertConfig) -> RobertaLMHead {
        let dense = nn::linear(p / "dense", config.hidden_size, config.hidden_size, Default::default());
        let layer_norm_config = nn::LayerNormConfig { eps: 1e-12, ..Default::default() };
        let layer_norm = nn::layer_norm(p / "layer_norm", vec![config.hidden_size], layer_norm_config);
        let decoder = linear_no_bias(&(p / "decoder"), config.hidden_size, config.vocab_size, Default::default());
        let bias = p.var("bias", &[config.vocab_size], Init::KaimingUniform);

        RobertaLMHead { dense, decoder, layer_norm, bias }
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Tensor {
        (_gelu(&hidden_states.apply(&self.dense))).apply(&self.layer_norm).apply(&self.decoder) + &self.bias
    }
}

pub struct RobertaForMaskedLM {
    roberta: BertModel<RobertaEmbeddings>,
    lm_head: RobertaLMHead,
}

impl RobertaForMaskedLM {
    pub fn new(p: &nn::Path, config: &BertConfig) -> RobertaForMaskedLM {
        let roberta = BertModel::<RobertaEmbeddings>::new(&(p / "roberta"), config);
        let lm_head = RobertaLMHead::new(&(p / "lm_head"), config);

        RobertaForMaskedLM { roberta, lm_head }
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
        let (hidden_state, _, all_hidden_states, all_attentions) = self.roberta.forward_t(input_ids, mask, token_type_ids, position_ids,
                                                                                          input_embeds, encoder_hidden_states, encoder_mask, train).unwrap();

        let prediction_scores = self.lm_head.forward(&hidden_state);
        (prediction_scores, all_hidden_states, all_attentions)
    }
}

pub struct RobertaClassificationHead {
    dense: nn::Linear,
    dropout: Dropout,
    out_proj: nn::Linear,
}

impl RobertaClassificationHead {
    pub fn new(p: &nn::Path, config: &BertConfig) -> RobertaClassificationHead {
        let dense = nn::linear(p / "dense", config.hidden_size, config.hidden_size, Default::default());
        let num_labels = config.num_labels.expect("num_labels not provided in configuration");
        let out_proj = nn::linear(p / "out_proj", config.hidden_size, num_labels, Default::default());
        let dropout = Dropout::new(config.hidden_dropout_prob);

        RobertaClassificationHead { dense, dropout, out_proj }
    }

    pub fn forward_t(&self, hidden_states: &Tensor, train: bool) -> Tensor {
        hidden_states
            .select(1, 0)
            .apply_t(&self.dropout, train)
            .apply(&self.dense)
            .tanh()
            .apply_t(&self.dropout, train)
            .apply(&self.out_proj)
    }
}

pub struct RobertaForSequenceClassification {
    roberta: BertModel<RobertaEmbeddings>,
    classifier: RobertaClassificationHead,
}

impl RobertaForSequenceClassification {
    pub fn new(p: &nn::Path, config: &BertConfig) -> RobertaForSequenceClassification {
        let roberta = BertModel::<RobertaEmbeddings>::new(&(p / "roberta"), config);
        let classifier = RobertaClassificationHead::new(&(p / "classifier"), config);

        RobertaForSequenceClassification { roberta, classifier }
    }

    pub fn forward_t(&self,
                     input_ids: Option<Tensor>,
                     mask: Option<Tensor>,
                     token_type_ids: Option<Tensor>,
                     position_ids: Option<Tensor>,
                     input_embeds: Option<Tensor>,
                     train: bool) -> (Tensor, Option<Vec<Tensor>>, Option<Vec<Tensor>>) {
        let (hidden_state, _, all_hidden_states, all_attentions) = self.roberta.forward_t(input_ids, mask, token_type_ids, position_ids,
                                                                                          input_embeds, &None, &None, train).unwrap();

        let output = self.classifier.forward_t(&hidden_state, train);
        (output, all_hidden_states, all_attentions)
    }
}

pub struct RobertaForMultipleChoice {
    roberta: BertModel<RobertaEmbeddings>,
    dropout: Dropout,
    classifier: nn::Linear,
}

impl RobertaForMultipleChoice {
    pub fn new(p: &nn::Path, config: &BertConfig) -> RobertaForMultipleChoice {
        let roberta = BertModel::<RobertaEmbeddings>::new(&(p / "roberta"), config);
        let dropout = Dropout::new(config.hidden_dropout_prob);
        let classifier = nn::linear(p / "classifier", config.hidden_size, 1, Default::default());

        RobertaForMultipleChoice { roberta, dropout, classifier }
    }

    pub fn forward_t(&self,
                     input_ids: Tensor,
                     mask: Option<Tensor>,
                     token_type_ids: Option<Tensor>,
                     position_ids: Option<Tensor>,
                     train: bool) -> (Tensor, Option<Vec<Tensor>>, Option<Vec<Tensor>>) {
        let num_choices = input_ids.size()[1];

        let flat_input_ids = Some(input_ids.view((-1i64, *input_ids.size().last().unwrap())));
        let flat_position_ids = match position_ids {
            Some(value) => Some(value.view((-1i64, *value.size().last().unwrap()))),
            None => None
        };
        let flat_token_type_ids = match token_type_ids {
            Some(value) => Some(value.view((-1i64, *value.size().last().unwrap()))),
            None => None
        };
        let flat_mask = match mask {
            Some(value) => Some(value.view((-1i64, *value.size().last().unwrap()))),
            None => None
        };

        let (_, pooled_output, all_hidden_states, all_attentions) = self.roberta.forward_t(flat_input_ids, flat_mask, flat_token_type_ids, flat_position_ids,
                                                                                           None, &None, &None, train).unwrap();

        let output = pooled_output.apply_t(&self.dropout, train).apply(&self.classifier).view((-1, num_choices));
        (output, all_hidden_states, all_attentions)
    }
}