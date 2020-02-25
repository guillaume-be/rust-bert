// Copyright 2019-present, the HuggingFace Inc. team, The Google AI Language Team and Facebook, Inc.
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

extern crate tch;

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use crate::distilbert::embeddings::DistilBertEmbedding;
use crate::distilbert::transformer::Transformer;
use self::tch::{nn, Tensor};
use crate::common::config::Config;
use crate::common::dropout::Dropout;

#[allow(non_camel_case_types)]
#[derive(Debug, Serialize, Deserialize)]
pub enum Activation {
    gelu,
    relu,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DistilBertConfig {
    pub activation: Activation,
    pub attention_dropout: f64,
    pub dim: i64,
    pub dropout: f64,
    pub hidden_dim: i64,
    pub id2label: Option<HashMap<i32, String>>,
    pub initializer_range: f32,
    pub is_decoder: Option<bool>,
    pub label2id: Option<HashMap<String, i32>>,
    pub max_position_embeddings: i64,
    pub n_heads: i64,
    pub n_layers: i64,
    pub num_labels: i64,
    pub output_attentions: bool,
    pub output_hidden_states: bool,
    pub output_past: Option<bool>,
    pub qa_dropout: f64,
    pub seq_classif_dropout: f64,
    pub sinusoidal_pos_embds: bool,
    pub tie_weights_: bool,
    pub torchscript: bool,
    pub use_bfloat16: Option<bool>,
    pub vocab_size: i64,
}

impl Config<DistilBertConfig> for DistilBertConfig {}

pub struct DistilBertModel {
    embeddings: DistilBertEmbedding,
    transformer: Transformer,
}

impl DistilBertModel {
    pub fn new(p: &nn::Path, config: &DistilBertConfig) -> DistilBertModel {
        let p = &(p / "distilbert");
        let embeddings = DistilBertEmbedding::new(&(p / "embeddings"), config);
        let transformer = Transformer::new(&(p / "transformer"), config);
        DistilBertModel { embeddings, transformer }
    }

    pub fn _get_embeddings(&self) -> &nn::Embedding {
        self.embeddings._get_word_embeddings()
    }

    pub fn _set_embeddings(&mut self, new_embeddings: nn::Embedding) {
        &self.embeddings._set_word_embeddings(new_embeddings);
    }

    pub fn forward_t(&self, input: Option<Tensor>, mask: Option<Tensor>, input_embeds: Option<Tensor>, train: bool)
                     -> Result<(Tensor, Option<Vec<Tensor>>, Option<Vec<Tensor>>), &'static str> {
        let input_embeddings = match input {
            Some(input_value) => match input_embeds {
                Some(_) => { return Err("Only one of input ids or input embeddings may be set"); }
                None => input_value.apply_t(&self.embeddings, train)
            }
            None => match input_embeds {
                Some(embeds) => embeds.copy(),
                None => { return Err("Only one of input ids or input embeddings may be set"); }
            }
        };


        let transformer_output = (&self.transformer).forward_t(&input_embeddings, mask, train);
        Ok(transformer_output)
    }
}

pub struct DistilBertModelClassifier {
    distil_bert_model: DistilBertModel,
    pre_classifier: nn::Linear,
    classifier: nn::Linear,
    dropout: Dropout,
}

impl DistilBertModelClassifier {
    pub fn new(p: &nn::Path, config: &DistilBertConfig) -> DistilBertModelClassifier {
        let distil_bert_model = DistilBertModel::new(&p, config);
        let pre_classifier = nn::linear(&(p / "pre_classifier"), config.dim, config.dim, Default::default());
        let classifier = nn::linear(&(p / "classifier"), config.dim, config.num_labels, Default::default());
        let dropout = Dropout::new(config.seq_classif_dropout);

        DistilBertModelClassifier { distil_bert_model, pre_classifier, classifier, dropout }
    }

    pub fn forward_t(&self, input: Option<Tensor>, mask: Option<Tensor>, input_embeds: Option<Tensor>, train: bool)
                     -> Result<(Tensor, Option<Vec<Tensor>>, Option<Vec<Tensor>>), &'static str> {
        let (output, all_hidden_states, all_attentions) = match self.distil_bert_model.forward_t(input, mask, input_embeds, train) {
            Ok(value) => value,
            Err(err) => return Err(err)
        };

        let output = output
            .select(1, 0)
            .apply(&self.pre_classifier)
            .relu()
            .apply_t(&self.dropout, train)
            .apply(&self.classifier);

        Ok((output, all_hidden_states, all_attentions))
    }
}

pub struct DistilBertModelMaskedLM {
    distil_bert_model: DistilBertModel,
    vocab_transform: nn::Linear,
    vocab_layer_norm: nn::LayerNorm,
    vocab_projector: nn::Linear,
}

impl DistilBertModelMaskedLM {
    pub fn new(p: &nn::Path, config: &DistilBertConfig) -> DistilBertModelMaskedLM {
        let distil_bert_model = DistilBertModel::new(&p, config);
        let vocab_transform = nn::linear(&(p / "vocab_transform"), config.dim, config.dim, Default::default());
        let layer_norm_config = nn::LayerNormConfig { eps: 1e-12, ..Default::default() };
        let vocab_layer_norm = nn::layer_norm(p / "vocab_layer_norm", vec![config.dim], layer_norm_config);
        let vocab_projector = nn::linear(&(p / "vocab_projector"), config.dim, config.vocab_size, Default::default());

        DistilBertModelMaskedLM { distil_bert_model, vocab_transform, vocab_layer_norm, vocab_projector }
    }

    pub fn forward_t(&self, input: Option<Tensor>, mask: Option<Tensor>, input_embeds: Option<Tensor>, train: bool)
                     -> Result<(Tensor, Option<Vec<Tensor>>, Option<Vec<Tensor>>), &'static str> {
        let (output, all_hidden_states, all_attentions) = match self.distil_bert_model.forward_t(input, mask, input_embeds, train) {
            Ok(value) => value,
            Err(err) => return Err(err)
        };

        let output = output
            .apply(&self.vocab_transform)
            .gelu()
            .apply(&self.vocab_layer_norm)
            .apply(&self.vocab_projector);

        Ok((output, all_hidden_states, all_attentions))
    }
}


pub struct DistilBertForQuestionAnswering {
    distil_bert_model: DistilBertModel,
    qa_outputs: nn::Linear,
    dropout: Dropout,
}

impl DistilBertForQuestionAnswering {
    pub fn new(p: &nn::Path, config: &DistilBertConfig) -> DistilBertForQuestionAnswering {
        let distil_bert_model = DistilBertModel::new(&p, config);
        let qa_outputs = nn::linear(&(p / "qa_outputs"), config.dim, config.num_labels, Default::default());
        assert_eq!(config.num_labels, 2, "num_labels should be set to 2 in the configuration provided");
        let dropout = Dropout::new(config.qa_dropout);

        DistilBertForQuestionAnswering { distil_bert_model, qa_outputs, dropout }
    }

    pub fn forward_t(&self,
                     input: Option<Tensor>,
                     mask: Option<Tensor>,
                     input_embeds: Option<Tensor>,
                     train: bool)
                     -> Result<(Tensor, Tensor, Option<Vec<Tensor>>, Option<Vec<Tensor>>), &'static str> {
        let (output, all_hidden_states, all_attentions) = match self.distil_bert_model.forward_t(input, mask, input_embeds, train) {
            Ok(value) => value,
            Err(err) => return Err(err)
        };

        let output = output
            .apply_t(&self.dropout, train)
            .apply(&self.qa_outputs);

        let logits = output.split(1, -1);
        let (start_logits, end_logits) = (&logits[0], &logits[1]);
        let start_logits = start_logits.squeeze1(-1);
        let end_logits = end_logits.squeeze1(-1);


        Ok((start_logits, end_logits, all_hidden_states, all_attentions))
    }
}

pub struct DistilBertForTokenClassification {
    distil_bert_model: DistilBertModel,
    classifier: nn::Linear,
    dropout: Dropout,
}

impl DistilBertForTokenClassification {
    pub fn new(p: &nn::Path, config: &DistilBertConfig) -> DistilBertForTokenClassification {
        let distil_bert_model = DistilBertModel::new(&p, config);
        let classifier = nn::linear(&(p / "classifier"), config.dim, config.num_labels, Default::default());
        let dropout = Dropout::new(config.seq_classif_dropout);

        DistilBertForTokenClassification { distil_bert_model, classifier, dropout }
    }

    pub fn forward_t(&self, input: Option<Tensor>, mask: Option<Tensor>, input_embeds: Option<Tensor>, train: bool)
                     -> Result<(Tensor, Option<Vec<Tensor>>, Option<Vec<Tensor>>), &'static str> {
        let (output, all_hidden_states, all_attentions) = match self.distil_bert_model.forward_t(input, mask, input_embeds, train) {
            Ok(value) => value,
            Err(err) => return Err(err)
        };

        let output = output
            .apply_t(&self.dropout, train)
            .apply(&self.classifier);

        Ok((output, all_hidden_states, all_attentions))
    }
}