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
use crate::common::linear::{linear_no_bias, LinearNoBias};
use tch::nn::Init;
use crate::common::activations::_gelu;
use crate::roberta::embeddings::RobertaEmbeddings;
use crate::common::dropout::Dropout;
use crate::bert::{BertConfig, BertModel};

/// # RoBERTa Pretrained model weight files
pub struct RobertaModelResources;

/// # RoBERTa Pretrained model config files
pub struct RobertaConfigResources;

/// # RoBERTa Pretrained model vocab files
pub struct RobertaVocabResources;

/// # RoBERTa Pretrained model merges files
pub struct RobertaMergesResources;

impl RobertaModelResources {
    pub const ROBERTA: (&'static str, &'static str) = ("roberta/model.ot", "https://cdn.huggingface.co/roberta-base-rust_model.ot");
}

impl RobertaConfigResources {
    pub const ROBERTA: (&'static str, &'static str) = ("roberta/config.json", "https://cdn.huggingface.co/roberta-base-config.json");
}

impl RobertaVocabResources {
    pub const ROBERTA: (&'static str, &'static str) = ("roberta/vocab.txt", "https://cdn.huggingface.co/roberta-base-vocab.json");
}

impl RobertaMergesResources {
    pub const ROBERTA: (&'static str, &'static str) = ("roberta/merges.txt", "https://cdn.huggingface.co/roberta-base-merges.txt");
}

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

/// # RoBERTa for masked language model
/// Base RoBERTa model with a RoBERTa masked language model head to predict missing tokens, for example `"Looks like one [MASK] is missing" -> "person"`
/// It is made of the following blocks:
/// - `roberta`: Base BertModel with RoBERTa embeddings
/// - `lm_head`: RoBERTa LM prediction head
pub struct RobertaForMaskedLM {
    roberta: BertModel<RobertaEmbeddings>,
    lm_head: RobertaLMHead,
}

impl RobertaForMaskedLM {
    /// Build a new `RobertaForMaskedLM`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the RobertaForMaskedLM model
    /// * `config` - `BertConfig` object defining the model architecture and vocab size
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::bert::BertConfig;
    /// use tch::{nn, Device};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use rust_bert::roberta::RobertaForMaskedLM;
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = BertConfig::from_file(config_path);
    /// let roberta = RobertaForMaskedLM::new(&(&p.root() / "roberta"), &config);
    /// ```
    ///
    pub fn new(p: &nn::Path, config: &BertConfig) -> RobertaForMaskedLM {
        let roberta = BertModel::<RobertaEmbeddings>::new(&(p / "roberta"), config);
        let lm_head = RobertaLMHead::new(&(p / "lm_head"), config);

        RobertaForMaskedLM { roberta, lm_head }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see *input_embeds*)
    /// * `mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `token_type_ids` -Optional segment id of shape (*batch size*, *sequence_length*). Convention is value of 0 for the first sentence (incl. *</s>*) and 1 for the second sentence. If None set to 0.
    /// * `position_ids` - Optional position ids of shape (*batch size*, *sequence_length*). If None, will be incremented from 0.
    /// * `input_embeds` - Optional pre-computed input embeddings of shape (*batch size*, *sequence_length*, *hidden_size*). If None, input ids must be provided (see *input_ids*)
    /// * `encoder_hidden_states` - Optional encoder hidden state of shape (*batch size*, *encoder_sequence_length*, *hidden_size*). If the model is defined as a decoder and the *encoder_hidden_states* is not None, used in the cross-attention layer as keys and values (query from the decoder).
    /// * `encoder_mask` - Optional encoder attention mask of shape (*batch size*, *encoder_sequence_length*). If the model is defined as a decoder and the *encoder_hidden_states* is not None, used to mask encoder values. Positions with value 0 will be masked.
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `output` - `Tensor` of shape (*batch size*, *num_labels*, *vocab_size*)
    /// * `hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    /// * `attentions` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    ///# use rust_bert::bert::BertConfig;
    ///# use tch::{nn, Device, Tensor, no_grad};
    ///# use rust_bert::Config;
    ///# use std::path::Path;
    ///# use tch::kind::Kind::Int64;
    /// use rust_bert::roberta::RobertaForMaskedLM;
    ///# let config_path = Path::new("path/to/config.json");
    ///# let vocab_path = Path::new("path/to/vocab.txt");
    ///# let device = Device::Cpu;
    ///# let vs = nn::VarStore::new(device);
    ///# let config = BertConfig::from_file(config_path);
    ///# let roberta_model = RobertaForMaskedLM::new(&vs.root(), &config);
    ///  let (batch_size, sequence_length) = (64, 128);
    ///  let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    ///  let mask = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    ///  let token_type_ids = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    ///  let position_ids = Tensor::arange(sequence_length, (Int64, device)).expand(&[batch_size, sequence_length], true);
    ///
    ///  let (output, all_hidden_states, all_attentions) = no_grad(|| {
    ///    roberta_model
    ///         .forward_t(Some(input_tensor),
    ///                    Some(mask),
    ///                    Some(token_type_ids),
    ///                    Some(position_ids),
    ///                    None,
    ///                    &None,
    ///                    &None,
    ///                    false)
    ///    });
    ///
    /// ```
    ///
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
        let num_labels = config.id2label.as_ref().expect("num_labels not provided in configuration").len() as i64;
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

/// # RoBERTa for sequence classification
/// Base RoBERTa model with a classifier head to perform sentence or document-level classification
/// It is made of the following blocks:
/// - `roberta`: Base RoBERTa model
/// - `classifier`: RoBERTa classification head made of 2 linear layers
pub struct RobertaForSequenceClassification {
    roberta: BertModel<RobertaEmbeddings>,
    classifier: RobertaClassificationHead,
}

impl RobertaForSequenceClassification {
    /// Build a new `RobertaForSequenceClassification`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the RobertaForSequenceClassification model
    /// * `config` - `BertConfig` object defining the model architecture and vocab size
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::bert::BertConfig;
    /// use tch::{nn, Device};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use rust_bert::roberta::RobertaForSequenceClassification;
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = BertConfig::from_file(config_path);
    /// let roberta = RobertaForSequenceClassification::new(&(&p.root() / "roberta"), &config);
    /// ```
    ///
    pub fn new(p: &nn::Path, config: &BertConfig) -> RobertaForSequenceClassification {
        let roberta = BertModel::<RobertaEmbeddings>::new(&(p / "roberta"), config);
        let classifier = RobertaClassificationHead::new(&(p / "classifier"), config);

        RobertaForSequenceClassification { roberta, classifier }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see `input_embeds`)
    /// * `mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `token_type_ids` -Optional segment id of shape (*batch size*, *sequence_length*). Convention is value of 0 for the first sentence (incl. *</s>*) and 1 for the second sentence. If None set to 0.
    /// * `position_ids` - Optional position ids of shape (*batch size*, *sequence_length*). If None, will be incremented from 0.
    /// * `input_embeds` - Optional pre-computed input embeddings of shape (*batch size*, *sequence_length*, *hidden_size*). If None, input ids must be provided (see `input_ids`)
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `labels` - `Tensor` of shape (*batch size*, *num_labels*)
    /// * `hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    /// * `attentions` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    ///# use rust_bert::bert::BertConfig;
    ///# use tch::{nn, Device, Tensor, no_grad};
    ///# use rust_bert::Config;
    ///# use std::path::Path;
    ///# use tch::kind::Kind::Int64;
    /// use rust_bert::roberta::RobertaForSequenceClassification;
    ///# let config_path = Path::new("path/to/config.json");
    ///# let vocab_path = Path::new("path/to/vocab.txt");
    ///# let device = Device::Cpu;
    ///# let vs = nn::VarStore::new(device);
    ///# let config = BertConfig::from_file(config_path);
    ///# let roberta_model = RobertaForSequenceClassification::new(&vs.root(), &config);
    ///  let (batch_size, sequence_length) = (64, 128);
    ///  let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    ///  let mask = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    ///  let token_type_ids = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    ///  let position_ids = Tensor::arange(sequence_length, (Int64, device)).expand(&[batch_size, sequence_length], true);
    ///
    ///  let (labels, all_hidden_states, all_attentions) = no_grad(|| {
    ///    roberta_model
    ///         .forward_t(Some(input_tensor),
    ///                    Some(mask),
    ///                    Some(token_type_ids),
    ///                    Some(position_ids),
    ///                    None,
    ///                    false)
    ///    });
    ///
    /// ```
    ///
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

/// # RoBERTa for multiple choices
/// Multiple choices model using a RoBERTa base model and a linear classifier.
/// Input should be in the form `<s> Context </s> Possible choice </s>`. The choice is made along the batch axis,
/// assuming all elements of the batch are alternatives to be chosen from for a given context.
/// It is made of the following blocks:
/// - `roberta`: Base RoBERTa model
/// - `classifier`: Linear layer for multiple choices
pub struct RobertaForMultipleChoice {
    roberta: BertModel<RobertaEmbeddings>,
    dropout: Dropout,
    classifier: nn::Linear,
}

impl RobertaForMultipleChoice {
    /// Build a new `RobertaForMultipleChoice`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the RobertaForMultipleChoice model
    /// * `config` - `BertConfig` object defining the model architecture and vocab size
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::bert::BertConfig;
    /// use tch::{nn, Device};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use rust_bert::roberta::RobertaForMultipleChoice;
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = BertConfig::from_file(config_path);
    /// let roberta = RobertaForMultipleChoice::new(&(&p.root() / "roberta"), &config);
    /// ```
    ///
    pub fn new(p: &nn::Path, config: &BertConfig) -> RobertaForMultipleChoice {
        let roberta = BertModel::<RobertaEmbeddings>::new(&(p / "roberta"), config);
        let dropout = Dropout::new(config.hidden_dropout_prob);
        let classifier = nn::linear(p / "classifier", config.hidden_size, 1, Default::default());

        RobertaForMultipleChoice { roberta, dropout, classifier }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Input tensor of shape (*batch size*, *sequence_length*).
    /// * `mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `token_type_ids` -Optional segment id of shape (*batch size*, *sequence_length*). Convention is value of 0 for the first sentence (incl. *</s>*) and 1 for the second sentence. If None set to 0.
    /// * `position_ids` - Optional position ids of shape (*batch size*, *sequence_length*). If None, will be incremented from 0.
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `output` - `Tensor` of shape (*1*, *batch size*) containing the logits for each of the alternatives given
    /// * `hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    /// * `attentions` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    ///# use rust_bert::bert::BertConfig;
    ///# use tch::{nn, Device, Tensor, no_grad};
    ///# use rust_bert::Config;
    ///# use std::path::Path;
    ///# use tch::kind::Kind::Int64;
    /// use rust_bert::roberta::RobertaForMultipleChoice;
    ///# let config_path = Path::new("path/to/config.json");
    ///# let vocab_path = Path::new("path/to/vocab.txt");
    ///# let device = Device::Cpu;
    ///# let vs = nn::VarStore::new(device);
    ///# let config = BertConfig::from_file(config_path);
    ///# let roberta_model = RobertaForMultipleChoice::new(&vs.root(), &config);
    ///  let (num_choices, sequence_length) = (3, 128);
    ///  let input_tensor = Tensor::rand(&[num_choices, sequence_length], (Int64, device));
    ///  let mask = Tensor::zeros(&[num_choices, sequence_length], (Int64, device));
    ///  let token_type_ids = Tensor::zeros(&[num_choices, sequence_length], (Int64, device));
    ///  let position_ids = Tensor::arange(sequence_length, (Int64, device)).expand(&[num_choices, sequence_length], true);
    ///
    ///  let (choices, all_hidden_states, all_attentions) = no_grad(|| {
    ///    roberta_model
    ///         .forward_t(input_tensor,
    ///                    Some(mask),
    ///                    Some(token_type_ids),
    ///                    Some(position_ids),
    ///                    false)
    ///    });
    ///
    /// ```
    ///
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

/// # RoBERTa for token classification (e.g. NER, POS)
/// Token-level classifier predicting a label for each token provided. Note that because of bpe tokenization, the labels predicted are
/// not necessarily aligned with words in the sentence.
/// It is made of the following blocks:
/// - `roberta`: Base RoBERTa model
/// - `classifier`: Linear layer for token classification
pub struct RobertaForTokenClassification {
    roberta: BertModel<RobertaEmbeddings>,
    dropout: Dropout,
    classifier: nn::Linear,
}

impl RobertaForTokenClassification {
    /// Build a new `RobertaForTokenClassification`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the RobertaForTokenClassification model
    /// * `config` - `BertConfig` object defining the model architecture and vocab size
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::bert::BertConfig;
    /// use tch::{nn, Device};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use rust_bert::roberta::RobertaForTokenClassification;
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = BertConfig::from_file(config_path);
    /// let roberta = RobertaForTokenClassification::new(&(&p.root() / "roberta"), &config);
    /// ```
    ///
    pub fn new(p: &nn::Path, config: &BertConfig) -> RobertaForTokenClassification {
        let roberta = BertModel::<RobertaEmbeddings>::new(&(p / "roberta"), config);
        let dropout = Dropout::new(config.hidden_dropout_prob);
        let num_labels = config.id2label.as_ref().expect("num_labels not provided in configuration").len() as i64;
        let classifier = nn::linear(p / "classifier", config.hidden_size, num_labels, Default::default());

        RobertaForTokenClassification { roberta, dropout, classifier }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see `input_embeds`)
    /// * `mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `token_type_ids` -Optional segment id of shape (*batch size*, *sequence_length*). Convention is value of 0 for the first sentence (incl. *</s>*) and 1 for the second sentence. If None set to 0.
    /// * `position_ids` - Optional position ids of shape (*batch size*, *sequence_length*). If None, will be incremented from 0.
    /// * `input_embeds` - Optional pre-computed input embeddings of shape (*batch size*, *sequence_length*, *hidden_size*). If None, input ids must be provided (see `input_ids`)
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `output` - `Tensor` of shape (*batch size*, *sequence_length*, *num_labels*) containing the logits for each of the input tokens and classes
    /// * `hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    /// * `attentions` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    ///# use rust_bert::bert::BertConfig;
    ///# use tch::{nn, Device, Tensor, no_grad};
    ///# use rust_bert::Config;
    ///# use std::path::Path;
    ///# use tch::kind::Kind::Int64;
    /// use rust_bert::roberta::RobertaForTokenClassification;
    ///# let config_path = Path::new("path/to/config.json");
    ///# let vocab_path = Path::new("path/to/vocab.txt");
    ///# let device = Device::Cpu;
    ///# let vs = nn::VarStore::new(device);
    ///# let config = BertConfig::from_file(config_path);
    ///# let roberta_model = RobertaForTokenClassification::new(&vs.root(), &config);
    ///  let (batch_size, sequence_length) = (64, 128);
    ///  let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    ///  let mask = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    ///  let token_type_ids = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    ///  let position_ids = Tensor::arange(sequence_length, (Int64, device)).expand(&[batch_size, sequence_length], true);
    ///
    ///  let (token_labels, all_hidden_states, all_attentions) = no_grad(|| {
    ///    roberta_model
    ///         .forward_t(Some(input_tensor),
    ///                    Some(mask),
    ///                    Some(token_type_ids),
    ///                    Some(position_ids),
    ///                    None,
    ///                    false)
    ///    });
    ///
    /// ```
    ///
    pub fn forward_t(&self,
                     input_ids: Option<Tensor>,
                     mask: Option<Tensor>,
                     token_type_ids: Option<Tensor>,
                     position_ids: Option<Tensor>,
                     input_embeds: Option<Tensor>,
                     train: bool) -> (Tensor, Option<Vec<Tensor>>, Option<Vec<Tensor>>) {
        let (hidden_state, _, all_hidden_states, all_attentions) = self.roberta.forward_t(input_ids, mask, token_type_ids, position_ids,
                                                                                          input_embeds, &None, &None, train).unwrap();

        let sequence_output = hidden_state.apply_t(&self.dropout, train).apply(&self.classifier);
        (sequence_output, all_hidden_states, all_attentions)
    }
}

/// # RoBERTa for question answering
/// Extractive question-answering model based on a RoBERTa language model. Identifies the segment of a context that answers a provided question.
/// Please note that a significant amount of pre- and post-processing is required to perform end-to-end question answering.
/// See the question answering pipeline (also provided in this crate) for more details.
/// It is made of the following blocks:
/// - `roberta`: Base RoBERTa model
/// - `qa_outputs`: Linear layer for question answering
pub struct RobertaForQuestionAnswering {
    roberta: BertModel<RobertaEmbeddings>,
    qa_outputs: nn::Linear,
}

impl RobertaForQuestionAnswering {
    /// Build a new `RobertaForQuestionAnswering`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the RobertaForQuestionAnswering model
    /// * `config` - `BertConfig` object defining the model architecture and vocab size
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::bert::BertConfig;
    /// use tch::{nn, Device};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use rust_bert::roberta::RobertaForQuestionAnswering;
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = BertConfig::from_file(config_path);
    /// let roberta = RobertaForQuestionAnswering::new(&(&p.root() / "roberta"), &config);
    /// ```
    ///
    pub fn new(p: &nn::Path, config: &BertConfig) -> RobertaForQuestionAnswering {
        let roberta = BertModel::<RobertaEmbeddings>::new(&(p / "roberta"), config);
        let num_labels = 2;
        let qa_outputs = nn::linear(p / "qa_outputs", config.hidden_size, num_labels, Default::default());

        RobertaForQuestionAnswering { roberta, qa_outputs }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see `input_embeds`)
    /// * `mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `token_type_ids` -Optional segment id of shape (*batch size*, *sequence_length*). Convention is value of 0 for the first sentence (incl. *</s>*) and 1 for the second sentence. If None set to 0.
    /// * `position_ids` - Optional position ids of shape (*batch size*, *sequence_length*). If None, will be incremented from 0.
    /// * `input_embeds` - Optional pre-computed input embeddings of shape (*batch size*, *sequence_length*, *hidden_size*). If None, input ids must be provided (see `input_ids`)
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `start_scores` - `Tensor` of shape (*batch size*, *sequence_length*) containing the logits for start of the answer
    /// * `end_scores` - `Tensor` of shape (*batch size*, *sequence_length*) containing the logits for end of the answer
    /// * `hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    /// * `attentions` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    ///# use rust_bert::bert::BertConfig;
    ///# use tch::{nn, Device, Tensor, no_grad};
    ///# use rust_bert::Config;
    ///# use std::path::Path;
    ///# use tch::kind::Kind::Int64;
    /// use rust_bert::roberta::RobertaForQuestionAnswering;
    ///# let config_path = Path::new("path/to/config.json");
    ///# let vocab_path = Path::new("path/to/vocab.txt");
    ///# let device = Device::Cpu;
    ///# let vs = nn::VarStore::new(device);
    ///# let config = BertConfig::from_file(config_path);
    ///# let roberta_model = RobertaForQuestionAnswering::new(&vs.root(), &config);
    ///  let (batch_size, sequence_length) = (64, 128);
    ///  let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    ///  let mask = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    ///  let token_type_ids = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    ///  let position_ids = Tensor::arange(sequence_length, (Int64, device)).expand(&[batch_size, sequence_length], true);
    ///
    ///  let (start_scores, end_scores, all_hidden_states, all_attentions) = no_grad(|| {
    ///    roberta_model
    ///         .forward_t(Some(input_tensor),
    ///                    Some(mask),
    ///                    Some(token_type_ids),
    ///                    Some(position_ids),
    ///                    None,
    ///                    false)
    ///    });
    ///
    /// ```
    ///
    pub fn forward_t(&self,
                     input_ids: Option<Tensor>,
                     mask: Option<Tensor>,
                     token_type_ids: Option<Tensor>,
                     position_ids: Option<Tensor>,
                     input_embeds: Option<Tensor>,
                     train: bool) -> (Tensor, Tensor, Option<Vec<Tensor>>, Option<Vec<Tensor>>) {
        let (hidden_state, _, all_hidden_states, all_attentions) = self.roberta.forward_t(input_ids, mask, token_type_ids, position_ids,
                                                                                          input_embeds, &None, &None, train).unwrap();

        let sequence_output = hidden_state.apply(&self.qa_outputs);
        let logits = sequence_output.split(1, -1);
        let (start_logits, end_logits) = (&logits[0], &logits[1]);
        let start_logits = start_logits.squeeze1(-1);
        let end_logits = end_logits.squeeze1(-1);

        (start_logits, end_logits, all_hidden_states, all_attentions)
    }
}