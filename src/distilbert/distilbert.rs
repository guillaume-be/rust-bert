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
use crate::common::dropout::Dropout;
use crate::Config;

/// # DistilBERT Pretrained model weight files
pub struct DistilBertModelResources;

/// # DistilBERT Pretrained model config files
pub struct DistilBertConfigResources;

/// # DistilBERT Pretrained model vocab files
pub struct DistilBertVocabResources;

impl DistilBertModelResources {
    pub const DISTIL_BERT_SST2: (&'static str, &'static str) = ("distilbert-sst2/model.ot", "https://cdn.huggingface.co/distilbert-base-uncased-finetuned-sst-2-english-rust_model.ot");
    pub const DISTIL_BERT: (&'static str, &'static str) = ("distilbert/model.ot", "https://cdn.huggingface.co/distilbert-base-uncased-rust_model.ot");
    pub const DISTIL_BERT_SQUAD: (&'static str, &'static str) = ("distilbert-qa/model.ot", "https://cdn.huggingface.co/distilbert-base-cased-distilled-squad-rust_model.ot");
}

impl DistilBertConfigResources {
    pub const DISTIL_BERT_SST2: (&'static str, &'static str) = ("distilbert-sst2/config.json", "https://cdn.huggingface.co/distilbert-base-uncased-finetuned-sst-2-english-config.json");
    pub const DISTIL_BERT: (&'static str, &'static str) = ("distilbert/config.json", "https://cdn.huggingface.co/distilbert-base-uncased-config.json");
    pub const DISTIL_BERT_SQUAD: (&'static str, &'static str) = ("distilbert-qa/config.json", "https://cdn.huggingface.co/distilbert-base-cased-distilled-squad-config.json");
}

impl DistilBertVocabResources {
    pub const DISTIL_BERT_SST2: (&'static str, &'static str) = ("distilbert-sst2/vocab.txt", "https://cdn.huggingface.co/distilbert-base-uncased-finetuned-sst-2-english-vocab.txt");
    pub const DISTIL_BERT: (&'static str, &'static str) = ("distilbert/vocab.txt", "https://cdn.huggingface.co/bert-base-uncased-vocab.txt");
    pub const DISTIL_BERT_SQUAD: (&'static str, &'static str) = ("distilbert-qa/vocab.txt", "https://cdn.huggingface.co/bert-large-cased-vocab.txt");
}

#[allow(non_camel_case_types)]
#[derive(Debug, Serialize, Deserialize)]
/// # Activation function used in the feed-forward layer in the transformer blocks
pub enum Activation {
    /// Gaussian Error Linear Unit ([Hendrycks et al., 2016,](https://arxiv.org/abs/1606.08415))
    gelu,
    /// Rectified Linear Unit
    relu,
}

#[derive(Debug, Serialize, Deserialize)]
/// # DistilBERT model configuration
/// Defines the DistilBERT model architecture (e.g. number of layers, hidden layer size, label mapping...)
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
    pub output_attentions: Option<bool>,
    pub output_hidden_states: Option<bool>,
    pub output_past: Option<bool>,
    pub qa_dropout: f64,
    pub seq_classif_dropout: f64,
    pub sinusoidal_pos_embds: bool,
    pub tie_weights_: bool,
    pub torchscript: Option<bool>,
    pub use_bfloat16: Option<bool>,
    pub vocab_size: i64,
}

impl Config<DistilBertConfig> for DistilBertConfig {}

/// # DistilBERT Base model
/// Base architecture for DistilBERT models. Task-specific models will be built from this common base model
/// It is made of the following blocks:
/// - `embeddings`: `token`, `position` embeddings
/// - `transformer`: Transformer made of a vector of layers. Each layer is made of a multi-head self-attention layer, layer norm and linear layers.
pub struct DistilBertModel {
    embeddings: DistilBertEmbedding,
    transformer: Transformer,
}

/// Defines the implementation of the DistilBertModel.
impl DistilBertModel {
    /// Build a new `DistilBertModel`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the DistilBERT model
    /// * `config` - `DistilBertConfig` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use tch::{nn, Device};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use rust_bert::distilbert::{DistilBertConfig, DistilBertModel};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = DistilBertConfig::from_file(config_path);
    /// let distil_bert: DistilBertModel = DistilBertModel::new(&(&p.root() / "distilbert"), &config);
    /// ```
    ///
    pub fn new(p: &nn::Path, config: &DistilBertConfig) -> DistilBertModel {
        let p = &(p / "distilbert");
        let embeddings = DistilBertEmbedding::new(&(p / "embeddings"), config);
        let transformer = Transformer::new(&(p / "transformer"), config);
        DistilBertModel { embeddings, transformer }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see `input_embeds`)
    /// * `mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `input_embeds` - Optional pre-computed input embeddings of shape (*batch size*, *sequence_length*, *hidden_size*). If None, input ids must be provided (see `input_ids`)
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
    ///# use tch::{nn, Device, Tensor, no_grad};
    ///# use rust_bert::Config;
    ///# use std::path::Path;
    ///# use tch::kind::Kind::Int64;
    /// use rust_bert::distilbert::{DistilBertConfig, DistilBertModel};
    ///# let config_path = Path::new("path/to/config.json");
    ///# let vocab_path = Path::new("path/to/vocab.txt");
    ///# let device = Device::Cpu;
    ///# let vs = nn::VarStore::new(device);
    ///# let config = DistilBertConfig::from_file(config_path);
    ///# let distilbert_model: DistilBertModel = DistilBertModel::new(&vs.root(), &config);
    ///  let (batch_size, sequence_length) = (64, 128);
    ///  let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    ///  let mask = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    ///
    ///  let (output, all_hidden_states, all_attentions) = no_grad(|| {
    ///    distilbert_model
    ///         .forward_t(Some(input_tensor),
    ///                    Some(mask),
    ///                    None,
    ///                    false).unwrap()
    ///    });
    ///
    /// ```
    ///
    pub fn forward_t(&self, input: Option<Tensor>, mask: Option<Tensor>, input_embeds: Option<Tensor>, train: bool)
                     -> Result<(Tensor, Option<Vec<Tensor>>, Option<Vec<Tensor>>), &'static str> {
        let input_embeddings = match input {
            Some(input_value) => match input_embeds {
                Some(_) => { return Err("Only one of input ids or input embeddings may be set"); }
                None => input_value.apply_t(&self.embeddings, train)
            }
            None => match input_embeds {
                Some(embeds) => embeds,
                None => { return Err("At least one of input ids or input embeddings must be set"); }
            }
        };


        let transformer_output = (&self.transformer).forward_t(&input_embeddings, mask, train);
        Ok(transformer_output)
    }
}

/// # DistilBERT for sequence classification
/// Base DistilBERT model with a pre-classifier and classifier heads to perform sentence or document-level classification
/// It is made of the following blocks:
/// - `distil_bert_model`: Base DistilBertModel
/// - `pre_classifier`: DistilBERT linear layer for classification
/// - `classifier`: DistilBERT linear layer for classification
pub struct DistilBertModelClassifier {
    distil_bert_model: DistilBertModel,
    pre_classifier: nn::Linear,
    classifier: nn::Linear,
    dropout: Dropout,
}

impl DistilBertModelClassifier {
    /// Build a new `DistilBertModelClassifier` for sequence classification
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the DistilBertModelClassifier model
    /// * `config` - `DistilBertConfig` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use tch::{nn, Device};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use rust_bert::distilbert::{DistilBertConfig, DistilBertModelClassifier};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = DistilBertConfig::from_file(config_path);
    /// let distil_bert: DistilBertModelClassifier = DistilBertModelClassifier::new(&(&p.root() / "distilbert"), &config);
    /// ```
    ///
    pub fn new(p: &nn::Path, config: &DistilBertConfig) -> DistilBertModelClassifier {
        let distil_bert_model = DistilBertModel::new(&p, config);

        let num_labels = config.id2label.as_ref().expect("id2label must be provided for classifiers").len() as i64;

        let pre_classifier = nn::linear(&(p / "pre_classifier"), config.dim, config.dim, Default::default());
        let classifier = nn::linear(&(p / "classifier"), config.dim, num_labels, Default::default());
        let dropout = Dropout::new(config.seq_classif_dropout);

        DistilBertModelClassifier { distil_bert_model, pre_classifier, classifier, dropout }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see `input_embeds`)
    /// * `mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `input_embeds` - Optional pre-computed input embeddings of shape (*batch size*, *sequence_length*, *hidden_size*). If None, input ids must be provided (see `input_ids`)
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `output` - `Tensor` of shape (*batch size*, *num_labels*) representing the logits for each class to predict
    /// * `hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    /// * `attentions` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    ///# use tch::{nn, Device, Tensor, no_grad};
    ///# use rust_bert::Config;
    ///# use std::path::Path;
    ///# use tch::kind::Kind::Int64;
    /// use rust_bert::distilbert::{DistilBertConfig, DistilBertModelClassifier};
    ///# let config_path = Path::new("path/to/config.json");
    ///# let vocab_path = Path::new("path/to/vocab.txt");
    ///# let device = Device::Cpu;
    ///# let vs = nn::VarStore::new(device);
    ///# let config = DistilBertConfig::from_file(config_path);
    ///# let distilbert_model: DistilBertModelClassifier = DistilBertModelClassifier::new(&vs.root(), &config);
    ///  let (batch_size, sequence_length) = (64, 128);
    ///  let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    ///  let mask = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    ///
    ///  let (output, all_hidden_states, all_attentions) = no_grad(|| {
    ///    distilbert_model
    ///         .forward_t(Some(input_tensor),
    ///                    Some(mask),
    ///                    None,
    ///                    false).unwrap()
    ///    });
    ///
    /// ```
    ///
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

/// # DistilBERT for masked language model
/// Base DistilBERT model with a masked language model head to predict missing tokens, for example `"Looks like one [MASK] is missing" -> "person"`
/// It is made of the following blocks:
/// - `distil_bert_model`: Base DistilBertModel
/// - `vocab_transform`:linear layer for classification of size (*hidden_dim*, *hidden_dim*)
/// - `vocab_layer_norm`: layer normalization
/// - `vocab_projector`: linear layer for classification of size (*hidden_dim*, *vocab_size*) with weights tied to the token embeddings
pub struct DistilBertModelMaskedLM {
    distil_bert_model: DistilBertModel,
    vocab_transform: nn::Linear,
    vocab_layer_norm: nn::LayerNorm,
    vocab_projector: nn::Linear,
}


impl DistilBertModelMaskedLM {
    /// Build a new `DistilBertModelMaskedLM` for sequence classification
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the DistilBertModelMaskedLM model
    /// * `config` - `DistilBertConfig` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use tch::{nn, Device};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use rust_bert::distilbert::{DistilBertConfig, DistilBertModelMaskedLM};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = DistilBertConfig::from_file(config_path);
    /// let distil_bert = DistilBertModelMaskedLM::new(&(&p.root() / "distilbert"), &config);
    /// ```
    ///
    pub fn new(p: &nn::Path, config: &DistilBertConfig) -> DistilBertModelMaskedLM {
        let distil_bert_model = DistilBertModel::new(&p, config);
        let vocab_transform = nn::linear(&(p / "vocab_transform"), config.dim, config.dim, Default::default());
        let layer_norm_config = nn::LayerNormConfig { eps: 1e-12, ..Default::default() };
        let vocab_layer_norm = nn::layer_norm(p / "vocab_layer_norm", vec![config.dim], layer_norm_config);
        let vocab_projector = nn::linear(&(p / "vocab_projector"), config.dim, config.vocab_size, Default::default());

        DistilBertModelMaskedLM { distil_bert_model, vocab_transform, vocab_layer_norm, vocab_projector }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see `input_embeds`)
    /// * `mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `input_embeds` - Optional pre-computed input embeddings of shape (*batch size*, *sequence_length*, *hidden_size*). If None, input ids must be provided (see `input_ids`)
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `output` - `Tensor` of shape (*batch size*, *sequence_length*, *vocab_size*) representing the logits for position and vocabulary index
    /// * `hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    /// * `attentions` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    ///# use tch::{nn, Device, Tensor, no_grad};
    ///# use rust_bert::Config;
    ///# use std::path::Path;
    ///# use tch::kind::Kind::Int64;
    /// use rust_bert::distilbert::{DistilBertConfig, DistilBertModelMaskedLM};
    ///# let config_path = Path::new("path/to/config.json");
    ///# let vocab_path = Path::new("path/to/vocab.txt");
    ///# let device = Device::Cpu;
    ///# let vs = nn::VarStore::new(device);
    ///# let config = DistilBertConfig::from_file(config_path);
    ///# let distilbert_model = DistilBertModelMaskedLM::new(&vs.root(), &config);
    ///  let (batch_size, sequence_length) = (64, 128);
    ///  let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    ///  let mask = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    ///
    ///  let (output, all_hidden_states, all_attentions) = no_grad(|| {
    ///    distilbert_model
    ///         .forward_t(Some(input_tensor),
    ///                    Some(mask),
    ///                    None,
    ///                    false).unwrap()
    ///    });
    ///
    /// ```
    ///
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

/// # DistilBERT for question answering
/// Extractive question-answering model based on a DistilBERT language model. Identifies the segment of a context that answers a provided question.
/// Please note that a significant amount of pre- and post-processing is required to perform end-to-end question answering.
/// See the question answering pipeline (also provided in this crate) for more details.
/// It is made of the following blocks:
/// - `distil_bert_model`: Base DistilBertModel
/// - `qa_outputs`: Linear layer for question answering
pub struct DistilBertForQuestionAnswering {
    distil_bert_model: DistilBertModel,
    qa_outputs: nn::Linear,
    dropout: Dropout,
}

impl DistilBertForQuestionAnswering {
    /// Build a new `DistilBertForQuestionAnswering` for sequence classification
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the DistilBertForQuestionAnswering model
    /// * `config` - `DistilBertConfig` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use tch::{nn, Device};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use rust_bert::distilbert::{DistilBertConfig, DistilBertForQuestionAnswering};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = DistilBertConfig::from_file(config_path);
    /// let distil_bert = DistilBertForQuestionAnswering::new(&(&p.root() / "distilbert"), &config);
    /// ```
    ///
    pub fn new(p: &nn::Path, config: &DistilBertConfig) -> DistilBertForQuestionAnswering {
        let distil_bert_model = DistilBertModel::new(&p, config);
        let qa_outputs = nn::linear(&(p / "qa_outputs"), config.dim, 2, Default::default());
        let dropout = Dropout::new(config.qa_dropout);

        DistilBertForQuestionAnswering { distil_bert_model, qa_outputs, dropout }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see `input_embeds`)
    /// * `mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
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
    ///# use tch::{nn, Device, Tensor, no_grad};
    ///# use rust_bert::Config;
    ///# use std::path::Path;
    ///# use tch::kind::Kind::Int64;
    /// use rust_bert::distilbert::{DistilBertConfig, DistilBertForQuestionAnswering};
    ///# let config_path = Path::new("path/to/config.json");
    ///# let vocab_path = Path::new("path/to/vocab.txt");
    ///# let device = Device::Cpu;
    ///# let vs = nn::VarStore::new(device);
    ///# let config = DistilBertConfig::from_file(config_path);
    ///# let distilbert_model = DistilBertForQuestionAnswering::new(&vs.root(), &config);
    ///  let (batch_size, sequence_length) = (64, 128);
    ///  let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    ///  let mask = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    ///
    ///  let (start_scores, end_score, all_hidden_states, all_attentions) = no_grad(|| {
    ///    distilbert_model
    ///         .forward_t(Some(input_tensor),
    ///                    Some(mask),
    ///                    None,
    ///                    false).unwrap()
    ///    });
    ///
    /// ```
    ///
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

/// # DistilBERT for token classification (e.g. NER, POS)
/// Token-level classifier predicting a label for each token provided. Note that because of wordpiece tokenization, the labels predicted are
/// not necessarily aligned with words in the sentence.
/// It is made of the following blocks:
/// - `distil_bert_model`: Base DistilBertModel
/// - `classifier`: Linear layer for token classification
pub struct DistilBertForTokenClassification {
    distil_bert_model: DistilBertModel,
    classifier: nn::Linear,
    dropout: Dropout,
}

impl DistilBertForTokenClassification {
    /// Build a new `DistilBertForTokenClassification` for sequence classification
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the DistilBertForTokenClassification model
    /// * `config` - `DistilBertConfig` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use tch::{nn, Device};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use rust_bert::distilbert::{DistilBertConfig, DistilBertForTokenClassification};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = DistilBertConfig::from_file(config_path);
    /// let distil_bert = DistilBertForTokenClassification::new(&(&p.root() / "distilbert"), &config);
    /// ```
    ///
    pub fn new(p: &nn::Path, config: &DistilBertConfig) -> DistilBertForTokenClassification {
        let distil_bert_model = DistilBertModel::new(&p, config);

        let num_labels = config.id2label.as_ref().expect("id2label must be provided for classifiers").len() as i64;

        let classifier = nn::linear(&(p / "classifier"), config.dim, num_labels, Default::default());
        let dropout = Dropout::new(config.seq_classif_dropout);

        DistilBertForTokenClassification { distil_bert_model, classifier, dropout }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see `input_embeds`)
    /// * `mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `input_embeds` - Optional pre-computed input embeddings of shape (*batch size*, *sequence_length*, *hidden_size*). If None, input ids must be provided (see `input_ids`)
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `output` - `Tensor` of shape (*batch size*, *sequence_length*, *num_labels*) representing the logits for position and class
    /// * `hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    /// * `attentions` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    ///# use tch::{nn, Device, Tensor, no_grad};
    ///# use rust_bert::Config;
    ///# use std::path::Path;
    ///# use tch::kind::Kind::Int64;
    /// use rust_bert::distilbert::{DistilBertConfig, DistilBertForTokenClassification};
    ///# let config_path = Path::new("path/to/config.json");
    ///# let vocab_path = Path::new("path/to/vocab.txt");
    ///# let device = Device::Cpu;
    ///# let vs = nn::VarStore::new(device);
    ///# let config = DistilBertConfig::from_file(config_path);
    ///# let distilbert_model = DistilBertForTokenClassification::new(&vs.root(), &config);
    ///  let (batch_size, sequence_length) = (64, 128);
    ///  let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    ///  let mask = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    ///
    ///  let (output, all_hidden_states, all_attentions) = no_grad(|| {
    ///    distilbert_model
    ///         .forward_t(Some(input_tensor),
    ///                    Some(mask),
    ///                    None,
    ///                    false).unwrap()
    ///    });
    ///
    /// ```
    ///
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