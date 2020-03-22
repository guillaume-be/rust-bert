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
use crate::bert::embeddings::{BertEmbeddings, BertEmbedding};
use crate::bert::encoder::{BertEncoder, BertPooler};
use tch::{nn, Tensor, Kind};
use tch::kind::Kind::Float;
use crate::common::activations::{_gelu, _relu, _mish};
use crate::common::linear::{LinearNoBias, linear_no_bias};
use tch::nn::Init;
use crate::common::dropout::Dropout;
use std::collections::HashMap;
use crate::Config;

#[allow(non_camel_case_types)]
#[derive(Debug, Serialize, Deserialize)]
/// # Activation function used in the attention layer and masked language model head
pub enum Activation {
    /// Gaussian Error Linear Unit ([Hendrycks et al., 2016,](https://arxiv.org/abs/1606.08415))
    gelu,
    /// Rectified Linear Unit
    relu,
    /// Mish ([Misra, 2019](https://arxiv.org/abs/1908.08681))
    mish,
}

#[derive(Debug, Serialize, Deserialize)]
/// # BERT model configuration
/// Defines the BERT model architecture (e.g. number of layers, hidden layer size, label mapping...)
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

/// # BERT Base model
/// Base architecture for BERT models. Task-specific models will be built from this common base model
/// It is made of the following blocks:
/// - `embeddings`: `token`, `position` and `segment_id` embeddings
/// - `encoder`: Encoder (transformer) made of a vector of layers. Each layer is made of a self-attention layer, an intermediate (linear) and output (linear + layer norm) layers
/// - `pooler`: linear layer applied to the first element of the sequence (*[MASK]* token)
/// - `is_decoder`: Flag indicating if the model is used as a decoder. If set to true, a causal mask will be applied to hide future positions that should not be attended to.
pub struct BertModel<T: BertEmbedding> {
    embeddings: T,
    encoder: BertEncoder,
    pooler: BertPooler,
    is_decoder: bool,
}

/// Defines the implementation of the BertModel. The BERT model shares many similarities with RoBERTa, main difference being the embeddings.
/// Therefore the forward pass of the model is shared and the type of embedding used is abstracted away. This allows to create
/// `BertModel<RobertaEmbeddings>` or `BertModel<BertEmbeddings>` for each model type.
impl<T: BertEmbedding> BertModel<T> {
    /// Build a new `BertModel`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the BERT model
    /// * `config` - `BertConfig` object defining the model architecture and decoder status
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::bert::{BertModel, BertConfig, BertEmbeddings};
    /// use tch::{nn, Device};
    /// use rust_bert::Config;
    /// use std::path::Path;
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = BertConfig::from_file(config_path);
    /// let bert: BertModel<BertEmbeddings> = BertModel::new(&(&p.root() / "bert"), &config);
    /// ```
    ///
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

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see `input_embeds`)
    /// * `mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `token_type_ids` -Optional segment id of shape (*batch size*, *sequence_length*). Convention is value of 0 for the first sentence (incl. *[SEP]*) and 1 for the second sentence. If None set to 0.
    /// * `position_ids` - Optional position ids of shape (*batch size*, *sequence_length*). If None, will be incremented from 0.
    /// * `input_embeds` - Optional pre-computed input embeddings of shape (*batch size*, *sequence_length*, *hidden_size*). If None, input ids must be provided (see `input_ids`)
    /// * `encoder_hidden_states` - Optional encoder hidden state of shape (*batch size*, *encoder_sequence_length*, *hidden_size*). If the model is defined as a decoder and the `encoder_hidden_states` is not None, used in the cross-attention layer as keys and values (query from the decoder).
    /// * `encoder_mask` - Optional encoder attention mask of shape (*batch size*, *encoder_sequence_length*). If the model is defined as a decoder and the `encoder_hidden_states` is not None, used to mask encoder values. Positions with value 0 will be masked.
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `output` - `Tensor` of shape (*batch size*, *sequence_length*, *hidden_size*)
    /// * `pooled_output` - `Tensor` of shape (*batch size*, *hidden_size*)
    /// * `hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    /// * `attentions` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    ///# use rust_bert::bert::{BertModel, BertConfig, BertEmbeddings};
    ///# use tch::{nn, Device, Tensor, no_grad};
    ///# use rust_bert::Config;
    ///# use std::path::Path;
    ///# use rust_tokenizers::{BertTokenizer, TruncationStrategy, Tokenizer};
    ///# let config_path = Path::new("path/to/config.json");
    ///# let vocab_path = Path::new("path/to/vocab.txt");
    ///# let device = Device::Cpu;
    ///# let vs = nn::VarStore::new(device);
    ///# let tokenizer: BertTokenizer = BertTokenizer::from_file(vocab_path.to_str().unwrap(), true);
    ///# let config = BertConfig::from_file(config_path);
    ///# let bert_model: BertModel<BertEmbeddings> = BertModel::new(&vs.root(), &config);
    /// let input = ["One sentence", "Another sentence"];
    /// let tokenized_input = tokenizer.encode_list(input.to_vec(), 128, &TruncationStrategy::LongestFirst, 0);
    /// let max_len = tokenized_input.iter().map(|input| input.token_ids.len()).max().unwrap();
    /// let mut tokenized_input = tokenized_input
    ///     .iter()
    ///     // retrieve input ids from TokenizedInput
    ///     .map(|input| input.token_ids.clone())
    ///     // Padding inputs to same length
    ///     .map(|mut input| {
    ///         input.extend(vec![0; max_len - input.len()]);
    ///         input
    ///     })
    ///     // Map to Tensor
    ///     .map(|input|
    ///         Tensor::of_slice(&(input)))
    ///     .collect::<Vec<_>>();
    ///  let input_tensor = Tensor::stack(tokenized_input.as_slice(), 0).to(device);
    ///
    ///  let (output, _, _, _) = no_grad(|| {
    ///    bert_model
    ///         .forward_t(Some(input_tensor),
    ///                    None,
    ///                    None,
    ///                    None,
    ///                    None,
    ///                    &None,
    ///                    &None,
    ///                    false).unwrap()
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

/// # BERT for masked language model
/// Base BERT model with a masked language model head to predict missing tokens, for example `"Looks like one [MASK] is missing" -> "person"`
/// It is made of the following blocks:
/// - `bert`: Base BertModel
/// - `cls`: BERT LM prediction head
pub struct BertForMaskedLM {
    bert: BertModel<BertEmbeddings>,
    cls: BertLMPredictionHead,
}

impl BertForMaskedLM {
    /// Build a new `BertForMaskedLM`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the BertForMaskedLM model
    /// * `config` - `BertConfig` object defining the model architecture and vocab size
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::bert::{BertConfig, BertForMaskedLM};
    /// use tch::{nn, Device};
    /// use rust_bert::Config;
    /// use std::path::Path;
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = BertConfig::from_file(config_path);
    /// let bert = BertForMaskedLM::new(&(&p.root() / "bert"), &config);
    /// ```
    ///
    pub fn new(p: &nn::Path, config: &BertConfig) -> BertForMaskedLM {
        let bert = BertModel::new(&(p / "bert"), config);
        let cls = BertLMPredictionHead::new(&(p / "cls"), config);

        BertForMaskedLM { bert, cls }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see *input_embeds*)
    /// * `mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `token_type_ids` -Optional segment id of shape (*batch size*, *sequence_length*). Convention is value of 0 for the first sentence (incl. *[SEP]*) and 1 for the second sentence. If None set to 0.
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
    ///# use rust_bert::bert::{BertModel, BertConfig, BertForMaskedLM};
    ///# use tch::{nn, Device, Tensor, no_grad};
    ///# use rust_bert::Config;
    ///# use std::path::Path;
    ///# use rust_tokenizers::{BertTokenizer, TruncationStrategy, Tokenizer};
    ///# let config_path = Path::new("path/to/config.json");
    ///# let vocab_path = Path::new("path/to/vocab.txt");
    ///# let device = Device::Cpu;
    ///# let vs = nn::VarStore::new(device);
    ///# let tokenizer: BertTokenizer = BertTokenizer::from_file(vocab_path.to_str().unwrap(), true);
    ///# let config = BertConfig::from_file(config_path);
    ///# let bert_model = BertForMaskedLM::new(&vs.root(), &config);
    /// let input = ["Looks like one [MASK] is missing", "It was a very nice and [MASK] day"];
    /// let tokenized_input = tokenizer.encode_list(input.to_vec(), 128, &TruncationStrategy::LongestFirst, 0);
    /// let max_len = tokenized_input.iter().map(|input| input.token_ids.len()).max().unwrap();
    /// let mut tokenized_input = tokenized_input
    ///     .iter()
    ///     // retrieve input ids from TokenizedInput
    ///     .map(|input| input.token_ids.clone())
    ///     // Padding inputs to same length
    ///     .map(|mut input| {
    ///         input.extend(vec![0; max_len - input.len()]);
    ///         input
    ///     })
    ///     // Map to Tensor
    ///     .map(|input|
    ///         Tensor::of_slice(&(input)))
    ///     .collect::<Vec<_>>();
    ///  let input_tensor = Tensor::stack(tokenized_input.as_slice(), 0).to(device);
    ///
    ///  let (output, _, _) = no_grad(|| {
    ///    bert_model
    ///         .forward_t(Some(input_tensor),
    ///                    None,
    ///                    None,
    ///                    None,
    ///                    None,
    ///                    &None,
    ///                    &None,
    ///                    false)
    ///    });
    ///
    /// ```
    ///
    /// Outputs: `Looks like one [person] is missing` and `It was a very nice and [pleasant] day`.
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
        let (hidden_state, _, all_hidden_states, all_attentions) = self.bert.forward_t(input_ids, mask, token_type_ids, position_ids,
                                                                                       input_embeds, encoder_hidden_states, encoder_mask, train).unwrap();

        let prediction_scores = self.cls.forward(&hidden_state);
        (prediction_scores, all_hidden_states, all_attentions)
    }
}

/// # BERT for sequence classification
/// Base BERT model with a classifier head to perform sentence or document-level classification
/// It is made of the following blocks:
/// - `bert`: Base BertModel
/// - `classifier`: BERT linear layer for classification
pub struct BertForSequenceClassification {
    bert: BertModel<BertEmbeddings>,
    dropout: Dropout,
    classifier: nn::Linear,
}

impl BertForSequenceClassification {
    /// Build a new `BertForSequenceClassification`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the BertForMaskedLM model
    /// * `config` - `BertConfig` object defining the model architecture and number of classes
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::bert::{BertConfig, BertForSequenceClassification};
    /// use tch::{nn, Device};
    /// use rust_bert::Config;
    /// use std::path::Path;
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = BertConfig::from_file(config_path);
    /// let bert = BertForSequenceClassification::new(&(&p.root() / "bert"), &config);
    /// ```
    pub fn new(p: &nn::Path, config: &BertConfig) -> BertForSequenceClassification {
        let bert = BertModel::new(&(p / "bert"), config);
        let dropout = Dropout::new(config.hidden_dropout_prob);
        let num_labels = config.num_labels.expect("num_labels not provided in configuration");
        let classifier = nn::linear(p / "classifier", config.hidden_size, num_labels, Default::default());

        BertForSequenceClassification { bert, dropout, classifier }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length`). If None, pre-computed embeddings must be provided (see `input_embeds`)
    /// * `mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `token_type_ids` -Optional segment id of shape (*batch size*, *sequence_length*). Convention is value of 0 for the first sentence (incl. *[SEP]*) and 1 for the second sentence. If None set to 0.
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
    ///# use rust_bert::bert::{BertModel, BertConfig, BertForSequenceClassification};
    ///# use tch::{nn, Device, Tensor, no_grad};
    ///# use rust_bert::Config;
    ///# use std::path::Path;
    ///# use rust_tokenizers::{BertTokenizer, TruncationStrategy, Tokenizer};
    ///# let config_path = Path::new("path/to/config.json");
    ///# let vocab_path = Path::new("path/to/vocab.txt");
    ///# let device = Device::Cpu;
    ///# let vs = nn::VarStore::new(device);
    ///# let tokenizer: BertTokenizer = BertTokenizer::from_file(vocab_path.to_str().unwrap(), true);
    ///# let config = BertConfig::from_file(config_path);
    ///# let bert_model = BertForSequenceClassification::new(&vs.root(), &config);
    /// let input = ["First sentence to classify", "Second sentence to classify"];
    /// let tokenized_input = tokenizer.encode_list(input.to_vec(), 128, &TruncationStrategy::LongestFirst, 0);
    /// let max_len = tokenized_input.iter().map(|input| input.token_ids.len()).max().unwrap();
    /// let mut tokenized_input = tokenized_input
    ///     .iter()
    ///     // retrieve input ids from TokenizedInput
    ///     .map(|input| input.token_ids.clone())
    ///     // Padding inputs to same length
    ///     .map(|mut input| {
    ///         input.extend(vec![0; max_len - input.len()]);
    ///         input
    ///     })
    ///     // Map to Tensor
    ///     .map(|input|
    ///         Tensor::of_slice(&(input)))
    ///     .collect::<Vec<_>>();
    ///  let input_tensor = Tensor::stack(tokenized_input.as_slice(), 0).to(device);
    ///
    ///  let (output, _, _) = no_grad(|| {
    ///    bert_model
    ///         .forward_t(Some(input_tensor),
    ///                    None,
    ///                    None,
    ///                    None,
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