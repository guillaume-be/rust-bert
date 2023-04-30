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

use crate::bert::encoder::{BertEncoder, BertPooler};
use crate::common::activations::Activation;
use crate::common::dropout::Dropout;
use crate::common::embeddings::get_shape_and_device_from_ids_embeddings_pair;
use crate::common::linear::{linear_no_bias, LinearNoBias};
use crate::{
    bert::embeddings::{BertEmbedding, BertEmbeddings},
    common::activations::TensorFunction,
};
use crate::{Config, RustBertError};
use serde::{Deserialize, Serialize};
use std::borrow::Borrow;
use std::collections::HashMap;
use tch::nn::init::DEFAULT_KAIMING_UNIFORM;
use tch::{nn, Kind, Tensor};

/// # BERT Pretrained model weight files
pub struct BertModelResources;

/// # BERT Pretrained model config files
pub struct BertConfigResources;

/// # BERT Pretrained model vocab files
pub struct BertVocabResources;

impl BertModelResources {
    /// Shared under Apache 2.0 license by the Google team at <https://github.com/google-research/bert>. Modified with conversion to C-array format.
    pub const BERT: (&'static str, &'static str) = (
        "bert/model",
        "https://huggingface.co/bert-base-uncased/resolve/main/rust_model.ot",
    );
    /// Shared under MIT license by the MDZ Digital Library team at the Bavarian State Library at <https://github.com/dbmdz/berts>. Modified with conversion to C-array format.
    pub const BERT_NER: (&'static str, &'static str) = (
        "bert-ner/model",
        "https://huggingface.co/dbmdz/bert-large-cased-finetuned-conll03-english/resolve/main/rust_model.ot",
    );
    /// Shared under Apache 2.0 license by Hugging Face Inc at <https://github.com/huggingface/transformers/tree/master/examples/question-answering>. Modified with conversion to C-array format.
    pub const BERT_QA: (&'static str, &'static str) = (
        "bert-qa/model",
        "https://huggingface.co/bert-large-cased-whole-word-masking-finetuned-squad/resolve/main/rust_model.ot",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens>. Modified with conversion to C-array format.
    pub const BERT_BASE_NLI_MEAN_TOKENS: (&'static str, &'static str) = (
        "bert-base-nli-mean-tokens/model",
        "https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens/resolve/main/rust_model.ot",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2>. Modified with conversion to C-array format.
    pub const ALL_MINI_LM_L12_V2: (&'static str, &'static str) = (
        "all-mini-lm-l12-v2/model",
        "https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2/resolve/main/rust_model.ot",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>. Modified with conversion to C-array format.
    pub const ALL_MINI_LM_L6_V2: (&'static str, &'static str) = (
        "all-mini-lm-l6-v2/model",
        "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/rust_model.ot",
    );
}

impl BertConfigResources {
    /// Shared under Apache 2.0 license by the Google team at <https://github.com/google-research/bert>. Modified with conversion to C-array format.
    pub const BERT: (&'static str, &'static str) = (
        "bert/config",
        "https://huggingface.co/bert-base-uncased/resolve/main/config.json",
    );
    /// Shared under MIT license by the MDZ Digital Library team at the Bavarian State Library at <https://github.com/dbmdz/berts>. Modified with conversion to C-array format.
    pub const BERT_NER: (&'static str, &'static str) = (
        "bert-ner/config",
        "https://huggingface.co/dbmdz/bert-large-cased-finetuned-conll03-english/resolve/main/config.json",
    );
    /// Shared under Apache 2.0 license by Hugging Face Inc at <https://github.com/huggingface/transformers/tree/master/examples/question-answering>. Modified with conversion to C-array format.
    pub const BERT_QA: (&'static str, &'static str) = (
        "bert-qa/config",
        "https://huggingface.co/bert-large-cased-whole-word-masking-finetuned-squad/resolve/main/config.json",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens>. Modified with conversion to C-array format.
    pub const BERT_BASE_NLI_MEAN_TOKENS: (&'static str, &'static str) = (
        "bert-base-nli-mean-tokens/config",
        "https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens/resolve/main/config.json",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2>. Modified with conversion to C-array format.
    pub const ALL_MINI_LM_L12_V2: (&'static str, &'static str) = (
        "all-mini-lm-l12-v2/config",
        "https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2/resolve/main/config.json",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>. Modified with conversion to C-array format.
    pub const ALL_MINI_LM_L6_V2: (&'static str, &'static str) = (
        "all-mini-lm-l6-v2/config",
        "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json",
    );
}

impl BertVocabResources {
    /// Shared under Apache 2.0 license by the Google team at <https://github.com/google-research/bert>. Modified with conversion to C-array format.
    pub const BERT: (&'static str, &'static str) = (
        "bert/vocab",
        "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt",
    );
    /// Shared under MIT license by the MDZ Digital Library team at the Bavarian State Library at <https://github.com/dbmdz/berts>. Modified with conversion to C-array format.
    pub const BERT_NER: (&'static str, &'static str) = (
        "bert-ner/vocab",
        "https://huggingface.co/dbmdz/bert-large-cased-finetuned-conll03-english/resolve/main/vocab.txt",
    );
    /// Shared under Apache 2.0 license by Hugging Face Inc at <https://github.com/huggingface/transformers/tree/master/examples/question-answering>. Modified with conversion to C-array format.
    pub const BERT_QA: (&'static str, &'static str) = (
        "bert-qa/vocab",
        "https://huggingface.co/bert-large-cased-whole-word-masking-finetuned-squad/resolve/main/vocab.txt",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens>. Modified with conversion to C-array format.
    pub const BERT_BASE_NLI_MEAN_TOKENS: (&'static str, &'static str) = (
        "bert-base-nli-mean-tokens/vocab",
        "https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens/resolve/main/vocab.txt",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2>. Modified with conversion to C-array format.
    pub const ALL_MINI_LM_L12_V2: (&'static str, &'static str) = (
        "all-mini-lm-l12-v2/vocab",
        "https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2/resolve/main/vocab.txt",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>. Modified with conversion to C-array format.
    pub const ALL_MINI_LM_L6_V2: (&'static str, &'static str) = (
        "all-mini-lm-l6-v2/vocab",
        "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/vocab.txt",
    );
}

#[derive(Debug, Serialize, Deserialize, Clone)]
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
}

impl Config for BertConfig {}

impl Default for BertConfig {
    fn default() -> Self {
        BertConfig {
            hidden_act: Activation::gelu,
            attention_probs_dropout_prob: 0.1,
            hidden_dropout_prob: 0.1,
            hidden_size: 768,
            initializer_range: 0.02,
            intermediate_size: 3072,
            max_position_embeddings: 512,
            num_attention_heads: 12,
            num_hidden_layers: 12,
            type_vocab_size: 2,
            vocab_size: 30522,
            output_attentions: None,
            output_hidden_states: None,
            is_decoder: None,
            id2label: None,
            label2id: None,
        }
    }
}

/// # BERT Base model
/// Base architecture for BERT models. Task-specific models will be built from this common base model
/// It is made of the following blocks:
/// - `embeddings`: `token`, `position` and `segment_id` embeddings
/// - `encoder`: Encoder (transformer) made of a vector of layers. Each layer is made of a self-attention layer, an intermediate (linear) and output (linear + layer norm) layers
/// - `pooler`: linear layer applied to the first element of the sequence (*MASK* token)
/// - `is_decoder`: Flag indicating if the model is used as a decoder. If set to true, a causal mask will be applied to hide future positions that should not be attended to.
pub struct BertModel<T: BertEmbedding> {
    embeddings: T,
    encoder: BertEncoder,
    pooler: Option<BertPooler>,
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
    /// use rust_bert::bert::{BertConfig, BertEmbeddings, BertModel};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = BertConfig::from_file(config_path);
    /// let bert: BertModel<BertEmbeddings> = BertModel::new(&p.root() / "bert", &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &BertConfig) -> BertModel<T>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let is_decoder = config.is_decoder.unwrap_or(false);
        let embeddings = T::new(p / "embeddings", config);
        let encoder = BertEncoder::new(p / "encoder", config);
        let pooler = Some(BertPooler::new(p / "pooler", config));

        BertModel {
            embeddings,
            encoder,
            pooler,
            is_decoder,
        }
    }

    /// Build a new `BertModel` with an optional Pooling layer
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the BERT model
    /// * `config` - `BertConfig` object defining the model architecture and decoder status
    /// * `add_pooling_layer` - Enable/Disable an optional pooling layer at the end of the model
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::bert::{BertConfig, BertEmbeddings, BertModel};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = BertConfig::from_file(config_path);
    /// let bert: BertModel<BertEmbeddings> =
    ///     BertModel::new_with_optional_pooler(&p.root() / "bert", &config, false);
    /// ```
    pub fn new_with_optional_pooler<'p, P>(
        p: P,
        config: &BertConfig,
        add_pooling_layer: bool,
    ) -> BertModel<T>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let is_decoder = config.is_decoder.unwrap_or(false);
        let embeddings = T::new(p / "embeddings", config);
        let encoder = BertEncoder::new(p / "encoder", config);

        let pooler = {
            if add_pooling_layer {
                Some(BertPooler::new(p / "pooler", config))
            } else {
                None
            }
        };

        BertModel {
            embeddings,
            encoder,
            pooler,
            is_decoder,
        }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see `input_embeds`)
    /// * `mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `token_type_ids` - Optional segment id of shape (*batch size*, *sequence_length*). Convention is value of 0 for the first sentence (incl. *SEP*) and 1 for the second sentence. If None set to 0.
    /// * `position_ids` - Optional position ids of shape (*batch size*, *sequence_length*). If None, will be incremented from 0.
    /// * `input_embeds` - Optional pre-computed input embeddings of shape (*batch size*, *sequence_length*, *hidden_size*). If None, input ids must be provided (see `input_ids`)
    /// * `encoder_hidden_states` - Optional encoder hidden state of shape (*batch size*, *encoder_sequence_length*, *hidden_size*). If the model is defined as a decoder and the `encoder_hidden_states` is not None, used in the cross-attention layer as keys and values (query from the decoder).
    /// * `encoder_mask` - Optional encoder attention mask of shape (*batch size*, *encoder_sequence_length*). If the model is defined as a decoder and the `encoder_hidden_states` is not None, used to mask encoder values. Positions with value 0 will be masked.
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `BertOutput` containing:
    ///   - `hidden_state` - `Tensor` of shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `pooled_output` - `Tensor` of shape (*batch size*, *hidden_size*)
    ///   - `all_hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `all_attentions` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rust_bert::bert::{BertModel, BertConfig, BertEmbeddings};
    /// # use tch::{nn, Device, Tensor, no_grad, Kind};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # let config_path = Path::new("path/to/config.json");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = BertConfig::from_file(config_path);
    /// # let bert_model: BertModel<BertEmbeddings> = BertModel::new(&vs.root(), &config);
    /// let (batch_size, sequence_length) = (64, 128);
    /// let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Kind::Int64, device));
    /// let mask = Tensor::zeros(&[batch_size, sequence_length], (Kind::Int64, device));
    /// let token_type_ids = Tensor::zeros(&[batch_size, sequence_length], (Kind::Int64, device));
    /// let position_ids = Tensor::arange(sequence_length, (Kind::Int64, device))
    ///     .expand(&[batch_size, sequence_length], true);
    ///
    /// let model_output = no_grad(|| {
    ///     bert_model
    ///         .forward_t(
    ///             Some(&input_tensor),
    ///             Some(&mask),
    ///             Some(&token_type_ids),
    ///             Some(&position_ids),
    ///             None,
    ///             None,
    ///             None,
    ///             false,
    ///         )
    ///         .unwrap()
    /// });
    /// ```
    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        mask: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
        encoder_mask: Option<&Tensor>,
        train: bool,
    ) -> Result<BertModelOutput, RustBertError> {
        let (input_shape, device) =
            get_shape_and_device_from_ids_embeddings_pair(input_ids, input_embeds)?;

        let calc_mask = Tensor::ones(&input_shape, (Kind::Int8, device));
        let mask = mask.unwrap_or(&calc_mask);

        let extended_attention_mask = match mask.dim() {
            3 => mask.unsqueeze(1),
            2 => {
                if self.is_decoder {
                    let seq_ids = Tensor::arange(input_shape[1], (Kind::Int8, device));
                    let causal_mask = seq_ids.unsqueeze(0).unsqueeze(0).repeat(&[
                        input_shape[0],
                        input_shape[1],
                        1,
                    ]);
                    let causal_mask = causal_mask.le_tensor(&seq_ids.unsqueeze(0).unsqueeze(-1));
                    causal_mask * mask.unsqueeze(1).unsqueeze(1)
                } else {
                    mask.unsqueeze(1).unsqueeze(1)
                }
            }
            _ => {
                return Err(RustBertError::ValueError(
                    "Invalid attention mask dimension, must be 2 or 3".into(),
                ));
            }
        };

        let embedding_output = self.embeddings.forward_t(
            input_ids,
            token_type_ids,
            position_ids,
            input_embeds,
            train,
        )?;

        let extended_attention_mask: Tensor =
            ((extended_attention_mask.ones_like() - extended_attention_mask) * -10000.0)
                .to_kind(embedding_output.kind());

        let encoder_extended_attention_mask: Option<Tensor> =
            if self.is_decoder & encoder_hidden_states.is_some() {
                let encoder_hidden_states = encoder_hidden_states.as_ref().unwrap();
                let encoder_hidden_states_shape = encoder_hidden_states.size();
                let encoder_mask = match encoder_mask {
                    Some(value) => value.copy(),
                    None => Tensor::ones(
                        &[
                            encoder_hidden_states_shape[0],
                            encoder_hidden_states_shape[1],
                        ],
                        (Kind::Int8, device),
                    ),
                };
                match encoder_mask.dim() {
                    2 => Some(encoder_mask.unsqueeze(1).unsqueeze(1)),
                    3 => Some(encoder_mask.unsqueeze(1)),
                    _ => {
                        return Err(RustBertError::ValueError(
                            "Invalid attention mask dimension, must be 2 or 3".into(),
                        ));
                    }
                }
            } else {
                None
            };

        let encoder_output = self.encoder.forward_t(
            &embedding_output,
            Some(&extended_attention_mask),
            encoder_hidden_states,
            encoder_extended_attention_mask.as_ref(),
            train,
        );

        let pooled_output = self
            .pooler
            .as_ref()
            .map(|pooler| pooler.forward(&encoder_output.hidden_state));

        Ok(BertModelOutput {
            hidden_state: encoder_output.hidden_state,
            pooled_output,
            all_hidden_states: encoder_output.all_hidden_states,
            all_attentions: encoder_output.all_attentions,
        })
    }
}

pub struct BertPredictionHeadTransform {
    dense: nn::Linear,
    activation: TensorFunction,
    layer_norm: nn::LayerNorm,
}

impl BertPredictionHeadTransform {
    pub fn new<'p, P>(p: P, config: &BertConfig) -> BertPredictionHeadTransform
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let dense = nn::linear(
            p / "dense",
            config.hidden_size,
            config.hidden_size,
            Default::default(),
        );
        let activation = config.hidden_act.get_function();
        let layer_norm_config = nn::LayerNormConfig {
            eps: 1e-12,
            ..Default::default()
        };
        let layer_norm =
            nn::layer_norm(p / "LayerNorm", vec![config.hidden_size], layer_norm_config);

        BertPredictionHeadTransform {
            dense,
            activation,
            layer_norm,
        }
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Tensor {
        self.activation.get_fn()(&hidden_states.apply(&self.dense)).apply(&self.layer_norm)
    }
}

pub struct BertLMPredictionHead {
    transform: BertPredictionHeadTransform,
    decoder: LinearNoBias,
    bias: Tensor,
}

impl BertLMPredictionHead {
    pub fn new<'p, P>(p: P, config: &BertConfig) -> BertLMPredictionHead
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow() / "predictions";
        let transform = BertPredictionHeadTransform::new(&p / "transform", config);
        let decoder = linear_no_bias(
            &p / "decoder",
            config.hidden_size,
            config.vocab_size,
            Default::default(),
        );
        let bias = p.var("bias", &[config.vocab_size], DEFAULT_KAIMING_UNIFORM);

        BertLMPredictionHead {
            transform,
            decoder,
            bias,
        }
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Tensor {
        self.transform.forward(hidden_states).apply(&self.decoder) + &self.bias
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
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = BertConfig::from_file(config_path);
    /// let bert = BertForMaskedLM::new(&p.root() / "bert", &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &BertConfig) -> BertForMaskedLM
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let bert = BertModel::new_with_optional_pooler(p / "bert", config, false);
        let cls = BertLMPredictionHead::new(p / "cls", config);

        BertForMaskedLM { bert, cls }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see *input_embeds*)
    /// * `mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `token_type_ids` -Optional segment id of shape (*batch size*, *sequence_length*). Convention is value of 0 for the first sentence (incl. *SEP*) and 1 for the second sentence. If None set to 0.
    /// * `position_ids` - Optional position ids of shape (*batch size*, *sequence_length*). If None, will be incremented from 0.
    /// * `input_embeds` - Optional pre-computed input embeddings of shape (*batch size*, *sequence_length*, *hidden_size*). If None, input ids must be provided (see *input_ids*)
    /// * `encoder_hidden_states` - Optional encoder hidden state of shape (*batch size*, *encoder_sequence_length*, *hidden_size*). If the model is defined as a decoder and the *encoder_hidden_states* is not None, used in the cross-attention layer as keys and values (query from the decoder).
    /// * `encoder_mask` - Optional encoder attention mask of shape (*batch size*, *encoder_sequence_length*). If the model is defined as a decoder and the *encoder_hidden_states* is not None, used to mask encoder values. Positions with value 0 will be masked.
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `BertMaskedLMOutput` containing:
    ///   - `prediction_scores` - `Tensor` of shape (*batch size*, *sequence_length*, *vocab_size*)
    ///   - `all_hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `all_attentions` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rust_bert::bert::{BertForMaskedLM, BertConfig};
    /// # use tch::{nn, Device, Tensor, no_grad, Kind};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # let config_path = Path::new("path/to/config.json");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = BertConfig::from_file(config_path);
    /// # let bert_model = BertForMaskedLM::new(&vs.root(), &config);
    /// let (batch_size, sequence_length) = (64, 128);
    /// let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Kind::Int64, device));
    /// let mask = Tensor::zeros(&[batch_size, sequence_length], (Kind::Int64, device));
    /// let token_type_ids = Tensor::zeros(&[batch_size, sequence_length], (Kind::Int64, device));
    /// let position_ids = Tensor::arange(sequence_length, (Kind::Int64, device))
    ///     .expand(&[batch_size, sequence_length], true);
    ///
    /// let model_output = no_grad(|| {
    ///     bert_model.forward_t(
    ///         Some(&input_tensor),
    ///         Some(&mask),
    ///         Some(&token_type_ids),
    ///         Some(&position_ids),
    ///         None,
    ///         None,
    ///         None,
    ///         false,
    ///     )
    /// });
    /// ```
    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        mask: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
        encoder_mask: Option<&Tensor>,
        train: bool,
    ) -> BertMaskedLMOutput {
        let base_model_output = self
            .bert
            .forward_t(
                input_ids,
                mask,
                token_type_ids,
                position_ids,
                input_embeds,
                encoder_hidden_states,
                encoder_mask,
                train,
            )
            .unwrap();

        let prediction_scores = self.cls.forward(&base_model_output.hidden_state);
        BertMaskedLMOutput {
            prediction_scores,
            all_hidden_states: base_model_output.all_hidden_states,
            all_attentions: base_model_output.all_attentions,
        }
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
    /// * `p` - Variable store path for the root of the BertForSequenceClassification model
    /// * `config` - `BertConfig` object defining the model architecture and number of classes
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::bert::{BertConfig, BertForSequenceClassification};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = BertConfig::from_file(config_path);
    /// let bert = BertForSequenceClassification::new(&p.root() / "bert", &config).unwrap();
    /// ```
    pub fn new<'p, P>(
        p: P,
        config: &BertConfig,
    ) -> Result<BertForSequenceClassification, RustBertError>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let bert = BertModel::new(p / "bert", config);
        let dropout = Dropout::new(config.hidden_dropout_prob);
        let num_labels = config
            .id2label
            .as_ref()
            .ok_or_else(|| {
                RustBertError::InvalidConfigurationError(
                    "num_labels not provided in configuration".to_string(),
                )
            })?
            .len() as i64;
        let classifier = nn::linear(
            p / "classifier",
            config.hidden_size,
            num_labels,
            Default::default(),
        );

        Ok(BertForSequenceClassification {
            bert,
            dropout,
            classifier,
        })
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see `input_embeds`)
    /// * `mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `token_type_ids` -Optional segment id of shape (*batch size*, *sequence_length*). Convention is value of 0 for the first sentence (incl. *SEP*) and 1 for the second sentence. If None set to 0.
    /// * `position_ids` - Optional position ids of shape (*batch size*, *sequence_length*). If None, will be incremented from 0.
    /// * `input_embeds` - Optional pre-computed input embeddings of shape (*batch size*, *sequence_length*, *hidden_size*). If None, input ids must be provided (see `input_ids`)
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `BertSequenceClassificationOutput` containing:
    ///   - `logits` - `Tensor` of shape (*batch size*, *num_labels*)
    ///   - `all_hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `all_attentions` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rust_bert::bert::{BertForSequenceClassification, BertConfig};
    /// # use tch::{nn, Device, Tensor, no_grad, Kind};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # let config_path = Path::new("path/to/config.json");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = BertConfig::from_file(config_path);
    /// # let bert_model = BertForSequenceClassification::new(&vs.root(), &config).unwrap();;
    /// let (batch_size, sequence_length) = (64, 128);
    /// let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Kind::Int64, device));
    /// let mask = Tensor::zeros(&[batch_size, sequence_length], (Kind::Int64, device));
    /// let token_type_ids = Tensor::zeros(&[batch_size, sequence_length], (Kind::Int64, device));
    /// let position_ids = Tensor::arange(sequence_length, (Kind::Int64, device))
    ///     .expand(&[batch_size, sequence_length], true);
    ///
    /// let model_output = no_grad(|| {
    ///     bert_model.forward_t(
    ///         Some(&input_tensor),
    ///         Some(&mask),
    ///         Some(&token_type_ids),
    ///         Some(&position_ids),
    ///         None,
    ///         false,
    ///     )
    /// });
    /// ```
    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        mask: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        train: bool,
    ) -> BertSequenceClassificationOutput {
        let base_model_output = self
            .bert
            .forward_t(
                input_ids,
                mask,
                token_type_ids,
                position_ids,
                input_embeds,
                None,
                None,
                train,
            )
            .unwrap();

        let logits = base_model_output
            .pooled_output
            .unwrap()
            .apply_t(&self.dropout, train)
            .apply(&self.classifier);
        BertSequenceClassificationOutput {
            logits,
            all_hidden_states: base_model_output.all_hidden_states,
            all_attentions: base_model_output.all_attentions,
        }
    }
}

/// # BERT for multiple choices
/// Multiple choices model using a BERT base model and a linear classifier.
/// Input should be in the form `[CLS] Context [SEP] Possible choice [SEP]`. The choice is made along the batch axis,
/// assuming all elements of the batch are alternatives to be chosen from for a given context.
/// It is made of the following blocks:
/// - `bert`: Base BertModel
/// - `classifier`: Linear layer for multiple choices
pub struct BertForMultipleChoice {
    bert: BertModel<BertEmbeddings>,
    dropout: Dropout,
    classifier: nn::Linear,
}

impl BertForMultipleChoice {
    /// Build a new `BertForMultipleChoice`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the BertForMultipleChoice model
    /// * `config` - `BertConfig` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::bert::{BertConfig, BertForMultipleChoice};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = BertConfig::from_file(config_path);
    /// let bert = BertForMultipleChoice::new(&p.root() / "bert", &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &BertConfig) -> BertForMultipleChoice
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let bert = BertModel::new(p / "bert", config);
        let dropout = Dropout::new(config.hidden_dropout_prob);
        let classifier = nn::linear(p / "classifier", config.hidden_size, 1, Default::default());

        BertForMultipleChoice {
            bert,
            dropout,
            classifier,
        }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Input tensor of shape (*batch size*, *sequence_length*).
    /// * `mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `token_type_ids` -Optional segment id of shape (*batch size*, *sequence_length*). Convention is value of 0 for the first sentence (incl. *SEP*) and 1 for the second sentence. If None set to 0.
    /// * `position_ids` - Optional position ids of shape (*batch size*, *sequence_length*). If None, will be incremented from 0.
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `BertSequenceClassificationOutput` containing:
    ///   - `logits` - `Tensor` of shape (*1*, *batch size*) containing the logits for each of the alternatives given
    ///   - `all_hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `all_attentions` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rust_bert::bert::{BertForMultipleChoice, BertConfig};
    /// # use tch::{nn, Device, Tensor, no_grad};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::Int64;
    /// # let config_path = Path::new("path/to/config.json");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = BertConfig::from_file(config_path);
    /// # let bert_model = BertForMultipleChoice::new(&vs.root(), &config);
    /// let (num_choices, sequence_length) = (3, 128);
    /// let input_tensor = Tensor::rand(&[num_choices, sequence_length], (Int64, device));
    /// let mask = Tensor::zeros(&[num_choices, sequence_length], (Int64, device));
    /// let token_type_ids = Tensor::zeros(&[num_choices, sequence_length], (Int64, device));
    /// let position_ids = Tensor::arange(sequence_length, (Int64, device))
    ///     .expand(&[num_choices, sequence_length], true);
    ///
    /// let model_output = no_grad(|| {
    ///     bert_model.forward_t(
    ///         &input_tensor,
    ///         Some(&mask),
    ///         Some(&token_type_ids),
    ///         Some(&position_ids),
    ///         false,
    ///     )
    /// });
    /// ```
    pub fn forward_t(
        &self,
        input_ids: &Tensor,
        mask: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        train: bool,
    ) -> BertSequenceClassificationOutput {
        let num_choices = input_ids.size()[1];

        let input_ids = input_ids.view((-1, *input_ids.size().last().unwrap()));
        let mask = mask.map(|tensor| tensor.view((-1, *tensor.size().last().unwrap())));
        let token_type_ids =
            token_type_ids.map(|tensor| tensor.view((-1, *tensor.size().last().unwrap())));
        let position_ids =
            position_ids.map(|tensor| tensor.view((-1, *tensor.size().last().unwrap())));

        let base_model_output = self
            .bert
            .forward_t(
                Some(&input_ids),
                mask.as_ref(),
                token_type_ids.as_ref(),
                position_ids.as_ref(),
                None,
                None,
                None,
                train,
            )
            .unwrap();

        let logits = base_model_output
            .pooled_output
            .unwrap()
            .apply_t(&self.dropout, train)
            .apply(&self.classifier)
            .view((-1, num_choices));
        BertSequenceClassificationOutput {
            logits,
            all_hidden_states: base_model_output.all_hidden_states,
            all_attentions: base_model_output.all_attentions,
        }
    }
}

/// # BERT for token classification (e.g. NER, POS)
/// Token-level classifier predicting a label for each token provided. Note that because of wordpiece tokenization, the labels predicted are
/// not necessarily aligned with words in the sentence.
/// It is made of the following blocks:
/// - `bert`: Base BertModel
/// - `classifier`: Linear layer for token classification
pub struct BertForTokenClassification {
    bert: BertModel<BertEmbeddings>,
    dropout: Dropout,
    classifier: nn::Linear,
}

impl BertForTokenClassification {
    /// Build a new `BertForTokenClassification`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the BertForTokenClassification model
    /// * `config` - `BertConfig` object defining the model architecture, number of output labels and label mapping
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::bert::{BertConfig, BertForTokenClassification};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = BertConfig::from_file(config_path);
    /// let bert = BertForTokenClassification::new(&p.root() / "bert", &config).unwrap();
    /// ```
    pub fn new<'p, P>(
        p: P,
        config: &BertConfig,
    ) -> Result<BertForTokenClassification, RustBertError>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let bert = BertModel::new_with_optional_pooler(p / "bert", config, false);
        let dropout = Dropout::new(config.hidden_dropout_prob);
        let num_labels = config
            .id2label
            .as_ref()
            .ok_or_else(|| {
                RustBertError::InvalidConfigurationError(
                    "num_labels not provided in configuration".to_string(),
                )
            })?
            .len() as i64;
        let classifier = nn::linear(
            p / "classifier",
            config.hidden_size,
            num_labels,
            Default::default(),
        );

        Ok(BertForTokenClassification {
            bert,
            dropout,
            classifier,
        })
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see `input_embeds`)
    /// * `mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `token_type_ids` -Optional segment id of shape (*batch size*, *sequence_length*). Convention is value of 0 for the first sentence (incl. *SEP*) and 1 for the second sentence. If None set to 0.
    /// * `position_ids` - Optional position ids of shape (*batch size*, *sequence_length*). If None, will be incremented from 0.
    /// * `input_embeds` - Optional pre-computed input embeddings of shape (*batch size*, *sequence_length*, *hidden_size*). If None, input ids must be provided (see `input_ids`)
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `BertTokenClassificationOutput` containing:
    ///   - `logits` - `Tensor` of shape (*batch size*, *sequence_length*, *num_labels*) containing the logits for each of the input tokens and classes
    ///   - `all_hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `all_attentions` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rust_bert::bert::{BertForTokenClassification, BertConfig};
    /// # use tch::{nn, Device, Tensor, no_grad};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::Int64;
    /// # let config_path = Path::new("path/to/config.json");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = BertConfig::from_file(config_path);
    /// # let bert_model = BertForTokenClassification::new(&vs.root(), &config).unwrap();
    /// let (batch_size, sequence_length) = (64, 128);
    /// let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    /// let mask = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    /// let token_type_ids = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    /// let position_ids = Tensor::arange(sequence_length, (Int64, device))
    ///     .expand(&[batch_size, sequence_length], true);
    ///
    /// let model_output = no_grad(|| {
    ///     bert_model.forward_t(
    ///         Some(&input_tensor),
    ///         Some(&mask),
    ///         Some(&token_type_ids),
    ///         Some(&position_ids),
    ///         None,
    ///         false,
    ///     )
    /// });
    /// ```
    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        mask: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        train: bool,
    ) -> BertTokenClassificationOutput {
        let base_model_output = self
            .bert
            .forward_t(
                input_ids,
                mask,
                token_type_ids,
                position_ids,
                input_embeds,
                None,
                None,
                train,
            )
            .unwrap();

        let logits = base_model_output
            .hidden_state
            .apply_t(&self.dropout, train)
            .apply(&self.classifier);
        BertTokenClassificationOutput {
            logits,
            all_hidden_states: base_model_output.all_hidden_states,
            all_attentions: base_model_output.all_attentions,
        }
    }
}

/// # BERT for question answering
/// Extractive question-answering model based on a BERT language model. Identifies the segment of a context that answers a provided question.
/// Please note that a significant amount of pre- and post-processing is required to perform end-to-end question answering.
/// See the question answering pipeline (also provided in this crate) for more details.
/// It is made of the following blocks:
/// - `bert`: Base BertModel
/// - `qa_outputs`: Linear layer for question answering
pub struct BertForQuestionAnswering {
    bert: BertModel<BertEmbeddings>,
    qa_outputs: nn::Linear,
}

impl BertForQuestionAnswering {
    /// Build a new `BertForQuestionAnswering`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the BertForQuestionAnswering model
    /// * `config` - `BertConfig` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::bert::{BertConfig, BertForQuestionAnswering};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = BertConfig::from_file(config_path);
    /// let bert = BertForQuestionAnswering::new(&p.root() / "bert", &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &BertConfig) -> BertForQuestionAnswering
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let bert = BertModel::new(p / "bert", config);
        let num_labels = 2;
        let qa_outputs = nn::linear(
            p / "qa_outputs",
            config.hidden_size,
            num_labels,
            Default::default(),
        );

        BertForQuestionAnswering { bert, qa_outputs }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see `input_embeds`)
    /// * `mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `token_type_ids` -Optional segment id of shape (*batch size*, *sequence_length*). Convention is value of 0 for the first sentence (incl. *SEP*) and 1 for the second sentence. If None set to 0.
    /// * `position_ids` - Optional position ids of shape (*batch size*, *sequence_length*). If None, will be incremented from 0.
    /// * `input_embeds` - Optional pre-computed input embeddings of shape (*batch size*, *sequence_length*, *hidden_size*). If None, input ids must be provided (see `input_ids`)
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `BertQuestionAnsweringOutput` containing:
    ///   - `start_logits` - `Tensor` of shape (*batch size*, *sequence_length*) containing the logits for start of the answer
    ///   - `end_logits` - `Tensor` of shape (*batch size*, *sequence_length*) containing the logits for end of the answer
    ///   - `all_hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `all_attentions` - `Option<Vec<Vec<Tensor>>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rust_bert::bert::{BertForQuestionAnswering, BertConfig};
    /// # use tch::{nn, Device, Tensor, no_grad};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::Int64;
    /// # let config_path = Path::new("path/to/config.json");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = BertConfig::from_file(config_path);
    /// # let bert_model = BertForQuestionAnswering::new(&vs.root(), &config);
    /// let (batch_size, sequence_length) = (64, 128);
    /// let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    /// let mask = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    /// let token_type_ids = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    /// let position_ids = Tensor::arange(sequence_length, (Int64, device))
    ///     .expand(&[batch_size, sequence_length], true);
    ///
    /// let model_output = no_grad(|| {
    ///     bert_model.forward_t(
    ///         Some(&input_tensor),
    ///         Some(&mask),
    ///         Some(&token_type_ids),
    ///         Some(&position_ids),
    ///         None,
    ///         false,
    ///     )
    /// });
    /// ```
    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        mask: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        train: bool,
    ) -> BertQuestionAnsweringOutput {
        let base_model_output = self
            .bert
            .forward_t(
                input_ids,
                mask,
                token_type_ids,
                position_ids,
                input_embeds,
                None,
                None,
                train,
            )
            .unwrap();

        let sequence_output = base_model_output.hidden_state.apply(&self.qa_outputs);
        let logits = sequence_output.split(1, -1);
        let (start_logits, end_logits) = (&logits[0], &logits[1]);
        let start_logits = start_logits.squeeze_dim(-1);
        let end_logits = end_logits.squeeze_dim(-1);

        BertQuestionAnsweringOutput {
            start_logits,
            end_logits,
            all_hidden_states: base_model_output.all_hidden_states,
            all_attentions: base_model_output.all_attentions,
        }
    }
}

/// # BERT for sentence embeddings
/// Transformer usable in [`SentenceEmbeddingsModel`](crate::pipelines::sentence_embeddings::SentenceEmbeddingsModel).
pub type BertForSentenceEmbeddings = BertModel<BertEmbeddings>;

/// Container for the BERT model output.
pub struct BertModelOutput {
    /// Last hidden states from the model
    pub hidden_state: Tensor,
    /// Pooled output (hidden state for the first token)
    pub pooled_output: Option<Tensor>,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all intermediate layers
    pub all_attentions: Option<Vec<Tensor>>,
}

/// Container for the BERT masked LM model output.
pub struct BertMaskedLMOutput {
    /// Logits for the vocabulary items at each sequence position
    pub prediction_scores: Tensor,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all intermediate layers
    pub all_attentions: Option<Vec<Tensor>>,
}

/// Container for the BERT sequence classification model output.
pub struct BertSequenceClassificationOutput {
    /// Logits for each input (sequence) for each target class
    pub logits: Tensor,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all intermediate layers
    pub all_attentions: Option<Vec<Tensor>>,
}

/// Container for the BERT token classification model output.
pub struct BertTokenClassificationOutput {
    /// Logits for each sequence item (token) for each target class
    pub logits: Tensor,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all intermediate layers
    pub all_attentions: Option<Vec<Tensor>>,
}

/// Container for the BERT question answering model output.
pub struct BertQuestionAnsweringOutput {
    /// Logits for the start position for token of each input sequence
    pub start_logits: Tensor,
    /// Logits for the end position for token of each input sequence
    pub end_logits: Tensor,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all intermediate layers
    pub all_attentions: Option<Vec<Tensor>>,
}

#[cfg(test)]
mod test {
    use tch::Device;

    use crate::{
        resources::{RemoteResource, ResourceProvider},
        Config,
    };

    use super::*;

    #[test]
    #[ignore] // compilation is enough, no need to run
    fn bert_model_send() {
        let config_resource = Box::new(RemoteResource::from_pretrained(BertConfigResources::BERT));
        let config_path = config_resource.get_local_path().expect("");

        //    Set-up masked LM model
        let device = Device::cuda_if_available();
        let vs = nn::VarStore::new(device);
        let config = BertConfig::from_file(config_path);

        let _: Box<dyn Send> = Box::new(BertModel::<BertEmbeddings>::new(vs.root(), &config));
    }
}
