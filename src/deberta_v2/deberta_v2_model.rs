// Copyright 2020, Microsoft and the HuggingFace Inc. team.
// Copyright 2022 Guillaume Becquin
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::common::dropout::{Dropout, XDropout};
use crate::common::embeddings::get_shape_and_device_from_ids_embeddings_pair;
use crate::deberta::{
    deserialize_attention_type, ContextPooler, DebertaConfig, DebertaLMPredictionHead,
    DebertaMaskedLMOutput, DebertaModelOutput, DebertaQuestionAnsweringOutput,
    DebertaSequenceClassificationOutput, DebertaTokenClassificationOutput, PositionAttentionTypes,
};
use crate::deberta_v2::embeddings::DebertaV2Embeddings;
use crate::deberta_v2::encoder::DebertaV2Encoder;
use crate::{Activation, Config, RustBertError};
use serde::de::{SeqAccess, Visitor};
use serde::{de, Deserialize, Deserializer, Serialize};
use std::borrow::Borrow;
use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;
use tch::{nn, Kind, Tensor};

/// # DeBERTaV2 Pretrained model weight files
pub struct DebertaV2ModelResources;

/// # DeBERTaV2 Pretrained model config files
pub struct DebertaV2ConfigResources;

/// # DeBERTaV2 Pretrained model vocab files
pub struct DebertaV2VocabResources;

impl DebertaV2ModelResources {
    /// Shared under MIT license by the Microsoft team at <https://huggingface.co/microsoft/deberta-v3-base>. Modified with conversion to C-array format.
    pub const DEBERTA_V3_BASE: (&'static str, &'static str) = (
        "deberta-v3-base/model",
        "https://huggingface.co/microsoft/deberta-v3-base/resolve/main/rust_model.ot",
    );
}

impl DebertaV2ConfigResources {
    /// Shared under MIT license by the Microsoft team at <https://huggingface.co/microsoft/deberta-v3-base>. Modified with conversion to C-array format.
    pub const DEBERTA_V3_BASE: (&'static str, &'static str) = (
        "deberta-v3-base/config",
        "https://huggingface.co/microsoft/deberta-v3-base/resolve/main/config.json",
    );
}

impl DebertaV2VocabResources {
    /// Shared under MIT license by the Microsoft team at <https://huggingface.co/microsoft/deberta-v3-base>. Modified with conversion to C-array format.
    pub const DEBERTA_V3_BASE: (&'static str, &'static str) = (
        "deberta-v3-base/vocab",
        "https://huggingface.co/microsoft/deberta-v3-base/resolve/main/spm.model",
    );
}

#[derive(Debug, Serialize, Deserialize)]
/// # DeBERTa (v2) model configuration
/// Defines the DeBERTa (v2) model architecture (e.g. number of layers, hidden layer size, label mapping...)
pub struct DebertaV2Config {
    pub vocab_size: i64,
    pub hidden_size: i64,
    pub num_hidden_layers: i64,
    pub hidden_act: Activation,
    pub attention_probs_dropout_prob: f64,
    pub hidden_dropout_prob: f64,
    pub initializer_range: f64,
    pub intermediate_size: i64,
    pub max_position_embeddings: i64,
    pub position_buckets: Option<i64>,
    pub num_attention_heads: i64,
    pub type_vocab_size: i64,
    pub position_biased_input: Option<bool>,
    #[serde(default, deserialize_with = "deserialize_attention_type")]
    pub pos_att_type: Option<PositionAttentionTypes>,
    #[serde(default, deserialize_with = "deserialize_norm_type")]
    pub norm_rel_ebd: Option<NormRelEmbedTypes>,
    pub share_att_key: Option<bool>,
    pub conv_kernel_size: Option<i64>,
    pub conv_groups: Option<i64>,
    pub conv_act: Option<Activation>,
    pub pooler_dropout: Option<f64>,
    pub pooler_hidden_act: Option<Activation>,
    pub pooler_hidden_size: Option<i64>,
    pub layer_norm_eps: Option<f64>,
    pub pad_token_id: Option<i64>,
    pub relative_attention: Option<bool>,
    pub max_relative_positions: Option<i64>,
    pub embedding_size: Option<i64>,
    pub talking_head: Option<bool>,
    pub output_hidden_states: Option<bool>,
    pub output_attentions: Option<bool>,
    pub classifier_activation: Option<bool>,
    pub classifier_dropout: Option<f64>,
    pub is_decoder: Option<bool>,
    pub id2label: Option<HashMap<i64, String>>,
    pub label2id: Option<HashMap<String, i64>>,
}

#[allow(non_camel_case_types)]
#[derive(Clone, Debug, Serialize, Deserialize, Copy, PartialEq)]
/// # Layer normalization layer for the DeBERTa model's relative embeddings.
pub enum NormRelEmbedType {
    layer_norm,
}

impl FromStr for NormRelEmbedType {
    type Err = RustBertError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "layer_norm" => Ok(NormRelEmbedType::layer_norm),
            _ => Err(RustBertError::InvalidConfigurationError(format!(
                "Layer normalization type `{}` not in accepted variants (`layer_norm`)",
                s
            ))),
        }
    }
}

#[allow(non_camel_case_types)]
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct NormRelEmbedTypes {
    types: Vec<NormRelEmbedType>,
}

impl FromStr for NormRelEmbedTypes {
    type Err = RustBertError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let types = s
            .to_lowercase()
            .split('|')
            .map(NormRelEmbedType::from_str)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(NormRelEmbedTypes { types })
    }
}

impl NormRelEmbedTypes {
    pub fn has_type(&self, norm_type: NormRelEmbedType) -> bool {
        self.types.iter().any(|self_type| *self_type == norm_type)
    }

    pub fn len(&self) -> usize {
        self.types.len()
    }
}

pub fn deserialize_norm_type<'de, D>(deserializer: D) -> Result<Option<NormRelEmbedTypes>, D::Error>
where
    D: Deserializer<'de>,
{
    struct NormTypeVisitor;

    impl<'de> Visitor<'de> for NormTypeVisitor {
        type Value = NormRelEmbedTypes;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("null, string or sequence")
        }

        fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(FromStr::from_str(value).unwrap())
        }

        fn visit_seq<S>(self, mut seq: S) -> Result<Self::Value, S::Error>
        where
            S: SeqAccess<'de>,
        {
            let mut types = vec![];
            while let Some(norm_type) = seq.next_element::<String>()? {
                types.push(FromStr::from_str(norm_type.as_str()).unwrap())
            }
            Ok(NormRelEmbedTypes { types })
        }
    }

    deserializer.deserialize_any(NormTypeVisitor).map(Some)
}

impl Config for DebertaV2Config {}

impl From<DebertaV2Config> for DebertaConfig {
    fn from(v2_config: DebertaV2Config) -> Self {
        DebertaConfig {
            hidden_act: v2_config.hidden_act,
            attention_probs_dropout_prob: v2_config.attention_probs_dropout_prob,
            hidden_dropout_prob: v2_config.hidden_dropout_prob,
            hidden_size: v2_config.hidden_size,
            initializer_range: v2_config.initializer_range,
            intermediate_size: v2_config.intermediate_size,
            max_position_embeddings: v2_config.max_position_embeddings,
            num_attention_heads: v2_config.num_attention_heads,
            num_hidden_layers: v2_config.num_hidden_layers,
            type_vocab_size: v2_config.type_vocab_size,
            vocab_size: v2_config.vocab_size,
            position_biased_input: v2_config.position_biased_input,
            pos_att_type: v2_config.pos_att_type,
            pooler_dropout: v2_config.pooler_dropout,
            pooler_hidden_act: v2_config.pooler_hidden_act,
            pooler_hidden_size: v2_config.pooler_hidden_size,
            layer_norm_eps: v2_config.layer_norm_eps,
            pad_token_id: v2_config.pad_token_id,
            relative_attention: v2_config.relative_attention,
            max_relative_positions: v2_config.max_relative_positions,
            embedding_size: v2_config.embedding_size,
            talking_head: v2_config.talking_head,
            output_hidden_states: v2_config.output_hidden_states,
            output_attentions: v2_config.output_attentions,
            classifier_activation: v2_config.classifier_activation,
            classifier_dropout: v2_config.classifier_dropout,
            is_decoder: v2_config.is_decoder,
            id2label: v2_config.id2label,
            label2id: v2_config.label2id,
            share_att_key: v2_config.share_att_key,
            position_buckets: v2_config.position_buckets,
        }
    }
}

impl From<&DebertaV2Config> for DebertaConfig {
    fn from(v2_config: &DebertaV2Config) -> Self {
        DebertaConfig {
            hidden_act: v2_config.hidden_act,
            attention_probs_dropout_prob: v2_config.attention_probs_dropout_prob,
            hidden_dropout_prob: v2_config.hidden_dropout_prob,
            hidden_size: v2_config.hidden_size,
            initializer_range: v2_config.initializer_range,
            intermediate_size: v2_config.intermediate_size,
            max_position_embeddings: v2_config.max_position_embeddings,
            num_attention_heads: v2_config.num_attention_heads,
            num_hidden_layers: v2_config.num_hidden_layers,
            type_vocab_size: v2_config.type_vocab_size,
            vocab_size: v2_config.vocab_size,
            position_biased_input: v2_config.position_biased_input,
            pos_att_type: v2_config.pos_att_type.clone(),
            pooler_dropout: v2_config.pooler_dropout,
            pooler_hidden_act: v2_config.pooler_hidden_act,
            pooler_hidden_size: v2_config.pooler_hidden_size,
            layer_norm_eps: v2_config.layer_norm_eps,
            pad_token_id: v2_config.pad_token_id,
            relative_attention: v2_config.relative_attention,
            max_relative_positions: v2_config.max_relative_positions,
            embedding_size: v2_config.embedding_size,
            talking_head: v2_config.talking_head,
            output_hidden_states: v2_config.output_hidden_states,
            output_attentions: v2_config.output_attentions,
            classifier_activation: v2_config.classifier_activation,
            classifier_dropout: v2_config.classifier_dropout,
            is_decoder: v2_config.is_decoder,
            id2label: v2_config.id2label.clone(),
            label2id: v2_config.label2id.clone(),
            share_att_key: v2_config.share_att_key,
            position_buckets: v2_config.position_buckets,
        }
    }
}

/// # DeBERTa V2 Base model
/// Base architecture for DeBERTa V2 models. Task-specific models will be built from this common base model
/// It is made of the following blocks:
/// - `embeddings`: `DeBERTa` V2 embeddings
/// - `encoder`: `DeBERTaV2Encoder` (transformer) made of a vector of layers.
pub struct DebertaV2Model {
    embeddings: DebertaV2Embeddings,
    encoder: DebertaV2Encoder,
}

impl DebertaV2Model {
    /// Build a new `DebertaV2Model`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the BERT model
    /// * `config` - `DebertaV2Config` object defining the model architecture and decoder status
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::deberta_v2::{DebertaV2Config, DebertaV2Model};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = DebertaV2Config::from_file(config_path);
    /// let model: DebertaV2Model = DebertaV2Model::new(&p.root() / "deberta", &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &DebertaV2Config) -> DebertaV2Model
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let embeddings = DebertaV2Embeddings::new(p / "embeddings", &config.into());
        let encoder = DebertaV2Encoder::new(p / "encoder", config);

        DebertaV2Model {
            embeddings,
            encoder,
        }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see `input_embeds`)
    /// * `attention_mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `token_type_ids` - Optional segment id of shape (*batch size*, *sequence_length*). Convention is value of 0 for the first sentence (incl. *SEP*) and 1 for the second sentence. If None set to 0.
    /// * `position_ids` - Optional position ids of shape (*batch size*, *sequence_length*). If None, will be incremented from 0.
    /// * `input_embeds` - Optional pre-computed input embeddings of shape (*batch size*, *sequence_length*, *hidden_size*). If None, input ids must be provided (see `input_ids`)
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `DebertaV2Output` containing:
    ///   - `hidden_state` - `Tensor` of shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `all_hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `all_attentions` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rust_bert::deberta_v2::{DebertaV2Model, DebertaV2Config};
    /// # use tch::{nn, Device, Tensor, no_grad, Kind};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # let config_path = Path::new("path/to/config.json");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = DebertaV2Config::from_file(config_path);
    /// # let model = DebertaV2Model::new(&vs.root(), &config);
    /// let (batch_size, sequence_length) = (64, 128);
    /// let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Kind::Int64, device));
    /// let attention_mask = Tensor::ones(&[batch_size, sequence_length], (Kind::Int64, device));
    /// let token_type_ids = Tensor::zeros(&[batch_size, sequence_length], (Kind::Int64, device));
    /// let position_ids = Tensor::arange(sequence_length, (Kind::Int64, device))
    ///     .expand(&[batch_size, sequence_length], true);
    ///
    /// let model_output = no_grad(|| {
    ///     model
    ///         .forward_t(
    ///             Some(&input_tensor),
    ///             Some(&attention_mask),
    ///             Some(&token_type_ids),
    ///             Some(&position_ids),
    ///             None,
    ///             false,
    ///         )
    ///         .unwrap()
    /// });
    /// ```
    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        train: bool,
    ) -> Result<DebertaV2ModelOutput, RustBertError> {
        let (input_shape, device) =
            get_shape_and_device_from_ids_embeddings_pair(input_ids, input_embeds)?;

        let calc_attention_mask = if attention_mask.is_none() {
            Some(Tensor::ones(input_shape.as_slice(), (Kind::Bool, device)))
        } else {
            None
        };

        let attention_mask =
            attention_mask.unwrap_or_else(|| calc_attention_mask.as_ref().unwrap());

        let embedding_output = self.embeddings.forward_t(
            input_ids,
            token_type_ids,
            position_ids,
            attention_mask,
            input_embeds,
            train,
        )?;

        let encoder_output =
            self.encoder
                .forward_t(&embedding_output, attention_mask, None, None, train)?;

        Ok(encoder_output)
    }
}

/// # DeBERTa V2 for masked language model
/// Base DeBERTa V2 model with a masked language model head to predict missing tokens, for example `"Looks like one [MASK] is missing" -> "person"`
/// It is made of the following blocks:
/// - `deberta`: Base DeBERTa V2 model
/// - `cls`: LM prediction head
pub struct DebertaV2ForMaskedLM {
    deberta: DebertaV2Model,
    cls: DebertaLMPredictionHead,
}

impl DebertaV2ForMaskedLM {
    /// Build a new `DebertaV2ForMaskedLM`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the BertForMaskedLM model
    /// * `config` - `DebertaConfig` object defining the model architecture and vocab size
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::deberta_v2::{DebertaV2Config, DebertaV2ForMaskedLM};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = DebertaV2Config::from_file(config_path);
    /// let model = DebertaV2ForMaskedLM::new(&p.root(), &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &DebertaV2Config) -> DebertaV2ForMaskedLM
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let deberta = DebertaV2Model::new(p / "deberta", config);
        let cls =
            DebertaLMPredictionHead::new(p.sub("cls").sub("predictions"), &config.into(), false);

        DebertaV2ForMaskedLM { deberta, cls }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see *input_embeds*)
    /// * `attention_mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `token_type_ids` -Optional segment id of shape (*batch size*, *sequence_length*). Convention is value of 0 for the first sentence (incl. *SEP*) and 1 for the second sentence. If None set to 0.
    /// * `position_ids` - Optional position ids of shape (*batch size*, *sequence_length*). If None, will be incremented from 0.
    /// * `input_embeds` - Optional pre-computed input embeddings of shape (*batch size*, *sequence_length*, *hidden_size*). If None, input ids must be provided (see *input_ids*)
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `DebertaMaskedLMOutput` containing:
    ///   - `prediction_scores` - `Tensor` of shape (*batch size*, *sequence_length*, *vocab_size*)
    ///   - `all_hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `all_attentions` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rust_bert::deberta_v2::{DebertaV2ForMaskedLM, DebertaV2Config};
    /// # use tch::{nn, Device, Tensor, no_grad, Kind};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # let config_path = Path::new("path/to/config.json");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = DebertaV2Config::from_file(config_path);
    /// # let model = DebertaV2ForMaskedLM::new(&vs.root(), &config);
    /// let (batch_size, sequence_length) = (64, 128);
    /// let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Kind::Int64, device));
    /// let mask = Tensor::zeros(&[batch_size, sequence_length], (Kind::Int64, device));
    /// let token_type_ids = Tensor::zeros(&[batch_size, sequence_length], (Kind::Int64, device));
    /// let position_ids = Tensor::arange(sequence_length, (Kind::Int64, device))
    ///     .expand(&[batch_size, sequence_length], true);
    ///
    /// let model_output = no_grad(|| {
    ///     model.forward_t(
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
        attention_mask: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        train: bool,
    ) -> Result<DebertaV2MaskedLMOutput, RustBertError> {
        let model_outputs = self.deberta.forward_t(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            input_embeds,
            train,
        )?;

        let logits = model_outputs.hidden_state.apply(&self.cls);
        Ok(DebertaV2MaskedLMOutput {
            logits,
            all_hidden_states: model_outputs.all_hidden_states,
            all_attentions: model_outputs.all_attentions,
        })
    }
}

/// # DeBERTa V2 for sequence classification
/// Base DeBERTa V2 model with a classifier head to perform sentence or document-level classification
/// It is made of the following blocks:
/// - `deberta`: Base Deberta (V2) Model
/// - `classifier`: BERT linear layer for classification
pub struct DebertaV2ForSequenceClassification {
    deberta: DebertaV2Model,
    pooler: ContextPooler,
    classifier: nn::Linear,
    dropout: XDropout,
}

impl DebertaV2ForSequenceClassification {
    /// Build a new `DebertaV2ForSequenceClassification`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the DebertaForSequenceClassification model
    /// * `config` - `DebertaV2Config` object defining the model architecture and number of classes
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::deberta_v2::{DebertaV2Config, DebertaV2ForSequenceClassification};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = DebertaV2Config::from_file(config_path);
    /// let model = DebertaV2ForSequenceClassification::new(&p.root(), &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &DebertaV2Config) -> DebertaV2ForSequenceClassification
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let deberta = DebertaV2Model::new(p / "deberta", config);
        let pooler = ContextPooler::new(p / "pooler", &config.into());
        let dropout = XDropout::new(
            config
                .classifier_dropout
                .unwrap_or(config.hidden_dropout_prob),
        );

        let num_labels = config
            .id2label
            .as_ref()
            .expect("num_labels not provided in configuration")
            .len() as i64;

        let classifier = nn::linear(
            p / "classifier",
            pooler.output_dim,
            num_labels,
            Default::default(),
        );

        DebertaV2ForSequenceClassification {
            deberta,
            pooler,
            classifier,
            dropout,
        }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see `input_embeds`)
    /// * `attention_mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `token_type_ids` -Optional segment id of shape (*batch size*, *sequence_length*). Convention is value of 0 for the first sentence (incl. *SEP*) and 1 for the second sentence. If None set to 0.
    /// * `position_ids` - Optional position ids of shape (*batch size*, *sequence_length*). If None, will be incremented from 0.
    /// * `input_embeds` - Optional pre-computed input embeddings of shape (*batch size*, *sequence_length*, *hidden_size*). If None, input ids must be provided (see `input_ids`)
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `DebertaV2SequenceClassificationOutput` containing:
    ///   - `logits` - `Tensor` of shape (*batch size*, *num_labels*)
    ///   - `all_hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `all_attentions` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rust_bert::deberta_v2::{DebertaV2ForSequenceClassification, DebertaV2Config};
    /// # use tch::{nn, Device, Tensor, no_grad, Kind};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # let config_path = Path::new("path/to/config.json");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = DebertaV2Config::from_file(config_path);
    /// # let model = DebertaV2ForSequenceClassification::new(&vs.root(), &config);
    /// let (batch_size, sequence_length) = (64, 128);
    /// let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Kind::Int64, device));
    /// let mask = Tensor::zeros(&[batch_size, sequence_length], (Kind::Int64, device));
    /// let token_type_ids = Tensor::zeros(&[batch_size, sequence_length], (Kind::Int64, device));
    /// let position_ids = Tensor::arange(sequence_length, (Kind::Int64, device))
    ///     .expand(&[batch_size, sequence_length], true);
    ///
    /// let model_output = no_grad(|| {
    ///     model.forward_t(
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
        attention_mask: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        train: bool,
    ) -> Result<DebertaV2SequenceClassificationOutput, RustBertError> {
        let base_model_output = self.deberta.forward_t(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            input_embeds,
            train,
        )?;

        let logits = base_model_output
            .hidden_state
            .apply_t(&self.pooler, train)
            .apply_t(&self.dropout, train)
            .apply(&self.classifier);

        Ok(DebertaV2SequenceClassificationOutput {
            logits,
            all_hidden_states: base_model_output.all_hidden_states,
            all_attentions: base_model_output.all_attentions,
        })
    }
}

/// # DeBERTa V2 for token classification (e.g. NER, POS)
/// Token-level classifier predicting a label for each token provided. Note that because of wordpiece tokenization, the labels predicted are
/// not necessarily aligned with words in the sentence.
/// It is made of the following blocks:
/// - `deberta`: Base DeBERTa (V2) model
/// - `dropout`: Dropout layer before the last token-level predictions layer
/// - `classifier`: Linear layer for token classification
pub struct DebertaV2ForTokenClassification {
    deberta: DebertaV2Model,
    dropout: Dropout,
    classifier: nn::Linear,
}

impl DebertaV2ForTokenClassification {
    /// Build a new `DebertaV2ForTokenClassification`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the Deberta V2 model
    /// * `config` - `DebertaV2Config` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::deberta_v2::{DebertaV2Config, DebertaV2ForTokenClassification};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = DebertaV2Config::from_file(config_path);
    /// let model = DebertaV2ForTokenClassification::new(&p.root(), &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &DebertaV2Config) -> DebertaV2ForTokenClassification
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let deberta = DebertaV2Model::new(p / "deberta", config);
        let dropout = Dropout::new(config.hidden_dropout_prob);
        let num_labels = config
            .id2label
            .as_ref()
            .expect("num_labels not provided in configuration")
            .len() as i64;
        let classifier = nn::linear(
            p / "classifier",
            config.hidden_size,
            num_labels,
            Default::default(),
        );

        DebertaV2ForTokenClassification {
            deberta,
            dropout,
            classifier,
        }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see `input_embeds`)
    /// * `attention_mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `token_type_ids` -Optional segment id of shape (*batch size*, *sequence_length*). Convention is value of 0 for the first sentence (incl. *SEP*) and 1 for the second sentence. If None set to 0.
    /// * `position_ids` - Optional position ids of shape (*batch size*, *sequence_length*). If None, will be incremented from 0.
    /// * `input_embeds` - Optional pre-computed input embeddings of shape (*batch size*, *sequence_length*, *hidden_size*). If None, input ids must be provided (see `input_ids`)
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `DebertaV2TokenClassificationOutput` containing:
    ///   - `logits` - `Tensor` of shape (*batch size*, *sequence_length*, *num_labels*)
    ///   - `all_hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `all_attentions` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rust_bert::deberta_v2::{DebertaV2ForTokenClassification, DebertaV2Config};
    /// # use tch::{nn, Device, Tensor, no_grad, Kind};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # let config_path = Path::new("path/to/config.json");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = DebertaV2Config::from_file(config_path);
    /// # let model = DebertaV2ForTokenClassification::new(&vs.root(), &config);
    /// let (batch_size, sequence_length) = (64, 128);
    /// let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Kind::Int64, device));
    /// let mask = Tensor::zeros(&[batch_size, sequence_length], (Kind::Int64, device));
    /// let token_type_ids = Tensor::zeros(&[batch_size, sequence_length], (Kind::Int64, device));
    /// let position_ids = Tensor::arange(sequence_length, (Kind::Int64, device))
    ///     .expand(&[batch_size, sequence_length], true);
    ///
    /// let model_output = no_grad(|| {
    ///     model.forward_t(
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
        attention_mask: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        train: bool,
    ) -> Result<DebertaV2TokenClassificationOutput, RustBertError> {
        let base_model_output = self.deberta.forward_t(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            input_embeds,
            train,
        )?;

        let logits = base_model_output
            .hidden_state
            .apply_t(&self.dropout, train)
            .apply(&self.classifier);

        Ok(DebertaV2TokenClassificationOutput {
            logits,
            all_hidden_states: base_model_output.all_hidden_states,
            all_attentions: base_model_output.all_attentions,
        })
    }
}

/// # DeBERTa V2 for question answering
/// Extractive question-answering model based on a DeBERTa V2 language model. Identifies the segment of a context that answers a provided question.
/// Please note that a significant amount of pre- and post-processing is required to perform end-to-end question answering.
/// See the question answering pipeline (also provided in this crate) for more details.
/// It is made of the following blocks:
/// - `deberta`: Base DeBERTa V2 model
/// - `qa_outputs`: Linear layer for question answering
pub struct DebertaV2ForQuestionAnswering {
    deberta: DebertaV2Model,
    qa_outputs: nn::Linear,
}

impl DebertaV2ForQuestionAnswering {
    /// Build a new `DebertaV2ForQuestionAnswering`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the BertForQuestionAnswering model
    /// * `config` - `DebertaV2Config` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::deberta_v2::{DebertaV2Config, DebertaV2ForQuestionAnswering};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = DebertaV2Config::from_file(config_path);
    /// let model = DebertaV2ForQuestionAnswering::new(&p.root(), &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &DebertaV2Config) -> DebertaV2ForQuestionAnswering
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let deberta = DebertaV2Model::new(p / "deberta", config);
        let num_labels = 2;
        let qa_outputs = nn::linear(
            p / "qa_outputs",
            config.hidden_size,
            num_labels,
            Default::default(),
        );

        DebertaV2ForQuestionAnswering {
            deberta,
            qa_outputs,
        }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see `input_embeds`)
    /// * `attention_mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `token_type_ids` -Optional segment id of shape (*batch size*, *sequence_length*). Convention is value of 0 for the first sentence (incl. *SEP*) and 1 for the second sentence. If None set to 0.
    /// * `position_ids` - Optional position ids of shape (*batch size*, *sequence_length*). If None, will be incremented from 0.
    /// * `input_embeds` - Optional pre-computed input embeddings of shape (*batch size*, *sequence_length*, *hidden_size*). If None, input ids must be provided (see `input_ids`)
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `DebertaQuestionAnsweringOutput` containing:
    ///   - `start_logits` - `Tensor` of shape (*batch size*, *sequence_length*) containing the logits for start of the answer
    ///   - `end_logits` - `Tensor` of shape (*batch size*, *sequence_length*) containing the logits for end of the answer
    ///   - `all_hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `all_attentions` - `Option<Vec<Vec<Tensor>>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rust_bert::deberta_v2::{DebertaV2ForQuestionAnswering, DebertaV2Config};
    /// # use tch::{nn, Device, Tensor, no_grad};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::Int64;
    /// # let config_path = Path::new("path/to/config.json");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = DebertaV2Config::from_file(config_path);
    /// # let model = DebertaV2ForQuestionAnswering::new(&vs.root(), &config);
    /// let (batch_size, sequence_length) = (64, 128);
    /// let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    /// let mask = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    /// let token_type_ids = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    /// let position_ids = Tensor::arange(sequence_length, (Int64, device))
    ///     .expand(&[batch_size, sequence_length], true);
    ///
    /// let model_output = no_grad(|| {
    ///     model.forward_t(
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
        attention_mask: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        train: bool,
    ) -> Result<DebertaV2QuestionAnsweringOutput, RustBertError> {
        let base_model_output = self.deberta.forward_t(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            input_embeds,
            train,
        )?;

        let sequence_output = base_model_output.hidden_state.apply(&self.qa_outputs);
        let logits = sequence_output.split(1, -1);
        let (start_logits, end_logits) = (&logits[0], &logits[1]);
        let start_logits = start_logits.squeeze_dim(-1);
        let end_logits = end_logits.squeeze_dim(-1);

        Ok(DebertaV2QuestionAnsweringOutput {
            start_logits,
            end_logits,
            all_hidden_states: base_model_output.all_hidden_states,
            all_attentions: base_model_output.all_attentions,
        })
    }
}

/// Container for the DeBERTa V2 model output.
pub type DebertaV2ModelOutput = DebertaModelOutput;

/// Container for the DeBERTa V2masked LM model output.
pub type DebertaV2MaskedLMOutput = DebertaMaskedLMOutput;

/// Container for the DeBERTa sequence classification model output.
pub type DebertaV2SequenceClassificationOutput = DebertaSequenceClassificationOutput;

/// Container for the DeBERTa token classification model output.
pub type DebertaV2TokenClassificationOutput = DebertaTokenClassificationOutput;

/// Container for the DeBERTa question answering model output.
pub type DebertaV2QuestionAnsweringOutput = DebertaQuestionAnsweringOutput;
