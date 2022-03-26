// Copyright 2020, Microsoft and the HuggingFace Inc. team.
// Copyright 2020 Guillaume Becquin
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::bert::{
    BertQuestionAnsweringOutput, BertSequenceClassificationOutput, BertTokenClassificationOutput,
};
use crate::common::activations::TensorFunction;
use crate::common::dropout::{Dropout, XDropout};
use crate::common::embeddings::get_shape_and_device_from_ids_embeddings_pair;
use crate::common::kind::get_negative_infinity;
use crate::deberta::embeddings::DebertaEmbeddings;
use crate::deberta::encoder::{DebertaEncoder, DebertaEncoderOutput};
use crate::{Activation, Config, RustBertError};
use serde::de::{SeqAccess, Visitor};
use serde::{de, Deserialize, Deserializer, Serialize};
use std::borrow::Borrow;
use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;
use tch::nn::{Init, Module, ModuleT};
use tch::{nn, Kind, Tensor};

/// # DeBERTa Pretrained model weight files
pub struct DebertaModelResources;

/// # DeBERTa Pretrained model config files
pub struct DebertaConfigResources;

/// # DeBERTa Pretrained model vocab files
pub struct DebertaVocabResources;

/// # DeBERTa Pretrained model merges files
pub struct DebertaMergesResources;

impl DebertaModelResources {
    /// Shared under MIT license by the Microsoft team at <https://huggingface.co/microsoft/deberta-base>. Modified with conversion to C-array format.
    pub const DEBERTA_BASE: (&'static str, &'static str) = (
        "deberta-base/model",
        "https://huggingface.co/microsoft/deberta-base/resolve/main/rust_model.ot",
    );
    /// Shared under MIT license by the Microsoft team at <https://huggingface.co/microsoft/deberta-base-mnli>. Modified with conversion to C-array format.
    pub const DEBERTA_BASE_MNLI: (&'static str, &'static str) = (
        "deberta-base-mnli/model",
        "https://huggingface.co/microsoft/deberta-base-mnli/resolve/main/rust_model.ot",
    );
}

impl DebertaConfigResources {
    /// Shared under MIT license by the Microsoft team at <https://huggingface.co/microsoft/deberta-base>. Modified with conversion to C-array format.
    pub const DEBERTA_BASE: (&'static str, &'static str) = (
        "deberta-base/config",
        "https://huggingface.co/microsoft/deberta-base/resolve/main/config.json",
    );
    /// Shared under MIT license by the Microsoft team at <https://huggingface.co/microsoft/deberta-base-mnli>. Modified with conversion to C-array format.
    pub const DEBERTA_BASE_MNLI: (&'static str, &'static str) = (
        "deberta-base-mnli/config",
        "https://huggingface.co/microsoft/deberta-base-mnli/resolve/main/config.json",
    );
}

impl DebertaVocabResources {
    /// Shared under MIT license by the Microsoft team at <https://huggingface.co/microsoft/deberta-base>. Modified with conversion to C-array format.
    pub const DEBERTA_BASE: (&'static str, &'static str) = (
        "deberta-base/vocab",
        "https://huggingface.co/microsoft/deberta-base/resolve/main/vocab.json",
    );
    /// Shared under MIT license by the Microsoft team at <https://huggingface.co/microsoft/deberta-base-mnli>. Modified with conversion to C-array format.
    pub const DEBERTA_BASE_MNLI: (&'static str, &'static str) = (
        "deberta-base-mnli/vocab",
        "https://huggingface.co/microsoft/deberta-base-mnli/resolve/main/vocab.json",
    );
}

impl DebertaMergesResources {
    /// Shared under MIT license by the Microsoft team at <https://huggingface.co/microsoft/deberta-base>. Modified with conversion to C-array format.
    pub const DEBERTA_BASE: (&'static str, &'static str) = (
        "deberta-base/merges",
        "https://huggingface.co/microsoft/deberta-base/resolve/main/merges.txt",
    );
    /// Shared under MIT license by the Microsoft team at <https://huggingface.co/microsoft/deberta-base-mnli>. Modified with conversion to C-array format.
    pub const DEBERTA_BASE_MNLI: (&'static str, &'static str) = (
        "deberta-base-mnli/merges",
        "https://huggingface.co/microsoft/deberta-base-mnli/resolve/main/merges.txt",
    );
}

#[allow(non_camel_case_types)]
#[derive(Clone, Debug, Serialize, Deserialize, Copy, PartialEq)]
/// # Position attention type to use for the DeBERTa model.
pub enum PositionAttentionType {
    p2c,
    c2p,
    p2p,
}

impl FromStr for PositionAttentionType {
    type Err = RustBertError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "p2c" => Ok(PositionAttentionType::p2c),
            "c2p" => Ok(PositionAttentionType::c2p),
            "p2p" => Ok(PositionAttentionType::p2p),
            _ => Err(RustBertError::InvalidConfigurationError(format!(
                "Position attention type `{}` not in accepted variants (`p2c`, `c2p`, `p2p`)",
                s
            ))),
        }
    }
}

#[allow(non_camel_case_types)]
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct PositionAttentionTypes {
    types: Vec<PositionAttentionType>,
}

impl FromStr for PositionAttentionTypes {
    type Err = RustBertError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let types = s
            .to_lowercase()
            .split('|')
            .map(PositionAttentionType::from_str)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(PositionAttentionTypes { types })
    }
}

impl PositionAttentionTypes {
    pub fn has_type(&self, attention_type: PositionAttentionType) -> bool {
        self.types
            .iter()
            .any(|self_type| *self_type == attention_type)
    }

    pub fn len(&self) -> usize {
        self.types.len()
    }
}

#[derive(Debug, Serialize, Deserialize)]
/// # DeBERTa model configuration
/// Defines the DeBERTa model architecture (e.g. number of layers, hidden layer size, label mapping...)
pub struct DebertaConfig {
    pub hidden_act: Activation,
    pub attention_probs_dropout_prob: f64,
    pub hidden_dropout_prob: f64,
    pub hidden_size: i64,
    pub initializer_range: f64,
    pub intermediate_size: i64,
    pub max_position_embeddings: i64,
    pub num_attention_heads: i64,
    pub num_hidden_layers: i64,
    pub type_vocab_size: i64,
    pub vocab_size: i64,
    pub position_biased_input: Option<bool>,
    #[serde(default, deserialize_with = "deserialize_attention_type")]
    pub pos_att_type: Option<PositionAttentionTypes>,
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
    pub share_att_key: Option<bool>,
    pub position_buckets: Option<i64>,
}

pub fn deserialize_attention_type<'de, D>(
    deserializer: D,
) -> Result<Option<PositionAttentionTypes>, D::Error>
where
    D: Deserializer<'de>,
{
    struct AttentionTypeVisitor;

    impl<'de> Visitor<'de> for AttentionTypeVisitor {
        type Value = PositionAttentionTypes;

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
            while let Some(attention_type) = seq.next_element::<String>()? {
                types.push(FromStr::from_str(attention_type.as_str()).unwrap())
            }
            Ok(PositionAttentionTypes { types })
        }
    }

    deserializer.deserialize_any(AttentionTypeVisitor).map(Some)
}

impl Config for DebertaConfig {}

pub fn x_softmax(input: &Tensor, mask: &Tensor, dim: i64) -> Tensor {
    let inverse_mask = ((1 - mask) as Tensor).to_kind(Kind::Bool);
    input
        .masked_fill(&inverse_mask, get_negative_infinity(input.kind()).unwrap())
        .softmax(dim, input.kind())
        .masked_fill(&inverse_mask, 0.0)
}

pub trait BaseDebertaLayerNorm {
    fn new<'p, P>(p: P, size: i64, variance_epsilon: f64) -> Self
    where
        P: Borrow<nn::Path<'p>>;
}

#[derive(Debug)]
pub struct DebertaLayerNorm {
    weight: Tensor,
    bias: Tensor,
    variance_epsilon: f64,
}

impl BaseDebertaLayerNorm for DebertaLayerNorm {
    fn new<'p, P>(p: P, size: i64, variance_epsilon: f64) -> DebertaLayerNorm
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let weight = p.var("weight", &[size], Init::Const(1.0));
        let bias = p.var("bias", &[size], Init::Const(0.0));
        DebertaLayerNorm {
            weight,
            bias,
            variance_epsilon,
        }
    }
}

impl Module for DebertaLayerNorm {
    fn forward(&self, hidden_states: &Tensor) -> Tensor {
        let input_type = hidden_states.kind();
        let hidden_states = hidden_states.to_kind(Kind::Float);
        let mean = hidden_states.mean_dim(&[-1], true, hidden_states.kind());
        let variance = (&hidden_states - &mean).pow_tensor_scalar(2.0).mean_dim(
            &[-1],
            true,
            hidden_states.kind(),
        );
        let hidden_states = (hidden_states - mean)
            / (variance + self.variance_epsilon)
                .sqrt()
                .to_kind(input_type);
        &self.weight * hidden_states + &self.bias
    }
}

/// # DeBERTa Base model
/// Base architecture for DeBERTa models. Task-specific models will be built from this common base model
/// It is made of the following blocks:
/// - `embeddings`: `DeBERTa` embeddings
/// - `encoder`: `DeBERTaEncoder` (transformer) made of a vector of layers.
pub struct DebertaModel {
    embeddings: DebertaEmbeddings,
    encoder: DebertaEncoder,
}

impl DebertaModel {
    /// Build a new `DebertaModel`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the BERT model
    /// * `config` - `DebertaConfig` object defining the model architecture and decoder status
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::deberta::{DebertaConfig, DebertaModel};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = DebertaConfig::from_file(config_path);
    /// let model: DebertaModel = DebertaModel::new(&p.root() / "deberta", &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &DebertaConfig) -> DebertaModel
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let embeddings = DebertaEmbeddings::new(p / "embeddings", config);
        let encoder = DebertaEncoder::new(p / "encoder", config);

        DebertaModel {
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
    /// * `DebertaOutput` containing:
    ///   - `hidden_state` - `Tensor` of shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `all_hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `all_attentions` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rust_bert::deberta::{DebertaModel, DebertaConfig};
    /// # use tch::{nn, Device, Tensor, no_grad, Kind};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # let config_path = Path::new("path/to/config.json");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = DebertaConfig::from_file(config_path);
    /// # let model = DebertaModel::new(&vs.root(), &config);
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
    ) -> Result<DebertaModelOutput, RustBertError> {
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

#[derive(Debug)]
struct DebertaPredictionHeadTransform {
    dense: nn::Linear,
    activation: TensorFunction,
    layer_norm: nn::LayerNorm,
}

impl DebertaPredictionHeadTransform {
    pub fn new<'p, P>(
        p: P,
        config: &DebertaConfig,
        transform_bias: bool,
    ) -> DebertaPredictionHeadTransform
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let dense = nn::linear(
            p / "dense",
            config.hidden_size,
            config.hidden_size,
            nn::LinearConfig {
                bias: transform_bias,
                ..Default::default()
            },
        );
        let activation = config.hidden_act.get_function();
        let layer_norm_config = nn::LayerNormConfig {
            eps: 1e-7,
            ..Default::default()
        };
        let layer_norm =
            nn::layer_norm(p / "LayerNorm", vec![config.hidden_size], layer_norm_config);

        DebertaPredictionHeadTransform {
            dense,
            activation,
            layer_norm,
        }
    }
}

impl Module for DebertaPredictionHeadTransform {
    fn forward(&self, hidden_states: &Tensor) -> Tensor {
        self.activation.get_fn()(&hidden_states.apply(&self.dense)).apply(&self.layer_norm)
    }
}

#[derive(Debug)]
pub(crate) struct DebertaLMPredictionHead {
    transform: DebertaPredictionHeadTransform,
    decoder: nn::Linear,
}

impl DebertaLMPredictionHead {
    pub fn new<'p, P>(p: P, config: &DebertaConfig, transform_bias: bool) -> DebertaLMPredictionHead
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let transform =
            DebertaPredictionHeadTransform::new(p / "transform", config, transform_bias);
        let decoder = nn::linear(
            p / "decoder",
            config.hidden_size,
            config.vocab_size,
            Default::default(),
        );

        DebertaLMPredictionHead { transform, decoder }
    }
}

impl Module for DebertaLMPredictionHead {
    fn forward(&self, hidden_states: &Tensor) -> Tensor {
        hidden_states.apply(&self.transform).apply(&self.decoder)
    }
}

/// # DeBERTa for masked language model
/// Base DeBERTa model with a masked language model head to predict missing tokens, for example `"Looks like one [MASK] is missing" -> "person"`
/// It is made of the following blocks:
/// - `deberta`: Base DeBERTa model
/// - `cls`: LM prediction head
pub struct DebertaForMaskedLM {
    deberta: DebertaModel,
    cls: DebertaLMPredictionHead,
}

impl DebertaForMaskedLM {
    /// Build a new `DebertaForMaskedLM`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the BertForMaskedLM model
    /// * `config` - `DebertaConfig` object defining the model architecture and vocab size
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::deberta::{DebertaConfig, DebertaForMaskedLM};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = DebertaConfig::from_file(config_path);
    /// let model = DebertaForMaskedLM::new(&p.root(), &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &DebertaConfig) -> DebertaForMaskedLM
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let deberta = DebertaModel::new(p / "deberta", config);
        let cls = DebertaLMPredictionHead::new(p.sub("cls").sub("predictions"), config, false);

        DebertaForMaskedLM { deberta, cls }
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
    /// # use rust_bert::deberta::{DebertaForMaskedLM, DebertaConfig};
    /// # use tch::{nn, Device, Tensor, no_grad, Kind};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # let config_path = Path::new("path/to/config.json");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = DebertaConfig::from_file(config_path);
    /// # let model = DebertaForMaskedLM::new(&vs.root(), &config);
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
    ) -> Result<DebertaMaskedLMOutput, RustBertError> {
        let model_outputs = self.deberta.forward_t(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            input_embeds,
            train,
        )?;

        let logits = model_outputs.hidden_state.apply(&self.cls);
        Ok(DebertaMaskedLMOutput {
            logits,
            all_hidden_states: model_outputs.all_hidden_states,
            all_attentions: model_outputs.all_attentions,
        })
    }
}

#[derive(Debug)]
pub struct ContextPooler {
    dense: nn::Linear,
    dropout: XDropout,
    activation: TensorFunction,
    pub output_dim: i64,
}

impl ContextPooler {
    pub fn new<'p, P>(p: P, config: &DebertaConfig) -> ContextPooler
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let pooler_hidden_size = config.pooler_hidden_size.unwrap_or(config.hidden_size);

        let dense = nn::linear(
            p / "dense",
            pooler_hidden_size,
            pooler_hidden_size,
            Default::default(),
        );
        let dropout = XDropout::new(config.pooler_dropout.unwrap_or(0.0));
        let activation = config
            .pooler_hidden_act
            .unwrap_or(Activation::gelu)
            .get_function();

        ContextPooler {
            dense,
            dropout,
            activation,
            output_dim: pooler_hidden_size,
        }
    }
}

impl ModuleT for ContextPooler {
    fn forward_t(&self, hidden_states: &Tensor, train: bool) -> Tensor {
        self.activation.get_fn()(
            &hidden_states
                .select(1, 0)
                .apply_t(&self.dropout, train)
                .apply(&self.dense),
        )
    }
}

/// # DeBERTa for sequence classification
/// Base DeBERTa model with a classifier head to perform sentence or document-level classification
/// It is made of the following blocks:
/// - `deberta`: Base BertModel
/// - `classifier`: BERT linear layer for classification
pub struct DebertaForSequenceClassification {
    deberta: DebertaModel,
    pooler: ContextPooler,
    classifier: nn::Linear,
    dropout: XDropout,
}

impl DebertaForSequenceClassification {
    /// Build a new `DebertaForSequenceClassification`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the DebertaForSequenceClassification model
    /// * `config` - `DebertaConfig` object defining the model architecture and number of classes
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::deberta::{DebertaConfig, DebertaForSequenceClassification};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = DebertaConfig::from_file(config_path);
    /// let model = DebertaForSequenceClassification::new(&p.root(), &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &DebertaConfig) -> DebertaForSequenceClassification
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let deberta = DebertaModel::new(p / "deberta", config);
        let pooler = ContextPooler::new(p / "pooler", config);
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

        DebertaForSequenceClassification {
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
    /// * `DebertaSequenceClassificationOutput` containing:
    ///   - `logits` - `Tensor` of shape (*batch size*, *num_labels*)
    ///   - `all_hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `all_attentions` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rust_bert::deberta::{DebertaForSequenceClassification, DebertaConfig};
    /// # use tch::{nn, Device, Tensor, no_grad, Kind};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # let config_path = Path::new("path/to/config.json");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = DebertaConfig::from_file(config_path);
    /// # let model = DebertaForSequenceClassification::new(&vs.root(), &config);
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
    ) -> Result<DebertaSequenceClassificationOutput, RustBertError> {
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

        Ok(DebertaSequenceClassificationOutput {
            logits,
            all_hidden_states: base_model_output.all_hidden_states,
            all_attentions: base_model_output.all_attentions,
        })
    }
}

/// # DeBERTa for token classification (e.g. NER, POS)
/// Token-level classifier predicting a label for each token provided. Note that because of wordpiece tokenization, the labels predicted are
/// not necessarily aligned with words in the sentence.
/// It is made of the following blocks:
/// - `deberta`: Base DeBERTa
/// - `dropout`: Dropout layer before the last token-level predictions layer
/// - `classifier`: Linear layer for token classification
pub struct DebertaForTokenClassification {
    deberta: DebertaModel,
    dropout: Dropout,
    classifier: nn::Linear,
}

impl DebertaForTokenClassification {
    /// Build a new `DebertaForTokenClassification`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the Deberta model
    /// * `config` - `DebertaConfig` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::deberta::{DebertaConfig, DebertaForTokenClassification};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = DebertaConfig::from_file(config_path);
    /// let model = DebertaForTokenClassification::new(&p.root(), &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &DebertaConfig) -> DebertaForTokenClassification
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let deberta = DebertaModel::new(p / "deberta", config);
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

        DebertaForTokenClassification {
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
    /// * `DebertaTokenClassificationOutput` containing:
    ///   - `logits` - `Tensor` of shape (*batch size*, *sequence_length*, *num_labels*)
    ///   - `all_hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `all_attentions` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rust_bert::deberta::{DebertaForTokenClassification, DebertaConfig};
    /// # use tch::{nn, Device, Tensor, no_grad, Kind};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # let config_path = Path::new("path/to/config.json");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = DebertaConfig::from_file(config_path);
    /// # let model = DebertaForTokenClassification::new(&vs.root(), &config);
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
    ) -> Result<DebertaTokenClassificationOutput, RustBertError> {
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

        Ok(DebertaTokenClassificationOutput {
            logits,
            all_hidden_states: base_model_output.all_hidden_states,
            all_attentions: base_model_output.all_attentions,
        })
    }
}

/// # DeBERTa for question answering
/// Extractive question-answering model based on a DeBERTa language model. Identifies the segment of a context that answers a provided question.
/// Please note that a significant amount of pre- and post-processing is required to perform end-to-end question answering.
/// See the question answering pipeline (also provided in this crate) for more details.
/// It is made of the following blocks:
/// - `deberta`: Base DeBERTa model
/// - `qa_outputs`: Linear layer for question answering
pub struct DebertaForQuestionAnswering {
    deberta: DebertaModel,
    qa_outputs: nn::Linear,
}

impl DebertaForQuestionAnswering {
    /// Build a new `DebertaForQuestionAnswering`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the BertForQuestionAnswering model
    /// * `config` - `DebertaConfig` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::deberta::{DebertaConfig, DebertaForQuestionAnswering};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = DebertaConfig::from_file(config_path);
    /// let model = DebertaForQuestionAnswering::new(&p.root(), &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &DebertaConfig) -> DebertaForQuestionAnswering
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let deberta = DebertaModel::new(p / "deberta", config);
        let num_labels = 2;
        let qa_outputs = nn::linear(
            p / "qa_outputs",
            config.hidden_size,
            num_labels,
            Default::default(),
        );

        DebertaForQuestionAnswering {
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
    /// # use rust_bert::deberta::{DebertaForQuestionAnswering, DebertaConfig};
    /// # use tch::{nn, Device, Tensor, no_grad};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::Int64;
    /// # let config_path = Path::new("path/to/config.json");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = DebertaConfig::from_file(config_path);
    /// # let model = DebertaForQuestionAnswering::new(&vs.root(), &config);
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
    ) -> Result<DebertaQuestionAnsweringOutput, RustBertError> {
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

        Ok(DebertaQuestionAnsweringOutput {
            start_logits,
            end_logits,
            all_hidden_states: base_model_output.all_hidden_states,
            all_attentions: base_model_output.all_attentions,
        })
    }
}

/// Container for the DeBERTa model output.
pub type DebertaModelOutput = DebertaEncoderOutput;

/// Container for the DeBERTa masked LM model output.
pub struct DebertaMaskedLMOutput {
    /// Logits for the vocabulary items at each sequence position
    pub logits: Tensor,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all intermediate layers
    pub all_attentions: Option<Vec<Tensor>>,
}

/// Container for the DeBERTa sequence classification model output.
pub type DebertaSequenceClassificationOutput = BertSequenceClassificationOutput;

/// Container for the DeBERTa token classification model output.
pub type DebertaTokenClassificationOutput = BertTokenClassificationOutput;

/// Container for the DeBERTa question answering model output.
pub type DebertaQuestionAnsweringOutput = BertQuestionAnsweringOutput;
