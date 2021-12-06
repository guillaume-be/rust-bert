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

use crate::common::dropout::XDropout;
use crate::common::kind::get_negative_infinity;
use crate::{Activation, Config, RustBertError};
use serde::de::{SeqAccess, Visitor};
use serde::{de, Deserialize, Deserializer, Serialize};
use std::borrow::Borrow;
use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;
use tch::nn::{Init, Module};
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
}

impl DebertaConfigResources {
    /// Shared under MIT license by the Microsoft team at <https://huggingface.co/microsoft/deberta-base>. Modified with conversion to C-array format.
    pub const DEBERTA_BASE: (&'static str, &'static str) = (
        "deberta-base/config",
        "https://huggingface.co/microsoft/deberta-base/resolve/main/config.json",
    );
}

impl DebertaVocabResources {
    /// Shared under MIT license by the Microsoft team at <https://huggingface.co/microsoft/deberta-base>. Modified with conversion to C-array format.
    pub const DEBERTA_BASE: (&'static str, &'static str) = (
        "deberta-base/vocab",
        "https://huggingface.co/microsoft/deberta-base/resolve/main/vocab.json",
    );
}

impl DebertaMergesResources {
    /// Shared under MIT license by the Microsoft team at <https://huggingface.co/microsoft/deberta-base>. Modified with conversion to C-array format.
    pub const DEBERTA_BASE: (&'static str, &'static str) = (
        "deberta-base/merges",
        "https://huggingface.co/microsoft/deberta-base/resolve/main/merges.txt",
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
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PositionAttentionTypes {
    types: Vec<PositionAttentionType>,
}

impl FromStr for PositionAttentionTypes {
    type Err = RustBertError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let types = s
            .to_lowercase()
            .split('|')
            .map(|s| PositionAttentionType::from_str(s))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(PositionAttentionTypes { types })
    }
}

impl Default for PositionAttentionTypes {
    fn default() -> Self {
        PositionAttentionTypes { types: vec![] }
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
    pub position_biased_input: bool,
    #[serde(default, deserialize_with = "deserialize_attention_type")]
    pub pos_att_type: Option<PositionAttentionTypes>,
    pub pooler_dropout: Option<f64>,
    pub pooler_hidden: Option<Activation>,
    pub pooler_hidden_size: Option<i64>,
    pub layer_norm_eps: Option<f64>,
    pub pad_token_id: Option<i64>,
    pub relative_attention: Option<bool>,
    pub max_relative_positions: Option<i64>,
    pub talking_head: Option<bool>,
    pub output_attentions: Option<bool>,
    pub classifier_activation: Option<bool>,
    pub is_decoder: Option<bool>,
    pub id2label: Option<HashMap<i64, String>>,
    pub label2id: Option<HashMap<String, i64>>,
}

fn deserialize_attention_type<'de, D>(
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

#[derive(Debug)]
pub struct DebertaLayerNorm {
    weight: Tensor,
    bias: Tensor,
    variance_epsilon: f64,
}

impl DebertaLayerNorm {
    pub fn new<'p, P>(p: P, size: i64, variance_epsilon: f64) -> DebertaLayerNorm
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

pub struct DebertaSelfOutput {
    dense: nn::Linear,
    layer_norm: DebertaLayerNorm,
    dropout: XDropout,
}

impl DebertaSelfOutput {
    pub fn new<'p, P>(p: P, config: &DebertaConfig) -> DebertaSelfOutput
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
        let layer_norm = DebertaLayerNorm::new(
            p / "LayerNorm",
            config.hidden_size,
            config.layer_norm_eps.unwrap_or(1e-7),
        );
        let dropout = XDropout::new(config.hidden_dropout_prob);
        DebertaSelfOutput {
            dense,
            layer_norm,
            dropout,
        }
    }

    pub fn forward_t(&self, hidden_states: &Tensor, input_tensor: &Tensor, train: bool) -> Tensor {
        self.layer_norm.forward(
            &(hidden_states
                .apply(&self.dense)
                .apply_t(&self.dropout, train)
                + input_tensor),
        )
    }
}
