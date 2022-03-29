use std::collections::HashMap;
use std::convert::TryFrom;

use serde::{Deserialize, Serialize};

use crate::albert::AlbertConfig;
use crate::bert::BertConfig;
use crate::distilbert::DistilBertConfig;
use crate::t5::{FeedForwardProj, T5Config};
use crate::{Activation, Config, RustBertError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SBertTokenizerConfig {
    pub max_seq_length: usize,
    pub do_lower_case: bool,
}

impl Config for SBertTokenizerConfig {}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SBertModelConfig {
    DistillBert {
        activation: Activation,
        attention_dropout: f64,
        dim: i64,
        dropout: f64,
        hidden_dim: i64,
        initializer_range: f32,
        max_position_embeddings: i64,
        n_heads: i64,
        n_layers: i64,
        output_hidden_states: Option<bool>,
        output_past: Option<bool>,
        pad_token_id: i64,
        qa_dropout: f64,
        seq_classif_dropout: f64,
        sinusoidal_pos_embds: bool,
        tie_weights_: bool,
        vocab_size: i64,
        output_attentions: Option<bool>,
    },
    Albert {
        hidden_act: Activation,
        attention_probs_dropout_prob: f64,
        classifier_dropout_prob: Option<f64>,
        bos_token_id: i64,
        eos_token_id: i64,
        down_scale_factor: i64,
        embedding_size: i64,
        gap_size: i64,
        hidden_dropout_prob: f64,
        hidden_size: i64,
        initializer_range: f32,
        inner_group_num: i64,
        intermediate_size: i64,
        layer_norm_eps: Option<f64>,
        max_position_embeddings: i64,
        net_structure_type: i64,
        num_attention_heads: i64,
        num_hidden_groups: i64,
        num_hidden_layers: i64,
        num_memory_blocks: i64,
        pad_token_id: i64,
        type_vocab_size: i64,
        vocab_size: i64,
        output_attentions: Option<bool>,
        output_hidden_states: Option<bool>,
        is_decoder: Option<bool>,
        id2label: Option<HashMap<i64, String>>,
        label2id: Option<HashMap<String, i64>>,
    },
    Bert {
        attention_probs_dropout_prob: f64,
        hidden_act: Activation,
        hidden_dropout_prob: f64,
        hidden_size: i64,
        initializer_range: f32,
        intermediate_size: i64,
        max_position_embeddings: i64,
        num_attention_heads: i64,
        num_hidden_layers: i64,
        pad_token_id: i64,
        type_vocab_size: i64,
        vocab_size: i64,
        output_attentions: Option<bool>,
    },
    T5 {
        dropout_rate: f64,
        d_model: i64,
        d_ff: i64,
        d_kv: i64,
        decoder_start_token_id: Option<i64>,
        bos_token_id: Option<i64>,
        eos_token_id: Option<i64>,
        initializer_factor: f64,
        is_encoder_decoder: Option<bool>,
        layer_norm_epsilon: f64,
        num_heads: i64,
        num_layers: i64,
        output_past: Option<bool>,
        pad_token_id: Option<i64>,
        relative_attention_num_buckets: i64,
        vocab_size: i64,
        feed_forward_proj: Option<FeedForwardProj>,
        tie_word_embeddings: Option<bool>,
        output_attentions: Option<bool>,
    },
}

impl Config for SBertModelConfig {}

impl SBertModelConfig {
    pub fn pad_token_id(&self) -> i64 {
        match self {
            &Self::DistillBert { pad_token_id, .. } => pad_token_id,
            &Self::Bert { pad_token_id, .. } => pad_token_id,
            &Self::Albert { pad_token_id, .. } => pad_token_id,
            &Self::T5 { pad_token_id, .. } => pad_token_id.unwrap_or(0),
        }
    }

    pub fn nb_layers(&self) -> usize {
        match self {
            &Self::DistillBert { n_layers, .. } => n_layers as usize,
            &Self::Bert {
                num_hidden_layers, ..
            } => num_hidden_layers as usize,
            &Self::Albert {
                num_hidden_layers, ..
            } => num_hidden_layers as usize,
            &Self::T5 { num_layers, .. } => num_layers as usize,
        }
    }

    pub fn nb_heads(&self) -> usize {
        match self {
            &Self::DistillBert { n_heads, .. } => n_heads as usize,
            &Self::Bert {
                num_attention_heads,
                ..
            } => num_attention_heads as usize,
            &Self::Albert {
                num_attention_heads,
                ..
            } => num_attention_heads as usize,
            &Self::T5 { num_heads, .. } => num_heads as usize,
        }
    }

    pub fn output_attentions(&self) -> bool {
        match self {
            &Self::DistillBert {
                output_attentions, ..
            } => output_attentions.unwrap_or(false),
            &Self::Bert {
                output_attentions, ..
            } => output_attentions.unwrap_or(false),
            &Self::Albert {
                output_attentions, ..
            } => output_attentions.unwrap_or(false),
            &Self::T5 {
                output_attentions, ..
            } => output_attentions.unwrap_or(false),
        }
    }
}

impl TryFrom<SBertModelConfig> for DistilBertConfig {
    type Error = RustBertError;

    fn try_from(c: SBertModelConfig) -> Result<Self, Self::Error> {
        match c {
            SBertModelConfig::DistillBert {
                activation,
                attention_dropout,
                dim,
                dropout,
                hidden_dim,
                initializer_range,
                max_position_embeddings,
                n_heads,
                n_layers,
                output_hidden_states,
                output_past,
                qa_dropout,
                seq_classif_dropout,
                sinusoidal_pos_embds,
                tie_weights_,
                vocab_size,
                output_attentions,
                ..
            } => Ok(DistilBertConfig {
                activation,
                attention_dropout,
                dim,
                dropout,
                hidden_dim,
                id2label: None,
                initializer_range,
                is_decoder: None,
                label2id: None,
                max_position_embeddings,
                n_heads,
                n_layers,
                output_attentions,
                output_hidden_states,
                output_past,
                qa_dropout,
                seq_classif_dropout,
                sinusoidal_pos_embds,
                tie_weights_,
                torchscript: None,
                use_bfloat16: None,
                vocab_size,
            }),
            _ => Err(RustBertError::InvalidConfigurationError(
                "Expected DistillBertConfig based config".to_string(),
            )),
        }
    }
}

impl TryFrom<SBertModelConfig> for BertConfig {
    type Error = RustBertError;

    fn try_from(c: SBertModelConfig) -> Result<Self, Self::Error> {
        match c {
            SBertModelConfig::Bert {
                attention_probs_dropout_prob,
                hidden_act,
                hidden_dropout_prob,
                hidden_size,
                initializer_range,
                intermediate_size,
                max_position_embeddings,
                num_attention_heads,
                num_hidden_layers,
                type_vocab_size,
                vocab_size,
                output_attentions,
                ..
            } => Ok(BertConfig {
                hidden_act,
                attention_probs_dropout_prob,
                hidden_dropout_prob,
                hidden_size,
                initializer_range,
                intermediate_size,
                max_position_embeddings,
                num_attention_heads,
                num_hidden_layers,
                type_vocab_size,
                vocab_size,
                output_attentions,
                output_hidden_states: Some(true),
                is_decoder: None,
                id2label: None,
                label2id: None,
            }),
            _ => Err(RustBertError::InvalidConfigurationError(
                "Expected BertConfig based config".to_string(),
            )),
        }
    }
}

impl TryFrom<SBertModelConfig> for AlbertConfig {
    type Error = RustBertError;

    fn try_from(c: SBertModelConfig) -> Result<Self, Self::Error> {
        match c {
            SBertModelConfig::Albert {
                hidden_act,
                attention_probs_dropout_prob,
                classifier_dropout_prob,
                bos_token_id,
                eos_token_id,
                down_scale_factor,
                embedding_size,
                gap_size,
                hidden_dropout_prob,
                hidden_size,
                initializer_range,
                inner_group_num,
                intermediate_size,
                layer_norm_eps,
                max_position_embeddings,
                net_structure_type,
                num_attention_heads,
                num_hidden_groups,
                num_hidden_layers,
                num_memory_blocks,
                pad_token_id,
                type_vocab_size,
                vocab_size,
                output_attentions,
                output_hidden_states,
                is_decoder,
                id2label,
                label2id,
                ..
            } => Ok(AlbertConfig {
                hidden_act,
                attention_probs_dropout_prob,
                classifier_dropout_prob,
                bos_token_id,
                eos_token_id,
                down_scale_factor,
                embedding_size,
                gap_size,
                hidden_dropout_prob,
                hidden_size,
                initializer_range,
                inner_group_num,
                intermediate_size,
                layer_norm_eps,
                max_position_embeddings,
                net_structure_type,
                num_attention_heads,
                num_hidden_groups,
                num_hidden_layers,
                num_memory_blocks,
                pad_token_id,
                type_vocab_size,
                vocab_size,
                output_attentions,
                output_hidden_states,
                is_decoder,
                id2label,
                label2id,
            }),
            _ => Err(RustBertError::InvalidConfigurationError(
                "Expected AlbertConfig based config".to_string(),
            )),
        }
    }
}

impl TryFrom<SBertModelConfig> for T5Config {
    type Error = RustBertError;

    fn try_from(c: SBertModelConfig) -> Result<Self, Self::Error> {
        match c {
            SBertModelConfig::T5 {
                dropout_rate,
                d_model,
                d_ff,
                d_kv,
                decoder_start_token_id,
                bos_token_id,
                eos_token_id,
                initializer_factor,
                is_encoder_decoder,
                layer_norm_epsilon,
                num_heads,
                num_layers,
                output_past,
                pad_token_id,
                relative_attention_num_buckets,
                vocab_size,
                feed_forward_proj,
                tie_word_embeddings,
                ..
            } => Ok(T5Config {
                dropout_rate,
                d_model,
                d_ff,
                d_kv,
                decoder_start_token_id,
                bos_token_id,
                eos_token_id,
                initializer_factor,
                is_encoder_decoder,
                layer_norm_epsilon,
                num_heads,
                num_layers,
                output_past,
                pad_token_id,
                relative_attention_num_buckets,
                vocab_size,
                feed_forward_proj,
                tie_word_embeddings,
                task_specific_params: None,
            }),
            _ => Err(RustBertError::InvalidConfigurationError(
                "Expected T5Config based config".to_string(),
            )),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SBertModulesConfig(Vec<SBertModule>);

impl Config for SBertModulesConfig {}

impl std::ops::Deref for SBertModulesConfig {
    type Target = Vec<SBertModule>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SBertModule {
    pub idx: usize,
    pub name: String,
    pub path: String,
    #[serde(rename = "type")]
    #[serde(with = "serde_sbert_module_type")]
    pub mod_type: SBertModuleType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SBertModuleType {
    Transformer,
    Pooling,
    Dense,
    Normalize,
}

mod serde_sbert_module_type {
    use super::SBertModuleType;
    use serde::{de, Deserializer, Serializer};

    pub fn serialize<S>(module_type: &SBertModuleType, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&format!("sentence_transformers.models.{:?}", module_type))
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<SBertModuleType, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct SBertModuleTypeVisitor;

        impl de::Visitor<'_> for SBertModuleTypeVisitor {
            type Value = SBertModuleType;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a sbert module type")
            }

            fn visit_str<E: de::Error>(self, s: &str) -> Result<Self::Value, E> {
                s.split('.')
                    .last()
                    .map(|s| serde_json::from_value(serde_json::Value::String(s.to_string())))
                    .transpose()
                    .map_err(de::Error::custom)?
                    .ok_or_else(|| format!("Invalid SBertModuleType: {}", s))
                    .map_err(de::Error::custom)
            }
        }

        deserializer.deserialize_str(SBertModuleTypeVisitor)
    }
}
