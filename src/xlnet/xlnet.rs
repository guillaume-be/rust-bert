// Copyright 2018 Google AI and Google Brain team.
// Copyright 2018 Carnegie Mellon University Authors.
// Copyright 2020-present, the HuggingFace Inc. team.
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

use crate::common::dropout::Dropout;
use crate::xlnet::attention::LayerState;
use crate::xlnet::encoder::XLNetLayer;
use crate::{Config, RustBertError};
use serde::{Deserialize, Serialize};
use std::borrow::Borrow;
use std::collections::HashMap;
use tch::nn::Init;
use tch::{nn, Device, Kind, Tensor};

/// # XLNet Pretrained model weight files
pub struct XLNetModelResources;

/// # XLNet Pretrained model config files
pub struct XLNetConfigResources;

/// # XLNet Pretrained model vocab files
pub struct XLNetVocabResources;

impl XLNetModelResources {
    /// Shared under Apache 2.0 license by the XLNet Authors at https://github.com/zihangdai/xlnet. Modified with conversion to C-array format.
    pub const XLNET_BASE_V2: (&'static str, &'static str) = (
        "xlnet-base-cased/model.ot",
        "https://cdn.huggingface.co/xlnet-base-cased/rust_model.ot",
    );
}

impl XLNetConfigResources {
    /// Shared under Apache 2.0 license by the XLNet Authors at https://github.com/zihangdai/xlnet. Modified with conversion to C-array format.
    pub const XLNET_BASE_V2: (&'static str, &'static str) = (
        "xlnet-base-cased/config.json",
        "https://cdn.huggingface.co/xlnet-base-cased-config.json",
    );
}

impl XLNetVocabResources {
    /// Shared under Apache 2.0 license by the XLNet Authors at https://github.com/zihangdai/xlnet. Modified with conversion to C-array format.
    pub const XLNET_BASE_V2: (&'static str, &'static str) = (
        "xlnet-base-cased/spiece.model",
        "https://cdn.huggingface.co/xlnet-base-cased-spiece.model",
    );
}

#[allow(non_camel_case_types)]
#[derive(Clone, Debug, Serialize, Deserialize)]
/// # Activation function used in the attention layer and masked language model head
pub enum Activation {
    /// Gaussian Error Linear Unit ([Hendrycks et al., 2016,](https://arxiv.org/abs/1606.08415))
    gelu,
    /// Rectified Linear Unit
    relu,
    /// Swish ([Ramachandran, 2017](https://arxiv.org/abs/1710.05941))
    swish,
}

#[allow(non_camel_case_types)]
#[derive(Clone, Debug, Serialize, Deserialize, Copy)]
/// # Attention type for the model (bidirectional or unidirectional)
pub enum AttentionType {
    /// Bidirectional (XLNet)
    bi,
    /// Unidirectional (Transformer-XL)
    uni,
}

#[allow(non_camel_case_types)]
#[derive(Clone, Debug, Serialize, Deserialize, Copy)]
/// # Summary type for the model when used for summarization
pub enum SummaryType {
    /// Hidden state stored in the last token
    last,
    /// Hidden state stored in the first token
    first,
    /// Mean of all token hidden states
    mean,
    /// Hidden state stored in the CLS token
    cls_index,
}

#[derive(Debug, Serialize, Deserialize)]
/// # XLNet model configuration
/// Defines the XLNet model architecture (e.g. number of layers, hidden layer size, label mapping...)
pub struct XLNetConfig {
    pub vocab_size: i64,
    pub d_model: i64,
    pub n_layer: i64,
    pub d_head: i64,
    pub n_head: i64,
    pub d_inner: i64,
    pub ff_activation: Activation,
    pub untie_r: bool,
    pub attn_type: AttentionType,
    pub initializer_range: f32,
    pub layer_norm_eps: Option<f64>,
    pub dropout: f64,
    pub mem_len: Option<i64>,
    pub reuse_len: Option<i64>,
    pub clamp_len: Option<i64>,
    pub bi_data: bool,
    pub same_length: bool,
    pub summary_type: SummaryType,
    pub summary_use_proj: bool,
    pub summary_activation: Option<String>,
    pub summary_proj_to_labels: Option<bool>,
    pub summary_last_dropout: Option<f64>,
    pub start_n_top: Option<i64>,
    pub end_n_top: Option<i64>,
    pub use_cache: Option<bool>,
    pub bos_token_id: i64,
    pub eos_token_id: i64,
    pub pad_token_id: i64,
    pub id2label: Option<HashMap<i64, String>>,
    pub label2id: Option<HashMap<String, i64>>,
    pub output_attentions: Option<bool>,
    pub output_hidden_states: Option<bool>,
    pub chunk_size_feed_forward: Option<i64>,
}

impl Config<XLNetConfig> for XLNetConfig {}

pub struct XLNetModel {
    mem_len: Option<i64>,
    reuse_len: Option<i64>,
    same_length: bool,
    attention_type: AttentionType,
    bi_data: bool,
    clamp_len: Option<i64>,
    d_model: i64,
    word_embeddings: nn::Embedding,
    mask_emb: Tensor,
    layers: Vec<XLNetLayer>,
    dropout: Dropout,
}

impl XLNetModel {
    pub fn new<'p, P>(p: P, config: &XLNetConfig) -> XLNetModel
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let mem_len = config.mem_len;
        let reuse_len = config.reuse_len;
        let same_length = config.same_length;
        let attention_type = config.attn_type;
        let bi_data = config.bi_data;
        let clamp_len = config.clamp_len;
        let d_model = config.d_model;

        let word_embeddings: nn::Embedding = nn::embedding(
            p / "word_embedding",
            config.vocab_size,
            config.d_model,
            Default::default(),
        );

        let mask_emb = p.var("mask_emb", &[1, 1, config.d_model], Init::Const(0f64));
        let mut layers: Vec<XLNetLayer> = vec![];
        let p_layers = p / "layer";
        for layer_index in 0..config.n_layer {
            layers.push(XLNetLayer::new(&p_layers / layer_index, config));
        }

        let dropout = Dropout::new(config.dropout);

        XLNetModel {
            mem_len,
            reuse_len,
            same_length,
            attention_type,
            bi_data,
            clamp_len,
            d_model,
            word_embeddings,
            mask_emb,
            layers,
            dropout,
        }
    }

    fn create_mask(&self, q_len: i64, m_len: i64, device: Device) -> Tensor {
        let attention_mask = Tensor::ones(&[q_len, q_len], (Kind::Int64, device));
        let attention_mask_pad = Tensor::zeros(&[q_len, m_len], (Kind::Int64, device));
        let mask_up = attention_mask.triu(1);
        let mut output = Tensor::cat(&[&attention_mask_pad, &mask_up], 1);
        if self.same_length {
            let mask_low = attention_mask.tril(-1);
            output = Tensor::cat(
                &[
                    output.slice(1, 0, q_len, 1) + mask_low,
                    output.slice(1, q_len, q_len + m_len, 1),
                ],
                1,
            );
        }
        output
    }

    fn cache_mem(
        &self,
        current_output: &Tensor,
        previous_cached_state: Option<LayerState>,
    ) -> LayerState {
        let cutoff = match self.mem_len {
            None => 0i64,
            Some(0) => 0i64,
            Some(value) => -value,
        };
        let cur_length = current_output.size()[0];
        LayerState {
            prev_content: match (self.reuse_len, previous_cached_state) {
                (Some(value), Some(previous_past)) if value > 0 => {
                    let current_output = current_output.slice(0, 0, value, 1);
                    Tensor::cat(&[&previous_past.prev_content, &current_output], 0)
                        .slice(0, cutoff, cur_length, 1)
                }
                (Some(_), Some(previous_past)) | (None, Some(previous_past)) => {
                    Tensor::cat(&[&previous_past.prev_content, current_output], 0)
                        .slice(0, cutoff, cur_length, 1)
                }
                (Some(value), None) if value > 0 => {
                    let current_output = current_output.slice(0, 0, value, 1);
                    current_output.slice(0, cutoff, cur_length, 1)
                }
                (Some(_), None) | (None, None) => current_output.slice(0, cutoff, cur_length, 1),
            },
        }
    }

    fn positional_embedding(
        &self,
        position_sequence: &Tensor,
        inverse_frequency: &Tensor,
        batch_size: Option<i64>,
    ) -> Tensor {
        let sinusoid = Tensor::einsum("i,d->id", &[position_sequence, inverse_frequency]);
        let mut positional_embeddings =
            Tensor::cat(&[sinusoid.sin(), sinusoid.cos()], -1).unsqueeze(1);

        if let Some(bsz) = batch_size {
            positional_embeddings = positional_embeddings.expand(&[-1, bsz, -1], true)
        };
        positional_embeddings
    }

    fn relative_positional_encoding(
        &self,
        q_len: i64,
        k_len: i64,
        batch_size: Option<i64>,
        device: Device,
    ) -> Tensor {
        let frequency_sequence = Tensor::arange2(0, self.d_model, 2, (Kind::Float, device));
        let inverse_frequency = 1f64 / Tensor::pow2(10000f64, &(frequency_sequence / self.d_model));

        let (begin, end) = match self.attention_type {
            AttentionType::bi => (k_len, -q_len),
            AttentionType::uni => (k_len, -1),
        };
        let mut forward_positions_sequence = Tensor::arange2(begin, end, -1, (Kind::Float, device));
        match self.clamp_len {
            Some(clamp_value) if clamp_value > 0 => {
                forward_positions_sequence.clamp(-clamp_value, clamp_value);
            }
            _ => {}
        }
        if self.bi_data {
            let mut backward_positions_sequence =
                Tensor::arange2(-begin, -end, 1, (Kind::Float, device));
            match self.clamp_len {
                Some(clamp_value) if clamp_value > 0 => {
                    backward_positions_sequence.clamp(-clamp_value, clamp_value);
                }
                _ => {}
            }
            let bsz = match batch_size {
                Some(value) => Some(value / 2),
                None => None,
            };

            let forward_positions_embeddings =
                self.positional_embedding(&forward_positions_sequence, &inverse_frequency, bsz);
            let backward_positions_embeddings =
                self.positional_embedding(&backward_positions_sequence, &inverse_frequency, bsz);
            Tensor::cat(
                &[forward_positions_embeddings, backward_positions_embeddings],
                1,
            )
        } else {
            self.positional_embedding(&forward_positions_sequence, &inverse_frequency, batch_size)
        }
    }

    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        old_layer_states: Option<Vec<Option<LayerState>>>,
        perm_mask: Option<Tensor>,
        target_mapping: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        input_embeds: Option<Tensor>,
        train: bool,
    ) -> Result<(), RustBertError> {
        let (input_embeddings, input_shape) = match input_ids {
            Some(input_value) => match input_embeds {
                Some(_) => {
                    return Err(RustBertError::ValueError(
                        "Only one of input ids or input embeddings may be set".into(),
                    ));
                }
                None => (
                    input_value
                        .transpose(0, 1)
                        .contiguous()
                        .apply_t(&self.word_embeddings, train),
                    input_value.size(),
                ),
            },
            None => match input_embeds {
                Some(embeds) => {
                    let size = vec![embeds.size()[1], embeds.size()[0]];
                    (embeds.transpose(0, 1).contiguous(), size)
                }
                None => {
                    return Err(RustBertError::ValueError(
                        "At least one of input ids or input embeddings must be set".into(),
                    ));
                }
            },
        };

        let token_type_ids = match token_type_ids {
            Some(token_type_ids) => Some(token_type_ids.transpose(0, 1).contiguous()),
            None => None,
        };
        let attention_mask = match attention_mask {
            Some(attention_mask) => Some(attention_mask.transpose(0, 1).contiguous()),
            None => None,
        };
        let perm_mask = match perm_mask {
            Some(perm_mask) => Some(perm_mask.permute(&[1, 2, 0]).contiguous()),
            None => None,
        };
        let target_mapping = match target_mapping {
            Some(target_mapping) => Some(target_mapping.permute(&[1, 2, 0]).contiguous()),
            None => None,
        };

        let m_len = if let Some(mems) = &old_layer_states {
            if let Some(mem_0) = &mems[0] {
                mem_0.prev_content.size()[0]
            } else {
                0
            }
        } else {
            0
        };
        let (q_len, batch_size) = (input_shape[0], input_shape[1]);
        let k_len = q_len + m_len;

        let mut attn_mask = match self.attention_type {
            AttentionType::uni => Some(
                self.create_mask(q_len, m_len, input_embeddings.device())
                    .unsqueeze(-1)
                    .unsqueeze(-1),
            ),
            AttentionType::bi => None,
        };

        let input_mask: Option<Tensor> = if let Some(attention_mask) = attention_mask {
            Some(1.0 - attention_mask)
        } else {
            None
        };

        let mut data_mask: Option<Tensor> = match (input_mask, perm_mask) {
            (Some(input_mask_value), Some(perm_mask_value)) => {
                Some(input_mask_value.unsqueeze(0) + perm_mask_value)
            }
            (Some(input_mask_value), None) => Some(input_mask_value.unsqueeze(0)),
            (None, Some(perm_mask_value)) => Some(perm_mask_value),
            (None, None) => None,
        };

        let (data_mask, attn_mask) = if let Some(data_mask) = data_mask {
            let data_mask = if m_len > 0 {
                let mems_mask = Tensor::zeros(
                    &[data_mask.size()[0], m_len, batch_size],
                    (Kind::Bool, data_mask.device()),
                );
                Tensor::cat(&[mems_mask, data_mask], 1)
            } else {
                data_mask
            };
            let attn_mask = Some(if let Some(attn_mask) = attn_mask {
                attn_mask + data_mask.unsqueeze(-1)
            } else {
                data_mask.unsqueeze(-1)
            });
            (Some(data_mask), attn_mask)
        } else {
            (data_mask, attn_mask)
        };

        Ok(())
    }
}
