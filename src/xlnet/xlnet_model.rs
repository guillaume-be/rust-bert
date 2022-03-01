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

use crate::common::activations::Activation;
use crate::common::dropout::Dropout;
use crate::common::summary::{SequenceSummary, SummaryConfig, SummaryType};
use crate::pipelines::common::{ModelType, TokenizerOption};
use crate::pipelines::generation_utils::private_generation_utils::{
    PreparedInput, PrivateLanguageGenerator,
};
use crate::pipelines::generation_utils::{
    Cache, GenerateConfig, LMHeadModel, LMModelOutput, LanguageGenerator,
};
use crate::xlnet::attention::LayerState;
use crate::xlnet::encoder::XLNetLayer;
use crate::{Config, RustBertError};
use rust_tokenizers::tokenizer::XLNetTokenizer;
use rust_tokenizers::vocab::XLNetVocab;
use serde::{Deserialize, Serialize};
use std::borrow::{Borrow, BorrowMut};
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
    /// Shared under Apache 2.0 license by the XLNet Authors at <https://github.com/zihangdai/xlnet>. Modified with conversion to C-array format.
    pub const XLNET_BASE_CASED: (&'static str, &'static str) = (
        "xlnet-base-cased/model",
        "https://huggingface.co/xlnet-base-cased/resolve/main/rust_model.ot",
    );
}

impl XLNetConfigResources {
    /// Shared under Apache 2.0 license by the XLNet Authors at <https://github.com/zihangdai/xlnet>. Modified with conversion to C-array format.
    pub const XLNET_BASE_CASED: (&'static str, &'static str) = (
        "xlnet-base-cased/config",
        "https://huggingface.co/xlnet-base-cased/resolve/main/config.json",
    );
}

impl XLNetVocabResources {
    /// Shared under Apache 2.0 license by the XLNet Authors at <https://github.com/zihangdai/xlnet>. Modified with conversion to C-array format.
    pub const XLNET_BASE_CASED: (&'static str, &'static str) = (
        "xlnet-base-cased/spiece",
        "https://huggingface.co/xlnet-base-cased/resolve/main/spiece.model",
    );
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

#[derive(Debug, Serialize, Deserialize, Clone)]
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
    pub summary_type: Option<SummaryType>,
    pub summary_use_proj: Option<bool>,
    pub summary_activation: Option<Activation>,
    pub summary_proj_to_labels: Option<bool>,
    pub summary_first_dropout: Option<f64>,
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

impl Config for XLNetConfig {}

/// # XLNet Base model
/// Base architecture for XLNet models. Task-specific models will be built from this common base model
/// It is made of the following blocks:
/// - `word_embeddings`: Word embeddings
/// - `mask_emb`: Embedding for the query stream
/// - `layers`: Vector of `XLNetLayer`. Each layer is made of a self-attention layers on the visible and hidden states and a post-attention layer
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
    output_attentions: bool,
    output_hidden_states: bool,
    use_cache: bool,
}

impl XLNetModel {
    /// Build a new `XLNetModel`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the XLNet model
    /// * `config` - `XLNetConfig` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::xlnet::{XLNetConfig, XLNetModel};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = XLNetConfig::from_file(config_path);
    /// let xlnet_model = XLNetModel::new(&p.root(), &config);
    /// ```
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
        let use_cache = config.use_cache.unwrap_or(true);
        let output_attentions = config.output_attentions.unwrap_or(false);
        let output_hidden_states = config.output_hidden_states.unwrap_or(false);
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
            output_attentions,
            output_hidden_states,
            use_cache,
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
        previous_cached_state: &Option<LayerState>,
    ) -> LayerState {
        let cutoff = match self.mem_len {
            None => 0i64,
            Some(0) => 0i64,
            Some(value) => -value,
        };
        let mut cur_length = current_output.size()[0];
        LayerState {
            prev_content: match (self.reuse_len, previous_cached_state) {
                (Some(value), Some(previous_past)) if value > 0 => {
                    let current_output = current_output.slice(0, 0, value, 1);
                    cur_length += &previous_past.prev_content.size()[0];
                    Tensor::cat(&[&previous_past.prev_content, &current_output], 0)
                        .slice(0, cutoff, cur_length, 1)
                }
                (Some(_), Some(previous_past)) | (None, Some(previous_past)) => {
                    cur_length += &previous_past.prev_content.size()[0];
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
        kind: Kind,
        device: Device,
    ) -> Tensor {
        let frequency_sequence =
            Tensor::arange_start_step(0, self.d_model, 2, (Kind::Float, device));
        let inverse_frequency =
            1f64 / Tensor::pow_scalar(10000f64, &(frequency_sequence / self.d_model));
        let (begin, end) = match self.attention_type {
            AttentionType::bi => (k_len, -q_len),
            AttentionType::uni => (k_len, -1),
        };
        let mut forward_positions_sequence =
            Tensor::arange_start_step(begin, end, -1, (Kind::Float, device));
        match self.clamp_len {
            Some(clamp_value) if clamp_value > 0 => {
                let _ = forward_positions_sequence.clamp_(-clamp_value, clamp_value);
            }
            _ => {}
        }
        let position_embeddings = if self.bi_data {
            let mut backward_positions_sequence =
                Tensor::arange_start(-begin, -end, (Kind::Float, device));
            match self.clamp_len {
                Some(clamp_value) if clamp_value > 0 => {
                    let _ = backward_positions_sequence.clamp_(-clamp_value, clamp_value);
                }
                _ => {}
            }
            let bsz = batch_size.map(|value| value / 2);

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
        };
        position_embeddings.to_kind(kind)
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). This or `input_embeds` must be provided.
    /// * `attention_mask` - Optional attention mask of shape (*batch size*, *sequence_length*) for the encoder positions. Positions with a mask with value 0 will be masked.
    /// * `perm_mask` - Optional tensor of shape (*batch size*, *sequence_length*, *sequence_length*). Mask to indicate the attention pattern for each input token (only used for pre-training over permutations, rather than simple token masking).
    /// * `target_mapping ` - Optional tensor of shape (*batch size*, *num_tokens*, *sequence_length*) indicating the position of the masked words to predict.
    /// * `token_type_ids` - Optional tensor (*batch size*, *sequence_length*) indicating the sentence ID of the token (0: first sentence, 1: second sentence).
    /// * `input_embeds` - Optional input tensor of shape (*batch size*, *sequence_length*, *embeddings dimension*). This or `input_ids` must be provided.
    /// * `old_layer_states` - Optional vector of length `n_layer` containing optional `LayerStates` containing the last calculated content for the attention layers. This avoids recomputing attention weights at past positions and speeds up decoding.
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `XLNetModelOutput` containing:
    ///   - `hidden_state` - `Tensor` of shape (*batch size*, *sequence_length*, *hidden_size*) representing the activations of the last hidden state
    ///   - `next_cache` - `Option<Vec<Option<LayerState>>>` of length *n_layer* containing the past content for the the attention layers with shape (*past_sequence_length*, *batch size*, *hidden_size*)
    ///   - `all_hidden_states` - `Option<Vec<(Tensor, Option<Tensor>)>>` of length *n_layer* with shape (*batch size*, *sequence_length*, *hidden_size*) (with optional query stream states if used)
    ///   - `all_attentions` - `Option<Vec<(Tensor, Option<Tensor>)>>` of length *n_layer* with shape (*batch size*, *sequence_length*, *hidden_size*) (with optional query stream states if used)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use tch::{nn, Device, Tensor, no_grad, Kind};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::{Int64, Double};
    /// use rust_bert::xlnet::{XLNetConfig, XLNetModel};
    /// # let config_path = Path::new("path/to/config.json");
    /// # let vocab_path = Path::new("path/to/vocab.txt");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = XLNetConfig::from_file(config_path);
    /// # let xlnet_model: XLNetModel = XLNetModel::new(&vs.root(), &config);
    /// let (batch_size, sequence_length) = (64, 128);
    /// let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    /// let attention_mask = Tensor::ones(&[batch_size, sequence_length], (Int64, device));
    /// let target_tensor = Tensor::ones(&[batch_size, sequence_length], (Int64, device));
    /// let target_mapping = Tensor::zeros(&[64, 1, 128], (Kind::Float, device));
    /// let _ = target_mapping.narrow(2, 3, 1).fill_(1.0);
    ///
    /// let model_output = no_grad(|| {
    ///     xlnet_model.forward_t(
    ///         Some(&input_tensor),
    ///         Some(&attention_mask),
    ///         None,
    ///         Some(&target_mapping),
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
        attention_mask: Option<&Tensor>,
        old_layer_states: Option<Vec<Option<LayerState>>>,
        perm_mask: Option<&Tensor>,
        target_mapping: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        train: bool,
    ) -> Result<XLNetModelOutput, RustBertError> {
        let (word_emb_k, input_shape) = match (input_ids, input_embeds) {
            (Some(_), Some(_)) => {
                return Err(RustBertError::ValueError(
                    "Only one of input ids or input embeddings may be set".into(),
                ));
            }
            (Some(input_value), None) => {
                let size = input_value.size();
                (
                    input_value
                        .transpose(0, 1)
                        .contiguous()
                        .apply_t(&self.word_embeddings, train),
                    vec![size[1], size[0]],
                )
            }
            (None, Some(embeds)) => {
                let size = vec![embeds.size()[1], embeds.size()[0]];
                (embeds.transpose(0, 1).contiguous(), size)
            }
            (None, None) => {
                return Err(RustBertError::ValueError(
                    "At least one of input ids or input embeddings must be set".into(),
                ));
            }
        };

        let token_type_ids =
            token_type_ids.map(|token_type_ids| token_type_ids.transpose(0, 1).contiguous());
        let attention_mask =
            attention_mask.map(|attention_mask| attention_mask.transpose(0, 1).contiguous());
        let perm_mask = perm_mask.map(|perm_mask| {
            perm_mask
                .to_kind(word_emb_k.kind())
                .permute(&[1, 2, 0])
                .contiguous()
        });
        let target_mapping = target_mapping.map(|target_mapping| {
            target_mapping
                .to_kind(word_emb_k.kind())
                .permute(&[1, 2, 0])
                .contiguous()
        });

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
                self.create_mask(q_len, m_len, word_emb_k.device())
                    .unsqueeze(-1)
                    .unsqueeze(-1),
            ),
            AttentionType::bi => None,
        };

        let input_mask: Option<Tensor> = attention_mask.map(|attention_mask| 1.0 - attention_mask);

        let mut data_mask: Option<Tensor> = match (input_mask, perm_mask) {
            (Some(input_mask_value), Some(perm_mask_value)) => {
                Some(input_mask_value.unsqueeze(0) + perm_mask_value)
            }
            (Some(input_mask_value), None) => Some(input_mask_value.unsqueeze(0)),
            (None, Some(perm_mask_value)) => Some(perm_mask_value),
            (None, None) => None,
        };

        if let Some(data_mask_value) = &data_mask {
            if m_len > 0 {
                let mems_mask = Tensor::zeros(
                    &[data_mask_value.size()[0], m_len, batch_size],
                    (Kind::Bool, data_mask_value.device()),
                );
                data_mask = Some(Tensor::cat(&[&mems_mask, data_mask_value], 1))
            }
            attn_mask = Some(if let Some(attn_mask) = attn_mask {
                attn_mask + data_mask.unwrap().unsqueeze(-1)
            } else {
                data_mask.unwrap().unsqueeze(-1)
            });
        }

        let non_tgt_mask = if let Some(attn_mask_value) = &attn_mask {
            let mut non_tgt_mask = -Tensor::eye(q_len, (Kind::Int64, attn_mask_value.device()));
            if m_len > 0 {
                non_tgt_mask = Tensor::cat(
                    &[
                        Tensor::zeros(&[q_len, m_len], (Kind::Int64, attn_mask_value.device())),
                        non_tgt_mask,
                    ],
                    -1,
                );
            }
            Some((attn_mask_value + non_tgt_mask.unsqueeze(-1).unsqueeze(-1)).gt(0))
        } else {
            None
        };

        let mut output_h = word_emb_k.apply_t(&self.dropout, train);
        let mut output_g = target_mapping.as_ref().map(|target_mapping_value| {
            (&self
                .mask_emb
                .expand(&[target_mapping_value.size()[0], batch_size, -1], true))
                .apply_t(&self.dropout, train)
        });

        let seg_mat = if let Some(token_type_ids_value) = token_type_ids {
            let cat_ids = if m_len > 0 {
                let mem_pad = Tensor::zeros(
                    &[m_len, batch_size],
                    (Kind::Int64, token_type_ids_value.device()),
                );
                Tensor::cat(&[mem_pad, token_type_ids_value.copy()], 0)
            } else {
                token_type_ids_value.copy()
            };
            let seg_mat = token_type_ids_value
                .unsqueeze(-1)
                .ne_tensor(&cat_ids.unsqueeze(0))
                .to_kind(Kind::Int64);
            Some(seg_mat.one_hot(2).to_kind(output_h.kind()))
        } else {
            None
        };

        let pos_emb = self
            .relative_positional_encoding(
                q_len,
                k_len,
                Some(batch_size),
                output_h.kind(),
                output_h.device(),
            )
            .apply_t(&self.dropout, train);

        let mut all_hidden_states: Option<Vec<(Tensor, Option<Tensor>)>> =
            if self.output_hidden_states {
                Some(vec![])
            } else {
                None
            };
        let mut all_attentions: Option<Vec<(Tensor, Option<Tensor>)>> = if self.output_attentions {
            Some(vec![])
        } else {
            None
        };

        let mut next_cache: Option<Vec<Option<LayerState>>> = if self.use_cache {
            if old_layer_states.is_some() {
                old_layer_states
            } else {
                Some(vec![None; self.layers.len()])
            }
        } else {
            None
        };

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let layer_state = match &next_cache {
                Some(values) => values[layer_idx].to_owned(),
                None => None,
            };
            if let Some(next_cache_value) = next_cache.borrow_mut() {
                next_cache_value[layer_idx] = Some(self.cache_mem(&output_h, &layer_state));
            }
            let temp = layer.forward_t(
                &output_h,
                output_g.as_ref(),
                non_tgt_mask.as_ref(),
                attn_mask.as_ref(),
                &pos_emb,
                seg_mat.as_ref(),
                layer_state,
                target_mapping.as_ref(),
                train,
            );
            output_h = temp.0;
            output_g = temp.1;
            let attention_probas_h = temp.2;
            let attention_probas_g = temp.3;
            if let Some(hidden_states) = all_hidden_states.borrow_mut() {
                hidden_states.push((
                    output_h.copy(),
                    output_g.as_ref().map(|output| output.copy()),
                ));
            };
            if let Some(attentions) = all_attentions.borrow_mut() {
                attentions.push((attention_probas_h.unwrap(), attention_probas_g));
            };
        }
        let hidden_state = if let Some(output_g_value) = output_g {
            output_g_value
        } else {
            output_h
        }
        .apply_t(&self.dropout, train)
        .permute(&[1, 0, 2])
        .contiguous();

        Ok(XLNetModelOutput {
            hidden_state,
            next_cache,
            all_hidden_states,
            all_attentions,
        })
    }
}

/// # XLNetLMHeadModel
/// XLNet model with a language model head for language generation tasks
/// It is made of the following blocks:
/// - `base_model`: `XLNetModel`
/// - `lm_head`: Linear language modeling head, projecting the hidden state logits to the vocabulary space
pub struct XLNetLMHeadModel {
    base_model: XLNetModel,
    lm_head: nn::Linear,
}

impl XLNetLMHeadModel {
    /// Build a new `XLNetLMHeadModel`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the XLNet model
    /// * `config` - `XLNetConfig` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::xlnet::{XLNetConfig, XLNetLMHeadModel};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = XLNetConfig::from_file(config_path);
    /// let xlnet_model = XLNetLMHeadModel::new(&p.root(), &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &XLNetConfig) -> XLNetLMHeadModel
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let base_model = XLNetModel::new(p / "transformer", config);
        let lm_head = nn::linear(
            p / "lm_loss",
            config.d_model,
            config.vocab_size,
            Default::default(),
        );

        XLNetLMHeadModel {
            base_model,
            lm_head,
        }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). This or `input_embeds` must be provided.
    /// * `attention_mask` - Optional attention mask of shape (*batch size*, *sequence_length*) for the encoder positions. Positions with a mask with value 0 will be masked.
    /// * `perm_mask` - Optional tensor of shape (*batch size*, *sequence_length*, *sequence_length*). Mask to indicate the attention pattern for each input token (only used for pre-training over permutations, rather than simple token masking).
    /// * `target_mapping ` - Optional tensor of shape (*batch size*, *num_tokens*, *sequence_length*) indicating the position of the masked words to predict.
    /// * `token_type_ids` - Optional tensor (*batch size*, *sequence_length*) indicating the sentence ID of the token (0: first sentence, 1: second sentence).
    /// * `input_embeds` - Optional input tensor of shape (*batch size*, *sequence_length*, *embeddings dimension*). This or `input_ids` must be provided.
    /// * `old_layer_states` - Optional vector of length `num_layers` containing optional `LayerStates` containing the last calculated content for the attention layers. This avoids recomputing attention weights at past positions and speeds up decoding.
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `LMModelOutput` containing:
    ///   - `lm_logits` - `Tensor` of shape (*batch size*, *sequence_length*, *vocab_size*) representing the logits for each vocab item and position
    ///   - `cache` - `XLNetCache` made of `Option<Vec<Option<LayerState>>>` of length *n_layers*  and shape (*past_sequence_length*, *batch size*, *hidden_size*) containing the previous content
    ///   - `encoder_hidden_states` - None
    ///   - `all_hidden_states` - `Option<Vec<Tensor>>` of length *n_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `all_attentions` - `Option<Vec<Tensor>>` of length *n_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use tch::{nn, Device, Tensor, no_grad, Kind};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::{Int64, Double};
    /// use rust_bert::xlnet::{XLNetConfig, XLNetLMHeadModel};
    /// # let config_path = Path::new("path/to/config.json");
    /// # let vocab_path = Path::new("path/to/vocab.txt");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = XLNetConfig::from_file(config_path);
    /// # let xlnet_model: XLNetLMHeadModel = XLNetLMHeadModel::new(&vs.root(), &config);
    /// let (batch_size, sequence_length) = (64, 128);
    /// let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    /// let attention_mask = Tensor::ones(&[batch_size, sequence_length], (Int64, device));
    /// let target_tensor = Tensor::ones(&[batch_size, sequence_length], (Int64, device));
    /// let target_mapping = Tensor::zeros(&[64, 1, 128], (Kind::Float, device));
    /// let _ = target_mapping.narrow(2, 3, 1).fill_(1.0);
    ///
    /// let model_output = no_grad(|| {
    ///     xlnet_model.forward_t(
    ///         Some(&input_tensor),
    ///         Some(&attention_mask),
    ///         None,
    ///         Some(&target_mapping),
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
        attention_mask: Option<&Tensor>,
        old_layer_states: Option<Vec<Option<LayerState>>>,
        perm_mask: Option<&Tensor>,
        target_mapping: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        train: bool,
    ) -> Result<LMModelOutput, RustBertError> {
        let base_model_output = self.base_model.forward_t(
            input_ids,
            attention_mask,
            old_layer_states,
            perm_mask,
            target_mapping,
            token_type_ids,
            input_embeds,
            train,
        )?;

        let lm_logits = base_model_output.hidden_state.apply(&self.lm_head);

        Ok(LMModelOutput {
            lm_logits,
            cache: Cache::XLNetCache(base_model_output.next_cache),
        })
    }
}

impl LMHeadModel for XLNetLMHeadModel {
    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). This or `input_embeds` must be provided.
    /// * `attention_mask` - Optional attention mask of shape (*batch size*, *sequence_length*) for the encoder positions. Positions with a mask with value 0 will be masked.
    /// * `perm_mask` - Optional tensor of shape (*batch size*, *sequence_length*, *sequence_length*). Mask to indicate the attention pattern for each input token (only used for pre-training over permutations, rather than simple token masking).
    /// * `target_mapping ` - Optional tensor of shape (*batch size*, *num_tokens*, *sequence_length*) indicating the position of the masked words to predict.
    /// * `token_type_ids` - Optional tensor (*batch size*, *sequence_length*) indicating the sentence ID of the token (0: first sentence, 1: second sentence).
    /// * `input_embeds` - Optional input tensor of shape (*batch size*, *sequence_length*, *embeddings dimension*). This or `input_ids` must be provided.
    /// * `old_layer_states` - Optional vector of length `num_layers` containing optional `LayerStates` containing the last calculated content for the attention layers. This avoids recomputing attention weights at past positions and speeds up decoding.
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `LMModelOutput` containing:
    ///   - `lm_logits` - `Tensor` of shape (*batch size*, *sequence_length*, *vocab_size*) representing the logits for each vocab item and position
    ///   - `cache` - `XLNetCache` made of `Option<Vec<Option<LayerState>>>` of length *n_layers*  and shape (*past_sequence_length*, *batch size*, *hidden_size*) containing the previous content
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use tch::{nn, Device, Tensor, no_grad, Kind};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::{Int64, Double};
    /// use rust_bert::xlnet::{XLNetConfig, XLNetLMHeadModel};
    /// # let config_path = Path::new("path/to/config.json");
    /// # let vocab_path = Path::new("path/to/vocab.txt");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = XLNetConfig::from_file(config_path);
    /// # let xlnet_model: XLNetLMHeadModel = XLNetLMHeadModel::new(&vs.root(), &config);
    /// let (batch_size, sequence_length) = (64, 128);
    /// let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    /// let attention_mask = Tensor::ones(&[batch_size, sequence_length], (Int64, device));
    /// let target_tensor = Tensor::ones(&[batch_size, sequence_length], (Int64, device));
    /// let target_mapping = Tensor::zeros(&[64, 1, 128], (Kind::Float, device));
    /// let _ = target_mapping.narrow(2, 3, 1).fill_(1.0);
    ///
    /// let model_output = no_grad(|| {
    ///     xlnet_model.forward_t(
    ///         Some(&input_tensor),
    ///         Some(&attention_mask),
    ///         None,
    ///         Some(&target_mapping),
    ///         None,
    ///         None,
    ///         None,
    ///         false,
    ///     )
    /// });
    /// ```
    fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        layer_past: Cache,
        attention_mask: Option<&Tensor>,
        _token_type_ids: Option<&Tensor>,
        _position_ids: Option<&Tensor>,
        _input_embeds: Option<&Tensor>,
        _encoder_outputs: Option<&Tensor>,
        decoder_input_ids: Option<&Tensor>,
        train: bool,
    ) -> Result<LMModelOutput, RustBertError> {
        match layer_past {
            Cache::XLNetCache(layer_past) => self.forward_t(
                input_ids,
                None,
                layer_past,
                attention_mask,
                // For XLNet the decoder_input_ids are used as a placeholder for the target mapping
                decoder_input_ids,
                None,
                None,
                train,
            ),
            Cache::None => self.forward_t(
                input_ids,
                None,
                None,
                attention_mask,
                // For XLNet the decoder_input_ids are used as a placeholder for the target mapping
                decoder_input_ids,
                None,
                None,
                train,
            ),
            _ => Err(RustBertError::ValueError(
                "Cache not compatible with XLNet Model".into(),
            )),
        }
    }
}

/// # XLNetForSequenceClassification
/// XLNet model with a classification head for sequence classification tasks
/// It is made of the following blocks:
/// - `base_model`: `XLNetModel`
/// - `sequence_summary`: `SequenceSummary` to pool the base model hidden states
/// - `logits_proj`: Linear layer projecting the hidden layer pooled output to the target space
pub struct XLNetForSequenceClassification {
    base_model: XLNetModel,
    sequence_summary: SequenceSummary,
    logits_proj: nn::Linear,
}

impl XLNetForSequenceClassification {
    /// Build a new `XLNetForSequenceClassification`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the XLNet model
    /// * `config` - `XLNetConfig` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::xlnet::{XLNetConfig, XLNetForSequenceClassification};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = XLNetConfig::from_file(config_path);
    /// let xlnet_model = XLNetForSequenceClassification::new(&p.root(), &config);
    /// ```
    pub fn new<'p, P>(
        p: P,
        config: &XLNetConfig,
    ) -> Result<XLNetForSequenceClassification, RustBertError>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let base_model = XLNetModel::new(p / "transformer", config);
        let sequence_summary =
            SequenceSummary::new(p / "sequence_summary", &SummaryConfig::from(config))?;
        let num_labels = config
            .id2label
            .as_ref()
            .expect("num_labels not provided in configuration")
            .len() as i64;

        let logits_proj = nn::linear(
            p / "logits_proj",
            config.d_model,
            num_labels,
            Default::default(),
        );

        Ok(XLNetForSequenceClassification {
            base_model,
            sequence_summary,
            logits_proj,
        })
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). This or `input_embeds` must be provided.
    /// * `attention_mask` - Optional attention mask of shape (*batch size*, *sequence_length*) for the encoder positions. Positions with a mask with value 0 will be masked.
    /// * `perm_mask` - Optional tensor of shape (*batch size*, *sequence_length*, *sequence_length*). Mask to indicate the attention pattern for each input token (only used for pre-training over permutations, rather than simple token masking).
    /// * `target_mapping ` - Optional tensor of shape (*batch size*, *num_tokens*, *sequence_length*) indicating the position of the masked words to predict.
    /// * `token_type_ids` - Optional tensor (*batch size*, *sequence_length*) indicating the sentence ID of the token (0: first sentence, 1: second sentence).
    /// * `input_embeds` - Optional input tensor of shape (*batch size*, *sequence_length*, *embeddings dimension*). This or `input_ids` must be provided.
    /// * `old_layer_states` - Optional vector of length `num_layers` containing optional `LayerStates` containing the last calculated content for the attention layers. This avoids recomputing attention weights at past positions and speeds up decoding.
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `XLNetSequenceClassificationOutput` containing:
    ///   - `logits` - `Tensor` of shape (*batch size*, *num_classes*) representing the logits for each batch item and class
    ///   - `next_cache` - `Option<Vec<Option<LayerState>>>` of length *n_layer* containing the past content for the the attention layers with shape (*past_sequence_length*, *batch size*, *hidden_size*)
    ///   - `all_hidden_states` - `Option<Vec<(Tensor, Option<Tensor>)>>` of length *n_layer* with shape (*batch size*, *sequence_length*, *hidden_size*) (with optional query stream states if used)
    ///   - `all_attentions` - `Option<Vec<(Tensor, Option<Tensor>)>>` of length *n_layer* with shape (*batch size*, *sequence_length*, *hidden_size*) (with optional query stream states if used)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use tch::{nn, Device, Tensor, no_grad, Kind};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::{Int64, Double};
    /// use rust_bert::xlnet::{XLNetConfig, XLNetForSequenceClassification};
    /// # fn main() -> anyhow::Result<()> {
    /// # let config_path = Path::new("path/to/config.json");
    /// # let vocab_path = Path::new("path/to/vocab.txt");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = XLNetConfig::from_file(config_path);
    /// # let xlnet_model: XLNetForSequenceClassification = XLNetForSequenceClassification::new(&vs.root(), &config)?;
    /// let (batch_size, sequence_length) = (64, 128);
    /// let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    /// let attention_mask = Tensor::ones(&[batch_size, sequence_length], (Int64, device));
    /// let target_tensor = Tensor::ones(&[batch_size, sequence_length], (Int64, device));
    /// let target_mapping = Tensor::zeros(&[64, 1, 128], (Kind::Float, device));
    /// let _ = target_mapping.narrow(2, 3, 1).fill_(1.0);
    ///
    /// let model_output = no_grad(|| {
    ///     xlnet_model.forward_t(
    ///         Some(&input_tensor),
    ///         Some(&attention_mask),
    ///         None,
    ///         Some(&target_mapping),
    ///         None,
    ///         None,
    ///         None,
    ///         false
    ///     )
    /// });
    /// # Ok(())
    /// # }
    /// ```
    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        old_layer_states: Option<Vec<Option<LayerState>>>,
        perm_mask: Option<&Tensor>,
        target_mapping: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        train: bool,
    ) -> XLNetSequenceClassificationOutput {
        let base_model_output = self
            .base_model
            .forward_t(
                input_ids,
                attention_mask,
                old_layer_states,
                perm_mask,
                target_mapping,
                token_type_ids,
                input_embeds,
                train,
            )
            .unwrap();

        let logits = self
            .sequence_summary
            .forward_t(&base_model_output.hidden_state, None, train)
            .apply(&self.logits_proj);

        XLNetSequenceClassificationOutput {
            logits,
            next_cache: base_model_output.next_cache,
            all_hidden_states: base_model_output.all_hidden_states,
            all_attentions: base_model_output.all_attentions,
        }
    }
}

/// # XLNetForTokenClassification
/// XLNet model with a classification head for token-level classification tasks
/// It is made of the following blocks:
/// - `base_model`: `XLNetModel`
/// - `classifier`: Linear layer projecting the hidden layer output to the target space
pub struct XLNetForTokenClassification {
    base_model: XLNetModel,
    classifier: nn::Linear,
}

impl XLNetForTokenClassification {
    /// Build a new `XLNetForTokenClassification`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the XLNet model
    /// * `config` - `XLNetConfig` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::xlnet::{XLNetConfig, XLNetForTokenClassification};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = XLNetConfig::from_file(config_path);
    /// let xlnet_model = XLNetForTokenClassification::new(&p.root(), &config);
    /// ```
    pub fn new<'p, P>(
        p: P,
        config: &XLNetConfig,
    ) -> Result<XLNetForTokenClassification, RustBertError>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let base_model = XLNetModel::new(p / "transformer", config);
        let num_labels = config
            .id2label
            .as_ref()
            .expect("num_labels not provided in configuration")
            .len() as i64;

        let classifier = nn::linear(
            p / "classifier",
            config.d_model,
            num_labels,
            Default::default(),
        );

        Ok(XLNetForTokenClassification {
            base_model,
            classifier,
        })
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). This or `input_embeds` must be provided.
    /// * `attention_mask` - Optional attention mask of shape (*batch size*, *sequence_length*) for the encoder positions. Positions with a mask with value 0 will be masked.
    /// * `perm_mask` - Optional tensor of shape (*batch size*, *sequence_length*, *sequence_length*). Mask to indicate the attention pattern for each input token (only used for pre-training over permutations, rather than simple token masking).
    /// * `target_mapping ` - Optional tensor of shape (*batch size*, *num_tokens*, *sequence_length*) indicating the position of the masked words to predict.
    /// * `token_type_ids` - Optional tensor (*batch size*, *sequence_length*) indicating the sentence ID of the token (0: first sentence, 1: second sentence).
    /// * `input_embeds` - Optional input tensor of shape (*batch size*, *sequence_length*, *embeddings dimension*). This or `input_ids` must be provided.
    /// * `old_layer_states` - Optional vector of length `num_layers` containing optional `LayerStates` containing the last calculated content for the attention layers. This avoids recomputing attention weights at past positions and speeds up decoding.
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `XLNetTokenClassificationOutput` containing:
    ///   - `logits` - `Tensor` of shape (*batch size*, *sequence_length*, *num_classes*) representing the logits for each batch item, token position and class
    ///   - `next_cache` - `Option<Vec<Option<LayerState>>>` of length *n_layer* containing the past content for the the attention layers with shape (*past_sequence_length*, *batch size*, *hidden_size*)
    ///   - `all_hidden_states` - `Option<Vec<(Tensor, Option<Tensor>)>>` of length *n_layer* with shape (*batch size*, *sequence_length*, *hidden_size*) (with optional query stream states if used)
    ///   - `all_attentions` - `Option<Vec<(Tensor, Option<Tensor>)>>` of length *n_layer* with shape (*batch size*, *sequence_length*, *hidden_size*) (with optional query stream states if used)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use tch::{nn, Device, Tensor, no_grad, Kind};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::{Int64, Double};
    /// use rust_bert::xlnet::{XLNetConfig, XLNetForTokenClassification};
    /// # fn main() -> anyhow::Result<()> {
    /// # let config_path = Path::new("path/to/config.json");
    /// # let vocab_path = Path::new("path/to/vocab.txt");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = XLNetConfig::from_file(config_path);
    /// # let xlnet_model: XLNetForTokenClassification = XLNetForTokenClassification::new(&vs.root(), &config)?;
    /// let (batch_size, sequence_length) = (64, 128);
    /// let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    /// let attention_mask = Tensor::ones(&[batch_size, sequence_length], (Int64, device));
    /// let target_tensor = Tensor::ones(&[batch_size, sequence_length], (Int64, device));
    /// let target_mapping = Tensor::zeros(&[64, 1, 128], (Kind::Float, device));
    /// let _ = target_mapping.narrow(2, 3, 1).fill_(1.0);
    ///
    /// let model_output = no_grad(|| {
    ///     xlnet_model.forward_t(
    ///         Some(&input_tensor),
    ///         Some(&attention_mask),
    ///         None,
    ///         Some(&target_mapping),
    ///         None,
    ///         None,
    ///         None,
    ///         false
    ///     )
    /// });
    /// # Ok(())
    /// # }
    /// ```
    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        old_layer_states: Option<Vec<Option<LayerState>>>,
        perm_mask: Option<&Tensor>,
        target_mapping: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        train: bool,
    ) -> XLNetTokenClassificationOutput {
        let base_model_output = self
            .base_model
            .forward_t(
                input_ids,
                attention_mask,
                old_layer_states,
                perm_mask,
                target_mapping,
                token_type_ids,
                input_embeds,
                train,
            )
            .unwrap();

        let logits = base_model_output.hidden_state.apply(&self.classifier);

        XLNetTokenClassificationOutput {
            logits,
            next_cache: base_model_output.next_cache,
            all_hidden_states: base_model_output.all_hidden_states,
            all_attentions: base_model_output.all_attentions,
        }
    }
}

/// # XLNetForMultipleChoice
/// Multiple choices model using a XLNet base model and a linear classifier.
/// Input should be in the form `[CLS] Context [SEP] Possible choice [SEP]`. The choice is made along the batch axis,
/// assuming all elements of the batch are alternatives to be chosen from for a given context.
/// It is made of the following blocks:
/// - `base_model`: `XLNetModel`
/// - `sequence_summary`: `SequenceSummary` to pool the base model hidden states
/// - `logits_proj`: Linear layer projecting the hidden layer pooled output to a single value
pub struct XLNetForMultipleChoice {
    base_model: XLNetModel,
    sequence_summary: SequenceSummary,
    logits_proj: nn::Linear,
}

impl XLNetForMultipleChoice {
    /// Build a new `XLNetForMultipleChoice`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the XLNet model
    /// * `config` - `XLNetConfig` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::xlnet::{XLNetConfig, XLNetForMultipleChoice};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = XLNetConfig::from_file(config_path);
    /// let xlnet_model = XLNetForMultipleChoice::new(&p.root(), &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &XLNetConfig) -> Result<XLNetForMultipleChoice, RustBertError>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let base_model = XLNetModel::new(p / "transformer", config);
        let sequence_summary =
            SequenceSummary::new(p / "sequence_summary", &SummaryConfig::from(config))?;

        let logits_proj = nn::linear(p / "logits_proj", config.d_model, 1, Default::default());

        Ok(XLNetForMultipleChoice {
            base_model,
            sequence_summary,
            logits_proj,
        })
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). This or `input_embeds` must be provided.
    /// * `attention_mask` - Optional attention mask of shape (*batch size*, *sequence_length*) for the encoder positions. Positions with a mask with value 0 will be masked.
    /// * `perm_mask` - Optional tensor of shape (*batch size*, *sequence_length*, *sequence_length*). Mask to indicate the attention pattern for each input token (only used for pre-training over permutations, rather than simple token masking).
    /// * `target_mapping ` - Optional tensor of shape (*batch size*, *num_tokens*, *sequence_length*) indicating the position of the masked words to predict.
    /// * `token_type_ids` - Optional tensor (*batch size*, *sequence_length*) indicating the sentence ID of the token (0: first sentence, 1: second sentence).
    /// * `input_embeds` - Optional input tensor of shape (*batch size*, *sequence_length*, *embeddings dimension*). This or `input_ids` must be provided.
    /// * `old_layer_states` - Optional vector of length `num_layers` containing optional `LayerStates` containing the last calculated content for the attention layers. This avoids recomputing attention weights at past positions and speeds up decoding.
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `XLNetSequenceClassificationOutput` containing:
    ///   - `logits` - `Tensor` of shape (*1*, *batch size*) containing the logits for each of the alternatives given
    ///   - `next_cache` - `Option<Vec<Option<LayerState>>>` of length *n_layer* containing the past content for the the attention layers with shape (*past_sequence_length*, *batch size*, *hidden_size*)
    ///   - `all_hidden_states` - `Option<Vec<(Tensor, Option<Tensor>)>>` of length *n_layer* with shape (*batch size*, *sequence_length*, *hidden_size*) (with optional query stream states if used)
    ///   - `all_attentions` - `Option<Vec<(Tensor, Option<Tensor>)>>` of length *n_layer* with shape (*batch size*, *sequence_length*, *hidden_size*) (with optional query stream states if used)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use tch::{nn, Device, Tensor, no_grad, Kind};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::{Int64, Double};
    /// use rust_bert::xlnet::{XLNetConfig, XLNetForMultipleChoice};
    /// # fn main() -> anyhow::Result<()> {
    /// # let config_path = Path::new("path/to/config.json");
    /// # let vocab_path = Path::new("path/to/vocab.txt");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = XLNetConfig::from_file(config_path);
    /// # let xlnet_model: XLNetForMultipleChoice = XLNetForMultipleChoice::new(&vs.root(), &config)?;
    /// let (batch_size, sequence_length) = (64, 128);
    /// let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    /// let attention_mask = Tensor::ones(&[batch_size, sequence_length], (Int64, device));
    /// let target_tensor = Tensor::ones(&[batch_size, sequence_length], (Int64, device));
    /// let target_mapping = Tensor::zeros(&[64, 1, 128], (Kind::Float, device));
    /// let _ = target_mapping.narrow(2, 3, 1).fill_(1.0);
    ///
    /// let model_output = no_grad(|| {
    ///     xlnet_model.forward_t(
    ///         Some(&input_tensor),
    ///         Some(&attention_mask),
    ///         None,
    ///         Some(&target_mapping),
    ///         None,
    ///         None,
    ///         None,
    ///         false
    ///     )
    /// });
    /// # Ok(())
    /// # }
    /// ```
    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        old_layer_states: Option<Vec<Option<LayerState>>>,
        perm_mask: Option<&Tensor>,
        target_mapping: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        train: bool,
    ) -> XLNetSequenceClassificationOutput {
        let (input_ids, num_choices) = match input_ids {
            Some(value) => (
                Some(value.view((-1, *value.size().last().unwrap()))),
                value.size()[1],
            ),
            None => (
                None,
                input_embeds
                    .as_ref()
                    .expect("At least one of input ids or input_embeds must be provided")
                    .size()[1],
            ),
        };

        let attention_mask =
            attention_mask.map(|tensor| tensor.view((-1, *tensor.size().last().unwrap())));
        let token_type_ids =
            token_type_ids.map(|tensor| tensor.view((-1, *tensor.size().last().unwrap())));
        let input_embeds =
            input_embeds.map(|tensor| tensor.view((-1, *tensor.size().last().unwrap())));
        let base_model_output = self
            .base_model
            .forward_t(
                input_ids.as_ref(),
                attention_mask.as_ref(),
                old_layer_states,
                perm_mask,
                target_mapping,
                token_type_ids.as_ref(),
                input_embeds.as_ref(),
                train,
            )
            .unwrap();

        let logits = self
            .sequence_summary
            .forward_t(&base_model_output.hidden_state, None, train)
            .apply(&self.logits_proj)
            .view((-1, num_choices));

        XLNetSequenceClassificationOutput {
            logits,
            next_cache: base_model_output.next_cache,
            all_hidden_states: base_model_output.all_hidden_states,
            all_attentions: base_model_output.all_attentions,
        }
    }
}

/// # XLNet for question answering
/// Extractive question-answering model based on a XLNet language model. Identifies the segment of a context that answers a provided question.
/// Please note that a significant amount of pre- and post-processing is required to perform end-to-end question answering.
/// See the question answering pipeline (also provided in this crate) for more details.
/// It is made of the following blocks:
/// - `base_model`: Base `XLNetModel`
/// - `qa_outputs`: Linear layer for question answering
pub struct XLNetForQuestionAnswering {
    base_model: XLNetModel,
    qa_outputs: nn::Linear,
}

impl XLNetForQuestionAnswering {
    /// Build a new `XLNetForQuestionAnswering`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the XLNet model
    /// * `config` - `XLNetConfig` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::xlnet::{XLNetConfig, XLNetForQuestionAnswering};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = XLNetConfig::from_file(config_path);
    /// let xlnet_model = XLNetForQuestionAnswering::new(&p.root(), &config);
    /// ```
    pub fn new<'p, P>(
        p: P,
        config: &XLNetConfig,
    ) -> Result<XLNetForQuestionAnswering, RustBertError>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let base_model = XLNetModel::new(p / "transformer", config);
        let qa_outputs = nn::linear(p / "qa_outputs", config.d_model, 2, Default::default());

        Ok(XLNetForQuestionAnswering {
            base_model,
            qa_outputs,
        })
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). This or `input_embeds` must be provided.
    /// * `attention_mask` - Optional attention mask of shape (*batch size*, *sequence_length*) for the encoder positions. Positions with a mask with value 0 will be masked.
    /// * `perm_mask` - Optional tensor of shape (*batch size*, *sequence_length*, *sequence_length*). Mask to indicate the attention pattern for each input token (only used for pre-training over permutations, rather than simple token masking).
    /// * `target_mapping ` - Optional tensor of shape (*batch size*, *num_tokens*, *sequence_length*) indicating the position of the masked words to predict.
    /// * `token_type_ids` - Optional tensor (*batch size*, *sequence_length*) indicating the sentence ID of the token (0: first sentence, 1: second sentence).
    /// * `input_embeds` - Optional input tensor of shape (*batch size*, *sequence_length*, *embeddings dimension*). This or `input_ids` must be provided.
    /// * `old_layer_states` - Optional vector of length `num_layers` containing optional `LayerStates` containing the last calculated content for the attention layers. This avoids recomputing attention weights at past positions and speeds up decoding.
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `XLNetQuestionAnsweringOutput` containing:
    ///   - `start_logits` - `Tensor` of shape (*batch size*, *sequence_length*) containing the logits for start of the answer
    ///   - `end_logits` - `Tensor` of shape (*batch size*, *sequence_length*) containing the logits for end of the answer
    ///   - `next_cache` - `Option<Vec<Option<LayerState>>>` of length *n_layer* containing the past content for the the attention layers with shape (*past_sequence_length*, *batch size*, *hidden_size*)
    ///   - `all_hidden_states` - `Option<Vec<(Tensor, Option<Tensor>)>>` of length *n_layer* with shape (*batch size*, *sequence_length*, *hidden_size*) (with optional query stream states if used)
    ///   - `all_attentions` - `Option<Vec<(Tensor, Option<Tensor>)>>` of length *n_layer* with shape (*batch size*, *sequence_length*, *hidden_size*) (with optional query stream states if used)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use tch::{nn, Device, Tensor, no_grad, Kind};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::{Int64, Double};
    /// use rust_bert::xlnet::{XLNetConfig, XLNetForMultipleChoice};
    /// # fn main() -> anyhow::Result<()> {
    /// # let config_path = Path::new("path/to/config.json");
    /// # let vocab_path = Path::new("path/to/vocab.txt");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = XLNetConfig::from_file(config_path);
    /// # let xlnet_model: XLNetForMultipleChoice = XLNetForMultipleChoice::new(&vs.root(), &config)?;
    /// let (batch_size, sequence_length) = (64, 128);
    /// let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    /// let attention_mask = Tensor::ones(&[batch_size, sequence_length], (Int64, device));
    /// let target_tensor = Tensor::ones(&[batch_size, sequence_length], (Int64, device));
    /// let target_mapping = Tensor::zeros(&[64, 1, 128], (Kind::Float, device));
    /// let _ = target_mapping.narrow(2, 3, 1).fill_(1.0);
    ///
    /// let model_output = no_grad(|| {
    ///     xlnet_model.forward_t(
    ///         Some(&input_tensor),
    ///         Some(&attention_mask),
    ///         None,
    ///         Some(&target_mapping),
    ///         None,
    ///         None,
    ///         None,
    ///         false
    ///     )
    /// });
    /// # Ok(())
    /// # }
    /// ```
    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        old_layer_states: Option<Vec<Option<LayerState>>>,
        perm_mask: Option<&Tensor>,
        target_mapping: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        train: bool,
    ) -> XLNetQuestionAnsweringOutput {
        let base_model_output = self
            .base_model
            .forward_t(
                input_ids,
                attention_mask,
                old_layer_states,
                perm_mask,
                target_mapping,
                token_type_ids,
                input_embeds,
                train,
            )
            .unwrap();

        let sequence_output = base_model_output.hidden_state.apply(&self.qa_outputs);
        let logits = sequence_output.split(1, -1);
        let (start_logits, end_logits) = (&logits[0], &logits[1]);
        let start_logits = start_logits.squeeze_dim(-1);
        let end_logits = end_logits.squeeze_dim(-1);

        XLNetQuestionAnsweringOutput {
            start_logits,
            end_logits,
            next_cache: base_model_output.next_cache,
            all_hidden_states: base_model_output.all_hidden_states,
            all_attentions: base_model_output.all_attentions,
        }
    }
}

/// Container for the XLNet model output.
pub struct XLNetModelOutput {
    /// Last hidden states from the model
    pub hidden_state: Tensor,
    /// Cached hiden layer states for generation tasks
    pub next_cache: Option<Vec<Option<LayerState>>>,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<(Tensor, Option<Tensor>)>>,
    /// Attention weights for all intermediate layers
    pub all_attentions: Option<Vec<(Tensor, Option<Tensor>)>>,
}

/// Container for the XLNet sequence classification model output.
pub struct XLNetSequenceClassificationOutput {
    /// Logits for each input (sequence) for each target class
    pub logits: Tensor,
    /// Cached hiden layer states for generation tasks
    pub next_cache: Option<Vec<Option<LayerState>>>,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<(Tensor, Option<Tensor>)>>,
    /// Attention weights for all intermediate layers
    pub all_attentions: Option<Vec<(Tensor, Option<Tensor>)>>,
}

/// Container for the XLNet token classification model output.
pub struct XLNetTokenClassificationOutput {
    /// Logits for each sequence item (token) for each target class
    pub logits: Tensor,
    /// Cached hiden layer states for generation tasks
    pub next_cache: Option<Vec<Option<LayerState>>>,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<(Tensor, Option<Tensor>)>>,
    /// Attention weights for all intermediate layers
    pub all_attentions: Option<Vec<(Tensor, Option<Tensor>)>>,
}

/// Container for the XLNet question answering model output.
pub struct XLNetQuestionAnsweringOutput {
    /// Logits for the start position for token of each input sequence
    pub start_logits: Tensor,
    /// Logits for the end position for token of each input sequence
    pub end_logits: Tensor,
    /// Cached hiden layer states for generation tasks
    pub next_cache: Option<Vec<Option<LayerState>>>,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<(Tensor, Option<Tensor>)>>,
    /// Attention weights for all intermediate layers
    pub all_attentions: Option<Vec<(Tensor, Option<Tensor>)>>,
}

/// # Language generation model based on the XLNet architecture
pub struct XLNetGenerator {
    model: XLNetLMHeadModel,
    tokenizer: TokenizerOption,
    var_store: nn::VarStore,
    generate_config: GenerateConfig,
    bos_token_id: Option<i64>,
    eos_token_ids: Option<Vec<i64>>,
    pad_token_id: Option<i64>,
    is_encoder_decoder: bool,
    vocab_size: i64,
    decoder_start_id: Option<i64>,
    max_position_embeddings: i64,
}

impl XLNetGenerator {
    /// Build a new `XLNetGenerator`
    ///
    /// # Arguments
    ///
    /// * `generate_config` - `GenerateConfig` object containing the resource references (model, vocabulary, configuration), generation options and device placement (CPU/GPU)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> anyhow::Result<()> {
    /// use rust_bert::pipelines::generation_utils::GenerateConfig;
    /// use rust_bert::xlnet::XLNetGenerator;
    ///
    /// let generate_config = GenerateConfig {
    ///     max_length: 30,
    ///     do_sample: true,
    ///     num_beams: 5,
    ///     temperature: 1.1,
    ///     num_return_sequences: 3,
    ///     ..Default::default()
    /// };
    /// let xlnet_generator = XLNetGenerator::new(generate_config)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(generate_config: GenerateConfig) -> Result<XLNetGenerator, RustBertError> {
        let vocab_path = generate_config.vocab_resource.get_local_path()?;

        let tokenizer = TokenizerOption::from_file(
            ModelType::XLNet,
            vocab_path.to_str().unwrap(),
            None,
            false,
            true,
            None,
        )?;

        Self::new_with_tokenizer(generate_config, tokenizer)
    }

    pub fn new_with_tokenizer(
        generate_config: GenerateConfig,
        tokenizer: TokenizerOption,
    ) -> Result<XLNetGenerator, RustBertError> {
        let config_path = generate_config.config_resource.get_local_path()?;
        let weights_path = generate_config.model_resource.get_local_path()?;
        let device = generate_config.device;

        generate_config.validate();
        let mut var_store = nn::VarStore::new(device);

        let config = XLNetConfig::from_file(config_path);
        let model = XLNetLMHeadModel::new(&var_store.root(), &config);
        var_store.load(weights_path)?;

        let bos_token_id = Some(config.bos_token_id);
        let eos_token_ids = Some(vec![config.eos_token_id]);
        let pad_token_id = Some(config.pad_token_id);
        let is_encoder_decoder = false;
        let vocab_size = config.vocab_size;
        let decoder_start_id = None;
        // XLNet do not have an embedding matrix for position IDs and relies on trigonometric methods instead
        let max_position_embeddings = i64::MAX;

        Ok(XLNetGenerator {
            model,
            tokenizer,
            var_store,
            generate_config,
            bos_token_id,
            eos_token_ids,
            pad_token_id,
            is_encoder_decoder,
            vocab_size,
            decoder_start_id,
            max_position_embeddings,
        })
    }
}

impl PrivateLanguageGenerator<XLNetLMHeadModel, XLNetVocab, XLNetTokenizer> for XLNetGenerator {
    fn get_model(&self) -> &XLNetLMHeadModel {
        &self.model
    }
    fn _get_tokenizer(&self) -> &TokenizerOption {
        &self.tokenizer
    }
    fn get_var_store(&self) -> &nn::VarStore {
        &self.var_store
    }
    fn get_var_store_mut(&mut self) -> &mut nn::VarStore {
        &mut self.var_store
    }
    fn get_config(&self) -> &GenerateConfig {
        &self.generate_config
    }
    fn get_bos_id(&self) -> Option<i64> {
        self.bos_token_id
    }
    fn get_eos_ids(&self) -> Option<&Vec<i64>> {
        self.eos_token_ids.as_ref()
    }
    fn get_pad_id(&self) -> Option<i64> {
        self.pad_token_id
    }
    fn is_encoder_decoder(&self) -> bool {
        self.is_encoder_decoder
    }
    fn get_vocab_size(&self) -> i64 {
        self.vocab_size
    }
    fn get_decoder_start_id(&self) -> Option<i64> {
        self.decoder_start_id
    }

    fn get_max_positions_embeddings(&self) -> i64 {
        self.max_position_embeddings
    }

    fn prepare_inputs_for_generation<'a>(
        &self,
        input_ids: Tensor,
        _encoder_outputs: Option<&'a Tensor>,
        past: Cache,
        _attention_mask: Tensor,
    ) -> PreparedInput<'a> {
        let effective_batch_size = input_ids.size()[0];
        let sequence_length = input_ids.size()[1];
        let dummy_token = Tensor::zeros(
            &[effective_batch_size, 1],
            (Kind::Int64, input_ids.device()),
        );
        let offset = 2i64;
        let input_ids = match &past {
            Cache::XLNetCache(past) => {
                if past.is_some() {
                    Tensor::cat(
                        &[
                            input_ids.slice(1, sequence_length - offset, sequence_length, 1),
                            dummy_token,
                        ],
                        1,
                    )
                } else {
                    Tensor::cat(&[input_ids, dummy_token], 1)
                }
            }
            _ => Tensor::cat(&[input_ids, dummy_token], 1),
        };
        let sequence_length = input_ids.size()[1];
        let perm_mask = Tensor::zeros(
            &[effective_batch_size, sequence_length, sequence_length],
            (Kind::Float, input_ids.device()),
        );
        let _ = perm_mask.narrow(2, sequence_length - 1, 1).fill_(1.0);

        let target_mapping = Tensor::zeros(
            &[effective_batch_size, 1, sequence_length],
            (Kind::Float, input_ids.device()),
        );
        let _ = target_mapping.narrow(2, sequence_length - 1, 1).fill_(1.0);

        match past {
            Cache::XLNetCache(past) => {
                if let Some(past) = past {
                    let past = if let Some(first_past) = &past[0] {
                        let past_len = first_past.prev_content.size()[0];
                        past.iter()
                            .map(|old_layer_state| {
                                Some(LayerState {
                                    prev_content: old_layer_state
                                        .as_ref()
                                        .unwrap()
                                        .prev_content
                                        .slice(0, 0, past_len - offset, 1),
                                })
                            })
                            .collect()
                    } else {
                        past
                    };
                    PreparedInput {
                        prepared_input: Some(input_ids),
                        prepared_attention_mask: Some(perm_mask),
                        prepared_encoder_output: None,
                        prepared_decoder_input: Some(target_mapping),
                        prepared_position_ids: None,
                        prepared_past: Cache::XLNetCache(Some(past)),
                    }
                } else {
                    PreparedInput {
                        prepared_input: Some(input_ids),
                        prepared_attention_mask: Some(perm_mask),
                        prepared_encoder_output: None,
                        prepared_decoder_input: Some(target_mapping),
                        prepared_position_ids: None,
                        prepared_past: Cache::XLNetCache(None),
                    }
                }
            }
            Cache::None => PreparedInput {
                prepared_input: Some(input_ids),
                prepared_attention_mask: Some(perm_mask),
                prepared_encoder_output: None,
                prepared_decoder_input: Some(target_mapping),
                prepared_position_ids: None,
                prepared_past: Cache::XLNetCache(None),
            },
            _ => panic!("Cache type incompatible with XLNet"),
        }
    }

    fn reorder_cache(
        &self,
        past: &mut Cache,
        _encoder_outputs: Option<Tensor>,
        beam_indices: &Tensor,
    ) -> Option<Tensor> {
        match past {
            Cache::XLNetCache(old_cache_option) => match old_cache_option {
                Some(old_cache) => {
                    for layer_state in old_cache.iter_mut() {
                        if layer_state.is_some() {
                            layer_state.as_mut().unwrap().reorder_cache(beam_indices)
                        };
                    }
                    None
                }
                None => None,
            },
            Cache::None => None,
            _ => {
                panic!("Invalid cache for XLNet model");
            }
        }
    }
}

impl LanguageGenerator<XLNetLMHeadModel, XLNetVocab, XLNetTokenizer> for XLNetGenerator {}
