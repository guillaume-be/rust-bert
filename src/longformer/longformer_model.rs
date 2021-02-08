// Copyright 2020 The Allen Institute for AI team and The HuggingFace Inc. team.
// Copyright 2021 Guillaume Becquin
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::common::activations::{TensorFunction, _tanh};
use crate::common::dropout::Dropout;
use crate::longformer::embeddings::LongformerEmbeddings;
use crate::longformer::encoder::LongformerEncoder;
use crate::{Activation, Config, RustBertError};
use serde::{Deserialize, Serialize};
use std::borrow::Borrow;
use std::collections::HashMap;
use tch::nn::{Init, Module, ModuleT};
use tch::{nn, Kind, Tensor};

/// # Longformer Pretrained model weight files
pub struct LongformerModelResources;

/// # Longformer Pretrained model config files
pub struct LongformerConfigResources;

/// # Longformer Pretrained model vocab files
pub struct LongformerVocabResources;

/// # Longformer Pretrained model merges files
pub struct LongformerMergesResources;

impl LongformerModelResources {
    /// Shared under Apache 2.0 license by the AllenAI team at https://github.com/allenai/longformer. Modified with conversion to C-array format.
    pub const LONGFORMER_BASE_4096: (&'static str, &'static str) = (
        "longformer-base-4096/model",
        "https://huggingface.co/allenai/longformer-base-4096/resolve/main/rust_model.ot",
    );
}

impl LongformerConfigResources {
    /// Shared under Apache 2.0 license by the AllenAI team at https://github.com/allenai/longformer. Modified with conversion to C-array format.
    pub const LONGFORMER_BASE_4096: (&'static str, &'static str) = (
        "longformer-base-4096/config",
        "https://huggingface.co/allenai/longformer-base-4096/resolve/main/config.json",
    );
}

impl LongformerVocabResources {
    /// Shared under Apache 2.0 license by the AllenAI team at https://github.com/allenai/longformer. Modified with conversion to C-array format.
    pub const LONGFORMER_BASE_4096: (&'static str, &'static str) = (
        "longformer-base-4096/vocab",
        "https://huggingface.co/allenai/longformer-base-4096/resolve/main/vocab.json",
    );
}

impl LongformerMergesResources {
    /// Shared under Apache 2.0 license by the AllenAI team at https://github.com/allenai/longformer. Modified with conversion to C-array format.
    pub const LONGFORMER_BASE_4096: (&'static str, &'static str) = (
        "longformer-base-4096/merges",
        "https://huggingface.co/allenai/longformer-base-4096/resolve/main/merges.txt",
    );
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
#[serde(rename_all = "camelCase")]
pub enum PositionEmbeddingType {
    Absolute,
    RelativeKey,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
/// # Longformer model configuration
/// Defines the Longformer model architecture (e.g. number of layers, hidden layer size, label mapping...)
pub struct LongformerConfig {
    pub hidden_act: Activation,
    pub attention_window: Vec<i64>,
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
    pub sep_token_id: i64,
    pub pad_token_id: Option<i64>,
    pub layer_norm_eps: Option<f64>,
    pub output_attentions: Option<bool>,
    pub output_hidden_states: Option<bool>,
    pub position_embedding_type: Option<PositionEmbeddingType>,
    pub is_decoder: Option<bool>,
    pub id2label: Option<HashMap<i64, String>>,
    pub label2id: Option<HashMap<String, i64>>,
}

impl Config<LongformerConfig> for LongformerConfig {}

fn _get_question_end_index(input_ids: &Tensor, sep_token_id: i64) -> Tensor {
    input_ids
        .eq(sep_token_id)
        .nonzero()
        .view([input_ids.size()[0], 3, 2])
        .select(2, 1)
        .select(1, 0)
}

fn _compute_global_attention_mask(
    input_ids: &Tensor,
    sep_token_id: i64,
    before_sep_token: bool,
) -> Tensor {
    let question_end_index = _get_question_end_index(input_ids, sep_token_id).unsqueeze(1);
    let attention_mask = Tensor::arange(input_ids.size()[1], (Kind::Int8, input_ids.device()));

    if before_sep_token {
        attention_mask.expand_as(input_ids).lt1(&question_end_index)
    } else {
        attention_mask
            .expand_as(input_ids)
            .gt1(&(question_end_index + 1))
            * attention_mask
                .expand_as(input_ids)
                .lt(*input_ids.size().last().unwrap())
    }
}

#[derive(Debug)]
pub struct LongformerPooler {
    dense: nn::Linear,
    activation: TensorFunction,
}

impl LongformerPooler {
    pub fn new<'p, P>(p: P, config: &LongformerConfig) -> LongformerPooler
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

        let activation = TensorFunction::new(Box::new(_tanh));

        LongformerPooler { dense, activation }
    }
}

impl Module for LongformerPooler {
    fn forward(&self, hidden_states: &Tensor) -> Tensor {
        self.activation.get_fn()(&hidden_states.select(1, 0).apply(&self.dense))
    }
}

#[derive(Debug)]
pub struct LongformerLMHead {
    dense: nn::Linear,
    layer_norm: nn::LayerNorm,
    decoder: nn::Linear,
    bias: Tensor,
}

impl LongformerLMHead {
    pub fn new<'p, P>(p: P, config: &LongformerConfig) -> LongformerLMHead
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

        let layer_norm_config = nn::LayerNormConfig {
            eps: config.layer_norm_eps.unwrap_or(1e-12),
            ..Default::default()
        };

        let layer_norm = nn::layer_norm(
            p / "layer_norm",
            vec![config.hidden_size],
            layer_norm_config,
        );

        let linear_config = nn::LinearConfig {
            bias: false,
            ..Default::default()
        };

        let decoder = nn::linear(
            p / "decoder",
            config.hidden_size,
            config.vocab_size,
            linear_config,
        );

        let bias = p.var("bias", &[config.vocab_size], Init::Const(0f64));

        LongformerLMHead {
            dense,
            layer_norm,
            decoder,
            bias,
        }
    }
}

impl Module for LongformerLMHead {
    fn forward(&self, hidden_states: &Tensor) -> Tensor {
        hidden_states
            .apply(&self.dense)
            .gelu()
            .apply(&self.layer_norm)
            .apply(&self.decoder)
            + &self.bias
    }
}

pub struct LongformerModel {
    embeddings: LongformerEmbeddings,
    encoder: LongformerEncoder,
    pooler: Option<LongformerPooler>,
    max_attention_window: i64,
    pad_token_id: i64,
    is_decoder: bool,
}

impl LongformerModel {
    pub fn new<'p, P>(p: P, config: &LongformerConfig, add_pooling_layer: bool) -> LongformerModel
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let embeddings = LongformerEmbeddings::new(p / "embeddings", config);
        let encoder = LongformerEncoder::new(p / "encoder", config);
        let pooler = if add_pooling_layer {
            Some(LongformerPooler::new(p / "pooler", config))
        } else {
            None
        };

        let max_attention_window = *config.attention_window.iter().max().unwrap();
        let pad_token_id = config.pad_token_id.unwrap_or(1);
        let is_decoder = config.is_decoder.unwrap_or(false);

        LongformerModel {
            embeddings,
            encoder,
            pooler,
            max_attention_window,
            pad_token_id,
            is_decoder,
        }
    }

    fn pad_with_nonzero_value(
        &self,
        tensor: &Tensor,
        padding: &[i64],
        padding_value: i64,
    ) -> Tensor {
        (tensor - padding_value).constant_pad_nd(padding) + padding_value
    }

    fn pad_with_boolean(&self, tensor: &Tensor, padding: &[i64], padding_value: bool) -> Tensor {
        if !padding_value {
            tensor.constant_pad_nd(padding)
        } else {
            ((tensor.logical_not()).constant_pad_nd(padding)).logical_not()
        }
    }

    fn pad_to_window_size(
        &self,
        input_ids: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        pad_token_id: i64,
        padding_length: i64,
        train: bool,
    ) -> Result<
        (
            Option<Tensor>,
            Option<Tensor>,
            Option<Tensor>,
            Option<Tensor>,
            Option<Tensor>,
        ),
        RustBertError,
    > {
        let input_shape = if let Some(input_ids) = input_ids {
            if input_embeds.is_none() {
                input_ids.size()
            } else {
                return Err(RustBertError::ValueError(
                    "Only one of input ids or input embeddings may be set".into(),
                ));
            }
        } else if let Some(input_embeds) = input_embeds {
            input_embeds.size()[..2].to_vec()
        } else {
            return Err(RustBertError::ValueError(
                "At least one of input ids or input embeddings must be set".into(),
            ));
        };
        let batch_size = input_shape[0];

        let input_ids = input_ids
            .map(|value| self.pad_with_nonzero_value(value, &[0, padding_length], pad_token_id));
        let position_ids = position_ids
            .map(|value| self.pad_with_nonzero_value(value, &[0, padding_length], pad_token_id));
        let inputs_embeds = input_embeds.map(|value| {
            let input_ids_padding = Tensor::full(
                &[batch_size, padding_length],
                pad_token_id,
                (Kind::Int64, value.device()),
            );
            let input_embeds_padding = self
                .embeddings
                .forward_t(Some(&input_ids_padding), None, None, None, train)
                .unwrap();

            Tensor::cat(&[value, &input_embeds_padding], -2)
        });

        let attention_mask =
            attention_mask.map(|value| self.pad_with_boolean(&value, &[0, padding_length], false));
        let token_type_ids =
            token_type_ids.map(|value| value.constant_pad_nd(&[0, padding_length]));
        Ok((
            input_ids,
            position_ids,
            inputs_embeds,
            attention_mask,
            token_type_ids,
        ))
    }

    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        global_attention_mask: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        train: bool,
    ) -> Result<LongformerModelOutput, RustBertError> {
        let (input_shape, device) = if let Some(input_ids) = input_ids {
            if input_embeds.is_none() {
                (input_ids.size(), input_ids.device())
            } else {
                return Err(RustBertError::ValueError(
                    "Only one of input ids or input embeddings may be set".into(),
                ));
            }
        } else if let Some(input_embeds) = input_embeds {
            (input_embeds.size()[..2].to_vec(), input_embeds.device())
        } else {
            return Err(RustBertError::ValueError(
                "At least one of input ids or input embeddings must be set".into(),
            ));
        };

        let (batch_size, sequence_length) = (input_shape[0], input_shape[1]);

        let calc_attention_mask = if attention_mask.is_none() {
            Some(Tensor::ones(input_shape.as_slice(), (Kind::Int, device)))
        } else {
            None
        };
        let calc_token_type_ids = if token_type_ids.is_none() {
            Some(Tensor::zeros(input_shape.as_slice(), (Kind::Int64, device)))
        } else {
            None
        };
        let attention_mask = if attention_mask.is_some() {
            attention_mask
        } else {
            calc_attention_mask.as_ref()
        };
        let token_type_ids = if token_type_ids.is_some() {
            token_type_ids
        } else {
            calc_token_type_ids.as_ref()
        };

        let merged_attention_mask = if let Some(global_attention_mask) = global_attention_mask {
            attention_mask.map(|tensor| tensor.multiply(&(global_attention_mask + 1)))
        } else {
            None
        };
        let attention_mask = if merged_attention_mask.is_some() {
            merged_attention_mask.as_ref()
        } else {
            attention_mask
        };

        let padding_length = (self.max_attention_window
            - sequence_length % self.max_attention_window)
            % self.max_attention_window;
        let (
            calc_padded_input_ids,
            calc_padded_position_ids,
            calc_padded_inputs_embeds,
            calc_padded_attention_mask,
            calc_padded_token_type_ids,
        ) = if padding_length > 0 {
            self.pad_to_window_size(
                input_ids,
                attention_mask,
                token_type_ids,
                position_ids,
                input_embeds,
                self.pad_token_id,
                padding_length,
                train,
            )?
        } else {
            (None, None, None, None, None)
        };
        let padded_input_ids = if calc_padded_input_ids.is_some() {
            calc_padded_input_ids.as_ref()
        } else {
            input_ids
        };
        let padded_position_ids = if calc_padded_position_ids.is_some() {
            calc_padded_position_ids.as_ref()
        } else {
            position_ids
        };
        let padded_inputs_embeds = if calc_padded_inputs_embeds.is_some() {
            calc_padded_inputs_embeds.as_ref()
        } else {
            input_embeds
        };
        let padded_attention_mask = calc_padded_attention_mask
            .as_ref()
            .unwrap_or(attention_mask.as_ref().unwrap());
        let padded_token_type_ids = if calc_padded_token_type_ids.is_some() {
            calc_padded_token_type_ids.as_ref()
        } else {
            token_type_ids
        };

        let extended_attention_mask = match padded_attention_mask.dim() {
            3 => padded_attention_mask.unsqueeze(1),
            2 => {
                if !self.is_decoder {
                    padded_attention_mask.unsqueeze(1).unsqueeze(1)
                } else {
                    let sequence_ids = Tensor::arange(sequence_length, (Kind::Int64, device));
                    let mut causal_mask = sequence_ids
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .repeat(&[batch_size, sequence_length, 1])
                        .le1(&sequence_ids.unsqueeze(-1).unsqueeze(0))
                        .totype(Kind::Int);
                    if causal_mask.size()[1] < padded_attention_mask.size()[1] {
                        let prefix_sequence_length =
                            padded_attention_mask.size()[1] - causal_mask.size()[1];
                        causal_mask = Tensor::cat(
                            &[
                                Tensor::ones(
                                    &[batch_size, sequence_length, prefix_sequence_length],
                                    (Kind::Int, device),
                                ),
                                causal_mask,
                            ],
                            -1,
                        );
                    }
                    causal_mask.unsqueeze(1) * padded_attention_mask.unsqueeze(1).unsqueeze(1)
                }
            }
            _ => {
                return Err(RustBertError::ValueError(
                    "Invalid attention mask dimension, must be 2 or 3".into(),
                ));
            }
        }
        .select(2, 0)
        .select(1, 0);
        let extended_attention_mask =
            (extended_attention_mask.ones_like() - extended_attention_mask) * -10000.0;

        let embedding_output = self.embeddings.forward_t(
            padded_input_ids,
            padded_token_type_ids,
            padded_position_ids,
            padded_inputs_embeds,
            train,
        )?;

        let encoder_outputs =
            self.encoder
                .forward_t(&embedding_output, &extended_attention_mask, train);

        let pooled_output = self
            .pooler
            .as_ref()
            .map(|pooler| pooler.forward(&encoder_outputs.hidden_states));

        let sequence_output = if padding_length > 0 {
            encoder_outputs
                .hidden_states
                .slice(1, 0, -padding_length, 1)
        } else {
            encoder_outputs.hidden_states
        };

        Ok(LongformerModelOutput {
            hidden_state: sequence_output,
            pooled_output,
            all_hidden_states: encoder_outputs.all_hidden_states,
            all_attentions: encoder_outputs.all_attentions,
            all_global_attentions: encoder_outputs.all_global_attentions,
        })
    }
}

pub struct LongformerForMaskedLM {
    longformer: LongformerModel,
    lm_head: LongformerLMHead,
}

impl LongformerForMaskedLM {
    pub fn new<'p, P>(p: P, config: &LongformerConfig) -> LongformerForMaskedLM
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let longformer = LongformerModel::new(p / "longformer", config, false);
        let lm_head = LongformerLMHead::new(p / "lm_head", config);

        LongformerForMaskedLM {
            longformer,
            lm_head,
        }
    }

    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        global_attention_mask: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        train: bool,
    ) -> Result<LongformerMaskedLMOutput, RustBertError> {
        let longformer_outputs = self.longformer.forward_t(
            input_ids,
            attention_mask,
            global_attention_mask,
            token_type_ids,
            position_ids,
            input_embeds,
            train,
        )?;

        let prediction_scores = self
            .lm_head
            .forward_t(&longformer_outputs.hidden_state, train);

        Ok(LongformerMaskedLMOutput {
            prediction_scores,
            all_hidden_states: longformer_outputs.all_hidden_states,
            all_attentions: longformer_outputs.all_attentions,
            all_global_attentions: longformer_outputs.all_global_attentions,
        })
    }
}

pub struct LongformerClassificationHead {
    dense: nn::Linear,
    dropout: Dropout,
    out_proj: nn::Linear,
}

impl LongformerClassificationHead {
    pub fn new<'p, P>(p: P, config: &LongformerConfig) -> LongformerClassificationHead
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
        let dropout = Dropout::new(config.hidden_dropout_prob);

        let num_labels = config
            .id2label
            .as_ref()
            .expect("num_labels not provided in configuration")
            .len() as i64;
        let out_proj = nn::linear(
            p / "out_proj",
            config.hidden_size,
            num_labels,
            Default::default(),
        );

        LongformerClassificationHead {
            dense,
            dropout,
            out_proj,
        }
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

pub struct LongformerForSequenceClassification {
    longformer: LongformerModel,
    classifier: LongformerClassificationHead,
}

impl LongformerForSequenceClassification {
    pub fn new<'p, P>(p: P, config: &LongformerConfig) -> LongformerForSequenceClassification
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let longformer = LongformerModel::new(p / "longformer", config, false);
        let classifier = LongformerClassificationHead::new(p / "classifier", config);

        LongformerForSequenceClassification {
            longformer,
            classifier,
        }
    }

    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        global_attention_mask: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        train: bool,
    ) -> Result<LongformerTokenClassificationOutput, RustBertError> {
        let calc_global_attention_mask = if global_attention_mask.is_none() {
            let (input_shape, device) = if let Some(input_ids) = input_ids {
                if input_embeds.is_none() {
                    (input_ids.size(), input_ids.device())
                } else {
                    return Err(RustBertError::ValueError(
                        "Only one of input ids or input embeddings may be set".into(),
                    ));
                }
            } else if let Some(input_embeds) = input_embeds {
                (input_embeds.size()[..2].to_vec(), input_embeds.device())
            } else {
                return Err(RustBertError::ValueError(
                    "At least one of input ids or input embeddings must be set".into(),
                ));
            };

            let (batch_size, sequence_length) = (input_shape[0], input_shape[1]);
            let global_attention_mask =
                Tensor::zeros(&[batch_size, sequence_length], (Kind::Int, device));
            let _ = global_attention_mask.select(1, 0).fill_(1);
            Some(global_attention_mask)
        } else {
            None
        };

        let global_attention_mask = if global_attention_mask.is_some() {
            global_attention_mask
        } else {
            calc_global_attention_mask.as_ref()
        };

        let base_model_output = self.longformer.forward_t(
            input_ids,
            attention_mask,
            global_attention_mask,
            token_type_ids,
            position_ids,
            input_embeds,
            train,
        )?;

        let logits = self
            .classifier
            .forward_t(&base_model_output.hidden_state, train);
        Ok(LongformerTokenClassificationOutput {
            logits,
            all_hidden_states: base_model_output.all_hidden_states,
            all_attentions: base_model_output.all_attentions,
            all_global_attentions: base_model_output.all_global_attentions,
        })
    }
}

/// Container for the Longformer model output.
pub struct LongformerModelOutput {
    /// Last hidden states from the model
    pub hidden_state: Tensor,
    /// Pooled output (hidden state for the first token)
    pub pooled_output: Option<Tensor>,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all intermediate layers
    pub all_attentions: Option<Vec<Tensor>>,
    /// Global attention weights for all intermediate layers
    pub all_global_attentions: Option<Vec<Tensor>>,
}

/// Container for the Longformer masked LM model output.
pub struct LongformerMaskedLMOutput {
    /// Logits for the vocabulary items at each sequence position
    pub prediction_scores: Tensor,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all intermediate layers
    pub all_attentions: Option<Vec<Tensor>>,
    /// Global attention weights for all intermediate layers
    pub all_global_attentions: Option<Vec<Tensor>>,
}

/// Container for the Longformer token classification model output.
pub struct LongformerTokenClassificationOutput {
    /// Logits for each sequence item (token) for each target class
    pub logits: Tensor,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all intermediate layers
    pub all_attentions: Option<Vec<Tensor>>,
    /// Global attention weights for all intermediate layers
    pub all_global_attentions: Option<Vec<Tensor>>,
}
