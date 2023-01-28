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

use crate::common::activations::TensorFunction;
use crate::common::dropout::Dropout;
use crate::longformer::attention::LongformerSelfAttention;
use crate::longformer::LongformerConfig;
use std::borrow::{Borrow, BorrowMut};
use tch::nn::Module;
use tch::{nn, Tensor};

pub struct LongformerSelfOutput {
    dense: nn::Linear,
    layer_norm: nn::LayerNorm,
    dropout: Dropout,
}

impl LongformerSelfOutput {
    pub fn new<'p, P>(p: P, config: &LongformerConfig) -> LongformerSelfOutput
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

        let layer_norm =
            nn::layer_norm(p / "LayerNorm", vec![config.hidden_size], layer_norm_config);

        let dropout = Dropout::new(config.hidden_dropout_prob);
        LongformerSelfOutput {
            dense,
            layer_norm,
            dropout,
        }
    }

    pub fn forward_t(&self, hidden_states: &Tensor, input_tensor: &Tensor, train: bool) -> Tensor {
        let hidden_states = hidden_states
            .apply(&self.dense)
            .apply_t(&self.dropout, train);

        (hidden_states + input_tensor).apply(&self.layer_norm)
    }
}

pub struct LongformerAttention {
    self_attention: LongformerSelfAttention,
    output: LongformerSelfOutput,
}

impl LongformerAttention {
    pub fn new<'p, P>(p: P, config: &LongformerConfig, layer_id: i64) -> LongformerAttention
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let self_attention = LongformerSelfAttention::new(p / "self", config, layer_id);
        let output = LongformerSelfOutput::new(p / "output", config);

        LongformerAttention {
            self_attention,
            output,
        }
    }

    pub fn forward_t(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
        is_index_masked: &Tensor,
        is_index_global_attention: &Tensor,
        is_global_attention: bool,
        train: bool,
    ) -> (Tensor, Option<Tensor>, Option<Tensor>) {
        let (attention_outputs, attention_scores, global_attention_scores) =
            self.self_attention.forward_t(
                hidden_states,
                attention_mask,
                is_index_masked,
                is_index_global_attention,
                is_global_attention,
                train,
            );

        let attention_outputs = self
            .output
            .forward_t(&attention_outputs, hidden_states, train);

        (attention_outputs, attention_scores, global_attention_scores)
    }
}

#[derive(Debug)]
pub struct LongformerIntermediate {
    dense: nn::Linear,
    activation_function: TensorFunction,
}

impl LongformerIntermediate {
    pub fn new<'p, P>(p: P, config: &LongformerConfig) -> LongformerIntermediate
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let dense = nn::linear(
            p / "dense",
            config.hidden_size,
            config.intermediate_size,
            Default::default(),
        );
        let activation_function = config.hidden_act.get_function();

        LongformerIntermediate {
            dense,
            activation_function,
        }
    }
}

impl Module for LongformerIntermediate {
    fn forward(&self, hidden_states: &Tensor) -> Tensor {
        self.activation_function.get_fn()(&hidden_states.apply(&self.dense))
    }
}

pub struct LongformerOutput {
    dense: nn::Linear,
    layer_norm: nn::LayerNorm,
    dropout: Dropout,
}

impl LongformerOutput {
    pub fn new<'p, P>(p: P, config: &LongformerConfig) -> LongformerOutput
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let dense = nn::linear(
            p / "dense",
            config.intermediate_size,
            config.hidden_size,
            Default::default(),
        );
        let layer_norm_config = nn::LayerNormConfig {
            eps: config.layer_norm_eps.unwrap_or(1e-12),
            ..Default::default()
        };

        let layer_norm =
            nn::layer_norm(p / "LayerNorm", vec![config.hidden_size], layer_norm_config);

        let dropout = Dropout::new(config.hidden_dropout_prob);

        LongformerOutput {
            dense,
            layer_norm,
            dropout,
        }
    }

    pub fn forward_t(&self, hidden_states: &Tensor, input_tensor: &Tensor, train: bool) -> Tensor {
        let hidden_states = hidden_states
            .apply(&self.dense)
            .apply_t(&self.dropout, train);
        (hidden_states + input_tensor).apply(&self.layer_norm)
    }
}

pub struct LongformerLayer {
    attention: LongformerAttention,
    intermediate: LongformerIntermediate,
    output: LongformerOutput,
}

impl LongformerLayer {
    pub fn new<'p, P>(p: P, config: &LongformerConfig, layer_id: i64) -> LongformerLayer
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let attention = LongformerAttention::new(p / "attention", config, layer_id);
        let intermediate = LongformerIntermediate::new(p / "intermediate", config);
        let output = LongformerOutput::new(p / "output", config);

        LongformerLayer {
            attention,
            intermediate,
            output,
        }
    }

    pub fn forward_t(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
        is_index_masked: &Tensor,
        is_index_global_attention: &Tensor,
        is_global_attention: bool,
        train: bool,
    ) -> (Tensor, Option<Tensor>, Option<Tensor>) {
        let (attention_outputs, attention_scores, global_attention_scores) =
            self.attention.forward_t(
                hidden_states,
                attention_mask,
                is_index_masked,
                is_index_global_attention,
                is_global_attention,
                train,
            );

        let intermediate_output = attention_outputs.apply(&self.intermediate);
        let attention_outputs =
            self.output
                .forward_t(&intermediate_output, &attention_outputs, train);
        (attention_outputs, attention_scores, global_attention_scores)
    }
}

/// Container for the Longformer encoder output.
pub struct LongformerEncoderOutput {
    /// Last hidden states from the model
    pub hidden_states: Tensor,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all intermediate layers
    pub all_attentions: Option<Vec<Tensor>>,
    /// Global attention weights for all intermediate layers
    pub all_global_attentions: Option<Vec<Tensor>>,
}

pub struct LongformerEncoder {
    layers: Vec<LongformerLayer>,
    output_attentions: bool,
    output_hidden_states: bool,
}

impl LongformerEncoder {
    pub fn new<'p, P>(p: P, config: &LongformerConfig) -> LongformerEncoder
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let p_layers = p / "layer";

        let mut layers: Vec<LongformerLayer> =
            Vec::with_capacity(config.num_hidden_layers as usize);
        for layer_index in 0..config.num_hidden_layers {
            layers.push(LongformerLayer::new(
                &p_layers / layer_index,
                config,
                layer_index,
            ));
        }
        let output_attentions = config.output_attentions.unwrap_or(false);
        let output_hidden_states = config.output_hidden_states.unwrap_or(false);

        LongformerEncoder {
            layers,
            output_attentions,
            output_hidden_states,
        }
    }

    pub fn forward_t(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
        train: bool,
    ) -> LongformerEncoderOutput {
        let is_index_masked = attention_mask.lt(0);
        let is_index_global_attention = attention_mask.gt(0);
        let is_global_attention = bool::from(is_index_global_attention.any());

        let mut all_hidden_states: Option<Vec<Tensor>> = if self.output_hidden_states {
            Some(vec![])
        } else {
            None
        };
        let mut all_attentions: Option<Vec<Tensor>> = if self.output_attentions {
            Some(vec![])
        } else {
            None
        };
        let mut all_global_attentions: Option<Vec<Tensor>> =
            if self.output_attentions & is_global_attention {
                Some(vec![])
            } else {
                None
            };

        let mut x: Option<Tensor> = None;
        let mut attention_weights: Option<Tensor>;
        let mut global_attention_weights: Option<Tensor>;

        for layer in &self.layers {
            let temp = if let Some(x_value) = &x {
                layer.forward_t(
                    x_value,
                    attention_mask,
                    &is_index_masked,
                    &is_index_global_attention,
                    is_global_attention,
                    train,
                )
            } else {
                layer.forward_t(
                    hidden_states,
                    attention_mask,
                    &is_index_masked,
                    &is_index_global_attention,
                    is_global_attention,
                    train,
                )
            };
            x = Some(temp.0);
            attention_weights = temp.1;
            global_attention_weights = temp.2;
            if let Some(attentions) = all_attentions.borrow_mut() {
                attentions.push(std::mem::take(&mut attention_weights.unwrap()).transpose(1, 2));
            };
            if let Some(global_attentions) = all_global_attentions.borrow_mut() {
                global_attentions
                    .push(std::mem::take(&mut global_attention_weights.unwrap()).transpose(2, 3));
            };
            if let Some(all_hidden_states) = all_hidden_states.borrow_mut() {
                all_hidden_states.push(x.as_ref().unwrap().copy());
            };
        }

        LongformerEncoderOutput {
            hidden_states: x.unwrap(),
            all_hidden_states,
            all_attentions,
            all_global_attentions,
        }
    }
}
