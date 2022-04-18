// Copyright 2018 Google AI and Google Brain team.
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

use crate::albert::AlbertConfig;
use crate::{albert::attention::AlbertSelfAttention, common::activations::TensorFunction};
use std::borrow::{Borrow, BorrowMut};
use tch::{nn, Tensor};

pub struct AlbertLayer {
    attention: AlbertSelfAttention,
    full_layer_layer_norm: nn::LayerNorm,
    ffn: nn::Linear,
    ffn_output: nn::Linear,
    activation: TensorFunction,
}

impl AlbertLayer {
    pub fn new<'p, P>(p: P, config: &AlbertConfig) -> AlbertLayer
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let attention = AlbertSelfAttention::new(p / "attention", config);

        let layer_norm_eps = config.layer_norm_eps.unwrap_or(1e-12);
        let layer_norm_config = nn::LayerNormConfig {
            eps: layer_norm_eps,
            ..Default::default()
        };
        let full_layer_layer_norm = nn::layer_norm(
            p / "full_layer_layer_norm",
            vec![config.hidden_size],
            layer_norm_config,
        );

        let ffn = nn::linear(
            p / "ffn",
            config.hidden_size,
            config.intermediate_size,
            Default::default(),
        );
        let ffn_output = nn::linear(
            p / "ffn_output",
            config.intermediate_size,
            config.hidden_size,
            Default::default(),
        );

        let activation = config.hidden_act.get_function();

        AlbertLayer {
            attention,
            full_layer_layer_norm,
            ffn,
            ffn_output,
            activation,
        }
    }

    pub fn forward_t(
        &self,
        hidden_states: &Tensor,
        mask: Option<&Tensor>,
        train: bool,
    ) -> (Tensor, Option<Tensor>) {
        let (attention_output, attention_weights) =
            self.attention.forward_t(hidden_states, mask, train);
        let ffn_output = attention_output.apply(&self.ffn);
        let ffn_output: Tensor = (self.activation.get_fn())(&ffn_output);
        let ffn_output = ffn_output.apply(&self.ffn_output);
        let ffn_output = (ffn_output + attention_output).apply(&self.full_layer_layer_norm);

        (ffn_output, attention_weights)
    }
}

pub struct AlbertLayerGroup {
    output_hidden_states: bool,
    output_attentions: bool,
    layers: Vec<AlbertLayer>,
}

impl AlbertLayerGroup {
    pub fn new<'p, P>(p: P, config: &AlbertConfig) -> AlbertLayerGroup
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow() / "albert_layers";

        let output_attentions = config.output_attentions.unwrap_or(false);
        let output_hidden_states = config.output_hidden_states.unwrap_or(false);

        let mut layers: Vec<AlbertLayer> = vec![];
        for layer_index in 0..config.inner_group_num {
            layers.push(AlbertLayer::new(&p / layer_index, config));
        }

        AlbertLayerGroup {
            output_hidden_states,
            output_attentions,
            layers,
        }
    }

    pub fn forward_t(
        &self,
        hidden_states: &Tensor,
        mask: Option<&Tensor>,
        train: bool,
    ) -> (Tensor, Option<Vec<Tensor>>, Option<Vec<Tensor>>) {
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

        let mut hidden_state = hidden_states.copy();
        let mut attention_weights: Option<Tensor>;

        for layer in &self.layers {
            let temp = layer.forward_t(&hidden_state, mask, train);
            hidden_state = temp.0;
            attention_weights = temp.1;
            if let Some(attentions) = all_attentions.borrow_mut() {
                attentions.push(attention_weights.as_ref().unwrap().copy());
            };
            if let Some(hidden_states) = all_hidden_states.borrow_mut() {
                hidden_states.push(hidden_state.as_ref().copy());
            };
        }

        (hidden_state, all_hidden_states, all_attentions)
    }
}

pub struct AlbertTransformer {
    output_hidden_states: bool,
    output_attentions: bool,
    num_hidden_layers: i64,
    num_hidden_groups: i64,
    embedding_hidden_mapping_in: nn::Linear,
    layers: Vec<AlbertLayerGroup>,
}

impl AlbertTransformer {
    pub fn new<'p, P>(p: P, config: &AlbertConfig) -> AlbertTransformer
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let p_layers = p / "albert_layer_groups";

        let output_attentions = config.output_attentions.unwrap_or(false);
        let output_hidden_states = config.output_hidden_states.unwrap_or(false);

        let embedding_hidden_mapping_in = nn::linear(
            p / "embedding_hidden_mapping_in",
            config.embedding_size,
            config.hidden_size,
            Default::default(),
        );

        let mut layers: Vec<AlbertLayerGroup> = vec![];
        for layer_index in 0..config.inner_group_num {
            layers.push(AlbertLayerGroup::new(&p_layers / layer_index, config));
        }

        AlbertTransformer {
            output_hidden_states,
            output_attentions,
            num_hidden_layers: config.num_hidden_layers,
            num_hidden_groups: config.num_hidden_groups,
            embedding_hidden_mapping_in,
            layers,
        }
    }

    pub fn forward_t(
        &self,
        hidden_states: &Tensor,
        mask: Option<Tensor>,
        train: bool,
    ) -> AlbertTransformerOutput {
        let mut hidden_state = hidden_states.apply(&self.embedding_hidden_mapping_in);

        let mut all_hidden_states: Option<Vec<Tensor>> = if self.output_hidden_states {
            Some(vec![])
        } else {
            None
        };
        let mut all_attentions: Option<Vec<Vec<Tensor>>> = if self.output_attentions {
            Some(vec![])
        } else {
            None
        };

        for i in 0..self.num_hidden_layers {
            let group_idx = i / (self.num_hidden_layers / self.num_hidden_groups);
            let layer = &self.layers[group_idx as usize];

            if let Some(hidden_states) = all_hidden_states.borrow_mut() {
                hidden_states.push(hidden_state.as_ref().copy());
            };

            let temp = layer.forward_t(&hidden_state, mask.as_ref(), train);
            hidden_state = temp.0;
            let attention_weights = temp.2;
            if let Some(attentions) = all_attentions.borrow_mut() {
                attentions.push(attention_weights.unwrap());
            };
        }

        AlbertTransformerOutput {
            hidden_state,
            all_hidden_states,
            all_attentions,
        }
    }
}

/// Container holding the ALBERT transformer output
pub struct AlbertTransformerOutput {
    /// Last hidden states of the transformer
    pub hidden_state: Tensor,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all intermediate layers. As layers in ALBERT can be made of a number of sub-layers, a vector of vector is used to store al of the attentions
    pub all_attentions: Option<Vec<Vec<Tensor>>>,
}
