// Copyright 2019-present, the HuggingFace Inc. team, The Google AI Language Team and Facebook, Inc.
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

use crate::common::activations::TensorFunction;
use crate::common::dropout::Dropout;
use crate::distilbert::attention::MultiHeadSelfAttention;
use crate::distilbert::distilbert_model::DistilBertConfig;
use std::borrow::{Borrow, BorrowMut};
use tch::nn::LayerNorm;
use tch::{nn, Tensor};

pub struct FeedForwardNetwork {
    lin1: nn::Linear,
    lin2: nn::Linear,
    dropout: Dropout,
    activation: TensorFunction,
}

impl FeedForwardNetwork {
    pub fn new<'p, P>(p: P, config: &DistilBertConfig) -> FeedForwardNetwork
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let lin1 = nn::linear(
            p / "lin1",
            config.dim,
            config.hidden_dim,
            Default::default(),
        );
        let lin2 = nn::linear(
            p / "lin2",
            config.hidden_dim,
            config.dim,
            Default::default(),
        );
        let dropout = Dropout::new(config.dropout);
        let activation = config.activation.get_function();
        FeedForwardNetwork {
            lin1,
            lin2,
            dropout,
            activation,
        }
    }

    pub fn forward_t(&self, input: &Tensor, train: bool) -> Tensor {
        (self.activation.get_fn())(&input.apply(&self.lin1))
            .apply(&self.lin2)
            .apply_t(&self.dropout, train)
    }
}

pub struct TransformerBlock {
    attention: MultiHeadSelfAttention,
    sa_layer_norm: LayerNorm,
    ffn: FeedForwardNetwork,
    output_layer_norm: LayerNorm,
}

impl TransformerBlock {
    pub fn new<'p, P>(p: P, config: &DistilBertConfig) -> TransformerBlock
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let attention = MultiHeadSelfAttention::new(p / "attention", config);
        let layer_norm_config = nn::LayerNormConfig {
            eps: 1e-12,
            ..Default::default()
        };
        let sa_layer_norm =
            nn::layer_norm(p / "sa_layer_norm", vec![config.dim], layer_norm_config);
        let ffn = FeedForwardNetwork::new(p / "ffn", config);
        let output_layer_norm =
            nn::layer_norm(p / "output_layer_norm", vec![config.dim], layer_norm_config);

        TransformerBlock {
            attention,
            sa_layer_norm,
            ffn,
            output_layer_norm,
        }
    }

    pub fn forward_t(
        &self,
        input: &Tensor,
        mask: Option<&Tensor>,
        train: bool,
    ) -> (Tensor, Option<Tensor>) {
        let (output, sa_weights) = self.attention.forward_t(input, input, input, mask, train);
        let output = (input + &output).apply(&self.sa_layer_norm);
        let output = (&output + self.ffn.forward_t(&output, train)).apply(&self.output_layer_norm);
        (output, sa_weights)
    }
}

pub struct Transformer {
    output_attentions: bool,
    output_hidden_states: bool,
    layers: Vec<TransformerBlock>,
}

impl Transformer {
    pub fn new<'p, P>(p: P, config: &DistilBertConfig) -> Transformer
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow() / "layer";
        let output_attentions = config.output_attentions.unwrap_or(false);
        let output_hidden_states = config.output_hidden_states.unwrap_or(false);

        let mut layers: Vec<TransformerBlock> = vec![];
        for layer_index in 0..config.n_layers {
            layers.push(TransformerBlock::new(&p / layer_index, config));
        }

        Transformer {
            output_attentions,
            output_hidden_states,
            layers,
        }
    }

    pub fn forward_t(
        &self,
        input: &Tensor,
        mask: Option<&Tensor>,
        train: bool,
    ) -> DistilBertTransformerOutput {
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

        // let mut hidden_state = input.copy();
        let mut hidden_state: Option<Tensor> = None;
        let mut attention_weights: Option<Tensor>;

        for layer in &self.layers {
            let temp = if let Some(hidden_state) = &hidden_state {
                layer.forward_t(hidden_state, mask, train)
            } else {
                layer.forward_t(input, mask, train)
            };

            hidden_state = Some(temp.0);
            attention_weights = temp.1;
            if let Some(attentions) = all_attentions.borrow_mut() {
                attentions.push(std::mem::take(&mut attention_weights.unwrap()));
            };
            if let Some(hidden_states) = all_hidden_states.borrow_mut() {
                hidden_states.push(hidden_state.as_ref().unwrap().copy());
            };
        }

        DistilBertTransformerOutput {
            hidden_state: hidden_state.unwrap(),
            all_hidden_states,
            all_attentions,
        }
    }
}

/// Container for the DistilBert transformer output.
pub struct DistilBertTransformerOutput {
    /// Last hidden states from the model
    pub hidden_state: Tensor,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all intermediate layers
    pub all_attentions: Option<Vec<Tensor>>,
}
