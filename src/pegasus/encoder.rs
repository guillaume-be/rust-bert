// Copyright 2021, Google and The HuggingFace Inc. team. All rights reserved.
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

use crate::bart::{BartEncoderOutput, _expand_mask};
use crate::common::dropout::Dropout;
use crate::mbart::MBartEncoderLayer;
use crate::pegasus::embeddings::SinusoidalPositionalEmbedding;
use crate::pegasus::PegasusConfig;
use std::borrow::{Borrow, BorrowMut};
use tch::{nn, Tensor};

pub type PegasusEncoderLayer = MBartEncoderLayer;

pub struct PegasusEncoder {
    dropout: Dropout,
    layer_norm: nn::LayerNorm,
    layers: Vec<PegasusEncoderLayer>,
    embed_positions: SinusoidalPositionalEmbedding,
    output_attentions: bool,
    output_hidden_states: bool,
    scale_embedding: f64,
}

impl PegasusEncoder {
    pub fn new<'p, P>(p: P, config: &PegasusConfig) -> PegasusEncoder
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let output_attentions = config.output_attentions.unwrap_or(false);
        let output_hidden_states = config.output_hidden_states.unwrap_or(false);
        let scale_embedding = match config.scale_embedding {
            Some(value) if value => (config.d_model as f64).sqrt(),
            _ => 1.0,
        };

        let dropout = Dropout::new(config.dropout);
        let layer_norm_config = nn::LayerNormConfig {
            eps: 1e-5,
            ..Default::default()
        };
        let layer_norm = nn::layer_norm(p / "layer_norm", vec![config.d_model], layer_norm_config);

        let embed_positions = SinusoidalPositionalEmbedding::new(
            p / "embed_positions",
            config.max_position_embeddings,
            config.d_model,
        );

        let mut layers: Vec<PegasusEncoderLayer> = vec![];
        let p_layers = p / "layers";
        for layer_index in 0..config.encoder_layers {
            layers.push(PegasusEncoderLayer::new(&p_layers / layer_index, config));
        }

        PegasusEncoder {
            dropout,
            layer_norm,
            layers,
            embed_positions,
            output_attentions,
            output_hidden_states,
            scale_embedding,
        }
    }

    pub fn forward_t(
        &self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
        embeddings: &nn::Embedding,
        train: bool,
    ) -> PegasusEncoderOutput {
        let x = input_ids.apply(embeddings) * self.scale_embedding;
        let positions = self.embed_positions.forward(input_ids, 0);
        let x = if positions.kind() != x.kind() {
            positions.to_kind(x.kind()) + x
        } else {
            positions + x
        };

        let attention_mask = attention_mask.map(|mask| _expand_mask(mask, None, x.kind()));

        let mut hidden_state = x.apply_t(&self.dropout, train);

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

        let mut attention_weights: Option<Tensor>;

        for layer in &self.layers {
            let temp = layer.forward_t(&hidden_state, attention_mask.as_ref(), train);
            hidden_state = temp.0;
            attention_weights = temp.1;
            if let Some(attentions) = all_attentions.borrow_mut() {
                attentions.push(attention_weights.as_ref().unwrap().copy());
            };
            if let Some(hidden_states) = all_hidden_states.borrow_mut() {
                hidden_states.push(hidden_state.as_ref().copy());
            };
        }

        PegasusEncoderOutput {
            hidden_state: hidden_state.apply(&self.layer_norm),
            all_hidden_states,
            all_attentions,
        }
    }
}

/// Container holding a Pegasus encoder output
pub type PegasusEncoderOutput = BartEncoderOutput;
