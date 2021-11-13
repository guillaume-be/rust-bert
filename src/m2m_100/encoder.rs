// Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
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

use crate::bart::{BartEncoderOutput, _expand_mask};
use crate::common::dropout::Dropout;
use crate::m2m_100::embeddings::SinusoidalPositionalEmbedding;
use crate::m2m_100::M2M100Config;
use crate::mbart::MBartEncoderLayer;
use std::borrow::{Borrow, BorrowMut};
use tch::{nn, Tensor};

pub type M2M100EncoderLayer = MBartEncoderLayer;

pub struct M2M100Encoder {
    dropout: Dropout,
    layer_norm: nn::LayerNorm,
    layers: Vec<M2M100EncoderLayer>,
    embed_positions: SinusoidalPositionalEmbedding,
    output_attentions: bool,
    output_hidden_states: bool,
    scale_embedding: f64,
}

impl M2M100Encoder {
    pub fn new<'p, P>(p: P, config: &M2M100Config) -> M2M100Encoder
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let output_attentions = config.output_attentions.unwrap_or(false);
        let output_hidden_states = config.output_hidden_states.unwrap_or(false);

        let scale_embedding = if let Some(scale_embeddings) = config.scale_embedding {
            if scale_embeddings {
                (config.d_model as f64).sqrt()
            } else {
                1.0
            }
        } else {
            1.0
        };

        let dropout = Dropout::new(config.dropout);

        let layer_norm = nn::layer_norm(p / "layer_norm", vec![config.d_model], Default::default());

        let embed_positions = SinusoidalPositionalEmbedding::new(
            p / "embed_positions",
            config.max_position_embeddings,
            config.d_model,
            config.pad_token_id.unwrap_or(1),
        );

        let mut layers: Vec<M2M100EncoderLayer> = vec![];
        let p_layers = p / "layers";
        for layer_index in 0..config.encoder_layers {
            layers.push(M2M100EncoderLayer::new(&p_layers / layer_index, config));
        }

        M2M100Encoder {
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
    ) -> M2M100EncoderOutput {
        let x = input_ids.apply(embeddings) * self.scale_embedding;
        let x = &self.embed_positions.forward(input_ids, 0, x.kind()) + x;
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

        hidden_state = hidden_state.apply(&self.layer_norm);

        M2M100EncoderOutput {
            hidden_state,
            all_hidden_states,
            all_attentions,
        }
    }
}

/// Container holding a M2M100 encoder output
pub type M2M100EncoderOutput = BartEncoderOutput;
