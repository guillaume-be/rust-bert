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

use crate::bart::{BartDecoderOutput, _expand_mask, _prepare_decoder_attention_mask};
use crate::common::dropout::Dropout;
use crate::mbart::MBartDecoderLayer;
use crate::pegasus::attention::LayerState;
use crate::pegasus::embeddings::SinusoidalPositionalEmbedding;
use crate::pegasus::PegasusConfig;
use std::borrow::{Borrow, BorrowMut};
use tch::{nn, Tensor};

pub type PegasusDecoderLayer = MBartDecoderLayer;

pub struct PegasusDecoder {
    dropout: Dropout,
    layer_norm: nn::LayerNorm,
    layers: Vec<PegasusDecoderLayer>,
    embed_positions: SinusoidalPositionalEmbedding,
    output_attentions: bool,
    output_hidden_states: bool,
    output_past: bool,
    scale_embedding: f64,
}

impl PegasusDecoder {
    pub fn new<'p, P>(p: P, config: &PegasusConfig) -> PegasusDecoder
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let output_past = config.output_past.unwrap_or(true);
        let output_attentions = config.output_attentions.unwrap_or(false);
        let output_hidden_states = config.output_hidden_states.unwrap_or(false);

        let scale_embedding = match config.scale_embedding {
            Some(value) => {
                if value {
                    (config.d_model as f64).sqrt()
                } else {
                    1.0
                }
            }
            None => 1.0,
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

        let mut layers: Vec<PegasusDecoderLayer> = vec![];
        let p_layers = p / "layers";
        for layer_index in 0..config.decoder_layers {
            layers.push(PegasusDecoderLayer::new(&p_layers / layer_index, config));
        }

        PegasusDecoder {
            dropout,
            layer_norm,
            layers,
            embed_positions,
            output_attentions,
            output_hidden_states,
            output_past,
            scale_embedding,
        }
    }

    pub fn forward_t(
        &self,
        input_ids: &Tensor,
        encoder_hidden_states: &Tensor,
        encoder_attention_mask: Option<&Tensor>,
        decoder_attention_mask: Option<&Tensor>,
        embeddings: &nn::Embedding,
        old_layer_states: Option<Vec<(Option<LayerState>, Option<LayerState>)>>,
        train: bool,
    ) -> PegasusDecoderOutput {
        let past_key_values_length = if let Some(old_layer_states_values) = &old_layer_states {
            if let Some(old_value_state) = &old_layer_states_values[0].0 {
                old_value_state.prev_key.size()[2]
            } else {
                0
            }
        } else {
            0
        };

        let x = input_ids.apply(embeddings) * self.scale_embedding;
        let positions = self
            .embed_positions
            .forward(input_ids, past_key_values_length);
        let x = if positions.kind() != x.kind() {
            positions.to_kind(x.kind()) + x
        } else {
            positions + x
        };

        let decoder_attention_mask = _prepare_decoder_attention_mask(
            decoder_attention_mask,
            input_ids.size().as_slice(),
            &x,
            past_key_values_length,
        );

        let encoder_attention_mask = encoder_attention_mask
            .map(|mask| _expand_mask(mask, Some(*input_ids.size().last().unwrap()), x.kind()));

        let mut hidden_state = x.apply_t(&self.dropout, train);
        let mut all_hidden_states: Option<Vec<Tensor>> = if self.output_hidden_states {
            Some(Vec::with_capacity(self.layers.len()))
        } else {
            None
        };
        let mut all_attentions: Option<Vec<Tensor>> = if self.output_attentions {
            Some(Vec::with_capacity(self.layers.len()))
        } else {
            None
        };
        let mut next_decoder_cache: Option<Vec<(Option<LayerState>, Option<LayerState>)>> =
            if self.output_past {
                if old_layer_states.is_some() {
                    old_layer_states
                } else {
                    Some(vec![(None, None); self.layers.len()])
                }
            } else {
                None
            };

        let mut attention_weights: Option<Tensor>;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let layer_state = match &next_decoder_cache {
                Some(values) => values[layer_idx].to_owned(),
                None => (None, None),
            };
            let temp = layer.forward_t(
                &hidden_state,
                encoder_hidden_states,
                encoder_attention_mask.as_ref(),
                decoder_attention_mask.as_ref(),
                layer_state,
                train,
            );
            hidden_state = temp.0;
            attention_weights = temp.1;
            if let Some(hidden_states) = all_hidden_states.borrow_mut() {
                hidden_states.push(hidden_state.as_ref().copy());
            };
            if let Some(attentions) = all_attentions.borrow_mut() {
                attentions.push(std::mem::take(&mut attention_weights.unwrap()));
            };
            if let Some(value) = &mut next_decoder_cache {
                value[layer_idx] = temp.2
            };
        }

        hidden_state = hidden_state.apply(&self.layer_norm);

        PegasusDecoderOutput {
            hidden_state,
            encoder_attention_mask,
            next_decoder_cache,
            all_hidden_states,
            all_attentions,
        }
    }
}

/// Container holding a Pegasus decoder output
pub type PegasusDecoderOutput = BartDecoderOutput;
