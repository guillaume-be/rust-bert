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

use crate::bart::{BartDecoderOutput, _expand_mask, _make_causal_mask};
use crate::common::dropout::Dropout;
use crate::m2m_100::embeddings::SinusoidalPositionalEmbedding;
use crate::m2m_100::{LayerState, M2M100Config};
use crate::mbart::MBartDecoderLayer;
use std::borrow::{Borrow, BorrowMut};
use tch::{nn, Tensor};

pub type M2M100DecoderLayer = MBartDecoderLayer;

pub struct M2M100Decoder {
    dropout: Dropout,
    layer_norm: nn::LayerNorm,
    layers: Vec<M2M100DecoderLayer>,
    embed_positions: SinusoidalPositionalEmbedding,
    output_attentions: bool,
    output_hidden_states: bool,
    output_past: bool,
    scale_embedding: f64,
}

impl M2M100Decoder {
    pub fn new<'p, P>(p: P, config: &M2M100Config) -> M2M100Decoder
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let output_past = config.output_past.unwrap_or(true);
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

        let mut layers: Vec<M2M100DecoderLayer> = vec![];
        let p_layers = p / "layers";
        for layer_index in 0..config.decoder_layers {
            layers.push(M2M100DecoderLayer::new(&p_layers / layer_index, config));
        }

        M2M100Decoder {
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
    ) -> M2M100DecoderOutput {
        let past_key_values_length = if let Some(old_layer_states_values) = &old_layer_states {
            if let Some(old_value_state) = &old_layer_states_values[0].0 {
                old_value_state.prev_key.size()[2]
            } else {
                0
            }
        } else {
            0
        };
        let input_shape = input_ids.size();
        let sequence_length = input_shape[1];

        let x = input_ids.apply(embeddings) * self.scale_embedding;
        let positions = self
            .embed_positions
            .forward(input_ids, past_key_values_length, x.kind());
        let x = x + positions;

        let causal_mask = if sequence_length > 1 {
            Some(_make_causal_mask(
                input_ids.size().as_slice(),
                x.kind(),
                x.device(),
                past_key_values_length,
            ))
        } else {
            None
        };

        let decoder_attention_mask = decoder_attention_mask.map(|attention_mask| {
            if let Some(causal_mask) = causal_mask {
                causal_mask + _expand_mask(attention_mask, Some(sequence_length), x.kind())
            } else {
                _expand_mask(attention_mask, Some(sequence_length), x.kind())
            }
        });

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

        M2M100DecoderOutput {
            hidden_state: hidden_state.apply(&self.layer_norm),
            encoder_attention_mask,
            next_decoder_cache,
            all_hidden_states,
            all_attentions,
        }
    }
}

/// Container holding a M2M100 decoder output
pub type M2M100DecoderOutput = BartDecoderOutput;
