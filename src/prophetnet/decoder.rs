// Copyright 2020 The Microsoft Authors and The HuggingFace Inc. team.
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
use crate::common::embeddings::process_ids_embeddings_pair;
use crate::common::kind::get_negative_infinity;
use crate::prophetnet::attention::{
    compute_all_stream_relative_buckets, LayerState, ProphetNetAttention, ProphetNetFeedForward,
    ProphetNetNgramAttention,
};
use crate::prophetnet::embeddings::ProphetNetPositionalEmbeddings;
use crate::prophetnet::ProphetNetConfig;
use crate::RustBertError;
use std::borrow::{Borrow, BorrowMut};
use tch::nn::Init;
use tch::{nn, Device, Kind, Tensor};

fn ngram_attention_bias(sequence_length: i64, ngram: i64, device: Device, kind: Kind) -> Tensor {
    let left_block = Tensor::ones(&[ngram, sequence_length, sequence_length], (kind, device))
        * get_negative_infinity(kind).unwrap();
    let right_block = left_block.copy();
    for stream_idx in 0..ngram {
        let _ = right_block.get(stream_idx).fill_diagonal_(0, false);
        let _ = left_block.get(stream_idx).triu_(-stream_idx + 1);
    }
    let _ = left_block.slice(2, 0, sequence_length, 1).fill_(0);
    Tensor::cat(&[left_block, right_block], 2)
}

pub struct ProphetNetDecoderLayer {
    self_attention: ProphetNetNgramAttention,
    self_attention_layer_norm: nn::LayerNorm,
    cross_attention: Option<ProphetNetAttention>,
    cross_attention_layer_norm: Option<nn::LayerNorm>,
    feed_forward: ProphetNetFeedForward,
    feed_forward_layer_norm: nn::LayerNorm,
}

impl ProphetNetDecoderLayer {
    pub fn new<'p, P>(
        p: P,
        config: &ProphetNetConfig,
    ) -> Result<ProphetNetDecoderLayer, RustBertError>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let self_attention = ProphetNetNgramAttention::new(p / "self_attn", config);
        let self_attention_layer_norm = nn::layer_norm(
            p / "self_attn_layer_norm",
            vec![config.hidden_size],
            Default::default(),
        );

        let (cross_attention, cross_attention_layer_norm) =
            if config.add_cross_attention.unwrap_or(true) {
                let cross_attention = ProphetNetAttention::new(
                    p / "cross_attn",
                    config,
                    config.num_decoder_attention_heads,
                )?;
                let cross_attention_layer_norm = nn::layer_norm(
                    p / "cross_attn_layer_norm",
                    vec![config.hidden_size],
                    Default::default(),
                );
                (Some(cross_attention), Some(cross_attention_layer_norm))
            } else {
                (None, None)
            };

        let feed_forward =
            ProphetNetFeedForward::new(p / "feed_forward", config, config.decoder_ffn_dim);
        let feed_forward_layer_norm = nn::layer_norm(
            p / "feed_forward_layer_norm",
            vec![config.hidden_size],
            Default::default(),
        );

        Ok(ProphetNetDecoderLayer {
            self_attention,
            self_attention_layer_norm,
            cross_attention,
            cross_attention_layer_norm,
            feed_forward,
            feed_forward_layer_norm,
        })
    }

    pub fn forward_t(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: Option<&Tensor>,
        encoder_attention_mask: Option<&Tensor>,
        layer_states: (Option<LayerState>, Option<LayerState>),
        attention_mask: Option<&Tensor>,
        extended_predict_attention_mask: Option<&Tensor>,
        main_relative_position_buckets: Option<&Tensor>,
        predict_relative_position_buckets: Option<&Tensor>,
        position_ids: &Tensor,
        train: bool,
    ) -> ProphetNetDecoderLayerOutput {
        let (
            ngram_attention_output,
            self_attention_weights,
            self_attention_weights_ngram,
            new_self_layer_state,
        ) = self.self_attention.forward_t(
            hidden_states,
            layer_states.0,
            attention_mask,
            extended_predict_attention_mask,
            main_relative_position_buckets,
            predict_relative_position_buckets,
            position_ids,
            train,
        );

        let mut hidden_states =
            (hidden_states + ngram_attention_output).apply(&self.self_attention_layer_norm);
        let (cross_attention_weights, new_cross_layer_state) = if let Some(encoder_hidden_states) =
            encoder_hidden_states
        {
            let (attention_output, cross_attention_weights, new_cross_layer_state) = self.cross_attention.as_ref().expect("Encoder hidden states were provided but model was not set up with a cross attention layer").forward_t(&hidden_states, Some(encoder_hidden_states), encoder_attention_mask,layer_states.1, train);
            hidden_states =
                (attention_output + hidden_states).apply(self.cross_attention_layer_norm.as_ref().expect("Encoder hidden states were provided but model was not set up with a cross attention layer"));
            (cross_attention_weights, new_cross_layer_state)
        } else {
            (None, None)
        };

        let feed_forward_output = hidden_states.apply_t(&self.feed_forward, train);

        let hidden_states =
            (feed_forward_output + hidden_states).apply(&self.feed_forward_layer_norm);

        ProphetNetDecoderLayerOutput {
            hidden_states,
            self_attention_weights,
            self_attention_weights_ngram,
            cross_attention_weights,
            layer_states: (new_self_layer_state, new_cross_layer_state),
        }
    }
}

///Container holding a ProphetNet decoder layer output
pub struct ProphetNetDecoderLayerOutput {
    pub hidden_states: Tensor,
    pub self_attention_weights: Option<Tensor>,
    pub self_attention_weights_ngram: Option<Tensor>,
    pub cross_attention_weights: Option<Tensor>,
    pub layer_states: (Option<LayerState>, Option<LayerState>),
}

pub struct ProphetNetDecoder {
    ngram: i64,
    num_buckets: i64,
    relative_max_distance: i64,
    max_target_positions: i64,
    position_embeddings: ProphetNetPositionalEmbeddings,
    embeddings_layer_norm: nn::LayerNorm,
    ngram_embeddings: Tensor,
    layers: Vec<ProphetNetDecoderLayer>,
    dropout: Dropout,
    output_attentions: bool,
    output_hidden_states: bool,
    num_attention_heads: i64,
    add_cross_attention: bool,
}

impl ProphetNetDecoder {
    pub fn new<'p, P>(p: P, config: &ProphetNetConfig) -> Result<ProphetNetDecoder, RustBertError>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let position_embeddings =
            ProphetNetPositionalEmbeddings::new(p / "position_embeddings", config);
        let embeddings_layer_norm = nn::layer_norm(
            p / "embeddings_layer_norm",
            vec![config.hidden_size],
            Default::default(),
        );

        let mut layers: Vec<ProphetNetDecoderLayer> =
            Vec::with_capacity(config.num_decoder_layers as usize);
        let p_layers = p / "layers";
        for layer_index in 0..config.num_decoder_layers {
            layers.push(ProphetNetDecoderLayer::new(
                &p_layers / layer_index,
                config,
            )?);
        }

        let dropout = Dropout::new(config.dropout);

        let p_ngram_embedding = p / "ngram_embeddings";
        let ngram_embeddings = p_ngram_embedding.var(
            "weight",
            &[config.ngram, config.hidden_size],
            Init::KaimingUniform,
        );

        let output_attentions = config.output_attentions.unwrap_or(false);
        let output_hidden_states = config.output_hidden_states.unwrap_or(false);

        let num_attention_heads = config.num_decoder_attention_heads;
        let ngram = config.ngram;
        let num_buckets = config.num_buckets;
        let relative_max_distance = config.relative_max_distance;
        let max_target_positions = config.max_position_embeddings;
        let add_cross_attention = config.add_cross_attention.unwrap_or(true);

        Ok(ProphetNetDecoder {
            ngram,
            num_buckets,
            relative_max_distance,
            max_target_positions,
            position_embeddings,
            embeddings_layer_norm,
            ngram_embeddings,
            layers,
            dropout,
            output_attentions,
            output_hidden_states,
            num_attention_heads,
            add_cross_attention,
        })
    }

    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
        encoder_attention_mask: Option<&Tensor>,
        old_layer_states: Option<Vec<(Option<LayerState>, Option<LayerState>)>>,
        input_embeds: Option<&Tensor>,
        word_embeddings: Option<&nn::Embedding>,
        train: bool,
    ) -> Result<ProphetNetDecoderOutput, RustBertError> {
        let (calc_input_embeddings, _, _) = process_ids_embeddings_pair(
            input_ids,
            input_embeds,
            word_embeddings.ok_or_else(|| {
                RustBertError::ValueError(
                    "Embeddings must be provided if input_embeds is not given".into(),
                )
            })?,
        )?;
        let input_embeds = input_embeds.unwrap_or_else(|| calc_input_embeddings.as_ref().unwrap());

        let input_size = input_embeds.size();
        let (batch_size, sequence_length) = (input_size[0], input_size[1]);

        let prev_num_input_ids = if let Some(old_layer_states_vec) = &old_layer_states {
            old_layer_states_vec[0]
                .0
                .as_ref()
                .map(|layer_states| layer_states.prev_key.size()[2])
        } else {
            None
        };

        let (main_stream_pos_embed, position_ids) = self.position_embeddings.forward(
            &input_size[..2],
            input_embeds.device(),
            None,
            prev_num_input_ids,
            None,
        );

        let (main_relative_position_buckets, predict_relative_position_buckets) =
            if old_layer_states.is_some() {
                (None, None)
            } else {
                let (main_relative_buckets, predict_relative_buckets) =
                    self.compute_buffered_relative_buckets(&position_ids);
                (Some(main_relative_buckets), Some(predict_relative_buckets))
            };

        let predicting_stream_pos_embed = self.position_embeddings._forward(&(&position_ids + 1));

        let hidden_states = (input_embeds + main_stream_pos_embed).transpose(0, 1);

        let (mut ngram_hidden_states, extended_attention_mask, extended_predict_attention_mask) = {
            let mut ngram_hidden_states = Vec::with_capacity(self.ngram as usize);
            if old_layer_states.is_some() {
                for ngram in 0..self.ngram {
                    ngram_hidden_states.push(
                        (&self.ngram_embeddings.get(ngram - 1) + &predicting_stream_pos_embed)
                            .transpose(0, 1)
                            .repeat(&[1, batch_size, 1]),
                    );
                }
                (ngram_hidden_states, None, None)
            } else {
                for ngram in 0..self.ngram {
                    ngram_hidden_states.push(
                        (&self.ngram_embeddings.get(ngram - 1) + &predicting_stream_pos_embed)
                            .transpose(0, 1),
                    );
                }
                let extended_attention_mask =
                    self.prepare_attention_mask(&hidden_states, attention_mask);
                let extended_predict_attention_mask =
                    self.prepare_predict_attention_mask(&hidden_states, attention_mask);
                (
                    ngram_hidden_states,
                    Some(extended_attention_mask),
                    Some(extended_predict_attention_mask),
                )
            }
        };

        let extended_encoder_attention_mask =
            encoder_attention_mask.map(|encoder_attention_mask_value| {
                encoder_attention_mask_value.ones_like()
                    - encoder_attention_mask_value.unsqueeze(1).repeat(&[
                        self.num_attention_heads,
                        1,
                        1,
                    ])
            });

        ngram_hidden_states.insert(0, hidden_states);
        let hidden_states = Tensor::cat(ngram_hidden_states.as_slice(), 0)
            .apply(&self.embeddings_layer_norm)
            .apply_t(&self.dropout, train);

        let encoder_hidden_states = encoder_hidden_states.map(|tensor| tensor.transpose(0, 1));

        let mut all_main_stream_hidden_states: Option<Vec<Tensor>> = if self.output_hidden_states {
            Some(vec![])
        } else {
            None
        };
        let mut all_ngram_stream_hidden_states: Option<Vec<Tensor>> =
            if self.output_hidden_states & (self.ngram > 0) {
                Some(vec![])
            } else {
                None
            };
        let mut all_main_stream_attentions: Option<Vec<Tensor>> = if self.output_attentions {
            Some(vec![])
        } else {
            None
        };
        let mut all_ngram_stream_attentions: Option<Vec<Tensor>> = if self.output_attentions {
            Some(vec![])
        } else {
            None
        };
        let mut all_cross_attentions: Option<Vec<Tensor>> =
            if self.output_attentions & self.add_cross_attention {
                Some(vec![])
            } else {
                None
            };

        let mut next_decoder_cache: Vec<(Option<LayerState>, Option<LayerState>)> =
            Vec::with_capacity(self.layers.len());
        let mut old_layer_states = old_layer_states.map(|mut layer_states_vec| {
            layer_states_vec.reverse();
            layer_states_vec
        });

        let mut x: Option<Tensor> = None;

        for layer in &self.layers {
            let layer_state = if let Some(layer_states_vec) = old_layer_states.borrow_mut() {
                layer_states_vec.pop().unwrap()
            } else {
                (None, None)
            };
            let temp = if let Some(x_value) = &x {
                layer.forward_t(
                    x_value,
                    encoder_hidden_states.as_ref(),
                    extended_encoder_attention_mask.as_ref(),
                    layer_state,
                    extended_attention_mask.as_ref(),
                    extended_predict_attention_mask.as_ref(),
                    main_relative_position_buckets.as_ref(),
                    predict_relative_position_buckets.as_ref(),
                    &position_ids,
                    train,
                )
            } else {
                layer.forward_t(
                    &hidden_states,
                    encoder_hidden_states.as_ref(),
                    extended_encoder_attention_mask.as_ref(),
                    layer_state,
                    extended_attention_mask.as_ref(),
                    extended_predict_attention_mask.as_ref(),
                    main_relative_position_buckets.as_ref(),
                    predict_relative_position_buckets.as_ref(),
                    &position_ids,
                    train,
                )
            };
            x = Some(temp.hidden_states);

            if let Some(all_main_stream_attentions) = all_main_stream_attentions.borrow_mut() {
                all_main_stream_attentions.push(temp.self_attention_weights.unwrap());
            };
            if let Some(all_ngram_stream_attentions) = all_ngram_stream_attentions.borrow_mut() {
                all_ngram_stream_attentions.push(temp.self_attention_weights_ngram.unwrap());
            };
            if let Some(all_cross_attentions) = all_cross_attentions.borrow_mut() {
                all_cross_attentions.push(temp.cross_attention_weights.unwrap());
            };
            if let Some(main_stream_hidden_states) = all_main_stream_hidden_states.borrow_mut() {
                main_stream_hidden_states.push(
                    x.as_ref()
                        .unwrap()
                        .slice(0, 0, sequence_length, 1)
                        .transpose(0, 1),
                );
            };
            if let Some(ngram_stream_hidden_states) = all_ngram_stream_hidden_states.borrow_mut() {
                ngram_stream_hidden_states.push(
                    x.as_ref()
                        .unwrap()
                        .slice(0, sequence_length, x.as_ref().unwrap().size()[0], 1)
                        .transpose(0, 1),
                );
            };
            next_decoder_cache.push(temp.layer_states);
        }
        let x = x.unwrap();

        let last_hidden_state = x.slice(0, 0, sequence_length, 1).transpose(0, 1);
        let last_hidden_state_ngram = if self.ngram > 0 {
            Some(x.slice(0, sequence_length, x.size()[0], 1).transpose(0, 1))
        } else {
            None
        };

        Ok(ProphetNetDecoderOutput {
            hidden_states: last_hidden_state,
            ngram_hidden_states: last_hidden_state_ngram,
            all_hidden_states: all_main_stream_hidden_states,
            all_ngram_hidden_states: all_ngram_stream_hidden_states,
            all_attentions: all_main_stream_attentions,
            all_ngram_attentions: all_ngram_stream_attentions,
            all_cross_attentions,
            next_decoder_cache: Some(next_decoder_cache),
        })
    }

    fn compute_buffered_relative_buckets(&self, position_ids: &Tensor) -> (Tensor, Tensor) {
        let input_size = position_ids.size();
        let (batch_size, sequence_length) = (input_size[0], input_size[1]);

        let position_ids = Tensor::arange_start(
            1,
            self.max_target_positions,
            (Kind::Int64, position_ids.device()),
        )
        .repeat(&[1, 1]);

        let (main_relative_buckets, predict_relative_buckets) = compute_all_stream_relative_buckets(
            self.num_buckets,
            self.relative_max_distance,
            &position_ids,
        );

        let main_relative_buckets = main_relative_buckets
            .slice(1, 0, sequence_length, 1)
            .slice(2, 0, sequence_length, 1)
            .repeat(&[batch_size, 1, 1]);

        let predict_relative_buckets = Tensor::cat(
            &[
                predict_relative_buckets
                    .slice(1, 0, sequence_length, 1)
                    .slice(2, 0, sequence_length, 1),
                predict_relative_buckets
                    .slice(1, 0, sequence_length, 1)
                    .slice(
                        2,
                        self.max_target_positions,
                        self.max_target_positions + sequence_length,
                        1,
                    ),
            ],
            2,
        )
        .repeat(&[batch_size, 1, 1]);

        (main_relative_buckets, predict_relative_buckets)
    }

    fn prepare_attention_mask(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Tensor {
        let input_size = hidden_states.size();
        let (sequence_length, batch_size) = (input_size[0], input_size[1]);

        let causal_mask = Tensor::full(
            &[sequence_length, sequence_length],
            get_negative_infinity(hidden_states.kind()).unwrap(),
            (hidden_states.kind(), hidden_states.device()),
        )
        .triu_(1);

        let extended_causal_mask = causal_mask
            .unsqueeze(0)
            .expand(&[batch_size, sequence_length, sequence_length], true);

        let extended_attention_mask = if let Some(attention_mask_value) = attention_mask {
            let extended_attention_mask =
                ((attention_mask_value.ones_like() - attention_mask_value.unsqueeze(1)) * -10000.0)
                    .to_kind(causal_mask.kind());
            extended_causal_mask + extended_attention_mask
        } else {
            extended_causal_mask
        };

        extended_attention_mask.repeat(&[self.num_attention_heads, 1, 1])
    }

    fn prepare_predict_attention_mask(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Tensor {
        let input_size = hidden_states.size();
        let (sequence_length, batch_size) = (input_size[0], input_size[1]);

        let predict_causal_mask = ngram_attention_bias(
            self.max_target_positions,
            self.ngram,
            hidden_states.device(),
            hidden_states.kind(),
        );

        let predict_causal_mask = Tensor::cat(
            &[
                predict_causal_mask
                    .slice(1, 0, sequence_length, 1)
                    .slice(2, 0, sequence_length, 1),
                predict_causal_mask.slice(1, 0, sequence_length, 1).slice(
                    2,
                    self.max_target_positions,
                    self.max_target_positions + sequence_length,
                    1,
                ),
            ],
            -1,
        );

        let predict_causal_mask_shape = predict_causal_mask.size();
        let mut extended_shape = vec![predict_causal_mask_shape[0]];
        extended_shape.push(batch_size);
        extended_shape.extend_from_slice(&predict_causal_mask_shape[1..]);
        let extended_predict_causal_mask = predict_causal_mask
            .unsqueeze(1)
            .expand(extended_shape.as_slice(), true);

        let extended_attention_mask = if let Some(attention_mask_value) = attention_mask {
            let extended_attention_mask = (attention_mask_value.ones_like()
                - attention_mask_value.unsqueeze(0).unsqueeze(2))
                * -10000.0;
            let extended_attention_mask = extended_attention_mask.expand(
                &[self.ngram, batch_size, sequence_length, sequence_length],
                true,
            );
            let extended_attention_mask = Tensor::cat(
                &[
                    &extended_attention_mask,
                    &extended_attention_mask.zeros_like(),
                ],
                -1,
            );
            extended_predict_causal_mask + extended_attention_mask
        } else {
            extended_predict_causal_mask
        };

        extended_attention_mask.repeat(&[1, self.num_attention_heads, 1, 1])
    }
}

///Container holding a ProphetNet decoder  output
pub struct ProphetNetDecoderOutput {
    /// last decoder layer hidden state
    pub hidden_states: Tensor,
    /// last decoder layer ngram hidden state
    pub ngram_hidden_states: Option<Tensor>,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<Tensor>>,
    /// Hidden states (ngram) for all intermediate layers
    pub all_ngram_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all intermediate layers
    pub all_attentions: Option<Vec<Tensor>>,
    /// Ngram attention weights for all intermediate layers
    pub all_ngram_attentions: Option<Vec<Tensor>>,
    /// Cross attention weights for all intermediate layers
    pub all_cross_attentions: Option<Vec<Tensor>>,
    /// Cached outputs of the model (attention layers keys and values) if the model is used for generation
    pub next_decoder_cache: Option<Vec<(Option<LayerState>, Option<LayerState>)>>,
}
