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

use crate::prophetnet::attention::{
    LayerState, ProphetNetAttention, ProphetNetFeedForward, ProphetNetNgramAttention,
};
use crate::prophetnet::ProphetNetConfig;
use crate::RustBertError;
use std::borrow::Borrow;
use tch::{nn, Device, Kind, Tensor};

fn ngram_attention_bias(sequence_length: i64, ngram: i64, device: Device) -> Tensor {
    let left_block = Tensor::ones(
        &[ngram, sequence_length, sequence_length],
        (Kind::Float, device),
    ) * f64::NEG_INFINITY;
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

        let hidden_states = hidden_states
            .apply_t(&self.feed_forward, train)
            .apply(&self.feed_forward_layer_norm);

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
