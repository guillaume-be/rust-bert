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

use crate::prophetnet::attention::{ProphetNetAttention, ProphetNetFeedForward};
use crate::prophetnet::ProphetNetConfig;
use crate::RustBertError;
use std::borrow::Borrow;
use tch::{nn, Tensor};

pub struct ProphetNetEncoderLayer {
    self_attention: ProphetNetAttention,
    self_attention_layer_norm: nn::LayerNorm,
    feed_forward: ProphetNetFeedForward,
    feed_forward_layer_norm: nn::LayerNorm,
}

impl ProphetNetEncoderLayer {
    pub fn new<'p, P>(
        p: P,
        config: &ProphetNetConfig,
    ) -> Result<ProphetNetEncoderLayer, RustBertError>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let self_attention =
            ProphetNetAttention::new(p / "self_attn", config, config.num_encoder_attention_heads)?;
        let self_attention_layer_norm = nn::layer_norm(
            p / "self_attn_layer_norm",
            vec![config.hidden_size],
            Default::default(),
        );

        let feed_forward =
            ProphetNetFeedForward::new(p / "feed_forward", config, config.encoder_ffn_dim);
        let feed_forward_layer_norm = nn::layer_norm(
            p / "feed_forward_layer_norm",
            vec![config.hidden_size],
            Default::default(),
        );

        Ok(ProphetNetEncoderLayer {
            self_attention,
            self_attention_layer_norm,
            feed_forward,
            feed_forward_layer_norm,
        })
    }

    pub fn forward_t(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        train: bool,
    ) -> (Tensor, Option<Tensor>) {
        let (attention_output, attention_weights) =
            self.self_attention
                .forward_t(hidden_states, None, attention_mask, None, train);
        let hidden_states = (attention_output + hidden_states)
            .apply(&self.self_attention_layer_norm)
            .apply_t(&self.feed_forward, train)
            .apply(&self.feed_forward_layer_norm);

        (hidden_states, attention_weights)
    }
}
