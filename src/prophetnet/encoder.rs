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
use crate::prophetnet::attention::{ProphetNetAttention, ProphetNetFeedForward};
use crate::prophetnet::embeddings::ProphetNetPositionalEmbeddings;
use crate::prophetnet::ProphetNetConfig;
use crate::RustBertError;
use std::borrow::{Borrow, BorrowMut};
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
        let (attention_output, attention_weights, _) =
            self.self_attention
                .forward_t(hidden_states, None, attention_mask, None, train);

        let hidden_states =
            (attention_output + hidden_states).apply(&self.self_attention_layer_norm);
        let feed_forward_output = hidden_states.apply_t(&self.feed_forward, train);
        let hidden_states =
            (hidden_states + feed_forward_output).apply(&self.feed_forward_layer_norm);

        (hidden_states, attention_weights)
    }
}

pub struct ProphetNetEncoder {
    position_embeddings: ProphetNetPositionalEmbeddings,
    embeddings_layer_norm: nn::LayerNorm,
    layers: Vec<ProphetNetEncoderLayer>,
    dropout: Dropout,
    output_attentions: bool,
    output_hidden_states: bool,
    num_attention_heads: i64,
}

impl ProphetNetEncoder {
    pub fn new<'p, P>(p: P, config: &ProphetNetConfig) -> Result<ProphetNetEncoder, RustBertError>
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

        let mut layers: Vec<ProphetNetEncoderLayer> =
            Vec::with_capacity(config.num_encoder_layers as usize);
        let p_layers = p / "layers";
        for layer_index in 0..config.num_encoder_layers {
            layers.push(ProphetNetEncoderLayer::new(
                &p_layers / layer_index,
                config,
            )?);
        }

        let dropout = Dropout::new(config.dropout);

        let output_attentions = config.output_attentions.unwrap_or(false);
        let output_hidden_states = config.output_hidden_states.unwrap_or(false);

        let num_attention_heads = config.num_encoder_attention_heads;

        Ok(ProphetNetEncoder {
            position_embeddings,
            embeddings_layer_norm,
            layers,
            dropout,
            output_attentions,
            output_hidden_states,
            num_attention_heads,
        })
    }

    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        word_embeddings: Option<&nn::Embedding>,
        train: bool,
    ) -> Result<ProphetNetEncoderOutput, RustBertError> {
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

        let extended_attention_mask = attention_mask.map(|mask| {
            ((mask.ones_like() - mask.unsqueeze(1).repeat(&[self.num_attention_heads, 1, 1]))
                * -10000.0)
                .to_kind(input_embeds.kind())
        });

        let (position_embeddings, _) = self.position_embeddings.forward(
            &input_embeds.size()[..2],
            input_embeds.device(),
            None,
            None,
            None,
        );

        let hidden_state = (input_embeds + position_embeddings)
            .apply(&self.embeddings_layer_norm)
            .apply_t(&self.dropout, train)
            .transpose(0, 1);

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

        let mut x: Option<Tensor> = None;
        let mut attention_weights: Option<Tensor>;

        for layer in &self.layers {
            let temp = if let Some(x_value) = &x {
                layer.forward_t(x_value, extended_attention_mask.as_ref(), train)
            } else {
                layer.forward_t(&hidden_state, extended_attention_mask.as_ref(), train)
            };
            x = Some(temp.0);
            attention_weights = temp.1;
            if let Some(attentions) = all_attentions.borrow_mut() {
                attentions.push(attention_weights.as_ref().unwrap().copy());
            };
            if let Some(hidden_states) = all_hidden_states.borrow_mut() {
                hidden_states.push(x.as_ref().unwrap().transpose(0, 1));
            };
        }

        Ok(ProphetNetEncoderOutput {
            hidden_states: x.unwrap().transpose(0, 1),
            all_hidden_states,
            all_attentions,
        })
    }
}

/// Container for the ProphetNet encoder output.
pub struct ProphetNetEncoderOutput {
    /// Last hidden states from the model
    pub hidden_states: Tensor,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all intermediate layers
    pub all_attentions: Option<Vec<Tensor>>,
}
