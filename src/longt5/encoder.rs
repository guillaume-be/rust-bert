// Copyright 2022 Google LLC., LongT5 Authors and HuggingFace Inc. team.
// Copyright 2022 Guillaume Becquin
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::longt5::attention::{
    LayerState, LongT5LayerCrossAttention, LongT5LayerLocalSelfAttention, LongT5LayerSelfAttention,
    LongT5LayerTransientGlobalSelfAttention,
};
use crate::longt5::longt5_model::EncoderAttentionType;
use crate::longt5::LongT5Config;
use crate::t5::{T5Block, T5BlockOutput, T5LayerFF};
use std::borrow::Borrow;
use tch::{nn, Kind, Scalar, Tensor};

pub type LongT5LayerFF = T5LayerFF;

enum LongT5AttentionLayer {
    SelfAttention(LongT5LayerSelfAttention),
    LocalSelfAttention(LongT5LayerLocalSelfAttention),
    GlobalSelfAttention(LongT5LayerTransientGlobalSelfAttention),
}

impl LongT5AttentionLayer {
    pub fn forward_t(
        &self,
        hidden_states: &Tensor,
        position_bias: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        layer_head_mask: Option<&Tensor>,
        layer_state: Option<LayerState>,
        train: bool,
    ) -> (Tensor, Option<Tensor>, Option<Tensor>, Option<LayerState>) {
        match self {
            LongT5AttentionLayer::SelfAttention(ref layer) => layer.forward_t(
                hidden_states,
                position_bias,
                attention_mask,
                layer_state,
                train,
            ),
            LongT5AttentionLayer::LocalSelfAttention(ref layer) => {
                let (output, position_bias, attention_weights) = layer.forward_t(
                    hidden_states,
                    attention_mask,
                    position_bias,
                    layer_head_mask,
                    train,
                );
                (output, attention_weights, position_bias, None)
            }
            LongT5AttentionLayer::GlobalSelfAttention(ref layer) => {
                let (output, position_bias, attention_weights) = layer.forward_t(
                    hidden_states,
                    attention_mask,
                    position_bias,
                    layer_head_mask,
                    train,
                );
                (output, attention_weights, position_bias, None)
            }
        }
    }
}

pub struct LongT5Block {
    attention_layer: LongT5AttentionLayer,
    cross_attention: Option<LongT5LayerCrossAttention>,
    ff_layer: LongT5LayerFF,
}

impl LongT5Block {
    pub fn new<'p, P>(
        p: P,
        config: &LongT5Config,
        has_relative_attention_bias: bool,
        is_decoder: bool,
        store_cache: bool,
        output_attentions: bool,
    ) -> LongT5Block
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow() / "layer";
        let mut module_index = 0;

        let attention_layer = if is_decoder {
            LongT5AttentionLayer::SelfAttention(LongT5LayerSelfAttention::new(
                &p / module_index,
                config,
                has_relative_attention_bias,
                is_decoder,
                store_cache,
                output_attentions,
            ))
        } else {
            match config.encoder_attention_type {
                Some(EncoderAttentionType::Local) | None => {
                    LongT5AttentionLayer::LocalSelfAttention(LongT5LayerLocalSelfAttention::new(
                        &p / module_index,
                        config,
                        has_relative_attention_bias,
                        is_decoder,
                        store_cache,
                    ))
                }
                Some(EncoderAttentionType::TransientGlobal) => {
                    LongT5AttentionLayer::GlobalSelfAttention(
                        LongT5LayerTransientGlobalSelfAttention::new(
                            &p / module_index,
                            config,
                            has_relative_attention_bias,
                            is_decoder,
                            store_cache,
                        ),
                    )
                }
            }
        };

        let cross_attention = if is_decoder {
            module_index += 1;
            Some(LongT5LayerCrossAttention::new(
                &p / module_index,
                &config.into(),
                false,
                is_decoder,
                store_cache,
                output_attentions,
            ))
        } else {
            None
        };
        module_index += 1;

        let ff_layer = LongT5LayerFF::new(&p / module_index, &config.into());

        LongT5Block {
            attention_layer,
            cross_attention,
            ff_layer,
        }
    }

    pub fn forward_t(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        position_bias: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
        encoder_attention_mask: Option<&Tensor>,
        encoder_decoder_position_bias: Option<&Tensor>,
        layer_head_mask: Option<&Tensor>,
        mut layer_states: (Option<LayerState>, Option<LayerState>),
        train: bool,
    ) -> LongT5BlockOutput {
        let (
            mut hidden_states,
            self_attention_weights,
            self_attention_position_bias,
            self_attention_layer_past,
        ) = self.attention_layer.forward_t(
            hidden_states,
            position_bias,
            attention_mask,
            layer_head_mask,
            layer_states.0,
            train,
        );

        hidden_states = T5Block::clamp_hidden_states(hidden_states);

        let (
            mut hidden_states,
            cross_attention_weights,
            cross_attention_position_bias,
            cross_attention_layer_past,
        ) = if self.cross_attention.is_some() & encoder_hidden_states.is_some() {
            let query_length = self_attention_layer_past
                .as_ref()
                .map(|value| value.prev_key.size()[2]);
            self.cross_attention.as_ref().unwrap().forward_t(
                &hidden_states,
                encoder_hidden_states,
                encoder_decoder_position_bias,
                encoder_attention_mask,
                layer_states.1,
                query_length,
                train,
            )
        } else {
            (hidden_states, None, None, None)
        };

        hidden_states = T5Block::clamp_hidden_states(hidden_states);

        layer_states = (self_attention_layer_past, cross_attention_layer_past);
        let mut hidden_states = self.ff_layer.forward_t(&hidden_states, train);

        hidden_states = T5Block::clamp_hidden_states(hidden_states);

        LongT5BlockOutput {
            hidden_states,
            self_attention_weights,
            cross_attention_weights,
            self_attention_position_bias,
            cross_attention_position_bias,
            cache: layer_states,
        }
    }
}

pub type LongT5BlockOutput = T5BlockOutput;
