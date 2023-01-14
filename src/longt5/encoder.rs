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

use crate::common::dropout::Dropout;
use crate::common::embeddings::process_ids_embeddings_pair;
use crate::longt5::attention::{
    LayerState, LongT5LayerCrossAttention, LongT5LayerLocalSelfAttention, LongT5LayerSelfAttention,
    LongT5LayerTransientGlobalSelfAttention,
};
use crate::longt5::layer_norm::LongT5LayerNorm;
use crate::longt5::longt5_model::EncoderAttentionType;
use crate::longt5::LongT5Config;
use crate::t5::{T5Block, T5BlockOutput, T5LayerFF, T5StackOutput};
use crate::RustBertError;
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

pub struct LongT5Stack {
    blocks: Vec<LongT5Block>,
    final_layer_norm: LongT5LayerNorm,
    dropout: Dropout,
    output_attentions: bool,
    output_hidden_states: bool,
    is_decoder: bool,
    store_cache: bool,
}

impl LongT5Stack {
    pub fn new<'p, P>(
        p: P,
        config: &LongT5Config,
        is_decoder: bool,
        store_cache: bool,
        output_attentions: bool,
        output_hidden_states: bool,
    ) -> LongT5Stack
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let dropout = Dropout::new(config.dropout_rate);

        let mut blocks: Vec<LongT5Block> = vec![];
        let p_layers = p / "block";
        for layer_index in 0..config.num_layers {
            blocks.push(LongT5Block::new(
                &p_layers / layer_index,
                config,
                layer_index == 0,
                is_decoder,
                store_cache,
                output_attentions,
            ));
        }

        let final_layer_norm = LongT5LayerNorm::new(
            p / "final_layer_norm",
            config.d_model,
            config.layer_norm_epsilon,
        );

        LongT5Stack {
            blocks,
            final_layer_norm,
            dropout,
            output_attentions,
            output_hidden_states,
            is_decoder,
            store_cache,
        }
    }

    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
        encoder_attention_mask: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        head_mask: Option<&Tensor>,
        embeddings: &nn::Embedding,
        old_layer_states: Option<Vec<(Option<LayerState>, Option<LayerState>)>>,
        train: bool,
    ) -> Result<LongT5StackOutput, RustBertError> {
        let (calc_input_embeddings, input_shape, _) =
            process_ids_embeddings_pair(input_ids, input_embeds, embeddings)?;
        let input_embeddings =
            input_embeds.unwrap_or_else(|| calc_input_embeddings.as_ref().unwrap());

        let (batch_size, sequence_length) = (input_shape[0], input_shape[1]);

        let mask_seq_length = if old_layer_states.is_some() {
            if old_layer_states.as_ref().unwrap()[0].0.is_some() {
                old_layer_states.as_ref().unwrap()[0]
                    .0
                    .as_ref()
                    .unwrap()
                    .prev_key
                    .size()[2]
                    + sequence_length
            } else {
                sequence_length
            }
        } else {
            sequence_length
        };

        let calculated_attention_mask = if attention_mask.is_none() {
            Some(Tensor::ones(
                &[batch_size, mask_seq_length],
                (Kind::Int64, input_embeddings.device()),
            ))
        } else {
            None
        };
        let attention_mask =
            attention_mask.unwrap_or_else(|| calculated_attention_mask.as_ref().unwrap());
        //     let extended_attention_mask = match attention_mask.dim() {
        //         3 => attention_mask.unsqueeze(1),
        //         2 => {
        //             if self.is_decoder {
        //                 let seq_ids = Tensor::arange(
        //                     input_shape[1],
        //                     (input_embeddings.kind(), input_embeddings.device()),
        //                 );
        //                 let causal_mask = seq_ids.unsqueeze(0).unsqueeze(0).repeat(&[
        //                     input_shape[0],
        //                     input_shape[1],
        //                     1,
        //                 ]);
        //                 let causal_mask = causal_mask.le_tensor(&seq_ids.unsqueeze(0).unsqueeze(-1));
        //                 causal_mask.unsqueeze(1) * attention_mask.unsqueeze(1).unsqueeze(1)
        //             } else {
        //                 attention_mask.unsqueeze(1).unsqueeze(1)
        //             }
        //         }
        //         _ => {
        //             return Err(RustBertError::ValueError(
        //                 "Invalid attention mask dimension, must be 2 or 3".into(),
        //             ));
        //         }
        //     };
        //
        //     let extended_attention_mask: Option<Tensor> = Some(
        //         ((extended_attention_mask.ones_like() - extended_attention_mask) * -1e4)
        //             .to_kind(input_embeddings.kind()),
        //     );
        //
        //     let extended_encoder_attention_mask = if self.is_decoder & encoder_hidden_states.is_some() {
        //         let encoder_hidden_states = encoder_hidden_states.as_ref().unwrap();
        //         let encoder_hidden_states_shape = encoder_hidden_states.size();
        //         let encoder_mask = match encoder_attention_mask {
        //             Some(value) => value.copy(),
        //             None => Tensor::ones(
        //                 &[
        //                     encoder_hidden_states_shape[0],
        //                     encoder_hidden_states_shape[1],
        //                 ],
        //                 (Kind::Int8, input_embeddings.device()),
        //             ),
        //         };
        //         let encoder_mask = match encoder_mask.dim() {
        //             2 => encoder_mask.unsqueeze(1).unsqueeze(1),
        //             3 => encoder_mask.unsqueeze(1),
        //             _ => {
        //                 return Err(RustBertError::ValueError(
        //                     "Invalid attention mask dimension, must be 2 or 3".into(),
        //                 ));
        //             }
        //         };
        //         Some(
        //             ((encoder_mask.ones_like() - encoder_mask) * -1e4).to_kind(input_embeddings.kind()),
        //         )
        //     } else {
        //         None
        //     };
        //
        //     let mut all_hidden_states: Option<Vec<Tensor>> = if self.output_hidden_states {
        //         Some(Vec::with_capacity(self.blocks.len()))
        //     } else {
        //         None
        //     };
        //     let mut all_attentions: Option<Vec<Tensor>> = if self.output_attentions {
        //         Some(Vec::with_capacity(self.blocks.len()))
        //     } else {
        //         None
        //     };
        //     let mut next_cache: Option<Vec<(Option<LayerState>, Option<LayerState>)>> =
        //         if self.store_cache {
        //             if old_layer_states.is_some() {
        //                 old_layer_states
        //             } else {
        //                 Some(vec![(None, None); self.blocks.len()])
        //             }
        //         } else {
        //             None
        //         };
        //     let mut position_bias = None;
        //     let mut encoder_decoder_position_bias = None;
        //     let mut attention_weights: Option<Tensor>;
        //     let mut hidden_state = input_embeddings.apply_t(&self.dropout, train);
        //
        //     for (layer_idx, layer) in self.blocks.iter().enumerate() {
        //         let layer_state = match &next_cache {
        //             Some(values) => values[layer_idx].to_owned(),
        //             None => (None, None),
        //         };
        //         let block_output = layer.forward_t(
        //             &hidden_state,
        //             position_bias.as_ref(),
        //             extended_attention_mask.as_ref(),
        //             encoder_hidden_states,
        //             extended_encoder_attention_mask.as_ref(),
        //             encoder_decoder_position_bias.as_ref(),
        //             layer_state,
        //             train,
        //         );
        //         if layer_idx == 0 {
        //             position_bias = block_output.self_attention_position_bias;
        //             encoder_decoder_position_bias = block_output.cross_attention_position_bias;
        //         }
        //         hidden_state = block_output.hidden_states;
        //         attention_weights = block_output.cross_attention_weights;
        //         if let Some(hidden_states) = all_hidden_states.borrow_mut() {
        //             hidden_states.push(hidden_state.as_ref().copy().transpose(0, 1));
        //         };
        //         if let Some(attentions) = all_attentions.borrow_mut() {
        //             attentions.push(std::mem::take(&mut attention_weights.unwrap()));
        //         };
        //         if let Some(value) = &mut next_cache {
        //             value[layer_idx] = block_output.cache
        //         };
        //     }
        //
        //     let hidden_state = hidden_state
        //         .apply(&self.final_layer_norm)
        //         .apply_t(&self.dropout, train);
        //
        //     Ok(LongT5StackOutput {
        //         hidden_state,
        //         all_hidden_states,
        //         all_attentions,
        //         next_cache,
        //     })
    }
}

pub type LongT5BlockOutput = T5BlockOutput;
pub type LongT5StackOutput = T5StackOutput;
