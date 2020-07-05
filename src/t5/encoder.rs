// Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
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
use crate::t5::attention::{LayerState, T5LayerCrossAttention, T5LayerSelfAttention};
use crate::t5::T5Config;
use std::borrow::{Borrow, BorrowMut};
use tch::nn::LinearConfig;
use tch::{nn, Kind, Tensor};

pub struct T5DenseReluDense {
    wi: nn::Linear,
    wo: nn::Linear,
    dropout: Dropout,
}

impl T5DenseReluDense {
    pub fn new<'p, P>(p: P, config: &T5Config) -> T5DenseReluDense
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let linear_config = LinearConfig {
            bias: false,
            ..Default::default()
        };
        let wi = nn::linear(p / "wi", config.d_model, config.d_ff, linear_config);
        let wo = nn::linear(p / "wi", config.d_ff, config.d_model, linear_config);
        let dropout = Dropout::new(config.dropout_rate);

        T5DenseReluDense { wi, wo, dropout }
    }

    pub fn forward_t(&self, hidden_states: &Tensor, train: bool) -> Tensor {
        hidden_states
            .apply(&self.wi)
            .relu()
            .apply_t(&self.dropout, train)
            .apply(&self.wo)
    }
}

pub struct T5LayerFF {
    dense_relu_dense: T5DenseReluDense,
    layer_norm: nn::LayerNorm,
    dropout: Dropout,
}

impl T5LayerFF {
    pub fn new<'p, P>(p: P, config: &T5Config) -> T5LayerFF
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let dense_relu_dense = T5DenseReluDense::new(p / "DenseReluDense", config);
        let layer_norm_config = nn::LayerNormConfig {
            eps: config.layer_norm_epsilon,
            ..Default::default()
        };
        let layer_norm = nn::layer_norm(p / "layer_norm", vec![config.d_model], layer_norm_config);
        let dropout = Dropout::new(config.dropout_rate);

        T5LayerFF {
            dense_relu_dense,
            layer_norm,
            dropout,
        }
    }

    pub fn forward_t(&self, hidden_states: &Tensor, train: bool) -> Tensor {
        let y = &self
            .dense_relu_dense
            .forward_t(&hidden_states.apply(&self.layer_norm), train);

        hidden_states + y.apply_t(&self.dropout, train)
    }
}

pub struct T5Block {
    self_attention: T5LayerSelfAttention,
    cross_attention: Option<T5LayerCrossAttention>,
    ff_layer: T5LayerFF,
}

impl T5Block {
    pub fn new<'p, P>(
        p: P,
        config: &T5Config,
        has_relative_attention_bias: bool,
        is_decoder: bool,
        store_cache: bool,
        output_attentions: bool,
    ) -> T5Block
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow() / "layer";
        let mut module_index = 0;

        let self_attention = T5LayerSelfAttention::new(
            &p / module_index,
            config,
            is_decoder,
            store_cache,
            output_attentions,
            has_relative_attention_bias,
        );

        let cross_attention = if is_decoder {
            module_index += 1;
            Some(T5LayerCrossAttention::new(
                &p / module_index,
                config,
                is_decoder,
                store_cache,
                output_attentions,
                has_relative_attention_bias,
            ))
        } else {
            None
        };
        module_index += 1;

        let ff_layer = T5LayerFF::new(&p / module_index, config);

        T5Block {
            self_attention,
            cross_attention,
            ff_layer,
        }
    }

    pub fn forward_t(
        &self,
        hidden_states: &Tensor,
        position_bias: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
        encoder_attention_mask: Option<&Tensor>,
        encoder_decoder_position_bias: Option<&Tensor>,
        mut layer_states: (Option<LayerState>, Option<LayerState>),
        train: bool,
    ) -> (
        Tensor,
        (Option<Tensor>, Option<Tensor>),
        (Option<Tensor>, Option<Tensor>),
        (Option<LayerState>, Option<LayerState>),
    ) {
        let (
            hidden_states,
            self_attention_weights,
            self_attention_position_bias,
            self_attention_layer_past,
        ) = self.self_attention.forward_t(
            hidden_states,
            position_bias,
            attention_mask,
            layer_states.0,
            train,
        );

        let (
            hidden_states,
            cross_attention_weights,
            cross_attention_position_bias,
            cross_attention_layer_past,
        ) = if self.cross_attention.is_some() & encoder_hidden_states.is_some() {
            let query_length = match &self_attention_layer_past {
                Some(value) => Some(value.prev_key.size()[2]),
                None => None,
            };
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

        let attention_weights = (self_attention_weights, cross_attention_weights);
        let position_bias = (self_attention_position_bias, cross_attention_position_bias);
        layer_states = (self_attention_layer_past, cross_attention_layer_past);
        let hidden_states = self.ff_layer.forward_t(&hidden_states, train);

        (
            hidden_states,
            attention_weights,
            position_bias,
            layer_states,
        )
    }
}

pub struct T5Stack {
    blocks: Vec<T5Block>,
    final_layer_norm: nn::LayerNorm,
    dropout: Dropout,
    output_attentions: bool,
    output_hidden_states: bool,
    is_decoder: bool,
    store_cache: bool,
}

impl T5Stack {
    pub fn new<'p, P>(
        p: P,
        config: &T5Config,
        is_decoder: bool,
        store_cache: bool,
        output_attentions: bool,
        output_hidden_states: bool,
    ) -> T5Stack
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let dropout = Dropout::new(config.dropout_rate);

        let mut blocks: Vec<T5Block> = vec![];
        let p_layers = p / "block";
        for layer_index in 0..config.num_layers {
            blocks.push(T5Block::new(
                &p_layers / layer_index,
                config,
                layer_index == 0,
                is_decoder,
                store_cache,
                output_attentions,
            ));
        }

        let layer_norm_config = nn::LayerNormConfig {
            eps: config.layer_norm_epsilon,
            ..Default::default()
        };

        let final_layer_norm = nn::layer_norm(
            p / "final_layer_norm",
            vec![config.d_model],
            layer_norm_config,
        );

        T5Stack {
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
        input_embeds: Option<Tensor>,
        embeddings: &nn::Embedding,
        old_layer_states: Option<Vec<(Option<LayerState>, Option<LayerState>)>>,
        train: bool,
    ) -> Result<
        (
            Tensor,
            Option<Vec<Tensor>>,
            Option<Vec<Tensor>>,
            Option<Vec<(Option<LayerState>, Option<LayerState>)>>,
        ),
        &'static str,
    > {
        let (input_embeddings, input_shape) = match input_ids {
            Some(input_ids_value) => match input_embeds {
                Some(_) => {
                    return Err("Only one of input ids or input embeddings may be set");
                }
                None => (input_ids_value.apply(embeddings), input_ids_value.size()),
            },
            None => match input_embeds {
                Some(embeds) => {
                    let size = vec![embeds.size()[0], embeds.size()[1]];
                    (embeds, size)
                }
                None => {
                    return Err("Only one of input ids or input embeddings may be set");
                }
            },
        };

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
        let attention_mask = match attention_mask {
            Some(value) => value,
            None => &calculated_attention_mask.as_ref().unwrap(),
        };

        let extended_attention_mask = match attention_mask.dim() {
            3 => attention_mask.unsqueeze(1),
            2 => {
                if self.is_decoder {
                    let seq_ids =
                        Tensor::arange(input_shape[1], (Kind::Float, input_embeddings.device()));
                    let causal_mask = seq_ids.unsqueeze(0).unsqueeze(0).repeat(&vec![
                        input_shape[0],
                        input_shape[1],
                        1,
                    ]);
                    let causal_mask = causal_mask.le1(&seq_ids.unsqueeze(0).unsqueeze(-1));
                    causal_mask * attention_mask.unsqueeze(1).unsqueeze(1)
                } else {
                    attention_mask.unsqueeze(1).unsqueeze(1)
                }
            }
            _ => {
                return Err("Invalid attention mask dimension, must be 2 or 3");
            }
        };

        let extended_attention_mask: Option<Tensor> =
            Some((extended_attention_mask.ones_like() - extended_attention_mask) * -10000.0);

        let extended_encoder_attention_mask = if self.is_decoder & encoder_hidden_states.is_some() {
            let encoder_hidden_states = encoder_hidden_states.as_ref().unwrap();
            let encoder_hidden_states_shape = encoder_hidden_states.size();
            let encoder_mask = match encoder_attention_mask {
                Some(value) => value.copy(),
                None => Tensor::ones(
                    &[
                        encoder_hidden_states_shape[0],
                        encoder_hidden_states_shape[1],
                    ],
                    (Kind::Int64, input_embeddings.device()),
                ),
            };
            let encoder_mask = match encoder_mask.dim() {
                2 => encoder_mask.unsqueeze(1).unsqueeze(1),
                3 => encoder_mask.unsqueeze(1),
                _ => {
                    return Err("Invalid encoder attention mask dimension, must be 2 or 3");
                }
            };
            Some((encoder_mask.ones_like() - encoder_mask) * -1e9)
        } else {
            None
        };

        let mut all_hidden_states: Option<Vec<Tensor>> = if self.output_hidden_states {
            Some(Vec::with_capacity(self.blocks.len()))
        } else {
            None
        };
        let mut all_attentions: Option<Vec<Tensor>> = if self.output_attentions {
            Some(Vec::with_capacity(self.blocks.len()))
        } else {
            None
        };
        let mut next_decoder_cache: Option<Vec<(Option<LayerState>, Option<LayerState>)>> =
            if self.store_cache {
                if old_layer_states.is_some() {
                    old_layer_states
                } else {
                    Some(vec![(None, None); self.blocks.len()])
                }
            } else {
                None
            };
        let mut position_bias = None;
        let mut encoder_decoder_position_bias = None;
        let mut attention_weights: Option<Tensor>;
        let mut hidden_state = input_embeddings.apply_t(&self.dropout, train);
        let mut blocks = self.blocks.iter().enumerate();

        loop {
            match blocks.next() {
                Some((layer_idx, layer)) => {
                    let layer_state = match &next_decoder_cache {
                        Some(values) => values[layer_idx].to_owned(),
                        None => (None, None),
                    };
                    let temp = layer.forward_t(
                        &hidden_state,
                        position_bias.as_ref(),
                        extended_attention_mask.as_ref(),
                        encoder_hidden_states,
                        extended_encoder_attention_mask.as_ref(),
                        encoder_decoder_position_bias.as_ref(),
                        layer_state,
                        train,
                    );
                    if layer_idx == 0 {
                        position_bias = (temp.2).0;
                        encoder_decoder_position_bias = (temp.2).1;
                    }
                    hidden_state = temp.0;
                    attention_weights = (temp.1).1;
                    if let Some(hidden_states) = all_hidden_states.borrow_mut() {
                        hidden_states.push(hidden_state.as_ref().copy().transpose(0, 1));
                    };
                    if let Some(attentions) = all_attentions.borrow_mut() {
                        attentions.push(attention_weights.as_ref().unwrap().copy());
                    };
                    if let Some(value) = &mut next_decoder_cache {
                        value[layer_idx] = temp.3
                    };
                }
                None => break,
            };
        }

        let hidden_state = hidden_state
            .apply(&self.final_layer_norm)
            .apply_t(&self.dropout, train);

        Ok((
            hidden_state,
            all_hidden_states,
            all_attentions,
            next_decoder_cache,
        ))
    }
}
