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

use crate::common::activations::TensorFunction;
use crate::common::dropout::Dropout;
use crate::prophetnet::ProphetNetConfig;
use crate::RustBertError;
use std::borrow::Borrow;
use tch::nn::ModuleT;
use tch::{nn, Kind, Tensor};

#[derive(Debug)]
/// # Cache for ProphetNet attention layers
/// Stores the cached value of key and value
pub struct LayerState {
    /// Cached keys
    pub prev_key: Tensor,
    /// Cached values
    pub prev_value: Tensor,
}

impl Clone for LayerState {
    fn clone(&self) -> Self {
        LayerState {
            prev_key: self.prev_key.copy(),
            prev_value: self.prev_value.copy(),
        }
    }
}

impl LayerState {
    pub(crate) fn reorder_cache(&mut self, new_indices: &Tensor) {
        self.prev_key = self.prev_key.index_select(0, new_indices);
        self.prev_value = self.prev_value.index_select(0, new_indices);
    }
}

pub struct ProphetNetAttention {
    key_proj: nn::Linear,
    value_proj: nn::Linear,
    query_proj: nn::Linear,
    out_proj: nn::Linear,
    dropout: Dropout,
    attention_dropout: Dropout,
    num_attention_heads: i64,
    head_dim: i64,
    output_attentions: bool,
}

impl ProphetNetAttention {
    pub fn new<'p, P>(
        p: P,
        config: &ProphetNetConfig,
        num_attention_heads: i64,
    ) -> Result<ProphetNetAttention, RustBertError>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let dropout = Dropout::new(config.dropout);
        let attention_dropout = Dropout::new(config.attention_dropout);

        if config.hidden_size % num_attention_heads != 0 {
            return Err(RustBertError::InvalidConfigurationError(format!(
                "Invalid number of heads for self attention, {} not a multiple of {}",
                config.hidden_size, num_attention_heads
            )));
        }

        let head_dim = config.hidden_size / num_attention_heads;

        let key_proj = nn::linear(
            p / "key_proj",
            config.hidden_size,
            config.hidden_size,
            Default::default(),
        );

        let value_proj = nn::linear(
            p / "value_proj",
            config.hidden_size,
            config.hidden_size,
            Default::default(),
        );

        let query_proj = nn::linear(
            p / "query_proj",
            config.hidden_size,
            config.hidden_size,
            Default::default(),
        );

        let out_proj = nn::linear(
            p / "out_proj",
            config.hidden_size,
            config.hidden_size,
            Default::default(),
        );

        let output_attentions = config.output_attentions.unwrap_or(false);

        Ok(ProphetNetAttention {
            key_proj,
            value_proj,
            query_proj,
            out_proj,
            dropout,
            attention_dropout,
            num_attention_heads,
            head_dim,
            output_attentions,
        })
    }

    fn flatten(&self, x: Tensor, dim_0: i64, bs: i64) -> Tensor {
        x.contiguous()
            .view((dim_0, bs * self.num_attention_heads, self.head_dim))
            .transpose(0, 1)
    }

    pub fn forward_t(
        &self,
        hidden_states: &Tensor,
        key_value_states: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        mut layer_state: Option<LayerState>,
        train: bool,
    ) -> (Tensor, Option<Tensor>, Option<LayerState>) {
        let hidden_states_size = hidden_states.size();
        let (sequence_length, batch_size, hidden_size) = (
            hidden_states_size[0],
            hidden_states_size[1],
            hidden_states_size[2],
        );
        let is_cross_attention = key_value_states.is_some();
        let query_states = hidden_states.apply(&self.query_proj) / (self.head_dim as f64).sqrt();
        let query_states = self.flatten(query_states, sequence_length, batch_size);

        let (key_states, value_states) = if !is_cross_attention {
            let key_states = self.flatten(hidden_states.apply(&self.key_proj), -1, batch_size);
            let value_states = self.flatten(hidden_states.apply(&self.value_proj), -1, batch_size);
            (key_states, value_states)
        } else if layer_state.is_none() {
            let key_states = self.flatten(
                key_value_states.unwrap().apply(&self.key_proj),
                -1,
                batch_size,
            );
            let value_states = self.flatten(
                key_value_states.unwrap().apply(&self.value_proj),
                -1,
                batch_size,
            );
            (key_states, value_states)
        } else {
            let past_state = layer_state.as_ref().unwrap();
            (
                past_state.prev_key.view([
                    batch_size * self.num_attention_heads,
                    -1,
                    self.head_dim,
                ]),
                past_state.prev_value.view([
                    batch_size * self.num_attention_heads,
                    -1,
                    self.head_dim,
                ]),
            )
        };

        if is_cross_attention {
            if layer_state.is_some() {
                layer_state.as_mut().unwrap().prev_key =
                    key_states.view([batch_size, self.num_attention_heads, -1, self.head_dim]);
                layer_state.as_mut().unwrap().prev_value =
                    value_states.view([batch_size, self.num_attention_heads, -1, self.head_dim]);
            } else {
                layer_state = Some(LayerState {
                    prev_key: key_states.view([
                        batch_size,
                        self.num_attention_heads,
                        -1,
                        self.head_dim,
                    ]),
                    prev_value: value_states.view([
                        batch_size,
                        self.num_attention_heads,
                        -1,
                        self.head_dim,
                    ]),
                })
            }
        };

        let key_sequence_key = key_states.size()[1];
        let mut attention_weights = query_states.bmm(&key_states.transpose(1, 2));

        if let Some(attention_mask) = attention_mask {
            attention_weights = attention_weights + attention_mask;
        };

        let attention_weights_reshaped = attention_weights.view([
            batch_size,
            self.num_attention_heads,
            sequence_length,
            key_sequence_key,
        ]);

        let attention_probs = attention_weights_reshaped
            .view([
                batch_size * self.num_attention_heads,
                sequence_length,
                key_sequence_key,
            ])
            .softmax(-1, attention_weights_reshaped.kind())
            .apply_t(&self.attention_dropout, train);

        let attention_output = attention_probs
            .bmm(&value_states)
            .transpose(0, 1)
            .contiguous()
            .view([sequence_length, batch_size, hidden_size])
            .apply(&self.out_proj)
            .apply_t(&self.dropout, train);

        let attention_weights = if self.output_attentions {
            Some(attention_weights_reshaped)
        } else {
            None
        };
        (attention_output, attention_weights, layer_state)
    }
}

#[derive(Debug)]
pub struct ProphetNetFeedForward {
    activation_function: TensorFunction,
    intermediate: nn::Linear,
    output: nn::Linear,
    activation_dropout: Dropout,
    dropout: Dropout,
}

impl ProphetNetFeedForward {
    pub fn new<'p, P>(p: P, config: &ProphetNetConfig, ffn_dim: i64) -> ProphetNetFeedForward
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let activation_function = config.activation_function.get_function();
        let intermediate = nn::linear(
            p / "intermediate",
            config.hidden_size,
            ffn_dim,
            Default::default(),
        );
        let output = nn::linear(
            p / "output",
            ffn_dim,
            config.hidden_size,
            Default::default(),
        );
        let activation_dropout = Dropout::new(config.activation_dropout);
        let dropout = Dropout::new(config.dropout);
        ProphetNetFeedForward {
            activation_function,
            intermediate,
            output,
            activation_dropout,
            dropout,
        }
    }
}

impl ModuleT for ProphetNetFeedForward {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let hidden_states = (self.activation_function.get_fn())(&xs.apply(&self.intermediate));
        hidden_states
            .apply_t(&self.activation_dropout, train)
            .apply(&self.output)
            .apply_t(&self.dropout, train)
    }
}

pub struct ProphetNetNgramAttention {
    num_buckets: i64,
    ngram: i64,
    relative_max_distance: i64,
    num_attention_heads: i64,
    dropout: Dropout,
    attention_dropout: Dropout,
    head_dim: i64,
    key_proj: nn::Linear,
    value_proj: nn::Linear,
    query_proj: nn::Linear,
    out_proj: nn::Linear,
    relative_pos_embeddings: nn::Linear,
    output_attentions: bool,
}

impl ProphetNetNgramAttention {
    pub fn new<'p, P>(p: P, config: &ProphetNetConfig) -> ProphetNetNgramAttention
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let num_buckets = config.num_buckets;
        let ngram = config.ngram;
        let relative_max_distance = config.relative_max_distance;
        let num_attention_heads = config.num_decoder_attention_heads;
        let dropout = Dropout::new(config.dropout);
        let attention_dropout = Dropout::new(config.attention_dropout);
        let head_dim = config.hidden_size / num_attention_heads;

        let key_proj = nn::linear(
            p / "key_proj",
            config.hidden_size,
            config.hidden_size,
            Default::default(),
        );

        let value_proj = nn::linear(
            p / "value_proj",
            config.hidden_size,
            config.hidden_size,
            Default::default(),
        );

        let query_proj = nn::linear(
            p / "query_proj",
            config.hidden_size,
            config.hidden_size,
            Default::default(),
        );

        let out_proj = nn::linear(
            p / "out_proj",
            config.hidden_size,
            config.hidden_size,
            Default::default(),
        );

        let relative_pos_embeddings = nn::linear(
            p / "relative_pos_embeddings",
            config.hidden_size,
            num_buckets * num_attention_heads,
            Default::default(),
        );

        let output_attentions = config.output_attentions.unwrap_or(false);

        ProphetNetNgramAttention {
            num_buckets,
            ngram,
            relative_max_distance,
            num_attention_heads,
            dropout,
            attention_dropout,
            head_dim,
            key_proj,
            value_proj,
            query_proj,
            out_proj,
            relative_pos_embeddings,
            output_attentions,
        }
    }

    fn flatten<T>(&self, x: T, dim_0: i64, bs: i64) -> Tensor
    where
        T: Borrow<Tensor>,
    {
        x.borrow()
            .contiguous()
            .view((dim_0, bs * self.num_attention_heads, self.head_dim))
            .transpose(0, 1)
    }

    pub fn forward_t(
        &self,
        hidden_states: &Tensor,
        mut layer_state: Option<LayerState>,
        attention_mask: Option<&Tensor>,
        extended_predict_attention_mask: Option<&Tensor>,
        main_relative_position_buckets: Option<&Tensor>,
        predict_relative_position_buckets: Option<&Tensor>,
        position_ids: &Tensor,
        train: bool,
    ) -> (Tensor, Option<Tensor>, Option<Tensor>, Option<LayerState>) {
        let hidden_states_size = hidden_states.size();
        let (sequence_length, batch_size, hidden_size) = (
            hidden_states_size[0],
            hidden_states_size[1],
            hidden_states_size[2],
        );

        let query_states = hidden_states.apply(&self.query_proj) / (self.head_dim as f64).sqrt();
        let key_states = hidden_states.apply(&self.key_proj);
        let value_states = hidden_states.apply(&self.value_proj);

        let mut main_hidden_states = hidden_states.chunk(1 + self.ngram, 0);
        let mut main_query_states = self
            .flatten(query_states, sequence_length, batch_size)
            .chunk(1 + self.ngram, 1);
        let mut main_key_states = self
            .flatten(key_states, -1, batch_size)
            .chunk(1 + self.ngram, 1);
        let mut main_value_states = self
            .flatten(value_states, -1, batch_size)
            .chunk(1 + self.ngram, 1);

        let hidden_states_predict_list = main_hidden_states.split_off(1);
        let predict_query_states_list = main_query_states.split_off(1);
        let predict_key_states_list = main_key_states.split_off(1);
        let predict_value_states_list = main_value_states.split_off(1);

        let main_hidden_states = main_hidden_states.pop().unwrap();
        let main_query_states = main_query_states.pop().unwrap();
        let mut main_key_states = main_key_states.pop().unwrap();
        let mut main_value_states = main_value_states.pop().unwrap();

        if let Some(layer_state_value) = &layer_state {
            let prev_main_key_states = layer_state_value.prev_key.view([
                batch_size * self.num_attention_heads,
                -1,
                self.head_dim,
            ]);
            let prev_main_value_states = layer_state_value.prev_value.view([
                batch_size * self.num_attention_heads,
                -1,
                self.head_dim,
            ]);
            main_key_states = Tensor::cat(&[prev_main_key_states, main_key_states], 1);
            main_value_states = Tensor::cat(&[prev_main_value_states, main_value_states], 1);
        };

        if layer_state.is_some() {
            layer_state.as_mut().unwrap().prev_key =
                main_key_states.view([batch_size, self.num_attention_heads, -1, self.head_dim]);
            layer_state.as_mut().unwrap().prev_value =
                main_value_states.view([batch_size, self.num_attention_heads, -1, self.head_dim]);
        } else {
            layer_state = Some(LayerState {
                prev_key: main_key_states.view([
                    batch_size,
                    self.num_attention_heads,
                    -1,
                    self.head_dim,
                ]),
                prev_value: main_value_states.view([
                    batch_size,
                    self.num_attention_heads,
                    -1,
                    self.head_dim,
                ]),
            })
        };
        let main_sequence_length = sequence_length / (1 + self.ngram);

        let main_attention_weights = main_query_states.bmm(&main_key_states.transpose(1, 2));

        let main_relative_pos_embeddings = self.get_main_relative_position_embeddings(
            &main_hidden_states,
            &main_attention_weights,
            position_ids,
            main_relative_position_buckets,
        );
        let mut main_attention_weights = main_attention_weights + main_relative_pos_embeddings;
        if let Some(attention_mask_value) = attention_mask {
            main_attention_weights = main_attention_weights + attention_mask_value;
        };

        let main_attention_probas = main_attention_weights
            .softmax(-1, main_attention_weights.kind())
            .apply_t(&self.attention_dropout, train);

        let main_attention_output = main_attention_probas
            .bmm(&main_value_states)
            .transpose(0, 1)
            .contiguous()
            .view([-1, main_sequence_length, batch_size, hidden_size])
            .apply(&self.out_proj);

        let predict_hidden_states = Tensor::cat(hidden_states_predict_list.as_slice(), 0).view([
            self.ngram,
            main_sequence_length,
            batch_size,
            hidden_size,
        ]);

        let predict_query_states = Tensor::cat(predict_query_states_list.as_slice(), 0).view([
            self.ngram,
            -1,
            main_sequence_length,
            self.head_dim,
        ]);

        let predict_key_states = Tensor::cat(
            predict_key_states_list
                .iter()
                .map(|predict_key_state| {
                    Tensor::cat(&[&main_key_states, predict_key_state], 1).unsqueeze(0)
                })
                .collect::<Vec<Tensor>>()
                .as_slice(),
            0,
        );

        let predict_value_states = Tensor::cat(
            predict_value_states_list
                .iter()
                .map(|predict_value_state| {
                    Tensor::cat(&[&main_value_states, predict_value_state], 1).unsqueeze(0)
                })
                .collect::<Vec<Tensor>>()
                .as_slice(),
            0,
        );

        let predict_attention_weights = Tensor::einsum(
            "nbtc,nbsc->nbts",
            &[predict_query_states, predict_key_states],
        );

        let predict_relative_pos_embeddings = self.get_predict_relative_pos_embeddings(
            &predict_hidden_states,
            &predict_attention_weights,
            position_ids,
            predict_relative_position_buckets,
        );

        let mut predict_attention_weights =
            predict_attention_weights + predict_relative_pos_embeddings;
        if let Some(extended_predict_attention_mask_value) = extended_predict_attention_mask {
            predict_attention_weights =
                predict_attention_weights + extended_predict_attention_mask_value;
        };

        let predict_attention_probas = predict_attention_weights
            .softmax(-1, predict_attention_weights.kind())
            .apply_t(&self.attention_dropout, train);

        let predict_attention_output = Tensor::einsum(
            "nbts,nbsc->nbtc",
            &[&predict_attention_probas, &predict_value_states],
        )
        .transpose(1, 2)
        .contiguous()
        .view([self.ngram, main_sequence_length, batch_size, hidden_size])
        .apply(&self.out_proj);

        let attention_output = Tensor::cat(&[main_attention_output, predict_attention_output], 0)
            .view([-1, batch_size, hidden_size])
            .apply_t(&self.dropout, train);

        let (main_attention_probas, predict_attention_probas) = if self.output_attentions {
            let main_attention_probas = main_attention_probas.view([
                batch_size,
                self.num_attention_heads,
                main_sequence_length,
                -1,
            ]);
            let predict_attention_probas = predict_attention_probas
                .view([
                    self.ngram,
                    batch_size,
                    self.num_attention_heads,
                    main_sequence_length,
                    -1,
                ])
                .transpose(0, 1);
            (Some(main_attention_probas), Some(predict_attention_probas))
        } else {
            (None, None)
        };
        (
            attention_output,
            main_attention_probas,
            predict_attention_probas,
            layer_state,
        )
    }

    fn get_main_relative_position_embeddings(
        &self,
        hidden_states: &Tensor,
        attention_weights: &Tensor,
        position_ids: &Tensor,
        main_relative_position_buckets: Option<&Tensor>,
    ) -> Tensor {
        let hidden_states_size = hidden_states.size();
        let (sequence_length, batch_size) = (hidden_states_size[0], hidden_states_size[1]);
        let calc_main_relative_position_buckets = if main_relative_position_buckets.is_none() {
            let relative_positions = Tensor::arange_start(
                1,
                attention_weights.size().last().unwrap() + 1,
                (Kind::Int64, hidden_states.device()),
            )
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(&[batch_size, sequence_length, 1]);
            let relative_positions = relative_positions
                - position_ids
                    .unsqueeze(0)
                    .repeat(&[batch_size, sequence_length, 1]);
            Some(compute_relative_buckets(
                self.num_buckets,
                self.relative_max_distance,
                &relative_positions,
                false,
            ))
        } else {
            None
        };
        let main_relative_position_buckets = main_relative_position_buckets
            .unwrap_or_else(|| calc_main_relative_position_buckets.as_ref().unwrap());

        let rel_pos_embeddings = hidden_states
            .transpose(0, 1)
            .apply(&self.relative_pos_embeddings)
            .view([
                batch_size,
                sequence_length,
                self.num_buckets,
                self.num_attention_heads,
            ])
            .permute(&[0, 3, 1, 2])
            .reshape(&[-1, self.num_buckets]);

        let main_relative_position_buckets = main_relative_position_buckets
            .repeat(&[1, self.num_attention_heads, 1])
            .view([-1, *main_relative_position_buckets.size().last().unwrap()]);

        let mut new_shape = attention_weights
            .size()
            .into_iter()
            .take(2)
            .collect::<Vec<i64>>();
        new_shape.push(-1);
        rel_pos_embeddings
            .gather(1, &main_relative_position_buckets, false)
            .view(new_shape.as_slice())
    }

    fn get_predict_relative_pos_embeddings(
        &self,
        hidden_states: &Tensor,
        attention_weights: &Tensor,
        position_ids: &Tensor,
        predict_relative_position_buckets: Option<&Tensor>,
    ) -> Tensor {
        let hidden_states_size = hidden_states.size();
        let (sequence_length, batch_size) = (hidden_states_size[1], hidden_states_size[2]);

        let calc_predict_relative_position_buckets = if predict_relative_position_buckets.is_none()
        {
            let key_sequence_length = *attention_weights.size().last().unwrap();
            let relative_positions =
                Tensor::arange(key_sequence_length, (Kind::Int64, hidden_states.device()))
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .repeat(&[batch_size, sequence_length, 1]);
            let relative_positions = relative_positions
                - position_ids
                    .unsqueeze(0)
                    .repeat(&[batch_size, sequence_length, 1]);
            Some(compute_relative_buckets(
                self.num_buckets,
                self.relative_max_distance,
                &relative_positions,
                false,
            ))
        } else {
            None
        };

        let predict_relative_position_buckets = predict_relative_position_buckets
            .unwrap_or_else(|| calc_predict_relative_position_buckets.as_ref().unwrap());

        let rel_pos_embeddings = hidden_states
            .transpose(0, 1)
            .apply(&self.relative_pos_embeddings)
            .view([
                self.ngram,
                batch_size,
                sequence_length,
                self.num_buckets,
                self.num_attention_heads,
            ])
            .permute(&[0, 1, 4, 2, 3])
            .reshape(&[-1, self.num_buckets]);

        let predict_relative_position_buckets = predict_relative_position_buckets
            .unsqueeze(0)
            .repeat(&[self.ngram, 1, self.num_attention_heads, 1])
            .view([
                -1,
                *predict_relative_position_buckets.size().last().unwrap(),
            ]);

        rel_pos_embeddings
            .gather(1, &predict_relative_position_buckets, true)
            .view([
                self.ngram,
                batch_size * self.num_attention_heads,
                sequence_length,
                -1,
            ])
    }
}

pub(crate) fn compute_relative_buckets(
    num_buckets: i64,
    max_distance: i64,
    relative_positions: &Tensor,
    bidirectional: bool,
) -> Tensor {
    let inverse_relative_positions = -relative_positions;

    let (num_buckets, relative_positions_bucket, inverse_relative_positions) = if bidirectional {
        let num_buckets = num_buckets / 2;
        let relative_position_bucket =
            inverse_relative_positions.lt(0).totype(Kind::Int) * num_buckets;
        let inverse_relative_position = inverse_relative_positions.abs();
        (
            num_buckets,
            relative_position_bucket,
            inverse_relative_position,
        )
    } else {
        (
            num_buckets,
            relative_positions.zeros_like(),
            inverse_relative_positions.max_other(&inverse_relative_positions.zeros_like()),
        )
    };
    let max_exact = num_buckets / 2;
    let is_small = inverse_relative_positions.lt(max_exact);
    let max_exact_f64 = max_exact as f64;
    let val_if_large = (inverse_relative_positions.totype(Kind::Float) / max_exact_f64).log2()
        / (max_distance as f64 / max_exact_f64).log2()
        * (num_buckets as f64 - max_exact_f64)
        + max_exact_f64;

    let val_if_large = val_if_large
        .min_other(&(val_if_large.ones_like() * (num_buckets as f64 - 1.0)))
        .totype(Kind::Int64);

    relative_positions_bucket + inverse_relative_positions.where_self(&is_small, &val_if_large)
}

pub(crate) fn compute_all_stream_relative_buckets(
    num_buckets: i64,
    max_distance: i64,
    position_ids: &Tensor,
) -> (Tensor, Tensor) {
    let main_stream_relative_positions =
        position_ids
            .unsqueeze(1)
            .repeat(&[1, *position_ids.size().last().unwrap(), 1])
            - position_ids.unsqueeze(-1);

    let predicting_stream_relative_positions = Tensor::cat(&[&(position_ids - 1), position_ids], 1)
        .unsqueeze(1)
        .repeat(&[1, *position_ids.size().last().unwrap(), 1])
        - position_ids.unsqueeze(-1);

    let main_relative_position_buckets = compute_relative_buckets(
        num_buckets,
        max_distance,
        &main_stream_relative_positions,
        false,
    );

    let predict_relative_position_buckets = compute_relative_buckets(
        num_buckets,
        max_distance,
        &predicting_stream_relative_positions,
        false,
    );

    (
        main_relative_position_buckets,
        predict_relative_position_buckets,
    )
}
