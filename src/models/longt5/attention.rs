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
use crate::longt5::layer_norm::LongT5LayerNorm;
use crate::longt5::LongT5Config;
use crate::t5::{
    get_relative_position_bucket, LayerState as T5layerState, T5Attention, T5LayerCrossAttention,
};
use std::borrow::Borrow;
use tch::nn::LinearConfig;
use tch::{nn, Device, IndexOp, Kind, Tensor};

pub type LongT5Attention = T5Attention;
pub type LongT5LayerCrossAttention = T5LayerCrossAttention;
pub type LayerState = T5layerState;

fn pad_to_multiple(x: &Tensor, block_length: i64, dim: usize, pad_value: f64) -> Tensor {
    let mut x_size = x.size();
    let pad_length = (-x_size[dim]).rem_euclid(block_length);

    if x_size.iter().any(|&el| el == 0) {
        x_size[dim] += pad_length;
        Tensor::zeros(x_size.as_slice(), (x.kind(), x.device()))
    } else {
        let mut pad = vec![0i64; 2 * x.dim()];
        pad[2 * dim] = pad_length;
        pad.reverse();
        x.pad(pad.as_slice(), "constant", pad_value)
    }
}

fn split_into_blocks(x: &Tensor, block_length: i64, dim: usize) -> Tensor {
    let x_size = x.size();
    let padded_x = if x_size[dim] % block_length != 0 {
        Some(pad_to_multiple(x, block_length, dim, 0f64))
    } else {
        None
    };
    let x = padded_x.as_ref().unwrap_or(x);
    let mut x_size = x.size();
    let num_blocks = x_size[dim] / block_length;
    x_size.remove(dim);
    x_size.insert(dim, block_length);
    x_size.insert(dim, num_blocks);
    if x_size.iter().any(|&el| el == 0) {
        Tensor::empty(x_size.as_slice(), (x.kind(), x.device()))
    } else {
        x.reshape(x_size.as_slice())
    }
}

fn concatenate_3_blocks(
    x: &Tensor,
    block_dim: usize,
    sequence_dim: i64,
    pad_value: Option<f64>,
) -> Tensor {
    let x_size = x.size();
    let num_blocks = x_size[block_dim];
    let mut pad = vec![0i64; 2 * x.dim()];
    pad[2 * block_dim] = 1;
    pad[2 * block_dim + 1] = 1;
    pad.reverse();
    let x = x.pad(pad.as_slice(), "constant", pad_value.unwrap_or(0f64));
    let mut block_list: Vec<Tensor> = Vec::with_capacity(3);
    for i in 0..3 {
        block_list.push(x.narrow(block_dim as i64, i, num_blocks));
    }
    Tensor::cat(block_list.as_slice(), sequence_dim)
}

fn make_3blocks_relative_position_ids(block_length: i64, device: Device) -> Tensor {
    let position_ids = Tensor::arange(3 * block_length, (Kind::Int, device));
    let center_position_ids = position_ids.i(block_length..2 * block_length);
    position_ids.unsqueeze(0) - center_position_ids.unsqueeze(1)
}

fn mask_local_attention_mask(local_attention_mask: &Tensor, block_length: i64) -> Tensor {
    let relative_position_ids =
        make_3blocks_relative_position_ids(block_length, local_attention_mask.device());
    let locality_mask = relative_position_ids
        .abs()
        .lt(block_length)
        .unsqueeze(0)
        .unsqueeze(0);
    local_attention_mask.logical_and(&locality_mask)
}

pub(crate) fn get_local_attention_mask(attention_mask: &Tensor, block_length: i64) -> Tensor {
    let blocked_attention_mask = split_into_blocks(attention_mask, block_length, 1);
    let three_blocked_attention_mask = concatenate_3_blocks(&blocked_attention_mask, 1, 2, None);

    let blocked_attention_mask = blocked_attention_mask.unsqueeze(-1);
    let three_blocked_attention_mask = three_blocked_attention_mask.unsqueeze(-2);

    let local_attention_mask = mask_local_attention_mask(
        &blocked_attention_mask.logical_and(&three_blocked_attention_mask),
        block_length,
    );
    local_attention_mask.unsqueeze(1)
}

fn make_global_fixed_block_ids(
    attention_mask: &Tensor,
    global_block_size: i64,
) -> (Tensor, Tensor) {
    let &[batch_size, seq_length, ..] = attention_mask.size().as_slice() else {
        unreachable!()
    };

    let handle_orphan_tokens = |block_ids: Tensor| -> Tensor {
        let block_ends = Tensor::arange(seq_length, (Kind::Int64, block_ids.device()))
            .remainder(global_block_size)
            .eq(global_block_size - 1);
        let true_block_ends = block_ends.logical_and(&block_ids.ge(0));
        let full_blocks = true_block_ends
            .sum_dim_intlist([-1].as_slice(), false, block_ids.kind())
            .unsqueeze(-1)
            - 1;
        block_ids.where_self(&block_ids.lt_tensor(&full_blocks), &full_blocks)
    };

    let fixed_block_mask = attention_mask.ones_like() / global_block_size;
    let fixed_block_mask = fixed_block_mask.cumsum(1, fixed_block_mask.kind()) - fixed_block_mask;
    let mask = attention_mask
        .ones_like()
        .where_scalarother(&attention_mask.not_equal(0.0), -1000.0);

    let mut global_block_ids = (mask + fixed_block_mask - 1.0).floor();
    global_block_ids = global_block_ids.where_scalarother(&global_block_ids.gt(-1.0), -1.0);
    global_block_ids = global_block_ids * attention_mask + attention_mask - 1;
    global_block_ids = handle_orphan_tokens(global_block_ids);
    let num_globals = seq_length / global_block_size;
    let sequence_block_ids_max = if num_globals > 0 {
        global_block_ids
            .max_dim(-1, false)
            .0
            .repeat([num_globals, 1])
            .transpose(0, 1)
    } else {
        Tensor::zeros(
            [batch_size, 0],
            (global_block_ids.kind(), global_block_ids.device()),
        )
    };
    let global_segment_ids = Tensor::ones(
        [batch_size, num_globals],
        (attention_mask.kind(), attention_mask.device()),
    )
    .cumsum(-1, attention_mask.kind())
        - 1;
    let global_segment_ids = global_segment_ids
        .ones_like()
        .where_scalarother(&global_segment_ids.le_tensor(&sequence_block_ids_max), 0.0);
    (
        global_block_ids.to_kind(Kind::Int),
        global_segment_ids.to_kind(Kind::Int),
    )
}

fn make_side_relative_position_ids(attention_mask: &Tensor, global_block_size: i64) -> Tensor {
    let (block_ids, global_segment_ids) =
        make_global_fixed_block_ids(attention_mask, global_block_size);
    let global_seq_length = *global_segment_ids.size().last().unwrap();
    let global_positions = Tensor::arange(global_seq_length, (Kind::Int64, block_ids.device()));
    global_positions - block_ids.unsqueeze(-1)
}

fn create_global_aggregates(
    hidden_states: &Tensor,
    block_ids: &Tensor,
    global_seq_length: i64,
) -> Tensor {
    let block_ids = block_ids.where_scalarother(&block_ids.ge(0), global_seq_length);
    let one_hot_block_ids = block_ids
        .to_kind(Kind::Int64)
        .one_hot(global_seq_length + 1);
    let one_hot_block_ids = one_hot_block_ids.narrow(2, 0, one_hot_block_ids.size()[2] - 1);
    Tensor::einsum(
        "...nd,...ng->...gd",
        &[
            hidden_states,
            &one_hot_block_ids.to_kind(hidden_states.kind()),
        ],
        None::<i64>,
    )
}

fn compute_bias(
    block_length: i64,
    relative_attention_bias: &nn::Embedding,
    is_decoder: bool,
    relative_attention_num_buckets: i64,
    relative_attention_max_distance: i64,
) -> Tensor {
    let device = relative_attention_bias.ws.device();
    let memory_position = Tensor::arange(3 * block_length, (Kind::Int64, device));
    let context_position = memory_position.narrow(0, block_length, block_length);
    let relative_position = memory_position.unsqueeze(0) - context_position.unsqueeze(-1);

    let rp_bucket = get_relative_position_bucket(
        &relative_position,
        !is_decoder,
        relative_attention_num_buckets,
        relative_attention_max_distance,
    );
    rp_bucket
        .apply(relative_attention_bias)
        .permute([2, 0, 1])
        .unsqueeze(0)
        .unsqueeze(0)
}

pub struct LongT5LocalAttention {
    is_decoder: bool,
    has_relative_attention_bias: bool,
    relative_attention_num_buckets: i64,
    relative_attention_max_distance: i64,
    key_value_proj_dim: i64,
    n_heads: i64,
    block_length: i64,
    dropout: Dropout,
    inner_dim: i64,
    output_attentions: bool,
    query: nn::Linear,
    key: nn::Linear,
    value: nn::Linear,
    output: nn::Linear,
    relative_attention_bias: Option<nn::Embedding>,
}

impl LongT5LocalAttention {
    pub fn new<'p, P>(
        p: P,
        config: &LongT5Config,
        is_decoder: bool,
        has_relative_attention_bias: bool,
    ) -> LongT5LocalAttention
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let linear_config = LinearConfig {
            bias: false,
            ..Default::default()
        };

        let block_length = config.local_radius + 1;
        let key_value_proj_dim = config.d_kv;

        let inner_dim = config.num_heads * config.d_kv;
        let key = nn::linear(p / "k", config.d_model, inner_dim, linear_config);
        let value = nn::linear(p / "v", config.d_model, inner_dim, linear_config);
        let query = nn::linear(p / "q", config.d_model, inner_dim, linear_config);
        let output = nn::linear(p / "o", inner_dim, config.d_model, linear_config);

        let dropout = Dropout::new(config.dropout_rate);
        let relative_attention_bias = if has_relative_attention_bias {
            Some(nn::embedding(
                p / "relative_attention_bias",
                config.relative_attention_num_buckets,
                config.num_heads,
                Default::default(),
            ))
        } else {
            None
        };

        LongT5LocalAttention {
            is_decoder,
            has_relative_attention_bias,
            relative_attention_num_buckets: config.relative_attention_num_buckets,
            relative_attention_max_distance: config.relative_attention_max_distance.unwrap_or(128),
            key_value_proj_dim,
            n_heads: config.num_heads,
            block_length,
            dropout,
            inner_dim,
            output_attentions: config.output_attentions.unwrap_or(false),
            query,
            key,
            value,
            output,
            relative_attention_bias,
        }
    }

    pub fn forward_t(
        &self,
        hidden_states: &Tensor,
        mask: Option<&Tensor>,
        position_bias: Option<&Tensor>,
        train: bool,
    ) -> (Tensor, Option<Tensor>, Option<Tensor>) {
        let input_size = hidden_states.size();
        let (batch_size, seq_length) = (input_size[0], input_size[1]);

        let shape = |states: &Tensor| -> Tensor {
            states.view([batch_size, -1, self.n_heads, self.key_value_proj_dim])
        };
        let unshape = |states: &Tensor| -> Tensor {
            states.contiguous().view([batch_size, -1, self.inner_dim])
        };

        let query_states = shape(&hidden_states.apply(&self.query));
        let key_states = shape(&hidden_states.apply(&self.key));
        let value_states = shape(&hidden_states.apply(&self.value));

        let query_states = split_into_blocks(&query_states, self.block_length, 1);
        let key_states = split_into_blocks(&key_states, self.block_length, 1);
        let value_states = split_into_blocks(&value_states, self.block_length, 1);

        let key_states = concatenate_3_blocks(&key_states, 1, 2, None);
        let value_states = concatenate_3_blocks(&value_states, 1, 2, None);

        let mut scores = Tensor::einsum(
            "...qhd,...khd->...hqk",
            &[query_states, key_states],
            None::<i64>,
        );
        let calc_position_bias = if position_bias.is_none() {
            let mut position_bias = if !self.has_relative_attention_bias {
                Tensor::zeros(
                    [1, 1, self.n_heads, self.block_length, 3 * self.block_length],
                    (scores.kind(), scores.device()),
                )
            } else {
                compute_bias(
                    self.block_length,
                    self.relative_attention_bias.as_ref().unwrap(),
                    self.is_decoder,
                    self.relative_attention_num_buckets,
                    self.relative_attention_max_distance,
                )
            };
            if let Some(mask) = mask {
                let mask = mask.zeros_like().where_scalarother(&mask.gt(0), -1e10);
                position_bias = position_bias + mask.transpose(1, 2);
            }
            Some(position_bias)
        } else {
            None
        };
        let position_bias = position_bias.unwrap_or_else(|| calc_position_bias.as_ref().unwrap());
        scores += position_bias;
        let attention_weights = scores
            .to_kind(Kind::Float)
            .softmax(-1, scores.kind())
            .apply_t(&self.dropout, train)
            .to_kind(value_states.kind());
        let attention_output = unshape(&Tensor::einsum(
            "...hqk,...khd->...qhd",
            &[&attention_weights, &value_states],
            None::<i64>,
        ))
        .narrow(1, 0, seq_length)
        .apply(&self.output);

        let attention_weights = if self.output_attentions {
            Some(attention_weights)
        } else {
            None
        };

        let position_bias = if self.has_relative_attention_bias {
            calc_position_bias
        } else {
            None
        };
        (attention_output, position_bias, attention_weights)
    }
}

pub struct LongT5TransientGlobalAttention {
    is_decoder: bool,
    has_relative_attention_bias: bool,
    relative_attention_num_buckets: i64,
    relative_attention_max_distance: i64,
    key_value_proj_dim: i64,
    n_heads: i64,
    block_length: i64,
    global_block_size: i64,
    dropout: Dropout,
    inner_dim: i64,
    output_attentions: bool,
    query: nn::Linear,
    key: nn::Linear,
    value: nn::Linear,
    output: nn::Linear,
    relative_attention_bias: Option<nn::Embedding>,
    global_relative_attention_bias: Option<nn::Embedding>,
    global_input_layer_norm: LongT5LayerNorm,
}

impl LongT5TransientGlobalAttention {
    pub fn new<'p, P>(
        p: P,
        config: &LongT5Config,
        is_decoder: bool,
        has_relative_attention_bias: bool,
    ) -> LongT5TransientGlobalAttention
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let linear_config = LinearConfig {
            bias: false,
            ..Default::default()
        };

        let block_length = config.local_radius + 1;
        let global_block_size = config.global_block_size;
        let key_value_proj_dim = config.d_kv;

        let inner_dim = config.num_heads * config.d_kv;
        let key = nn::linear(p / "k", config.d_model, inner_dim, linear_config);
        let value = nn::linear(p / "v", config.d_model, inner_dim, linear_config);
        let query = nn::linear(p / "q", config.d_model, inner_dim, linear_config);
        let output = nn::linear(p / "o", inner_dim, config.d_model, linear_config);

        let dropout = Dropout::new(config.dropout_rate);
        let global_relative_attention_bias = if has_relative_attention_bias {
            Some(nn::embedding(
                p / "global_relative_attention_bias",
                config.relative_attention_num_buckets,
                config.num_heads,
                Default::default(),
            ))
        } else {
            None
        };
        let relative_attention_bias = if has_relative_attention_bias {
            Some(nn::embedding(
                p / "relative_attention_bias",
                config.relative_attention_num_buckets,
                config.num_heads,
                Default::default(),
            ))
        } else {
            None
        };
        let global_input_layer_norm = LongT5LayerNorm::new(
            p / "global_input_layer_norm",
            config.d_model,
            config.layer_norm_epsilon,
        );

        LongT5TransientGlobalAttention {
            is_decoder,
            has_relative_attention_bias,
            relative_attention_num_buckets: config.relative_attention_num_buckets,
            relative_attention_max_distance: config.relative_attention_max_distance.unwrap_or(128),
            key_value_proj_dim,
            n_heads: config.num_heads,
            block_length,
            global_block_size,
            dropout,
            inner_dim,
            output_attentions: config.output_attentions.unwrap_or(false),
            query,
            key,
            value,
            output,
            relative_attention_bias,
            global_relative_attention_bias,
            global_input_layer_norm,
        }
    }

    fn compute_side_bias(&self, mask: &Tensor, global_segment_ids: &Tensor) -> Tensor {
        let side_attention_mask = mask
            .unsqueeze(-1)
            .eq_tensor(&global_segment_ids.unsqueeze(1))
            .unsqueeze(1);

        let attention_side_bias = side_attention_mask
            .zeros_like()
            .where_scalarother(&side_attention_mask.gt(0), -1e10);

        let side_relative_position = make_side_relative_position_ids(mask, self.global_block_size);
        let side_relative_position_bucket = get_relative_position_bucket(
            &side_relative_position,
            !self.is_decoder,
            self.relative_attention_num_buckets,
            self.relative_attention_max_distance,
        );
        let side_bias = side_relative_position_bucket
            .apply(self.global_relative_attention_bias.as_ref().unwrap())
            .permute([0, 3, 1, 2]);
        attention_side_bias + side_bias
    }

    pub fn forward_t(
        &self,
        hidden_states: &Tensor,
        mask: Option<&Tensor>,
        position_bias: Option<&Tensor>,
        train: bool,
    ) -> (Tensor, Option<Tensor>, Option<Tensor>) {
        let input_size = hidden_states.size();
        let (batch_size, seq_length) = (input_size[0], input_size[1]);

        let shape = |states: &Tensor| -> Tensor {
            states.view([batch_size, -1, self.n_heads, self.key_value_proj_dim])
        };
        let unshape = |states: &Tensor| -> Tensor {
            states.contiguous().view([batch_size, -1, self.inner_dim])
        };
        let calc_mask = if mask.is_none() {
            let mut mask_size = input_size;
            let _ = mask_size.pop();
            Some(Tensor::ones(
                mask_size.as_slice(),
                (Kind::Bool, hidden_states.device()),
            ))
        } else {
            None
        };
        let (block_ids, global_segment_ids) = make_global_fixed_block_ids(
            mask.unwrap_or_else(|| calc_mask.as_ref().unwrap()),
            self.global_block_size,
        );
        let global_seq_length = *global_segment_ids.size().last().unwrap();
        let global_inputs = create_global_aggregates(hidden_states, &block_ids, global_seq_length)
            .apply(&self.global_input_layer_norm);

        let query_states = shape(&hidden_states.apply(&self.query));
        let key_states = shape(&hidden_states.apply(&self.key));
        let value_states = shape(&hidden_states.apply(&self.value));

        let side_key_states = shape(&global_inputs.apply(&self.key));
        let side_value_states = shape(&global_inputs.apply(&self.value));

        let query_states = split_into_blocks(&query_states, self.block_length, 1);
        let key_states = split_into_blocks(&key_states, self.block_length, 1);
        let value_states = split_into_blocks(&value_states, self.block_length, 1);

        let key_states = concatenate_3_blocks(&key_states, 1, 2, None);
        let value_states = concatenate_3_blocks(&value_states, 1, 2, None);

        let mut reps = vec![1; side_key_states.dim() + 1];
        reps[1] = key_states.size()[1];
        let side_key_states = side_key_states.unsqueeze(1).repeat(reps.as_slice());
        let side_value_states = side_value_states.unsqueeze(1).repeat(reps.as_slice());
        let key_states = Tensor::cat(&[key_states, side_key_states], 2);
        let value_states = Tensor::cat(&[value_states, side_value_states], 2);

        let mut scores = Tensor::einsum(
            "...qhd,...khd->...hqk",
            &[query_states, key_states],
            None::<i64>,
        );
        let local_attention_mask = mask.map(|mask| {
            let local_attention_mask = get_local_attention_mask(mask, self.block_length);
            local_attention_mask
                .zeros_like()
                .where_scalarother(&local_attention_mask.gt(0), -1e10)
        });

        let calc_position_bias = if position_bias.is_none() {
            let mut position_bias = if !self.has_relative_attention_bias {
                Tensor::zeros(
                    [1, 1, self.n_heads, self.block_length, 3 * self.block_length],
                    (scores.kind(), scores.device()),
                )
            } else {
                compute_bias(
                    self.block_length,
                    self.relative_attention_bias.as_ref().unwrap(),
                    self.is_decoder,
                    self.relative_attention_num_buckets,
                    self.relative_attention_max_distance,
                )
            };
            if let Some(local_attention_mask) = local_attention_mask {
                position_bias = position_bias + local_attention_mask.transpose(1, 2);
            }
            let calc_mask = if mask.is_none() {
                Some(Tensor::ones(
                    [batch_size, seq_length],
                    (global_segment_ids.kind(), global_segment_ids.device()),
                ))
            } else {
                None
            };
            let mask = mask.unwrap_or_else(|| calc_mask.as_ref().unwrap());
            let side_position_bias = self.compute_side_bias(mask, &global_segment_ids);
            let side_position_bias = split_into_blocks(
                &side_position_bias,
                self.block_length,
                side_position_bias.dim() - 2,
            )
            .transpose(1, 2);
            let position_bias = Tensor::cat(&[position_bias, side_position_bias], -1);

            Some(position_bias)
        } else {
            None
        };
        let position_bias = position_bias.unwrap_or_else(|| calc_position_bias.as_ref().unwrap());

        scores += position_bias;
        let attention_weights = scores
            .to_kind(Kind::Float)
            .softmax(-1, scores.kind())
            .apply_t(&self.dropout, train);

        let attention_output = unshape(&Tensor::einsum(
            "...hqk,...khd->...qhd",
            &[&attention_weights, &value_states],
            None::<i64>,
        ))
        .narrow(1, 0, seq_length)
        .apply(&self.output);

        let attention_weights = if self.output_attentions {
            Some(attention_weights)
        } else {
            None
        };

        let position_bias = if self.has_relative_attention_bias {
            calc_position_bias
        } else {
            None
        };
        (attention_output, position_bias, attention_weights)
    }
}

pub struct LongT5LayerSelfAttention {
    self_attention: LongT5Attention,
    layer_norm: LongT5LayerNorm,
    dropout: Dropout,
}

impl LongT5LayerSelfAttention {
    pub fn new<'p, P>(
        p: P,
        config: &LongT5Config,
        has_relative_attention_bias: bool,
        is_decoder: bool,
        store_cache: bool,
        output_attentions: bool,
    ) -> LongT5LayerSelfAttention
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let self_attention = LongT5Attention::new(
            p / "SelfAttention",
            &config.into(),
            is_decoder,
            !is_decoder,
            store_cache,
            output_attentions,
            has_relative_attention_bias,
        );

        let layer_norm =
            LongT5LayerNorm::new(p / "layer_norm", config.d_model, config.layer_norm_epsilon);
        let dropout = Dropout::new(config.dropout_rate);

        LongT5LayerSelfAttention {
            self_attention,
            layer_norm,
            dropout,
        }
    }

    pub fn forward_t(
        &self,
        hidden_states: &Tensor,
        position_bias: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        layer_state: Option<LayerState>,
        train: bool,
    ) -> (Tensor, Option<Tensor>, Option<Tensor>, Option<LayerState>) {
        let norm_x = hidden_states.apply(&self.layer_norm);

        let (y, attention_weights, position_bias, layer_state) = self.self_attention.forward_t(
            &norm_x,
            None,
            position_bias,
            attention_mask,
            layer_state,
            None,
            train,
        );

        let output = hidden_states + y.apply_t(&self.dropout, train);

        (output, attention_weights, position_bias, layer_state)
    }
}

pub struct LongT5LayerLocalSelfAttention {
    local_self_attention: LongT5LocalAttention,
    layer_norm: LongT5LayerNorm,
    dropout: Dropout,
}

impl LongT5LayerLocalSelfAttention {
    pub fn new<'p, P>(
        p: P,
        config: &LongT5Config,
        has_relative_attention_bias: bool,
        is_decoder: bool,
    ) -> LongT5LayerLocalSelfAttention
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let local_self_attention = LongT5LocalAttention::new(
            p / "LocalSelfAttention",
            config,
            is_decoder,
            has_relative_attention_bias,
        );

        let layer_norm =
            LongT5LayerNorm::new(p / "layer_norm", config.d_model, config.layer_norm_epsilon);
        let dropout = Dropout::new(config.dropout_rate);

        LongT5LayerLocalSelfAttention {
            local_self_attention,
            layer_norm,
            dropout,
        }
    }

    pub fn forward_t(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        position_bias: Option<&Tensor>,
        train: bool,
    ) -> (Tensor, Option<Tensor>, Option<Tensor>) {
        let normed_hidden_states = hidden_states.apply(&self.layer_norm);

        let (attention_output, position_bias, attention_weights) = self
            .local_self_attention
            .forward_t(&normed_hidden_states, attention_mask, position_bias, train);

        let output = hidden_states + attention_output.apply_t(&self.dropout, train);

        (output, position_bias, attention_weights)
    }
}

pub struct LongT5LayerTransientGlobalSelfAttention {
    transient_global_sef_attention: LongT5TransientGlobalAttention,
    layer_norm: LongT5LayerNorm,
    dropout: Dropout,
}

impl LongT5LayerTransientGlobalSelfAttention {
    pub fn new<'p, P>(
        p: P,
        config: &LongT5Config,
        has_relative_attention_bias: bool,
        is_decoder: bool,
    ) -> LongT5LayerTransientGlobalSelfAttention
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let transient_global_sef_attention = LongT5TransientGlobalAttention::new(
            p / "TransientGlobalSelfAttention",
            config,
            is_decoder,
            has_relative_attention_bias,
        );

        let layer_norm =
            LongT5LayerNorm::new(p / "layer_norm", config.d_model, config.layer_norm_epsilon);
        let dropout = Dropout::new(config.dropout_rate);

        LongT5LayerTransientGlobalSelfAttention {
            transient_global_sef_attention,
            layer_norm,
            dropout,
        }
    }

    pub fn forward_t(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        position_bias: Option<&Tensor>,
        train: bool,
    ) -> (Tensor, Option<Tensor>, Option<Tensor>) {
        let normed_hidden_states = hidden_states.apply(&self.layer_norm);
        let (attention_output, position_bias, attention_weights) = self
            .transient_global_sef_attention
            .forward_t(&normed_hidden_states, attention_mask, position_bias, train);

        let output = hidden_states + attention_output.apply_t(&self.dropout, train);

        (output, position_bias, attention_weights)
    }
}
