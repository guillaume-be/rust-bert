// Copyright 2020, Microsoft and the HuggingFace Inc. team.
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

use crate::common::dropout::XDropout;
use crate::deberta::deberta_model::{PositionAttentionType, PositionAttentionTypes};
use crate::deberta::DebertaConfig;
use std::borrow::Borrow;
use tch::nn::Init;
use tch::{nn, Tensor};

pub struct DisentangledSelfAttention {
    in_proj: nn::Linear,
    q_bias: Tensor,
    v_bias: Tensor,
    num_attention_heads: i64,
    head_logits_proj: Option<nn::Linear>,
    head_weights_proj: Option<nn::Linear>,
    pos_proj: Option<nn::Linear>,
    pos_q_proj: Option<nn::Linear>,
    max_relative_positions: Option<i64>,
    pos_dropout: Option<XDropout>,
    dropout: XDropout,
}

impl DisentangledSelfAttention {
    pub fn new<'p, P>(p: P, config: &DebertaConfig) -> DisentangledSelfAttention
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let num_attention_heads = config.num_attention_heads;
        let attention_head_size = config.hidden_size / num_attention_heads;
        let all_head_size = num_attention_heads * attention_head_size;

        let linear_no_bias_config = nn::LinearConfig {
            bias: false,
            ..Default::default()
        };

        let in_proj = nn::linear(
            p / "in_proj",
            config.hidden_size,
            all_head_size * 3,
            linear_no_bias_config,
        );
        let q_bias = p.var("q_bias", &[all_head_size], Init::Const(0.0));
        let v_bias = p.var("v_bias", &[all_head_size], Init::Const(0.0));
        let pos_att_type = config
            .pos_att_type
            .clone()
            .unwrap_or(PositionAttentionTypes::default());

        let relative_attention = config.relative_attention.unwrap_or(false);
        let talking_head = config.talking_head.unwrap_or(false);

        let (head_logits_proj, head_weights_proj) = if talking_head {
            (
                Some(nn::linear(
                    p / "head_logits_proj",
                    num_attention_heads,
                    num_attention_heads,
                    linear_no_bias_config,
                )),
                Some(nn::linear(
                    p / "head_weights_proj",
                    num_attention_heads,
                    num_attention_heads,
                    linear_no_bias_config,
                )),
            )
        } else {
            (None, None)
        };

        let (max_relative_positions, pos_dropout, pos_proj, pos_q_proj) = if relative_attention {
            let mut max_relative_positions = config.max_relative_positions.unwrap_or(-1);
            if max_relative_positions < 1 {
                max_relative_positions = config.max_position_embeddings;
            }
            let pos_dropout = Some(XDropout::new(config.hidden_dropout_prob));
            let pos_proj = if pos_att_type.has_type(PositionAttentionType::c2p)
                | pos_att_type.has_type(PositionAttentionType::p2p)
            {
                Some(nn::linear(
                    p / "pos_proj",
                    config.hidden_size,
                    all_head_size,
                    linear_no_bias_config,
                ))
            } else {
                None
            };
            let pos_q_proj = if pos_att_type.has_type(PositionAttentionType::p2c)
                | pos_att_type.has_type(PositionAttentionType::p2p)
            {
                Some(nn::linear(
                    p / "pos_q_proj",
                    config.hidden_size,
                    all_head_size,
                    Default::default(),
                ))
            } else {
                None
            };
            (
                Some(max_relative_positions),
                pos_dropout,
                pos_proj,
                pos_q_proj,
            )
        } else {
            (None, None, None, None)
        };
        let dropout = XDropout::new(config.attention_probs_dropout_prob);
        DisentangledSelfAttention {
            in_proj,
            q_bias,
            v_bias,
            num_attention_heads,
            head_logits_proj,
            head_weights_proj,
            pos_proj,
            pos_q_proj,
            max_relative_positions,
            pos_dropout,
            dropout,
        }
    }
}
