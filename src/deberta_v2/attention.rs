// Copyright 2020, Microsoft and the HuggingFace Inc. team.
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

use crate::common::dropout::XDropout;
use crate::deberta::{PositionAttentionType, PositionAttentionTypes};
use crate::deberta_v2::DebertaV2Config;
use std::borrow::Borrow;
use tch::nn;

pub struct DisentangledSelfAttention {
    query_proj: nn::Linear,
    key_proj: nn::Linear,
    value_proj: nn::Linear,
    pos_key_proj: Option<nn::Linear>,
    pos_query_proj: Option<nn::Linear>,
    position_buckets: Option<i64>,
    pos_embed_size: Option<i64>,
    dropout: XDropout,
    num_attention_heads: i64,
    pos_att_type: PositionAttentionTypes,
    max_relative_positions: Option<i64>,
    pos_dropout: Option<XDropout>,
    output_attentions: bool,
}

impl DisentangledSelfAttention {
    pub fn new<'p, P>(p: P, config: &DebertaV2Config) -> DisentangledSelfAttention
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let num_attention_heads = config.num_attention_heads;
        let query_proj = nn::linear(
            p / "query_proj",
            config.hidden_size,
            config.hidden_size,
            Default::default(),
        );
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
        let share_attention_key = config.share_att_key.unwrap_or(false);
        let pos_att_type = config.pos_att_type.clone().unwrap_or_default();
        let relative_attention = config.relative_attention.unwrap_or(false);

        let (
            max_relative_positions,
            pos_dropout,
            pos_key_proj,
            pos_query_proj,
            position_buckets,
            pos_embed_size,
        ) = if relative_attention {
            let position_buckets = config.position_buckets.unwrap_or(-1);
            let mut max_relative_positions = config.max_relative_positions.unwrap_or(-1);
            if max_relative_positions < 1 {
                max_relative_positions = config.max_position_embeddings;
            }
            let pos_ebd_size = if position_buckets > 0 {
                position_buckets
            } else {
                max_relative_positions
            };
            let pos_dropout = Some(XDropout::new(config.hidden_dropout_prob));

            let (pos_key_proj, pos_query_proj) = if !share_attention_key {
                let pos_key_proj = if pos_att_type.has_type(PositionAttentionType::c2p)
                    | pos_att_type.has_type(PositionAttentionType::p2p)
                {
                    Some(nn::linear(
                        p / "pos_key_proj",
                        config.hidden_size,
                        config.hidden_size,
                        Default::default(),
                    ))
                } else {
                    None
                };
                let pos_query_proj = if pos_att_type.has_type(PositionAttentionType::p2c)
                    | pos_att_type.has_type(PositionAttentionType::p2p)
                {
                    Some(nn::linear(
                        p / "pos_query_proj",
                        config.hidden_size,
                        config.hidden_size,
                        Default::default(),
                    ))
                } else {
                    None
                };
                (pos_key_proj, pos_query_proj)
            } else {
                (None, None)
            };

            (
                Some(max_relative_positions),
                pos_dropout,
                pos_key_proj,
                pos_query_proj,
                Some(position_buckets),
                Some(pos_ebd_size),
            )
        } else {
            (None, None, None, None, None, None)
        };
        let dropout = XDropout::new(config.attention_probs_dropout_prob);

        let output_attentions = config.output_attentions.unwrap_or(false);

        DisentangledSelfAttention {
            query_proj,
            key_proj,
            num_attention_heads,
            pos_att_type,
            max_relative_positions,
            pos_dropout,
            dropout,
            output_attentions,
            value_proj,
            pos_key_proj,
            pos_query_proj,
            position_buckets,
            pos_embed_size,
        }
    }
}
