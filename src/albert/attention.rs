// Copyright 2018 Google AI and Google Brain team.
// Copyright 2020-present, the HuggingFace Inc. team.
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

use crate::albert::AlbertConfig;
use crate::common::dropout::Dropout;
use std::borrow::Borrow;
use tch::{nn, Tensor};

#[derive(Debug)]
pub struct AlbertSelfAttention {
    num_attention_heads: i64,
    attention_head_size: i64,
    hidden_size: i64,
    dropout: Dropout,
    output_attentions: bool,
    query: nn::Linear,
    key: nn::Linear,
    value: nn::Linear,
    dense: nn::Linear,
    layer_norm: nn::LayerNorm,
}

impl AlbertSelfAttention {
    pub fn new<'p, P>(p: P, config: &AlbertConfig) -> AlbertSelfAttention
    where
        P: Borrow<nn::Path<'p>>,
    {
        assert_eq!(
            config.hidden_size % config.num_attention_heads,
            0,
            "Hidden size not a multiple of the number of attention heads"
        );
        let p = p.borrow();

        let query = nn::linear(
            p / "query",
            config.hidden_size,
            config.hidden_size,
            Default::default(),
        );
        let key = nn::linear(
            p / "key",
            config.hidden_size,
            config.hidden_size,
            Default::default(),
        );
        let value = nn::linear(
            p / "value",
            config.hidden_size,
            config.hidden_size,
            Default::default(),
        );
        let dense = nn::linear(
            p / "dense",
            config.hidden_size,
            config.hidden_size,
            Default::default(),
        );
        let dropout = Dropout::new(config.attention_probs_dropout_prob);
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let output_attentions = config.output_attentions.unwrap_or(false);
        let layer_norm_eps = config.layer_norm_eps.unwrap_or(1e-12);
        let layer_norm_config = nn::LayerNormConfig {
            eps: layer_norm_eps,
            ..Default::default()
        };
        let layer_norm =
            nn::layer_norm(p / "LayerNorm", vec![config.hidden_size], layer_norm_config);

        AlbertSelfAttention {
            num_attention_heads: config.num_attention_heads,
            attention_head_size,
            hidden_size: config.hidden_size,
            dropout,
            output_attentions,
            query,
            key,
            value,
            dense,
            layer_norm,
        }
    }

    fn split_heads(&self, x: Tensor, bs: i64, dim_per_head: i64) -> Tensor {
        x.view((bs, -1, self.num_attention_heads, dim_per_head))
            .transpose(1, 2)
    }

    pub fn forward_t(
        &self,
        input_ids: &Tensor,
        mask: Option<&Tensor>,
        train: bool,
    ) -> (Tensor, Option<Tensor>) {
        let bs = *input_ids.size().first().unwrap();

        let key_layer = self.split_heads(input_ids.apply(&self.key), bs, self.attention_head_size);
        let value_layer =
            self.split_heads(input_ids.apply(&self.value), bs, self.attention_head_size);
        let query_layer =
            self.split_heads(input_ids.apply(&self.query), bs, self.attention_head_size);

        let query_layer: Tensor = query_layer / (self.attention_head_size as f64).sqrt();

        let scores = if let Some(mask) = mask {
            query_layer.matmul(&key_layer.transpose(-1, -2)) + mask
        } else {
            query_layer.matmul(&key_layer.transpose(-1, -2))
        };

        let weights = scores
            .softmax(-1, scores.kind())
            .apply_t(&self.dropout, train);

        let context = weights.matmul(&value_layer).transpose(1, 2).contiguous();

        let w = self.dense.ws.transpose(0, 1).view((
            self.num_attention_heads,
            self.attention_head_size,
            self.hidden_size,
        ));

        let context: Tensor =
            Tensor::einsum("bfnd,ndh->bfh", &[context, w]) + self.dense.bs.as_ref().unwrap();
        let context = (input_ids + context.apply_t(&self.dropout, train)).apply(&self.layer_norm);

        if !self.output_attentions {
            (context, None)
        } else {
            (context, Some(weights))
        }
    }
}
