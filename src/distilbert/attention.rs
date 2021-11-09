// Copyright 2019-present, the HuggingFace Inc. team, The Google AI Language Team and Facebook, Inc.
// Copyright 2019 Guillaume Becquin
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
use crate::distilbert::distilbert_model::DistilBertConfig;
use std::borrow::Borrow;
use tch::{nn, Tensor};

#[derive(Debug)]
pub struct MultiHeadSelfAttention {
    n_heads: i64,
    dim_per_head: i64,
    dropout: Dropout,
    output_attentions: bool,
    q_lin: nn::Linear,
    k_lin: nn::Linear,
    v_lin: nn::Linear,
    out_lin: nn::Linear,
}

impl MultiHeadSelfAttention {
    pub fn new<'p, P>(p: P, config: &DistilBertConfig) -> MultiHeadSelfAttention
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let q_lin = nn::linear(p / "q_lin", config.dim, config.dim, Default::default());
        let k_lin = nn::linear(p / "k_lin", config.dim, config.dim, Default::default());
        let v_lin = nn::linear(p / "v_lin", config.dim, config.dim, Default::default());
        let out_lin = nn::linear(p / "out_lin", config.dim, config.dim, Default::default());

        let dropout = Dropout::new(config.attention_dropout);
        let output_attentions = config.output_attentions.unwrap_or(false);

        MultiHeadSelfAttention {
            n_heads: config.n_heads,
            dim_per_head: config.dim / config.n_heads,
            dropout,
            output_attentions,
            q_lin,
            k_lin,
            v_lin,
            out_lin,
        }
    }

    fn split_heads(&self, x: Tensor, bs: i64, dim_per_head: i64) -> Tensor {
        x.view((bs, -1, self.n_heads, dim_per_head)).transpose(1, 2)
    }

    fn flatten(&self, x: Tensor, bs: i64, dim_per_head: i64) -> Tensor {
        x.transpose(1, 2)
            .contiguous()
            .view((bs, -1, self.n_heads * dim_per_head))
    }

    pub fn forward_t(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: Option<&Tensor>,
        train: bool,
    ) -> (Tensor, Option<Tensor>) {
        let bs = query.size()[0];
        let k_length = key.size()[1];

        let q = self.split_heads(query.apply(&self.q_lin), bs, self.dim_per_head);
        let k = self.split_heads(key.apply(&self.k_lin), bs, self.dim_per_head);
        let v = self.split_heads(value.apply(&self.v_lin), bs, self.dim_per_head);
        let q: Tensor = q / (self.dim_per_head as f64).sqrt();

        let scores = if let Some(mask) = mask {
            let unmasked_scores = q.matmul(&k.transpose(2, 3));
            let mask = mask
                .le_tensor(&(mask.zeros_like() + 0.1))
                .view((bs, 1i64, 1i64, k_length))
                .expand_as(&unmasked_scores);
            unmasked_scores.masked_fill(&mask, f64::NEG_INFINITY)
        } else {
            q.matmul(&k.transpose(2, 3))
        };

        let weights = scores
            .softmax(-1, scores.kind())
            .apply_t(&self.dropout, train);
        let context = self
            .flatten(weights.matmul(&v), bs, self.dim_per_head)
            .apply(&self.out_lin);

        if !self.output_attentions {
            (context, None)
        } else {
            (context, Some(weights))
        }
    }
}
