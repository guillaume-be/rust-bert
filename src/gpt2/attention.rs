// Copyright 2018-present, the HuggingFace Inc. team
// Copyright 2018-present, The OpenAI Team Authors
// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
use crate::gpt2::gpt2_model::Gpt2Config;
use std::borrow::Borrow;
use tch::kind::Kind::Float;
use tch::nn::{Init, Module};
use tch::{nn, Tensor};

#[derive(Debug)]
pub struct GPTConv1D {
    weight: Tensor,
    bias: Tensor,
}

impl GPTConv1D {
    pub fn new<'p, P>(p: P, nf: i64, nx: i64) -> GPTConv1D
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let weight = p.var(
            "weight",
            &[nx, nf],
            Init::Randn {
                mean: 0.,
                stdev: 0.02,
            },
        );
        let bias = p.var("bias", &[nf], Init::Const(0.));
        GPTConv1D { weight, bias }
    }
}

impl Module for GPTConv1D {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.matmul(&self.weight) + &self.bias
    }
}

pub struct Attention {
    bias: Tensor,
    c_attn: GPTConv1D,
    c_proj: GPTConv1D,
    attn_dropout: Dropout,
    resid_dropout: Dropout,
    output_attentions: bool,
    n_state: i64,
    dim_per_head: i64,
    n_head: i64,
    scale: bool,
}

impl Attention {
    pub fn new<'p, P>(p: P, config: &Gpt2Config, scale: bool) -> Attention
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let bias = Tensor::ones(&[config.n_ctx, config.n_ctx], (Float, p.device()))
            .tril(0)
            .view((1, 1, config.n_ctx, config.n_ctx));

        let bias = p.var_copy("bias", &bias);

        let c_attn = GPTConv1D::new(p / "c_attn", config.n_embd * 3, config.n_embd);
        let c_proj = GPTConv1D::new(p / "c_proj", config.n_embd, config.n_embd);

        let attn_pdrop = config.attn_pdrop.unwrap_or(0.1);
        let resid_pdrop = config.resid_pdrop.unwrap_or(0.1);
        let output_attentions = config.output_attentions.unwrap_or(false);

        let attn_dropout = Dropout::new(attn_pdrop);
        let resid_dropout = Dropout::new(resid_pdrop);

        assert_eq!(
            config.n_embd % config.n_head,
            0,
            "Attention hidden states not a multiple of the number of heads"
        );
        let dim_per_head = config.n_embd / config.n_head;

        Attention {
            bias,
            c_attn,
            c_proj,
            attn_dropout,
            resid_dropout,
            output_attentions,
            n_state: config.n_embd,
            dim_per_head,
            n_head: config.n_head,
            scale,
        }
    }

    fn split_heads(&self, x: &Tensor, k: bool) -> Tensor {
        let x = x.view((x.size()[0], -1, self.n_head, self.dim_per_head));
        if k {
            x.permute(&[0, 2, 3, 1])
        } else {
            x.permute(&[0, 2, 1, 3])
        }
    }

    fn flatten(&self, x: Tensor) -> Tensor {
        x.transpose(1, 2)
            .contiguous()
            .view((x.size()[0], -1, self.n_head * self.dim_per_head))
    }

    fn attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attention_mask: Option<&Tensor>,
        train: bool,
    ) -> (Tensor, Option<Tensor>) {
        let mut w = query.matmul(key);
        if self.scale {
            w = w / (*value.size().last().unwrap() as f64).sqrt();
        }

        let (nd, ns) = (w.size()[2], w.size()[3]);
        let b = self.bias.narrow(2, ns - nd, nd).narrow(3, 0, ns);
        let mut w: Tensor = w * &b + 1e4 * (&b - 1);
        if let Some(mask) = attention_mask {
            w = w + mask;
        }
        w = w.softmax(-1, w.kind()).apply_t(&self.attn_dropout, train);

        let output = w.matmul(value);

        if self.output_attentions {
            (output, Some(w))
        } else {
            (output, None)
        }
    }

    pub fn forward_t(
        &self,
        x: &Tensor,
        layer_past: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        train: bool,
    ) -> (Tensor, Tensor, Option<Tensor>) {
        let x = x.apply(&self.c_attn).split(self.n_state, 2);

        let (query, key, value) = (
            self.split_heads(&x[0], false),
            self.split_heads(&x[1], true),
            self.split_heads(&x[2], false),
        );
        let (key, value) = match layer_past {
            Some(past) => {
                let key = Tensor::cat(&[past.get(0).transpose(-2, -1), key], -1);
                let value = Tensor::cat(&[past.get(1), value], -2);
                (key, value)
            }
            None => (key, value),
        };
        let present = Tensor::stack(&[key.transpose(-2, -1), value.copy()], 0);
        let (a, attentions) = self.attention(&query, &key, &value, attention_mask, train);

        let a = self
            .flatten(a)
            .apply(&self.c_proj)
            .apply_t(&self.resid_dropout, train);

        (a, present, attentions)
    }
}
