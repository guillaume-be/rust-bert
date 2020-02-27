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

use tch::{Tensor, nn};
use crate::common::dropout::Dropout;
use crate::gpt2::gpt2::Gpt2Config;
use tch::kind::Kind::Float;
use tch::nn::{conv1d, Init, Module};

#[derive(Debug)]
pub struct GPTConv1D {
    weight: Tensor,
    bias: Tensor,
    nf: i64,
}

impl GPTConv1D {
    pub fn new(p: &nn::Path, nf: i64, nx: i64) -> GPTConv1D {
        let weight = p.var("weight", &[nx, nf], Init::Randn { mean: 0., stdev: 0.02 });
        let bias = p.var("bias", &[nf], Init::Const(0.));
        GPTConv1D { weight, bias, nf }
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
}

impl Attention {
    pub fn new(p: &nn::Path, nx: i64, n_ctx: i64, config: &Gpt2Config) -> Attention {
        let bias = Tensor::ones(&[n_ctx, n_ctx], (Float, p.device())).tril(0).view((1, 1, n_ctx, n_ctx));
        let c_attn = GPTConv1D::new(&(p / "c_attn"), nx * 3, nx);
        let c_proj = GPTConv1D::new(&(p / "c_proj"), nx, nx);

        let attn_pdrop = match config.attn_pdrop {
            Some(value) => value,
            None => 0.1
        };

        let resid_pdrop = match config.resid_pdrop {
            Some(value) => value,
            None => 0.1
        };

        let output_attentions = match config.output_attentions {
            Some(value) => value,
            None => false
        };

        let attn_dropout = Dropout::new(attn_pdrop);
        let resid_dropout = Dropout::new(resid_pdrop);

        Attention { bias, c_attn, c_proj, attn_dropout, resid_dropout, output_attentions }
    }
}