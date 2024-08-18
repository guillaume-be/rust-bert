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

use crate::common::activations::{Activation, TensorFunction};
use crate::common::dropout::Dropout;
use crate::gpt2::attention::{Attention, GPTConv1D};
use crate::gpt2::gpt2_model::Gpt2Config;
use std::borrow::Borrow;
use tch::{nn, Tensor};

pub struct MLP {
    c_fc: GPTConv1D,
    c_proj: GPTConv1D,
    activation: TensorFunction,
    dropout: Dropout,
}

impl MLP {
    pub fn new<'p, P>(p: P, config: &Gpt2Config) -> MLP
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let c_fc = GPTConv1D::new(p / "c_fc", config.n_embd * 4, config.n_embd);
        let c_proj = GPTConv1D::new(p / "c_proj", config.n_embd, config.n_embd * 4);
        let activation = match &config.afn {
            Some(activation_enum) => match activation_enum {
                Activation::gelu => &Activation::gelu_new,
                default => default,
            },
            None => &Activation::gelu_new,
        }
        .get_function();
        let resid_pdrop = config.resid_pdrop.unwrap_or(0.1);
        let dropout = Dropout::new(resid_pdrop);
        MLP {
            c_fc,
            c_proj,
            activation,
            dropout,
        }
    }

    pub fn forward_t(&self, x: &Tensor, train: bool) -> Tensor {
        let h = (self.activation.get_fn())(&x.apply(&self.c_fc));
        h.apply(&self.c_proj).apply_t(&self.dropout, train)
    }
}

pub struct Block {
    ln_1: nn::LayerNorm,
    attn: Attention,
    ln_2: nn::LayerNorm,
    mlp: MLP,
}

impl Block {
    pub fn new<'p, P>(p: P, config: &Gpt2Config, scale: bool) -> Block
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let layer_norm_config = nn::LayerNormConfig {
            eps: config.layer_norm_epsilon,
            ..Default::default()
        };
        let ln_1 = nn::layer_norm(p / "ln_1", vec![config.n_embd], layer_norm_config);
        let ln_2 = nn::layer_norm(p / "ln_2", vec![config.n_embd], layer_norm_config);
        let attn = Attention::new(p / "attn", config, scale);
        let mlp = MLP::new(p / "mlp", config);

        Block {
            ln_1,
            attn,
            ln_2,
            mlp,
        }
    }

    pub fn forward_t(
        &self,
        x: &Tensor,
        layer_past: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        train: bool,
    ) -> (Tensor, Tensor, Option<Tensor>) {
        let (output, present, attentions) =
            self.attn
                .forward_t(&x.apply(&self.ln_1), layer_past, attention_mask, train);
        let x = x + output;
        let m = self.mlp.forward_t(&x.apply(&self.ln_2), train);
        let x = x + m;
        (x, present, attentions)
    }
}
