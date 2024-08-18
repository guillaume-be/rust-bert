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

use crate::gpt2::attention::Attention;
use crate::gpt2::transformer::MLP;
use crate::gpt2::Gpt2Config;
use std::borrow::Borrow;
use tch::{nn, Tensor};

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
        attention_mask: Option<&Tensor>,
        train: bool,
    ) -> (Tensor, Option<Tensor>) {
        let (output, _, attentions) = self.attn.forward_t(x, None, attention_mask, train);
        let x = (x + output).apply(&self.ln_1);
        let m = self.mlp.forward_t(&x, train);
        let x = (x + m).apply(&self.ln_2);
        (x, attentions)
    }
}
