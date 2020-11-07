// Copyright 2020 The Trax Authors and The HuggingFace Inc. team.
// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

use crate::common::dropout::Dropout;
use crate::reformer::ReformerConfig;
use std::borrow::Borrow;
use tch::nn::LinearConfig;
use tch::{nn, Tensor};

#[derive(Debug)]
/// # Reformer attention dense layer
pub struct ReformerSelfOutput {
    dense: nn::Linear,
    dropout: Dropout,
}

impl ReformerSelfOutput {
    pub fn new<'p, P>(p: P, config: &ReformerConfig) -> ReformerSelfOutput
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let linear_config = LinearConfig {
            bias: false,
            ..Default::default()
        };
        let dense = nn::linear(
            p / "dense",
            config.num_attention_heads * config.attention_head_size,
            config.hidden_size,
            linear_config,
        );
        let dropout = Dropout::new(config.hidden_dropout_prob);

        ReformerSelfOutput { dense, dropout }
    }

    pub fn forward_t(&self, hidden_states: &Tensor, train: bool) -> Tensor {
        hidden_states
            .apply(&self.dense)
            .apply_t(&self.dropout, train)
    }
}
