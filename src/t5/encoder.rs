// Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
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
use crate::t5::T5Config;
use std::borrow::Borrow;
use tch::nn::LinearConfig;
use tch::{nn, Tensor};

pub struct T5DenseReluDense {
    wi: nn::Linear,
    wo: nn::Linear,
    dropout: Dropout,
}

impl T5DenseReluDense {
    pub fn new<'p, P>(p: P, config: &T5Config) -> T5DenseReluDense
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let linear_config = LinearConfig {
            bias: false,
            ..Default::default()
        };
        let wi = nn::linear(p / "wi", config.d_model, config.d_ff, linear_config);
        let wo = nn::linear(p / "wi", config.d_ff, config.d_model, linear_config);
        let dropout = Dropout::new(config.dropout_rate);

        T5DenseReluDense { wi, wo, dropout }
    }

    pub fn forward_t(&self, hidden_states: &Tensor, train: bool) -> Tensor {
        hidden_states
            .apply(&self.wi)
            .relu()
            .apply_t(&self.dropout, train)
            .apply(&self.wo)
    }
}

pub struct T5LayerFF {
    dense_relu_dense: T5DenseReluDense,
    layer_norm: nn::LayerNorm,
    dropout: Dropout,
}

impl T5LayerFF {
    pub fn new<'p, P>(p: P, config: &T5Config) -> T5LayerFF
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let dense_relu_dense = T5DenseReluDense::new(p / "DenseReluDense", config);
        let layer_norm_config = nn::LayerNormConfig {
            eps: config.layer_norm_epsilon,
            ..Default::default()
        };
        let layer_norm = nn::layer_norm(p / "layer_norm", vec![config.d_model], layer_norm_config);
        let dropout = Dropout::new(config.dropout_rate);

        T5LayerFF {
            dense_relu_dense,
            layer_norm,
            dropout,
        }
    }

    pub fn forward_t(&self, hidden_states: &Tensor, train: bool) -> Tensor {
        let y = &self
            .dense_relu_dense
            .forward_t(&hidden_states.apply(&self.layer_norm), train);

        hidden_states + y.apply_t(&self.dropout, train)
    }
}
