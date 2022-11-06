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

use std::borrow::Borrow;
use tch::nn::{Init, Module};
use tch::{nn, Kind, Tensor};

#[derive(Debug)]
pub struct T5LayerNorm {
    weight: Tensor,
    epsilon: f64,
}

impl T5LayerNorm {
    pub fn new<'p, P>(p: P, hidden_size: i64, epsilon: f64) -> T5LayerNorm
    where
        P: Borrow<nn::Path<'p>>,
    {
        let weight = p.borrow().var("weight", &[hidden_size], Init::Const(1.0));
        T5LayerNorm { weight, epsilon }
    }
}

impl Module for T5LayerNorm {
    fn forward(&self, x: &Tensor) -> Tensor {
        let input_type = x.kind();
        let variance = x.to_kind(Kind::Float).pow_tensor_scalar(2.0_f64).mean_dim(
            [-1].as_slice(),
            true,
            Kind::Float,
        );
        let x = x * (variance + self.epsilon).rsqrt();
        if input_type != Kind::Float {
            (&self.weight * x).to_kind(input_type)
        } else {
            &self.weight * x
        }
    }
}
