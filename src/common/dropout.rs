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

use tch::nn::ModuleT;
use tch::{Kind, Tensor};

#[derive(Debug)]
pub struct Dropout {
    dropout_prob: f64,
}

impl Dropout {
    pub fn new(p: f64) -> Dropout {
        Dropout { dropout_prob: p }
    }
}

impl ModuleT for Dropout {
    fn forward_t(&self, input: &Tensor, train: bool) -> Tensor {
        input.dropout(self.dropout_prob, train)
    }
}

#[derive(Debug)]
pub struct XDropout {
    dropout_prob: f64,
}

impl XDropout {
    pub fn new(p: f64) -> XDropout {
        XDropout { dropout_prob: p }
    }
}

impl ModuleT for XDropout {
    fn forward_t(&self, input: &Tensor, train: bool) -> Tensor {
        if train {
            let mask = (Tensor::ones(&[1], (input.kind(), input.device()))
                - input
                    .empty_like()
                    .bernoulli_float_(1_f64 - self.dropout_prob))
            .to_kind(Kind::Bool);

            input.masked_fill(&mask, 0) / (1_f64 - self.dropout_prob)
        } else {
            input.shallow_clone()
        }
    }
}
