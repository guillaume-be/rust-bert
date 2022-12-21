// Copyright 2019 Laurent Mazare.
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

use std::borrow::Borrow;
use tch::nn::init::DEFAULT_KAIMING_UNIFORM;
use tch::nn::{Init, Module, Path};
use tch::Tensor;

#[derive(Debug, Clone, Copy)]
pub struct LinearNoBiasConfig {
    pub ws_init: Init,
}

impl Default for LinearNoBiasConfig {
    fn default() -> Self {
        LinearNoBiasConfig {
            ws_init: DEFAULT_KAIMING_UNIFORM,
        }
    }
}

#[derive(Debug)]
pub struct LinearNoBias {
    pub ws: Tensor,
}

pub fn linear_no_bias<'a, T: Borrow<Path<'a>>>(
    vs: T,
    in_dim: i64,
    out_dim: i64,
    c: LinearNoBiasConfig,
) -> LinearNoBias {
    let vs = vs.borrow();
    LinearNoBias {
        ws: vs.var("weight", &[out_dim, in_dim], c.ws_init),
    }
}

impl Module for LinearNoBias {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.matmul(&self.ws.tr())
    }
}
