// Copyright 2021 Google Research
// Copyright 2020-present, the HuggingFace Inc. team.
// Copyright 2021 Guillaume Becquin
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::fnet::FNetConfig;
use std::borrow::Borrow;
use tch::nn::LayerNormConfig;
use tch::{nn, Tensor};

pub struct FNetFourierTransform {
    layer_norm: nn::LayerNorm,
}

impl FNetFourierTransform {
    pub fn new<'p, P>(p: P, config: &FNetConfig) -> FNetFourierTransform
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let layer_norm_config = LayerNormConfig {
            eps: config.layer_norm_eps.unwrap_or(1e-12),
            ..Default::default()
        };
        let layer_norm = nn::layer_norm(
            p.sub("output").sub("LayerNorm"),
            vec![config.hidden_size],
            layer_norm_config,
        );
        FNetFourierTransform { layer_norm }
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Tensor {
        let self_outputs = hidden_states.fft_fft2(None, &[1, 2], "backward").real();
        (self_outputs + hidden_states).apply(&self.layer_norm)
    }
}
