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

use crate::fnet::attention::FNetLayer;
use crate::fnet::FNetConfig;
use std::borrow::{Borrow, BorrowMut};
use tch::{nn, Tensor};

pub struct FNetEncoder {
    layers: Vec<FNetLayer>,
    output_hidden_states: bool,
}

impl FNetEncoder {
    pub fn new<'p, P>(p: P, config: &FNetConfig) -> FNetEncoder
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let p_layers = p / "layer";

        let mut layers: Vec<FNetLayer> = Vec::with_capacity(config.num_hidden_layers as usize);

        for layer_index in 0..config.num_hidden_layers {
            layers.push(FNetLayer::new(&p_layers / layer_index, config));
        }
        let output_hidden_states = config.output_hidden_states.unwrap_or(false);

        FNetEncoder {
            layers,
            output_hidden_states,
        }
    }

    pub fn forward_t(&self, hidden_states: &Tensor, train: bool) -> FNetEncoderOutput {
        let mut all_hidden_states: Option<Vec<Tensor>> = if self.output_hidden_states {
            Some(vec![])
        } else {
            None
        };

        let mut x: Option<Tensor> = None;

        for layer in &self.layers {
            let temp = if let Some(x_value) = &x {
                layer.forward_t(x_value, train)
            } else {
                layer.forward_t(hidden_states, train)
            };
            x = Some(temp);

            if let Some(all_hidden_states) = all_hidden_states.borrow_mut() {
                all_hidden_states.push(x.as_ref().unwrap().copy());
            };
        }

        FNetEncoderOutput {
            hidden_states: x.unwrap(),
            all_hidden_states,
        }
    }
}

/// Container for the FNet encoder output.
pub struct FNetEncoderOutput {
    /// Last hidden states from the model
    pub hidden_states: Tensor,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<Tensor>>,
}
