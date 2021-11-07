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

use crate::common::activations::TensorFunction;
use crate::common::dropout::Dropout;
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

pub struct FNetIntermediate {
    dense: nn::Linear,
    intermediate_activation_function: TensorFunction,
}

impl FNetIntermediate {
    pub fn new<'p, P>(p: P, config: &FNetConfig) -> FNetIntermediate
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let dense = nn::linear(
            p / "dense",
            config.hidden_size,
            config.intermediate_size,
            Default::default(),
        );

        let intermediate_activation_function = config.hidden_act.get_function();

        FNetIntermediate {
            dense,
            intermediate_activation_function,
        }
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Tensor {
        self.intermediate_activation_function.get_fn()(&hidden_states.apply(&self.dense))
    }
}

pub struct FNetOutput {
    dense: nn::Linear,
    layer_norm: nn::LayerNorm,
    dropout: Dropout,
}

impl FNetOutput {
    pub fn new<'p, P>(p: P, config: &FNetConfig) -> FNetOutput
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let dense = nn::linear(
            p / "dense",
            config.intermediate_size,
            config.hidden_size,
            Default::default(),
        );

        let layer_norm_config = LayerNormConfig {
            eps: config.layer_norm_eps.unwrap_or(1e-12),
            ..Default::default()
        };
        let layer_norm =
            nn::layer_norm(p / "LayerNorm", vec![config.hidden_size], layer_norm_config);

        let dropout = Dropout::new(config.hidden_dropout_prob);

        FNetOutput {
            dense,
            layer_norm,
            dropout,
        }
    }

    pub fn forward_t(&self, hidden_states: &Tensor, input_tensor: &Tensor, train: bool) -> Tensor {
        let hidden_states = hidden_states
            .apply(&self.dense)
            .apply_t(&self.dropout, train);

        (input_tensor + hidden_states).apply(&self.layer_norm)
    }
}

pub struct FNetLayer {
    fourier: FNetFourierTransform,
    intermediate: FNetIntermediate,
    output: FNetOutput,
}

impl FNetLayer {
    pub fn new<'p, P>(p: P, config: &FNetConfig) -> FNetLayer
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let fourier = FNetFourierTransform::new(p / "fourier", config);
        let intermediate = FNetIntermediate::new(p / "intermediate", config);
        let output = FNetOutput::new(p / "output", config);

        FNetLayer {
            fourier,
            intermediate,
            output,
        }
    }

    pub fn forward_t(&self, hidden_states: &Tensor, train: bool) -> Tensor {
        let fourier_outputs = self.fourier.forward(hidden_states);
        let intermediate_output = self.intermediate.forward(&fourier_outputs);
        self.output
            .forward_t(&intermediate_output, &fourier_outputs, train)
    }
}
