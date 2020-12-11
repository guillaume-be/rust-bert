// Copyright (c) 2020  The Google AI Language Team Authors, The HuggingFace Inc. team and github/lonePatient
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

use crate::common::activations::TensorFunction;
use crate::common::dropout::Dropout;
use crate::mobilebert::mobilebert_model::{NormalizationLayer, NormalizationType};
use crate::mobilebert::MobileBertConfig;
use std::borrow::Borrow;
use tch::{nn, Tensor};

pub struct MobileBertIntermediate {
    pub dense: nn::Linear,
    pub activation: TensorFunction,
}

impl MobileBertIntermediate {
    pub fn new<'p, P>(p: P, config: &MobileBertConfig) -> MobileBertIntermediate
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let true_hidden_size = if config.use_bottleneck.unwrap_or(true) {
            config.intra_bottleneck_size.unwrap_or(128)
        } else {
            config.hidden_size
        };
        let dense = nn::linear(
            p / "dense",
            true_hidden_size,
            config.intermediate_size,
            Default::default(),
        );
        let activation_function = config.hidden_act;
        let activation = activation_function.get_function();
        MobileBertIntermediate { dense, activation }
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Tensor {
        self.activation.get_fn()(&hidden_states.apply(&self.dense))
    }
}

pub struct OutputBottleneck {
    pub dense: nn::Linear,
    pub layer_norm: NormalizationLayer,
    pub dropout: Dropout,
}

impl OutputBottleneck {
    pub fn new<'p, P>(p: P, config: &MobileBertConfig) -> OutputBottleneck
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let true_hidden_size = if config.use_bottleneck.unwrap_or(true) {
            config.intra_bottleneck_size.unwrap_or(128)
        } else {
            config.hidden_size
        };
        let dense = nn::linear(
            p / "dense",
            true_hidden_size,
            config.hidden_size,
            Default::default(),
        );

        let layer_norm = NormalizationLayer::new(
            p / "LayerNorm",
            config
                .normalization_type
                .unwrap_or(NormalizationType::no_norm),
            config.hidden_size,
            config.layer_norm_eps,
        );

        let dropout = Dropout::new(config.hidden_dropout_prob);
        OutputBottleneck {
            dense,
            layer_norm,
            dropout,
        }
    }

    pub fn forward_t(
        &self,
        hidden_states: &Tensor,
        residual_tensor: &Tensor,
        train: bool,
    ) -> Tensor {
        let layer_outputs = hidden_states
            .apply(&self.dense)
            .apply_t(&self.dropout, train);
        self.layer_norm.forward(&(layer_outputs + residual_tensor))
    }
}

pub struct MobileBertOutput {
    pub dense: nn::Linear,
    pub layer_norm: NormalizationLayer,
    pub dropout: Option<Dropout>,
    pub bottleneck: Option<OutputBottleneck>,
}

impl MobileBertOutput {
    pub fn new<'p, P>(p: P, config: &MobileBertConfig) -> MobileBertOutput
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let true_hidden_size = if config.use_bottleneck.unwrap_or(true) {
            config.intra_bottleneck_size.unwrap_or(128)
        } else {
            config.hidden_size
        };
        let dense = nn::linear(
            p / "dense",
            config.intermediate_size,
            true_hidden_size,
            Default::default(),
        );
        let layer_norm = NormalizationLayer::new(
            p / "LayerNorm",
            config
                .normalization_type
                .unwrap_or(NormalizationType::no_norm),
            true_hidden_size,
            None,
        );
        let (bottleneck, dropout) = if config.use_bottleneck.unwrap_or(true) {
            (Some(OutputBottleneck::new(p / "bottleneck", config)), None)
        } else {
            (None, Some(Dropout::new(config.hidden_dropout_prob)))
        };

        MobileBertOutput {
            dense,
            layer_norm,
            dropout,
            bottleneck,
        }
    }

    pub fn forward(
        &self,
        intermediate_states: &Tensor,
        residual_tensor_1: &Tensor,
        residual_tensor_2: &Tensor,
        train: bool,
    ) -> Tensor {
        let layer_output = intermediate_states.apply(&self.dense);
        if let Some(bottleneck) = &self.bottleneck {
            let layer_output = self.layer_norm.forward(&(layer_output + residual_tensor_1));
            bottleneck.forward_t(&layer_output, residual_tensor_2, train)
        } else {
            self.layer_norm.forward(
                &(layer_output.apply_t(self.dropout.as_ref().unwrap(), train) + residual_tensor_1),
            )
        }
    }
}
