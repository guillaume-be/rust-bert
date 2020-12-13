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
use crate::mobilebert::attention::MobileBertAttention;
use crate::mobilebert::encoder::BottleneckOutput::BottleNeckSharedAttn;
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

pub struct BottleneckLayer {
    pub dense: nn::Linear,
    pub layer_norm: NormalizationLayer,
}

impl BottleneckLayer {
    pub fn new<'p, P>(p: P, config: &MobileBertConfig) -> BottleneckLayer
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let intra_bottleneck_size = config.intra_bottleneck_size.unwrap_or(128);

        let dense = nn::linear(
            p / "dense",
            config.hidden_size,
            intra_bottleneck_size,
            Default::default(),
        );
        let layer_norm = NormalizationLayer::new(
            p / "LayerNorm",
            config
                .normalization_type
                .unwrap_or(NormalizationType::no_norm),
            intra_bottleneck_size,
            config.layer_norm_eps,
        );

        BottleneckLayer { dense, layer_norm }
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Tensor {
        self.layer_norm.forward(&hidden_states.apply(&self.dense))
    }
}

pub enum BottleneckOutput {
    Bottleneck(Tensor),
    BottleNeckSharedAttn(Tensor, Tensor),
}

pub struct Bottleneck {
    pub input: BottleneckLayer,
    pub attention: Option<BottleneckLayer>,
}

impl Bottleneck {
    pub fn new<'p, P>(p: P, config: &MobileBertConfig) -> Bottleneck
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let key_query_shared_bottleneck = config.key_query_shared_bottleneck.unwrap_or(true);

        let input = BottleneckLayer::new(p / "input", config);
        let attention = if key_query_shared_bottleneck {
            Some(BottleneckLayer::new(p / "attention", config))
        } else {
            None
        };

        Bottleneck { input, attention }
    }

    pub fn forward(&self, hidden_states: &Tensor) -> BottleneckOutput {
        let bottleneck_hidden_states = self.input.forward(hidden_states);
        if let Some(attention) = &self.attention {
            let shared_attention_input = attention.forward(hidden_states);
            BottleneckOutput::BottleNeckSharedAttn(bottleneck_hidden_states, shared_attention_input)
        } else {
            BottleneckOutput::Bottleneck(bottleneck_hidden_states)
        }
    }
}

pub struct FFNOutput {
    pub dense: nn::Linear,
    pub layer_norm: NormalizationLayer,
}

impl FFNOutput {
    pub fn new<'p, P>(p: P, config: &MobileBertConfig) -> FFNOutput
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
            config.layer_norm_eps,
        );

        FFNOutput { dense, layer_norm }
    }

    pub fn forward(&self, hidden_states: &Tensor, residual_tensor: &Tensor) -> Tensor {
        self.layer_norm
            .forward(&(hidden_states.apply(&self.dense) + residual_tensor))
    }
}

pub struct FFNLayer {
    pub intermediate: MobileBertIntermediate,
    pub output: FFNOutput,
}

impl FFNLayer {
    pub fn new<'p, P>(p: P, config: &MobileBertConfig) -> FFNLayer
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let intermediate = MobileBertIntermediate::new(p / "intermediate", config);
        let output = FFNOutput::new(p / "output", config);

        FFNLayer {
            intermediate,
            output,
        }
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Tensor {
        let intermediate_output = self.intermediate.forward(hidden_states);
        self.output.forward(&intermediate_output, hidden_states)
    }
}

pub struct MobileBertLayer {
    pub attention: MobileBertAttention,
    pub intermediate: MobileBertIntermediate,
    pub output: MobileBertOutput,
    pub bottleneck: Option<Bottleneck>,
    pub ffn: Option<Vec<FFNLayer>>,
    pub use_bottleneck_attention: bool,
}

impl MobileBertLayer {
    pub fn new<'p, P>(p: P, config: &MobileBertConfig) -> MobileBertLayer
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let attention = MobileBertAttention::new(p / "attention", config);
        let intermediate = MobileBertIntermediate::new(p / "intermediate", config);
        let output = MobileBertOutput::new(p / "output", config);
        let bottleneck = if config.use_bottleneck.unwrap_or(true) {
            Some(Bottleneck::new(p / "bottleneck", config))
        } else {
            None
        };
        let num_feedforward_networks = config.num_feedforward_networks.unwrap_or(4);
        let ffn = if num_feedforward_networks > 1 {
            let mut layers = Vec::with_capacity(num_feedforward_networks as usize);
            let p_layers = p / "ffn";
            for layer_index in 0..num_feedforward_networks {
                layers.push(FFNLayer::new(&p_layers / layer_index, config));
            }
            Some(layers)
        } else {
            None
        };

        let use_bottleneck_attention = config.use_bottleneck_attention.unwrap_or(false);

        MobileBertLayer {
            attention,
            intermediate,
            output,
            bottleneck,
            ffn,
            use_bottleneck_attention,
        }
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        train: bool,
    ) -> (Tensor, Option<Tensor>) {
        let (mut attention_output, attention_weights) = if let Some(bottleneck) = &self.bottleneck {
            let bottleneck_output = bottleneck.forward(hidden_states);
            let (query, key, value, layer_input) = match &bottleneck_output {
                BottleneckOutput::Bottleneck(bottleneck_hidden_states) => {
                    if self.use_bottleneck_attention {
                        (
                            bottleneck_hidden_states,
                            bottleneck_hidden_states,
                            bottleneck_hidden_states,
                            bottleneck_hidden_states,
                        )
                    } else {
                        (
                            hidden_states,
                            hidden_states,
                            hidden_states,
                            bottleneck_hidden_states,
                        )
                    }
                }
                BottleneckOutput::BottleNeckSharedAttn(
                    bottleneck_hidden_states,
                    shared_attention_input,
                ) => (
                    shared_attention_input,
                    shared_attention_input,
                    hidden_states,
                    bottleneck_hidden_states,
                ),
            };
            self.attention
                .forward_t(query, key, value, layer_input, attention_mask, train)
        } else {
            self.attention.forward_t(
                hidden_states,
                hidden_states,
                hidden_states,
                hidden_states,
                attention_mask,
                train,
            )
        };

        if let Some(additional_feedforward_networks) = &self.ffn {
            for layer in additional_feedforward_networks {
                attention_output = layer.forward(&attention_output);
            }
        };

        let layer_output = self.output.forward(
            &self.intermediate.forward(&attention_output),
            &attention_output,
            hidden_states,
            train,
        );
        (layer_output, attention_weights)
    }
}
