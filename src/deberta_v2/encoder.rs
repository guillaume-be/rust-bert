// Copyright 2020, Microsoft and the HuggingFace Inc. team.
// Copyright 2022 Guillaume Becquin
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
use crate::common::dropout::XDropout;
use crate::deberta::{BaseDebertaLayer, BaseDebertaLayerNorm};
use crate::deberta_v2::attention::DebertaV2DisentangledSelfAttention;
use crate::deberta_v2::DebertaV2Config;
use crate::Activation;
use std::borrow::Borrow;
use tch::nn;
use tch::nn::{ConvConfig, LayerNorm, LayerNormConfig, Path};

pub type DebertaV2Layer = BaseDebertaLayer<DebertaV2DisentangledSelfAttention, LayerNorm>;

pub struct ConvLayer {
    conv: nn::Conv1D,
    layer_norm: nn::LayerNorm,
    dropout: XDropout,
    conv_act: TensorFunction,
}

impl ConvLayer {
    pub fn new<'p, P>(p: P, config: &DebertaV2Config) -> ConvLayer
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let conv_act = config.conv_act.unwrap_or(Activation::tanh).get_function();
        let kernel_size = config.conv_kernel_size.unwrap_or(3);
        let groups = config.conv_groups.unwrap_or(1);

        let conv_config = ConvConfig {
            padding: kernel_size - 1 / 2,
            groups,
            ..Default::default()
        };
        let conv = nn::conv1d(
            p / "conv",
            config.hidden_size,
            config.hidden_size,
            kernel_size,
            conv_config,
        );

        let layer_norm_config = LayerNormConfig {
            eps: config.layer_norm_eps.unwrap_or(1e-7),
            ..Default::default()
        };
        let layer_norm =
            nn::layer_norm(p / "LayerNorm", vec![config.hidden_size], layer_norm_config);

        let dropout = XDropout::new(config.hidden_dropout_prob);

        ConvLayer {
            conv,
            layer_norm,
            dropout,
            conv_act,
        }
    }
}

impl BaseDebertaLayerNorm for LayerNorm {
    fn new<'p, P>(p: P, size: i64, variance_epsilon: f64) -> Self
    where
        P: Borrow<Path<'p>>,
    {
        let layer_norm_config = nn::LayerNormConfig {
            eps: variance_epsilon,
            ..Default::default()
        };

        nn::layer_norm(p, vec![size], layer_norm_config)
    }
}
