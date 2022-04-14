// Copyright 2018 Google AI and Google Brain team.
// Copyright 2018 Carnegie Mellon University Authors.
// Copyright 2020-present, the HuggingFace Inc. team.
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
use crate::xlnet::attention::{LayerState, XLNetRelativeAttention};
use crate::xlnet::XLNetConfig;
use std::borrow::Borrow;
use tch::{nn, Tensor};

pub struct XLNetFeedForward {
    layer_1: nn::Linear,
    layer_2: nn::Linear,
    layer_norm: nn::LayerNorm,
    dropout: Dropout,
    activation: TensorFunction,
}

impl XLNetFeedForward {
    pub fn new<'p, P>(p: P, config: &XLNetConfig) -> XLNetFeedForward
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let layer_1 = nn::linear(
            p / "layer_1",
            config.d_model,
            config.d_inner,
            Default::default(),
        );
        let layer_2 = nn::linear(
            p / "layer_2",
            config.d_inner,
            config.d_model,
            Default::default(),
        );

        let dropout = Dropout::new(config.dropout);
        let layer_norm_eps = config.layer_norm_eps.unwrap_or(1e-12);
        let layer_norm_config = nn::LayerNormConfig {
            eps: layer_norm_eps,
            ..Default::default()
        };
        let layer_norm = nn::layer_norm(p / "layer_norm", vec![config.d_model], layer_norm_config);
        let activation = config.ff_activation.get_function();

        XLNetFeedForward {
            layer_1,
            layer_2,
            layer_norm,
            dropout,
            activation,
        }
    }

    pub fn forward_t(&self, input: &Tensor, train: bool) -> Tensor {
        let output = input.apply(&self.layer_1);
        let output: Tensor = self.activation.get_fn()(&output);
        let output = output
            .apply_t(&self.dropout, train)
            .apply(&self.layer_2)
            .apply_t(&self.dropout, train);
        (output + input).apply(&self.layer_norm)
    }
}

pub struct XLNetLayer {
    rel_attn: XLNetRelativeAttention,
    ff: XLNetFeedForward,
}

impl XLNetLayer {
    pub fn new<'p, P>(p: P, config: &XLNetConfig) -> XLNetLayer
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let rel_attn = XLNetRelativeAttention::new(p / "rel_attn", config);
        let ff = XLNetFeedForward::new(p / "ff", config);
        XLNetLayer { rel_attn, ff }
    }

    pub fn forward_t(
        &self,
        output_h: &Tensor,
        output_g: Option<&Tensor>,
        attn_mask_h: Option<&Tensor>,
        attn_mask_g: Option<&Tensor>,
        r: &Tensor,
        seg_mat: Option<&Tensor>,
        layer_state: Option<LayerState>,
        target_mapping: Option<&Tensor>,
        train: bool,
    ) -> (Tensor, Option<Tensor>, Option<Tensor>, Option<Tensor>) {
        let (output_h, output_g, attention_probas_h, attention_probas_g) = self.rel_attn.forward_t(
            output_h,
            output_g,
            attn_mask_h,
            attn_mask_g,
            r,
            seg_mat,
            layer_state,
            target_mapping,
            train,
        );
        let output_h = self.ff.forward_t(&output_h, train);
        let output_g = output_g.map(|value| self.ff.forward_t(&value, train));
        (output_h, output_g, attention_probas_h, attention_probas_g)
    }
}
