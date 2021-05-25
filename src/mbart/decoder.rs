// Copyright 2021, The Facebook AI Research Team and The HuggingFace Inc. team. All rights reserved.
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
use crate::mbart::attention::MBartAttention;
use crate::mbart::{LayerState, MBartConfig};
use crate::Activation;
use std::borrow::Borrow;
use tch::{nn, Tensor};

pub struct MBartDecoderLayer {
    self_attention: MBartAttention,
    encoder_attention: MBartAttention,
    self_attention_layer_norm: nn::LayerNorm,
    encoder_attention_layer_norm: nn::LayerNorm,
    dropout: Dropout,
    activation_dropout: Dropout,
    activation: TensorFunction,
    fc1: nn::Linear,
    fc2: nn::Linear,
    final_layer_norm: nn::LayerNorm,
}

impl MBartDecoderLayer {
    pub fn new<'p, P>(p: P, config: &MBartConfig) -> MBartDecoderLayer
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let layer_norm_config = nn::LayerNormConfig {
            eps: 1e-5,
            ..Default::default()
        };
        let output_attention = config.output_attentions.unwrap_or(false);
        let self_attention = MBartAttention::new(
            p / "self_attn",
            config.d_model,
            config.decoder_attention_heads,
            config.attention_dropout,
            false,
            true,
            output_attention,
        );
        let encoder_attention = MBartAttention::new(
            p / "encoder_attn",
            config.d_model,
            config.decoder_attention_heads,
            config.attention_dropout,
            true,
            true,
            output_attention,
        );
        let self_attention_layer_norm = nn::layer_norm(
            p / "self_attn_layer_norm",
            vec![config.d_model],
            layer_norm_config,
        );
        let encoder_attention_layer_norm = nn::layer_norm(
            p / "encoder_attn_layer_norm",
            vec![config.d_model],
            layer_norm_config,
        );

        let dropout = Dropout::new(config.dropout);
        let activation_dropout = Dropout::new(config.activation_dropout);
        let activation_function = config.activation_function.unwrap_or(Activation::gelu);
        let activation = activation_function.get_function();
        let fc1 = nn::linear(
            p / "fc1",
            config.d_model,
            config.decoder_ffn_dim,
            Default::default(),
        );
        let fc2 = nn::linear(
            p / "fc2",
            config.decoder_ffn_dim,
            config.d_model,
            Default::default(),
        );

        let final_layer_norm = nn::layer_norm(
            p / "final_layer_norm",
            vec![config.d_model],
            layer_norm_config,
        );

        MBartDecoderLayer {
            self_attention,
            encoder_attention,
            self_attention_layer_norm,
            encoder_attention_layer_norm,
            dropout,
            activation_dropout,
            activation,
            fc1,
            fc2,
            final_layer_norm,
        }
    }

    pub fn forward_t(
        &self,
        x: &Tensor,
        encoder_hidden_states: &Tensor,
        encoder_attention_mask: Option<&Tensor>,
        decoder_attention_mask: Option<&Tensor>,
        layer_states: (Option<LayerState>, Option<LayerState>),
        train: bool,
    ) -> (
        Tensor,
        Option<Tensor>,
        (Option<LayerState>, Option<LayerState>),
    ) {
        let output = x.apply(&self.self_attention_layer_norm);

        let (output, attention_weights, new_self_layer_states) = self.self_attention.forward_t(
            &output,
            None,
            decoder_attention_mask,
            layer_states.0,
            train,
        );
        let output: Tensor = output.apply_t(&self.dropout, train) + x;

        let output1 = output.apply(&self.encoder_attention_layer_norm);
        let (output1, _, new_encoder_layer_states) = self.encoder_attention.forward_t(
            &output1,
            Some(encoder_hidden_states),
            encoder_attention_mask,
            layer_states.1,
            train,
        );
        let output1: Tensor = output1.apply_t(&self.dropout, train) + output;

        let output2 = output1.apply(&self.final_layer_norm);
        let output2 = (self.activation.get_fn())(&output2.apply(&self.fc1));
        let output2 = output2
            .apply_t(&self.activation_dropout, train)
            .apply(&self.fc2)
            .apply_t(&self.dropout, train);
        let output2: Tensor = output2 + output1;
        (
            output2,
            attention_weights,
            (new_self_layer_states, new_encoder_layer_states),
        )
    }
}
