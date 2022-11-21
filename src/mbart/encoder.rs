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

use crate::bart::{BartEncoderOutput, _expand_mask};
use crate::common::activations::TensorFunction;
use crate::common::dropout::Dropout;
use crate::mbart::attention::MBartAttention;
use crate::mbart::embeddings::MBartLearnedPositionalEmbedding;
use crate::mbart::MBartConfig;
use crate::Activation;
use std::borrow::{Borrow, BorrowMut};
use tch::{nn, Tensor};

pub struct MBartEncoderLayer {
    self_attention: MBartAttention,
    self_attention_layer_norm: nn::LayerNorm,
    dropout: Dropout,
    activation_dropout: Dropout,
    activation: TensorFunction,
    fc1: nn::Linear,
    fc2: nn::Linear,
    final_layer_norm: nn::LayerNorm,
}

impl MBartEncoderLayer {
    pub fn new<'p, P>(p: P, config: &MBartConfig) -> MBartEncoderLayer
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
            config.encoder_attention_heads,
            config.attention_dropout,
            false,
            false,
            output_attention,
        );
        let self_attention_layer_norm = nn::layer_norm(
            p / "self_attn_layer_norm",
            vec![config.d_model],
            layer_norm_config,
        );
        let dropout = Dropout::new(config.dropout);
        let activation_dropout = Dropout::new(config.activation_dropout);
        let activation_function = match &config.activation_function {
            Some(act_function) => act_function,
            None => &Activation::gelu,
        };
        let activation = activation_function.get_function();
        let fc1 = nn::linear(
            p / "fc1",
            config.d_model,
            config.encoder_ffn_dim,
            Default::default(),
        );
        let fc2 = nn::linear(
            p / "fc2",
            config.encoder_ffn_dim,
            config.d_model,
            Default::default(),
        );

        let final_layer_norm = nn::layer_norm(
            p / "final_layer_norm",
            vec![config.d_model],
            layer_norm_config,
        );

        MBartEncoderLayer {
            self_attention,
            self_attention_layer_norm,
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
        encoder_attention_mask: Option<&Tensor>,
        train: bool,
    ) -> (Tensor, Option<Tensor>) {
        let output = x.apply(&self.self_attention_layer_norm);
        let (output, attention_weights, _) =
            self.self_attention
                .forward_t(&output, None, encoder_attention_mask, None, train);
        let output: Tensor = output.apply_t(&self.dropout, train) + x;

        let residual = output.copy();
        let output = output.apply(&self.final_layer_norm);
        let output = (self.activation.get_fn())(&output.apply(&self.fc1));
        let output = output
            .apply_t(&self.activation_dropout, train)
            .apply(&self.fc2)
            .apply_t(&self.dropout, train);
        let output = output + residual;
        (output, attention_weights)
    }
}

pub struct MBartEncoder {
    dropout: Dropout,
    layer_norm_embedding: nn::LayerNorm,
    layer_norm: nn::LayerNorm,
    layers: Vec<MBartEncoderLayer>,
    embed_positions: MBartLearnedPositionalEmbedding,
    output_attentions: bool,
    output_hidden_states: bool,
    scale_embedding: f64,
}

impl MBartEncoder {
    pub fn new<'p, P>(p: P, config: &MBartConfig) -> MBartEncoder
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let output_attentions = config.output_attentions.unwrap_or(false);
        let output_hidden_states = config.output_hidden_states.unwrap_or(false);

        let scale_embedding = if let Some(scale_embeddings) = config.scale_embedding {
            if scale_embeddings {
                (config.d_model as f64).sqrt()
            } else {
                1.0
            }
        } else {
            1.0
        };

        let dropout = Dropout::new(config.dropout);

        let layer_norm_embedding = nn::layer_norm(
            p / "layernorm_embedding",
            vec![config.d_model],
            Default::default(),
        );

        let layer_norm = nn::layer_norm(p / "layer_norm", vec![config.d_model], Default::default());

        let embed_positions = MBartLearnedPositionalEmbedding::new(
            p / "embed_positions",
            config.max_position_embeddings,
            config.d_model,
        );

        let mut layers: Vec<MBartEncoderLayer> = vec![];
        let p_layers = p / "layers";
        for layer_index in 0..config.encoder_layers {
            layers.push(MBartEncoderLayer::new(&p_layers / layer_index, config));
        }

        MBartEncoder {
            dropout,
            layer_norm_embedding,
            layer_norm,
            layers,
            embed_positions,
            output_attentions,
            output_hidden_states,
            scale_embedding,
        }
    }

    pub fn forward_t(
        &self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
        embeddings: &nn::Embedding,
        train: bool,
    ) -> MBartEncoderOutput {
        let x = input_ids.apply(embeddings) * self.scale_embedding;
        let x = x + &self.embed_positions.forward(input_ids, 0);

        let attention_mask = attention_mask.map(|mask| _expand_mask(mask, None, x.kind()));

        let mut hidden_state = x
            .apply(&self.layer_norm_embedding)
            .apply_t(&self.dropout, train);

        let mut all_hidden_states: Option<Vec<Tensor>> = if self.output_hidden_states {
            Some(vec![])
        } else {
            None
        };
        let mut all_attentions: Option<Vec<Tensor>> = if self.output_attentions {
            Some(vec![])
        } else {
            None
        };

        let mut attention_weights: Option<Tensor>;

        for layer in &self.layers {
            let temp = layer.forward_t(&hidden_state, attention_mask.as_ref(), train);
            hidden_state = temp.0;
            attention_weights = temp.1;
            if let Some(attentions) = all_attentions.borrow_mut() {
                attentions.push(std::mem::take(&mut attention_weights.unwrap()));
            };
            if let Some(hidden_states) = all_hidden_states.borrow_mut() {
                hidden_states.push(hidden_state.as_ref().copy());
            };
        }

        hidden_state = hidden_state.apply(&self.layer_norm);

        MBartEncoderOutput {
            hidden_state,
            all_hidden_states,
            all_attentions,
        }
    }
}

/// Container holding a MBART encoder output
pub type MBartEncoderOutput = BartEncoderOutput;
