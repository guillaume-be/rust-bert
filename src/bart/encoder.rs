// Copyright 2020 The Facebook AI Research Team Authors
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

use crate::bart::attention::SelfAttention;
use crate::bart::bart_model::Activation;
use crate::bart::embeddings::{
    EmbeddingOption, LearnedPositionalEmbedding, SinusoidalPositionalEmbedding,
};
use crate::bart::BartConfig;
use crate::common::activations::{_gelu, _gelu_new, _relu, _swish, _tanh};
use crate::common::dropout::Dropout;
use std::borrow::{Borrow, BorrowMut};
use tch::kind::Kind::Bool;
use tch::{nn, Tensor};

pub struct EncoderLayer {
    self_attention: SelfAttention,
    self_attention_layer_norm: nn::LayerNorm,
    dropout: Dropout,
    activation_dropout: Dropout,
    activation: Box<dyn Fn(&Tensor) -> Tensor>,
    fc1: nn::Linear,
    fc2: nn::Linear,
    final_layer_norm: nn::LayerNorm,
}

impl EncoderLayer {
    pub fn new<'p, P>(p: P, config: &BartConfig) -> EncoderLayer
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let layer_norm_config = nn::LayerNormConfig {
            eps: 1e-5,
            ..Default::default()
        };
        let output_attention = match config.output_attentions {
            Some(value) => value,
            None => false,
        };
        let self_attention = SelfAttention::new(
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
        let activation = Box::new(match activation_function {
            Activation::gelu => _gelu,
            Activation::relu => _relu,
            Activation::swish => _swish,
            Activation::gelu_new => _gelu_new,
            Activation::tanh => _tanh,
        });
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

        EncoderLayer {
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
        encoder_padding_mask: Option<&Tensor>,
        train: bool,
    ) -> (Tensor, Option<Tensor>) {
        let (output, attention_weights, _) =
            self.self_attention
                .forward_t(x, None, encoder_padding_mask, None, None, train);
        let output: Tensor = output.apply_t(&self.dropout, train) + x;
        let output = output.apply(&self.self_attention_layer_norm);

        let residual = output.copy();
        let output = (self.activation)(&output.apply(&self.fc1));
        let output = output
            .apply_t(&self.activation_dropout, train)
            .apply(&self.fc2)
            .apply_t(&self.dropout, train);
        let output: Tensor = output + residual;
        (output.apply(&self.final_layer_norm), attention_weights)
    }
}

pub struct BartEncoder {
    dropout: Dropout,
    layer_norm_embedding: Option<nn::LayerNorm>,
    layers: Vec<EncoderLayer>,
    embed_positions: EmbeddingOption,
    output_attentions: bool,
    output_hidden_states: bool,
    scale_embedding: f64,
}

impl BartEncoder {
    pub fn new<'p, P>(p: P, config: &BartConfig) -> BartEncoder
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let output_attentions = match config.output_attentions {
            Some(value) => value,
            None => false,
        };
        let output_hidden_states = match config.output_hidden_states {
            Some(value) => value,
            None => false,
        };
        let normalize_embedding = match config.normalize_embedding {
            Some(value) => value,
            None => true,
        };
        let static_position_embeddings = match config.static_position_embeddings {
            Some(value) => value,
            None => false,
        };
        let scale_embedding = match config.scale_embedding {
            Some(value) => {
                if value {
                    (config.d_model as f64).sqrt()
                } else {
                    1.0
                }
            }
            None => 1.0,
        };

        let dropout = Dropout::new(config.dropout);

        let layer_norm_embedding = if normalize_embedding {
            let layer_norm_config = nn::LayerNormConfig {
                eps: 1e-5,
                ..Default::default()
            };
            Some(nn::layer_norm(
                p / "layernorm_embedding",
                vec![config.d_model],
                layer_norm_config,
            ))
        } else {
            None
        };

        let pad_token_id = match config.pad_token_id {
            Some(value) => value,
            None => 1,
        };

        let embed_positions = if static_position_embeddings {
            EmbeddingOption::SinusoidalPositionalEmbedding(SinusoidalPositionalEmbedding::new(
                p / "embed_positions",
                config.max_position_embeddings,
                config.d_model,
            ))
        } else {
            EmbeddingOption::LearnedPositionalEmbedding(LearnedPositionalEmbedding::new(
                p / "embed_positions",
                config.max_position_embeddings,
                config.d_model,
                pad_token_id,
            ))
        };

        let mut layers: Vec<EncoderLayer> = vec![];
        let p_layers = p / "layers";
        for layer_index in 0..config.encoder_layers {
            layers.push(EncoderLayer::new(&p_layers / layer_index, config));
        }

        BartEncoder {
            dropout,
            layer_norm_embedding,
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
    ) -> BartEncoderOutput {
        let attention_mask = match attention_mask {
            Some(mask) => Some(mask.eq(0).to_kind(Bool)),
            None => None,
        };

        let x = input_ids.apply(embeddings) * self.scale_embedding;
        let x: Tensor = x + &self.embed_positions.forward(input_ids, false);
        let x = if let Some(layer_norm_embedding) = &self.layer_norm_embedding {
            x.apply(layer_norm_embedding)
        } else {
            x
        };
        let x = x.apply_t(&self.dropout, train).transpose(0, 1);

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

        let mut hidden_state = x.copy();
        let mut attention_weights: Option<Tensor>;

        for layer in &self.layers {
            if let Some(hidden_states) = all_hidden_states.borrow_mut() {
                hidden_states.push(hidden_state.as_ref().copy().transpose(0, 1));
            };

            let temp = layer.forward_t(&hidden_state, attention_mask.as_ref(), train);
            hidden_state = temp.0;
            attention_weights = temp.1;
            if let Some(attentions) = all_attentions.borrow_mut() {
                attentions.push(attention_weights.as_ref().unwrap().copy());
            };
        }

        if let Some(hidden_states) = all_hidden_states.borrow_mut() {
            hidden_states.push(hidden_state.as_ref().copy().transpose(0, 1));
        };

        BartEncoderOutput {
            hidden_state: hidden_state.transpose(0, 1),
            all_hidden_states,
            all_attentions,
        }
    }
}

pub struct BartEncoderOutput {
    pub hidden_state: Tensor,
    pub all_hidden_states: Option<Vec<Tensor>>,
    pub all_attentions: Option<Vec<Tensor>>,
}
