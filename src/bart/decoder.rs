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

use crate::bart::attention::{LayerState, SelfAttention};
use crate::bart::bart::Activation;
use crate::bart::embeddings::{
    EmbeddingOption, LearnedPositionalEmbedding, SinusoidalPositionalEmbedding,
};
use crate::bart::BartConfig;
use crate::common::activations::{_gelu, _gelu_new, _relu, _swish, _tanh};
use crate::common::dropout::Dropout;
use std::borrow::{Borrow, BorrowMut};
use tch::kind::Kind::Bool;
use tch::{nn, Tensor};

pub struct DecoderLayer {
    self_attention: SelfAttention,
    encoder_attention: SelfAttention,
    self_attention_layer_norm: nn::LayerNorm,
    encoder_attention_layer_norm: nn::LayerNorm,
    dropout: Dropout,
    activation_dropout: Dropout,
    activation: Box<dyn Fn(&Tensor) -> Tensor>,
    fc1: nn::Linear,
    fc2: nn::Linear,
    final_layer_norm: nn::LayerNorm,
}

impl DecoderLayer {
    pub fn new<'p, P>(p: P, config: &BartConfig) -> DecoderLayer
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
            config.decoder_attention_heads,
            config.attention_dropout,
            false,
            true,
            output_attention,
        );
        let encoder_attention = SelfAttention::new(
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

        DecoderLayer {
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
        encoder_attn_mask: Option<&Tensor>,
        causal_mask: Option<&Tensor>,
        decoder_padding_mask: Option<&Tensor>,
        layer_states: (Option<LayerState>, Option<LayerState>),
        train: bool,
    ) -> (
        Tensor,
        Option<Tensor>,
        (Option<LayerState>, Option<LayerState>),
    ) {
        let (output, attention_weights, new_self_layer_states) = self.self_attention.forward_t(
            x,
            Some(x),
            decoder_padding_mask,
            causal_mask,
            layer_states.0,
            train,
        );
        let output: Tensor = output.apply_t(&self.dropout, train) + x;
        let output = output.apply(&self.self_attention_layer_norm);
        let (output1, _, new_encoder_layer_states) = self.encoder_attention.forward_t(
            &output,
            Some(encoder_hidden_states),
            encoder_attn_mask,
            None,
            layer_states.1,
            train,
        );
        let output1: Tensor = output1.apply_t(&self.dropout, train) + output;
        let output1 = output1.apply(&self.encoder_attention_layer_norm);
        let output2 = (self.activation)(&output1.apply(&self.fc1));
        let output2 = output2
            .apply_t(&self.activation_dropout, train)
            .apply(&self.fc2)
            .apply_t(&self.dropout, train);
        let output2: Tensor = output2 + output1;
        (
            output2.apply(&self.final_layer_norm),
            attention_weights,
            (new_self_layer_states, new_encoder_layer_states),
        )
    }
}

pub struct BartDecoder {
    dropout: Dropout,
    layer_norm_embedding: Option<nn::LayerNorm>,
    layers: Vec<DecoderLayer>,
    embed_positions: EmbeddingOption,
    output_attentions: bool,
    output_hidden_states: bool,
    output_past: bool,
    generation_mode: bool,
    scale_embedding: f64,
}

impl BartDecoder {
    pub fn new<'p, P>(p: P, config: &BartConfig, generation_mode: bool) -> BartDecoder
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let output_past = match config.output_past {
            Some(value) => value,
            None => true,
        };
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

        let mut layers: Vec<DecoderLayer> = vec![];
        let p_layers = p / "layers";
        for layer_index in 0..config.decoder_layers {
            layers.push(DecoderLayer::new(&p_layers / layer_index, config));
        }

        BartDecoder {
            dropout,
            layer_norm_embedding,
            layers,
            embed_positions,
            output_attentions,
            output_hidden_states,
            output_past,
            generation_mode,
            scale_embedding,
        }
    }

    pub fn forward_t(
        &self,
        input_ids: &Tensor,
        encoder_hidden_states: &Tensor,
        encoder_padding_mask: Option<&Tensor>,
        decoder_padding_mask: Option<&Tensor>,
        decoder_causal_mask: Option<&Tensor>,
        embeddings: &nn::Embedding,
        old_layer_states: Option<Vec<(Option<LayerState>, Option<LayerState>)>>,
        train: bool,
    ) -> (
        Tensor,
        (
            Option<Tensor>,
            Option<Vec<(Option<LayerState>, Option<LayerState>)>>,
        ),
        Option<Vec<Tensor>>,
        Option<Vec<Tensor>>,
    ) {
        let encoder_padding_mask = match encoder_padding_mask {
            Some(mask) => Some(mask.eq(0).to_kind(Bool)),
            None => None,
        };

        let positions = self
            .embed_positions
            .forward(input_ids, self.generation_mode);
        let x: Tensor = if self.generation_mode {
            let end_inputs = input_ids.size()[1];
            let end_positions = positions.size()[1];
            input_ids.narrow(1, end_inputs - 1, 1).apply(embeddings) * self.scale_embedding
                + positions.narrow(1, end_positions - 1, 1)
        } else {
            input_ids.apply(embeddings) * self.scale_embedding + positions
        };

        let x = if let Some(layer_norm_embedding) = &self.layer_norm_embedding {
            x.apply(layer_norm_embedding)
        } else {
            x
        };
        let mut hidden_state = x.apply_t(&self.dropout, train).transpose(0, 1);
        let mut all_hidden_states: Option<Vec<Tensor>> = if self.output_hidden_states {
            Some(Vec::with_capacity(self.layers.len()))
        } else {
            None
        };
        let mut all_attentions: Option<Vec<Tensor>> = if self.output_attentions {
            Some(Vec::with_capacity(self.layers.len()))
        } else {
            None
        };
        let mut next_decoder_cache: Option<Vec<(Option<LayerState>, Option<LayerState>)>> =
            if self.output_past {
                if old_layer_states.is_some() {
                    old_layer_states
                } else {
                    Some(vec![(None, None); self.layers.len()])
                }
            } else {
                None
            };
        let encoder_hidden_states = encoder_hidden_states.transpose(0, 1);
        let mut attention_weights: Option<Tensor>;
        let mut layers = self.layers.iter().enumerate();

        loop {
            match layers.next() {
                Some((layer_idx, layer)) => {
                    let layer_state = match &next_decoder_cache {
                        Some(values) => values[layer_idx].to_owned(),
                        None => (None, None),
                    };
                    let temp = layer.forward_t(
                        &hidden_state,
                        &encoder_hidden_states,
                        encoder_padding_mask.as_ref(),
                        decoder_causal_mask,
                        decoder_padding_mask,
                        layer_state,
                        train,
                    );
                    hidden_state = temp.0;
                    attention_weights = temp.1;
                    if let Some(hidden_states) = all_hidden_states.borrow_mut() {
                        hidden_states.push(hidden_state.as_ref().copy().transpose(0, 1));
                    };
                    if let Some(attentions) = all_attentions.borrow_mut() {
                        attentions.push(attention_weights.as_ref().unwrap().copy());
                    };
                    if let Some(value) = &mut next_decoder_cache {
                        value[layer_idx] = temp.2
                    };
                }
                None => break,
            };
        }

        (
            hidden_state.transpose(0, 1),
            (encoder_padding_mask, next_decoder_cache),
            all_hidden_states,
            all_attentions,
        )
    }
}
