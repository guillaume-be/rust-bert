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

use crate::bart::attention::{SelfAttention, LayerState};
use tch::{nn, Tensor};
use crate::common::dropout::Dropout;
use crate::bart::BartConfig;
use crate::bart::bart::Activation;
use crate::common::activations::{_gelu, _relu, _swish, _gelu_new, _tanh};
use crate::bart::embeddings::PositionalEmbedding;
use tch::kind::Kind::Int64;
use std::borrow::BorrowMut;

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
    pub fn new(p: nn::Path, config: &BartConfig) -> DecoderLayer {
        let layer_norm_config = nn::LayerNormConfig { eps: 1e-5, ..Default::default() };
        let output_attention = match config.output_attentions {
            Some(value) => value,
            None => false
        };
        let self_attention = SelfAttention::new(&p / "self_attn",
                                                config.d_model,
                                                config.decoder_attention_heads,
                                                config.attention_dropout,
                                                false,
                                                true,
                                                output_attention);
        let encoder_attention = SelfAttention::new(&p / "encoder_attn ",
                                                   config.d_model,
                                                   config.decoder_attention_heads,
                                                   config.attention_dropout,
                                                   true,
                                                   true,
                                                   output_attention);
        let self_attention_layer_norm = nn::layer_norm(&p / "self_attn_layer_norm",
                                                       vec![config.d_model],
                                                       layer_norm_config);
        let encoder_attention_layer_norm = nn::layer_norm(&p / "self_attn_layer_norm",
                                                          vec![config.d_model],
                                                          layer_norm_config);

        let dropout = Dropout::new(config.dropout);
        let activation_dropout = Dropout::new(config.activation_dropout);
        let activation_function = match &config.activation_function {
            Some(act_function) => act_function,
            None => &Activation::gelu
        };
        let activation = Box::new(match activation_function {
            Activation::gelu => _gelu,
            Activation::relu => _relu,
            Activation::swish => _swish,
            Activation::gelu_new => _gelu_new,
            Activation::tanh => _tanh
        });
        let fc1 = nn::linear(&p / "fc1", config.d_model, config.decoder_ffn_dim, Default::default());
        let fc2 = nn::linear(&p / "fc2", config.decoder_ffn_dim, config.d_model, Default::default());

        let final_layer_norm = nn::layer_norm(&p / "final_layer_norm",
                                              vec![config.d_model],
                                              layer_norm_config);

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

    pub fn forward_t(&mut self,
                     x: &Tensor,
                     encoder_hidden_states: &Tensor,
                     encoder_attn_mask: Option<&Tensor>,
                     causal_mask: Option<&Tensor>,
                     decoder_padding_mask: Option<&Tensor>,
                     train: bool) -> (Tensor, Option<Tensor>) {
        let (output, attention_weights) = self.self_attention.forward_t(x, Some(x), decoder_padding_mask, causal_mask, train);
        let output: Tensor = output.apply_t(&self.dropout, train) + x;
        let output = output.apply(&self.self_attention_layer_norm);

        let residual = output.copy();
        let (output, _) = self.encoder_attention.forward_t(x, Some(encoder_hidden_states), encoder_attn_mask, None, train);
        let output: Tensor = output.apply_t(&self.dropout, train) + residual;
        let output = output.apply(&self.encoder_attention_layer_norm);

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

pub struct BartDecoder<'a> {
    dropout: Dropout,
    layer_norm_embedding: nn::LayerNorm,
    layers: Vec<DecoderLayer>,
    embed_positions: PositionalEmbedding,
    embed_tokens: &'a nn::Embedding,
    output_attentions: bool,
    output_hidden_states: bool,
    output_past: bool,
    generation_mode: bool,
}

impl<'a> BartDecoder<'a> {
    pub fn new(p: nn::Path, config: &BartConfig, embed_tokens: &'a nn::Embedding, generation_mode: bool) -> BartDecoder<'a> {
        let output_past = match config.output_past {
            Some(value) => value,
            None => false
        };
        let output_attentions = match config.output_attentions {
            Some(value) => value,
            None => false
        };
        let output_hidden_states = match config.output_hidden_states {
            Some(value) => value,
            None => false
        };

        let dropout = Dropout::new(config.dropout);

        let layer_norm_config = nn::LayerNormConfig { eps: 1e-5, ..Default::default() };
        let layer_norm_embedding = nn::layer_norm(&p / "layernorm_embedding",
                                                  vec![config.d_model],
                                                  layer_norm_config);

        let pad_token_id = match config.pad_token_id {
            Some(value) => value,
            None => 1
        };

        let embed_positions = PositionalEmbedding::new(&p / "embed_positions",
                                                       config.max_position_embeddings,
                                                       config.d_model,
                                                       pad_token_id);

        let mut layers: Vec<DecoderLayer> = vec!();
        let p_layers = &p / "layer";
        for layer_index in 0..config.decoder_layers {
            layers.push(DecoderLayer::new(&p_layers / layer_index, config));
        };

        BartDecoder {
            dropout,
            layer_norm_embedding,
            layers,
            embed_positions,
            embed_tokens,
            output_attentions,
            output_hidden_states,
            output_past,
            generation_mode,
        }
    }

    pub fn forward(&mut self,
                   input_ids: &Tensor,
                   encoder_hidden_states: &Tensor,
                   encoder_padding_mask: Option<&Tensor>,
                   decoder_padding_mask: Option<&Tensor>,
                   decoder_causal_mask: Option<&Tensor>,
                   train: bool)
                   -> (Tensor,
                       (Tensor, Option<Tensor>, Option<Vec<(&LayerState, &LayerState)>>),
                       Option<Vec<Tensor>>,
                       Option<Vec<Tensor>>) {
        let encoder_padding_mask = match encoder_padding_mask {
            Some(mask) => Some(mask.eq(0).to_kind(Int64)),
            None => None
        };

        let positions = &self.embed_positions.forward(input_ids, self.generation_mode);
        let (input_ids, positions) = if self.generation_mode {
            let end = input_ids.size()[1];
            (input_ids.slice(1, end - 1, end, 1), positions.slice(1, end - 1, end, 1))
        } else {
            (input_ids.copy(), positions.copy())
        };

        let x: Tensor = input_ids.as_ref().apply(self.embed_tokens) + positions;
        let x = x
            .apply(&self.layer_norm_embedding)
            .apply_t(&self.dropout, train)
            .transpose(0, 1);

        let mut all_hidden_states: Option<Vec<Tensor>> = if self.output_hidden_states { Some(vec!()) } else { None };
        let mut all_attentions: Option<Vec<Tensor>> = if self.output_attentions { Some(vec!()) } else { None };
        let mut next_decoder_cache: Option<Vec<(&LayerState, &LayerState)>> = if self.output_past { Some(vec!()) } else { None };

        let mut hidden_state = x.copy();
        let mut attention_weights: Option<Tensor>;
        let mut layers = self.layers.iter_mut();

        loop {
            match layers.next() {
                Some(layer) => {
                    let temp = layer.forward_t(&hidden_state,
                                               encoder_hidden_states,
                                               encoder_padding_mask.as_ref(),
                                               decoder_padding_mask,
                                               decoder_causal_mask, train);
                    hidden_state = temp.0;
                    attention_weights = temp.1;
                    if let Some(hidden_states) = all_hidden_states.borrow_mut() {
                        hidden_states.push(hidden_state.as_ref().copy().transpose(0, 1));
                    };
                    if let Some(attentions) = all_attentions.borrow_mut() {
                        attentions.push(attention_weights.as_ref().unwrap().copy());
                    };
                    if let Some(cache) = next_decoder_cache.borrow_mut() {
                        cache.push((layer.self_attention.prev_state.as_ref().unwrap(), layer.encoder_attention.prev_state.as_ref().unwrap()));
                    };
                }
                None => break
            };
        };

        (hidden_state.transpose(0, 1),
         (encoder_hidden_states.copy(), encoder_padding_mask, next_decoder_cache),
         all_hidden_states,
         all_attentions)
    }
}