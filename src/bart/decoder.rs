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

use crate::bart::bart_model::{_expand_mask, _prepare_decoder_attention_mask};
use crate::bart::embeddings::{
    EmbeddingOption, LearnedPositionalEmbedding, SinusoidalPositionalEmbedding,
};
use crate::bart::BartConfig;
use crate::common::activations::Activation;
use crate::common::dropout::Dropout;
use crate::{
    bart::attention::{BartAttention, LayerState},
    common::activations::TensorFunction,
};
use std::borrow::{Borrow, BorrowMut};
use tch::{nn, Tensor};

pub struct DecoderLayer {
    self_attention: BartAttention,
    encoder_attention: BartAttention,
    self_attention_layer_norm: nn::LayerNorm,
    encoder_attention_layer_norm: nn::LayerNorm,
    dropout: Dropout,
    activation_dropout: Dropout,
    activation: TensorFunction,
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
        let output_attention = config.output_attentions.unwrap_or(false);
        let self_attention = BartAttention::new(
            p / "self_attn",
            config.d_model,
            config.decoder_attention_heads,
            config.attention_dropout,
            false,
            true,
            output_attention,
        );
        let encoder_attention = BartAttention::new(
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
        encoder_attention_mask: Option<&Tensor>,
        decoder_attention_mask: Option<&Tensor>,
        layer_states: (Option<LayerState>, Option<LayerState>),
        train: bool,
    ) -> (
        Tensor,
        Option<Tensor>,
        (Option<LayerState>, Option<LayerState>),
    ) {
        let (output, attention_weights, new_self_layer_states) =
            self.self_attention
                .forward_t(x, None, decoder_attention_mask, layer_states.0, train);
        let output: Tensor = output.apply_t(&self.dropout, train) + x;
        let output = output.apply(&self.self_attention_layer_norm);

        let (output1, _, new_encoder_layer_states) = self.encoder_attention.forward_t(
            &output,
            Some(encoder_hidden_states),
            encoder_attention_mask,
            layer_states.1,
            train,
        );
        let output1: Tensor = output1.apply_t(&self.dropout, train) + output;
        let output1 = output1.apply(&self.encoder_attention_layer_norm);
        let output2 = (self.activation.get_fn())(&output1.apply(&self.fc1));
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
    scale_embedding: f64,
}

impl BartDecoder {
    pub fn new<'p, P>(p: P, config: &BartConfig) -> BartDecoder
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let output_past = config.output_past.unwrap_or(true);
        let output_attentions = config.output_attentions.unwrap_or(false);
        let output_hidden_states = config.output_hidden_states.unwrap_or(false);
        let normalize_embedding = config.normalize_embedding.unwrap_or(true);
        let static_position_embeddings = config.static_position_embeddings.unwrap_or(false);
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
            scale_embedding,
        }
    }

    pub fn forward_t(
        &self,
        input_ids: &Tensor,
        encoder_hidden_states: &Tensor,
        encoder_attention_mask: Option<&Tensor>,
        decoder_attention_mask: Option<&Tensor>,
        embeddings: &nn::Embedding,
        old_layer_states: Option<Vec<(Option<LayerState>, Option<LayerState>)>>,
        train: bool,
    ) -> BartDecoderOutput {
        let past_key_values_length = if let Some(old_layer_states_values) = &old_layer_states {
            if let Some(old_value_state) = &old_layer_states_values[0].0 {
                old_value_state.prev_key.size()[2]
            } else {
                0
            }
        } else {
            0
        };

        let positions = self
            .embed_positions
            .forward(input_ids, past_key_values_length);

        let x: Tensor = input_ids.apply(embeddings) * self.scale_embedding + positions;

        let decoder_attention_mask = _prepare_decoder_attention_mask(
            decoder_attention_mask,
            input_ids.size().as_slice(),
            &x,
            past_key_values_length,
        );

        let encoder_attention_mask = encoder_attention_mask
            .map(|mask| _expand_mask(mask, Some(*input_ids.size().last().unwrap()), x.kind()));

        let x = if let Some(layer_norm_embedding) = &self.layer_norm_embedding {
            x.apply(layer_norm_embedding)
        } else {
            x
        };
        let mut hidden_state = x.apply_t(&self.dropout, train);
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

        let mut attention_weights: Option<Tensor>;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let layer_state = match &next_decoder_cache {
                Some(values) => values[layer_idx].to_owned(),
                None => (None, None),
            };
            let temp = layer.forward_t(
                &hidden_state,
                encoder_hidden_states,
                encoder_attention_mask.as_ref(),
                decoder_attention_mask.as_ref(),
                layer_state,
                train,
            );
            hidden_state = temp.0;
            attention_weights = temp.1;
            if let Some(hidden_states) = all_hidden_states.borrow_mut() {
                hidden_states.push(hidden_state.as_ref().copy());
            };
            if let Some(attentions) = all_attentions.borrow_mut() {
                attentions.push(attention_weights.as_ref().unwrap().copy());
            };
            if let Some(value) = &mut next_decoder_cache {
                value[layer_idx] = temp.2
            };
        }

        BartDecoderOutput {
            hidden_state,
            encoder_attention_mask,
            next_decoder_cache,
            all_hidden_states,
            all_attentions,
        }
    }
}

///Container holding a BART decoder output
pub struct BartDecoderOutput {
    /// last decoder layer hidden state
    pub hidden_state: Tensor,
    /// Padding mask for the encoder positions to attend to
    pub encoder_attention_mask: Option<Tensor>,
    /// Cached outputs of the model (attention layers keys and values) if the model is used for generation
    pub next_decoder_cache: Option<Vec<(Option<LayerState>, Option<LayerState>)>>,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all intermediate layers
    pub all_attentions: Option<Vec<Tensor>>,
}
