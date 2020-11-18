// Copyright 2020 The Trax Authors and The HuggingFace Inc. team.
// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
use crate::reformer::attention::{AttentionType, LayerState, ReformerAttention};
use crate::reformer::ReformerConfig;
use crate::RustBertError;
use std::borrow::{Borrow, BorrowMut};
use tch::{nn, Tensor};

pub struct ReformerFeedForwardDense {
    dense: nn::Linear,
    dropout: Dropout,
    activation: TensorFunction,
}

impl ReformerFeedForwardDense {
    pub fn new<'p, P>(p: P, config: &ReformerConfig) -> ReformerFeedForwardDense
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let dense = nn::linear(
            p / "dense",
            config.hidden_size,
            config.feed_forward_size,
            Default::default(),
        );
        let dropout = Dropout::new(config.hidden_dropout_prob);
        let activation = config.hidden_act.get_function();
        ReformerFeedForwardDense {
            dense,
            dropout,
            activation,
        }
    }

    pub fn forward_t(&self, hidden_states: &Tensor, train: bool) -> Tensor {
        self.activation.get_fn()(
            &hidden_states
                .apply(&self.dense)
                .apply_t(&self.dropout, train),
        )
    }
}

pub struct ReformerFeedForwardOutput {
    dense: nn::Linear,
    dropout: Dropout,
}

impl ReformerFeedForwardOutput {
    pub fn new<'p, P>(p: P, config: &ReformerConfig) -> ReformerFeedForwardOutput
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let dense = nn::linear(
            p / "dense",
            config.feed_forward_size,
            config.hidden_size,
            Default::default(),
        );
        let dropout = Dropout::new(config.hidden_dropout_prob);

        ReformerFeedForwardOutput { dense, dropout }
    }

    pub fn forward_t(&self, hidden_states: &Tensor, train: bool) -> Tensor {
        hidden_states
            .apply(&self.dense)
            .apply_t(&self.dropout, train)
    }
}

pub struct ChunkReformerFeedForward {
    dense: ReformerFeedForwardDense,
    output: ReformerFeedForwardOutput,
    layer_norm: nn::LayerNorm,
    chunk_size_feed_forward: i64,
}

impl ChunkReformerFeedForward {
    pub fn new<'p, P>(p: P, config: &ReformerConfig) -> ChunkReformerFeedForward
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let dense = ReformerFeedForwardDense::new(p / "dense", config);
        let output = ReformerFeedForwardOutput::new(p / "output", config);
        let layer_norm_config = nn::LayerNormConfig {
            eps: config.layer_norm_eps.unwrap_or(1e-12),
            ..Default::default()
        };
        let layer_norm = nn::layer_norm(
            p / "layer_norm",
            vec![config.hidden_size],
            layer_norm_config,
        );

        let chunk_size_feed_forward = config.chunk_size_feed_forward.unwrap_or(0);

        ChunkReformerFeedForward {
            dense,
            output,
            layer_norm,
            chunk_size_feed_forward,
        }
    }

    pub fn forward_t(&self, hidden_states: &Tensor, train: bool) -> Tensor {
        if self.chunk_size_feed_forward > 0 {
            let num_chunks = hidden_states.size()[1] / self.chunk_size_feed_forward;
            let input_tensors_chunk = hidden_states.chunk(num_chunks, 1);
            let output_chunks = input_tensors_chunk
                .iter()
                .map(|v| self._forward_t(v, train))
                .collect::<Vec<Tensor>>();
            Tensor::cat(output_chunks.as_slice(), 1)
        } else {
            self._forward_t(hidden_states, train)
        }
    }

    fn _forward_t(&self, hidden_states: &Tensor, train: bool) -> Tensor {
        let hidden_states = hidden_states.apply(&self.layer_norm);
        let hidden_states = self.dense.forward_t(&hidden_states, train);
        self.output.forward_t(&hidden_states, train)
    }
}

pub struct ReformerLayerOutput {
    pub attention_output: Tensor,
    pub hidden_states: Tensor,
    pub attention_probs: Option<Tensor>,
    pub buckets: Option<Tensor>,
    pub new_layer_state: Option<LayerState>,
}

pub struct ReformerLayer {
    attention: ReformerAttention,
    feed_forward: ChunkReformerFeedForward,
}

impl ReformerLayer {
    pub fn new<'p, P>(
        p: P,
        config: &ReformerConfig,
        attention_type: &AttentionType,
        output_attentions: bool,
        use_past: bool,
    ) -> Result<ReformerLayer, RustBertError>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let attention = ReformerAttention::new(
            p / "attention",
            config,
            attention_type,
            output_attentions,
            use_past,
        )?;
        let feed_forward = ChunkReformerFeedForward::new(p / "feed_forward", config);

        Ok(ReformerLayer {
            attention,
            feed_forward,
        })
    }

    pub fn forward_t(
        &self,
        prev_attention_output: &Tensor,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        num_hashes: Option<i64>,
        layer_state: Option<LayerState>,
        original_sequence_length: i64,
        train: bool,
    ) -> Result<ReformerLayerOutput, RustBertError> {
        let attention_layer_output = self.attention.forward_t(
            hidden_states,
            attention_mask,
            num_hashes,
            None,
            layer_state,
            original_sequence_length,
            train,
        )?;

        let attention_output = prev_attention_output + attention_layer_output.attention_output;
        let hidden_states = hidden_states + self.feed_forward.forward_t(&attention_output, train);
        Ok(ReformerLayerOutput {
            attention_output,
            hidden_states,
            attention_probs: attention_layer_output.attention_probs,
            buckets: attention_layer_output.buckets,
            new_layer_state: attention_layer_output.new_layer_state,
        })
    }
}

///Container holding a Reformer model output
pub struct ReformerModelOutput {
    /// last encoder layer hidden state
    pub hidden_states: Tensor,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all intermediate layers
    pub all_attentions: Option<Vec<Tensor>>,
    /// Cached outputs of the model (attention layers keys and values) if the model is used for generation
    pub next_cache: Option<Vec<Option<LayerState>>>,
}

pub struct ReformerEncoder {
    layers: Vec<ReformerLayer>,
    layer_norm: nn::LayerNorm,
    dropout: Dropout,
    output_attentions: bool,
    output_hidden_states: bool,
    use_cache: bool,
}

impl ReformerEncoder {
    pub fn new<'p, P>(p: P, config: &ReformerConfig) -> Result<ReformerEncoder, RustBertError>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let use_cache = config.use_cache.unwrap_or(true);
        let output_attentions = config.output_attentions.unwrap_or(false);
        let output_hidden_states = config.output_hidden_states.unwrap_or(false);

        let layer_norm_config = nn::LayerNormConfig {
            eps: config.layer_norm_eps.unwrap_or(1e-12),
            ..Default::default()
        };
        let layer_norm = nn::layer_norm(
            p / "layer_norm",
            vec![2 * config.hidden_size],
            layer_norm_config,
        );

        let mut layers: Vec<ReformerLayer> = vec![];
        let p_layers = p / "layers";
        for (layer_index, attention_type) in config.attn_layers.iter().enumerate() {
            layers.push(ReformerLayer::new(
                &p_layers / layer_index,
                config,
                attention_type,
                output_attentions,
                use_cache,
            )?);
        }

        let dropout = Dropout::new(config.hidden_dropout_prob);

        Ok(ReformerEncoder {
            layers,
            layer_norm,
            dropout,
            output_attentions,
            output_hidden_states,
            use_cache,
        })
    }

    pub fn forward_t(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        num_hashes: Option<i64>,
        old_layer_states: Option<Vec<Option<LayerState>>>,
        original_sequence_length: i64,
        train: bool,
    ) -> Result<ReformerModelOutput, RustBertError> {
        let mut hidden_state = hidden_states.copy();
        let mut attention_output = hidden_states.copy();
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
        let old_cache = old_layer_states.unwrap_or_else(|| vec![None; self.layers.len()]);
        let mut next_cache = vec![None; self.layers.len()];
        for (layer_idx, (layer, old_cache)) in
            self.layers.iter().zip(old_cache.into_iter()).enumerate()
        {
            let temp = layer.forward_t(
                &attention_output,
                &hidden_state,
                attention_mask,
                num_hashes,
                old_cache,
                original_sequence_length,
                train,
            )?;
            attention_output = temp.attention_output;
            hidden_state = temp.hidden_states;

            if let Some(hidden_states) = all_hidden_states.borrow_mut() {
                hidden_states.push(hidden_state.copy());
            };
            if let Some(attentions) = all_attentions.borrow_mut() {
                attentions.push(temp.attention_probs.unwrap());
            };
            next_cache[layer_idx] = temp.new_layer_state;
        }

        hidden_state = Tensor::cat(&[attention_output, hidden_state], -1)
            .apply(&self.layer_norm)
            .apply_t(&self.dropout, train);

        let next_cache = if self.use_cache {
            Some(next_cache)
        } else {
            None
        };

        Ok(ReformerModelOutput {
            hidden_states: hidden_state,
            all_hidden_states,
            all_attentions,
            next_cache,
        })
    }
}
