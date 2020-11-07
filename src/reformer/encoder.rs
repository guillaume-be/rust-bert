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
use crate::reformer::attention::{AttentionType, LayerState};
use crate::reformer::{ReformerAttention, ReformerConfig};
use crate::RustBertError;
use std::borrow::Borrow;
use tch::{nn, Tensor};

#[derive(Debug)]
/// # Reformer attention dense layer
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
        attention_type: AttentionType,
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
