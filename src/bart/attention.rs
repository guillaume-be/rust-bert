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

use crate::common::dropout::Dropout;
use std::borrow::Borrow;
use tch::{nn, Tensor};

#[derive(Debug)]
/// # Cache for BART attention layers
/// Stores the cached value of key, value and key padding mask to avoid recalculation (e.g. at each generation step)
pub struct LayerState {
    /// Cached keys
    pub prev_key: Tensor,
    /// Cached values
    pub prev_value: Tensor,
}

impl Clone for LayerState {
    fn clone(&self) -> Self {
        LayerState {
            prev_key: self.prev_key.copy(),
            prev_value: self.prev_value.copy(),
        }
    }
}

impl LayerState {
    pub(crate) fn reorder_cache(&mut self, new_indices: &Tensor) {
        self.prev_key = self.prev_key.index_select(0, new_indices);
        self.prev_value = self.prev_value.index_select(0, new_indices);
    }
}

#[derive(Debug)]
pub struct BartAttention {
    num_heads: i64,
    head_dim: i64,
    dropout: Dropout,
    scaling: f64,
    encoder_decoder_attention: bool,
    output_attentions: bool,
    k_proj: nn::Linear,
    v_proj: nn::Linear,
    q_proj: nn::Linear,
    out_proj: nn::Linear,
    store_cache: bool,
}

impl BartAttention {
    pub fn new<'p, P>(
        p: P,
        embed_dim: i64,
        num_heads: i64,
        dropout: f64,
        encoder_decoder_attention: bool,
        store_cache: bool,
        output_attentions: bool,
    ) -> BartAttention
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let k_proj = nn::linear(p / "k_proj", embed_dim, embed_dim, Default::default());
        let v_proj = nn::linear(p / "v_proj", embed_dim, embed_dim, Default::default());
        let q_proj = nn::linear(p / "q_proj", embed_dim, embed_dim, Default::default());
        let out_proj = nn::linear(p / "out_proj", embed_dim, embed_dim, Default::default());

        let head_dim = embed_dim / num_heads;
        let scaling = (head_dim as f64).powf(-0.5);
        let dropout = Dropout::new(dropout);

        BartAttention {
            num_heads,
            head_dim,
            dropout,
            scaling,
            encoder_decoder_attention,
            output_attentions,
            k_proj,
            v_proj,
            q_proj,
            out_proj,
            store_cache,
        }
    }

    fn _shape(&self, x: Tensor, sequence_length: i64, batch_size: i64) -> Tensor {
        x.view((batch_size, sequence_length, self.num_heads, self.head_dim))
            .transpose(1, 2)
            .contiguous()
    }

    pub fn forward_t(
        &self,
        hidden_states: &Tensor,
        key_value_states: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        layer_state: Option<LayerState>,
        train: bool,
    ) -> (Tensor, Option<Tensor>, Option<LayerState>) {
        let (bs, target_length, embed_dim) = hidden_states.size3().unwrap();

        let query_states = hidden_states.apply(&self.q_proj) * self.scaling;

        let (key_states, value_states) = if self.encoder_decoder_attention {
            if let Some(layer_state_value) = layer_state {
                (layer_state_value.prev_key, layer_state_value.prev_value)
            } else {
                (
                    self._shape(key_value_states.unwrap().apply(&self.k_proj), -1, bs),
                    self._shape(key_value_states.unwrap().apply(&self.v_proj), -1, bs),
                )
            }
        } else if let Some(layer_state_value) = layer_state {
            let key_states = self._shape(hidden_states.apply(&self.k_proj), -1, bs);
            let value_states = self._shape(hidden_states.apply(&self.v_proj), -1, bs);
            (
                Tensor::cat(&[layer_state_value.prev_key, key_states], 2),
                Tensor::cat(&[layer_state_value.prev_value, value_states], 2),
            )
        } else {
            (
                self._shape(hidden_states.apply(&self.k_proj), -1, bs),
                self._shape(hidden_states.apply(&self.v_proj), -1, bs),
            )
        };

        let new_layer_state = if self.store_cache {
            Some(LayerState {
                prev_key: key_states.copy(),
                prev_value: value_states.copy(),
            })
        } else {
            None
        };

        let proj_shape = [bs * self.num_heads, -1, self.head_dim];
        let query_states = self
            ._shape(query_states, target_length, bs)
            .view(proj_shape);
        let key_states = key_states.view(proj_shape);
        let value_states = value_states.view(proj_shape);

        let source_length = key_states.size()[1];
        let mut attention_weights = query_states.bmm(&key_states.transpose(1, 2));

        if let Some(attention_mask_value) = attention_mask {
            attention_weights =
                attention_weights.view([bs, self.num_heads, target_length, source_length])
                    + attention_mask_value;
            attention_weights =
                attention_weights.view([bs * self.num_heads, target_length, source_length]);
        };

        attention_weights = attention_weights.softmax(-1, attention_weights.kind());

        let saved_attention_weights = if self.output_attentions {
            Some(attention_weights.view((bs, self.num_heads, target_length, source_length)))
        } else {
            None
        };

        let attention_probas = attention_weights.apply_t(&self.dropout, train);
        let attention_output = attention_probas
            .bmm(&value_states)
            .view([bs, self.num_heads, target_length, self.head_dim])
            .transpose(1, 2)
            .reshape(&[bs, target_length, embed_dim])
            .apply(&self.out_proj);

        (attention_output, saved_attention_weights, new_layer_state)
    }
}
