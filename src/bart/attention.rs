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
use tch::kind::Kind::Float;
use tch::{nn, Tensor};

#[derive(Debug)]
/// # Cache for BART attention layers
/// Stores the cached value of key, value and key padding mask to avoid recalculation (e.g. at each generation step)
pub struct LayerState {
    /// Cached keys
    pub prev_key: Tensor,
    /// Cached values
    pub prev_value: Tensor,
    /// Cached keys padding mask
    pub prev_key_padding_mask: Option<Tensor>,
}

impl Clone for LayerState {
    fn clone(&self) -> Self {
        let prev_key_padding_mask = match &self.prev_key_padding_mask {
            Some(key_padding_mask) => Some(key_padding_mask.copy()),
            None => None,
        };
        LayerState {
            prev_key: self.prev_key.copy(),
            prev_value: self.prev_value.copy(),
            prev_key_padding_mask,
        }
    }
}

impl LayerState {
    pub(crate) fn reorder_cache(&mut self, new_indices: &Tensor) {
        self.prev_key = self.prev_key.index_select(0, new_indices);
        self.prev_value = self.prev_value.index_select(0, new_indices);
        if self.prev_key_padding_mask.is_some() {
            self.prev_key_padding_mask = Some(
                self.prev_key_padding_mask
                    .as_ref()
                    .unwrap()
                    .index_select(0, new_indices),
            );
        }
    }
}

#[derive(Debug)]
pub struct SelfAttention {
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

impl SelfAttention {
    pub fn new(
        p: nn::Path,
        embed_dim: i64,
        num_heads: i64,
        dropout: f64,
        encoder_decoder_attention: bool,
        store_cache: bool,
        output_attentions: bool,
    ) -> SelfAttention {
        let k_proj = nn::linear(&p / "k_proj", embed_dim, embed_dim, Default::default());
        let v_proj = nn::linear(&p / "v_proj", embed_dim, embed_dim, Default::default());
        let q_proj = nn::linear(&p / "q_proj", embed_dim, embed_dim, Default::default());
        let out_proj = nn::linear(&p / "out_proj", embed_dim, embed_dim, Default::default());

        let head_dim = embed_dim / num_heads;
        let scaling = (head_dim as f64).powf(-0.5);
        let dropout = Dropout::new(dropout);

        SelfAttention {
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

    fn flatten(&self, x: Tensor, dim_0: i64, bs: i64) -> Tensor {
        x.contiguous()
            .view((dim_0, bs * self.num_heads, self.head_dim))
            .transpose(0, 1)
    }

    pub fn forward_t(
        &self,
        query: &Tensor,
        key: Option<&Tensor>,
        key_padding_mask: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        mut layer_state: Option<LayerState>,
        train: bool,
    ) -> (Tensor, Option<Tensor>, Option<LayerState>) {
        let query_size = query.size();
        let (target_sequence_length, bs) = (query_size[0], query_size[1]);
        let q: Tensor = self.flatten(
            query.as_ref().apply(&self.q_proj) * self.scaling,
            target_sequence_length,
            bs,
        );
        let key = match &layer_state {
            Some(_) => {
                if self.encoder_decoder_attention {
                    None
                } else {
                    key
                }
            }
            None => key,
        };

        let (k, v) = if self.encoder_decoder_attention {
            match key {
                Some(key) => (
                    Some(self.flatten(key.apply(&self.k_proj), -1, bs)),
                    Some(self.flatten(key.apply(&self.v_proj), -1, bs)),
                ),
                None => (None, None),
            }
        } else {
            (
                Some(self.flatten(query.apply(&self.k_proj), -1, bs)),
                Some(self.flatten(query.apply(&self.v_proj), -1, bs)),
            )
        };

        let (k, v, key_padding_mask) =
            self.use_saved_state(&layer_state, k, v, key_padding_mask, bs);

        let source_sequence_length = k.size()[1];
        let attention_weights = q.bmm(&k.transpose(1, 2));
        let attention_weights = match attention_mask {
            Some(mask) => {
                let attention_weights = attention_weights.view((
                    bs,
                    self.num_heads,
                    target_sequence_length,
                    source_sequence_length,
                )) + mask;
                attention_weights.view((
                    bs * self.num_heads,
                    target_sequence_length,
                    source_sequence_length,
                ))
            }
            None => attention_weights,
        };

        let attention_weights = match key_padding_mask.as_ref() {
            Some(mask) => attention_weights
                .view((
                    bs,
                    self.num_heads,
                    target_sequence_length,
                    source_sequence_length,
                ))
                .masked_fill(&mask.unsqueeze(1).unsqueeze(2), std::f64::NEG_INFINITY)
                .view((
                    bs * self.num_heads,
                    target_sequence_length,
                    source_sequence_length,
                )),
            None => attention_weights,
        };

        let attention_weights = attention_weights.softmax(-1, Float);
        let attention_probabilities = attention_weights.apply_t(&self.dropout, train);
        let output = attention_probabilities
            .bmm(&v)
            .transpose(0, 1)
            .contiguous()
            .view((target_sequence_length, bs, self.num_heads * self.head_dim))
            .apply(&self.out_proj);

        let attention_weights = if self.output_attentions {
            Some(attention_weights.view((
                bs,
                self.num_heads,
                target_sequence_length,
                source_sequence_length,
            )))
        } else {
            None
        };

        if self.store_cache {
            if layer_state.is_some() {
                layer_state.as_mut().unwrap().prev_key =
                    k.view((bs, self.num_heads, -1, self.head_dim));
                layer_state.as_mut().unwrap().prev_value =
                    v.view((bs, self.num_heads, -1, self.head_dim));
                layer_state.as_mut().unwrap().prev_key_padding_mask = match key_padding_mask {
                    Some(tensor) => Some(tensor),
                    None => None,
                };
            } else {
                layer_state = Some(LayerState {
                    prev_key: k.view((bs, self.num_heads, -1, self.head_dim)),
                    prev_value: v.view((bs, self.num_heads, -1, self.head_dim)),
                    prev_key_padding_mask: match key_padding_mask {
                        Some(tensor) => Some(tensor),
                        None => None,
                    },
                })
            };
        };

        (output, attention_weights, layer_state)
    }

    fn use_saved_state(
        &self,
        layer_state: &Option<LayerState>,
        k: Option<Tensor>,
        v: Option<Tensor>,
        key_padding_mask: Option<&Tensor>,
        bs: i64,
    ) -> (Tensor, Tensor, Option<Tensor>) {
        match &layer_state {
            Some(prev_state) => {
                let prev_key = prev_state
                    .prev_key
                    .view((bs * self.num_heads, -1, self.head_dim));
                let prev_value =
                    prev_state
                        .prev_value
                        .view((bs * self.num_heads, -1, self.head_dim));
                let k = if self.encoder_decoder_attention {
                    prev_key
                } else {
                    Tensor::cat(&[prev_key, k.unwrap()], 1)
                };
                let v = if self.encoder_decoder_attention {
                    prev_value
                } else {
                    Tensor::cat(&[prev_value, v.unwrap()], 1)
                };

                let key_padding_mask = self.use_saved_key_padding_mask(
                    key_padding_mask,
                    &prev_state.prev_key_padding_mask,
                    bs,
                    k.size()[1],
                );
                (k, v, key_padding_mask)
            }
            None => {
                let key_padding_mask = match key_padding_mask {
                    Some(value) => Some(value.copy()),
                    None => None,
                };
                (k.unwrap(), v.unwrap(), key_padding_mask)
            }
        }
    }

    fn use_saved_key_padding_mask(
        &self,
        key_padding_mask: Option<&Tensor>,
        prev_key_padding_mask: &Option<Tensor>,
        bs: i64,
        sequence_length: i64,
    ) -> Option<Tensor> {
        if prev_key_padding_mask.is_some() {
            if self.encoder_decoder_attention {
                Some(prev_key_padding_mask.as_ref().unwrap().copy())
            } else {
                Some(Tensor::cat(
                    &[
                        prev_key_padding_mask.as_ref().unwrap(),
                        key_padding_mask.as_ref().unwrap(),
                    ],
                    1,
                ))
            }
        } else {
            match key_padding_mask {
                Some(key_padding_mask) => {
                    let filler = Tensor::zeros(
                        &[bs, sequence_length - key_padding_mask.size()[1]],
                        (key_padding_mask.kind(), key_padding_mask.device()),
                    );
                    Some(Tensor::cat(&[filler, key_padding_mask.copy()], 1))
                }
                None => None,
            }
        }
    }
}
