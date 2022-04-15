// Copyright 2020 The Microsoft Authors and The HuggingFace Inc. team.
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

use crate::prophetnet::ProphetNetConfig;
use std::borrow::Borrow;
use tch::nn::{Embedding, EmbeddingConfig};
use tch::{nn, Device, Kind, Tensor};

pub struct ProphetNetPositionalEmbeddings {
    embeddings: Embedding,
    padding_idx: i64,
}

impl ProphetNetPositionalEmbeddings {
    pub fn new<'p, P>(p: P, config: &ProphetNetConfig) -> ProphetNetPositionalEmbeddings
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let embeddings_config = EmbeddingConfig {
            padding_idx: config.pad_token_id,
            ..Default::default()
        };
        let embeddings = nn::embedding(
            p,
            config.max_position_embeddings,
            config.hidden_size,
            embeddings_config,
        );
        ProphetNetPositionalEmbeddings {
            embeddings,
            padding_idx: config.pad_token_id,
        }
    }

    pub fn forward(
        &self,
        input_shape: &[i64],
        device: Device,
        attention_mask: Option<&Tensor>,
        prev_num_input_ids: Option<i64>,
        position_ids: Option<&Tensor>,
    ) -> (Tensor, Tensor) {
        let calc_position_ids = match position_ids {
            None => {
                if let Some(prev_num_input_ids_value) = prev_num_input_ids {
                    let num_input_ids = input_shape[1] + prev_num_input_ids_value;

                    Tensor::ones(&[1, 1], (Kind::Int64, device))
                        * (self.padding_idx + num_input_ids)
                } else {
                    let calc_attention_mask = if attention_mask.is_none() {
                        Some(Tensor::ones(input_shape, (Kind::Int64, device)))
                    } else {
                        None
                    };
                    let attention_mask =
                        attention_mask.unwrap_or_else(|| calc_attention_mask.as_ref().unwrap());
                    attention_mask.cumsum(1, Kind::Int64) * attention_mask + self.padding_idx
                }
            }
            Some(value) => value.copy(),
        };

        (calc_position_ids.apply(&self.embeddings), calc_position_ids)
    }

    pub fn _forward(&self, position_ids: &Tensor) -> Tensor {
        position_ids.apply(&self.embeddings)
    }
}
