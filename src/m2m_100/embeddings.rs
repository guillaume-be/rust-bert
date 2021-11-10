// Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
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

use std::borrow::Borrow;
use std::ops::Deref;
use std::sync::RwLock;
use tch::nn::embedding;
use tch::{nn, Device, Kind, Tensor};

#[derive(Debug)]
pub struct SinusoidalPositionalEmbedding {
    embedding: RwLock<nn::Embedding>,
    embedding_dim: i64,
    padding_idx: i64,
    offset: i64,
}

impl SinusoidalPositionalEmbedding {
    pub fn new<'p, P>(
        p: P,
        num_embeddings: i64,
        embedding_dim: i64,
        padding_idx: i64,
    ) -> SinusoidalPositionalEmbedding
    where
        P: Borrow<nn::Path<'p>>,
    {
        let device = p.borrow().device();
        let mut local_varstore = nn::VarStore::new(device);
        let offset = 2;

        let mut embedding = embedding(
            local_varstore.root(),
            num_embeddings + offset,
            embedding_dim,
            Default::default(),
        );

        embedding
            .ws
            .set_data(&SinusoidalPositionalEmbedding::build_positional_embeddings(
                num_embeddings + offset,
                embedding_dim,
                padding_idx,
                device,
            ));

        local_varstore.freeze();
        SinusoidalPositionalEmbedding {
            embedding: RwLock::new(embedding),
            embedding_dim,
            padding_idx,
            offset,
        }
    }

    fn build_positional_embeddings(
        num_embeddings: i64,
        embedding_dim: i64,
        padding_idx: i64,
        device: Device,
    ) -> Tensor {
        let half_dim = embedding_dim / 2;

        let emb = -(10000f64.ln() as f64) / ((half_dim - 1) as f64);
        let emb = (Tensor::arange(half_dim, (Kind::Float, device)) * emb).exp();
        let emb =
            Tensor::arange(num_embeddings, (Kind::Float, device)).unsqueeze(1) * emb.unsqueeze(0);
        let mut sinusoidal_embedding =
            Tensor::cat(&[&emb.sin(), &emb.cos()], 1).view([num_embeddings, -1]);

        if embedding_dim % 2 == 1 {
            sinusoidal_embedding = Tensor::cat(
                &[
                    sinusoidal_embedding,
                    Tensor::zeros(&[num_embeddings, 1], (Kind::Float, device)),
                ],
                1,
            );
        }
        let _ = sinusoidal_embedding.select(0, padding_idx).fill_(0);

        let _ = sinusoidal_embedding.requires_grad_(false);
        sinusoidal_embedding
    }

    fn create_position_ids_from_input_ids(
        &self,
        input_ids: &Tensor,
        past_key_values_length: i64,
    ) -> Tensor {
        let mask = input_ids.ne(self.padding_idx).to_kind(Kind::Int64);
        let incremental_indices = (mask.cumsum(1, Kind::Int64) + past_key_values_length) * mask;
        incremental_indices + self.padding_idx
    }

    pub fn forward(&self, input_ids: &Tensor, past_key_values_length: i64, kind: Kind) -> Tensor {
        let position_ids =
            self.create_position_ids_from_input_ids(input_ids, past_key_values_length);
        let input_size = input_ids.size();
        let seq_length = input_size[1];

        let max_pos = self.padding_idx + 1 + seq_length;
        let current_size = self.embedding.read().unwrap().ws.size()[0];
        if max_pos > current_size {
            self.embedding.write().unwrap().ws.set_data(
                &SinusoidalPositionalEmbedding::build_positional_embeddings(
                    max_pos + self.offset,
                    self.embedding_dim,
                    self.padding_idx,
                    input_ids.device(),
                ),
            );
        }
        let current_kind = self.embedding.read().unwrap().ws.kind();
        if current_kind != kind {
            let new_embeddings = &self.embedding.read().unwrap().ws.to_kind(kind);
            self.embedding.write().unwrap().ws.set_data(new_embeddings);
        }
        position_ids.apply(self.embedding.read().unwrap().deref())
    }
}
