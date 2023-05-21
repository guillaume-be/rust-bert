// Copyright 2021, Google and The HuggingFace Inc. team. All rights reserved.
// Copyright 2021 Guillaume Becquin
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
use tch::nn::embedding;
use tch::{nn, Device, Kind, Tensor};

#[derive(Debug)]
pub struct SinusoidalPositionalEmbedding {
    embedding: nn::Embedding,
}

impl SinusoidalPositionalEmbedding {
    pub fn new<'p, P>(
        p: P,
        num_embeddings: i64,
        embedding_dim: i64,
    ) -> SinusoidalPositionalEmbedding
    where
        P: Borrow<nn::Path<'p>>,
    {
        let device = p.borrow().device();
        let mut local_varstore = nn::VarStore::new(device);

        let mut embedding: nn::Embedding = embedding(
            local_varstore.root(),
            num_embeddings,
            embedding_dim,
            Default::default(),
        );

        embedding.ws = SinusoidalPositionalEmbedding::build_positional_embeddings(
            num_embeddings,
            embedding_dim,
            device,
        );

        local_varstore.freeze();
        SinusoidalPositionalEmbedding { embedding }
    }

    pub fn build_positional_embeddings(
        num_embeddings: i64,
        embedding_dim: i64,
        device: Device,
    ) -> Tensor {
        let mut sinusoidal_embedding: Vec<Tensor> = Vec::with_capacity(num_embeddings as usize);
        let sentinel = embedding_dim / 2 + embedding_dim % 2;
        for pos in 0..num_embeddings {
            let mut temp_vec: Vec<f64> = Vec::with_capacity(embedding_dim as usize);
            for j in 0..embedding_dim {
                let base_value =
                    pos as f64 / 10000_f64.powf((2 * (j / 2)) as f64 / embedding_dim as f64);
                if j % 2 == 0 {
                    temp_vec.push(base_value.sin());
                } else {
                    temp_vec.push(base_value.cos());
                }
            }
            let temp_vec = Tensor::from_slice(&temp_vec);

            sinusoidal_embedding.push(temp_vec);
        }
        let sinusoidal_embeddings = Tensor::stack(&sinusoidal_embedding, 0).to_kind(Kind::Float);

        let reordered_sinusoidal_embeddings =
            Tensor::empty([num_embeddings, embedding_dim], (Kind::Float, device));

        reordered_sinusoidal_embeddings
            .slice(1, 0, sentinel, 1)
            .copy_(&sinusoidal_embeddings.slice(1, 0, embedding_dim, 2));
        reordered_sinusoidal_embeddings
            .slice(1, sentinel, embedding_dim, 1)
            .copy_(&sinusoidal_embeddings.slice(1, 1, embedding_dim, 2));
        reordered_sinusoidal_embeddings.to_kind(Kind::Half)
    }

    pub fn forward(&self, input: &Tensor, past_key_values_length: i64) -> Tensor {
        let input_shape = input.size();
        let (_, sequence_length) = (input_shape[0], input_shape[1]);
        let positions = Tensor::arange_start(
            past_key_values_length,
            past_key_values_length + sequence_length,
            (Kind::Int64, input.device()),
        );
        positions.apply(&self.embedding)
    }
}
