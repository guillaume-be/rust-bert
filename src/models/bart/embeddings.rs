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

use std::borrow::Borrow;
use tch::nn::embedding;
use tch::{nn, Kind, Tensor};

/// # Abstraction that holds a embeddings configuration
pub enum EmbeddingOption {
    /// PositionalEmbedding
    LearnedPositionalEmbedding(LearnedPositionalEmbedding),
    SinusoidalPositionalEmbedding(SinusoidalPositionalEmbedding),
}

impl EmbeddingOption {
    /// Interface method to forward_t() of the particular models.
    pub fn forward(&self, input: &Tensor, past_key_values_length: i64) -> Tensor {
        match *self {
            Self::LearnedPositionalEmbedding(ref embeddings) => {
                embeddings.forward(input, past_key_values_length)
            }
            Self::SinusoidalPositionalEmbedding(ref embeddings) => {
                embeddings.forward(input, past_key_values_length)
            }
        }
    }
}

#[derive(Debug)]
pub struct LearnedPositionalEmbedding {
    embedding: nn::Embedding,
    offset: i64,
}

impl LearnedPositionalEmbedding {
    pub fn new<'p, P>(p: P, num_embeddings: i64, embedding_dim: i64) -> LearnedPositionalEmbedding
    where
        P: Borrow<nn::Path<'p>>,
    {
        let offset = 2;

        let num_embeddings = num_embeddings + offset;

        let embedding: nn::Embedding = embedding(
            p.borrow(),
            num_embeddings,
            embedding_dim,
            Default::default(),
        );
        LearnedPositionalEmbedding { embedding, offset }
    }

    pub fn forward(&self, input: &Tensor, past_key_values_length: i64) -> Tensor {
        let input_shape = input.size();
        let (_, sequence_length) = (input_shape[0], input_shape[1]);
        let positions = Tensor::arange_start(
            past_key_values_length,
            past_key_values_length + sequence_length,
            (Kind::Int64, input.device()),
        ) + self.offset;
        positions.apply(&self.embedding)
    }
}

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
        let embedding: nn::Embedding = embedding(
            p.borrow(),
            num_embeddings,
            embedding_dim,
            Default::default(),
        );
        SinusoidalPositionalEmbedding { embedding }
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
