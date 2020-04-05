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

use tch::{nn, Tensor};
use tch::nn::{EmbeddingConfig, embedding};
use tch::kind::Kind::Int64;


#[derive(Debug)]
pub struct PositionalEmbedding {
    embedding: nn::Embedding,
    padding_index: i64,
}

impl PositionalEmbedding {
    pub fn new(p: nn::Path, num_embeddings: i64, embedding_dim: i64, padding_index: i64) -> PositionalEmbedding {
        let embedding_config = EmbeddingConfig { padding_idx: padding_index, ..Default::default() };
        let num_embeddings = num_embeddings + padding_index + 1;

        let embedding: nn::Embedding = embedding(p,
                                                 num_embeddings,
                                                 embedding_dim,
                                                 embedding_config);
        PositionalEmbedding { embedding, padding_index }
    }

    pub fn forward(&self, input: &Tensor, generation_mode: bool) -> Tensor {
        let positions = if generation_mode {
            let positions = self.padding_index + input.size()[1];
            input.new_full(&[1, 1], positions, (Int64, input.device()))
        } else {
            self.create_position_ids_from_input_ids(input, self.padding_index)
        };
        positions.apply(&self.embedding)
    }

    fn create_position_ids_from_input_ids(&self, input_ids: &Tensor, padding_index: i64) -> Tensor {
        let mask = input_ids.ne(padding_index).to_kind(Int64);
        let position_ids: Tensor = mask.cumsum(1, Int64) * mask + padding_index;
        position_ids
    }
}
