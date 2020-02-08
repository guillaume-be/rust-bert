// Copyright 2019-present, the HuggingFace Inc. team, The Google AI Language Team and Facebook, Inc.
// Copyright 2019 Guillaume Becquin
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use tch::{nn, Tensor, Kind, Device};
use tch::nn::{ModuleT, embedding, EmbeddingConfig};
use crate::distilbert::distilbert::DistilBertConfig;
use crate::distilbert::dropout::Dropout;


fn create_sinusoidal_embeddings(config: &DistilBertConfig, device: Device) -> nn::Embedding {
    let sinusoidal_embedding = Tensor::arange(config.max_position_embeddings, (Kind::Float, device)).unsqueeze(1);
    let multiplier: Tensor = Tensor::arange2(0, config.dim, 2, (Kind::Float, device));
    let multiplier: Tensor = Tensor::from(1.0) / (Tensor::ones(&[1], (Kind::Float, device)) * 10000).pow1(&(multiplier / config.dim));
    let sinusoidal_embedding: Tensor = sinusoidal_embedding * multiplier;
    let cos_embeddings: Tensor = sinusoidal_embedding.cos();
    let sin_embeddings: Tensor = sinusoidal_embedding.sin();

    let sinusoidal_embedding: Tensor = Tensor::ones(&[config.max_position_embeddings, config.dim], (Kind::Float, device));
    sinusoidal_embedding.slice(1, 0, config.dim, 2).copy_(&sin_embeddings);
    sinusoidal_embedding.slice(1, 1, config.dim, 2).copy_(&cos_embeddings);

    let embedding_config = EmbeddingConfig { padding_idx: 0, ..Default::default() };
    let mut embeddings = embedding(&nn::VarStore::new(device).root(),
                                   config.max_position_embeddings,
                                   config.dim,
                                   embedding_config);
    embeddings.ws = sinusoidal_embedding;
    embeddings
}


#[derive(Debug)]
pub struct BertEmbedding {
    word_embeddings: nn::Embedding,
    position_embeddings: nn::Embedding,
    layer_norm: nn::LayerNorm,
    dropout: Dropout,
}

impl BertEmbedding {
    pub fn new(p: &nn::Path, config: &DistilBertConfig) -> BertEmbedding {
        let embedding_config = EmbeddingConfig { padding_idx: 0, ..Default::default() };

        let word_embeddings: nn::Embedding = embedding(p / "word_embeddings",
                                                       config.vocab_size,
                                                       config.dim,
                                                       embedding_config);
        let position_embeddings: nn::Embedding = match config.sinusoidal_pos_embds {
            false => embedding(p / "position_embeddings",
                               config.max_position_embeddings,
                               config.dim,
                               embedding_config),

            true => create_sinusoidal_embeddings(&config, p.device())
        };
        let layer_norm_config = nn::LayerNormConfig { eps: 1e-12, ..Default::default() };
        let layer_norm: nn::LayerNorm = nn::layer_norm(p / "LayerNorm", vec![config.dim], layer_norm_config);
        let dropout: Dropout = Dropout::new(config.dropout);
        BertEmbedding { word_embeddings, position_embeddings, layer_norm, dropout }
    }

    pub fn _get_word_embeddings(&self) -> &nn::Embedding {
        &self.word_embeddings
    }

    pub fn _set_word_embeddings(&mut self, new_embeddings: nn::Embedding) {
        self.word_embeddings = new_embeddings;
    }
}

impl ModuleT for BertEmbedding {
    fn forward_t(&self, input: &Tensor, train: bool) -> Tensor {
        let seq_length = (&input).size().last().unwrap().to_owned();
        let position_ids = Tensor::arange(seq_length, (Kind::Int64, input.device()));
        let position_ids = position_ids.unsqueeze(0).expand_as(input);

        let word_embed = input.apply(&self.word_embeddings);
        let position_embed = position_ids.apply(&self.position_embeddings);

        let embeddings = word_embed + position_embed;
        let embeddings = embeddings.apply(&self.layer_norm).apply_t(&self.dropout, train);

        embeddings
    }
}