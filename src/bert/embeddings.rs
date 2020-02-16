// Copyright 2019-present, the HuggingFace Inc. team, The Google AI Language Team and Facebook, Inc.
// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

use tch::{nn, Tensor, Kind};
use crate::BertConfig;
use tch::nn::{EmbeddingConfig, embedding};
use crate::common::dropout::Dropout;

#[derive(Debug)]
pub struct BertEmbeddings {
    word_embeddings: nn::Embedding,
    position_embeddings: nn::Embedding,
    token_type_embeddings: nn::Embedding,
    layer_norm: nn::LayerNorm,
    dropout: Dropout,
}

impl BertEmbeddings {
    pub fn new(p: &nn::Path, config: &BertConfig) -> BertEmbeddings {
        let embedding_config = EmbeddingConfig { padding_idx: 0, ..Default::default() };

        let word_embeddings: nn::Embedding = embedding(p / "word_embeddings",
                                                       config.vocab_size,
                                                       config.hidden_size,
                                                       embedding_config);

        let position_embeddings: nn::Embedding = embedding(p / "position_embeddings",
                                                           config.max_position_embeddings,
                                                           config.hidden_size,
                                                           Default::default());

        let token_type_embeddings: nn::Embedding = embedding(p / "token_type_embeddings",
                                                             config.type_vocab_size,
                                                             config.hidden_size,
                                                             Default::default());

        let layer_norm_config = nn::LayerNormConfig { eps: 1e-12, ..Default::default() };
        let layer_norm: nn::LayerNorm = nn::layer_norm(p / "LayerNorm", vec![config.hidden_size], layer_norm_config);
        let dropout: Dropout = Dropout::new(config.hidden_dropout_prob);
        BertEmbeddings { word_embeddings, position_embeddings, token_type_embeddings, layer_norm, dropout }
    }

    pub fn forward_t(&self,
                     input_ids: Option<Tensor>,
                     token_type_ids: Option<Tensor>,
                     position_ids: Option<Tensor>,
                     input_embeds: Option<Tensor>,
                     train: bool) -> Result<Tensor, &'static str> {
        let (input_embeddings, input_shape) = match input_ids {
            Some(input_value) => match input_embeds {
                Some(_) => { return Err("Only one of input ids or input embeddings may be set"); }
                None => (input_value.apply_t(&self.word_embeddings, train), input_value.size())
            }
            None => match input_embeds {
                Some(embeds) => (embeds.copy(), vec!(embeds.size()[0], embeds.size()[1])),
                None => { return Err("Only one of input ids or input embeddings may be set"); }
            }
        };

        let seq_length = (&input_embeddings).size()[1].to_owned();

        let position_ids = match position_ids {
            Some(value) => value,
            None => Tensor::arange(seq_length, (Kind::Int64, input_embeddings.device()))
                .unsqueeze(0).
                expand(&input_shape, true)
        };

        let token_type_ids = match token_type_ids {
            Some(value) => value,
            None => Tensor::zeros(&input_shape, (Kind::Int64, input_embeddings.device()))
        };

        let position_embeddings = position_ids.apply(&self.position_embeddings);
        let token_type_embeddings = token_type_ids.apply(&self.token_type_embeddings);

        let input_embeddings: Tensor = input_embeddings + position_embeddings + token_type_embeddings;
        Ok(input_embeddings.apply(&self.layer_norm).apply_t(&self.dropout, train))
    }
}