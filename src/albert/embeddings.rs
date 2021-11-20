// Copyright 2018 Google AI and Google Brain team.
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

use crate::albert::AlbertConfig;
use crate::common::dropout::Dropout;
use crate::common::embeddings::process_ids_embeddings_pair;
use crate::RustBertError;
use std::borrow::Borrow;
use tch::nn::{embedding, EmbeddingConfig};
use tch::{nn, Kind, Tensor};

/// # Embeddings implementation for Albert model
#[derive(Debug)]
pub struct AlbertEmbeddings {
    word_embeddings: nn::Embedding,
    position_embeddings: nn::Embedding,
    token_type_embeddings: nn::Embedding,
    layer_norm: nn::LayerNorm,
    dropout: Dropout,
}

impl AlbertEmbeddings {
    pub fn new<'p, P>(p: P, config: &AlbertConfig) -> AlbertEmbeddings
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let embedding_config = EmbeddingConfig {
            padding_idx: config.pad_token_id,
            ..Default::default()
        };

        let word_embeddings: nn::Embedding = embedding(
            p / "word_embeddings",
            config.vocab_size,
            config.embedding_size,
            embedding_config,
        );

        let position_embeddings: nn::Embedding = embedding(
            p / "position_embeddings",
            config.max_position_embeddings,
            config.embedding_size,
            Default::default(),
        );

        let token_type_embeddings: nn::Embedding = embedding(
            p / "token_type_embeddings",
            config.type_vocab_size,
            config.embedding_size,
            Default::default(),
        );

        let layer_norm_eps = config.layer_norm_eps.unwrap_or(1e-12);
        let layer_norm_config = nn::LayerNormConfig {
            eps: layer_norm_eps,
            ..Default::default()
        };
        let layer_norm: nn::LayerNorm = nn::layer_norm(
            p / "LayerNorm",
            vec![config.embedding_size],
            layer_norm_config,
        );
        let dropout: Dropout = Dropout::new(config.hidden_dropout_prob);
        AlbertEmbeddings {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
            dropout,
        }
    }

    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        train: bool,
    ) -> Result<Tensor, RustBertError> {
        let (calc_input_embeddings, input_shape, _) =
            process_ids_embeddings_pair(input_ids, input_embeds, &self.word_embeddings)?;
        let input_embeddings =
            input_embeds.unwrap_or_else(|| calc_input_embeddings.as_ref().unwrap());
        let seq_length = input_embeddings.as_ref().size()[1].to_owned();

        let calc_position_ids = if position_ids.is_none() {
            Some(
                Tensor::arange(seq_length, (Kind::Int64, input_embeddings.device()))
                    .unsqueeze(0)
                    .expand(&input_shape, true),
            )
        } else {
            None
        };
        let position_ids = position_ids.unwrap_or_else(|| calc_position_ids.as_ref().unwrap());

        let calc_token_type_ids = if token_type_ids.is_none() {
            Some(Tensor::zeros(
                &input_shape,
                (Kind::Int64, input_embeddings.device()),
            ))
        } else {
            None
        };
        let token_type_ids =
            token_type_ids.unwrap_or_else(|| calc_token_type_ids.as_ref().unwrap());

        let position_embeddings = position_ids.apply(&self.position_embeddings);
        let token_type_embeddings = token_type_ids.apply(&self.token_type_embeddings);

        let input_embeddings: Tensor =
            input_embeddings + position_embeddings + token_type_embeddings;
        Ok(input_embeddings
            .apply(&self.layer_norm)
            .apply_t(&self.dropout, train))
    }
}
