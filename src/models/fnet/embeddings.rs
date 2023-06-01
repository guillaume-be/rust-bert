// Copyright 2021 Google Research
// Copyright 2020-present, the HuggingFace Inc. team.
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

use crate::common::dropout::Dropout;
use crate::common::embeddings::process_ids_embeddings_pair;
use crate::fnet::FNetConfig;
use crate::RustBertError;
use std::borrow::Borrow;
use tch::nn::{EmbeddingConfig, LayerNormConfig};
use tch::{nn, Kind, Tensor};

pub struct FNetEmbeddings {
    word_embeddings: nn::Embedding,
    position_embeddings: nn::Embedding,
    token_type_embeddings: nn::Embedding,
    projection: nn::Linear,
    layer_norm: nn::LayerNorm,
    dropout: Dropout,
}

impl FNetEmbeddings {
    pub fn new<'p, P>(p: P, config: &FNetConfig) -> FNetEmbeddings
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let word_embeddings_config = EmbeddingConfig {
            padding_idx: config.pad_token_id.unwrap_or(3),
            ..Default::default()
        };
        let word_embeddings = nn::embedding(
            p / "word_embeddings",
            config.vocab_size,
            config.hidden_size,
            word_embeddings_config,
        );
        let position_embeddings = nn::embedding(
            p / "position_embeddings",
            config.max_position_embeddings,
            config.hidden_size,
            Default::default(),
        );
        let token_type_embeddings = nn::embedding(
            p / "token_type_embeddings",
            config.type_vocab_size,
            config.hidden_size,
            Default::default(),
        );

        let layer_norm_config = LayerNormConfig {
            eps: config.layer_norm_eps.unwrap_or(1e-12),
            ..Default::default()
        };
        let layer_norm =
            nn::layer_norm(p / "LayerNorm", vec![config.hidden_size], layer_norm_config);

        let projection = nn::linear(
            p / "projection",
            config.hidden_size,
            config.hidden_size,
            Default::default(),
        );

        let dropout = Dropout::new(config.hidden_dropout_prob);
        FNetEmbeddings {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            projection,
            layer_norm,
            dropout,
        }
    }

    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        input_embeddings: Option<&Tensor>,
        train: bool,
    ) -> Result<Tensor, RustBertError> {
        let (calc_input_embeddings, input_shape, _) =
            process_ids_embeddings_pair(input_ids, input_embeddings, &self.word_embeddings)?;

        let input_embeddings =
            input_embeddings.unwrap_or_else(|| calc_input_embeddings.as_ref().unwrap());

        let calc_token_type_ids = if token_type_ids.is_none() {
            Some(Tensor::zeros(
                input_shape.as_slice(),
                (Kind::Int64, input_embeddings.device()),
            ))
        } else {
            None
        };
        let token_type_embeddings = token_type_ids
            .unwrap_or_else(|| calc_token_type_ids.as_ref().unwrap())
            .apply(&self.token_type_embeddings);

        let calc_position_ids = if position_ids.is_none() {
            Some(Tensor::arange(
                input_shape[1],
                (Kind::Int64, input_embeddings.device()),
            ))
        } else {
            None
        };

        let position_embeddings = position_ids
            .unwrap_or_else(|| calc_position_ids.as_ref().unwrap())
            .apply(&self.position_embeddings);

        let embeddings = input_embeddings + token_type_embeddings + position_embeddings;
        Ok(embeddings
            .apply(&self.layer_norm)
            .apply(&self.projection)
            .apply_t(&self.dropout, train))
    }
}
