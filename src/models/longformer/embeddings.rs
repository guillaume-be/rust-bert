// Copyright 2020 The Allen Institute for AI team and The HuggingFace Inc. team.
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
use crate::longformer::LongformerConfig;
use crate::RustBertError;
use std::borrow::Borrow;
use tch::nn::EmbeddingConfig;
use tch::{nn, Kind, Tensor};

pub struct LongformerEmbeddings {
    word_embeddings: nn::Embedding,
    position_embeddings: nn::Embedding,
    token_type_embeddings: nn::Embedding,
    layer_norm: nn::LayerNorm,
    dropout: Dropout,
    pad_token_id: i64,
}

impl LongformerEmbeddings {
    pub fn new<'p, P>(p: P, config: &LongformerConfig) -> LongformerEmbeddings
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let pad_token_id = config.pad_token_id.unwrap_or(1);

        let embeddings_config = EmbeddingConfig {
            padding_idx: pad_token_id,
            ..Default::default()
        };
        let word_embeddings = nn::embedding(
            p / "word_embeddings",
            config.vocab_size,
            config.hidden_size,
            embeddings_config,
        );

        let position_embeddings = nn::embedding(
            p / "position_embeddings",
            config.max_position_embeddings,
            config.hidden_size,
            embeddings_config,
        );

        let token_type_embeddings = nn::embedding(
            p / "token_type_embeddings",
            config.type_vocab_size,
            config.hidden_size,
            Default::default(),
        );

        let layer_norm_config = nn::LayerNormConfig {
            eps: config.layer_norm_eps.unwrap_or(1e-12),
            ..Default::default()
        };
        let layer_norm =
            nn::layer_norm(p / "LayerNorm", vec![config.hidden_size], layer_norm_config);
        let dropout = Dropout::new(config.hidden_dropout_prob);

        LongformerEmbeddings {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
            dropout,
            pad_token_id,
        }
    }

    fn create_position_ids_from_input_ids(&self, input_ids: &Tensor) -> Tensor {
        let mask = input_ids.ne(self.pad_token_id);
        mask.cumsum(1, Kind::Int64) * mask + self.pad_token_id
    }

    fn create_position_ids_from_input_embeds(&self, inputs_embeds: &Tensor) -> Tensor {
        let input_shape = inputs_embeds.size();
        let (batch_size, sequence_length) = (input_shape[0], input_shape[1]);

        Tensor::arange_start(
            self.pad_token_id + 1,
            sequence_length + self.pad_token_id + 1,
            (Kind::Int64, inputs_embeds.device()),
        )
        .unsqueeze(0)
        .expand([batch_size, sequence_length], true)
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
        let input_embeds = input_embeds.unwrap_or_else(|| calc_input_embeddings.as_ref().unwrap());

        let calc_position_ids = if position_ids.is_none() {
            if let Some(input_ids) = input_ids {
                Some(self.create_position_ids_from_input_ids(input_ids))
            } else {
                Some(self.create_position_ids_from_input_embeds(input_embeds))
            }
        } else {
            None
        };
        let position_ids = position_ids.unwrap_or_else(|| calc_position_ids.as_ref().unwrap());

        let calc_token_type_ids = if token_type_ids.is_none() {
            Some(Tensor::zeros(
                input_shape.as_slice(),
                (Kind::Int64, input_embeds.device()),
            ))
        } else {
            None
        };
        let token_type_ids =
            token_type_ids.unwrap_or_else(|| calc_token_type_ids.as_ref().unwrap());

        let position_embeddings = position_ids.apply(&self.position_embeddings);
        let token_type_embeddings = token_type_ids.apply(&self.token_type_embeddings);
        Ok((input_embeds + position_embeddings + token_type_embeddings)
            .apply(&self.layer_norm)
            .apply_t(&self.dropout, train))
    }
}
