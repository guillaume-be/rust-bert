// Copyright 2020, Microsoft and the HuggingFace Inc. team.
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

use crate::common::dropout::XDropout;
use crate::common::embeddings::process_ids_embeddings_pair;
use crate::deberta::deberta_model::DebertaLayerNorm;
use crate::deberta::{BaseDebertaLayerNorm, DebertaConfig};
use crate::RustBertError;
use std::borrow::Borrow;
use tch::nn::{EmbeddingConfig, Module};
use tch::{nn, Kind, Tensor};

pub struct BaseDebertaEmbeddings<LN>
where
    LN: BaseDebertaLayerNorm + Module,
{
    word_embeddings: nn::Embedding,
    position_embeddings: Option<nn::Embedding>,
    token_type_embeddings: Option<nn::Embedding>,
    embed_proj: Option<nn::Linear>,
    layer_norm: LN,
    dropout: XDropout,
}

impl<LN> BaseDebertaEmbeddings<LN>
where
    LN: BaseDebertaLayerNorm + Module,
{
    pub fn new<'p, P>(p: P, config: &DebertaConfig) -> BaseDebertaEmbeddings<LN>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let embedding_config = EmbeddingConfig {
            padding_idx: config.pad_token_id.unwrap_or(0),
            ..Default::default()
        };
        let embedding_size = config.embedding_size.unwrap_or(config.hidden_size);

        let word_embeddings = nn::embedding(
            p / "word_embeddings",
            config.vocab_size,
            embedding_size,
            embedding_config,
        );

        let position_embeddings = if config.position_biased_input.unwrap_or(true) {
            Some(nn::embedding(
                p / "position_embeddings",
                config.max_position_embeddings,
                embedding_size,
                Default::default(),
            ))
        } else {
            None
        };

        let token_type_embeddings = if config.type_vocab_size > 0 {
            Some(nn::embedding(
                p / "token_type_embeddings",
                config.type_vocab_size,
                embedding_size,
                Default::default(),
            ))
        } else {
            None
        };

        let embed_proj = if embedding_size != config.hidden_size {
            let linear_config = nn::LinearConfig {
                bias: false,
                ..Default::default()
            };
            Some(nn::linear(
                p / "embed_proj",
                embedding_size,
                config.hidden_size,
                linear_config,
            ))
        } else {
            None
        };

        let layer_norm = LN::new(
            p / "LayerNorm",
            embedding_size,
            config.layer_norm_eps.unwrap_or(1e-7),
        );
        let dropout = XDropout::new(config.hidden_dropout_prob);
        BaseDebertaEmbeddings {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            embed_proj,
            layer_norm,
            dropout,
        }
    }

    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        attention_mask: &Tensor,
        input_embeds: Option<&Tensor>,
        train: bool,
    ) -> Result<Tensor, RustBertError> {
        let (calc_input_embeddings, input_shape, _) =
            process_ids_embeddings_pair(input_ids, input_embeds, &self.word_embeddings)?;

        let mut input_embeddings = input_embeds
            .unwrap_or_else(|| calc_input_embeddings.as_ref().unwrap())
            .shallow_clone();
        let seq_length = input_embeddings.size()[1];

        let calc_position_ids = if position_ids.is_none() {
            Some(
                Tensor::arange(seq_length, (Kind::Int64, input_embeddings.device()))
                    .expand(&[1, -1], true),
            )
        } else {
            None
        };

        let calc_token_type_ids = if token_type_ids.is_none() {
            Some(Tensor::zeros(
                &input_shape,
                (Kind::Int64, input_embeddings.device()),
            ))
        } else {
            None
        };

        let position_ids = position_ids.unwrap_or_else(|| calc_position_ids.as_ref().unwrap());
        let token_type_ids =
            token_type_ids.unwrap_or_else(|| calc_token_type_ids.as_ref().unwrap());

        if let Some(position_embeddings) = &self.position_embeddings {
            let position_embeddings = position_ids.apply(position_embeddings);
            input_embeddings = input_embeddings + position_embeddings;
        };

        if let Some(token_type_embeddings) = &self.token_type_embeddings {
            let token_type_embeddings = token_type_ids.apply(token_type_embeddings);
            input_embeddings = input_embeddings + token_type_embeddings;
        };

        if let Some(embed_proj) = &self.embed_proj {
            input_embeddings = input_embeddings.apply(embed_proj);
        };

        input_embeddings = input_embeddings.apply(&self.layer_norm);

        let mask = if attention_mask.dim() != input_embeddings.dim() {
            if attention_mask.dim() != 4 {
                attention_mask
                    .squeeze_dim(1)
                    .squeeze_dim(1)
                    .unsqueeze(2)
                    .to_kind(input_embeddings.kind())
            } else {
                attention_mask.unsqueeze(2).to_kind(input_embeddings.kind())
            }
        } else {
            attention_mask.to_kind(input_embeddings.kind())
        };
        input_embeddings = input_embeddings * mask;

        Ok(input_embeddings.apply_t(&self.dropout, train))
    }
}

pub type DebertaEmbeddings = BaseDebertaEmbeddings<DebertaLayerNorm>;
