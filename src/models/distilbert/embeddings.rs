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

use crate::common::dropout::Dropout;
use crate::common::embeddings::process_ids_embeddings_pair;
use crate::distilbert::distilbert_model::DistilBertConfig;
use crate::RustBertError;
use std::borrow::Borrow;
use tch::kind::Kind::Float;
use tch::nn::{embedding, EmbeddingConfig, Init, VarStore};
use tch::{nn, Device, Kind, Tensor};

fn create_sinusoidal_embeddings<'p, P>(
    config: &DistilBertConfig,
    p: P,
    device: Device,
) -> nn::Embedding
where
    P: Borrow<nn::Path<'p>>,
{
    let mut sinusoidal_embedding: Vec<Tensor> =
        Vec::with_capacity(config.max_position_embeddings as usize);
    for pos in 0..config.max_position_embeddings {
        let mut temp_vec: Vec<f64> = Vec::with_capacity(config.dim as usize);
        for j in 0..config.dim {
            if j % 2 == 0 {
                temp_vec.push(
                    (pos as f64 / 10000_f64.powf((2 * (j / 2)) as f64 / config.dim as f64)).sin(),
                );
            } else {
                temp_vec.push(
                    (pos as f64 / 10000_f64.powf((2 * (j / 2)) as f64 / config.dim as f64)).cos(),
                );
            }
        }
        let temp_vec = Tensor::from_slice(&temp_vec);
        sinusoidal_embedding.push(temp_vec);
    }
    let sinusoidal_embedding = Tensor::stack(&sinusoidal_embedding, 0)
        .to_kind(Float)
        .to_device(device);

    let p = p.borrow();
    let mut updated_weights = p.var(
        "weight",
        &[config.max_position_embeddings, config.dim],
        Init::Const(0.),
    );
    tch::no_grad(|| {
        updated_weights.copy_(&sinusoidal_embedding);
    });

    let embedding_config = EmbeddingConfig {
        padding_idx: 0,
        ..Default::default()
    };
    let mut embeddings = embedding(
        VarStore::new(Device::Cpu).root(),
        config.max_position_embeddings,
        config.dim,
        embedding_config,
    );

    embeddings.ws = updated_weights;
    embeddings
}

#[derive(Debug)]
pub struct DistilBertEmbedding {
    word_embeddings: nn::Embedding,
    position_embeddings: nn::Embedding,
    layer_norm: nn::LayerNorm,
    dropout: Dropout,
}

impl DistilBertEmbedding {
    pub fn new<'p, P>(p: P, config: &DistilBertConfig) -> DistilBertEmbedding
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let embedding_config = EmbeddingConfig {
            padding_idx: 0,
            ..Default::default()
        };

        let word_embeddings: nn::Embedding = embedding(
            p / "word_embeddings",
            config.vocab_size,
            config.dim,
            embedding_config,
        );
        let position_embeddings: nn::Embedding = match config.sinusoidal_pos_embds {
            false => embedding(
                p / "position_embeddings",
                config.max_position_embeddings,
                config.dim,
                embedding_config,
            ),
            true => create_sinusoidal_embeddings(config, p / "position_embeddings", p.device()),
        };
        let layer_norm_config = nn::LayerNormConfig {
            eps: 1e-12,
            ..Default::default()
        };
        let layer_norm: nn::LayerNorm =
            nn::layer_norm(p / "LayerNorm", vec![config.dim], layer_norm_config);
        let dropout: Dropout = Dropout::new(config.dropout);
        DistilBertEmbedding {
            word_embeddings,
            position_embeddings,
            layer_norm,
            dropout,
        }
    }

    pub fn _get_word_embeddings(&self) -> &nn::Embedding {
        &self.word_embeddings
    }

    pub fn _set_word_embeddings(&mut self, new_embeddings: nn::Embedding) {
        self.word_embeddings = new_embeddings;
    }

    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        train: bool,
    ) -> Result<Tensor, RustBertError> {
        let (calc_input_embeddings, input_size, device) =
            process_ids_embeddings_pair(input_ids, input_embeds, &self.word_embeddings)?;
        let word_embeds = input_embeds.unwrap_or_else(|| calc_input_embeddings.as_ref().unwrap());

        let seq_length = input_size[1];
        let position_ids = Tensor::arange(seq_length, (Kind::Int64, device));
        let position_ids = position_ids
            .unsqueeze(0)
            .expand(input_size.as_slice(), true);
        let position_embed = position_ids.apply(&self.position_embeddings);

        let embeddings = word_embeds + position_embed;
        Ok(embeddings
            .apply(&self.layer_norm)
            .apply_t(&self.dropout, train))
    }
}
