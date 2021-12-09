// Copyright 2020 The Trax Authors and The HuggingFace Inc. team.
// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

use crate::common::dropout::Dropout;
use crate::common::embeddings::process_ids_embeddings_pair;
use crate::reformer::ReformerConfig;
use crate::RustBertError;
use std::borrow::Borrow;
use tch::nn::Init;
use tch::{nn, Kind, Tensor};

#[derive(Debug)]
/// # Axial position embeddings implementation for Reformer model
pub struct AxialPositionEmbeddings {
    weights: Vec<Tensor>,
    axial_pos_shape: Vec<i64>,
    dropout_prob: f64,
}

impl AxialPositionEmbeddings {
    pub fn new<'p, P>(p: P, config: &ReformerConfig) -> Result<Self, RustBertError>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let axial_pos_shape = config.axial_pos_shape.clone();

        if config.axial_pos_embds_dim.iter().sum::<i64>() != config.hidden_size {
            return Err(RustBertError::InvalidConfigurationError(format!(
                "The sum of position embedding dimensions ({:?}) does not add up to the hidden size {}",
                config.axial_pos_embds_dim,
                config.hidden_size
            )));
        };

        let mut weights: Vec<Tensor> = vec![];
        let p_weights = p / "weights";
        for (axis_index, axial_pos_embd_dim) in config.axial_pos_embds_dim.iter().enumerate() {
            let mut axial_shape = vec![1i64; config.axial_pos_shape.len()];
            axial_shape[axis_index] = config.axial_pos_shape[axis_index];
            axial_shape.push(*axial_pos_embd_dim);
            weights.push(p_weights.var(&axis_index.to_string(), &axial_shape, Init::Const(1.0)));
        }

        Ok(AxialPositionEmbeddings {
            weights,
            axial_pos_shape,
            dropout_prob: config.hidden_dropout_prob,
        })
    }

    pub fn forward_t(&self, position_ids: &Tensor, train: bool) -> Tensor {
        let input_shape = position_ids.size();
        let (batch_size, sequence_length) = (input_shape[0], input_shape[1]);

        let broadcasted_weights = self
            .weights
            .iter()
            .map(|tensor| {
                let mut new_shape = vec![batch_size];
                new_shape.extend(&self.axial_pos_shape);
                new_shape.push(*tensor.size().last().unwrap());
                tensor.expand(new_shape.as_slice(), true)
            })
            .collect::<Vec<Tensor>>();

        if train {
            if self.dropout_prob > 0.0 {
                Tensor::cat(&broadcasted_weights, -1)
                    .transpose(2, 1)
                    .feature_dropout(self.dropout_prob, train)
                    .transpose(2, 1)
                    .reshape(&[batch_size, sequence_length, -1])
            } else {
                Tensor::cat(
                    &broadcasted_weights
                        .iter()
                        .map(|tensor| tensor.reshape(&[batch_size, sequence_length, -1]))
                        .collect::<Vec<Tensor>>(),
                    -1,
                )
            }
        } else {
            let max_position_id = i64::from(position_ids.max());
            let required_pos_encodings_columns =
                (max_position_id + 1) / self.axial_pos_shape[1] + 1;
            let position_encodings = Tensor::cat(
                &broadcasted_weights
                    .iter()
                    .map(|tensor| tensor.slice(1, 0, required_pos_encodings_columns, 1))
                    .collect::<Vec<Tensor>>(),
                -1,
            );
            let position_encodings = position_encodings.reshape(&[
                batch_size,
                -1,
                *position_encodings.size().last().unwrap(),
            ]);

            let mut output_tensors = vec![];
            for i in 0..batch_size {
                output_tensors.push(
                    position_encodings
                        .get(i)
                        .index_select(0, &position_ids.get(i))
                        .unsqueeze(0),
                );
            }
            Tensor::cat(&output_tensors, 0)
        }
    }
}

#[derive(Debug)]
/// # Position embeddings implementation for Reformer model
pub struct BasePositionEmbeddings {
    embeddings: nn::Embedding,
    dropout: Dropout,
}

impl BasePositionEmbeddings {
    pub fn new<'p, P>(p: P, config: &ReformerConfig) -> BasePositionEmbeddings
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let dropout = Dropout::new(config.hidden_dropout_prob);
        let embeddings = nn::embedding(
            p / "embedding",
            config.max_position_embeddings,
            config.hidden_size,
            Default::default(),
        );

        BasePositionEmbeddings {
            embeddings,
            dropout,
        }
    }

    pub fn forward_t(&self, position_ids: &Tensor, train: bool) -> Tensor {
        position_ids
            .apply(&self.embeddings)
            .apply_t(&self.dropout, train)
    }
}

#[derive(Debug)]
pub enum PositionEmbedding {
    AxialPositionEmbeddings(AxialPositionEmbeddings),
    BasePositionEmbeddings(BasePositionEmbeddings),
}

impl PositionEmbedding {
    pub fn forward_t(&self, position_ids: &Tensor, train: bool) -> Tensor {
        match self {
            PositionEmbedding::AxialPositionEmbeddings(ref embeddings) => {
                embeddings.forward_t(position_ids, train)
            }
            PositionEmbedding::BasePositionEmbeddings(ref embeddings) => {
                embeddings.forward_t(position_ids, train)
            }
        }
    }
}

#[derive(Debug)]
/// # Embeddings implementation for Reformer model
pub struct ReformerEmbeddings {
    position_embeddings: PositionEmbedding,
    word_embeddings: nn::Embedding,
    dropout: Dropout,
}

impl ReformerEmbeddings {
    pub fn new<'p, P>(p: P, config: &ReformerConfig) -> Result<ReformerEmbeddings, RustBertError>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let dropout = Dropout::new(config.hidden_dropout_prob);
        let word_embeddings = nn::embedding(
            p / "word_embeddings",
            config.vocab_size,
            config.hidden_size,
            Default::default(),
        );

        let position_embeddings = if config.axial_pos_embds {
            PositionEmbedding::AxialPositionEmbeddings(AxialPositionEmbeddings::new(
                p / "position_embeddings",
                config,
            )?)
        } else {
            PositionEmbedding::BasePositionEmbeddings(BasePositionEmbeddings::new(
                p / "position_embeddings",
                config,
            ))
        };

        Ok(ReformerEmbeddings {
            position_embeddings,
            word_embeddings,
            dropout,
        })
    }

    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        start_ids_pos_encoding: i64,
        train: bool,
    ) -> Result<Tensor, RustBertError> {
        let (calc_input_embeddings, input_shape, device) =
            process_ids_embeddings_pair(input_ids, input_embeds, &self.word_embeddings)?;
        let input_embeddings =
            input_embeds.unwrap_or_else(|| calc_input_embeddings.as_ref().unwrap());

        let calc_position_ids = if position_ids.is_none() {
            Some(
                Tensor::arange_start(
                    start_ids_pos_encoding,
                    start_ids_pos_encoding + input_shape[1],
                    (Kind::Int64, device),
                )
                .unsqueeze(0)
                .expand(&input_shape, true),
            )
        } else {
            None
        };

        let position_ids = position_ids.unwrap_or_else(|| calc_position_ids.as_ref().unwrap());

        Ok(self.position_embeddings.forward_t(position_ids, train)
            + input_embeddings.apply_t(&self.dropout, train))
    }
}
