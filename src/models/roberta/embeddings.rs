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

use crate::bert::{BertConfig, BertEmbedding};
use crate::common::dropout::Dropout;
use crate::common::embeddings::process_ids_embeddings_pair;
use crate::RustBertError;
use std::borrow::Borrow;
use tch::nn::{embedding, EmbeddingConfig};
use tch::{nn, Kind, Tensor};

#[derive(Debug)]
/// # BertEmbeddings implementation for RoBERTa model
/// Implementation of the `BertEmbedding` trait for RoBERTa models
pub struct RobertaEmbeddings {
    word_embeddings: nn::Embedding,
    position_embeddings: nn::Embedding,
    token_type_embeddings: nn::Embedding,
    layer_norm: nn::LayerNorm,
    dropout: Dropout,
    padding_index: i64,
}

impl RobertaEmbeddings {
    fn create_position_ids_from_input_ids(&self, x: &Tensor) -> Tensor {
        let mask: Tensor = x.ne(self.padding_index).to_kind(Kind::Int64);
        mask.cumsum(1, Kind::Int64) * mask + self.padding_index
    }

    fn create_position_ids_from_embeddings(&self, x: &Tensor) -> Tensor {
        let input_shape = x.size();
        let input_shape = vec![input_shape[0], input_shape[1]];
        let position_ids: Tensor = Tensor::arange_start(
            self.padding_index + 1,
            input_shape[0],
            (Kind::Int64, x.device()),
        );
        position_ids.unsqueeze(0).expand(&input_shape, true)
    }
}

impl BertEmbedding for RobertaEmbeddings {
    /// Build a new `RobertaEmbeddings`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the BertEmbeddings model
    /// * `config` - `BertConfig` object defining the model architecture and vocab/hidden size
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::bert::{BertConfig, BertEmbedding};
    /// use rust_bert::roberta::RobertaEmbeddings;
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = BertConfig::from_file(config_path);
    /// let robert_embeddings = RobertaEmbeddings::new(&p.root() / "bert_embeddings", &config);
    /// ```
    fn new<'p, P>(p: P, config: &BertConfig) -> RobertaEmbeddings
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let embedding_config = EmbeddingConfig {
            padding_idx: 1,
            ..Default::default()
        };

        let word_embeddings: nn::Embedding = embedding(
            p / "word_embeddings",
            config.vocab_size,
            config.hidden_size,
            embedding_config,
        );

        let position_embeddings: nn::Embedding = embedding(
            p / "position_embeddings",
            config.max_position_embeddings,
            config.hidden_size,
            Default::default(),
        );

        let token_type_embeddings: nn::Embedding = embedding(
            p / "token_type_embeddings",
            config.type_vocab_size,
            config.hidden_size,
            Default::default(),
        );

        let layer_norm_config = nn::LayerNormConfig {
            eps: 1e-12,
            ..Default::default()
        };
        let layer_norm: nn::LayerNorm =
            nn::layer_norm(p / "LayerNorm", vec![config.hidden_size], layer_norm_config);
        let dropout: Dropout = Dropout::new(config.hidden_dropout_prob);
        RobertaEmbeddings {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
            dropout,
            padding_index: 1,
        }
    }

    /// Forward pass through the embedding layer.
    /// This differs from the original BERT embeddings in how the position ids are calculated when not provided.
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see *input_embeds*)
    /// * `token_type_ids` -Optional segment id of shape (*batch size*, *sequence_length*). Convention is value of 0 for the first sentence (incl. *SEP*) and 1 for the second sentence. If None set to 0.
    /// * `position_ids` - Optional position ids of shape (*batch size*, *sequence_length*). If None, will be incremented from 0.
    /// * `input_embeds` - Optional pre-computed input embeddings of shape (*batch size*, *sequence_length*, *hidden_size*). If None, input ids must be provided (see *input_ids*)
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `embedded_output` - `Tensor` of shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rust_bert::bert::{BertConfig, BertEmbedding};
    /// # use tch::{nn, Device, Tensor, no_grad};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::Int64;
    /// use rust_bert::roberta::RobertaEmbeddings;
    /// # let config_path = Path::new("path/to/config.json");
    /// # let vocab_path = Path::new("path/to/vocab.txt");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = BertConfig::from_file(config_path);
    /// # let roberta_embeddings = RobertaEmbeddings::new(&vs.root(), &config);
    /// let (batch_size, sequence_length) = (64, 128);
    /// let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    /// let token_type_ids = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    /// let position_ids = Tensor::arange(sequence_length, (Int64, device))
    ///     .expand(&[batch_size, sequence_length], true);
    ///
    /// let embedded_output = no_grad(|| {
    ///     roberta_embeddings
    ///         .forward_t(
    ///             Some(&input_tensor),
    ///             Some(&token_type_ids),
    ///             Some(&position_ids),
    ///             None,
    ///             false,
    ///         )
    ///         .unwrap()
    /// });
    /// ```
    fn forward_t(
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

        let calc_position_ids = if position_ids.is_none() {
            Some(match input_ids {
                Some(value) => self.create_position_ids_from_input_ids(value),
                None => self.create_position_ids_from_embeddings(input_embeds.unwrap()),
            })
        } else {
            None
        };

        let calc_token_type_ids = if token_type_ids.is_none() {
            Some(Tensor::zeros(
                input_shape,
                (Kind::Int64, input_embeddings.device()),
            ))
        } else {
            None
        };

        let position_ids = position_ids.unwrap_or_else(|| calc_position_ids.as_ref().unwrap());
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
