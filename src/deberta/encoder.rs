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

use crate::common::activations::TensorFunction;
use crate::common::dropout::XDropout;
use crate::deberta::attention::DebertaAttention;
use crate::deberta::deberta_model::DebertaLayerNorm;
use crate::deberta::DebertaConfig;
use crate::RustBertError;
use std::borrow::Borrow;
use tch::{nn, Tensor};

pub struct DebertaIntermediate {
    dense: nn::Linear,
    activation: TensorFunction,
}

impl DebertaIntermediate {
    pub fn new<'p, P>(p: P, config: &DebertaConfig) -> DebertaIntermediate
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let dense = nn::linear(
            p / "dense",
            config.hidden_size,
            config.intermediate_size,
            Default::default(),
        );
        let activation = config.hidden_act.get_function();
        DebertaIntermediate { dense, activation }
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Tensor {
        (self.activation.get_fn())(&hidden_states.apply(&self.dense))
    }
}

pub struct DebertaOutput {
    dense: nn::Linear,
    layer_norm: DebertaLayerNorm,
    dropout: XDropout,
}

impl DebertaOutput {
    pub fn new<'p, P>(p: P, config: &DebertaConfig) -> DebertaOutput
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let dense = nn::linear(
            p / "dense",
            config.intermediate_size,
            config.hidden_size,
            Default::default(),
        );

        let layer_norm = DebertaLayerNorm::new(
            p / "LayerNorm",
            config.hidden_size,
            config.layer_norm_eps.unwrap_or(1e-7),
        );
        let dropout = XDropout::new(config.hidden_dropout_prob);

        DebertaOutput {
            dense,
            layer_norm,
            dropout,
        }
    }

    pub fn forward_t(&self, hidden_states: &Tensor, input_tensor: &Tensor, train: bool) -> Tensor {
        let hidden_states: Tensor = input_tensor
            + hidden_states
                .apply(&self.dense)
                .apply_t(&self.dropout, train);
        hidden_states.apply(&self.layer_norm)
    }
}

pub struct DebertaLayer {
    attention: DebertaAttention,
    intermediate: DebertaIntermediate,
    output: DebertaOutput,
}

impl DebertaLayer {
    pub fn new<'p, P>(p: P, config: &DebertaConfig) -> DebertaLayer
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let attention = DebertaAttention::new(p / "attention", config);
        let intermediate = DebertaIntermediate::new(p / "intermediate", config);
        let output = DebertaOutput::new(p / "output", config);

        DebertaLayer {
            attention,
            intermediate,
            output,
        }
    }

    pub fn forward_t(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
        query_states: Option<&Tensor>,
        relative_pos: Option<&Tensor>,
        relative_embeddings: Option<&Tensor>,
        train: bool,
    ) -> Result<(Tensor, Option<Tensor>), RustBertError> {
        let (attention_output, attention_matrix) = self.attention.forward_t(
            hidden_states,
            attention_mask,
            query_states,
            relative_pos,
            relative_embeddings,
            train,
        )?;

        let intermediate_output = self.intermediate.forward(&attention_output);
        let layer_output = self
            .output
            .forward_t(&intermediate_output, &attention_output, train);

        Ok((layer_output, attention_matrix))
    }
}
