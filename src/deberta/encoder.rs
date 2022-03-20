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
use crate::deberta::attention::{build_relative_position, DebertaAttention};
use crate::deberta::deberta_model::{BaseDebertaLayerNorm, DebertaLayerNorm};
use crate::deberta::{DebertaConfig, DebertaDisentangledSelfAttention, DisentangledSelfAttention};
use crate::RustBertError;
use std::borrow::{Borrow, BorrowMut};
use tch::nn::Module;
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

pub struct DebertaOutput<LN: BaseDebertaLayerNorm + Module> {
    dense: nn::Linear,
    layer_norm: LN,
    dropout: XDropout,
}

impl<LN: BaseDebertaLayerNorm + Module> DebertaOutput<LN> {
    pub fn new<'p, P>(p: P, config: &DebertaConfig) -> DebertaOutput<LN>
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

        let layer_norm = LN::new(
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

pub struct BaseDebertaLayer<SA, LN>
where
    SA: DisentangledSelfAttention,
    LN: BaseDebertaLayerNorm + Module,
{
    attention: DebertaAttention<SA, LN>,
    intermediate: DebertaIntermediate,
    output: DebertaOutput<LN>,
}

impl<SA, LN> BaseDebertaLayer<SA, LN>
where
    SA: DisentangledSelfAttention,
    LN: BaseDebertaLayerNorm + Module,
{
    pub fn new<'p, P>(p: P, config: &DebertaConfig) -> BaseDebertaLayer<SA, LN>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let attention = DebertaAttention::new(p / "attention", config);
        let intermediate = DebertaIntermediate::new(p / "intermediate", config);
        let output = DebertaOutput::new(p / "output", config);

        BaseDebertaLayer {
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

pub type DebertaLayer = BaseDebertaLayer<DebertaDisentangledSelfAttention, DebertaLayerNorm>;

pub struct DebertaEncoder {
    output_attentions: bool,
    output_hidden_states: bool,
    layers: Vec<DebertaLayer>,
    rel_embeddings: Option<nn::Embedding>,
}

impl DebertaEncoder {
    pub fn new<'p, P>(p: P, config: &DebertaConfig) -> DebertaEncoder
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let output_attentions = config.output_attentions.unwrap_or(false);
        let output_hidden_states = config.output_hidden_states.unwrap_or(false);

        let p_layer = p / "layer";
        let mut layers: Vec<DebertaLayer> = vec![];
        for layer_index in 0..config.num_hidden_layers {
            layers.push(DebertaLayer::new(&p_layer / layer_index, config));
        }

        let relative_attention = config.relative_attention.unwrap_or(false);
        let rel_embeddings = if relative_attention {
            let mut max_relative_positions = config.max_relative_positions.unwrap_or(-1);
            if max_relative_positions < 1 {
                max_relative_positions = config.max_position_embeddings;
            }
            Some(nn::embedding(
                p / "rel_embeddings",
                max_relative_positions * 2,
                config.hidden_size,
                Default::default(),
            ))
        } else {
            None
        };

        DebertaEncoder {
            output_attentions,
            output_hidden_states,
            layers,
            rel_embeddings,
        }
    }

    fn get_attention_mask(&self, attention_mask: &Tensor) -> Tensor {
        if attention_mask.dim() <= 2 {
            let extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2);
            &extended_attention_mask
                * &extended_attention_mask
                    .squeeze_dim(-2)
                    .unsqueeze(-1)
                    .internal_cast_byte(true)
        } else if attention_mask.dim() == 3 {
            attention_mask.unsqueeze(1)
        } else {
            attention_mask.shallow_clone()
        }
    }

    fn get_rel_pos(
        &self,
        hidden_states: &Tensor,
        query_states: Option<&Tensor>,
        relative_pos: Option<&Tensor>,
    ) -> Option<Tensor> {
        if self.rel_embeddings.is_some() & relative_pos.is_none() {
            let mut query_size = query_states.unwrap_or(hidden_states).size();
            query_size.reverse();
            let query_size = query_size[1];
            let mut key_size = hidden_states.size();
            key_size.reverse();
            let key_size = key_size[1];
            Some(build_relative_position(
                query_size,
                key_size,
                hidden_states.device(),
            ))
        } else {
            relative_pos.map(|tensor| tensor.shallow_clone())
        }
    }

    pub fn forward_t(
        &self,
        input: &Tensor,
        attention_mask: &Tensor,
        query_states: Option<&Tensor>,
        relative_pos: Option<&Tensor>,
        train: bool,
    ) -> Result<DebertaEncoderOutput, RustBertError> {
        let mut all_hidden_states: Option<Vec<Tensor>> = if self.output_hidden_states {
            Some(vec![])
        } else {
            None
        };
        let mut all_attentions: Option<Vec<Tensor>> = if self.output_attentions {
            Some(vec![])
        } else {
            None
        };

        let attention_mask = self.get_attention_mask(attention_mask);
        let relative_pos = self.get_rel_pos(input, query_states, relative_pos);
        let relative_embeddings = self
            .rel_embeddings
            .as_ref()
            .map(|embeddings| &embeddings.ws);

        let mut hidden_state = None::<Tensor>;
        let mut attention_weights: Option<Tensor>;

        for layer in &self.layers {
            let layer_output = if let Some(hidden_state) = &hidden_state {
                layer.forward_t(
                    hidden_state,
                    &attention_mask,
                    query_states,
                    relative_pos.as_ref(),
                    relative_embeddings,
                    train,
                )?
            } else {
                layer.forward_t(
                    input,
                    &attention_mask,
                    query_states,
                    relative_pos.as_ref(),
                    relative_embeddings,
                    train,
                )?
            };

            hidden_state = Some(layer_output.0);
            attention_weights = layer_output.1;
            if let Some(attentions) = all_attentions.borrow_mut() {
                attentions.push(attention_weights.as_ref().unwrap().copy());
            };
            if let Some(hidden_states) = all_hidden_states.borrow_mut() {
                hidden_states.push(hidden_state.as_ref().unwrap().copy());
            };
        }

        Ok(DebertaEncoderOutput {
            hidden_state: hidden_state.unwrap(),
            all_hidden_states,
            all_attentions,
        })
    }
}

/// Container for the DeBERTa encoder output.
pub struct DebertaEncoderOutput {
    /// Last hidden states from the model
    pub hidden_state: Tensor,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all intermediate layers
    pub all_attentions: Option<Vec<Tensor>>,
}
