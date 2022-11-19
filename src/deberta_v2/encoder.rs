// Copyright 2020, Microsoft and the HuggingFace Inc. team.
// Copyright 2022 Guillaume Becquin
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
use crate::deberta::{BaseDebertaLayer, BaseDebertaLayerNorm, DebertaEncoderOutput};
use crate::deberta_v2::attention::{build_relative_position, DebertaV2DisentangledSelfAttention};
use crate::deberta_v2::deberta_v2_model::NormRelEmbedType;
use crate::deberta_v2::DebertaV2Config;
use crate::{Activation, RustBertError};
use std::borrow::{Borrow, BorrowMut};
use tch::nn::{ConvConfig, LayerNorm, LayerNormConfig, Path};
use tch::{nn, Kind, Tensor};

pub type DebertaV2Layer = BaseDebertaLayer<DebertaV2DisentangledSelfAttention, LayerNorm>;

pub struct ConvLayer {
    conv: nn::Conv1D,
    layer_norm: nn::LayerNorm,
    dropout: XDropout,
    conv_act: TensorFunction,
}

impl ConvLayer {
    pub fn new<'p, P>(p: P, config: &DebertaV2Config) -> ConvLayer
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let conv_act = config.conv_act.unwrap_or(Activation::tanh).get_function();
        let kernel_size = config.conv_kernel_size.unwrap_or(3);
        let groups = config.conv_groups.unwrap_or(1);

        let conv_config = ConvConfig {
            padding: (kernel_size - 1) / 2,
            groups,
            ..Default::default()
        };
        let conv = nn::conv1d(
            p / "conv",
            config.hidden_size,
            config.hidden_size,
            kernel_size,
            conv_config,
        );

        let layer_norm_config = LayerNormConfig {
            eps: config.layer_norm_eps.unwrap_or(1e-7),
            ..Default::default()
        };
        let layer_norm =
            nn::layer_norm(p / "LayerNorm", vec![config.hidden_size], layer_norm_config);

        let dropout = XDropout::new(config.hidden_dropout_prob);

        ConvLayer {
            conv,
            layer_norm,
            dropout,
            conv_act,
        }
    }

    pub fn forward_t(
        &self,
        hidden_states: &Tensor,
        residual_states: &Tensor,
        input_mask: &Tensor,
        train: bool,
    ) -> Tensor {
        let out = hidden_states
            .permute(&[0, 2, 1])
            .contiguous()
            .apply(&self.conv)
            .permute(&[0, 2, 1])
            .contiguous();
        let reverse_mask: Tensor = 1 - input_mask;
        let out = out.masked_fill(
            &reverse_mask
                .to_kind(Kind::Bool)
                .unsqueeze(-1)
                .expand(out.size().as_slice(), true),
            0,
        );
        let out = self.conv_act.get_fn()(&out.apply_t(&self.dropout, train));

        let layer_norm_input = residual_states + out;
        let output = layer_norm_input.apply(&self.layer_norm);
        let new_input_mask = if input_mask.dim() != layer_norm_input.dim() {
            if input_mask.dim() == 4 {
                input_mask.squeeze_dim(1).squeeze_dim(1).unsqueeze(2)
            } else {
                input_mask.unsqueeze(2)
            }
            .to_kind(output.kind())
        } else {
            input_mask.to_kind(output.kind())
        };
        output * new_input_mask
    }
}

impl BaseDebertaLayerNorm for LayerNorm {
    fn new<'p, P>(p: P, size: i64, variance_epsilon: f64) -> Self
    where
        P: Borrow<Path<'p>>,
    {
        let layer_norm_config = nn::LayerNormConfig {
            eps: variance_epsilon,
            ..Default::default()
        };

        nn::layer_norm(p, vec![size], layer_norm_config)
    }
}

pub struct DebertaV2Encoder {
    output_attentions: bool,
    output_hidden_states: bool,
    layers: Vec<DebertaV2Layer>,
    max_relative_positions: Option<i64>,
    position_buckets: Option<i64>,
    rel_embeddings: Option<nn::Embedding>,
    layer_norm: Option<nn::LayerNorm>,
    conv: Option<ConvLayer>,
}

impl DebertaV2Encoder {
    pub fn new<'p, P>(p: P, config: &DebertaV2Config) -> DebertaV2Encoder
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let output_attentions = config.output_attentions.unwrap_or(false);
        let output_hidden_states = config.output_hidden_states.unwrap_or(false);
        let p_layer = p / "layer";
        let mut layers: Vec<DebertaV2Layer> = vec![];
        for layer_index in 0..config.num_hidden_layers {
            layers.push(DebertaV2Layer::new(
                &p_layer / layer_index,
                &(config.into()),
            ));
        }
        let (rel_embeddings, max_relative_positions, position_buckets) =
            if config.relative_attention.unwrap_or(false) {
                let mut max_relative_positions = config.max_relative_positions.unwrap_or(-1);
                if max_relative_positions < 1 {
                    max_relative_positions = config.max_position_embeddings;
                };
                let position_buckets = config.position_buckets.unwrap_or(-1);
                let position_embed_size = if position_buckets > 0 {
                    position_buckets * 2
                } else {
                    max_relative_positions * 2
                };
                let rel_embeddings = nn::embedding(
                    p / "rel_embeddings",
                    position_embed_size,
                    config.hidden_size,
                    Default::default(),
                );
                (
                    Some(rel_embeddings),
                    Some(max_relative_positions),
                    Some(position_buckets),
                )
            } else {
                (None, None, None)
            };

        let layer_norm = if config
            .norm_rel_ebd
            .clone()
            .unwrap_or_default()
            .has_type(NormRelEmbedType::layer_norm)
        {
            Some(nn::layer_norm(
                p / "LayerNorm",
                vec![config.hidden_size],
                LayerNormConfig {
                    eps: 1e-7,
                    elementwise_affine: true,
                    ..Default::default()
                },
            ))
        } else {
            None
        };

        let conv = if config.conv_kernel_size.unwrap_or(0) > 0 {
            Some(ConvLayer::new(p / "conv", config))
        } else {
            None
        };

        DebertaV2Encoder {
            output_attentions,
            output_hidden_states,
            layers,
            max_relative_positions,
            position_buckets,
            rel_embeddings,
            layer_norm,
            conv,
        }
    }

    fn get_rel_embedding(&self) -> Option<Tensor> {
        self.rel_embeddings.as_ref().map(|embeddings| {
            let rel_embeds = &embeddings.ws;
            if let Some(layer_norm) = &self.layer_norm {
                rel_embeds.apply(layer_norm)
            } else {
                rel_embeds.shallow_clone()
            }
        })
    }

    fn get_attention_mask(attention_mask: &Tensor) -> Tensor {
        match attention_mask.dim() {
            value if value <= 2 => {
                let extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2);
                extended_attention_mask.as_ref()
                    * extended_attention_mask
                        .squeeze_dim(-2)
                        .unsqueeze(-1)
                        .to_kind(Kind::Uint8)
            }
            value if value == 3 => attention_mask.unsqueeze(1),
            _ => attention_mask.shallow_clone(),
        }
    }

    fn reverse_vec<T>(mut input_vec: Vec<T>) -> Vec<T> {
        input_vec.reverse();
        input_vec
    }

    fn get_rel_pos(
        &self,
        hidden_states: &Tensor,
        query_states: Option<&Tensor>,
        relative_pos: Option<&Tensor>,
    ) -> Option<Tensor> {
        if self.rel_embeddings.is_some() & relative_pos.is_none() {
            let q = query_states
                .map(|query_states| DebertaV2Encoder::reverse_vec(query_states.size())[1])
                .unwrap_or_else(|| DebertaV2Encoder::reverse_vec(hidden_states.size())[1]);

            Some(build_relative_position(
                q,
                DebertaV2Encoder::reverse_vec(hidden_states.size())[1],
                self.position_buckets.unwrap(),
                self.max_relative_positions.unwrap(),
                hidden_states.device(),
            ))
        } else {
            relative_pos.map(|tensor| tensor.shallow_clone())
        }
    }

    pub fn forward_t(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
        query_states: Option<&Tensor>,
        relative_pos: Option<&Tensor>,
        train: bool,
    ) -> Result<DebertaV2EncoderOutput, RustBertError> {
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

        let input_mask = if attention_mask.dim() <= 2 {
            attention_mask.shallow_clone()
        } else {
            attention_mask
                .sum_dim_intlist([-2].as_slice(), false, attention_mask.kind())
                .gt(0)
                .to_kind(Kind::Uint8)
        };
        let attention_mask = Self::get_attention_mask(attention_mask);
        let relative_pos = self.get_rel_pos(hidden_states, query_states, relative_pos);
        let relative_embeddings = self.get_rel_embedding();

        let mut output_states = None::<Tensor>;
        let mut attention_weights: Option<Tensor>;

        for (layer_index, layer) in self.layers.iter().enumerate() {
            let layer_output = if let Some(output_states) = &output_states {
                layer.forward_t(
                    output_states,
                    &attention_mask,
                    query_states,
                    relative_pos.as_ref(),
                    relative_embeddings.as_ref(),
                    train,
                )?
            } else {
                layer.forward_t(
                    hidden_states,
                    &attention_mask,
                    query_states,
                    relative_pos.as_ref(),
                    relative_embeddings.as_ref(),
                    train,
                )?
            };

            output_states = Some(layer_output.0);
            if layer_index == 0 {
                if let Some(conv) = &self.conv {
                    output_states = output_states.map(|output_states| {
                        conv.forward_t(hidden_states, &output_states, &input_mask, train)
                    })
                }
            }
            attention_weights = layer_output.1;
            if let Some(attentions) = all_attentions.borrow_mut() {
                attentions.push(std::mem::take(&mut attention_weights.unwrap()));
            };
            if let Some(hidden_states) = all_hidden_states.borrow_mut() {
                hidden_states.push(output_states.as_ref().unwrap().copy());
            };
        }

        Ok(DebertaEncoderOutput {
            hidden_state: output_states.unwrap(),
            all_hidden_states,
            all_attentions,
        })
    }
}

/// Container for the DeBERTa V2 encoder output.
pub type DebertaV2EncoderOutput = DebertaEncoderOutput;
