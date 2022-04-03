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
use crate::deberta::deberta_model::{x_softmax, PositionAttentionType, PositionAttentionTypes};
use crate::deberta::{BaseDebertaLayerNorm, DebertaConfig};
use crate::RustBertError;
use std::borrow::Borrow;
use tch::nn::{Init, Module};
use tch::{nn, Device, Kind, Tensor};

pub trait DisentangledSelfAttention {
    fn new<'p, P>(p: P, config: &DebertaConfig) -> Self
    where
        P: Borrow<nn::Path<'p>>;

    fn forward_t(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
        query_states: Option<&Tensor>,
        relative_pos: Option<&Tensor>,
        relative_embeddings: Option<&Tensor>,
        train: bool,
    ) -> Result<(Tensor, Option<Tensor>), RustBertError>;
}

pub fn build_relative_position(query_size: i64, key_size: i64, device: Device) -> Tensor {
    let q_ids = Tensor::arange(query_size, (Kind::Int64, device));
    let k_ids = Tensor::arange(key_size, (Kind::Int64, device));
    let rel_pos_ids = q_ids.unsqueeze(-1) - k_ids.view([1, -1]).repeat(&[query_size, 1]);
    rel_pos_ids.slice(0, 0, query_size, 1).unsqueeze(0)
}

pub struct DebertaDisentangledSelfAttention {
    in_proj: nn::Linear,
    q_bias: Tensor,
    v_bias: Tensor,
    num_attention_heads: i64,
    head_logits_proj: Option<nn::Linear>,
    head_weights_proj: Option<nn::Linear>,
    pos_proj: Option<nn::Linear>,
    pos_q_proj: Option<nn::Linear>,
    pos_att_type: PositionAttentionTypes,
    max_relative_positions: Option<i64>,
    pos_dropout: Option<XDropout>,
    dropout: XDropout,
    output_attentions: bool,
}

impl DebertaDisentangledSelfAttention {
    fn transpose_for_scores(&self, x: &Tensor) -> Tensor {
        let mut new_shape = x.size();
        let _ = new_shape.pop();
        new_shape.extend_from_slice(&[self.num_attention_heads, -1]);
        x.view(new_shape.as_slice()).permute(&[0, 2, 1, 3])
    }

    fn linear(&self, weights: &Tensor, bias: Option<&Tensor>, x: &Tensor) -> Tensor {
        if let Some(bias) = bias {
            x.matmul(&weights.tr()) + bias
        } else {
            x.matmul(&weights.tr())
        }
    }

    fn c2p_dynamic_expand(
        &self,
        c2p_pos: &Tensor,
        query_layer: &Tensor,
        relative_pos: &Tensor,
    ) -> Tensor {
        let query_layer_size = query_layer.size();
        c2p_pos.expand(
            &[
                query_layer_size[0],
                query_layer_size[1],
                query_layer_size[2],
                *relative_pos.size().last().unwrap(),
            ],
            true,
        )
    }

    fn p2c_dynamic_expand(
        &self,
        c2p_pos: &Tensor,
        query_layer: &Tensor,
        key_layer: &Tensor,
    ) -> Tensor {
        let query_layer_size = query_layer.size();
        let mut key_layer_size = key_layer.size();
        key_layer_size.reverse();
        c2p_pos.expand(
            &[
                query_layer_size[0],
                query_layer_size[1],
                key_layer_size[1],
                key_layer_size[1],
            ],
            true,
        )
    }

    fn pos_dynamic_expand(
        &self,
        pos_index: &Tensor,
        p2c_att: &Tensor,
        key_layer: &Tensor,
    ) -> Tensor {
        let mut new_shape = p2c_att.size().iter().take(2).cloned().collect::<Vec<i64>>();
        let mut key_layer_size = key_layer.size();
        key_layer_size.reverse();
        let mut pos_index_size = pos_index.size();
        pos_index_size.reverse();
        new_shape.push(pos_index_size[1]);
        new_shape.push(key_layer_size[1]);

        pos_index.expand(&new_shape, true)
    }

    fn disentangled_att_bias(
        &self,
        query_layer: &Tensor,
        key_layer: &Tensor,
        relative_pos: Option<&Tensor>,
        relative_embeddings: &Tensor,
        scale_factor: f64,
    ) -> Result<Tensor, RustBertError> {
        let mut key_layer_size = key_layer.size();
        key_layer_size.reverse();
        let mut query_layer_size = query_layer.size();
        query_layer_size.reverse();
        let calc_relative_pos = if relative_pos.is_none() {
            Some(build_relative_position(
                query_layer_size[1],
                key_layer_size[1],
                query_layer.device(),
            ))
        } else {
            None
        };
        let relative_pos = relative_pos.unwrap_or_else(|| calc_relative_pos.as_ref().unwrap());
        let relative_pos = match &relative_pos.dim() {
            2 => relative_pos.unsqueeze(0).unsqueeze(0),
            3 => relative_pos.unsqueeze(1),
            4 => relative_pos.shallow_clone(),
            _ => {
                return Err(RustBertError::ValueError(format!(
                    "Expected relative position of dimensions 2, 3 or 4, got {}",
                    relative_pos.dim()
                )))
            }
        };

        let attention_span = *[
            *[query_layer.size()[1], key_layer.size()[1]]
                .iter()
                .max()
                .unwrap(),
            self.max_relative_positions.unwrap(),
        ]
        .iter()
        .min()
        .unwrap();

        let relative_embeddings = relative_embeddings
            .slice(
                0,
                self.max_relative_positions.unwrap() - attention_span,
                self.max_relative_positions.unwrap() + attention_span,
                1,
            )
            .unsqueeze(0);

        let mut score = Tensor::zeros(&[1], (query_layer.kind(), key_layer.device()));

        // content -> position
        if let Some(pos_proj) = &self.pos_proj {
            let pos_key_layer = self.transpose_for_scores(&relative_embeddings.apply(pos_proj));
            let c2p_att = query_layer.matmul(&pos_key_layer.transpose(-1, -2));
            let c2p_pos = (&relative_pos + attention_span).clamp(0, attention_span * 2 - 1);
            let c2p_att = c2p_att.gather(
                -1,
                &self.c2p_dynamic_expand(&c2p_pos, query_layer, &relative_pos),
                true,
            );
            score = score + c2p_att;
        }

        // position -> content
        if let Some(pos_q_proj) = &self.pos_q_proj {
            let pos_query_layer = self.transpose_for_scores(&relative_embeddings.apply(pos_q_proj));
            let pos_query_layer = &pos_query_layer
                / (*pos_query_layer.size().last().unwrap() as f64 * scale_factor).sqrt();
            let r_pos = if query_layer_size[1] != key_layer_size[1] {
                build_relative_position(key_layer_size[1], key_layer_size[1], query_layer.device())
            } else {
                relative_pos.copy()
            };
            let p2c_pos = (-r_pos + attention_span).clamp(0, attention_span * 2 - 1);
            let mut p2c_att = key_layer
                .matmul(&pos_query_layer.transpose(-1, -2))
                .gather(
                    -1,
                    &self.p2c_dynamic_expand(&p2c_pos, query_layer, key_layer),
                    true,
                )
                .transpose(-1, -2);
            if query_layer_size[1] != key_layer_size[1] {
                let pos_index = relative_pos.select(3, 0).unsqueeze(-1);
                p2c_att = p2c_att.gather(
                    -2,
                    &self.pos_dynamic_expand(&pos_index, &p2c_att, key_layer),
                    true,
                );
            }
            score = score + p2c_att;
        }

        Ok(score)
    }
}

impl DisentangledSelfAttention for DebertaDisentangledSelfAttention {
    fn new<'p, P>(p: P, config: &DebertaConfig) -> DebertaDisentangledSelfAttention
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let num_attention_heads = config.num_attention_heads;
        let attention_head_size = config.hidden_size / num_attention_heads;
        let all_head_size = num_attention_heads * attention_head_size;

        let linear_no_bias_config = nn::LinearConfig {
            bias: false,
            ..Default::default()
        };

        let in_proj = nn::linear(
            p / "in_proj",
            config.hidden_size,
            all_head_size * 3,
            linear_no_bias_config,
        );
        let q_bias = p.var("q_bias", &[all_head_size], Init::Const(0.0));
        let v_bias = p.var("v_bias", &[all_head_size], Init::Const(0.0));
        let pos_att_type = config.pos_att_type.clone().unwrap_or_default();

        let relative_attention = config.relative_attention.unwrap_or(false);
        let talking_head = config.talking_head.unwrap_or(false);

        let (head_logits_proj, head_weights_proj) = if talking_head {
            (
                Some(nn::linear(
                    p / "head_logits_proj",
                    num_attention_heads,
                    num_attention_heads,
                    linear_no_bias_config,
                )),
                Some(nn::linear(
                    p / "head_weights_proj",
                    num_attention_heads,
                    num_attention_heads,
                    linear_no_bias_config,
                )),
            )
        } else {
            (None, None)
        };

        let (max_relative_positions, pos_dropout, pos_proj, pos_q_proj) = if relative_attention {
            let mut max_relative_positions = config.max_relative_positions.unwrap_or(-1);
            if max_relative_positions < 1 {
                max_relative_positions = config.max_position_embeddings;
            }
            let pos_dropout = Some(XDropout::new(config.hidden_dropout_prob));
            let pos_proj = if pos_att_type.has_type(PositionAttentionType::c2p) {
                Some(nn::linear(
                    p / "pos_proj",
                    config.hidden_size,
                    all_head_size,
                    linear_no_bias_config,
                ))
            } else {
                None
            };
            let pos_q_proj = if pos_att_type.has_type(PositionAttentionType::p2c) {
                Some(nn::linear(
                    p / "pos_q_proj",
                    config.hidden_size,
                    all_head_size,
                    Default::default(),
                ))
            } else {
                None
            };
            (
                Some(max_relative_positions),
                pos_dropout,
                pos_proj,
                pos_q_proj,
            )
        } else {
            (None, None, None, None)
        };

        let dropout = XDropout::new(config.attention_probs_dropout_prob);

        let output_attentions = config.output_attentions.unwrap_or(false);

        DebertaDisentangledSelfAttention {
            in_proj,
            q_bias,
            v_bias,
            num_attention_heads,
            head_logits_proj,
            head_weights_proj,
            pos_proj,
            pos_q_proj,
            pos_att_type,
            max_relative_positions,
            pos_dropout,
            dropout,
            output_attentions,
        }
    }

    fn forward_t(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
        query_states: Option<&Tensor>,
        relative_pos: Option<&Tensor>,
        relative_embeddings: Option<&Tensor>,
        train: bool,
    ) -> Result<(Tensor, Option<Tensor>), RustBertError> {
        let (query_layer, key_layer, value_layer) = if let Some(query_states) = query_states {
            let ws = self.in_proj.ws.chunk(self.num_attention_heads * 3, 0);
            let query_key_value_weights = (0..3)
                .map(|k| {
                    Tensor::cat(
                        &{
                            (0..self.num_attention_heads)
                                .map(|i| ws.get((i * 3 + k) as usize).unwrap())
                                .collect::<Vec<&Tensor>>()
                        },
                        0,
                    )
                })
                .collect::<Vec<Tensor>>();

            let query_layer = self.transpose_for_scores(&self.linear(
                &query_key_value_weights[0],
                None,
                query_states,
            ));
            let key_layer = self.transpose_for_scores(&self.linear(
                &query_key_value_weights[1],
                None,
                hidden_states,
            ));
            let value_layer = self.transpose_for_scores(&self.linear(
                &query_key_value_weights[2],
                None,
                hidden_states,
            ));
            (query_layer, key_layer, value_layer)
        } else {
            let qp = hidden_states.apply(&self.in_proj);
            let mut layers = self.transpose_for_scores(&qp).chunk(3, -1);
            let value_layer = layers.pop().unwrap();
            let key_layer = layers.pop().unwrap();
            let query_layer = layers.pop().unwrap();
            (query_layer, key_layer, value_layer)
        };

        let query_layer =
            query_layer + self.transpose_for_scores(&self.q_bias.unsqueeze(0).unsqueeze(0));
        let value_layer =
            value_layer + self.transpose_for_scores(&self.v_bias.unsqueeze(0).unsqueeze(0));

        let scale_factor = 1.0 + self.pos_att_type.len() as f64;
        let scale = (*query_layer.size().last().unwrap() as f64 * scale_factor).sqrt();
        let query_layer = query_layer / scale;
        let mut attention_scores = query_layer.matmul(&key_layer.transpose(-1, -2));

        if let Some(relative_embeddings) = relative_embeddings {
            let relative_embeddings =
                relative_embeddings.apply_t(self.pos_dropout.as_ref().unwrap(), train);
            let relative_attention = self.disentangled_att_bias(
                &query_layer,
                &key_layer,
                relative_pos,
                &relative_embeddings,
                scale_factor,
            )?;
            attention_scores = attention_scores + relative_attention;
        }

        if let Some(head_logits_proj) = &self.head_logits_proj {
            attention_scores = attention_scores
                .permute(&[0, 2, 3, 1])
                .apply(head_logits_proj)
                .permute(&[0, 3, 1, 2]);
        }

        let mut attention_probs =
            x_softmax(&attention_scores, attention_mask, -1).apply_t(&self.dropout, train);

        if let Some(head_weights_proj) = &self.head_weights_proj {
            attention_probs = attention_probs
                .permute(&[0, 2, 3, 1])
                .apply(head_weights_proj)
                .permute(&[0, 3, 1, 2]);
        }

        let context_layer = attention_probs
            .matmul(&value_layer)
            .permute(&[0, 2, 1, 3])
            .contiguous();

        let mut new_context_layer_shape = context_layer.size();
        let _ = new_context_layer_shape.pop();
        let _ = new_context_layer_shape.pop();
        new_context_layer_shape.push(-1);
        let context_layer = context_layer.view(new_context_layer_shape.as_slice());

        let attention_probs = if self.output_attentions {
            Some(attention_probs)
        } else {
            None
        };

        Ok((context_layer, attention_probs))
    }
}

pub struct DebertaSelfOutput<LN: BaseDebertaLayerNorm + Module> {
    dense: nn::Linear,
    layer_norm: LN,
    dropout: XDropout,
}

impl<LN: BaseDebertaLayerNorm + Module> DebertaSelfOutput<LN> {
    pub fn new<'p, P>(p: P, config: &DebertaConfig) -> DebertaSelfOutput<LN>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let dense = nn::linear(
            p / "dense",
            config.hidden_size,
            config.hidden_size,
            Default::default(),
        );
        let layer_norm = LN::new(
            p / "LayerNorm",
            config.hidden_size,
            config.layer_norm_eps.unwrap_or(1e-7),
        );
        let dropout = XDropout::new(config.hidden_dropout_prob);
        DebertaSelfOutput {
            dense,
            layer_norm,
            dropout,
        }
    }

    pub fn forward_t(&self, hidden_states: &Tensor, input_tensor: &Tensor, train: bool) -> Tensor {
        self.layer_norm.forward(
            &(hidden_states
                .apply(&self.dense)
                .apply_t(&self.dropout, train)
                + input_tensor),
        )
    }
}

pub struct DebertaAttention<SA, LN>
where
    SA: DisentangledSelfAttention,
    LN: BaseDebertaLayerNorm + Module,
{
    self_attention: SA,
    self_output: DebertaSelfOutput<LN>,
}

impl<SA, LN> DebertaAttention<SA, LN>
where
    SA: DisentangledSelfAttention,
    LN: BaseDebertaLayerNorm + Module,
{
    pub fn new<'p, P>(p: P, config: &DebertaConfig) -> Self
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let self_attention = SA::new(p / "self", config);
        let self_output = DebertaSelfOutput::new(p / "output", config);

        DebertaAttention {
            self_attention,
            self_output,
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
        let (self_output, attention_matrix) = self.self_attention.forward_t(
            hidden_states,
            attention_mask,
            query_states,
            relative_pos,
            relative_embeddings,
            train,
        )?;

        let query_states = query_states.unwrap_or(hidden_states);

        Ok((
            self.self_output
                .forward_t(&self_output, query_states, train),
            attention_matrix,
        ))
    }
}
