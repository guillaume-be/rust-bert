// Copyright (c) 2020  The Google AI Language Team Authors, The HuggingFace Inc. team and github/lonePatient
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
use crate::mobilebert::mobilebert_model::{NormalizationLayer, NormalizationType};
use crate::mobilebert::MobileBertConfig;
use std::borrow::Borrow;
use tch::{nn, Tensor};

pub struct MobileBertSelfAttention {
    attention_head_size: i64,
    num_attention_heads: i64,
    query: nn::Linear,
    key: nn::Linear,
    value: nn::Linear,
    dropout: Dropout,
    output_attentions: bool,
}

impl MobileBertSelfAttention {
    pub fn new<'p, P>(p: P, config: &MobileBertConfig) -> MobileBertSelfAttention
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let all_head_size = if config.use_bottleneck.unwrap_or(true) {
            config.intra_bottleneck_size.unwrap_or(128)
        } else {
            config.hidden_size
        };
        let attention_head_size = all_head_size / config.num_attention_heads;
        let query = nn::linear(
            p / "query",
            all_head_size,
            all_head_size,
            Default::default(),
        );
        let key = nn::linear(p / "key", all_head_size, all_head_size, Default::default());
        let value = nn::linear(
            p / "value",
            if config.use_bottleneck_attention.unwrap_or(false) {
                all_head_size
            } else {
                config.hidden_size
            },
            all_head_size,
            Default::default(),
        );

        let dropout = Dropout::new(config.attention_probs_dropout_prob);
        let output_attentions = config.output_attentions.unwrap_or(false);

        MobileBertSelfAttention {
            attention_head_size,
            num_attention_heads: config.num_attention_heads,
            query,
            key,
            value,
            dropout,
            output_attentions,
        }
    }

    fn split_heads(&self, x: Tensor, bs: i64, dim_per_head: i64) -> Tensor {
        x.view((bs, -1, self.num_attention_heads, dim_per_head))
            .transpose(1, 2)
    }

    fn flatten(&self, x: Tensor, bs: i64, dim_per_head: i64) -> Tensor {
        x.transpose(1, 2)
            .contiguous()
            .view((bs, -1, self.num_attention_heads * dim_per_head))
    }

    pub fn forward_t(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attention_mask: Option<&Tensor>,
        train: bool,
    ) -> (Tensor, Option<Tensor>) {
        let bs = query.size()[0];

        let query = query.apply(&self.query);
        let key = key.apply(&self.key);
        let value = value.apply(&self.value);

        let query = self.split_heads(query, bs, self.attention_head_size);
        let key = self.split_heads(key, bs, self.attention_head_size);
        let value = self.split_heads(value, bs, self.attention_head_size);

        let mut attention_scores =
            query.matmul(&key.transpose(-1, -2)) / (self.attention_head_size as f64).sqrt();
        if let Some(attention_mask_value) = attention_mask {
            attention_scores = attention_scores + attention_mask_value;
        }
        let attention_probs = attention_scores
            .softmax(-1, attention_scores.kind())
            .apply_t(&self.dropout, train);
        let context = attention_probs.matmul(&value);
        let context = self.flatten(context, bs, self.attention_head_size);
        let attention_probs = if self.output_attentions {
            attention_probs.into()
        } else {
            None
        };
        (context, attention_probs)
    }
}

pub struct MobileBertSelfOutput {
    dense: nn::Linear,
    layer_norm: NormalizationLayer,
    dropout: Option<Dropout>,
}

impl MobileBertSelfOutput {
    pub fn new<'p, P>(p: P, config: &MobileBertConfig) -> MobileBertSelfOutput
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let use_bottleneck = config.use_bottleneck.unwrap_or(true);
        let true_hidden_size = if config.use_bottleneck.unwrap_or(true) {
            config.intra_bottleneck_size.unwrap_or(128)
        } else {
            config.hidden_size
        };
        let dense = nn::linear(
            p / "dense",
            true_hidden_size,
            true_hidden_size,
            Default::default(),
        );
        let layer_norm = NormalizationLayer::new(
            p / "LayerNorm",
            config
                .normalization_type
                .unwrap_or(NormalizationType::no_norm),
            true_hidden_size,
            None,
        );

        let dropout = if !use_bottleneck {
            Dropout::new(config.hidden_dropout_prob).into()
        } else {
            None
        };
        MobileBertSelfOutput {
            dense,
            layer_norm,
            dropout,
        }
    }

    pub fn forward_t(
        &self,
        hidden_states: &Tensor,
        residual_tensor: &Tensor,
        train: bool,
    ) -> Tensor {
        let mut layer_output = hidden_states.apply(&self.dense);
        if let Some(dropout) = &self.dropout {
            layer_output = layer_output.apply_t(dropout, train);
        };
        self.layer_norm.forward(&(layer_output + residual_tensor))
    }
}

pub struct MobileBertAttention {
    self_attention: MobileBertSelfAttention,
    output: MobileBertSelfOutput,
}

impl MobileBertAttention {
    pub fn new<'p, P>(p: P, config: &MobileBertConfig) -> MobileBertAttention
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let self_attention = MobileBertSelfAttention::new(p / "self", config);
        let output = MobileBertSelfOutput::new(p / "output", config);

        MobileBertAttention {
            self_attention,
            output,
        }
    }

    pub fn forward_t(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        layer_input: &Tensor,
        attention_mask: Option<&Tensor>,
        train: bool,
    ) -> (Tensor, Option<Tensor>) {
        let (self_outputs, attention_scores) =
            self.self_attention
                .forward_t(query, key, value, attention_mask, train);
        let attention_output = self.output.forward_t(&self_outputs, layer_input, train);
        (attention_output, attention_scores)
    }
}
