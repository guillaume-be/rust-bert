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

use crate::bert::attention::{BertAttention, BertIntermediate, BertOutput};
use crate::bert::bert::BertConfig;
use std::borrow::BorrowMut;
use tch::{nn, Tensor};

pub struct BertLayer {
    attention: BertAttention,
    is_decoder: bool,
    cross_attention: Option<BertAttention>,
    intermediate: BertIntermediate,
    output: BertOutput,
}

impl BertLayer {
    pub fn new(p: &nn::Path, config: &BertConfig) -> BertLayer {
        let attention = BertAttention::new(&(p / "attention"), &config);
        let (is_decoder, cross_attention) = match config.is_decoder {
            Some(value) => {
                if value == true {
                    (
                        value,
                        Some(BertAttention::new(&(p / "cross_attention"), &config)),
                    )
                } else {
                    (value, None)
                }
            }
            None => (false, None),
        };

        let intermediate = BertIntermediate::new(&(p / "intermediate"), &config);
        let output = BertOutput::new(&(p / "output"), &config);

        BertLayer {
            attention,
            is_decoder,
            cross_attention,
            intermediate,
            output,
        }
    }

    pub fn forward_t(
        &self,
        hidden_states: &Tensor,
        mask: &Option<Tensor>,
        encoder_hidden_states: &Option<Tensor>,
        encoder_mask: &Option<Tensor>,
        train: bool,
    ) -> (Tensor, Option<Tensor>, Option<Tensor>) {
        let (attention_output, attention_weights, cross_attention_weights) =
            if self.is_decoder & encoder_hidden_states.is_some() {
                let (attention_output, attention_weights) =
                    self.attention
                        .forward_t(hidden_states, mask, &None, &None, train);
                let (attention_output, cross_attention_weights) =
                    self.cross_attention.as_ref().unwrap().forward_t(
                        &attention_output,
                        mask,
                        encoder_hidden_states,
                        encoder_mask,
                        train,
                    );
                (attention_output, attention_weights, cross_attention_weights)
            } else {
                let (attention_output, attention_weights) =
                    self.attention
                        .forward_t(hidden_states, mask, &None, &None, train);
                (attention_output, attention_weights, None)
            };

        let output = self.intermediate.forward(&attention_output);
        let output = self.output.forward_t(&output, &attention_output, train);

        (output, attention_weights, cross_attention_weights)
    }
}

pub struct BertEncoder {
    output_attentions: bool,
    output_hidden_states: bool,
    layers: Vec<BertLayer>,
}

impl BertEncoder {
    pub fn new(p: &nn::Path, config: &BertConfig) -> BertEncoder {
        let p = &(p / "layer");
        let output_attentions = if let Some(value) = config.output_attentions {
            value
        } else {
            false
        };
        let output_hidden_states = if let Some(value) = config.output_hidden_states {
            value
        } else {
            false
        };

        let mut layers: Vec<BertLayer> = vec![];
        for layer_index in 0..config.num_hidden_layers {
            layers.push(BertLayer::new(&(p / layer_index), config));
        }

        BertEncoder {
            output_attentions,
            output_hidden_states,
            layers,
        }
    }

    pub fn forward_t(
        &self,
        hidden_states: &Tensor,
        mask: &Option<Tensor>,
        encoder_hidden_states: &Option<Tensor>,
        encoder_mask: &Option<Tensor>,
        train: bool,
    ) -> (Tensor, Option<Vec<Tensor>>, Option<Vec<Tensor>>) {
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

        let mut hidden_state = hidden_states.copy();
        let mut attention_weights: Option<Tensor>;
        let mut layers = self.layers.iter();
        loop {
            match layers.next() {
                Some(layer) => {
                    if let Some(hidden_states) = all_hidden_states.borrow_mut() {
                        hidden_states.push(hidden_state.as_ref().copy());
                    };

                    let temp = layer.forward_t(
                        &hidden_state,
                        &mask,
                        encoder_hidden_states,
                        encoder_mask,
                        train,
                    );
                    hidden_state = temp.0;
                    attention_weights = temp.1;
                    if let Some(attentions) = all_attentions.borrow_mut() {
                        attentions.push(attention_weights.as_ref().unwrap().copy());
                    };
                }
                None => break,
            };
        }

        (hidden_state, all_hidden_states, all_attentions)
    }
}

pub struct BertPooler {
    lin: nn::Linear,
}

impl BertPooler {
    pub fn new(p: &nn::Path, config: &BertConfig) -> BertPooler {
        let lin = nn::linear(
            &(p / "dense"),
            config.hidden_size,
            config.hidden_size,
            Default::default(),
        );
        BertPooler { lin }
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Tensor {
        hidden_states.select(1, 0).apply(&self.lin).tanh()
    }
}
