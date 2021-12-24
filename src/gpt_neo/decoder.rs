// Copyright 2021 The Eleuther AI and HuggingFace Inc. team. All rights reserved.
// Copyright 2021 Guillaume Becquin
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
use crate::common::dropout::Dropout;
use crate::gpt_neo::attention::{GptNeoSelfAttention, LayerState};
use crate::gpt_neo::GptNeoConfig;
use crate::RustBertError;
use std::borrow::Borrow;
use tch::nn::ModuleT;
use tch::{nn, Tensor};

#[derive(Debug)]
pub struct GptNeoMLP {
    c_fc: nn::Linear,
    c_proj: nn::Linear,
    activation_function: TensorFunction,
    dropout: Dropout,
}

impl GptNeoMLP {
    pub fn new<'p, P>(p: P, intermediate_size: i64, config: &GptNeoConfig) -> GptNeoMLP
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let c_fc = nn::linear(
            p / "c_fc",
            config.hidden_size,
            intermediate_size,
            Default::default(),
        );
        let c_proj = nn::linear(
            p / "c_proj",
            intermediate_size,
            config.hidden_size,
            Default::default(),
        );

        let activation_function = config.activation_function.get_function();
        let dropout = Dropout::new(config.resid_dropout);

        GptNeoMLP {
            c_fc,
            c_proj,
            activation_function,
            dropout,
        }
    }
}

impl ModuleT for GptNeoMLP {
    fn forward_t(&self, hidden_states: &Tensor, train: bool) -> Tensor {
        let hidden_states = hidden_states.apply(&self.c_fc);
        let hidden_states = self.activation_function.get_fn()(&hidden_states);
        hidden_states
            .apply(&self.c_proj)
            .apply_t(&self.dropout, train)
    }
}

pub struct GptNeoBlock {
    ln_1: nn::LayerNorm,
    ln_2: nn::LayerNorm,
    attention: GptNeoSelfAttention,
    mlp: GptNeoMLP,
}

impl GptNeoBlock {
    pub fn new<'p, P>(p: P, layer_id: usize, config: &GptNeoConfig) -> GptNeoBlock
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let layer_norm_config = nn::LayerNormConfig {
            eps: config.layer_norm_epsilon,
            ..Default::default()
        };

        let ln_1 = nn::layer_norm(p / "ln_1", vec![config.hidden_size], layer_norm_config);
        let ln_2 = nn::layer_norm(p / "ln_2", vec![config.hidden_size], layer_norm_config);
        let attention_type = &config.attention_layers[layer_id];
        let attention =
            GptNeoSelfAttention::new(p.sub("attn").sub("attention"), config, attention_type);

        let inner_dim = config.intermediate_size.unwrap_or(4 * config.hidden_size);

        let mlp = GptNeoMLP::new(p / "mlp", inner_dim, config);

        GptNeoBlock {
            ln_1,
            ln_2,
            attention,
            mlp,
        }
    }

    pub fn forward_t(
        &self,
        hidden_states: &Tensor,
        layer_state: Option<&LayerState>,
        attention_mask: Option<&Tensor>,
        train: bool,
    ) -> Result<(Tensor, Option<Tensor>, Option<LayerState>), RustBertError> {
        let intermediate = hidden_states.apply(&self.ln_1);
        let (intermediate, attention_weights, layer_state) =
            self.attention
                .forward_t(&intermediate, layer_state, attention_mask, train);
        let hidden_states = hidden_states + intermediate;

        let intermediate = hidden_states.apply(&self.ln_2).apply_t(&self.mlp, train);
        let output = hidden_states + intermediate;

        Ok((output, attention_weights, layer_state))
    }
}
