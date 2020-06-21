// Copyright 2018 Google AI and Google Brain team.
// Copyright 2020-present, the HuggingFace Inc. team.
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


use std::collections::HashMap;
use crate::Config;
use serde::{Deserialize, Serialize};
use crate::albert::embeddings::AlbertEmbeddings;
use crate::albert::encoder::AlbertTransformer;
use tch::{nn, Tensor, Kind};
use crate::common::activations::{_tanh, _gelu_new, _gelu, _relu, _mish};
use tch::nn::{Module, Init};

#[allow(non_camel_case_types)]
#[derive(Clone, Debug, Serialize, Deserialize)]
/// # Activation function used in the attention layer and masked language model head
pub enum Activation {
    /// Gaussian Error Linear Unit ([Hendrycks et al., 2016,](https://arxiv.org/abs/1606.08415))
    gelu_new,
    /// Gaussian Error Linear Unit ([Hendrycks et al., 2016,](https://arxiv.org/abs/1606.08415))
    gelu,
    /// Rectified Linear Unit
    relu,
    /// Mish ([Misra, 2019](https://arxiv.org/abs/1908.08681))
    mish,
}


#[derive(Debug, Serialize, Deserialize)]
/// # ALBERT model configuration
/// Defines the ALBERT model architecture (e.g. number of layers, hidden layer size, label mapping...)
pub struct AlbertConfig {
    pub hidden_act: Activation,
    pub attention_probs_dropout_prob: f64,
    pub bos_token_id: i64,
    pub eos_token_id: i64,
    pub down_scale_factor: i64,
    pub embedding_size: i64,
    pub gap_size: i64,
    pub hidden_dropout_prob: f64,
    pub hidden_size: i64,
    pub initializer_range: f32,
    pub inner_group_num: i64,
    pub intermediate_size: i64,
    pub layer_norm_eps: Option<f64>,
    pub max_position_embeddings: i64,
    pub net_structure_type: i64,
    pub num_attention_heads: i64,
    pub num_hidden_groups: i64,
    pub num_hidden_layers: i64,
    pub num_memory_blocks: i64,
    pub pad_token_id: i64,
    pub type_vocab_size: i64,
    pub vocab_size: i64,
    pub output_attentions: Option<bool>,
    pub output_hidden_states: Option<bool>,
    pub is_decoder: Option<bool>,
    pub id2label: Option<HashMap<i64, String>>,
    pub label2id: Option<HashMap<String, i64>>,
}

impl Config<AlbertConfig> for AlbertConfig {}

pub struct AlbertModel {
    embeddings: AlbertEmbeddings,
    encoder: AlbertTransformer,
    pooler: nn::Linear,
    pooler_activation: Box<dyn Fn(&Tensor) -> Tensor>,
}

impl AlbertModel {
    pub fn new(p: &nn::Path, config: &AlbertConfig) -> AlbertModel {
        let embeddings = AlbertEmbeddings::new(&(p / "embeddings"), config);
        let encoder = AlbertTransformer::new(&(p / "encoder"), config);
        let pooler = nn::linear(&(p / "pooler"), config.hidden_size, config.hidden_size, Default::default());
        let pooler_activation = Box::new(_tanh);

        AlbertModel { embeddings, encoder, pooler, pooler_activation }
    }

    pub fn forward_t(&self,
                     input_ids: Option<Tensor>,
                     mask: Option<Tensor>,
                     token_type_ids: Option<Tensor>,
                     position_ids: Option<Tensor>,
                     input_embeds: Option<Tensor>,
                     train: bool)
                     -> Result<(Tensor, Tensor, Option<Vec<Tensor>>, Option<Vec<Vec<Tensor>>>), &'static str> {
        let (input_shape, device) = match &input_ids {
            Some(input_value) => match &input_embeds {
                Some(_) => { return Err("Only one of input ids or input embeddings may be set"); }
                None => (input_value.size(), input_value.device())
            }
            None => match &input_embeds {
                Some(embeds) => (vec!(embeds.size()[0], embeds.size()[1]), embeds.device()),
                None => { return Err("At least one of input ids or input embeddings must be set"); }
            }
        };

        let mask = match mask {
            Some(value) => value,
            None => Tensor::ones(&input_shape, (Kind::Int64, device))
        };

        let extended_attention_mask = mask.unsqueeze(1).unsqueeze(2);
        let extended_attention_mask: Tensor = (extended_attention_mask.ones_like() - extended_attention_mask) * -10000.0;

        let embedding_output = match self.embeddings.forward_t(input_ids, token_type_ids, position_ids, input_embeds, train) {
            Ok(value) => value,
            Err(e) => { return Err(e); }
        };

        let (hidden_state, all_hidden_states, all_attentions) =
            self.encoder.forward_t(&embedding_output,
                                   Some(extended_attention_mask),
                                   train);

        let pooled_output = self.pooler.forward(&hidden_state.select(1, 0));
        let pooled_output = (self.pooler_activation)(&pooled_output);

        Ok((hidden_state, pooled_output, all_hidden_states, all_attentions))
    }
}

pub struct AlbertMLMHead {
    layer_norm: nn::LayerNorm,
    bias: nn::Tensor,
    dense: nn::Linear,
    decoder: nn::Linear,
    activation: Box<dyn Fn(&Tensor) -> Tensor>,
}

impl AlbertMLMHead {
    pub fn new(p: &nn::Path, config: &AlbertConfig) -> AlbertMLMHead {
        let layer_norm_eps = match config.layer_norm_eps {
            Some(value) => value,
            None => 1e-12
        };
        let layer_norm_config = nn::LayerNormConfig { eps: layer_norm_eps, ..Default::default() };
        let layer_norm = nn::layer_norm(&p / "LayerNorm", vec![config.embedding_size], layer_norm_config);
        let bias = p.var("bias", &[1, config.vocab_size], Init::Const(0.));
        let dense = nn::linear(&(p / "dense"), config.hidden_size, config.embedding_size, Default::default());
        let decoder = nn::linear(&(p / "decoder"), config.embedding_size, config.vocab_size, Default::default());

        let activation = Box::new(match &config.hidden_act {
            Activation::gelu_new => _gelu_new,
            Activation::gelu => _gelu,
            Activation::relu => _relu,
            Activation::mish => _mish
        });

        AlbertMLMHead { layer_norm, bias, dense, decoder, activation }
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Tensor {
        let output: Tensor = (self.activation)(&hidden_states.apply(&self.dense));
        output.apply(&self.layer_norm).apply(&self.decoder)
    }
}