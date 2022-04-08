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
use crate::bert::bert_model::BertConfig;
use std::borrow::{Borrow, BorrowMut};
use tch::{nn, Tensor};

/// # BERT Layer
/// Layer used in BERT encoders.
/// It is made of the following blocks:
/// - `attention`: self-attention `BertAttention` layer
/// - `cross_attention`: (optional) cross-attention `BertAttention` layer (if the model is used as a decoder)
/// - `is_decoder`: flag indicating if the model is used as a decoder
/// - `intermediate`: `BertIntermediate` intermediate layer
/// - `output`: `BertOutput` output layer
pub struct BertLayer {
    attention: BertAttention,
    is_decoder: bool,
    cross_attention: Option<BertAttention>,
    intermediate: BertIntermediate,
    output: BertOutput,
}

impl BertLayer {
    /// Build a new `BertLayer`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the BERT model
    /// * `config` - `BertConfig` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::bert::{BertConfig, BertLayer};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = BertConfig::from_file(config_path);
    /// let layer: BertLayer = BertLayer::new(&p.root(), &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &BertConfig) -> BertLayer
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let attention = BertAttention::new(p / "attention", config);
        let (is_decoder, cross_attention) = match config.is_decoder {
            Some(value) => {
                if value {
                    (
                        value,
                        Some(BertAttention::new(p / "cross_attention", config)),
                    )
                } else {
                    (value, None)
                }
            }
            None => (false, None),
        };

        let intermediate = BertIntermediate::new(p / "intermediate", config);
        let output = BertOutput::new(p / "output", config);

        BertLayer {
            attention,
            is_decoder,
            cross_attention,
            intermediate,
            output,
        }
    }

    /// Forward pass through the layer
    ///
    /// # Arguments
    ///
    /// * `hidden_states` - input tensor of shape (*batch size*, *sequence_length*, *hidden_size*).
    /// * `mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `encoder_hidden_states` - Optional encoder hidden state of shape (*batch size*, *encoder_sequence_length*, *hidden_size*). If the model is defined as a decoder and the `encoder_hidden_states` is not None, used in the cross-attention layer as keys and values (query from the decoder).
    /// * `encoder_mask` - Optional encoder attention mask of shape (*batch size*, *encoder_sequence_length*). If the model is defined as a decoder and the `encoder_hidden_states` is not None, used to mask encoder values. Positions with value 0 will be masked.
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `BertLayerOutput` containing:
    ///   - `hidden_state` - `Tensor` of shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `attention_scores` - `Option<Tensor>` of shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `cross_attention_scores` - `Option<Tensor>` of shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rust_bert::bert::{BertConfig, BertLayer};
    /// # use tch::{nn, Device, Tensor, no_grad, Kind};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # let config_path = Path::new("path/to/config.json");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = BertConfig::from_file(config_path);
    /// let layer: BertLayer = BertLayer::new(&vs.root(), &config);
    /// let (batch_size, sequence_length, hidden_size) = (64, 128, 512);
    /// let input_tensor = Tensor::rand(
    ///     &[batch_size, sequence_length, hidden_size],
    ///     (Kind::Float, device),
    /// );
    /// let mask = Tensor::zeros(&[batch_size, sequence_length], (Kind::Int64, device));
    ///
    /// let layer_output = no_grad(|| layer.forward_t(&input_tensor, Some(&mask), None, None, false));
    /// ```
    pub fn forward_t(
        &self,
        hidden_states: &Tensor,
        mask: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
        encoder_mask: Option<&Tensor>,
        train: bool,
    ) -> BertLayerOutput {
        let (attention_output, attention_weights) =
            self.attention
                .forward_t(hidden_states, mask, None, None, train);

        let (attention_output, attention_scores, cross_attention_scores) =
            if self.is_decoder & encoder_hidden_states.is_some() {
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
                (attention_output, attention_weights, None)
            };

        let output = self.intermediate.forward(&attention_output);
        let output = self.output.forward_t(&output, &attention_output, train);

        BertLayerOutput {
            hidden_state: output,
            attention_weights: attention_scores,
            cross_attention_weights: cross_attention_scores,
        }
    }
}

/// # BERT Encoder
/// Encoder used in BERT models.
/// It is made of a Vector of `BertLayer` through which hidden states will be passed. The encoder can also be
/// used as a decoder (with cross-attention) if `encoder_hidden_states` are provided.
pub struct BertEncoder {
    output_attentions: bool,
    output_hidden_states: bool,
    layers: Vec<BertLayer>,
}

impl BertEncoder {
    /// Build a new `BertEncoder`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the BERT model
    /// * `config` - `BertConfig` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::bert::{BertConfig, BertEncoder};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = BertConfig::from_file(config_path);
    /// let encoder: BertEncoder = BertEncoder::new(&p.root(), &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &BertConfig) -> BertEncoder
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow() / "layer";
        let output_attentions = config.output_attentions.unwrap_or(false);
        let output_hidden_states = config.output_hidden_states.unwrap_or(false);

        let mut layers: Vec<BertLayer> = vec![];
        for layer_index in 0..config.num_hidden_layers {
            layers.push(BertLayer::new(&p / layer_index, config));
        }

        BertEncoder {
            output_attentions,
            output_hidden_states,
            layers,
        }
    }

    /// Forward pass through the encoder
    ///
    /// # Arguments
    ///
    /// * `hidden_states` - input tensor of shape (*batch size*, *sequence_length*, *hidden_size*).
    /// * `mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `encoder_hidden_states` - Optional encoder hidden state of shape (*batch size*, *encoder_sequence_length*, *hidden_size*). If the model is defined as a decoder and the `encoder_hidden_states` is not None, used in the cross-attention layer as keys and values (query from the decoder).
    /// * `encoder_mask` - Optional encoder attention mask of shape (*batch size*, *encoder_sequence_length*). If the model is defined as a decoder and the `encoder_hidden_states` is not None, used to mask encoder values. Positions with value 0 will be masked.
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `BertEncoderOutput` containing:
    ///   - `hidden_state` - `Tensor` of shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `all_hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `all_attentions` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rust_bert::bert::{BertConfig, BertEncoder};
    /// # use tch::{nn, Device, Tensor, no_grad, Kind};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # let config_path = Path::new("path/to/config.json");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = BertConfig::from_file(config_path);
    /// let encoder: BertEncoder = BertEncoder::new(&vs.root(), &config);
    /// let (batch_size, sequence_length, hidden_size) = (64, 128, 512);
    /// let input_tensor = Tensor::rand(
    ///     &[batch_size, sequence_length, hidden_size],
    ///     (Kind::Float, device),
    /// );
    /// let mask = Tensor::zeros(&[batch_size, sequence_length], (Kind::Int8, device));
    ///
    /// let encoder_output =
    ///     no_grad(|| encoder.forward_t(&input_tensor, Some(&mask), None, None, false));
    /// ```
    pub fn forward_t(
        &self,
        input: &Tensor,
        mask: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
        encoder_mask: Option<&Tensor>,
        train: bool,
    ) -> BertEncoderOutput {
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

        let mut hidden_state = None::<Tensor>;
        let mut attention_weights: Option<Tensor>;

        for layer in &self.layers {
            let layer_output = if let Some(hidden_state) = &hidden_state {
                layer.forward_t(
                    hidden_state,
                    mask,
                    encoder_hidden_states,
                    encoder_mask,
                    train,
                )
            } else {
                layer.forward_t(input, mask, encoder_hidden_states, encoder_mask, train)
            };

            hidden_state = Some(layer_output.hidden_state);
            attention_weights = layer_output.attention_weights;
            if let Some(attentions) = all_attentions.borrow_mut() {
                attentions.push(attention_weights.as_ref().unwrap().copy());
            };
            if let Some(hidden_states) = all_hidden_states.borrow_mut() {
                hidden_states.push(hidden_state.as_ref().unwrap().copy());
            };
        }

        BertEncoderOutput {
            hidden_state: hidden_state.unwrap(),
            all_hidden_states,
            all_attentions,
        }
    }
}

/// # BERT Pooler
/// Pooler used in BERT models.
/// It is made of a fully connected layer which is applied to the first sequence element.
pub struct BertPooler {
    lin: nn::Linear,
}

impl BertPooler {
    /// Build a new `BertPooler`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the BERT model
    /// * `config` - `BertConfig` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::bert::{BertConfig, BertPooler};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = BertConfig::from_file(config_path);
    /// let pooler: BertPooler = BertPooler::new(&p.root(), &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &BertConfig) -> BertPooler
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let lin = nn::linear(
            p / "dense",
            config.hidden_size,
            config.hidden_size,
            Default::default(),
        );
        BertPooler { lin }
    }

    /// Forward pass through the pooler
    ///
    /// # Arguments
    ///
    /// * `hidden_states` - input tensor of shape (*batch size*, *sequence_length*, *hidden_size*).
    ///
    /// # Returns
    ///
    /// * `Tensor` of shape (*batch size*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rust_bert::bert::{BertConfig, BertPooler};
    /// # use tch::{nn, Device, Tensor, no_grad, Kind};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # let config_path = Path::new("path/to/config.json");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = BertConfig::from_file(config_path);
    /// let pooler: BertPooler = BertPooler::new(&vs.root(), &config);
    /// let (batch_size, sequence_length, hidden_size) = (64, 128, 512);
    /// let input_tensor = Tensor::rand(
    ///     &[batch_size, sequence_length, hidden_size],
    ///     (Kind::Float, device),
    /// );
    ///
    /// let pooler_output = no_grad(|| pooler.forward(&input_tensor));
    /// ```
    pub fn forward(&self, hidden_states: &Tensor) -> Tensor {
        hidden_states.select(1, 0).apply(&self.lin).tanh()
    }
}

/// Container for the BERT layer output.
pub struct BertLayerOutput {
    /// Hidden states
    pub hidden_state: Tensor,
    /// Self attention scores
    pub attention_weights: Option<Tensor>,
    /// Cross attention scores
    pub cross_attention_weights: Option<Tensor>,
}

/// Container for the BERT encoder output.
pub struct BertEncoderOutput {
    /// Last hidden states from the model
    pub hidden_state: Tensor,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all intermediate layers
    pub all_attentions: Option<Vec<Tensor>>,
}
