// Copyright 2021, Google and The HuggingFace Inc. team. All rights reserved.
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

use crate::bart::{BartConfig, BartModelOutput};
use crate::pegasus::decoder::PegasusDecoder;
use crate::pegasus::encoder::PegasusEncoder;
use crate::pegasus::LayerState;
use std::borrow::Borrow;
use tch::nn::{embedding, EmbeddingConfig};
use tch::{nn, Tensor};

/// # Pegasus Pretrained model weight files
pub struct PegasusModelResources;

/// # Pegasus Pretrained model config files
pub struct PegasusConfigResources;

/// # Pegasus Pretrained model vocab files
pub struct PegasusVocabResources;

impl PegasusModelResources {
    /// Shared under Apache 2.0 license by the Pegasus team at https://huggingface.co/google/pegasus-cnn_dailymail. Modified with conversion to C-array format.
    pub const CNN_DAILYMAIL: (&'static str, &'static str) = (
        "pegasus-cnn_dailymail/model",
        "https://huggingface.co/google/pegasus-cnn_dailymail/resolve/main/rust_model.ot",
    );
}

impl PegasusConfigResources {
    /// Shared under Apache 2.0 license by the Pegasus team at https://huggingface.co/google/pegasus-cnn_dailymail.
    pub const CNN_DAILYMAIL: (&'static str, &'static str) = (
        "pegasus-cnn_dailymail/config",
        "https://huggingface.co/google/pegasus-cnn_dailymail/resolve/main/config.json",
    );
}

impl PegasusVocabResources {
    /// Shared under Apache 2.0 license by the Pegasus team at https://huggingface.co/google/pegasus-cnn_dailymail.
    pub const CNN_DAILYMAIL: (&'static str, &'static str) = (
        "pegasus-cnn_dailymail/spiece",
        "https://huggingface.co/google/pegasus-cnn_dailymail/resolve/main/spiece.model",
    );
}

/// # Pegasus model configuration
/// Defines the Pegasus model architecture (e.g. number of layers, hidden layer size, label mapping...)
pub type PegasusConfig = BartConfig;

/// # Pegasus Base model
/// Base architecture for Pegasus model. Usually complemented with a task-specific head, such as a language model head.
/// It is made of the following blocks:
/// - `encoder`: `PegasusEncoder` (transformer) made of a vector of encoding layers
/// - `decoder`: `PegasusDecoder` (transformer)  made of a vector of decoding layers with self attention and encoder cross-attention.
/// caching is implemented for the decoder to avoid recalculating static states (encoder key/values and previously calculated decoder key/values)
pub struct PegasusModel {
    pub(crate) encoder: PegasusEncoder,
    decoder: PegasusDecoder,
    pub(crate) embeddings: nn::Embedding,
}

impl PegasusModel {
    /// Build a new `PegasusModel`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the Pegasus model
    /// * `config` - `PegasusConfig` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::pegasus::{PegasusConfig, PegasusModel};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = PegasusConfig::from_file(config_path);
    /// let pegasus: PegasusModel = PegasusModel::new(&p.root() / "pegasus", &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &PegasusConfig) -> PegasusModel
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let pad_token_id = config.pad_token_id.unwrap_or(0);
        let embedding_config = EmbeddingConfig {
            padding_idx: pad_token_id,
            ..Default::default()
        };
        let embeddings: nn::Embedding = embedding(
            p / "shared",
            config.vocab_size,
            config.d_model,
            embedding_config,
        );

        let encoder = PegasusEncoder::new(p / "encoder", config);
        let decoder = PegasusDecoder::new(p / "decoder", config);

        PegasusModel {
            encoder,
            decoder,
            embeddings,
        }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *source_sequence_length*). Must be provided when not running in generation mode
    /// * `attention_mask` - Optional attention mask of shape (*batch size*, *source_sequence_length*) for the encoder positions. Positions with a mask with value 0 will be masked.
    /// * `decoder_input_ids` - Optional input tensor of shape (*batch size*, *target_sequence_length*). Must be provided when running in generation mode (e.g. initialiazed with a BOS token)
    /// * `encoder_outputs` - Optional tuple made of a tensor of shape (*batch size*, *source_sequence_length*, *encoder_hidden_dim*) and optional vectors of tensors of length *num_encoder_layers* with shape (*batch size*, *source_sequence_length*, *hidden_size*).
    /// These correspond to the encoder last hidden state and optional hidden states/attention weights for encoder layers. When provided, the encoder hidden state will not be recalculated. Useful for generation tasks.
    /// * `decoder_attention_mask` - Optional attention mask of shape (*batch size*, *target_sequence_length*) for the decoder positions. Positions with a mask with value 0 will be masked.
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `PegasusModelOutput` containing:
    ///   - `decoder_output` - `Tensor` of shape (*batch size*, *target_sequence_length*, *hidden_size*) representing the activations of the last decoder hidden state
    ///   - `encoder_hidden_states` - `Option<Tensor>` of shape (*batch size*, *source_sequence_length*, *hidden_size*) representing the activations of the last encoder hidden state if it was not provided, otherwise None
    ///   - `cache` - `(Option<Tensor>, Option<Vec<&LayerState, &LayerState>>)` of length *n_layer* containing the encoder padding mask and past keys and values for both the self attention and the encoder cross attention of each layer of the decoder.
    ///   - `all_encoder_hidden_states` - `Option<Vec<Tensor>>` of length *num_encoder_layers* with shape (*batch size*, *source_sequence_length*, *hidden_size*)
    ///   - `all_encoder_attentions` - `Option<Vec<Tensor>>` of length *num_encoder_layers* with shape (*batch size*, *source_sequence_length*, *hidden_size*)
    ///   - `all_decoder_hidden_states` - `Option<Vec<Tensor>>` of length *num_decoder_layers* with shape (*batch size*, *target_sequence_length*, *hidden_size*)
    ///   - `all_decoder_attentions` - `Option<Vec<Tensor>>` of length *num_decoder_layers* with shape (*batch size*, *target_sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use tch::{nn, Device, Tensor, no_grad};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::{Int64, Double};
    /// use rust_bert::pegasus::{PegasusConfig, PegasusModel};
    /// # let config_path = Path::new("path/to/config.json");
    /// # let vocab_path = Path::new("path/to/vocab.txt");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = PegasusConfig::from_file(config_path);
    /// # let pegasus_model: PegasusModel = PegasusModel::new(&vs.root(), &config);
    /// let (batch_size, source_sequence_length, target_sequence_length) = (64, 128, 56);
    /// let input_tensor = Tensor::rand(&[batch_size, source_sequence_length], (Int64, device));
    /// let decoder_input_tensor = Tensor::rand(&[batch_size, target_sequence_length], (Int64, device));
    /// let encoder_attention_mask =
    ///     Tensor::ones(&[batch_size, source_sequence_length], (Int64, device));
    /// let decoder_attention_mask =
    ///     Tensor::ones(&[batch_size, source_sequence_length], (Int64, device));
    ///
    /// let model_output = no_grad(|| {
    ///     pegasus_model.forward_t(
    ///         Some(&input_tensor),
    ///         Some(&encoder_attention_mask),
    ///         &decoder_input_tensor,
    ///         None,
    ///         Some(&decoder_attention_mask),
    ///         None,
    ///         false,
    ///     )
    /// });
    /// ```
    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        decoder_input_ids: &Tensor,
        encoder_output: Option<&Tensor>,
        decoder_attention_mask: Option<&Tensor>,
        layer_states: Option<Vec<(Option<LayerState>, Option<LayerState>)>>,
        train: bool,
    ) -> PegasusModelOutput {
        let calc_encoder_output = if encoder_output.is_none() {
            Some(self.encoder.forward_t(
                input_ids.unwrap(),
                attention_mask,
                &self.embeddings,
                train,
            ))
        } else {
            None
        };

        let (calc_hidden_states, all_encoder_hidden_states, all_encoder_attentions) =
            if let Some(calc_encoder_output) = calc_encoder_output {
                (
                    Some(calc_encoder_output.hidden_state),
                    calc_encoder_output.all_hidden_states,
                    calc_encoder_output.all_attentions,
                )
            } else {
                (None, None, None)
            };

        let encoder_output = encoder_output.unwrap_or_else(|| calc_hidden_states.as_ref().unwrap());

        let decoder_output = self.decoder.forward_t(
            &decoder_input_ids,
            &encoder_output,
            attention_mask,
            decoder_attention_mask,
            &self.embeddings,
            layer_states,
            train,
        );
        PegasusModelOutput {
            decoder_output: decoder_output.hidden_state,
            encoder_hidden_state: calc_hidden_states,
            cache: decoder_output.next_decoder_cache,
            all_decoder_hidden_states: decoder_output.all_hidden_states,
            all_decoder_attentions: decoder_output.all_attentions,
            all_encoder_hidden_states,
            all_encoder_attentions,
        }
    }
}

/// Container holding a Pegasus model output. The decoder output may hold the hidden state of
/// the last layer of the decoder, or may hold logits for a custom head module after the
/// decoder (e.g. for classification or language modeling tasks)
pub type PegasusModelOutput = BartModelOutput;
