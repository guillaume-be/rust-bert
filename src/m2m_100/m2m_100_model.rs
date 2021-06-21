// Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
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

use crate::m2m_100::decoder::M2M100Decoder;
use crate::m2m_100::encoder::M2M100Encoder;
use crate::m2m_100::LayerState;
use crate::mbart::{MBartConfig, MBartModelOutput};
use std::borrow::Borrow;
use tch::nn::{embedding, EmbeddingConfig};
use tch::{nn, Kind, Tensor};

/// # M2M100 Pretrained model weight files
pub struct M2M100ModelResources;

/// # M2M100 Pretrained model config files
pub struct M2M100ConfigResources;

/// # M2M100 Pretrained model vocab files
pub struct M2M100VocabResources;

/// # M2M100 Pretrained model ,erges files
pub struct M2M100MergesResources;

impl M2M100ModelResources {
    /// Shared under MIT license by the Facebook AI Research Fairseq team at https://github.com/pytorch/fairseq. Modified with conversion to C-array format.
    pub const M2M100_418M: (&'static str, &'static str) = (
        "m2m100-418m/model",
        "https://huggingface.co/facebook/m2m100_418M/resolve/main/rust_model.ot",
    );
}

impl M2M100ConfigResources {
    /// Shared under MIT license by the Facebook AI Research Fairseq team at https://github.com/pytorch/fairseq. Modified with conversion to C-array format.
    pub const M2M100_418M: (&'static str, &'static str) = (
        "m2m100-418m/config",
        "https://huggingface.co/facebook/m2m100_418M/resolve/main/config.json",
    );
}

impl M2M100VocabResources {
    /// Shared under MIT license by the Facebook AI Research Fairseq team at https://github.com/pytorch/fairseq. Modified with conversion to C-array format.
    pub const M2M100_418M: (&'static str, &'static str) = (
        "m2m100-418m/vocab",
        "https://huggingface.co/facebook/m2m100_418M/resolve/main/vocab.json",
    );
}

impl M2M100MergesResources {
    /// Shared under MIT license by the Facebook AI Research Fairseq team at https://github.com/pytorch/fairseq. Modified with conversion to C-array format.
    pub const M2M100_418M: (&'static str, &'static str) = (
        "m2m100-418m/merges",
        "https://huggingface.co/facebook/m2m100_418M/resolve/main/sentencepiece.bpe.model",
    );
}

pub type M2M100Config = MBartConfig;

fn _shift_tokens_right(
    input_ids: &Tensor,
    pad_token_id: i64,
    decoder_start_token_id: i64,
) -> Tensor {
    let shifted_input_ids = Tensor::zeros(
        input_ids.size().as_slice(),
        (Kind::Int64, input_ids.device()),
    );
    let _ = shifted_input_ids.select(1, 0).fill_(decoder_start_token_id);
    let _ = shifted_input_ids
        .slice(1, 1, *shifted_input_ids.size().last().unwrap(), 1)
        .copy_(&input_ids.slice(1, 0, *input_ids.size().last().unwrap() - 1, 1));
    shifted_input_ids.masked_fill(&shifted_input_ids.eq(-100), pad_token_id)
}

/// # M2M100 Base model
/// Base architecture for M2M100 model. Usually complemented with a task-specific head, such as a language model head.
/// It is made of the following blocks:
/// - `encoder`: `M2M100Encoder` (transformer) made of a vector of encoding layers
/// - `decoder`: `M2M100Decoder` (transformer)  made of a vector of decoding layers with self attention and encoder cross-attention.
/// caching is implemented for the decoder to avoid recalculating static states (encoder key/values and previously calculated decoder key/values)
/// - `pad_token_id`: padding token id
pub struct M2M100Model {
    pub(crate) encoder: M2M100Encoder,
    decoder: M2M100Decoder,
    pub(crate) embeddings: nn::Embedding,
    pad_token_id: i64,
    decoder_start_token_id: i64,
}

impl M2M100Model {
    /// Build a new `M2M100Model`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the MBart model
    /// * `config` - `M2M100Config` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::m2m_100::{M2M100Config, M2M100Model};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = M2M100Config::from_file(config_path);
    /// let mbart: M2M100Model = M2M100Model::new(&p.root() / "bart", &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &MBartConfig) -> M2M100Model
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let pad_token_id = config.pad_token_id.unwrap_or(1);
        let decoder_start_token_id = config.decoder_start_token_id.unwrap_or(2);
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

        let encoder = M2M100Encoder::new(p / "encoder", config);
        let decoder = M2M100Decoder::new(p / "decoder", config);

        M2M100Model {
            encoder,
            decoder,
            embeddings,
            pad_token_id,
            decoder_start_token_id,
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
    /// * `M2M100ModelOutput` containing:
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
    /// use rust_bert::m2m_100::{M2M100Config, M2M100Model};
    /// # let config_path = Path::new("path/to/config.json");
    /// # let vocab_path = Path::new("path/to/vocab.txt");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = M2M100Config::from_file(config_path);
    /// # let mbart_model: M2M100Model = M2M100Model::new(&vs.root(), &config);
    /// let (batch_size, source_sequence_length, target_sequence_length) = (64, 128, 56);
    /// let input_tensor = Tensor::rand(&[batch_size, source_sequence_length], (Int64, device));
    /// let target_tensor = Tensor::rand(&[batch_size, target_sequence_length], (Int64, device));
    /// let encoder_attention_mask =
    ///     Tensor::ones(&[batch_size, source_sequence_length], (Int64, device));
    /// let decoder_attention_mask =
    ///     Tensor::ones(&[batch_size, source_sequence_length], (Int64, device));
    ///
    /// let model_output = no_grad(|| {
    ///     m2m100_model.forward_t(
    ///         Some(&input_tensor),
    ///         Some(&encoder_attention_mask),
    ///         Some(&target_tensor),
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
        decoder_input_ids: Option<&Tensor>,
        encoder_output: Option<&Tensor>,
        decoder_attention_mask: Option<&Tensor>,
        layer_states: Option<Vec<(Option<LayerState>, Option<LayerState>)>>,
        train: bool,
    ) -> M2M100ModelOutput {
        let calc_decoder_input_ids = if decoder_input_ids.is_none() {
            Some(_shift_tokens_right(
                input_ids.unwrap(),
                self.pad_token_id,
                self.decoder_start_token_id,
            ))
        } else {
            None
        };

        let decoder_input_ids =
            decoder_input_ids.unwrap_or_else(|| calc_decoder_input_ids.as_ref().unwrap());

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

        M2M100ModelOutput {
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

/// Container holding a M2M100 model output
pub type M2M100ModelOutput = MBartModelOutput;
