// Copyright 2020 The Facebook AI Research Team Authors
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

use crate::bart::attention::LayerState;
use crate::bart::decoder::BartDecoder;
use crate::bart::encoder::BartEncoder;
use crate::common::dropout::Dropout;
use crate::pipelines::generation::{Cache, LMHeadModel};
use crate::Config;
use serde::{Deserialize, Serialize};
use std::borrow::Borrow;
use std::collections::HashMap;
use tch::kind::Kind::{Float, Int64};
use tch::nn::{embedding, EmbeddingConfig};
use tch::{nn, Tensor};

/// # BART Pretrained model weight files
pub struct BartModelResources;

/// # BART Pretrained model config files
pub struct BartConfigResources;

/// # BART Pretrained model vocab files
pub struct BartVocabResources;

/// # BART Pretrained model merges files
pub struct BartMergesResources;

impl BartModelResources {
    /// Shared under MIT license by the Facebook AI Research Fairseq team at https://github.com/pytorch/fairseq. Modified with conversion to C-array format.
    pub const BART: (&'static str, &'static str) = (
        "bart/model.ot",
        "https://cdn.huggingface.co/facebook/bart-large/rust_model.ot",
    );
    /// Shared under MIT license by the Facebook AI Research Fairseq team at https://github.com/pytorch/fairseq. Modified with conversion to C-array format.
    pub const BART_CNN: (&'static str, &'static str) = (
        "bart-cnn/model.ot",
        "https://cdn.huggingface.co/facebook/bart-large-cnn/rust_model.ot",
    );
    /// Shared under MIT license by the Facebook AI Research Fairseq team at https://github.com/pytorch/fairseq. Modified with conversion to C-array format.
    pub const BART_XSUM: (&'static str, &'static str) = (
        "bart-xsum/model.ot",
        "https://cdn.huggingface.co/facebook/bart-large-xsum/rust_model.ot",
    );
    /// Shared under MIT license by the Facebook AI Research Fairseq team at https://github.com/pytorch/fairseq. Modified with conversion to C-array format.
    pub const BART_MNLI: (&'static str, &'static str) = (
        "bart-large-mnli/model.ot",
        "https://cdn.huggingface.co/facebook/bart-large-mnli/rust_model.ot",
    );
}

impl BartConfigResources {
    /// Shared under MIT license by the Facebook AI Research Fairseq team at https://github.com/pytorch/fairseq. Modified with conversion to C-array format.
    pub const BART: (&'static str, &'static str) = (
        "bart/config.json",
        "https://cdn.huggingface.co/facebook/bart-large/config.json",
    );
    /// Shared under MIT license by the Facebook AI Research Fairseq team at https://github.com/pytorch/fairseq. Modified with conversion to C-array format.
    pub const BART_CNN: (&'static str, &'static str) = (
        "bart-cnn/config.json",
        "https://cdn.huggingface.co/facebook/bart-large-cnn/config.json",
    );
    /// Shared under MIT license by the Facebook AI Research Fairseq team at https://github.com/pytorch/fairseq. Modified with conversion to C-array format.
    pub const BART_XSUM: (&'static str, &'static str) = (
        "bart-xsum/config.json",
        "https://cdn.huggingface.co/facebook/bart-large-xsum/config.json",
    );
    /// Shared under MIT license by the Facebook AI Research Fairseq team at https://github.com/pytorch/fairseq. Modified with conversion to C-array format.
    pub const BART_MNLI: (&'static str, &'static str) = (
        "bart-large-mnli/config.json",
        "https://cdn.huggingface.co/facebook/bart-large-mnli/config.json",
    );
}

impl BartVocabResources {
    /// Shared under MIT license by the Facebook AI Research Fairseq team at https://github.com/pytorch/fairseq. Modified with conversion to C-array format.
    pub const BART: (&'static str, &'static str) = (
        "bart/vocab.txt",
        "https://cdn.huggingface.co/roberta-large-vocab.json",
    );
    /// Shared under MIT license by the Facebook AI Research Fairseq team at https://github.com/pytorch/fairseq. Modified with conversion to C-array format.
    pub const BART_CNN: (&'static str, &'static str) = (
        "bart-cnn/vocab.txt",
        "https://cdn.huggingface.co/roberta-large-vocab.json",
    );
    /// Shared under MIT license by the Facebook AI Research Fairseq team at https://github.com/pytorch/fairseq. Modified with conversion to C-array format.
    pub const BART_XSUM: (&'static str, &'static str) = (
        "bart-xsum/vocab.txt",
        "https://cdn.huggingface.co/roberta-large-vocab.json",
    );
    /// Shared under MIT license by the Facebook AI Research Fairseq team at https://github.com/pytorch/fairseq. Modified with conversion to C-array format.
    pub const BART_MNLI: (&'static str, &'static str) = (
        "bart-large-mnli/vocab.txt",
        "https://cdn.huggingface.co/roberta-large-vocab.json",
    );
}

impl BartMergesResources {
    /// Shared under MIT license by the Facebook AI Research Fairseq team at https://github.com/pytorch/fairseq. Modified with conversion to C-array format.
    pub const BART: (&'static str, &'static str) = (
        "bart/merges.txt",
        "https://cdn.huggingface.co/roberta-large-merges.txt",
    );
    /// Shared under MIT license by the Facebook AI Research Fairseq team at https://github.com/pytorch/fairseq. Modified with conversion to C-array format.
    pub const BART_CNN: (&'static str, &'static str) = (
        "bart-cnn/merges.txt",
        "https://cdn.huggingface.co/roberta-large-merges.txt",
    );
    /// Shared under MIT license by the Facebook AI Research Fairseq team at https://github.com/pytorch/fairseq. Modified with conversion to C-array format.
    pub const BART_XSUM: (&'static str, &'static str) = (
        "bart-xsum/merges.txt",
        "https://cdn.huggingface.co/roberta-large-merges.txt",
    );
    /// Shared under MIT license by the Facebook AI Research Fairseq team at https://github.com/pytorch/fairseq. Modified with conversion to C-array format.
    pub const BART_MNLI: (&'static str, &'static str) = (
        "bart-large-mnli/merges.txt",
        "https://cdn.huggingface.co/roberta-large-merges.txt",
    );
}

#[allow(non_camel_case_types)]
#[derive(Debug, Serialize, Deserialize)]
pub enum Activation {
    /// Gaussian Error Linear Unit ([Hendrycks et al., 2016,](https://arxiv.org/abs/1606.08415))
    gelu,
    /// Rectified Linear Unit
    relu,
    /// Swish ([Ramachandran, 2017](https://arxiv.org/abs/1710.05941))
    swish,
    /// Gaussian Error Linear Unit - OpenAI version ([Hendrycks et al., 2016,](https://arxiv.org/abs/1606.08415))
    gelu_new,
    /// Tanh
    tanh,
}

#[derive(Debug, Serialize, Deserialize)]
/// # BART model configuration
/// Defines the BART model architecture (e.g. number of layers, hidden layer size, label mapping...)
pub struct BartConfig {
    pub num_labels: Option<i64>,
    pub activation_function: Option<Activation>,
    pub activation_dropout: f64,
    pub attention_dropout: f64,
    pub classif_dropout: f64,
    pub d_model: i64,
    pub decoder_attention_heads: i64,
    pub decoder_ffn_dim: i64,
    pub decoder_layerdrop: f64,
    pub decoder_layers: i64,
    pub decoder_start_token_id: Option<i64>,
    pub dropout: f64,
    pub encoder_attention_heads: i64,
    pub encoder_ffn_dim: i64,
    pub encoder_layerdrop: f64,
    pub encoder_layers: i64,
    pub bos_token_id: Option<i64>,
    pub eos_token_id: Option<i64>,
    pub pad_token_id: Option<i64>,
    pub id2label: Option<HashMap<i64, String>>,
    pub label2id: Option<HashMap<String, i64>>,
    pub init_std: f64,
    pub is_decoder: Option<bool>,
    pub is_encoder_decoder: Option<bool>,
    pub max_position_embeddings: i64,
    pub min_length: Option<i64>,
    pub no_repeat_ngram_size: Option<i64>,
    pub normalize_embedding: Option<bool>,
    pub num_hidden_layers: i64,
    pub output_attentions: Option<bool>,
    pub output_hidden_states: Option<bool>,
    pub output_past: Option<bool>,
    pub static_position_embeddings: Option<bool>,
    pub scale_embedding: Option<bool>,
    pub vocab_size: i64,
}

impl Config<BartConfig> for BartConfig {}

fn _prepare_bart_decoder_inputs(
    pad_token_id: i64,
    input_ids: &Tensor,
    decoder_input_ids: Option<&Tensor>,
    decoder_padding_mask: Option<&Tensor>,
) -> (Tensor, Option<Tensor>, Option<Tensor>) {
    let decoder_input_ids = match decoder_input_ids {
        Some(value) => value.copy(),
        None => _shift_tokens_right(input_ids, pad_token_id),
    };

    let decoder_padding_mask = match decoder_padding_mask {
        Some(value) => Some(value.eq(0).to_kind(Int64)),
        None => {
            let padding_mask = decoder_input_ids.eq(pad_token_id);
            if i64::from(padding_mask.any()) == 0 {
                None
            } else {
                Some(padding_mask)
            }
        }
    };
    let length = *input_ids.size().last().unwrap();
    let causal_mask = Tensor::empty(&[length, length], (Float, input_ids.device()))
        .fill_(std::f64::NEG_INFINITY)
        .triu(1);

    (decoder_input_ids, decoder_padding_mask, Some(causal_mask))
}

fn _shift_tokens_right(input_ids: &Tensor, pad_token_id: i64) -> Tensor {
    let index_eos: Tensor = input_ids.ne(pad_token_id).sum1(&[-1], true, Int64) - 1;
    let output = input_ids.empty_like().to_kind(Int64);
    output
        .select(1, 0)
        .copy_(&input_ids.gather(1, &index_eos, true).squeeze());
    output
        .slice(1, 1, *output.size().last().unwrap(), 1)
        .copy_(&input_ids.slice(1, 0, *output.size().last().unwrap() - 1, 1));
    output
}

/// # BART Base model
/// Base architecture for BART model. Usually complemented with a task-specific head, such as a language model head.
/// It is made of the following blocks:
/// - `encoder`: `BartEncoder` (transformer) made of a vector of encoding layers
/// - `decoder`: `BartDecoder` (transformer)  made of a vector of decoding layers with self attention and encoder cross-attention.
/// caching is implemented for the decoder to avoid recalculating static states (encoder key/values and previously calculated decoder key/values)
/// - `generation_mode`: flag indicating if the model should run in generation mode (a decoder start token must then be provided)
/// - `pad_token_id`: padding token id
pub struct BartModel {
    pub(crate) encoder: BartEncoder,
    decoder: BartDecoder,
    pub(crate) embeddings: nn::Embedding,
    generation_mode: bool,
    pad_token_id: i64,
}

impl BartModel {
    /// Build a new `BartModel`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the BART model
    /// * `config` - `BartConfig` object defining the model architecture
    /// * `generation_mode` - flag indicating if the model should run in generation mode (a decoder start token must then be provided)
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::bart::{BartConfig, BartModel};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = BartConfig::from_file(config_path);
    /// let generation_mode = true;
    /// let bart: BartModel = BartModel::new(&p.root() / "bart", &config, generation_mode);
    /// ```
    pub fn new<'p, P>(p: P, config: &BartConfig, generation_mode: bool) -> BartModel
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let pad_token_id = match config.pad_token_id {
            Some(value) => value,
            None => 1,
        };
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

        let encoder = BartEncoder::new(p / "encoder", config);
        let decoder = BartDecoder::new(p / "decoder", config, generation_mode);

        BartModel {
            encoder,
            decoder,
            generation_mode,
            pad_token_id,
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
    /// * `decoder_output` - `Tensor` of shape (*batch size*, *target_sequence_length*, *hidden_size*) representing the activations of the last decoder hidden state
    /// * `encoder_hidden_states` - `Tensor` of shape (*batch size*, *source_sequence_length*, *hidden_size*) representing the activations of the last encoder hidden state
    /// * `decoder_cache` - `(Option<Tensor>, Option<Vec<&LayerState, &LayerState>>)` of length *n_layer* containing the encoder padding mask and past keys and values for
    /// both the self attention and the encoder cross attention of each layer of the decoder.
    /// * `all_encoder_hidden_states` - `Option<Vec<Tensor>>` of length *num_encoder_layers* with shape (*batch size*, *source_sequence_length*, *hidden_size*)
    /// * `all_encoder_attentions` - `Option<Vec<Tensor>>` of length *num_encoder_layers* with shape (*batch size*, *source_sequence_length*, *hidden_size*)
    /// * `all_decoder_hidden_states` - `Option<Vec<Tensor>>` of length *num_decoder_layers* with shape (*batch size*, *target_sequence_length*, *hidden_size*)
    /// * `all_decoder_attentions` - `Option<Vec<Tensor>>` of length *num_decoder_layers* with shape (*batch size*, *target_sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use tch::{nn, Device, Tensor, no_grad};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::{Int64, Double};
    /// use rust_bert::bart::{BartConfig, BartModel};
    /// # let config_path = Path::new("path/to/config.json");
    /// # let vocab_path = Path::new("path/to/vocab.txt");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = BartConfig::from_file(config_path);
    /// # let bart_model: BartModel = BartModel::new(&vs.root(), &config, false);
    /// let (batch_size, source_sequence_length, target_sequence_length) = (64, 128, 56);
    /// let input_tensor = Tensor::rand(&[batch_size, source_sequence_length], (Int64, device));
    /// let target_tensor = Tensor::rand(&[batch_size, target_sequence_length], (Int64, device));
    /// let encoder_attention_mask =
    ///     Tensor::ones(&[batch_size, source_sequence_length], (Int64, device));
    /// let decoder_attention_mask =
    ///     Tensor::ones(&[batch_size, source_sequence_length], (Int64, device));
    ///
    /// let (
    ///     decoder_output,
    ///     encoder_hidden_states,
    ///     decoder_cache,
    ///     all_encoder_hidden_states,
    ///     all_encoder_attentions,
    ///     all_decoder_hidden_states,
    ///     all_decoder_attentions,
    /// ) = no_grad(|| {
    ///     bart_model.forward_t(
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
        encoder_outputs: Option<(Tensor, Option<Vec<Tensor>>, Option<Vec<Tensor>>)>,
        decoder_attention_mask: Option<&Tensor>,
        layer_states: Option<Vec<(Option<LayerState>, Option<LayerState>)>>,
        train: bool,
    ) -> (
        Tensor,
        Tensor,
        Option<Vec<(Option<LayerState>, Option<LayerState>)>>,
        Option<Vec<Tensor>>,
        Option<Vec<Tensor>>,
        Option<Vec<Tensor>>,
        Option<Vec<Tensor>>,
    ) {
        let (decoder_input_ids, decoder_padding_mask, causal_mask) = if self.generation_mode {
            (decoder_input_ids.unwrap().copy(), None, None)
        } else {
            assert!(
                input_ids.is_some(),
                "input_ids must be provided when not in generation mode"
            );
            _prepare_bart_decoder_inputs(
                self.pad_token_id,
                input_ids.unwrap(),
                decoder_input_ids,
                decoder_attention_mask,
            )
        };

        let (encoder_hidden_states, all_encoder_hidden_states, all_encoder_attentions) =
            match encoder_outputs {
                Some(value) => value,
                None => {
                    assert!(
                        input_ids.is_some(),
                        "input_ids must be provided when encoder output is not pre-computed"
                    );
                    self.encoder.forward_t(
                        input_ids.unwrap(),
                        attention_mask,
                        &self.embeddings,
                        train,
                    )
                }
            };

        let (decoder_outputs, decoder_cache, all_decoder_hidden_states, all_decoder_attentions) =
            self.decoder.forward_t(
                &decoder_input_ids,
                &encoder_hidden_states,
                attention_mask,
                decoder_padding_mask.as_ref(),
                causal_mask.as_ref(),
                &self.embeddings,
                layer_states,
                train,
            );
        (
            decoder_outputs,
            encoder_hidden_states,
            decoder_cache.1,
            all_decoder_hidden_states,
            all_decoder_attentions,
            all_encoder_hidden_states,
            all_encoder_attentions,
        )
    }
}

/// # BART Model for conditional generation
/// BART model with a vocabulary decoding head
/// It is made of the following blocks:
/// - `base_model`: `BartModel` Base BART model
/// - `linear`: Linear layer without bias tied to the weights of the token id embeddings
pub struct BartForConditionalGeneration {
    base_model: BartModel,
}

impl BartForConditionalGeneration {
    /// Build a new `BartForConditionalGeneration`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the BART model
    /// * `config` - `BartConfig` object defining the model architecture
    /// * `generation_mode` - flag indicating if the model should run in generation mode (a decoder start token must then be provided)
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::bart::{BartConfig, BartForConditionalGeneration};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = BartConfig::from_file(config_path);
    /// let generation_mode = true;
    /// let bart: BartForConditionalGeneration =
    ///     BartForConditionalGeneration::new(&p.root() / "bart", &config, generation_mode);
    /// ```
    pub fn new<'p, P>(
        p: P,
        config: &BartConfig,
        generation_mode: bool,
    ) -> BartForConditionalGeneration
    where
        P: Borrow<nn::Path<'p>>,
    {
        let base_model = BartModel::new(p.borrow() / "model", config, generation_mode);
        BartForConditionalGeneration { base_model }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *source_sequence_length*). Must be provided when not running in generation mode
    /// * `attention_mask` - Optional attention mask of shape (*batch size*, *source_sequence_length*) for the encoder positions. Positions with a mask with value 0 will be masked.
    /// * `encoder_outputs` - Optional tuple made of a tensor of shape (*batch size*, *source_sequence_length*, *encoder_hidden_dim*) and optional vectors of tensors of length *num_encoder_layers* with shape (*batch size*, *source_sequence_length*, *hidden_size*).
    /// These correspond to the encoder last hidden state and optional hidden states/attention weights for encoder layers. When provided, the encoder hidden state will not be recalculated. Useful for generation tasks.
    /// * `decoder_input_ids` - Optional input tensor of shape (*batch size*, *target_sequence_length*). Must be provided when running in generation mode (e.g. initialiazed with a BOS token)
    /// * `decoder_attention_mask` - Optional attention mask of shape (*batch size*, *target_sequence_length*) for the decoder positions. Positions with a mask with value 0 will be masked.
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `lm_logits` - `Tensor` of shape (*batch size*, *target_sequence_length*, *vocab_size*) representing the logits for each vocab item and position
    /// * `encoder_hidden_states` - `Tensor` of shape (*batch size*, *source_sequence_length*, *hidden_size*) representing the activations of the last encoder hidden state
    /// * `all_encoder_hidden_states` - `Option<Vec<Tensor>>` of length *num_encoder_layers* with shape (*batch size*, *source_sequence_length*, *hidden_size*)
    /// * `all_encoder_attentions` - `Option<Vec<Tensor>>` of length *num_encoder_layers* with shape (*batch size*, *source_sequence_length*, *hidden_size*)
    /// * `all_decoder_hidden_states` - `Option<Vec<Tensor>>` of length *num_decoder_layers* with shape (*batch size*, *target_sequence_length*, *hidden_size*)
    /// * `all_decoder_attentions` - `Option<Vec<Tensor>>` of length *num_decoder_layers* with shape (*batch size*, *target_sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use tch::{nn, Device, Tensor, no_grad};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::{Int64, Double};
    /// use rust_bert::bart::{BartConfig, BartForConditionalGeneration};
    /// # let config_path = Path::new("path/to/config.json");
    /// # let vocab_path = Path::new("path/to/vocab.txt");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = BartConfig::from_file(config_path);
    /// # let bart_model: BartForConditionalGeneration = BartForConditionalGeneration::new(&vs.root(), &config, false);
    ///  let (batch_size, source_sequence_length, target_sequence_length) = (64, 128, 56);
    ///  let input_tensor = Tensor::rand(&[batch_size, source_sequence_length], (Int64, device));
    ///  let target_tensor = Tensor::rand(&[batch_size, target_sequence_length], (Int64, device));
    ///  let encoder_attention_mask = Tensor::ones(&[batch_size, source_sequence_length], (Int64, device));
    ///  let decoder_attention_mask = Tensor::ones(&[batch_size, source_sequence_length], (Int64, device));
    ///
    ///  let (decoder_output, encoder_hidden_states, cache,
    ///       all_encoder_hidden_states, all_encoder_attentions,
    ///       all_decoder_hidden_states, all_decoder_attentions) = no_grad(|| {
    ///    bart_model
    ///         .forward_t(Some(&input_tensor),
    ///                    Some(&encoder_attention_mask),
    ///                    None,
    ///                    Some(&target_tensor),
    ///                    Some(&decoder_attention_mask),
    ///                    None,
    ///                    false)
    ///    });
    /// ```
    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        encoder_outputs: Option<(Tensor, Option<Vec<Tensor>>, Option<Vec<Tensor>>)>,
        decoder_input_ids: Option<&Tensor>,
        decoder_attention_mask: Option<&Tensor>,
        old_layer_states: Option<Vec<(Option<LayerState>, Option<LayerState>)>>,
        train: bool,
    ) -> (
        Tensor,
        Tensor,
        Option<Vec<(Option<LayerState>, Option<LayerState>)>>,
        Option<Vec<Tensor>>,
        Option<Vec<Tensor>>,
        Option<Vec<Tensor>>,
        Option<Vec<Tensor>>,
    ) {
        let (
            decoder_outputs,
            encoder_hidden_states,
            decoder_cache,
            all_decoder_hidden_states,
            all_decoder_attentions,
            all_encoder_hidden_states,
            all_encoder_attentions,
        ) = self.base_model.forward_t(
            input_ids,
            attention_mask,
            decoder_input_ids,
            encoder_outputs,
            decoder_attention_mask,
            old_layer_states,
            train,
        );

        let lm_logits = decoder_outputs.linear::<Tensor>(&self.base_model.embeddings.ws, None);
        (
            lm_logits,
            encoder_hidden_states,
            decoder_cache,
            all_decoder_hidden_states,
            all_decoder_attentions,
            all_encoder_hidden_states,
            all_encoder_attentions,
        )
    }

    pub fn encode(&self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> Tensor {
        let (encoder_hidden_states, _, _) = self.base_model.encoder.forward_t(
            input_ids,
            attention_mask,
            &self.base_model.embeddings,
            false,
        );
        encoder_hidden_states
    }
}

pub struct BartClassificationHead {
    dense: nn::Linear,
    dropout: Dropout,
    out_proj: nn::Linear,
}

impl BartClassificationHead {
    pub fn new<'p, P>(p: P, config: &BartConfig) -> BartClassificationHead
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let num_labels = config
            .id2label
            .as_ref()
            .expect("num_labels not provided in configuration")
            .len() as i64;
        let dense = nn::linear(
            p / "dense",
            config.d_model,
            config.d_model,
            Default::default(),
        );
        let dropout = Dropout::new(config.classif_dropout);
        let out_proj = nn::linear(
            p / "out_proj",
            config.d_model,
            num_labels,
            Default::default(),
        );

        BartClassificationHead {
            dense,
            dropout,
            out_proj,
        }
    }

    pub fn forward_t(&self, x: &Tensor, train: bool) -> Tensor {
        x.apply_t(&self.dropout, train)
            .apply(&self.dense)
            .tanh()
            .apply_t(&self.dropout, train)
            .apply(&self.out_proj)
    }
}

/// # BART Model for sequence classification
/// BART model with a classification head
/// It is made of the following blocks:
/// - `base_model`: `BartModel` Base BART model
/// - `classification_head`: `BartClassificationHead` made of 2 linear layers mapping hidden states to a target class
/// - `eos_token_id`: token id for the EOS token carrying the pooled representation for classification
pub struct BartForSequenceClassification {
    base_model: BartModel,
    classification_head: BartClassificationHead,
    eos_token_id: i64,
}

impl BartForSequenceClassification {
    /// Build a new `BartForSequenceClassification`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the BART model
    /// * `config` - `BartConfig` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::bart::{BartConfig, BartForSequenceClassification};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = BartConfig::from_file(config_path);
    /// let generation_mode = true;
    /// let bart: BartForSequenceClassification =
    ///     BartForSequenceClassification::new(&p.root() / "bart", &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &BartConfig) -> BartForSequenceClassification
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let base_model = BartModel::new(p / "model", config, false);
        let classification_head = BartClassificationHead::new(p / "classification_head", config);
        let eos_token_id = match config.eos_token_id {
            Some(value) => value,
            None => 3,
        };
        BartForSequenceClassification {
            base_model,
            classification_head,
            eos_token_id,
        }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *source_sequence_length*). Must be provided when not running in generation mode
    /// * `attention_mask` - Optional attention mask of shape (*batch size*, *source_sequence_length*) for the encoder positions. Positions with a mask with value 0 will be masked.
    /// * `encoder_outputs` - Optional tuple made of a tensor of shape (*batch size*, *source_sequence_length*, *encoder_hidden_dim*) and optional vectors of tensors of length *num_encoder_layers* with shape (*batch size*, *source_sequence_length*, *hidden_size*).
    /// These correspond to the encoder last hidden state and optional hidden states/attention weights for encoder layers. When provided, the encoder hidden state will not be recalculated. Useful for generation tasks.
    /// * `decoder_input_ids` - Optional input tensor of shape (*batch size*, *target_sequence_length*). Must be provided when running in generation mode (e.g. initialiazed with a BOS token)
    /// * `decoder_attention_mask` - Optional attention mask of shape (*batch size*, *target_sequence_length*) for the decoder positions. Positions with a mask with value 0 will be masked.
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `logits` - `Tensor` of shape (*batch size*, *num_classes*) representing the logits for each class item and batch item
    /// * `encoder_hidden_states` - `Tensor` of shape (*batch size*, *source_sequence_length*, *hidden_size*) representing the activations of the last encoder hidden state
    /// * `all_encoder_hidden_states` - `Option<Vec<Tensor>>` of length *num_encoder_layers* with shape (*batch size*, *source_sequence_length*, *hidden_size*)
    /// * `all_encoder_attentions` - `Option<Vec<Tensor>>` of length *num_encoder_layers* with shape (*batch size*, *source_sequence_length*, *hidden_size*)
    /// * `all_decoder_hidden_states` - `Option<Vec<Tensor>>` of length *num_decoder_layers* with shape (*batch size*, *target_sequence_length*, *hidden_size*)
    /// * `all_decoder_attentions` - `Option<Vec<Tensor>>` of length *num_decoder_layers* with shape (*batch size*, *target_sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use tch::{nn, Device, Tensor, no_grad};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::{Int64, Double};
    /// use rust_bert::bart::{BartConfig, BartForConditionalGeneration};
    /// # let config_path = Path::new("path/to/config.json");
    /// # let vocab_path = Path::new("path/to/vocab.txt");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = BartConfig::from_file(config_path);
    /// # let bart_model: BartForConditionalGeneration = BartForConditionalGeneration::new(&vs.root(), &config, false);
    ///  let (batch_size, source_sequence_length, target_sequence_length) = (64, 128, 56);
    ///  let input_tensor = Tensor::rand(&[batch_size, source_sequence_length], (Int64, device));
    ///  let target_tensor = Tensor::rand(&[batch_size, target_sequence_length], (Int64, device));
    ///  let encoder_attention_mask = Tensor::ones(&[batch_size, source_sequence_length], (Int64, device));
    ///  let decoder_attention_mask = Tensor::ones(&[batch_size, source_sequence_length], (Int64, device));
    ///
    ///  let (decoder_output, encoder_hidden_states, cache,
    ///       all_encoder_hidden_states, all_encoder_attentions,
    ///       all_decoder_hidden_states, all_decoder_attentions) = no_grad(|| {
    ///    bart_model
    ///         .forward_t(Some(&input_tensor),
    ///                    Some(&encoder_attention_mask),
    ///                    None,
    ///                    Some(&target_tensor),
    ///                    Some(&decoder_attention_mask),
    ///                    None,
    ///                    false)
    ///    });
    /// ```
    pub fn forward_t(
        &self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
        encoder_outputs: Option<(Tensor, Option<Vec<Tensor>>, Option<Vec<Tensor>>)>,
        decoder_input_ids: Option<&Tensor>,
        decoder_attention_mask: Option<&Tensor>,
        train: bool,
    ) -> (
        Tensor,
        Tensor,
        Option<Vec<Tensor>>,
        Option<Vec<Tensor>>,
        Option<Vec<Tensor>>,
        Option<Vec<Tensor>>,
    ) {
        let (
            decoder_outputs,
            encoder_hidden_states,
            _,
            all_decoder_hidden_states,
            all_decoder_attentions,
            all_encoder_hidden_states,
            all_encoder_attentions,
        ) = self.base_model.forward_t(
            Some(input_ids),
            attention_mask,
            decoder_input_ids,
            encoder_outputs,
            decoder_attention_mask,
            None,
            train,
        );

        let eos_mask = input_ids.eq(self.eos_token_id);
        let reshape = eos_mask.sum1(&[1], true, Int64);
        let sentence_representation = decoder_outputs
            .permute(&[2, 0, 1])
            .masked_select(&eos_mask)
            .view((-1, reshape.size()[0] * reshape.int64_value(&[0, 0])))
            .transpose(0, 1)
            .view((
                decoder_outputs.size()[0],
                -1,
                *decoder_outputs.size().last().unwrap(),
            ))
            .select(1, -1);

        let logits = self
            .classification_head
            .forward_t(&sentence_representation, train);
        (
            logits,
            encoder_hidden_states,
            all_decoder_hidden_states,
            all_decoder_attentions,
            all_encoder_hidden_states,
            all_encoder_attentions,
        )
    }
}

impl LMHeadModel for BartForConditionalGeneration {
    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see `input_embeds`)
    /// * `layer_past` - Optional vector of length `num_layers` containing tuples of optional `LayerStates` containing th elast calculated key and value pairs for the decoder. This avoids recomputing attention weights at past positions and speeds up decoding.
    /// * `attention_mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `input_embeds` - Unused for BART
    /// * `token_type_ids` - Unused for BART
    /// * `position_ids` - Unused for BART
    /// * `encoder_outputs` - Optional tensor of shape (*batch size*, *source_sequence_length*, *hidden_size*). When provided, the encoder hidden state will not be recalculated. Useful for generation tasks.
    /// * `decoder_input_ids` - Optional input tensor of shape (*batch size*, *target_sequence_length*). Must be provided when running in generation mode (e.g. initialized with a BOS token)
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    ///
    /// # Returns
    ///
    /// * `lm_logits` - `Tensor` of shape (*batch size*, *sequence_length*, *vocab_size*) representing the logits for each vocab item and position
    /// * `past` - `BartCache` made of `Option<Vec<(Option<Vec<&LayerState, &LayerState>>)>>` of length *n_layer* containing the encoder past keys and values for
    ///     both the self attention and the encoder cross attention of each layer of the decoder.
    /// * `encoder_hidden_states` - `Option<Tensor>` Hidden states for the encoder
    /// * `hidden_states` - None
    /// * `attentions` - None
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use tch::{nn, Device, Tensor, no_grad};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::{Int64, Double};
    /// use rust_bert::pipelines::generation::LMHeadModel;
    /// use rust_bert::bart::{BartForConditionalGeneration, BartConfig};
    /// # let config_path = Path::new("path/to/config.json");
    /// # let vocab_path = Path::new("path/to/vocab.txt");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = BartConfig::from_file(config_path);
    /// # let bart_model: BartForConditionalGeneration = BartForConditionalGeneration::new(&vs.root(), &config, false);
    ///  let (batch_size, source_sequence_length, target_sequence_length) = (64, 128, 56);
    ///  let input_tensor = Tensor::rand(&[batch_size, source_sequence_length], (Int64, device));
    ///  let target_tensor = Tensor::rand(&[batch_size, target_sequence_length], (Int64, device));
    ///  let encoder_attention_mask = Tensor::ones(&[batch_size, source_sequence_length], (Int64, device));
    ///  let decoder_attention_mask = Tensor::ones(&[batch_size, source_sequence_length], (Int64, device));
    ///
    ///  let (decoder_output, encoder_hidden_states, cache,
    ///       all_encoder_hidden_states, all_encoder_attentions,
    ///       all_decoder_hidden_states, all_decoder_attentions) = no_grad(|| {
    ///    bart_model
    ///         .forward_t(Some(&input_tensor),
    ///                    Some(&encoder_attention_mask),
    ///                    None,
    ///                    Some(&target_tensor),
    ///                    Some(&decoder_attention_mask),
    ///                    None,
    ///                    false)
    ///    });
    /// ```
    fn forward_t(
        &self,
        input_ids: &Option<Tensor>,
        cache: Cache,
        attention_mask: &Option<Tensor>,
        _token_type_ids: &Option<Tensor>,
        _position_ids: &Option<Tensor>,
        _input_embeds: &Option<Tensor>,
        encoder_outputs: Option<&Tensor>,
        decoder_input_ids: &Option<Tensor>,
        train: bool,
    ) -> Result<
        (
            Tensor,
            Option<Tensor>,
            Cache,
            Option<Vec<Tensor>>,
            Option<Vec<Tensor>>,
        ),
        &'static str,
    > {
        let (decoder_output, encoder_hidden_states, new_cache, _, _, _, _) = match cache {
            Cache::BARTCache(cached_layer_states) => self.base_model.forward_t(
                input_ids.as_ref(),
                attention_mask.as_ref(),
                decoder_input_ids.as_ref(),
                Some((encoder_outputs.as_ref().unwrap().copy(), None, None)),
                None,
                cached_layer_states,
                train,
            ),

            Cache::None => self.base_model.forward_t(
                input_ids.as_ref(),
                attention_mask.as_ref(),
                decoder_input_ids.as_ref(),
                Some((encoder_outputs.as_ref().unwrap().copy(), None, None)),
                None,
                None,
                train,
            ),
            _ => Err("Cache not compatible with BART Model")?,
        };

        let lm_logits = decoder_output.linear::<Tensor>(&self.base_model.embeddings.ws, None);
        Ok((
            lm_logits,
            Some(encoder_hidden_states),
            Cache::BARTCache(new_cache),
            None,
            None,
        ))
    }
}
