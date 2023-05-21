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
use crate::common::activations::Activation;
use crate::common::dropout::Dropout;
use crate::common::kind::get_min;
use crate::pipelines::common::{ModelType, TokenizerOption};
use crate::pipelines::generation_utils::private_generation_utils::{
    PreparedInput, PrivateLanguageGenerator,
};
use crate::pipelines::generation_utils::{Cache, GenerateConfig, LMModelOutput, LanguageGenerator};
use crate::{Config, RustBertError};
use rust_tokenizers::tokenizer::TruncationStrategy;

use serde::{Deserialize, Serialize};
use std::borrow::Borrow;
use std::collections::HashMap;
use tch::nn::{embedding, EmbeddingConfig};
use tch::{nn, Device, Kind, Tensor};

/// # BART Pretrained model weight files
pub struct BartModelResources;

/// # BART Pretrained model config files
pub struct BartConfigResources;

/// # BART Pretrained model vocab files
pub struct BartVocabResources;

/// # BART Pretrained model merges files
pub struct BartMergesResources;

impl BartModelResources {
    /// Shared under MIT license by the Facebook AI Research Fairseq team at <https://github.com/pytorch/fairseq>. Modified with conversion to C-array format.
    pub const BART: (&'static str, &'static str) = (
        "bart/model",
        "https://huggingface.co/facebook/bart-large/resolve/main/rust_model.ot",
    );
    /// Shared under MIT license by the Facebook AI Research Fairseq team at <https://github.com/pytorch/fairseq>. Modified with conversion to C-array format.
    pub const BART_CNN: (&'static str, &'static str) = (
        "bart-cnn/model",
        "https://huggingface.co/facebook/bart-large-cnn/resolve/main/rust_model.ot",
    );
    /// Shared under MIT license by the Facebook AI Research Fairseq team at <https://github.com/pytorch/fairseq>. Modified with conversion to C-array format.
    pub const BART_XSUM: (&'static str, &'static str) = (
        "bart-xsum/model",
        "https://huggingface.co/facebook/bart-large-xsum/resolve/main/rust_model.ot",
    );
    /// Shared under MIT license by the Facebook AI Research Fairseq team at <https://github.com/pytorch/fairseq>. Modified with conversion to C-array format.
    pub const BART_MNLI: (&'static str, &'static str) = (
        "bart-large-mnli/model",
        "https://huggingface.co/facebook/bart-large-mnli/resolve/main/rust_model.ot",
    );
    /// Shared under Apache 2.0 license by the Hugging Face team at <https://huggingface.co/sshleifer/distilbart-cnn-6-6>. Modified with conversion to C-array format.
    pub const DISTILBART_CNN_6_6: (&'static str, &'static str) = (
        "distilbart-cnn-6-6/model",
        "https://huggingface.co/sshleifer/distilbart-cnn-6-6/resolve/main/rust_model.ot",
    );
    /// Shared under Apache 2.0 license by the Hugging Face team at <https://huggingface.co/sshleifer/distilbart-cnn-12-6>. Modified with conversion to C-array format.
    pub const DISTILBART_CNN_12_6: (&'static str, &'static str) = (
        "distilbart-cnn-12-6/model",
        "https://huggingface.co/sshleifer/distilbart-cnn-12-6/resolve/main/rust_model.ot",
    );
}

impl BartConfigResources {
    /// Shared under MIT license by the Facebook AI Research Fairseq team at <https://github.com/pytorch/fairseq>. Modified with conversion to C-array format.
    pub const BART: (&'static str, &'static str) = (
        "bart/config",
        "https://huggingface.co/facebook/bart-large/resolve/main/config.json",
    );
    /// Shared under MIT license by the Facebook AI Research Fairseq team at <https://github.com/pytorch/fairseq>. Modified with conversion to C-array format.
    pub const BART_CNN: (&'static str, &'static str) = (
        "bart-cnn/config",
        "https://huggingface.co/facebook/bart-large-cnn/resolve/main/config.json",
    );
    /// Shared under MIT license by the Facebook AI Research Fairseq team at <https://github.com/pytorch/fairseq>. Modified with conversion to C-array format.
    pub const BART_XSUM: (&'static str, &'static str) = (
        "bart-xsum/config",
        "https://huggingface.co/facebook/bart-large-xsum/resolve/main/config.json",
    );
    /// Shared under MIT license by the Facebook AI Research Fairseq team at <https://github.com/pytorch/fairseq>. Modified with conversion to C-array format.
    pub const BART_MNLI: (&'static str, &'static str) = (
        "bart-large-mnli/config",
        "https://huggingface.co/facebook/bart-large-mnli/resolve/main/config.json",
    );
    /// Shared under Apache 2.0 license by the Hugging Face team at <https://huggingface.co/sshleifer/distilbart-cnn-6-6>. Modified with conversion to C-array format.
    pub const DISTILBART_CNN_6_6: (&'static str, &'static str) = (
        "distilbart-cnn-6-6/config",
        "https://cdn.huggingface.co/sshleifer/distilbart-cnn-6-6/config.json",
    );
    /// Shared under Apache 2.0 license by the Hugging Face team at <https://huggingface.co/sshleifer/distilbart-cnn-12-6>. Modified with conversion to C-array format.
    pub const DISTILBART_CNN_12_6: (&'static str, &'static str) = (
        "distilbart-cnn-12-6/config",
        "https://cdn.huggingface.co/sshleifer/distilbart-cnn-12-6/config.json",
    );
}

impl BartVocabResources {
    /// Shared under MIT license by the Facebook AI Research Fairseq team at <https://github.com/pytorch/fairseq>. Modified with conversion to C-array format.
    pub const BART: (&'static str, &'static str) = (
        "bart/vocab",
        "https://huggingface.co/roberta-large/resolve/main/vocab.json",
    );
    /// Shared under MIT license by the Facebook AI Research Fairseq team at <https://github.com/pytorch/fairseq>. Modified with conversion to C-array format.
    pub const BART_CNN: (&'static str, &'static str) = (
        "bart-cnn/vocab",
        "https://huggingface.co/roberta-large/resolve/main/vocab.json",
    );
    /// Shared under MIT license by the Facebook AI Research Fairseq team at <https://github.com/pytorch/fairseq>. Modified with conversion to C-array format.
    pub const BART_XSUM: (&'static str, &'static str) = (
        "bart-xsum/vocab",
        "https://huggingface.co/roberta-large/resolve/main/vocab.json",
    );
    /// Shared under MIT license by the Facebook AI Research Fairseq team at <https://github.com/pytorch/fairseq>. Modified with conversion to C-array format.
    pub const BART_MNLI: (&'static str, &'static str) = (
        "bart-large-mnli/vocab",
        "https://huggingface.co/roberta-large/resolve/main/vocab.json",
    );
    /// Shared under Apache 2.0 license by the Hugging Face team at <https://huggingface.co/sshleifer/distilbart-cnn-6-6>. Modified with conversion to C-array format.
    pub const DISTILBART_CNN_6_6: (&'static str, &'static str) = (
        "distilbart-cnn-6-6/vocab",
        "https://cdn.huggingface.co/sshleifer/distilbart-cnn-6-6/vocab.json",
    );
    /// Shared under Apache 2.0 license by the Hugging Face team at <https://huggingface.co/sshleifer/distilbart-cnn-12-6>. Modified with conversion to C-array format.
    pub const DISTILBART_CNN_12_6: (&'static str, &'static str) = (
        "distilbart-cnn-12-6/vocab",
        "https://cdn.huggingface.co/sshleifer/distilbart-cnn-12-6/vocab.json",
    );
}

impl BartMergesResources {
    /// Shared under MIT license by the Facebook AI Research Fairseq team at <https://github.com/pytorch/fairseq>. Modified with conversion to C-array format.
    pub const BART: (&'static str, &'static str) = (
        "bart/merges",
        "https://huggingface.co/roberta-large/resolve/main/merges.txt",
    );
    /// Shared under MIT license by the Facebook AI Research Fairseq team at <https://github.com/pytorch/fairseq>. Modified with conversion to C-array format.
    pub const BART_CNN: (&'static str, &'static str) = (
        "bart-cnn/merges",
        "https://huggingface.co/roberta-large/resolve/main/merges.txt",
    );
    /// Shared under MIT license by the Facebook AI Research Fairseq team at <https://github.com/pytorch/fairseq>. Modified with conversion to C-array format.
    pub const BART_XSUM: (&'static str, &'static str) = (
        "bart-xsum/merges",
        "https://huggingface.co/roberta-large/resolve/main/merges.txt",
    );
    /// Shared under MIT license by the Facebook AI Research Fairseq team at <https://github.com/pytorch/fairseq>. Modified with conversion to C-array format.
    pub const BART_MNLI: (&'static str, &'static str) = (
        "bart-large-mnli/merges",
        "https://huggingface.co/roberta-large/resolve/main/merges.txt",
    );
    /// Shared under Apache 2.0 license by the Hugging Face team at <https://huggingface.co/sshleifer/distilbart-cnn-6-6>. Modified with conversion to C-array format.
    pub const DISTILBART_CNN_6_6: (&'static str, &'static str) = (
        "distilbart-cnn-6-6/merges",
        "https://cdn.huggingface.co/sshleifer/distilbart-cnn-6-6/merges.txt",
    );
    /// Shared under Apache 2.0 license by the Hugging Face team at <https://huggingface.co/sshleifer/distilbart-cnn-12-6>. Modified with conversion to C-array format.
    pub const DISTILBART_CNN_12_6: (&'static str, &'static str) = (
        "distilbart-cnn-12-6/merges",
        "https://cdn.huggingface.co/sshleifer/distilbart-cnn-12-6/merges.txt",
    );
}

#[derive(Debug, Serialize, Deserialize, Clone)]
/// # BART model configuration
/// Defines the BART model architecture (e.g. number of layers, hidden layer size, label mapping...)
pub struct BartConfig {
    pub num_labels: Option<i64>,
    pub activation_function: Option<Activation>,
    pub activation_dropout: f64,
    pub attention_dropout: f64,
    pub classif_dropout: Option<f64>,
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

impl Config for BartConfig {}

impl Default for BartConfig {
    fn default() -> Self {
        BartConfig {
            num_labels: Some(3),
            activation_function: Some(Activation::gelu),
            activation_dropout: 0.0,
            attention_dropout: 0.0,
            classif_dropout: Some(0.0),
            d_model: 1024,
            decoder_attention_heads: 16,
            decoder_ffn_dim: 4096,
            decoder_layerdrop: 0.0,
            decoder_layers: 12,
            decoder_start_token_id: Some(2),
            dropout: 0.1,
            encoder_attention_heads: 16,
            encoder_ffn_dim: 4096,
            encoder_layerdrop: 0.0,
            encoder_layers: 12,
            bos_token_id: Some(0),
            eos_token_id: Some(2),
            pad_token_id: Some(1),
            id2label: None,
            label2id: None,
            init_std: 0.02,
            is_decoder: None,
            is_encoder_decoder: Some(true),
            max_position_embeddings: 1024,
            min_length: None,
            no_repeat_ngram_size: None,
            normalize_embedding: Some(true),
            num_hidden_layers: 12,
            output_attentions: None,
            output_hidden_states: None,
            output_past: None,
            static_position_embeddings: None,
            scale_embedding: Some(false),
            vocab_size: 50265,
        }
    }
}

pub(crate) fn _make_causal_mask(
    input_ids_shape: &[i64],
    dtype: Kind,
    device: Device,
    past_key_values_length: i64,
) -> Tensor {
    let batch_size = input_ids_shape[0];
    let target_length = input_ids_shape[1];

    let mut mask = Tensor::full(
        [target_length, target_length],
        get_min(dtype).unwrap(),
        (dtype, device),
    );
    let mask_cond = Tensor::arange(target_length, (dtype, device));
    let _ = mask.masked_fill_(
        &mask_cond.lt_tensor(&(&mask_cond + 1).view([target_length, 1])),
        0,
    );

    if past_key_values_length > 0 {
        mask = Tensor::cat(
            &[
                Tensor::zeros([target_length, past_key_values_length], (dtype, device)),
                mask,
            ],
            -1,
        );
    }
    mask.unsqueeze(0).unsqueeze(0).expand(
        [
            batch_size,
            1,
            target_length,
            target_length + past_key_values_length,
        ],
        true,
    )
}

pub(crate) fn _expand_mask(mask: &Tensor, target_length: Option<i64>, dtype: Kind) -> Tensor {
    let (batch_size, source_length) = mask.size2().unwrap();
    let target_length = target_length.unwrap_or(source_length);
    let expanded_mask = mask
        .unsqueeze(1)
        .unsqueeze(1)
        .expand([batch_size, 1, target_length, source_length], true)
        .totype(dtype);
    let inverted_mask: Tensor = 1 - expanded_mask;
    inverted_mask.masked_fill(&inverted_mask.to_kind(Kind::Bool), get_min(dtype).unwrap())
}

pub(crate) fn _prepare_decoder_attention_mask(
    attention_mask: Option<&Tensor>,
    input_shape: &[i64],
    input_embeds: &Tensor,
    past_key_values_length: i64,
) -> Option<Tensor> {
    let last_input_shape_dim = *input_shape.last().unwrap();
    let mut combined_attention_mask = if last_input_shape_dim > 1 {
        Some(_make_causal_mask(
            input_shape,
            input_embeds.kind(),
            input_embeds.device(),
            past_key_values_length,
        ))
    } else {
        None
    };

    if let Some(attention_mask) = attention_mask {
        let expanded_attention_mask = _expand_mask(
            attention_mask,
            Some(last_input_shape_dim),
            input_embeds.kind(),
        );
        combined_attention_mask = match combined_attention_mask {
            Some(value) => Some(value + expanded_attention_mask),
            None => Some(expanded_attention_mask),
        };
    }

    combined_attention_mask
}

fn _shift_tokens_right(input_ids: &Tensor, pad_token_id: i64) -> Tensor {
    let index_eos: Tensor =
        input_ids
            .ne(pad_token_id)
            .sum_dim_intlist([-1].as_slice(), true, Kind::Int64)
            - 1;
    let output = input_ids.empty_like().to_kind(Kind::Int64);
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
/// - `pad_token_id`: padding token id
pub struct BartModel {
    pub(crate) encoder: BartEncoder,
    decoder: BartDecoder,
    pub(crate) embeddings: nn::Embedding,
    pad_token_id: i64,
}

impl BartModel {
    /// Build a new `BartModel`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the BART model
    /// * `config` - `BartConfig` object defining the model architecture
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
    /// let bart: BartModel = BartModel::new(&p.root() / "bart", &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &BartConfig) -> BartModel
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let pad_token_id = config.pad_token_id.unwrap_or(1);
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
        let decoder = BartDecoder::new(p / "decoder", config);

        BartModel {
            encoder,
            decoder,
            embeddings,
            pad_token_id,
        }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *source_sequence_length*). Must be provided when not running in generation mode
    /// * `attention_mask` - Optional attention mask of shape (*batch size*, *source_sequence_length*) for the encoder positions. Positions with a mask with value 0 will be masked.
    /// * `decoder_input_ids` - Optional input tensor of shape (*batch size*, *target_sequence_length*). Must be provided when running in generation mode (e.g. initialized with a BOS token)
    /// * `encoder_outputs` - Optional tuple made of a tensor of shape (*batch size*, *source_sequence_length*, *encoder_hidden_dim*) and optional vectors of tensors of length *num_encoder_layers* with shape (*batch size*, *source_sequence_length*, *hidden_size*).
    /// These correspond to the encoder last hidden state and optional hidden states/attention weights for encoder layers. When provided, the encoder hidden state will not be recalculated. Useful for generation tasks.
    /// * `decoder_attention_mask` - Optional attention mask of shape (*batch size*, *target_sequence_length*) for the decoder positions. Positions with a mask with value 0 will be masked.
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `BartModelOutput` containing:
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
    /// use rust_bert::bart::{BartConfig, BartModel};
    /// # let config_path = Path::new("path/to/config.json");
    /// # let vocab_path = Path::new("path/to/vocab.txt");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = BartConfig::from_file(config_path);
    /// # let bart_model: BartModel = BartModel::new(&vs.root(), &config);
    /// let (batch_size, source_sequence_length, target_sequence_length) = (64, 128, 56);
    /// let input_tensor = Tensor::rand(&[batch_size, source_sequence_length], (Int64, device));
    /// let target_tensor = Tensor::rand(&[batch_size, target_sequence_length], (Int64, device));
    /// let encoder_attention_mask =
    ///     Tensor::ones(&[batch_size, source_sequence_length], (Int64, device));
    /// let decoder_attention_mask =
    ///     Tensor::ones(&[batch_size, source_sequence_length], (Int64, device));
    ///
    /// let model_output = no_grad(|| {
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
        encoder_output: Option<&Tensor>,
        decoder_attention_mask: Option<&Tensor>,
        layer_states: Option<Vec<(Option<LayerState>, Option<LayerState>)>>,
        train: bool,
    ) -> BartModelOutput {
        let calc_decoder_input_ids = if decoder_input_ids.is_none() {
            Some(_shift_tokens_right(input_ids.unwrap(), self.pad_token_id))
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
            decoder_input_ids,
            encoder_output,
            attention_mask,
            decoder_attention_mask,
            &self.embeddings,
            layer_states,
            train,
        );
        BartModelOutput {
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
    /// let bart: BartForConditionalGeneration =
    ///     BartForConditionalGeneration::new(&p.root() / "bart", &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &BartConfig) -> BartForConditionalGeneration
    where
        P: Borrow<nn::Path<'p>>,
    {
        let base_model = BartModel::new(p.borrow() / "model", config);
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
    /// * `decoder_input_ids` - Optional input tensor of shape (*batch size*, *target_sequence_length*). Must be provided when running in generation mode (e.g. initialized with a BOS token)
    /// * `decoder_attention_mask` - Optional attention mask of shape (*batch size*, *target_sequence_length*) for the decoder positions. Positions with a mask with value 0 will be masked.
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `BartModelOutput` containing:
    ///   - `decoder_output` - `Tensor` of shape (*batch size*, *target_sequence_length*, *vocab_size*) representing the logits for each vocabulary item and position
    ///   - `encoder_hidden_states` - `Tensor` of shape (*batch size*, *source_sequence_length*, *hidden_size*) representing the activations of the last encoder hidden state
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
    /// use rust_bert::bart::{BartConfig, BartForConditionalGeneration};
    /// # let config_path = Path::new("path/to/config.json");
    /// # let vocab_path = Path::new("path/to/vocab.txt");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = BartConfig::from_file(config_path);
    /// # let bart_model: BartForConditionalGeneration = BartForConditionalGeneration::new(&vs.root(), &config);
    ///  let (batch_size, source_sequence_length, target_sequence_length) = (64, 128, 56);
    ///  let input_tensor = Tensor::rand(&[batch_size, source_sequence_length], (Int64, device));
    ///  let target_tensor = Tensor::rand(&[batch_size, target_sequence_length], (Int64, device));
    ///  let encoder_attention_mask = Tensor::ones(&[batch_size, source_sequence_length], (Int64, device));
    ///  let decoder_attention_mask = Tensor::ones(&[batch_size, source_sequence_length], (Int64, device));
    ///
    ///  let model_output = no_grad(|| {
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
        encoder_output: Option<&Tensor>,
        decoder_input_ids: Option<&Tensor>,
        decoder_attention_mask: Option<&Tensor>,
        old_layer_states: Option<Vec<(Option<LayerState>, Option<LayerState>)>>,
        train: bool,
    ) -> BartModelOutput {
        let base_model_output = self.base_model.forward_t(
            input_ids,
            attention_mask,
            decoder_input_ids,
            encoder_output,
            decoder_attention_mask,
            old_layer_states,
            train,
        );

        let lm_logits = base_model_output
            .decoder_output
            .linear::<Tensor>(&self.base_model.embeddings.ws, None);
        BartModelOutput {
            decoder_output: lm_logits,
            ..base_model_output
        }
    }

    pub fn encode(&self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> Tensor {
        self.base_model
            .encoder
            .forward_t(
                input_ids,
                attention_mask,
                &self.base_model.embeddings,
                false,
            )
            .hidden_state
    }
}

pub struct BartClassificationHead {
    dense: nn::Linear,
    dropout: Dropout,
    out_proj: nn::Linear,
}

impl BartClassificationHead {
    pub fn new<'p, P>(p: P, config: &BartConfig) -> Result<BartClassificationHead, RustBertError>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let num_labels = config
            .id2label
            .as_ref()
            .ok_or_else(|| {
                RustBertError::InvalidConfigurationError(
                    "num_labels not provided in configuration".to_string(),
                )
            })?
            .len() as i64;
        let dense = nn::linear(
            p / "dense",
            config.d_model,
            config.d_model,
            Default::default(),
        );
        let dropout = Dropout::new(config.classif_dropout.unwrap_or(0.0));
        let out_proj = nn::linear(
            p / "out_proj",
            config.d_model,
            num_labels,
            Default::default(),
        );

        Ok(BartClassificationHead {
            dense,
            dropout,
            out_proj,
        })
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
    /// let bart: BartForSequenceClassification =
    ///     BartForSequenceClassification::new(&p.root() / "bart", &config).unwrap();
    /// ```
    pub fn new<'p, P>(
        p: P,
        config: &BartConfig,
    ) -> Result<BartForSequenceClassification, RustBertError>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let base_model = BartModel::new(p / "model", config);
        let classification_head = BartClassificationHead::new(p / "classification_head", config)?;
        let eos_token_id = config.eos_token_id.unwrap_or(3);
        Ok(BartForSequenceClassification {
            base_model,
            classification_head,
            eos_token_id,
        })
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *source_sequence_length*). Must be provided when not running in generation mode
    /// * `attention_mask` - Optional attention mask of shape (*batch size*, *source_sequence_length*) for the encoder positions. Positions with a mask with value 0 will be masked.
    /// * `encoder_outputs` - Optional tuple made of a tensor of shape (*batch size*, *source_sequence_length*, *encoder_hidden_dim*) and optional vectors of tensors of length *num_encoder_layers* with shape (*batch size*, *source_sequence_length*, *hidden_size*).
    /// These correspond to the encoder last hidden state and optional hidden states/attention weights for encoder layers. When provided, the encoder hidden state will not be recalculated. Useful for generation tasks.
    /// * `decoder_input_ids` - Optional input tensor of shape (*batch size*, *target_sequence_length*). Must be provided when running in generation mode (e.g. initialized with a BOS token)
    /// * `decoder_attention_mask` - Optional attention mask of shape (*batch size*, *target_sequence_length*) for the decoder positions. Positions with a mask with value 0 will be masked.
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `BartModelOutput` containing:
    ///   - `decoder_output` - `Tensor` of shape (*batch size*, *num_classes*) representing the activations for each class and batch item
    ///   - `encoder_hidden_states` - `Option<Tensor>` of shape (*batch size*, *source_sequence_length*, *hidden_size*) representing the activations of the last encoder hidden state if it was not provided, otherwise None.
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
    /// use rust_bert::bart::{BartConfig, BartForSequenceClassification};
    /// # let config_path = Path::new("path/to/config.json");
    /// # let vocab_path = Path::new("path/to/vocab.txt");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = BartConfig::from_file(config_path);
    /// # let bart_model: BartForSequenceClassification = BartForSequenceClassification::new(&vs.root(), &config).unwrap();
    ///  let (batch_size, source_sequence_length, target_sequence_length) = (64, 128, 56);
    ///  let input_tensor = Tensor::rand(&[batch_size, source_sequence_length], (Int64, device));
    ///  let target_tensor = Tensor::rand(&[batch_size, target_sequence_length], (Int64, device));
    ///  let encoder_attention_mask = Tensor::ones(&[batch_size, source_sequence_length], (Int64, device));
    ///  let decoder_attention_mask = Tensor::ones(&[batch_size, source_sequence_length], (Int64, device));
    ///
    ///  let model_output = no_grad(|| {
    ///    bart_model
    ///         .forward_t(&input_tensor,
    ///                    Some(&encoder_attention_mask),
    ///                    None,
    ///                    Some(&target_tensor),
    ///                    Some(&decoder_attention_mask),
    ///                    false)
    ///    });
    /// ```
    pub fn forward_t(
        &self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
        encoder_output: Option<&Tensor>,
        decoder_input_ids: Option<&Tensor>,
        decoder_attention_mask: Option<&Tensor>,
        train: bool,
    ) -> BartModelOutput {
        let base_model_output = self.base_model.forward_t(
            Some(input_ids),
            attention_mask,
            decoder_input_ids,
            encoder_output,
            decoder_attention_mask,
            None,
            train,
        );
        let eos_mask = input_ids.eq(self.eos_token_id);
        let reshape = eos_mask.sum_dim_intlist([1].as_slice(), true, input_ids.kind());
        let sentence_representation = base_model_output
            .decoder_output
            .permute([2, 0, 1])
            .masked_select(&eos_mask)
            .view((-1, reshape.size()[0] * reshape.int64_value(&[0, 0])))
            .transpose(0, 1)
            .view((
                base_model_output.decoder_output.size()[0],
                -1,
                *base_model_output.decoder_output.size().last().unwrap(),
            ))
            .select(1, -1);

        let logits = self
            .classification_head
            .forward_t(&sentence_representation, train);
        BartModelOutput {
            decoder_output: logits,
            encoder_hidden_state: base_model_output.encoder_hidden_state,
            cache: None,
            all_decoder_hidden_states: base_model_output.all_decoder_hidden_states,
            all_decoder_attentions: base_model_output.all_decoder_attentions,
            all_encoder_hidden_states: base_model_output.all_encoder_hidden_states,
            all_encoder_attentions: base_model_output.all_encoder_attentions,
        }
    }
}

/// Container holding a BART model output. The decoder output may hold the hidden state of
/// the last layer of the decoder, or may hold logits for a custom head module after the
/// decoder (e.g. for classification or language modeling tasks)
pub struct BartModelOutput {
    /// Hidden state of the last layer of the decoder, or logits for a custom head
    /// module after the decoder (e.g. for classification or language modeling tasks)
    pub decoder_output: Tensor,
    /// Hidden state for the last layer of the encoder if they are calculated (not provided), otherwise None
    pub encoder_hidden_state: Option<Tensor>,
    /// Cached outputs of the model (attention layers keys and values) if the model is used for generation
    pub cache: Option<Vec<(Option<LayerState>, Option<LayerState>)>>,
    /// Hidden states for all layers of the decoder
    pub all_decoder_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all layers of the decoder
    pub all_decoder_attentions: Option<Vec<Tensor>>,
    /// Hidden states for all layers of the encoder
    pub all_encoder_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all layers of the encoder
    pub all_encoder_attentions: Option<Vec<Tensor>>,
}

/// # Language generation model based on the Bart architecture
pub struct BartGenerator {
    model: BartForConditionalGeneration,
    tokenizer: TokenizerOption,
    var_store: nn::VarStore,
    generate_config: GenerateConfig,
    bos_token_id: Option<i64>,
    eos_token_ids: Option<Vec<i64>>,
    pad_token_id: Option<i64>,
    is_encoder_decoder: bool,
    vocab_size: i64,
    decoder_start_id: Option<i64>,
    max_position_embeddings: i64,
}

impl BartGenerator {
    /// Build a new `BartGenerator`
    ///
    /// # Arguments
    ///
    /// * `vocab_path` - Path to the model vocabulary, expected to have a structure following the [Transformers library](https://github.com/huggingface/transformers) convention
    /// * `merges_path` - Path to the bpe merges, expected to have a structure following the [Transformers library](https://github.com/huggingface/transformers) convention
    /// * `config_path` - Path to the model configuration, expected to have a structure following the [Transformers library](https://github.com/huggingface/transformers) convention
    /// * `weights_path` - Path to the model weight files. These need to be converted form the `.bin` to `.ot` format using the utility script provided.
    /// * `device` - Device to run the model on, e.g. `Device::Cpu` or `Device::Cuda(0)`
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use std::path::PathBuf;
    /// # use tch::Device;
    /// # fn main() -> anyhow::Result<()> {
    /// use rust_bert::bart::BartGenerator;
    /// use rust_bert::pipelines::generation_utils::GenerateConfig;
    /// # let mut home: PathBuf = dirs::home_dir().unwrap();
    /// # home.push("rustbert");
    /// # home.push("openai-gpt");
    /// # let config_path = &home.as_path().join("config.json");
    /// # let vocab_path = &home.as_path().join("vocab.txt");
    /// # let merges_path = &home.as_path().join("merges.txt");
    /// # let weights_path = &home.as_path().join("model.ot");
    /// let device = Device::cuda_if_available();
    /// let generate_config = GenerateConfig {
    ///     max_length: Some(30),
    ///     do_sample: true,
    ///     num_beams: 5,
    ///     temperature: 1.1,
    ///     num_return_sequences: 3,
    ///     ..Default::default()
    /// };
    /// let bart_generator = BartGenerator::new(generate_config)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(generate_config: GenerateConfig) -> Result<BartGenerator, RustBertError> {
        let vocab_path = generate_config.vocab_resource.get_local_path()?;
        let merges_path = generate_config
            .merges_resource
            .as_ref()
            .ok_or_else(|| {
                RustBertError::InvalidConfigurationError(
                    "BART expects a merges resources to be provided".to_string(),
                )
            })?
            .get_local_path()?;

        let tokenizer = TokenizerOption::from_file(
            ModelType::Bart,
            vocab_path.to_str().unwrap(),
            Some(merges_path.to_str().unwrap()),
            false,
            None,
            false,
        )?;

        Self::new_with_tokenizer(generate_config, tokenizer)
    }

    pub fn new_with_tokenizer(
        generate_config: GenerateConfig,
        tokenizer: TokenizerOption,
    ) -> Result<BartGenerator, RustBertError> {
        let config_path = generate_config.config_resource.get_local_path()?;
        let weights_path = generate_config.model_resource.get_local_path()?;
        let device = generate_config.device;

        generate_config.validate();
        let mut var_store = nn::VarStore::new(device);
        let config = BartConfig::from_file(config_path);
        let model = BartForConditionalGeneration::new(var_store.root(), &config);
        var_store.load(weights_path)?;

        let bos_token_id = Some(config.bos_token_id.unwrap_or(0));
        let eos_token_ids = Some(match config.eos_token_id {
            Some(value) => vec![value],
            None => vec![2],
        });
        let pad_token_id = Some(config.pad_token_id.unwrap_or(1));
        let vocab_size = config.vocab_size;
        let is_encoder_decoder = true;
        let decoder_start_id = Some(2);
        let max_position_embeddings = config.max_position_embeddings;

        Ok(BartGenerator {
            model,
            tokenizer,
            var_store,
            generate_config,
            bos_token_id,
            eos_token_ids,
            pad_token_id,
            is_encoder_decoder,
            vocab_size,
            decoder_start_id,
            max_position_embeddings,
        })
    }

    fn force_token_id_generation(&self, scores: &mut Tensor, token_ids: &[i64]) {
        let impossible_tokens: Vec<i64> = (0..self.get_vocab_size())
            .filter(|pos| !token_ids.contains(pos))
            .collect();
        let impossible_tokens = Tensor::from_slice(&impossible_tokens).to_device(scores.device());
        let _ = scores.index_fill_(1, &impossible_tokens, f64::NEG_INFINITY);
    }
}

impl PrivateLanguageGenerator for BartGenerator {
    fn _get_tokenizer(&self) -> &TokenizerOption {
        &self.tokenizer
    }
    fn _get_tokenizer_mut(&mut self) -> &mut TokenizerOption {
        &mut self.tokenizer
    }
    fn get_var_store(&self) -> &nn::VarStore {
        &self.var_store
    }
    fn get_var_store_mut(&mut self) -> &mut nn::VarStore {
        &mut self.var_store
    }
    fn get_config(&self) -> &GenerateConfig {
        &self.generate_config
    }
    fn get_bos_id(&self) -> Option<i64> {
        self.bos_token_id
    }
    fn get_eos_ids(&self) -> Option<&Vec<i64>> {
        self.eos_token_ids.as_ref()
    }
    fn get_pad_id(&self) -> Option<i64> {
        self.pad_token_id
    }
    fn is_encoder_decoder(&self) -> bool {
        self.is_encoder_decoder
    }
    fn get_vocab_size(&self) -> i64 {
        self.vocab_size
    }
    fn get_decoder_start_id(&self) -> Option<i64> {
        self.decoder_start_id
    }
    fn get_max_positions_embeddings(&self) -> i64 {
        self.max_position_embeddings
    }

    fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        cache: Cache,
        attention_mask: Option<&Tensor>,
        _token_type_ids: Option<&Tensor>,
        _position_ids: Option<&Tensor>,
        _input_embeds: Option<&Tensor>,
        encoder_outputs: Option<&Tensor>,
        decoder_input_ids: Option<&Tensor>,
        train: bool,
    ) -> Result<LMModelOutput, RustBertError> {
        let base_model_output = match cache {
            Cache::BARTCache(cached_layer_states) => self.model.forward_t(
                input_ids,
                attention_mask,
                encoder_outputs,
                decoder_input_ids,
                None,
                cached_layer_states,
                train,
            ),

            Cache::None => self.model.forward_t(
                input_ids,
                attention_mask,
                encoder_outputs,
                decoder_input_ids,
                None,
                None,
                train,
            ),
            _ => {
                return Err(RustBertError::ValueError(
                    "Cache not compatible with BART Model".into(),
                ));
            }
        };

        Ok(LMModelOutput {
            lm_logits: base_model_output.decoder_output,
            cache: Cache::BARTCache(base_model_output.cache),
        })
    }

    fn prepare_scores_for_generation(
        &self,
        scores: &mut Tensor,
        current_length: i64,
        max_length: Option<i64>,
        forced_bos_token_id: Option<i64>,
    ) {
        if current_length == 1 {
            self.force_token_id_generation(
                scores,
                &[forced_bos_token_id.unwrap_or_else(|| self.get_bos_id().unwrap())],
            );
        } else if let Some(max_length) = max_length {
            if current_length == max_length - 1 {
                self.force_token_id_generation(scores, self.get_eos_ids().as_ref().unwrap());
            }
        }
    }

    fn encode(&self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> Option<Tensor> {
        Some(self.model.encode(input_ids, attention_mask))
    }

    fn prepare_inputs_for_generation<'a>(
        &self,
        input_ids: Tensor,
        encoder_outputs: Option<&'a Tensor>,
        past: Cache,
        attention_mask: Tensor,
    ) -> PreparedInput<'a> {
        match past {
            Cache::BARTCache(past) => PreparedInput {
                prepared_input: None,
                prepared_attention_mask: Some(attention_mask),
                prepared_encoder_output: encoder_outputs,
                prepared_decoder_input: Some(input_ids.narrow(1, -1, 1)),
                prepared_position_ids: None,
                prepared_past: Cache::BARTCache(past),
            },
            Cache::None => PreparedInput {
                prepared_input: None,
                prepared_attention_mask: Some(attention_mask),
                prepared_encoder_output: encoder_outputs,
                prepared_decoder_input: Some(input_ids),
                prepared_position_ids: None,
                prepared_past: Cache::BARTCache(None),
            },
            _ => panic!("Cache type incompatible with BART"),
        }
    }

    fn encode_prompt_text<S>(
        &self,
        prompt_text: &[S],
        max_len: Option<i64>,
        pad_token_id: Option<i64>,
    ) -> Tensor
    where
        S: AsRef<str> + Sync,
    {
        let tokens = self._get_tokenizer().encode_list(
            prompt_text,
            max_len
                .map(|max_len| max_len as usize)
                .unwrap_or(usize::MAX),
            &TruncationStrategy::LongestFirst,
            0,
        );
        let token_ids = tokens
            .into_iter()
            .map(|tokenized_input| tokenized_input.token_ids)
            .collect::<Vec<Vec<i64>>>();

        let max_len = token_ids.iter().map(|input| input.len()).max().unwrap();

        let pad_token = match pad_token_id {
            Some(value) => value,
            None => self._get_tokenizer().get_unk_id(),
        };

        let token_ids = token_ids
            .into_iter()
            .map(|mut input| {
                let temp = vec![pad_token; max_len - input.len()];
                input.extend(temp);
                input
            })
            .map(|tokens| Tensor::from_slice(&tokens).to(self.get_var_store().device()))
            .collect::<Vec<Tensor>>();

        Tensor::stack(&token_ids, 0)
    }

    fn reorder_cache(
        &self,
        past: &mut Cache,
        encoder_outputs: Option<Tensor>,
        beam_indices: &Tensor,
    ) -> Option<Tensor> {
        let encoder_outputs = encoder_outputs.map(|value| value.index_select(0, beam_indices));
        match past {
            Cache::BARTCache(old_cache_option) => match old_cache_option {
                Some(old_cache) => {
                    for (self_layer_state, encoder_layer_state) in old_cache.iter_mut() {
                        if self_layer_state.is_some() {
                            self_layer_state
                                .as_mut()
                                .unwrap()
                                .reorder_cache(beam_indices)
                        };
                        if encoder_layer_state.is_some() {
                            encoder_layer_state
                                .as_mut()
                                .unwrap()
                                .reorder_cache(beam_indices)
                        };
                    }
                }
                None => {}
            },
            Cache::None => {}
            _ => {
                panic!("Invalid cache for BART model");
            }
        };
        encoder_outputs
    }
}

impl LanguageGenerator for BartGenerator {}

#[cfg(test)]
mod test {
    use tch::Device;

    use crate::{
        resources::{RemoteResource, ResourceProvider},
        Config,
    };

    use super::{BartConfig, BartConfigResources, BartModel};

    #[test]
    #[ignore] // compilation is enough, no need to run
    fn bart_model_send() {
        let config_resource = Box::new(RemoteResource::from_pretrained(BartConfigResources::BART));
        let config_path = config_resource.get_local_path().expect("");

        //    Set-up masked LM model
        let device = Device::cuda_if_available();
        let vs = tch::nn::VarStore::new(device);
        let config = BartConfig::from_file(config_path);

        let _: Box<dyn Send> = Box::new(BartModel::new(vs.root(), &config));
    }
}
