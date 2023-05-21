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
use crate::pipelines::common::{ModelType, TokenizerOption};
use crate::pipelines::generation_utils::private_generation_utils::{
    PreparedInput, PrivateLanguageGenerator,
};
use crate::pipelines::generation_utils::{Cache, GenerateConfig, LMModelOutput, LanguageGenerator};
use crate::pipelines::translation::Language;
use crate::{Config, RustBertError};
use rust_tokenizers::tokenizer::TruncationStrategy;
use std::borrow::Borrow;
use tch::nn::{embedding, EmbeddingConfig};
use tch::{nn, Kind, Tensor};

/// # M2M100 Pretrained model weight files
pub struct M2M100ModelResources;

/// # M2M100 Pretrained model config files
pub struct M2M100ConfigResources;

/// # M2M100 Pretrained model vocab files
pub struct M2M100VocabResources;

/// # M2M100 Pretrained model merges files
pub struct M2M100MergesResources;

/// # M2M100 source languages pre-sets
pub struct M2M100SourceLanguages;

/// # M2M100 target languages pre-sets
pub type M2M100TargetLanguages = M2M100SourceLanguages;

impl M2M100ModelResources {
    /// Shared under MIT license by the Facebook AI Research Fairseq team at <https://github.com/pytorch/fairseq>. Modified with conversion to C-array format.
    pub const M2M100_418M: (&'static str, &'static str) = (
        "m2m100-418m/model",
        "https://huggingface.co/facebook/m2m100_418M/resolve/main/rust_model.ot",
    );
    /// Shared under MIT license by the Facebook AI Research Fairseq team at <https://github.com/pytorch/fairseq>. Modified with conversion to C-array format.
    pub const M2M100_1_2B: (&'static str, &'static str) = (
        "m2m100-1_2b/model",
        "https://huggingface.co/facebook/m2m100_1.2B/resolve/main/rust_model.ot",
    );
}

impl M2M100ConfigResources {
    /// Shared under MIT license by the Facebook AI Research Fairseq team at <https://github.com/pytorch/fairseq>. Modified with conversion to C-array format.
    pub const M2M100_418M: (&'static str, &'static str) = (
        "m2m100-418m/config",
        "https://huggingface.co/facebook/m2m100_418M/resolve/main/config.json",
    );
    /// Shared under MIT license by the Facebook AI Research Fairseq team at <https://github.com/pytorch/fairseq>. Modified with conversion to C-array format.
    pub const M2M100_1_2B: (&'static str, &'static str) = (
        "m2m100-1_2b/config",
        "https://huggingface.co/facebook/m2m100_1.2B/resolve/main/config.json",
    );
}

impl M2M100VocabResources {
    /// Shared under MIT license by the Facebook AI Research Fairseq team at <https://github.com/pytorch/fairseq>. Modified with conversion to C-array format.
    pub const M2M100_418M: (&'static str, &'static str) = (
        "m2m100-418m/vocab",
        "https://huggingface.co/facebook/m2m100_418M/resolve/main/vocab.json",
    );
    /// Shared under MIT license by the Facebook AI Research Fairseq team at <https://github.com/pytorch/fairseq>. Modified with conversion to C-array format.
    pub const M2M100_1_2B: (&'static str, &'static str) = (
        "m2m100-1_2b/vocab",
        "https://huggingface.co/facebook/m2m100_1.2B/resolve/main/vocab.json",
    );
}

impl M2M100MergesResources {
    /// Shared under MIT license by the Facebook AI Research Fairseq team at <https://github.com/pytorch/fairseq>. Modified with conversion to C-array format.
    pub const M2M100_418M: (&'static str, &'static str) = (
        "m2m100-418m/merges",
        "https://huggingface.co/facebook/m2m100_418M/resolve/main/sentencepiece.bpe.model",
    );
    /// Shared under MIT license by the Facebook AI Research Fairseq team at <https://github.com/pytorch/fairseq>. Modified with conversion to C-array format.
    pub const M2M100_1_2B: (&'static str, &'static str) = (
        "m2m100-1_2b/merges",
        "https://huggingface.co/facebook/m2m100_1.2B/resolve/main/sentencepiece.bpe.model",
    );
}

#[rustfmt::skip]
impl M2M100SourceLanguages {
    pub const M2M100_418M: [Language; 100] = [Language::Afrikaans, Language::Danish, Language::Dutch, Language::German, Language::English, Language::Icelandic, Language::Luxembourgish, Language::Norwegian, Language::Swedish, Language::WesternFrisian, Language::Yiddish, Language::Asturian, Language::Catalan, Language::French, Language::Galician, Language::Italian, Language::Occitan, Language::Portuguese, Language::Romanian, Language::Spanish, Language::Belarusian, Language::Bosnian, Language::Bulgarian, Language::Croatian, Language::Czech, Language::Macedonian, Language::Polish, Language::Russian, Language::Serbian, Language::Slovak, Language::Slovenian, Language::Ukrainian, Language::Estonian, Language::Finnish, Language::Hungarian, Language::Latvian, Language::Lithuanian, Language::Albanian, Language::Armenian, Language::Georgian, Language::Greek, Language::Breton, Language::Irish, Language::ScottishGaelic, Language::Welsh, Language::Azerbaijani, Language::Bashkir, Language::Kazakh, Language::Turkish, Language::Uzbek, Language::Japanese, Language::Korean, Language::Vietnamese, Language::ChineseMandarin, Language::Bengali, Language::Gujarati, Language::Hindi, Language::Kannada, Language::Marathi, Language::Nepali, Language::Oriya, Language::Panjabi, Language::Sindhi, Language::Sinhala, Language::Urdu, Language::Tamil, Language::Cebuano, Language::Iloko, Language::Indonesian, Language::Javanese, Language::Malagasy, Language::Malay, Language::Malayalam, Language::Sundanese, Language::Tagalog, Language::Burmese, Language::CentralKhmer, Language::Lao, Language::Thai, Language::Mongolian, Language::Arabic, Language::Hebrew, Language::Pashto, Language::Farsi, Language::Amharic, Language::Fulah, Language::Hausa, Language::Igbo, Language::Lingala, Language::Luganda, Language::NorthernSotho, Language::Somali, Language::Swahili, Language::Swati, Language::Tswana, Language::Wolof, Language::Xhosa, Language::Yoruba, Language::Zulu, Language::HaitianCreole];
    pub const M2M100_1_2B: [Language; 100] = M2M100SourceLanguages::M2M100_418M;
}

/// # M2M100 model configuration
/// Defines the M2M100 model architecture (e.g. number of layers, hidden layer size, label mapping...)
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
    shifted_input_ids
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
    /// * `p` - Variable store path for the root of the M2M100 model
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
    /// let m2m100: M2M100Model = M2M100Model::new(&p.root() / "m2m100", &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &M2M100Config) -> M2M100Model
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
    /// * `decoder_input_ids` - Optional input tensor of shape (*batch size*, *target_sequence_length*). Must be provided when running in generation mode (e.g. initialized with a BOS token)
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
    /// # let m2m100_model: M2M100Model = M2M100Model::new(&vs.root(), &config);
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
            decoder_input_ids,
            encoder_output,
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

/// # M2M100 Model for conditional generation
/// M2M100 model with a vocabulary decoding head
/// It is made of the following blocks:
/// - `base_model`: `M2M100Model` Base M2M100 model
/// - `linear`: Linear layer without bias tied to the weights of the token id embeddings
pub struct M2M100ForConditionalGeneration {
    base_model: M2M100Model,
}

impl M2M100ForConditionalGeneration {
    /// Build a new `M2M100ForConditionalGeneration`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the M2M100 model
    /// * `config` - `M2M100Config` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::m2m_100::{M2M100Config, M2M100ForConditionalGeneration};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = M2M100Config::from_file(config_path);
    /// let m2m100: M2M100ForConditionalGeneration =
    ///     M2M100ForConditionalGeneration::new(&p.root(), &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &M2M100Config) -> M2M100ForConditionalGeneration
    where
        P: Borrow<nn::Path<'p>>,
    {
        let base_model = M2M100Model::new(p.borrow() / "model", config);
        M2M100ForConditionalGeneration { base_model }
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
    /// * `M2M100ModelOutput` containing:
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
    /// # use rust_bert::m2m_100::{M2M100Config, M2M100ForConditionalGeneration};
    /// # let config_path = Path::new("path/to/config.json");
    /// # let vocab_path = Path::new("path/to/vocab.txt");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = M2M100Config::from_file(config_path);
    /// # let m2m100_model: M2M100ForConditionalGeneration = M2M100ForConditionalGeneration::new(&vs.root(), &config);
    ///  let (batch_size, source_sequence_length, target_sequence_length) = (64, 128, 56);
    ///  let input_tensor = Tensor::rand(&[batch_size, source_sequence_length], (Int64, device));
    ///  let target_tensor = Tensor::rand(&[batch_size, target_sequence_length], (Int64, device));
    ///  let encoder_attention_mask = Tensor::ones(&[batch_size, source_sequence_length], (Int64, device));
    ///  let decoder_attention_mask = Tensor::ones(&[batch_size, source_sequence_length], (Int64, device));
    ///
    ///  let model_output = no_grad(|| {
    ///    m2m100_model
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
    ) -> M2M100ModelOutput {
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
        M2M100ModelOutput {
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

/// # Language generation model based on the M2M100 architecture
pub struct M2M100Generator {
    model: M2M100ForConditionalGeneration,
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

impl M2M100Generator {
    /// Build a new `M2M100Generator`
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
    /// use rust_bert::m2m_100::M2M100Generator;
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
    /// let m2m100_generator = M2M100Generator::new(generate_config)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(generate_config: GenerateConfig) -> Result<M2M100Generator, RustBertError> {
        let vocab_path = generate_config.vocab_resource.get_local_path()?;
        let merges_path = generate_config
            .merges_resource
            .as_ref()
            .ok_or_else(|| {
                RustBertError::InvalidConfigurationError(
                    "M2M100 expects a merges resources to be provided".to_string(),
                )
            })?
            .get_local_path()?;

        let tokenizer = TokenizerOption::from_file(
            ModelType::M2M100,
            vocab_path.to_str().unwrap(),
            Some(merges_path.to_str().unwrap()),
            false,
            None,
            None,
        )?;

        Self::new_with_tokenizer(generate_config, tokenizer)
    }

    pub fn new_with_tokenizer(
        generate_config: GenerateConfig,
        tokenizer: TokenizerOption,
    ) -> Result<M2M100Generator, RustBertError> {
        let config_path = generate_config.config_resource.get_local_path()?;
        let weights_path = generate_config.model_resource.get_local_path()?;
        let device = generate_config.device;

        generate_config.validate();
        let mut var_store = nn::VarStore::new(device);

        let config = M2M100Config::from_file(config_path);
        let model = M2M100ForConditionalGeneration::new(var_store.root(), &config);
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

        Ok(M2M100Generator {
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

impl PrivateLanguageGenerator for M2M100Generator {
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
                    "Cache not compatible with M2M100 Model".into(),
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
            self.force_token_id_generation(scores, &[forced_bos_token_id.unwrap_or(250004)]);
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
            _ => panic!("Cache type incompatible with M2M100"),
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
                panic!("Invalid cache for M2M100 model");
            }
        };
        encoder_outputs
    }
}

impl LanguageGenerator for M2M100Generator {}

#[cfg(test)]
mod test {
    use tch::Device;

    use crate::{
        resources::{RemoteResource, ResourceProvider},
        Config,
    };

    use super::*;

    #[test]
    #[ignore] // compilation is enough, no need to run
    fn mbart_model_send() {
        let config_resource = Box::new(RemoteResource::from_pretrained(
            M2M100ConfigResources::M2M100_418M,
        ));
        let config_path = config_resource.get_local_path().expect("");

        //    Set-up masked LM model
        let device = Device::cuda_if_available();
        let vs = nn::VarStore::new(device);
        let config = M2M100Config::from_file(config_path);

        let _: Box<dyn Send> = Box::new(M2M100Model::new(vs.root(), &config));
    }
}
