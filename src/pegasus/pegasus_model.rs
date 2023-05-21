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

use crate::bart::BartModelOutput;
use crate::common::kind::get_negative_infinity;
use crate::mbart::MBartConfig;
use crate::pegasus::decoder::PegasusDecoder;
use crate::pegasus::encoder::PegasusEncoder;
use crate::pegasus::LayerState;
use crate::pipelines::common::{ModelType, TokenizerOption};
use crate::pipelines::generation_utils::private_generation_utils::{
    PreparedInput, PrivateLanguageGenerator,
};
use crate::pipelines::generation_utils::{Cache, GenerateConfig, LMModelOutput, LanguageGenerator};
use crate::{Config, RustBertError};
use rust_tokenizers::tokenizer::TruncationStrategy;
use std::borrow::Borrow;
use tch::nn::{embedding, EmbeddingConfig, Init};
use tch::{nn, Tensor};

/// # Pegasus Pretrained model weight files
pub struct PegasusModelResources;

/// # Pegasus Pretrained model config files
pub struct PegasusConfigResources;

/// # Pegasus Pretrained model vocab files
pub struct PegasusVocabResources;

impl PegasusModelResources {
    /// Shared under Apache 2.0 license by the Pegasus team at <https://huggingface.co/google/pegasus-cnn_dailymail>. Modified with conversion to C-array format.
    pub const CNN_DAILYMAIL: (&'static str, &'static str) = (
        "pegasus-cnn_dailymail/model",
        "https://huggingface.co/google/pegasus-cnn_dailymail/resolve/main/rust_model.ot",
    );
}

impl PegasusConfigResources {
    /// Shared under Apache 2.0 license by the Pegasus team at <https://huggingface.co/google/pegasus-cnn_dailymail>.
    pub const CNN_DAILYMAIL: (&'static str, &'static str) = (
        "pegasus-cnn_dailymail/config",
        "https://huggingface.co/google/pegasus-cnn_dailymail/resolve/main/config.json",
    );
}

impl PegasusVocabResources {
    /// Shared under Apache 2.0 license by the Pegasus team at <https://huggingface.co/google/pegasus-cnn_dailymail>.
    pub const CNN_DAILYMAIL: (&'static str, &'static str) = (
        "pegasus-cnn_dailymail/spiece",
        "https://huggingface.co/google/pegasus-cnn_dailymail/resolve/main/spiece.model",
    );
}

/// # Pegasus model configuration
/// Defines the Pegasus model architecture (e.g. number of layers, hidden layer size, label mapping...)
pub type PegasusConfig = MBartConfig;

fn _shift_tokens_right(
    input_ids: &Tensor,
    pad_token_id: i64,
    decoder_start_token_id: i64,
) -> Tensor {
    let input_ids_length = input_ids.size()[1];
    let mut shifted_input_ids = Tensor::zeros(
        input_ids.size().as_slice(),
        (input_ids.kind(), input_ids.device()),
    );
    shifted_input_ids
        .slice(1, 1, input_ids_length, 1)
        .copy_(&input_ids.slice(1, 0, input_ids_length - 1, 1));

    let _ = shifted_input_ids.select(1, 0).fill_(decoder_start_token_id);
    let _ = shifted_input_ids.masked_fill_(&shifted_input_ids.eq(-100), pad_token_id);

    shifted_input_ids
}

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
    /// * `decoder_input_ids` - Optional input tensor of shape (*batch size*, *target_sequence_length*). Must be provided when running in generation mode (e.g. initialized with a BOS token)
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
            decoder_input_ids,
            encoder_output,
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

/// # Pegasus Model for conditional generation
/// Pegasus model with a vocabulary decoding head
/// It is made of the following blocks:
/// - `base_model`: `PegasusModel` Base Pegasus model
pub struct PegasusForConditionalGeneration {
    base_model: PegasusModel,
    final_logits_bias: Tensor,
    pad_token_id: i64,
    decoder_start_token_id: i64,
}

impl PegasusForConditionalGeneration {
    /// Build a new `PegasusForConditionalGeneration`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the BART model
    /// * `config` - `PegasusConfig` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::pegasus::{PegasusConfig, PegasusForConditionalGeneration};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = PegasusConfig::from_file(config_path);
    /// let pegasus: PegasusForConditionalGeneration =
    ///     PegasusForConditionalGeneration::new(&p.root(), &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &PegasusConfig) -> PegasusForConditionalGeneration
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let base_model = PegasusModel::new(p / "model", config);

        let final_logits_bias = p.var(
            "final_logits_bias",
            &[1, config.vocab_size],
            Init::Const(0.0),
        );

        let pad_token_id = config.pad_token_id.unwrap_or(0);
        let decoder_start_token_id = config.decoder_start_token_id.unwrap_or(0);

        PegasusForConditionalGeneration {
            base_model,
            final_logits_bias,
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
    /// * `encoder_outputs` - Optional tuple made of a tensor of shape (*batch size*, *source_sequence_length*, *encoder_hidden_dim*) and optional vectors of tensors of length *num_encoder_layers* with shape (*batch size*, *source_sequence_length*, *hidden_size*).
    /// These correspond to the encoder last hidden state and optional hidden states/attention weights for encoder layers. When provided, the encoder hidden state will not be recalculated. Useful for generation tasks.
    /// * `decoder_input_ids` - Optional input tensor of shape (*batch size*, *target_sequence_length*). Must be provided when running in generation mode (e.g. initialized with a BOS token)
    /// * `decoder_attention_mask` - Optional attention mask of shape (*batch size*, *target_sequence_length*) for the decoder positions. Positions with a mask with value 0 will be masked.
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `PegasusModelOutput` containing:
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
    /// use rust_bert::pegasus::{PegasusConfig, PegasusForConditionalGeneration};
    /// # let config_path = Path::new("path/to/config.json");
    /// # let vocab_path = Path::new("path/to/vocab.txt");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = PegasusConfig::from_file(config_path);
    /// # let pegasus_model: PegasusForConditionalGeneration = PegasusForConditionalGeneration::new(&vs.root(), &config);
    ///  let (batch_size, source_sequence_length, target_sequence_length) = (64, 128, 56);
    ///  let input_tensor = Tensor::rand(&[batch_size, source_sequence_length], (Int64, device));
    ///  let decoder_input_ids = Tensor::rand(&[batch_size, target_sequence_length], (Int64, device));
    ///  let encoder_attention_mask = Tensor::ones(&[batch_size, source_sequence_length], (Int64, device));
    ///  let decoder_attention_mask = Tensor::ones(&[batch_size, source_sequence_length], (Int64, device));
    ///
    ///  let model_output = no_grad(|| {
    ///    pegasus_model
    ///         .forward_t(Some(&input_tensor),
    ///                    Some(&encoder_attention_mask),
    ///                    None,
    ///                    Some(&decoder_input_ids),
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
    ) -> PegasusModelOutput {
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
            .linear::<Tensor>(&self.base_model.embeddings.ws, None)
            + &self.final_logits_bias;
        PegasusModelOutput {
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

/// # Language generation model based on the Pegasus architecture
pub struct PegasusConditionalGenerator {
    model: PegasusForConditionalGeneration,
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

impl PegasusConditionalGenerator {
    /// Build a new `PegasusGenerator`
    ///
    /// # Arguments
    ///
    /// * `vocab_path` - Path to the model vocabulary, expected to have a structure following the [Transformers library](https://github.com/huggingface/transformers) convention
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
    /// use rust_bert::pegasus::PegasusConditionalGenerator;
    /// use rust_bert::pipelines::generation_utils::GenerateConfig;
    /// # let mut home: PathBuf = dirs::home_dir().unwrap();
    /// # home.push("rustbert");
    /// # home.push("pegasus-cnn_dailymail");
    /// # let config_path = &home.as_path().join("config.json");
    /// # let vocab_path = &home.as_path().join("spiece.model");
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
    /// let pegasus_generator = PegasusConditionalGenerator::new(generate_config)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(
        generate_config: GenerateConfig,
    ) -> Result<PegasusConditionalGenerator, RustBertError> {
        let vocab_path = generate_config.vocab_resource.get_local_path()?;

        let tokenizer = TokenizerOption::from_file(
            ModelType::Pegasus,
            vocab_path.to_str().unwrap(),
            None,
            false,
            None,
            None,
        )?;

        Self::new_with_tokenizer(generate_config, tokenizer)
    }

    pub fn new_with_tokenizer(
        generate_config: GenerateConfig,
        tokenizer: TokenizerOption,
    ) -> Result<PegasusConditionalGenerator, RustBertError> {
        let config_path = generate_config.config_resource.get_local_path()?;
        let weights_path = generate_config.model_resource.get_local_path()?;
        let device = generate_config.device;

        generate_config.validate();
        let mut var_store = nn::VarStore::new(device);
        let config = PegasusConfig::from_file(config_path);
        let model = PegasusForConditionalGeneration::new(var_store.root(), &config);
        var_store.load(weights_path)?;

        let bos_token_id = Some(config.bos_token_id.unwrap_or(0));
        let eos_token_ids = Some(match config.eos_token_id {
            Some(value) => vec![value],
            None => vec![1],
        });
        let pad_token_id = Some(config.pad_token_id.unwrap_or(0));
        let vocab_size = config.vocab_size;
        let is_encoder_decoder = true;
        let decoder_start_id = Some(0);
        let max_position_embeddings = config.max_position_embeddings;

        Ok(PegasusConditionalGenerator {
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
        let _ = scores.index_fill_(
            1,
            &impossible_tokens,
            get_negative_infinity(scores.kind()).unwrap(),
        );
    }
}

impl PrivateLanguageGenerator for PegasusConditionalGenerator {
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
                    "Cache not compatible with Pegasus Model".into(),
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
        _forced_bos_token_id: Option<i64>,
    ) {
        if let Some(max_length) = max_length {
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
            _ => panic!("Cache type incompatible with Pegasus"),
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
            None => self
                ._get_tokenizer()
                .get_pad_id()
                .expect("A padding token must be provided to encode prompt texts."),
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
                panic!("Invalid cache for Pegasus model");
            }
        };
        encoder_outputs
    }
}

impl LanguageGenerator for PegasusConditionalGenerator {}

/// Container holding a Pegasus model output. The decoder output may hold the hidden state of
/// the last layer of the decoder, or may hold logits for a custom head module after the
/// decoder (e.g. for classification or language modeling tasks)
pub type PegasusModelOutput = BartModelOutput;
