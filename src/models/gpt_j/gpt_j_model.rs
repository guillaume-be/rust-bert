// Copyright 2021 The Eleuther AI and HuggingFace Inc. team. All rights reserved.
// Copyright 2022 Guillaume Becquin
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::common::activations::Activation;
use crate::common::dropout::Dropout;
use crate::common::embeddings::process_ids_embeddings_pair;
use crate::common::kind::get_min;
use crate::gpt_j::attention::LayerState;
use crate::gpt_j::transformer::GptJBlock;
use crate::pipelines::common::{ModelType, TokenizerOption};
use crate::pipelines::generation_utils::private_generation_utils::{
    PreparedInput, PrivateLanguageGenerator,
};
use crate::pipelines::generation_utils::{Cache, GenerateConfig, LMModelOutput, LanguageGenerator};
use crate::{Config, RustBertError};
use serde::{Deserialize, Serialize};
use std::borrow::{Borrow, BorrowMut};
use tch::nn::{embedding, Linear};
use tch::{nn, Device, Tensor};

/// # GPT-J Pretrained model weight files
pub struct GptJModelResources;

/// # GPT-J Pretrained model config files
pub struct GptJConfigResources;

/// # GPT-J Pretrained model vocab files
pub struct GptJVocabResources;

/// # GPT-J Pretrained model merges files
pub struct GptJMergesResources;

/// Model weights for Rust are not available out of the box for GPT-J but can be created
/// simply with the following command:
///
/// ```ignore
/// python utils/convert_model.py path/to/gpt_j/pytorch_model.bin
/// ```
///
/// Where `pytorch_model.bin` was downloaded from [EleutherAI GPT-J 6B][gpt-j-6B] or
/// [EleutherAI GPT-J 6B (float16)][gpt-j-6B-float16]. Note that to convert GPT-J 6B you
/// will need about 32 Gb of RAM, and converting GPT-J 6B float16 requires about 12 Gb
/// of RAM.
///
/// [gpt-j-6B]: https://huggingface.co/EleutherAI/gpt-j-6B/tree/main
/// [gpt-j-6B-float16]:https://huggingface.co/EleutherAI/gpt-j-6B/tree/float16
impl GptJModelResources {
    pub const GPT_J_TINY_RANDOM: (&'static str, &'static str) = (
        "gpt-j-tiny-random/model",
        "https://huggingface.co/anton-l/gpt-j-tiny-random/resolve/main/rust_model.ot",
    );
}

impl GptJConfigResources {
    /// Shared under Apache 2.0 license by the EleutherAI contributors at <https://www.eleuther.ai>. Modified with conversion to C-array format.
    pub const GPT_J_6B: (&'static str, &'static str) = (
        "gpt-j-6B/config",
        "https://huggingface.co/EleutherAI/gpt-j-6B/resolve/main/config.json",
    );
    pub const GPT_J_6B_FLOAT16: (&'static str, &'static str) = (
        "gpt-j-6B/config",
        "https://huggingface.co/EleutherAI/gpt-j-6B/resolve/float16/config.json",
    );
    pub const GPT_J_TINY_RANDOM: (&'static str, &'static str) = (
        "gpt-j-tiny-random/config",
        "https://huggingface.co/anton-l/gpt-j-tiny-random/resolve/main/config.json",
    );
}

impl GptJVocabResources {
    /// Shared under Apache 2.0 license by the EleutherAI contributors at <https://www.eleuther.ai>. Modified with conversion to C-array format.
    pub const GPT_J_6B: (&'static str, &'static str) = (
        "gpt-j-6B/vocab",
        "https://huggingface.co/EleutherAI/gpt-j-6B/resolve/main/vocab.json",
    );
    pub const GPT_J_6B_FLOAT16: (&'static str, &'static str) = (
        "gpt-j-6B/vocab",
        "https://huggingface.co/EleutherAI/gpt-j-6B/resolve/float16/vocab.json",
    );
    pub const GPT_J_TINY_RANDOM: (&'static str, &'static str) = (
        "gpt-j-tiny-random/vocab",
        "https://huggingface.co/anton-l/gpt-j-tiny-random/resolve/main/vocab.json",
    );
}

impl GptJMergesResources {
    /// Shared under Apache 2.0 license by the EleutherAI contributors at <https://www.eleuther.ai>. Modified with conversion to C-array format.
    pub const GPT_J_6B: (&'static str, &'static str) = (
        "gpt-j-6B/merges",
        "https://huggingface.co/EleutherAI/gpt-j-6B/resolve/main/merges.txt",
    );
    pub const GPT_J_6B_FLOAT16: (&'static str, &'static str) = (
        "gpt-j-6B/merges",
        "https://huggingface.co/EleutherAI/gpt-j-6B/resolve/float16/merges.txt",
    );
    pub const GPT_J_TINY_RANDOM: (&'static str, &'static str) = (
        "gpt-j-tiny-random/merges",
        "https://huggingface.co/anton-l/gpt-j-tiny-random/resolve/main/merges.txt",
    );
}

#[derive(Debug, Serialize, Deserialize, Clone)]
/// # GPT-J model configuration
/// Defines the GPT-J model architecture (e.g. number of layers, hidden layer size, vocab size...).
pub struct GptJConfig {
    pub attn_pdrop: Option<f64>,
    pub embd_pdrop: Option<f64>,
    pub hidden_dropout_prob: Option<f64>,
    pub afn: Option<Activation>,
    pub initializer_range: f64,
    pub layer_norm_epsilon: f64,
    pub n_embd: i64,
    pub n_head: i64,
    pub n_layer: i64,
    pub n_positions: i64,
    pub n_inner: Option<i64>,
    pub num_labels: Option<i64>,
    pub use_cache: Option<bool>,
    pub output_attentions: Option<bool>,
    pub output_hidden_states: Option<bool>,
    pub resid_pdrop: Option<f64>,
    pub rotary_dim: Option<i64>,
    pub vocab_size: i64,
    pub scale_attn_weights: Option<bool>,
    #[serde(default = "default_preload_on_cpu")]
    pub preload_on_cpu: bool,
    pub decoder_start_token_id: Option<i64>,
    pub forced_bos_token_id: Option<i64>,
    pub forced_eos_token_id: Option<i64>,
}

impl Config for GptJConfig {}

impl Default for GptJConfig {
    fn default() -> Self {
        GptJConfig {
            attn_pdrop: Some(0.1),
            embd_pdrop: Some(0.1),
            hidden_dropout_prob: None,
            afn: Some(Activation::gelu_new),
            initializer_range: 0.02,
            layer_norm_epsilon: 1e-5,
            n_embd: 4096,
            n_head: 16,
            n_layer: 28,
            n_positions: 2048,
            n_inner: None,
            num_labels: None,
            use_cache: None,
            output_attentions: None,
            output_hidden_states: None,
            resid_pdrop: Some(0.1),
            rotary_dim: Some(64),
            vocab_size: 50400,
            scale_attn_weights: Some(true),
            preload_on_cpu: default_preload_on_cpu(),
            decoder_start_token_id: None,
            forced_bos_token_id: None,
            forced_eos_token_id: None,
        }
    }
}

fn default_preload_on_cpu() -> bool {
    true
}

/// # GPT-J Base model
/// Base architecture for GPT-J model. Usually complemented with a task-specific head, such as a language model head.
/// It is made of the following blocks:
/// - `wte`: `token` embeddings
/// - `h`: Encoder (transformer) made of a vector of layers. Each layer is made of a multi-head attention layer, a layer-normalization layer, and a MLP made of linear layers.
/// - `output_past`: flag indicating if the model should return a past state. This can be fed back to the model to improve the quality of text generated.
/// - `output_hidden_states`: flag indicating if the model should return all hidden states (as opposed to only the last layer)
/// - `output_attentions`: flag indicating if the model should return activation weights
pub struct GptJModel {
    wte: nn::Embedding,
    drop: Dropout,
    ln_f: nn::LayerNorm,
    h: Vec<GptJBlock>,
    use_cache: bool,
    output_hidden_states: bool,
    output_attentions: bool,
}

impl GptJModel {
    /// Build a new `GptJModel`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the GPT-J model
    /// * `config` - `GptJConfig` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::gpt_j::{GptJConfig, GptJModel};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = GptJConfig::from_file(config_path);
    /// let gpt_j: GptJModel = GptJModel::new(&p.root() / "gpt_j", &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &GptJConfig) -> GptJModel
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow() / "transformer";

        let wte = embedding(
            &p / "wte",
            config.vocab_size,
            config.n_embd,
            Default::default(),
        );

        let embd_pdrop = config.embd_pdrop.unwrap_or(0.1);
        let drop = Dropout::new(embd_pdrop);

        let layer_norm_config = nn::LayerNormConfig {
            eps: config.layer_norm_epsilon,
            ..Default::default()
        };
        let ln_f = nn::layer_norm(&p / "ln_f", vec![config.n_embd], layer_norm_config);

        let mut h: Vec<GptJBlock> = vec![];
        let h_path = &p / "h";
        for layer_index in 0..config.n_layer {
            h.push(GptJBlock::new(&h_path / layer_index, config));
        }

        let use_cache = config.use_cache.unwrap_or(true);
        let output_attentions = config.output_attentions.unwrap_or(false);
        let output_hidden_states = config.output_hidden_states.unwrap_or(false);

        GptJModel {
            wte,
            drop,
            ln_f,
            h,
            use_cache,
            output_hidden_states,
            output_attentions,
        }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see `input_embeds`)
    /// * `layer_past` - Optional vector of length *n_layer* containing the past keys and values of each layer of shape (*2*, *batch size*, *number of heads*, *past_sequence_length*, *hidden size per head*). When provided, these are concatenated with the current input keys and values.
    /// * `attention_mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `input_embeds` - Optional pre-computed input embeddings of shape (*batch size*, *sequence_length*, *hidden_size*). If None, input ids must be provided (see `input_ids`)
    /// * `token_type_ids` - Optional token type ids used to indicate the portion of the input the token belongs to. If not None, token type embeddings will be added to the token and position embeddings.
    /// * `_position_ids` - Optional position ids of shape (*batch size*, *sequence_length*). If None, will be incremented starting from the length of the past input.
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `GptJModelOutput` containing:
    ///   - `output` - `Tensor` of shape (*batch size*, *sequence_length*, *vocab_size*) representing the activations of the last hidden state
    ///   - `cache` - `Option<Vec<Tensor>>` of length *n_layer* containing the past keys and values of each layer of shape (*2*, *batch size*, *number of heads*, *past_sequence_length*, *hidden size per head*)
    ///   - `all_hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `all_attentions` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use tch::{nn, Device, Tensor, no_grad};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::{Int64, Double};
    /// use rust_bert::gpt_j::{GptJConfig, GptJModel, LayerState};
    /// # let config_path = Path::new("path/to/config.json");
    /// # let vocab_path = Path::new("path/to/vocab.txt");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = GptJConfig::from_file(config_path);
    /// # let gpt_j_model: GptJModel = GptJModel::new(&vs.root(), &config);
    /// let (batch_size, sequence_length, past_sequence_length) = (64, 128, 56);
    /// let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    /// let mut past: Vec<Option<LayerState>> = Vec::with_capacity(config.n_layer as usize);
    /// for _ in 0..config.n_layer as usize {
    ///     past.push(Some(LayerState {
    ///         prev_key: Tensor::rand(
    ///             &[
    ///                 batch_size,
    ///                 config.n_head,
    ///                 past_sequence_length,
    ///                 config.n_embd / config.n_head,
    ///             ],
    ///             (Double, device),
    ///         ),
    ///         prev_value: Tensor::rand(
    ///             &[
    ///                 batch_size,
    ///                 config.n_head,
    ///                 past_sequence_length,
    ///                 config.n_embd / config.n_head,
    ///             ],
    ///             (Double, device),
    ///         ),
    ///     }))
    /// }
    /// let attention_mask = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    /// let token_type_ids = Tensor::ones(&[batch_size, sequence_length], (Int64, device));
    ///
    /// let model_output = no_grad(|| {
    ///     gpt_j_model
    ///         .forward_t(
    ///             Some(&input_tensor),
    ///             Some(past),
    ///             Some(&attention_mask),
    ///             Some(&token_type_ids),
    ///             None,
    ///             None,
    ///             false,
    ///         )
    ///         .unwrap()
    /// });
    /// ```
    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        layer_past: Option<Vec<Option<LayerState>>>,
        attention_mask: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        _position_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        train: bool,
    ) -> Result<GptJModelOutput, RustBertError> {
        let (calc_input_embeddings, _input_size, _device) =
            process_ids_embeddings_pair(input_ids, input_embeds, &self.wte)?;

        let input_embeddings =
            input_embeds.unwrap_or_else(|| calc_input_embeddings.as_ref().unwrap());

        let (layer_past, _layer_past_length) = match layer_past {
            Some(value) => {
                if value.len() != self.h.len() {
                    return Err(RustBertError::ValueError(format!(
                        "Past activations vector length ({}) must be equal to the number of layers ({})",
                        value.len(),
                        self.h.len()
                    )));
                } else {
                    let length = value.len();
                    (value, length)
                }
            }
            None => {
                let mut out = Vec::with_capacity(self.h.len());
                out.resize_with(self.h.len(), || None);
                (out, 0)
            }
        };

        let kind_min = get_min(input_embeddings.kind())?;
        let attention_mask: Option<Tensor> = attention_mask.map(|value| {
            let attention_mask = value
                .view((input_embeddings.size()[0], -1))
                .unsqueeze(1)
                .unsqueeze(2)
                .to_kind(input_embeddings.kind());

            (attention_mask.ones_like() - attention_mask.to_kind(input_embeddings.kind()))
                * kind_min
        });

        let mut hidden_state: Tensor = input_embeddings.copy();
        if let Some(token_type_ids) = token_type_ids {
            let token_type_embeds = token_type_ids.apply(&self.wte);
            hidden_state = hidden_state + token_type_embeds;
        }
        hidden_state = hidden_state.apply_t(&self.drop, train);

        let mut all_presents: Option<Vec<Option<LayerState>>> = self.use_cache.then(Vec::new);
        let mut all_hidden_states: Option<Vec<Tensor>> = self.output_hidden_states.then(Vec::new);
        let mut all_attentions: Option<Vec<Tensor>> = self.output_attentions.then(Vec::new);

        for (layer, past) in self.h.iter().zip(layer_past) {
            let temp =
                layer.forward_t(&hidden_state, past.as_ref(), attention_mask.as_ref(), train);
            hidden_state = temp.0;
            if let Some(presents) = all_presents.borrow_mut() {
                presents.push(temp.1);
            };
            if let Some(attentions) = all_attentions.borrow_mut() {
                attentions.push(std::mem::take(&mut temp.2.unwrap()));
            };
            if let Some(hidden_states) = all_hidden_states.borrow_mut() {
                hidden_states.push(std::mem::take(&mut hidden_state));
            };
        }

        let output = hidden_state.apply(&self.ln_f);

        Ok(GptJModelOutput {
            output,
            cache: all_presents,
            all_hidden_states,
            all_attentions,
        })
    }
}

/// # GPT-J Language Modeling head
/// GPT-J model with a decoding head (linear layer without bias). The weights of the linear layer are tied to the word embeddings
/// It is made of the following blocks:
/// - `transformer`: Base GptJModel
pub struct GptJLMHeadModel {
    transformer: GptJModel,
    lm_head: Linear,
}

impl GptJLMHeadModel {
    /// Build a new `GptJLMHeadModel`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the GPT-J model
    /// * `config` - `GptJConfig` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::gpt_j::{GptJConfig, GptJLMHeadModel};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = GptJConfig::from_file(config_path);
    /// let gpt_j: GptJLMHeadModel = GptJLMHeadModel::new(&p.root() / "gpt_j", &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &GptJConfig) -> GptJLMHeadModel
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let transformer = GptJModel::new(p, config);
        let lm_head = nn::linear(
            p / "lm_head",
            config.n_embd,
            config.vocab_size,
            Default::default(),
        );

        GptJLMHeadModel {
            transformer,
            lm_head,
        }
    }

    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        layer_past: Cache,
        attention_mask: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        _encoder_outputs: Option<&Tensor>,
        _decoder_input_ids: Option<&Tensor>,
        train: bool,
    ) -> Result<LMModelOutput, RustBertError> {
        let base_model_output = match layer_past {
            Cache::GPTJCache(layer_past) => self.transformer.forward_t(
                input_ids,
                layer_past,
                attention_mask,
                token_type_ids,
                position_ids,
                input_embeds,
                train,
            ),
            Cache::None => self.transformer.forward_t(
                input_ids,
                None,
                attention_mask,
                token_type_ids,
                position_ids,
                input_embeds,
                train,
            ),
            _ => {
                return Err(RustBertError::ValueError(
                    "Cache not compatible with GPT-J Model".into(),
                ));
            }
        }?;

        let lm_logits = base_model_output.output.apply(&self.lm_head);

        Ok(LMModelOutput {
            lm_logits,
            cache: Cache::GPTJCache(base_model_output.cache),
        })
    }
}

/// Container for the GPT-J model output.
pub struct GptJModelOutput {
    /// Hidden state of the last layer of the decoder, or logits for a custom head
    /// module after the decoder (e.g. vocabulary logits for language modeling tasks)
    pub output: Tensor,
    /// Cached attention layers keys and values if the model is used for generation
    pub cache: Option<Vec<Option<LayerState>>>,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all intermediate layers
    pub all_attentions: Option<Vec<Tensor>>,
}

/// # Language generation model based on the GPT-J architecture
pub struct GptJGenerator {
    model: GptJLMHeadModel,
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

impl GptJGenerator {
    /// Build a new `GptJGenerator`
    ///
    /// # Arguments
    ///
    /// * `generate_config` - `GenerateConfig` object containing the resource references (model, vocabulary, configuration), generation options and device placement (CPU/GPU)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> anyhow::Result<()> {
    /// use rust_bert::gpt_j::GptJGenerator;
    /// use rust_bert::pipelines::generation_utils::GenerateConfig;
    ///
    /// let generate_config = GenerateConfig {
    ///     max_length: Some(30),
    ///     do_sample: true,
    ///     num_beams: 5,
    ///     temperature: 1.1,
    ///     num_return_sequences: 3,
    ///     ..Default::default()
    /// };
    /// let gpt_j_generator = GptJGenerator::new(generate_config)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(generate_config: GenerateConfig) -> Result<GptJGenerator, RustBertError> {
        let vocab_path = generate_config.vocab_resource.get_local_path()?;
        let merges_path = generate_config
            .merges_resource
            .as_ref()
            .ok_or_else(|| {
                RustBertError::InvalidConfigurationError(
                    "GPT-J expects a merges resources to be provided".to_string(),
                )
            })?
            .get_local_path()?;

        let tokenizer = TokenizerOption::from_file(
            ModelType::GPTJ,
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
    ) -> Result<GptJGenerator, RustBertError> {
        let config_path = generate_config.config_resource.get_local_path()?;
        let device = generate_config.device;

        generate_config.validate();
        let mut var_store = nn::VarStore::new(device);

        let config = GptJConfig::from_file(config_path);
        let model = GptJLMHeadModel::new(var_store.root(), &config);
        if config.preload_on_cpu && device != Device::Cpu {
            var_store.set_device(Device::Cpu);
        }
        crate::resources::load_weights(
            &generate_config.model_resource,
            &mut var_store,
            generate_config.kind,
            device,
        )?;
        if device != Device::Cpu {
            var_store.set_device(device);
        }

        let bos_token_id = tokenizer.get_bos_id();
        let eos_token_ids = tokenizer.get_eos_id().map(|id| vec![id]);
        let pad_token_id = tokenizer.get_pad_id();
        let max_position_embeddings = config.n_positions;
        let is_encoder_decoder = false;
        let vocab_size = config.vocab_size;
        let decoder_start_id = config.decoder_start_token_id;

        Ok(GptJGenerator {
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
}

impl PrivateLanguageGenerator for GptJGenerator {
    fn _get_tokenizer(&self) -> &TokenizerOption {
        &self.tokenizer
    }
    fn _get_tokenizer_mut(&mut self) -> &mut TokenizerOption {
        &mut self.tokenizer
    }
    fn get_device(&self) -> Device {
        self.var_store.device()
    }
    fn get_var_store_mut(&mut self) -> Result<&mut nn::VarStore, RustBertError> {
        Ok(&mut self.var_store)
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
    fn get_max_positions_embeddings(&self) -> Option<i64> {
        Some(self.max_position_embeddings)
    }

    fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        layer_past: Cache,
        attention_mask: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        _encoder_outputs: Option<&Tensor>,
        _decoder_input_ids: Option<&Tensor>,
        train: bool,
    ) -> Result<LMModelOutput, RustBertError> {
        let base_model_output = match layer_past {
            Cache::GPTJCache(layer_past) => self.model.transformer.forward_t(
                input_ids,
                layer_past,
                attention_mask,
                token_type_ids,
                position_ids,
                input_embeds,
                train,
            ),
            Cache::None => self.model.transformer.forward_t(
                input_ids,
                None,
                attention_mask,
                token_type_ids,
                position_ids,
                input_embeds,
                train,
            ),
            _ => {
                return Err(RustBertError::ValueError(
                    "Cache not compatible with GPT-J Model".into(),
                ));
            }
        }?;

        let lm_logits = base_model_output.output.apply(&self.model.lm_head);

        Ok(LMModelOutput {
            lm_logits,
            cache: Cache::GPTJCache(base_model_output.cache),
        })
    }

    fn prepare_inputs_for_generation<'a>(
        &self,
        input_ids: Tensor,
        _encoder_outputs: Option<&'a Tensor>,
        past: Cache,
        attention_mask: Tensor,
    ) -> PreparedInput<'a> {
        match past {
            Cache::GPTJCache(past) => {
                if past.is_some() {
                    PreparedInput {
                        prepared_input: Some(input_ids.select(1, -1).unsqueeze(-1)),
                        prepared_attention_mask: Some(attention_mask),
                        prepared_encoder_output: None,
                        prepared_decoder_input: None,
                        prepared_position_ids: None,
                        prepared_past: Cache::GPTJCache(past),
                    }
                } else {
                    PreparedInput {
                        prepared_input: Some(input_ids),
                        prepared_attention_mask: Some(attention_mask),
                        prepared_encoder_output: None,
                        prepared_decoder_input: None,
                        prepared_position_ids: None,
                        prepared_past: Cache::GPTJCache(None),
                    }
                }
            }
            Cache::None => PreparedInput {
                prepared_input: Some(input_ids),
                prepared_attention_mask: Some(attention_mask),
                prepared_encoder_output: None,
                prepared_decoder_input: None,
                prepared_position_ids: None,
                prepared_past: Cache::GPTJCache(None),
            },
            _ => panic!("Cache type incompatible with GPT-J"),
        }
    }

    fn reorder_cache(
        &self,
        past: &mut Cache,
        _encoder_outputs: Option<Tensor>,
        beam_indices: &Tensor,
    ) -> Option<Tensor> {
        match past {
            Cache::GPTJCache(cached_decoder_state) => match cached_decoder_state {
                Some(old_cache) => {
                    for layer_state in old_cache.iter_mut() {
                        if layer_state.is_some() {
                            layer_state.as_mut().unwrap().reorder_cache(beam_indices)
                        };
                    }
                    None
                }
                None => None,
            },
            Cache::None => None,
            _ => {
                panic!("Invalid cache for GPT-J model");
            }
        }
    }
}

impl LanguageGenerator for GptJGenerator {}
