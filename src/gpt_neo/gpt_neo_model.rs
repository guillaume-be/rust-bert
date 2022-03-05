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

use crate::common::dropout::Dropout;
use crate::common::embeddings::process_ids_embeddings_pair;
use crate::gpt_neo::decoder::GptNeoBlock;
use crate::gpt_neo::LayerState;
use crate::pipelines::common::{ModelType, TokenizerOption};
use crate::pipelines::generation_utils::private_generation_utils::{
    PreparedInput, PrivateLanguageGenerator,
};
use crate::pipelines::generation_utils::{
    Cache, GenerateConfig, LMHeadModel, LMModelOutput, LanguageGenerator,
};
use crate::{Activation, Config, RustBertError};
use rust_tokenizers::tokenizer::Gpt2Tokenizer;
use rust_tokenizers::vocab::Gpt2Vocab;
use serde::{Deserialize, Serialize};
use std::borrow::{Borrow, BorrowMut};
use tch::{nn, Kind, Tensor};

/// # GPT-Neo Pretrained model weight files
pub struct GptNeoModelResources;

/// # GPT-Neo Pretrained model config files
pub struct GptNeoConfigResources;

/// # GPT-Neo Pretrained model vocab files
pub struct GptNeoVocabResources;

/// # GPT-Neo Pretrained model merges files
pub struct GptNeoMergesResources;

impl GptNeoModelResources {
    /// Shared under Apache 2.0 license by the EleutherAI contributors at <https://www.eleuther.ai>. Modified with conversion to C-array format.
    pub const GPT_NEO_125M: (&'static str, &'static str) = (
        "gpt-neo-125M/model",
        "https://huggingface.co/EleutherAI/gpt-neo-125M/resolve/main/rust_model.ot",
    );
    /// Shared under Apache 2.0 license by the EleutherAI contributors at <https://www.eleuther.ai>. Modified with conversion to C-array format.
    pub const GPT_NEO_1_3B: (&'static str, &'static str) = (
        "gpt-neo-1_3B/model",
        "https://huggingface.co/EleutherAI/gpt-neo-1.3B/resolve/main/rust_model.ot",
    );
    /// Shared under Apache 2.0 license by the EleutherAI contributors at <https://www.eleuther.ai>. Modified with conversion to C-array format.
    pub const GPT_NEO_2_7B: (&'static str, &'static str) = (
        "gpt-neo-2_7B/model",
        "https://huggingface.co/EleutherAI/gpt-neo-2.7B/resolve/main/rust_model.ot",
    );
}

impl GptNeoConfigResources {
    /// Shared under Apache 2.0 license by the EleutherAI contributors at <https://www.eleuther.ai>. Modified with conversion to C-array format.
    pub const GPT_NEO_125M: (&'static str, &'static str) = (
        "gpt-neo-125M/config",
        "https://huggingface.co/EleutherAI/gpt-neo-125M/resolve/main/config.json",
    );
    /// Shared under Apache 2.0 license by the EleutherAI contributors at <https://www.eleuther.ai>. Modified with conversion to C-array format.
    pub const GPT_NEO_1_3B: (&'static str, &'static str) = (
        "gpt-neo-1_3B/config",
        "https://huggingface.co/EleutherAI/gpt-neo-1.3B/resolve/main/config.json",
    );
    /// Shared under Apache 2.0 license by the EleutherAI contributors at <https://www.eleuther.ai>. Modified with conversion to C-array format.
    pub const GPT_NEO_2_7B: (&'static str, &'static str) = (
        "gpt-neo-2_7B/config",
        "https://huggingface.co/EleutherAI/gpt-neo-2.7B/resolve/main/config.json",
    );
}

impl GptNeoVocabResources {
    /// Shared under Modified MIT license by the OpenAI team at <https://github.com/openai/gpt-2/blob/master/LICENSE>. Modified with conversion to C-array format.
    pub const GPT_NEO_125M: (&'static str, &'static str) = (
        "gpt-neo-125M/vocab",
        "https://huggingface.co/EleutherAI/gpt-neo-125M/resolve/main/vocab.json",
    );
    /// Shared under Modified MIT license by the OpenAI team at <https://github.com/openai/gpt-2/blob/master/LICENSE>. Modified with conversion to C-array format.
    pub const GPT_NEO_1_3B: (&'static str, &'static str) = (
        "gpt-neo-1_3B/vocab",
        "https://huggingface.co/EleutherAI/gpt-neo-1.3B/resolve/main/vocab.json",
    );
    /// Shared under Modified MIT license by the OpenAI team at <https://github.com/openai/gpt-2/blob/master/LICENSE>. Modified with conversion to C-array format.
    pub const GPT_NEO_2_7B: (&'static str, &'static str) = (
        "gpt-neo-2_7B/vocab",
        "https://huggingface.co/EleutherAI/gpt-neo-2.7B/resolve/main/vocab.json",
    );
}

impl GptNeoMergesResources {
    /// Shared under Apache 2.0 license by the EleutherAI contributors at <https://www.eleuther.ai>. Modified with conversion to C-array format.
    pub const GPT_NEO_125M: (&'static str, &'static str) = (
        "gpt-neo-125M/merges",
        "https://huggingface.co/EleutherAI/gpt-neo-125M/resolve/main/merges.txt",
    );
    /// Shared under Apache 2.0 license by the EleutherAI contributors at <https://www.eleuther.ai>. Modified with conversion to C-array format.
    pub const GPT_NEO_1_3B: (&'static str, &'static str) = (
        "gpt-neo-1_3B/merges",
        "https://huggingface.co/EleutherAI/gpt-neo-1.3B/resolve/main/merges.txt",
    );
    /// Shared under Apache 2.0 license by the EleutherAI contributors at <https://www.eleuther.ai>. Modified with conversion to C-array format.
    pub const GPT_NEO_2_7B: (&'static str, &'static str) = (
        "gpt-neo-2_7B/merges",
        "https://huggingface.co/EleutherAI/gpt-neo-2.7B/resolve/main/merges.txt",
    );
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq)]
#[serde(rename_all = "camelCase")]
/// #GPT-Neo attention layer type
pub enum AttentionLayerType {
    Global,
    Local,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
/// # GPT-Neo model configuration
/// Defines the GPT-Neo model architecture (e.g. number of layers, hidden layer size, vocab size...).
pub struct GptNeoConfig {
    pub activation_function: Activation,
    pub attention_dropout: f64,
    pub attention_layers: Vec<AttentionLayerType>,
    pub attention_types: Vec<(Vec<AttentionLayerType>, i64)>,
    pub intermediate_size: Option<i64>,
    pub bos_token_id: i64,
    pub eos_token_id: i64,
    pub vocab_size: i64,
    pub num_layers: i64,
    pub num_heads: i64,
    pub hidden_size: i64,
    pub window_size: i64,
    pub embed_dropout: f64,
    pub initializer_range: f64,
    pub layer_norm_epsilon: f64,
    pub max_position_embeddings: i64,
    pub output_past: Option<bool>,
    pub output_attentions: Option<bool>,
    pub output_hidden_states: Option<bool>,
    pub resid_dropout: f64,
}

impl Config for GptNeoConfig {}

/// # GPT-Neo Base model
/// Base architecture for GPT-Neo models. Task-specific models will be built from this common base model
/// It is made of the following blocks:
/// - `word_embeddings`: Word embeddings
/// - `position_embeddings`: Position embeddings
/// - `layers`: Vector of `GptNeoBlock` (transformer part of the model)
pub struct GptNeoModel {
    word_embeddings: nn::Embedding,
    position_embeddings: nn::Embedding,
    layers: Vec<GptNeoBlock>,
    dropout: Dropout,
    layer_norm: nn::LayerNorm,
    output_attentions: bool,
    output_hidden_states: bool,
}

impl GptNeoModel {
    /// Build a new `GptNeoModel`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the GPT-Neo model
    /// * `config` - `GptNeoConfig` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::gpt_neo::{GptNeoConfig, GptNeoModel};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = GptNeoConfig::from_file(config_path);
    /// let gpt_neo_model = GptNeoModel::new(&p.root(), &config).unwrap();
    /// ```
    pub fn new<'p, P>(p: P, config: &GptNeoConfig) -> Result<GptNeoModel, RustBertError>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let word_embeddings = nn::embedding(
            p / "wte",
            config.vocab_size,
            config.hidden_size,
            Default::default(),
        );

        let position_embeddings = nn::embedding(
            p / "wpe",
            config.max_position_embeddings,
            config.hidden_size,
            Default::default(),
        );

        let dropout = Dropout::new(config.embed_dropout);

        let layer_norm_config = nn::LayerNormConfig {
            eps: config.layer_norm_epsilon,
            ..Default::default()
        };

        let layer_norm = nn::layer_norm(p / "ln_f", vec![config.hidden_size], layer_norm_config);

        let mut layers: Vec<GptNeoBlock> = Vec::with_capacity(config.num_layers as usize);
        let p_layers = p / "h";
        for layer_index in 0..config.num_layers {
            layers.push(GptNeoBlock::new(
                &p_layers / layer_index,
                layer_index as usize,
                config,
            ));
        }

        let output_attentions = config.output_attentions.unwrap_or(false);
        let output_hidden_states = config.output_hidden_states.unwrap_or(false);

        Ok(GptNeoModel {
            word_embeddings,
            position_embeddings,
            layers,
            dropout,
            layer_norm,
            output_attentions,
            output_hidden_states,
        })
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). This or `input_embeds` must be provided.
    /// * `input_embeds` - Optional input tensor of shape (*batch size*, *sequence_length*, *embeddings dimension*). This or `input_ids` must be provided.
    /// * `token_type_ids` - Optional token type ids used to indicate the portion of the input the token belongs to. If not None, token type embeddings will be added to the token and position embeddings.
    /// * `position_ids` - Optional position ids of shape (*batch size*, *sequence_length*). If None, will be incremented starting from the length of the past input.
    /// * `layer_states` - Optional Vector `Option<Vec<Option<&LayerState>>>` of length *n_layer* containing tuples with the past keys and values for both the self attention of each layer.
    /// * `attention_mask` - Optional attention mask of shape (*batch size*, *sequence_length*) for the encoder positions. Positions with a mask with value 0 will be masked.
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `Result<GptNeoModelOutput, RustBertError>` containing:
    ///   - `hidden_states` - `Tensor` of shape (*batch size*, *sequence_length*, *hidden_size*) representing the activations of the last hidden state
    ///   - `next_cache` - `Option<Vec<Option<LayerState>>>` of length *n_layer* containing the past content for the the attention layers
    ///   - `all_hidden_states` - `Option<Vec<Tensor>>` of length *n_layer + 1* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `all_attentions` - `Option<Vec<Tensor>>` of length *n_layer* containing the attention weights for each layer
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use tch::{nn, Device, Tensor, no_grad, Kind};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::{Int64, Double};
    /// use rust_bert::gpt_neo::{GptNeoConfig, GptNeoModel};
    /// # let config_path = Path::new("path/to/config.json");
    /// # let vocab_path = Path::new("path/to/vocab.txt");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = GptNeoConfig::from_file(config_path);
    /// # let gpt_neo_model = GptNeoModel::new(&vs.root(), &config).unwrap();
    /// let (batch_size, sequence_length) = (64, 128);
    /// let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    /// let attention_mask = Tensor::ones(&[batch_size, sequence_length], (Int64, device));
    ///
    /// let model_output = no_grad(|| {
    ///     gpt_neo_model.forward_t(
    ///         Some(&input_tensor),
    ///         Some(&attention_mask),
    ///         None,
    ///         None,
    ///         None,
    ///         None,
    ///         false,
    ///     )
    /// });
    /// ```
    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        layer_states: Option<Vec<Option<LayerState>>>,
        attention_mask: Option<&Tensor>,
        train: bool,
    ) -> Result<GptNeoModelOutput, RustBertError> {
        let (calc_input_embeddings, input_shape, device) =
            process_ids_embeddings_pair(input_ids, input_embeds, &self.word_embeddings)?;

        let (batch_size, current_sequence_length) = (input_shape[0], input_shape[1]);

        let past_length = if let Some(past_state_value) = &layer_states {
            if let Some(first_layer_state) = &past_state_value[0] {
                let mut size_iter = first_layer_state.prev_key.size().into_iter().rev();
                size_iter.next();
                size_iter.next().unwrap()
            } else {
                0
            }
        } else {
            0
        };

        let full_sequence_length = current_sequence_length + past_length;

        let calc_position_ids = if position_ids.is_none() {
            let position_ids =
                Tensor::arange_start(past_length, full_sequence_length, (Kind::Int64, device));
            Some(
                position_ids
                    .unsqueeze(0)
                    .view([-1, current_sequence_length]),
            )
        } else {
            None
        };

        let position_ids = position_ids.unwrap_or_else(|| calc_position_ids.as_ref().unwrap());

        let input_embeds = input_embeds.unwrap_or_else(|| calc_input_embeddings.as_ref().unwrap());
        let position_embeds = position_ids.apply(&self.position_embeddings);

        let attention_mask = attention_mask.map(|attention_mask_value| {
            let attention_mask = attention_mask_value
                .view([batch_size, -1])
                .unsqueeze(1)
                .unsqueeze(1);
            let attention_mask = attention_mask.to_kind(position_embeds.kind());
            (1 - attention_mask) * -1e4
        });

        let mut hidden_state = input_embeds + position_embeds;
        if let Some(token_type_ids) = token_type_ids {
            hidden_state = hidden_state + token_type_ids.apply(&self.word_embeddings);
        };
        hidden_state = hidden_state.apply_t(&self.dropout, train);
        let mut output_shape = input_shape;
        output_shape.push(*hidden_state.size().last().unwrap());

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
        let old_cache = layer_states.unwrap_or_else(|| vec![None; self.layers.len()]);
        let mut next_cache = vec![None; self.layers.len()];

        let mut x: Option<Tensor> = None;
        let mut attention_weights: Option<Tensor>;

        for ((layer_idx, layer), layer_state) in
            self.layers.iter().enumerate().zip(old_cache.into_iter())
        {
            let temp = if let Some(x_value) = &x {
                layer.forward_t(
                    x_value,
                    layer_state.as_ref(),
                    attention_mask.as_ref(),
                    train,
                )?
            } else {
                layer.forward_t(
                    &hidden_state,
                    layer_state.as_ref(),
                    attention_mask.as_ref(),
                    train,
                )?
            };
            x = Some(temp.0);
            attention_weights = temp.1;
            next_cache[layer_idx] = temp.2;
            if let Some(attentions) = all_attentions.borrow_mut() {
                attentions.push(attention_weights.as_ref().unwrap().copy());
            };
            if let Some(hidden_states) = all_hidden_states.borrow_mut() {
                hidden_states.push(x.as_ref().unwrap().copy());
            };
        }

        let hidden_states = x
            .unwrap()
            .apply(&self.layer_norm)
            .view(output_shape.as_slice());

        Ok(GptNeoModelOutput {
            hidden_states,
            next_cache: Some(next_cache),
            all_hidden_states,
            all_attentions,
        })
    }
}

/// # GPT-Neo Model for causal language modeling
/// Gpt-Neo model with a vocabulary decoding head. The language model decoding head is tied to the word embedding matrix weights
/// It is made of the following blocks:
/// - `transformer`: `GptNeoModel` Base ProphetNet model
pub struct GptNeoForCausalLM {
    transformer: GptNeoModel,
}

impl GptNeoForCausalLM {
    /// Build a new `GptNeoForCausalLM`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the GPT-Neo model
    /// * `config` - `GptNeoConfig` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::gpt_neo::{GptNeoConfig, GptNeoForCausalLM};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = GptNeoConfig::from_file(config_path);
    /// let gpt_neo_model = GptNeoForCausalLM::new(&p.root(), &config).unwrap();
    /// ```
    pub fn new<'p, P>(p: P, config: &GptNeoConfig) -> Result<GptNeoForCausalLM, RustBertError>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let transformer = GptNeoModel::new(p / "transformer", config)?;

        Ok(GptNeoForCausalLM { transformer })
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). This or `input_embeds` must be provided.
    /// * `input_embeds` - Optional input tensor of shape (*batch size*, *sequence_length*, *embeddings dimension*). This or `input_ids` must be provided.
    /// * `token_type_ids` - Optional token type ids used to indicate the portion of the input the token belongs to. If not None, token type embeddings will be added to the token and position embeddings.
    /// * `position_ids` - Optional position ids of shape (*batch size*, *sequence_length*). If None, will be incremented starting from the length of the past input.
    /// * `layer_states` - Optional Vector `Option<Vec<Option<&LayerState>>>` of length *n_layer* containing tuples with the past keys and values for both the self attention of each layer.
    /// * `attention_mask` - Optional attention mask of shape (*batch size*, *sequence_length*) for the encoder positions. Positions with a mask with value 0 will be masked.
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `Result<GptNeoModelLMOutput, RustBertError>` containing:
    ///   - `lm_logits` - `Tensor` of shape (*batch size*, *sequence_length*, *vocab_size*) representing the logits for each vocab item and position
    ///   - `next_cache` - `Option<Vec<Option<LayerState>>>` of length *n_layer* containing the past content for the the attention layers
    ///   - `all_hidden_states` - `Option<Vec<Tensor>>` of length *n_layer + 1* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `all_attentions` - `Option<Vec<Tensor>>` of length *n_layer* containing the attention weights for each layer
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use tch::{nn, Device, Tensor, no_grad, Kind};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::{Int64, Double};
    /// use rust_bert::gpt_neo::{GptNeoConfig, GptNeoForCausalLM};
    /// # let config_path = Path::new("path/to/config.json");
    /// # let vocab_path = Path::new("path/to/vocab.txt");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = GptNeoConfig::from_file(config_path);
    /// # let gpt_neo_model = GptNeoForCausalLM::new(&vs.root(), &config).unwrap();
    /// let (batch_size, sequence_length) = (64, 128);
    /// let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    /// let attention_mask = Tensor::ones(&[batch_size, sequence_length], (Int64, device));
    ///
    /// let model_output = no_grad(|| {
    ///     gpt_neo_model.forward_t(
    ///         Some(&input_tensor),
    ///         Some(&attention_mask),
    ///         None,
    ///         None,
    ///         None,
    ///         None,
    ///         false,
    ///     )
    /// });
    /// ```
    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        layer_states: Option<Vec<Option<LayerState>>>,
        attention_mask: Option<&Tensor>,
        train: bool,
    ) -> Result<GptNeoModelLMOutput, RustBertError> {
        let base_model_output = self.transformer.forward_t(
            input_ids,
            input_embeds,
            token_type_ids,
            position_ids,
            layer_states,
            attention_mask,
            train,
        )?;

        let lm_logits = base_model_output
            .hidden_states
            .linear::<Tensor>(&self.transformer.word_embeddings.ws, None);

        Ok(GptNeoModelLMOutput {
            lm_logits,
            next_cache: base_model_output.next_cache,
            all_hidden_states: base_model_output.all_hidden_states,
            all_attentions: base_model_output.all_attentions,
        })
    }
}

impl LMHeadModel for GptNeoForCausalLM {
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
            Cache::GPTNeoCache(layer_past) => self.forward_t(
                input_ids,
                input_embeds,
                token_type_ids,
                position_ids,
                layer_past,
                attention_mask,
                train,
            ),
            Cache::None => self.forward_t(
                input_ids,
                input_embeds,
                token_type_ids,
                position_ids,
                None,
                attention_mask,
                train,
            ),
            _ => {
                return Err(RustBertError::ValueError(
                    "Cache not compatible with GPT-Neo Model".into(),
                ));
            }
        }?;

        Ok(LMModelOutput {
            lm_logits: base_model_output.lm_logits,
            cache: Cache::GPTNeoCache(base_model_output.next_cache),
        })
    }
}

/// Container for the GPT-Neo model output.
pub struct GptNeoModelOutput {
    /// Last hidden states from the model
    pub hidden_states: Tensor,
    /// Cached outputs of the model (attention layers keys and values) if the model is used for generation
    pub next_cache: Option<Vec<Option<LayerState>>>,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all intermediate layers
    pub all_attentions: Option<Vec<Tensor>>,
}

///Container holding a GPT-Neo model with LM head output
pub struct GptNeoModelLMOutput {
    /// logits
    pub lm_logits: Tensor,
    /// Cached outputs of the model (attention layers keys and values) if the model is used for generation
    pub next_cache: Option<Vec<Option<LayerState>>>,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all intermediate layers
    pub all_attentions: Option<Vec<Tensor>>,
}

/// # Language generation model based on the GPT-Neo architecture
pub struct GptNeoGenerator {
    model: GptNeoForCausalLM,
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

impl GptNeoGenerator {
    /// Build a new `GPTNeoGenerator`
    ///
    /// # Arguments
    ///
    /// * `generate_config` - `GenerateConfig` object containing the resource references (model, vocabulary, configuration), generation options and device placement (CPU/GPU)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> anyhow::Result<()> {
    /// use rust_bert::gpt_neo::GptNeoGenerator;
    /// use rust_bert::pipelines::generation_utils::GenerateConfig;
    ///
    /// let generate_config = GenerateConfig {
    ///     max_length: 30,
    ///     do_sample: true,
    ///     num_beams: 5,
    ///     temperature: 1.1,
    ///     num_return_sequences: 3,
    ///     ..Default::default()
    /// };
    /// let gpt_neo_generator = GptNeoGenerator::new(generate_config)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(generate_config: GenerateConfig) -> Result<GptNeoGenerator, RustBertError> {
        let vocab_path = generate_config.vocab_resource.get_local_path()?;
        let merges_path = generate_config.merges_resource.get_local_path()?;

        let tokenizer = TokenizerOption::from_file(
            ModelType::GPTNeo,
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
    ) -> Result<GptNeoGenerator, RustBertError> {
        let config_path = generate_config.config_resource.get_local_path()?;
        let weights_path = generate_config.model_resource.get_local_path()?;
        let device = generate_config.device;

        generate_config.validate();
        let mut var_store = nn::VarStore::new(device);
        let config = GptNeoConfig::from_file(config_path);
        let model = GptNeoForCausalLM::new(&var_store.root(), &config)?;
        var_store.load(weights_path)?;

        let bos_token_id = tokenizer.get_bos_id();
        let eos_token_ids = tokenizer.get_eos_id().map(|id| vec![id]);
        let pad_token_id = tokenizer.get_pad_id();
        let is_encoder_decoder = false;
        let vocab_size = config.vocab_size;
        let decoder_start_id = None;
        let max_position_embeddings = config.max_position_embeddings;

        Ok(GptNeoGenerator {
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

impl PrivateLanguageGenerator<GptNeoForCausalLM, Gpt2Vocab, Gpt2Tokenizer> for GptNeoGenerator {
    fn get_model(&self) -> &GptNeoForCausalLM {
        &self.model
    }
    fn _get_tokenizer(&self) -> &TokenizerOption {
        &self.tokenizer
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

    fn prepare_inputs_for_generation<'a>(
        &self,
        input_ids: Tensor,
        _encoder_outputs: Option<&'a Tensor>,
        past: Cache,
        attention_mask: Tensor,
    ) -> PreparedInput<'a> {
        let position_ids = (attention_mask.totype(Kind::Int64).cumsum(-1, Kind::Int64) - 1)
            .masked_fill(&attention_mask.eq(0), 1);

        match past {
            Cache::GPTNeoCache(past) => {
                if past.is_some() {
                    PreparedInput {
                        prepared_input: Some(input_ids.select(1, -1).unsqueeze(-1)),
                        prepared_attention_mask: Some(attention_mask),
                        prepared_encoder_output: None,
                        prepared_decoder_input: None,
                        prepared_position_ids: Some(position_ids.select(1, -1).unsqueeze(-1)),
                        prepared_past: Cache::GPTNeoCache(past),
                    }
                } else {
                    PreparedInput {
                        prepared_input: Some(input_ids),
                        prepared_attention_mask: Some(attention_mask),
                        prepared_encoder_output: None,
                        prepared_decoder_input: None,
                        prepared_position_ids: Some(position_ids),
                        prepared_past: Cache::GPTNeoCache(None),
                    }
                }
            }
            Cache::None => PreparedInput {
                prepared_input: Some(input_ids),
                prepared_attention_mask: Some(attention_mask),
                prepared_encoder_output: None,
                prepared_decoder_input: None,
                prepared_position_ids: Some(position_ids),
                prepared_past: Cache::GPTNeoCache(None),
            },
            _ => panic!("Cache type incompatible with GPT-Neo"),
        }
    }

    fn reorder_cache(
        &self,
        past: &mut Cache,
        _encoder_outputs: Option<Tensor>,
        beam_indices: &Tensor,
    ) -> Option<Tensor> {
        match past {
            Cache::GPTNeoCache(cached_decoder_state) => match cached_decoder_state {
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
                panic!("Invalid cache for GPT-Neo model");
            }
        }
    }
}

impl LanguageGenerator<GptNeoForCausalLM, Gpt2Vocab, Gpt2Tokenizer> for GptNeoGenerator {}
