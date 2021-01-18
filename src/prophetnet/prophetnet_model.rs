// Copyright 2020 The Microsoft Authors and The HuggingFace Inc. team.
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
use crate::pipelines::generation_utils::{Cache, LMHeadModel, LMModelOutput};
use crate::prophetnet::attention::LayerState;
use crate::prophetnet::decoder::ProphetNetDecoder;
use crate::prophetnet::encoder::ProphetNetEncoder;
use crate::{Activation, Config, RustBertError};
use serde::{Deserialize, Serialize};
use std::borrow::Borrow;
use std::collections::HashMap;
use tch::{nn, Kind, Tensor};

/// # ProphetNet Pretrained model weight files
pub struct ProphetNetModelResources;

/// # ProphetNet Pretrained model config files
pub struct ProphetNetConfigResources;

/// # ProphetNet Pretrained model vocab files
pub struct ProphetNetVocabResources;

impl ProphetNetModelResources {
    /// Shared under MIT license by the Microsoft team at https://github.com/microsoft/ProphetNet. Modified with conversion to C-array format.
    pub const PROPHETNET_LARGE_UNCASED: (&'static str, &'static str) = (
        "prophetnet-large-uncased/model",
        "https://huggingface.co/microsoft/prophetnet-large-uncased/resolve/main/rust_model.ot",
    );
    /// Shared under MIT license by the Microsoft team at https://github.com/microsoft/ProphetNet. Modified with conversion to C-array format.
    pub const PROPHETNET_LARGE_CNN_DM: (&'static str, &'static str) = (
        "prophetnet-large-uncased-cnndm/model",
        "https://huggingface.co/microsoft/prophetnet-large-uncased-cnndm/resolve/main/rust_model.ot",
    );
}

impl ProphetNetConfigResources {
    /// Shared under MIT license by the Microsoft team at https://github.com/microsoft/ProphetNet. Modified with conversion to C-array format.
    pub const PROPHETNET_LARGE_UNCASED: (&'static str, &'static str) = (
        "prophetnet-large-uncased/config",
        "https://huggingface.co/microsoft/prophetnet-large-uncased/resolve/main/config.json",
    );
    /// Shared under MIT license by the Microsoft team at https://github.com/microsoft/ProphetNet. Modified with conversion to C-array format.
    pub const PROPHETNET_LARGE_CNN_DM: (&'static str, &'static str) = (
        "prophetnet-large-uncased-cnndm/config",
        "https://huggingface.co/microsoft/prophetnet-large-uncased-cnndm/resolve/main/config.json",
    );
}

impl ProphetNetVocabResources {
    /// Shared under MIT license by the Microsoft team at https://github.com/microsoft/ProphetNet. Modified with conversion to C-array format.
    pub const PROPHETNET_LARGE_UNCASED: (&'static str, &'static str) = (
        "prophetnet-large-uncased/vocab",
        "https://huggingface.co/microsoft/prophetnet-large-uncased/resolve/main/prophetnet.tokenizer",
    );
    /// Shared under MIT license by the Microsoft team at https://github.com/microsoft/ProphetNet. Modified with conversion to C-array format.
    pub const PROPHETNET_LARGE_CNN_DM: (&'static str, &'static str) = (
        "prophetnet-large-uncased-cnndm/vocab",
        "https://huggingface.co/microsoft/prophetnet-large-uncased-cnndm/resolve/main/prophetnet.tokenizer",
    );
}

#[derive(Debug, Serialize, Deserialize)]
/// # ProphetNet model configuration
/// Defines the ProphetNet model architecture (e.g. number of layers, hidden layer size, label mapping...)
pub struct ProphetNetConfig {
    pub activation_function: Activation,
    pub activation_dropout: f64,
    pub attention_dropout: f64,
    pub decoder_ffn_dim: i64,
    pub decoder_layerdrop: f64,
    pub decoder_max_position_embeddings: i64,
    pub decoder_start_token_id: i64,
    pub disable_ngram_loss: bool,
    pub dropout: f64,
    pub encoder_ffn_dim: i64,
    pub encoder_layerdrop: f64,
    pub encoder_max_position_embeddings: i64,
    pub eps: f64,
    pub hidden_size: i64,
    pub init_std: f64,
    pub is_encoder_decoder: bool,
    pub max_position_embeddings: i64,
    pub bos_token_id: i64,
    pub eos_token_id: i64,
    pub ngram: i64,
    pub id2label: Option<HashMap<i64, String>>,
    pub label2id: Option<HashMap<String, i64>>,
    pub num_buckets: i64,
    pub num_decoder_attention_heads: i64,
    pub num_decoder_layers: i64,
    pub num_encoder_attention_heads: i64,
    pub num_encoder_layers: i64,
    pub output_past: Option<bool>,
    pub pad_token_id: i64,
    pub relative_max_distance: i64,
    pub vocab_size: i64,
    pub output_attentions: Option<bool>,
    pub output_hidden_states: Option<bool>,
    pub add_cross_attention: Option<bool>,
}

impl Config<ProphetNetConfig> for ProphetNetConfig {}

pub struct ProphetNetModel {
    pub(crate) word_embeddings: nn::Embedding,
    pub(crate) encoder: ProphetNetEncoder,
    decoder: ProphetNetDecoder,
}

impl ProphetNetModel {
    pub fn new<'p, P>(p: P, config: &ProphetNetConfig) -> Result<ProphetNetModel, RustBertError>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let word_embeddings_config = nn::EmbeddingConfig {
            padding_idx: config.pad_token_id,
            ..Default::default()
        };
        let word_embeddings = nn::embedding(
            p / "word_embeddings",
            config.vocab_size,
            config.hidden_size,
            word_embeddings_config,
        );

        let encoder = ProphetNetEncoder::new(p / "encoder", config)?;
        let decoder = ProphetNetDecoder::new(p / "decoder", config)?;

        Ok(ProphetNetModel {
            word_embeddings,
            encoder,
            decoder,
        })
    }

    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        decoder_input_ids: Option<&Tensor>,
        decoder_attention_mask: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
        old_layer_states: Option<Vec<(Option<LayerState>, Option<LayerState>)>>,
        decoder_input_embeds: Option<&Tensor>,
        train: bool,
    ) -> Result<ProphetNetOutput, RustBertError> {
        let calc_encoder_hidden_states = if encoder_hidden_states.is_none() {
            Some(
                self.encoder
                    .forward_t(
                        input_ids,
                        attention_mask,
                        input_embeds,
                        Some(&self.word_embeddings),
                        train,
                    )?
                    .hidden_states,
            )
        } else {
            None
        };
        let encoder_hidden_states =
            encoder_hidden_states.unwrap_or_else(|| calc_encoder_hidden_states.as_ref().unwrap());

        let decoder_output = self.decoder.forward_t(
            decoder_input_ids,
            decoder_attention_mask,
            encoder_hidden_states.into(),
            decoder_attention_mask,
            old_layer_states,
            decoder_input_embeds,
            Some(&self.word_embeddings),
            train,
        )?;

        Ok(ProphetNetOutput {
            last_hidden_states: decoder_output.hidden_states,
            ngram_hidden_states: decoder_output.ngram_hidden_states,
            all_decoder_hidden_states: decoder_output.all_hidden_states,
            all_ngram_hidden_states: decoder_output.all_ngram_hidden_states,
            all_attentions: decoder_output.all_attentions,
            all_ngram_attentions: decoder_output.all_ngram_attentions,
            all_cross_attentions: decoder_output.all_cross_attentions,
            next_decoder_cache: decoder_output.next_decoder_cache,
        })
    }
}

pub struct ProphetNetForConditionalGeneration {
    base_model: ProphetNetModel,
    lm_head: nn::Linear,
    decoder_start_token_id: i64,
    pad_token_id: i64,
    ngram: i64,
}

impl ProphetNetForConditionalGeneration {
    pub fn new<'p, P>(
        p: P,
        config: &ProphetNetConfig,
    ) -> Result<ProphetNetForConditionalGeneration, RustBertError>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let base_model = ProphetNetModel::new(p / "prophetnet", config)?;
        let linear_config = nn::LinearConfig {
            bias: false,
            ..Default::default()
        };
        let lm_head = nn::linear(
            p / "lm_head",
            config.hidden_size,
            config.vocab_size,
            linear_config,
        );

        let decoder_start_token_id = config.decoder_start_token_id;
        let pad_token_id = config.pad_token_id;
        let ngram = config.ngram;

        Ok(ProphetNetForConditionalGeneration {
            base_model,
            lm_head,
            decoder_start_token_id,
            pad_token_id,
            ngram,
        })
    }

    fn shift_right(&self, input_ids: &Tensor) -> Tensor {
        let shifted_input_ids = Tensor::zeros(
            input_ids.size().as_slice(),
            (Kind::Int64, input_ids.device()),
        );

        shifted_input_ids
            .slice(-1, 1, *shifted_input_ids.size().last().unwrap(), 1)
            .copy_(&input_ids.slice(-1, 0, *input_ids.size().last().unwrap() - 1, 1));

        let _ = shifted_input_ids
            .get(-1)
            .get(0)
            .fill_(self.decoder_start_token_id);

        let _ = shifted_input_ids.masked_fill(&shifted_input_ids.eq(-100), self.pad_token_id);

        shifted_input_ids
    }

    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        decoder_input_ids: Option<&Tensor>,
        decoder_attention_mask: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
        old_layer_states: Option<Vec<(Option<LayerState>, Option<LayerState>)>>,
        decoder_input_embeds: Option<&Tensor>,
        train: bool,
    ) -> Result<ProphetNetForConditionalGenerationOutput, RustBertError> {
        let calc_decoder_input_ids = if decoder_input_ids.is_none() & decoder_input_embeds.is_none()
        {
            if let Some(input_ids) = input_ids {
                Some(self.shift_right(input_ids))
            } else {
                return Err(RustBertError::ValueError("input_ids must be provided if decoder_input_ids and decoder_input_embeds are not given.".into()));
            }
        } else {
            None
        };

        let decoder_input_ids = if decoder_input_ids.is_some() {
            decoder_input_ids
        } else {
            Some(calc_decoder_input_ids.as_ref().unwrap())
        };

        let base_model_output = self.base_model.forward_t(
            input_ids,
            attention_mask,
            input_embeds,
            decoder_input_ids,
            decoder_attention_mask,
            encoder_hidden_states,
            old_layer_states,
            decoder_input_embeds,
            train,
        )?;

        let (batch_size, sequence_length) = if let Some(decoder_input_ids) = decoder_input_ids {
            let shape = decoder_input_ids.size();
            (shape[0], shape[1])
        } else if let Some(decoder_input_embeds) = decoder_input_embeds {
            let shape = decoder_input_embeds.size();
            (shape[0], shape[1])
        } else {
            return Err(RustBertError::ValueError(
                "At least one of decoder_input_ids or decoder_input_embeds must be set".into(),
            ));
        };

        if base_model_output.ngram_hidden_states.is_none() {
            return Err(RustBertError::InvalidConfigurationError(
                "ngram must be set > 0 in the configuration for conditional generation".into(),
            ));
        }

        let predict_logits = base_model_output
            .ngram_hidden_states
            .as_ref()
            .unwrap()
            .view([batch_size, self.ngram, sequence_length, -1])
            .apply(&self.lm_head);

        let logits = predict_logits.select(1, 0).contiguous();

        let ngram_logits = if self.ngram > 1 {
            Some(predict_logits.slice(1, 1, predict_logits.size()[1], 1))
        } else {
            None
        };

        Ok(ProphetNetForConditionalGenerationOutput {
            logits,
            ngram_logits,
            ngram_hidden_states: base_model_output.ngram_hidden_states,
            all_decoder_hidden_states: base_model_output.all_decoder_hidden_states,
            all_ngram_hidden_states: base_model_output.all_ngram_hidden_states,
            all_attentions: base_model_output.all_attentions,
            all_ngram_attentions: base_model_output.all_ngram_attentions,
            all_cross_attentions: base_model_output.all_cross_attentions,
            next_decoder_cache: base_model_output.next_decoder_cache,
        })
    }

    pub fn encode(
        &self,
        input_ids: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
    ) -> Result<Tensor, RustBertError> {
        Ok(self
            .base_model
            .encoder
            .forward_t(
                input_ids,
                attention_mask,
                input_embeds,
                Some(&self.base_model.word_embeddings),
                false,
            )?
            .hidden_states)
    }
}

impl LMHeadModel for ProphetNetForConditionalGeneration {
    fn forward_t(
        &self,
        input_ids: &Option<Tensor>,
        cache: Cache,
        attention_mask: &Option<Tensor>,
        _token_type_ids: &Option<Tensor>,
        _position_ids: &Option<Tensor>,
        input_embeds: &Option<Tensor>,
        encoder_outputs: Option<&Tensor>,
        decoder_input_ids: &Option<Tensor>,
        train: bool,
    ) -> Result<LMModelOutput, RustBertError> {
        let base_model_output = match cache {
            Cache::ProphetNetCache(cached_layer_states) => self.forward_t(
                input_ids.as_ref(),
                attention_mask.as_ref(),
                input_embeds.as_ref(),
                decoder_input_ids.as_ref(),
                None,
                encoder_outputs,
                cached_layer_states,
                None,
                train,
            )?,
            Cache::None => self.forward_t(
                input_ids.as_ref(),
                attention_mask.as_ref(),
                input_embeds.as_ref(),
                decoder_input_ids.as_ref(),
                None,
                encoder_outputs,
                None,
                None,
                train,
            )?,
            _ => {
                return Err(RustBertError::ValueError(
                    "Cache not compatible with ProphetNet Model".into(),
                ));
            }
        };

        Ok(LMModelOutput {
            lm_logits: base_model_output.logits,
            cache: Cache::ProphetNetCache(base_model_output.next_decoder_cache),
        })
    }
}

///Container holding a ProphetNet model output
pub struct ProphetNetOutput {
    /// last decoder layer hidden state
    pub last_hidden_states: Tensor,
    /// last decoder layer ngram hidden state
    pub ngram_hidden_states: Option<Tensor>,
    /// Hidden states for all intermediate layers
    pub all_decoder_hidden_states: Option<Vec<Tensor>>,
    /// Hidden states (ngram) for all intermediate layers
    pub all_ngram_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all intermediate layers
    pub all_attentions: Option<Vec<Tensor>>,
    /// Ngram attention weights for all intermediate layers
    pub all_ngram_attentions: Option<Vec<Tensor>>,
    /// Cross attention weights for all intermediate layers
    pub all_cross_attentions: Option<Vec<Tensor>>,
    /// Cached outputs of the model (attention layers keys and values) if the model is used for generation
    pub next_decoder_cache: Option<Vec<(Option<LayerState>, Option<LayerState>)>>,
}

///Container holding a ProphetNet model output
pub struct ProphetNetForConditionalGenerationOutput {
    /// Prediction logits
    pub logits: Tensor,
    /// Ngram prediction logits
    pub ngram_logits: Option<Tensor>,
    /// last decoder layer ngram hidden state
    pub ngram_hidden_states: Option<Tensor>,
    /// Hidden states for all intermediate layers
    pub all_decoder_hidden_states: Option<Vec<Tensor>>,
    /// Hidden states (ngram) for all intermediate layers
    pub all_ngram_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all intermediate layers
    pub all_attentions: Option<Vec<Tensor>>,
    /// Ngram attention weights for all intermediate layers
    pub all_ngram_attentions: Option<Vec<Tensor>>,
    /// Cross attention weights for all intermediate layers
    pub all_cross_attentions: Option<Vec<Tensor>>,
    /// Cached outputs of the model (attention layers keys and values) if the model is used for generation
    pub next_decoder_cache: Option<Vec<(Option<LayerState>, Option<LayerState>)>>,
}
