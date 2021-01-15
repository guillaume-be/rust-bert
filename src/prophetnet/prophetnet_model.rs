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
use crate::prophetnet::attention::LayerState;
use crate::prophetnet::decoder::ProphetNetDecoder;
use crate::prophetnet::encoder::ProphetNetEncoder;
use crate::{Activation, Config, RustBertError};
use serde::{Deserialize, Serialize};
use std::borrow::Borrow;
use std::collections::HashMap;
use tch::{nn, Tensor};

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
}

impl ProphetNetConfigResources {
    /// Shared under MIT license by the Microsoft team at https://github.com/microsoft/ProphetNet. Modified with conversion to C-array format.
    pub const PROPHETNET_LARGE_UNCASED: (&'static str, &'static str) = (
        "prophetnet-large-uncased/config",
        "https://huggingface.co/microsoft/prophetnet-large-uncased/resolve/main/config.json",
    );
}

impl ProphetNetVocabResources {
    /// Shared under MIT license by the Microsoft team at https://github.com/microsoft/ProphetNet. Modified with conversion to C-array format.
    pub const PROPHETNET_LARGE_UNCASED: (&'static str, &'static str) = (
        "prophetnet-large-uncased/vocab",
        "https://huggingface.co/microsoft/prophetnet-large-uncased/resolve/main/prophetnet.tokenizer",
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
    word_embeddings: nn::Embedding,
    encoder: ProphetNetEncoder,
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
        encoder_hidden_states: Option<Tensor>,
        encoder_attention_mask: Option<&Tensor>,
        old_layer_states: Option<Vec<(Option<LayerState>, Option<LayerState>)>>,
        decoder_input_embeds: Option<&Tensor>,
        train: bool,
    ) -> Result<ProphetNetOutput, RustBertError> {
        let (encoder_hidden_states, all_encoder_hidden_states, all_encoder_attentions) =
            if let Some(encoder_hidden_states) = encoder_hidden_states {
                (encoder_hidden_states, None, None)
            } else {
                let encoder_hidden_states = self.encoder.forward_t(
                    input_ids,
                    attention_mask,
                    input_embeds,
                    Some(&self.word_embeddings),
                    train,
                )?;
                (
                    encoder_hidden_states.hidden_states,
                    encoder_hidden_states.all_hidden_states,
                    encoder_hidden_states.all_attentions,
                )
            };

        let decoder_output = self.decoder.forward_t(
            decoder_input_ids,
            decoder_attention_mask,
            encoder_hidden_states.as_ref().into(),
            encoder_attention_mask,
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
            last_encoder_hidden_states: encoder_hidden_states,
            all_encoder_hidden_states,
            all_encoder_attentions,
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
    /// last encoder layer hidden state
    pub last_encoder_hidden_states: Tensor,
    /// Hidden states for all encoder intermediate layers
    pub all_encoder_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all encoder intermediate layers
    pub all_encoder_attentions: Option<Vec<Tensor>>,
}
