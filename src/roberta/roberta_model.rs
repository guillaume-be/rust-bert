// Copyright 2019-present, the HuggingFace Inc. team, The Google AI Language Team and Facebook, Inc.
// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
// Copyright 2019 Guillaume Becquin
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::bert::{BertConfig, BertModel};
use crate::common::activations::_gelu;
use crate::common::dropout::Dropout;
use crate::common::linear::{linear_no_bias, LinearNoBias};
use crate::roberta::embeddings::RobertaEmbeddings;
use crate::RustBertError;
use std::borrow::Borrow;
use tch::nn::init::DEFAULT_KAIMING_UNIFORM;
use tch::{nn, Tensor};

/// # RoBERTa Pretrained model weight files
pub struct RobertaModelResources;

/// # RoBERTa Pretrained model config files
pub struct RobertaConfigResources;

/// # RoBERTa Pretrained model vocab files
pub struct RobertaVocabResources;

/// # RoBERTa Pretrained model merges files
pub struct RobertaMergesResources;

impl RobertaModelResources {
    /// Shared under MIT license by the Facebook AI Research Fairseq team at <https://github.com/pytorch/fairseq>. Modified with conversion to C-array format.
    pub const ROBERTA: (&'static str, &'static str) = (
        "roberta/model",
        "https://huggingface.co/roberta-base/resolve/main/rust_model.ot",
    );
    /// Shared under Apache 2.0 license by the Hugging Face Inc. team at <https://huggingface.co/distilroberta-base>. Modified with conversion to C-array format.
    pub const DISTILROBERTA_BASE: (&'static str, &'static str) = (
        "distilroberta-base/model",
        "https://cdn.huggingface.co/distilroberta-base-rust_model.ot",
    );
    /// Shared under Apache 2.0 license by [deepset](https://deepset.ai) at <https://huggingface.co/deepset/roberta-base-squad2>. Modified with conversion to C-array format.
    pub const ROBERTA_QA: (&'static str, &'static str) = (
        "roberta-qa/model",
        "https://huggingface.co/deepset/roberta-base-squad2/resolve/main/rust_model.ot",
    );
    /// Shared under Apache 2.0 license by the HuggingFace Inc. team at <https://huggingface.co/models>. Modified with conversion to C-array format.
    pub const XLM_ROBERTA_NER_EN: (&'static str, &'static str) = (
        "xlm-roberta-ner-en/model",
        "https://huggingface.co/xlm-roberta-large-finetuned-conll03-english/resolve/main/rust_model.ot",
    );
    /// Shared under Apache 2.0 license by the HuggingFace Inc. team at <https://huggingface.co/models>. Modified with conversion to C-array format.
    pub const XLM_ROBERTA_NER_DE: (&'static str, &'static str) = (
        "xlm-roberta-ner-de/model",
        "https://huggingface.co/xlm-roberta-large-finetuned-conll03-german/resolve/main/rust_model.ot",
    );
    /// Shared under Apache 2.0 license by the HuggingFace Inc. team at <https://huggingface.co/models>. Modified with conversion to C-array format.
    pub const XLM_ROBERTA_NER_NL: (&'static str, &'static str) = (
        "xlm-roberta-ner-nl/model",
        "https://huggingface.co/xlm-roberta-large-finetuned-conll02-dutch/resolve/main/rust_model.ot",
    );
    /// Shared under Apache 2.0 license by the HuggingFace Inc. team at <https://huggingface.co/models>. Modified with conversion to C-array format.
    pub const XLM_ROBERTA_NER_ES: (&'static str, &'static str) = (
        "xlm-roberta-ner-es/model",
        "https://huggingface.co/xlm-roberta-large-finetuned-conll02-spanish/resolve/main/rust_model.ot",
    );
    /// Shared under Apache 2.0 license by the HuggingFace Inc. team at <https://huggingface.co/sentence-transformers/all-distilroberta-v1>. Modified with conversion to C-array format.
    pub const ALL_DISTILROBERTA_V1: (&'static str, &'static str) = (
        "all-distilroberta-v1/model",
        "https://huggingface.co/sentence-transformers/all-distilroberta-v1/resolve/main/rust_model.ot",
    );
    /// Shared under Apache 2.0 license by the HuggingFace Inc. team at <https://huggingface.co/huggingface/CodeBERTa-language-id>. Modified with conversion to C-array format.
    pub const CODEBERTA_LANGUAGE_ID: (&'static str, &'static str) = (
        "codeberta-language-id/model",
        "https://huggingface.co/huggingface/CodeBERTa-language-id/resolve/main/rust_model.ot",
    );
    /// Shared under MIT license by the Microsoft team at <https://github.com/microsoft/CodeBERT>. Modified with conversion to C-array format.
    pub const CODEBERT_MLM: (&'static str, &'static str) = (
        "codebert-mlm/model",
        "https://huggingface.co/microsoft/codebert-base-mlm/resolve/main/rust_model.ot",
    );
}

impl RobertaConfigResources {
    /// Shared under MIT license by the Facebook AI Research Fairseq team at <https://github.com/pytorch/fairseq>. Modified with conversion to C-array format.
    pub const ROBERTA: (&'static str, &'static str) = (
        "roberta/config",
        "https://huggingface.co/roberta-base/resolve/main/config.json",
    );
    /// Shared under Apache 2.0 license by the Hugging Face Inc. team at <https://huggingface.co/distilroberta-base>. Modified with conversion to C-array format.
    pub const DISTILROBERTA_BASE: (&'static str, &'static str) = (
        "distilroberta-base/config",
        "https://cdn.huggingface.co/distilroberta-base-config.json",
    );
    /// Shared under Apache 2.0 license by [deepset](https://deepset.ai) at <https://huggingface.co/deepset/roberta-base-squad2>. Modified with conversion to C-array format.
    pub const ROBERTA_QA: (&'static str, &'static str) = (
        "roberta-qa/config",
        "https://huggingface.co/deepset/roberta-base-squad2/resolve/main/config.json",
    );
    /// Shared under Apache 2.0 license by the HuggingFace Inc. team at <https://huggingface.co/models>. Modified with conversion to C-array format.
    pub const XLM_ROBERTA_NER_EN: (&'static str, &'static str) = (
        "xlm-roberta-ner-en/config",
        "https://huggingface.co/xlm-roberta-large-finetuned-conll03-english/resolve/main/config.json",
    );
    /// Shared under Apache 2.0 license by the HuggingFace Inc. team at <https://huggingface.co/models>. Modified with conversion to C-array format.
    pub const XLM_ROBERTA_NER_DE: (&'static str, &'static str) = (
        "xlm-roberta-ner-de/config",
        "https://huggingface.co/xlm-roberta-large-finetuned-conll03-german/resolve/main/config.json",
    );
    /// Shared under Apache 2.0 license by the HuggingFace Inc. team at <https://huggingface.co/models>. Modified with conversion to C-array format.
    pub const XLM_ROBERTA_NER_NL: (&'static str, &'static str) = (
        "xlm-roberta-ner-nl/config",
        "https://huggingface.co/xlm-roberta-large-finetuned-conll02-dutch/resolve/main/config.json",
    );
    /// Shared under Apache 2.0 license by the HuggingFace Inc. team at <https://huggingface.co/models>. Modified with conversion to C-array format.
    pub const XLM_ROBERTA_NER_ES: (&'static str, &'static str) = (
        "xlm-roberta-ner-es/config",
        "https://huggingface.co/xlm-roberta-large-finetuned-conll02-spanish/resolve/main/config.json",
    );
    /// Shared under Apache 2.0 licenseat <https://huggingface.co/sentence-transformers/all-distilroberta-v1>. Modified with conversion to C-array format.
    pub const ALL_DISTILROBERTA_V1: (&'static str, &'static str) = (
        "all-distilroberta-v1/config",
        "https://huggingface.co/sentence-transformers/all-distilroberta-v1/resolve/main/config.json",
    );
    /// Shared under Apache 2.0 license by the HuggingFace Inc. team at <https://huggingface.co/huggingface/CodeBERTa-language-id>. Modified with conversion to C-array format.
    pub const CODEBERTA_LANGUAGE_ID: (&'static str, &'static str) = (
        "codeberta-language-id/config",
        "https://huggingface.co/huggingface/CodeBERTa-language-id/resolve/main/config.json",
    );
    /// Shared under MIT license by the Microsoft team at <https://github.com/microsoft/CodeBERT>. Modified with conversion to C-array format.
    pub const CODEBERT_MLM: (&'static str, &'static str) = (
        "codebert-mlm/config",
        "https://huggingface.co/microsoft/codebert-base-mlm/resolve/main/config.json",
    );
}

impl RobertaVocabResources {
    /// Shared under MIT license by the Facebook AI Research Fairseq team at <https://github.com/pytorch/fairseq>. Modified with conversion to C-array format.
    pub const ROBERTA: (&'static str, &'static str) = (
        "roberta/vocab",
        "https://huggingface.co/roberta-base/resolve/main/vocab.json",
    );
    /// Shared under Apache 2.0 license by the Hugging Face Inc. team at <https://huggingface.co/distilroberta-base>. Modified with conversion to C-array format.
    pub const DISTILROBERTA_BASE: (&'static str, &'static str) = (
        "distilroberta-base/vocab",
        "https://cdn.huggingface.co/distilroberta-base-vocab.json",
    );
    /// Shared under Apache 2.0 license by [deepset](https://deepset.ai) at <https://huggingface.co/deepset/roberta-base-squad2>. Modified with conversion to C-array format.
    pub const ROBERTA_QA: (&'static str, &'static str) = (
        "roberta-qa/vocab",
        "https://huggingface.co/deepset/roberta-base-squad2/resolve/main/vocab.json",
    );
    /// Shared under Apache 2.0 license by the HuggingFace Inc. team at <https://huggingface.co/models>. Modified with conversion to C-array format.
    pub const XLM_ROBERTA_NER_EN: (&'static str, &'static str) = (
        "xlm-roberta-ner-en/spiece",
        "https://huggingface.co/xlm-roberta-large-finetuned-conll03-english/resolve/main/sentencepiece.bpe.model",
    );
    /// Shared under Apache 2.0 license by the HuggingFace Inc. team at <https://huggingface.co/models>. Modified with conversion to C-array format.
    pub const XLM_ROBERTA_NER_DE: (&'static str, &'static str) = (
        "xlm-roberta-ner-de/spiece",
        "https://huggingface.co/xlm-roberta-large-finetuned-conll03-german/resolve/main/sentencepiece.bpe.model",
    );
    /// Shared under Apache 2.0 license by the HuggingFace Inc. team at <https://huggingface.co/models>. Modified with conversion to C-array format.
    pub const XLM_ROBERTA_NER_NL: (&'static str, &'static str) = (
        "xlm-roberta-ner-nl/spiece",
        "https://huggingface.co/xlm-roberta-large-finetuned-conll02-dutch/resolve/main/sentencepiece.bpe.model",
    );
    /// Shared under Apache 2.0 license by the HuggingFace Inc. team at <https://huggingface.co/models>. Modified with conversion to C-array format.
    pub const XLM_ROBERTA_NER_ES: (&'static str, &'static str) = (
        "xlm-roberta-ner-es/spiece",
        "https://huggingface.co/xlm-roberta-large-finetuned-conll02-spanish/resolve/main/sentencepiece.bpe.model",
    );
    /// Shared under Apache 2.0 licenseat <https://huggingface.co/sentence-transformers/all-distilroberta-v1>. Modified with conversion to C-array format.
    pub const ALL_DISTILROBERTA_V1: (&'static str, &'static str) = (
        "all-distilroberta-v1/vocab",
        "https://huggingface.co/sentence-transformers/all-distilroberta-v1/resolve/main/vocab.json",
    );
    /// Shared under Apache 2.0 license by the HuggingFace Inc. team at <https://huggingface.co/huggingface/CodeBERTa-language-id>. Modified with conversion to C-array format.
    pub const CODEBERTA_LANGUAGE_ID: (&'static str, &'static str) = (
        "codeberta-language-id/vocab",
        "https://huggingface.co/huggingface/CodeBERTa-language-id/resolve/main/vocab.json",
    );
    /// Shared under MIT license by the Microsoft team at <https://github.com/microsoft/CodeBERT>. Modified with conversion to C-array format.
    pub const CODEBERT_MLM: (&'static str, &'static str) = (
        "codebert-mlm/vocab",
        "https://huggingface.co/microsoft/codebert-base-mlm/resolve/main/vocab.json",
    );
}

impl RobertaMergesResources {
    /// Shared under MIT license by the Facebook AI Research Fairseq team at <https://github.com/pytorch/fairseq>. Modified with conversion to C-array format.
    pub const ROBERTA: (&'static str, &'static str) = (
        "roberta/merges",
        "https://huggingface.co/roberta-base/resolve/main/merges.txt",
    );
    /// Shared under Apache 2.0 license by the Hugging Face Inc. team at <https://huggingface.co/distilroberta-base>. Modified with conversion to C-array format.
    pub const DISTILROBERTA_BASE: (&'static str, &'static str) = (
        "distilroberta-base/merges",
        "https://cdn.huggingface.co/distilroberta-base-merges.txt",
    );
    /// Shared under Apache 2.0 license by [deepset](https://deepset.ai) at <https://huggingface.co/deepset/roberta-base-squad2>. Modified with conversion to C-array format.
    pub const ROBERTA_QA: (&'static str, &'static str) = (
        "roberta-qa/merges",
        "https://huggingface.co/deepset/roberta-base-squad2/resolve/main/merges.txt",
    );
    /// Shared under Apache 2.0 licenseat <https://huggingface.co/sentence-transformers/all-distilroberta-v1>. Modified with conversion to C-array format.
    pub const ALL_DISTILROBERTA_V1: (&'static str, &'static str) = (
        "all-distilroberta-v1/merges",
        "https://huggingface.co/sentence-transformers/all-distilroberta-v1/resolve/main/merges.txt",
    );
    /// Shared under Apache 2.0 license by the HuggingFace Inc. team at <https://huggingface.co/huggingface/CodeBERTa-language-id>. Modified with conversion to C-array format.
    pub const CODEBERTA_LANGUAGE_ID: (&'static str, &'static str) = (
        "codeberta-language-id/merges",
        "https://huggingface.co/huggingface/CodeBERTa-language-id/resolve/main/merges.txt",
    );
    /// Shared under MIT license by the Microsoft team at <https://github.com/microsoft/CodeBERT>. Modified with conversion to C-array format.
    pub const CODEBERT_MLM: (&'static str, &'static str) = (
        "codebert-mlm/merges",
        "https://huggingface.co/microsoft/codebert-base-mlm/resolve/main/merges.txt",
    );
}

pub struct RobertaLMHead {
    dense: nn::Linear,
    decoder: LinearNoBias,
    layer_norm: nn::LayerNorm,
    bias: Tensor,
}

impl RobertaLMHead {
    pub fn new<'p, P>(p: P, config: &BertConfig) -> RobertaLMHead
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let dense = nn::linear(
            p / "dense",
            config.hidden_size,
            config.hidden_size,
            Default::default(),
        );
        let layer_norm_config = nn::LayerNormConfig {
            eps: 1e-12,
            ..Default::default()
        };
        let layer_norm = nn::layer_norm(
            p / "layer_norm",
            vec![config.hidden_size],
            layer_norm_config,
        );
        let decoder = linear_no_bias(
            p / "decoder",
            config.hidden_size,
            config.vocab_size,
            Default::default(),
        );
        let bias = p.var("bias", &[config.vocab_size], DEFAULT_KAIMING_UNIFORM);

        RobertaLMHead {
            dense,
            decoder,
            layer_norm,
            bias,
        }
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Tensor {
        (_gelu(&hidden_states.apply(&self.dense)))
            .apply(&self.layer_norm)
            .apply(&self.decoder)
            + &self.bias
    }
}

/// # RoBERTa model configuration
/// Defines the RoBERTa model architecture (e.g. number of layers, hidden layer size, label mapping...)
pub type RobertaConfig = BertConfig;

/// # RoBERTa for masked language model
/// Base RoBERTa model with a RoBERTa masked language model head to predict missing tokens, for example `"Looks like one [MASK] is missing" -> "person"`
/// It is made of the following blocks:
/// - `roberta`: Base BertModel with RoBERTa embeddings
/// - `lm_head`: RoBERTa LM prediction head
pub struct RobertaForMaskedLM {
    roberta: BertModel<RobertaEmbeddings>,
    lm_head: RobertaLMHead,
}

impl RobertaForMaskedLM {
    /// Build a new `RobertaForMaskedLM`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the RobertaForMaskedLM model
    /// * `config` - `RobertaConfig` object defining the model architecture and vocab size
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::roberta::{RobertaConfig, RobertaForMaskedLM};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = RobertaConfig::from_file(config_path);
    /// let roberta = RobertaForMaskedLM::new(&p.root() / "roberta", &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &BertConfig) -> RobertaForMaskedLM
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let roberta =
            BertModel::<RobertaEmbeddings>::new_with_optional_pooler(p / "roberta", config, false);
        let lm_head = RobertaLMHead::new(p / "lm_head", config);

        RobertaForMaskedLM { roberta, lm_head }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see *input_embeds*)
    /// * `mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `token_type_ids` -Optional segment id of shape (*batch size*, *sequence_length*). Convention is value of 0 for the first sentence (incl. *</s>*) and 1 for the second sentence. If None set to 0.
    /// * `position_ids` - Optional position ids of shape (*batch size*, *sequence_length*). If None, will be incremented from 0.
    /// * `input_embeds` - Optional pre-computed input embeddings of shape (*batch size*, *sequence_length*, *hidden_size*). If None, input ids must be provided (see *input_ids*)
    /// * `encoder_hidden_states` - Optional encoder hidden state of shape (*batch size*, *encoder_sequence_length*, *hidden_size*). If the model is defined as a decoder and the *encoder_hidden_states* is not None, used in the cross-attention layer as keys and values (query from the decoder).
    /// * `encoder_mask` - Optional encoder attention mask of shape (*batch size*, *encoder_sequence_length*). If the model is defined as a decoder and the *encoder_hidden_states* is not None, used to mask encoder values. Positions with value 0 will be masked.
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `output` - `Tensor` of shape (*batch size*, *num_labels*, *vocab_size*)
    /// * `hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    /// * `attentions` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rust_bert::bert::BertConfig;
    /// # use tch::{nn, Device, Tensor, no_grad};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::Int64;
    /// use rust_bert::roberta::RobertaForMaskedLM;
    /// # let config_path = Path::new("path/to/config.json");
    /// # let vocab_path = Path::new("path/to/vocab.txt");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = BertConfig::from_file(config_path);
    /// # let roberta_model = RobertaForMaskedLM::new(&vs.root(), &config);
    /// let (batch_size, sequence_length) = (64, 128);
    /// let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    /// let mask = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    /// let token_type_ids = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    /// let position_ids = Tensor::arange(sequence_length, (Int64, device))
    ///     .expand(&[batch_size, sequence_length], true);
    ///
    /// let model_output = no_grad(|| {
    ///     roberta_model.forward_t(
    ///         Some(&input_tensor),
    ///         Some(&mask),
    ///         Some(&token_type_ids),
    ///         Some(&position_ids),
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
        mask: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
        encoder_mask: Option<&Tensor>,
        train: bool,
    ) -> RobertaMaskedLMOutput {
        let base_model_output = self
            .roberta
            .forward_t(
                input_ids,
                mask,
                token_type_ids,
                position_ids,
                input_embeds,
                encoder_hidden_states,
                encoder_mask,
                train,
            )
            .unwrap();

        let prediction_scores = self.lm_head.forward(&base_model_output.hidden_state);
        RobertaMaskedLMOutput {
            prediction_scores,
            all_hidden_states: base_model_output.all_hidden_states,
            all_attentions: base_model_output.all_attentions,
        }
    }
}

pub struct RobertaClassificationHead {
    dense: nn::Linear,
    dropout: Dropout,
    out_proj: nn::Linear,
}

impl RobertaClassificationHead {
    pub fn new<'p, P>(p: P, config: &BertConfig) -> Result<RobertaClassificationHead, RustBertError>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let dense = nn::linear(
            p / "dense",
            config.hidden_size,
            config.hidden_size,
            Default::default(),
        );
        let num_labels = config
            .id2label
            .as_ref()
            .ok_or_else(|| {
                RustBertError::InvalidConfigurationError(
                    "num_labels not provided in configuration".to_string(),
                )
            })?
            .len() as i64;
        let out_proj = nn::linear(
            p / "out_proj",
            config.hidden_size,
            num_labels,
            Default::default(),
        );
        let dropout = Dropout::new(config.hidden_dropout_prob);

        Ok(RobertaClassificationHead {
            dense,
            dropout,
            out_proj,
        })
    }

    pub fn forward_t(&self, hidden_states: &Tensor, train: bool) -> Tensor {
        hidden_states
            .select(1, 0)
            .apply_t(&self.dropout, train)
            .apply(&self.dense)
            .tanh()
            .apply_t(&self.dropout, train)
            .apply(&self.out_proj)
    }
}

/// # RoBERTa for sequence classification
/// Base RoBERTa model with a classifier head to perform sentence or document-level classification
/// It is made of the following blocks:
/// - `roberta`: Base RoBERTa model
/// - `classifier`: RoBERTa classification head made of 2 linear layers
pub struct RobertaForSequenceClassification {
    roberta: BertModel<RobertaEmbeddings>,
    classifier: RobertaClassificationHead,
}

impl RobertaForSequenceClassification {
    /// Build a new `RobertaForSequenceClassification`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the RobertaForMaskedLM model
    /// * `config` - `RobertaConfig` object defining the model architecture and vocab size
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::roberta::{RobertaConfig, RobertaForSequenceClassification};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = RobertaConfig::from_file(config_path);
    /// let roberta = RobertaForSequenceClassification::new(&p.root() / "roberta", &config).unwrap();
    /// ```
    pub fn new<'p, P>(
        p: P,
        config: &BertConfig,
    ) -> Result<RobertaForSequenceClassification, RustBertError>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let roberta =
            BertModel::<RobertaEmbeddings>::new_with_optional_pooler(p / "roberta", config, false);
        let classifier = RobertaClassificationHead::new(p / "classifier", config)?;

        Ok(RobertaForSequenceClassification {
            roberta,
            classifier,
        })
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see `input_embeds`)
    /// * `mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `token_type_ids` -Optional segment id of shape (*batch size*, *sequence_length*). Convention is value of 0 for the first sentence (incl. *</s>*) and 1 for the second sentence. If None set to 0.
    /// * `position_ids` - Optional position ids of shape (*batch size*, *sequence_length*). If None, will be incremented from 0.
    /// * `input_embeds` - Optional pre-computed input embeddings of shape (*batch size*, *sequence_length*, *hidden_size*). If None, input ids must be provided (see `input_ids`)
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `RobertaSequenceClassificationOutput` containing:
    ///   - `logits` - `Tensor` of shape (*batch size*, *num_labels*)
    ///   - `all_hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `all_attentions` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rust_bert::bert::BertConfig;
    /// # use tch::{nn, Device, Tensor, no_grad};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::Int64;
    /// use rust_bert::roberta::RobertaForSequenceClassification;
    /// # let config_path = Path::new("path/to/config.json");
    /// # let vocab_path = Path::new("path/to/vocab.txt");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = BertConfig::from_file(config_path);
    /// # let roberta_model = RobertaForSequenceClassification::new(&vs.root(), &config).unwrap();;
    /// let (batch_size, sequence_length) = (64, 128);
    /// let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    /// let mask = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    /// let token_type_ids = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    /// let position_ids = Tensor::arange(sequence_length, (Int64, device))
    ///     .expand(&[batch_size, sequence_length], true);
    ///
    /// let model_output = no_grad(|| {
    ///     roberta_model.forward_t(
    ///         Some(&input_tensor),
    ///         Some(&mask),
    ///         Some(&token_type_ids),
    ///         Some(&position_ids),
    ///         None,
    ///         false,
    ///     )
    /// });
    /// ```
    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        mask: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        train: bool,
    ) -> RobertaSequenceClassificationOutput {
        let base_model_output = self
            .roberta
            .forward_t(
                input_ids,
                mask,
                token_type_ids,
                position_ids,
                input_embeds,
                None,
                None,
                train,
            )
            .unwrap();

        let logits = self
            .classifier
            .forward_t(&base_model_output.hidden_state, train);
        RobertaSequenceClassificationOutput {
            logits,
            all_hidden_states: base_model_output.all_hidden_states,
            all_attentions: base_model_output.all_attentions,
        }
    }
}

/// # RoBERTa for multiple choices
/// Multiple choices model using a RoBERTa base model and a linear classifier.
/// Input should be in the form `<s> Context </s> Possible choice </s>`. The choice is made along the batch axis,
/// assuming all elements of the batch are alternatives to be chosen from for a given context.
/// It is made of the following blocks:
/// - `roberta`: Base RoBERTa model
/// - `classifier`: Linear layer for multiple choices
pub struct RobertaForMultipleChoice {
    roberta: BertModel<RobertaEmbeddings>,
    dropout: Dropout,
    classifier: nn::Linear,
}

impl RobertaForMultipleChoice {
    /// Build a new `RobertaForMultipleChoice`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the RobertaForMaskedLM model
    /// * `config` - `RobertaConfig` object defining the model architecture and vocab size
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::roberta::{RobertaConfig, RobertaForMultipleChoice};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = RobertaConfig::from_file(config_path);
    /// let roberta = RobertaForMultipleChoice::new(&p.root() / "roberta", &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &BertConfig) -> RobertaForMultipleChoice
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let roberta = BertModel::<RobertaEmbeddings>::new(p / "roberta", config);
        let dropout = Dropout::new(config.hidden_dropout_prob);
        let classifier = nn::linear(p / "classifier", config.hidden_size, 1, Default::default());

        RobertaForMultipleChoice {
            roberta,
            dropout,
            classifier,
        }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Input tensor of shape (*batch size*, *sequence_length*).
    /// * `mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `token_type_ids` -Optional segment id of shape (*batch size*, *sequence_length*). Convention is value of 0 for the first sentence (incl. *</s>*) and 1 for the second sentence. If None set to 0.
    /// * `position_ids` - Optional position ids of shape (*batch size*, *sequence_length*). If None, will be incremented from 0.
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `RobertaSequenceClassificationOutput` containing:
    ///   - `logits` - `Tensor` of shape (*1*, *batch size*) containing the logits for each of the alternatives given
    ///   - `all_hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `all_attentions` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rust_bert::bert::BertConfig;
    /// # use tch::{nn, Device, Tensor, no_grad};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::Int64;
    /// use rust_bert::roberta::RobertaForMultipleChoice;
    /// # let config_path = Path::new("path/to/config.json");
    /// # let vocab_path = Path::new("path/to/vocab.txt");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = BertConfig::from_file(config_path);
    /// # let roberta_model = RobertaForMultipleChoice::new(&vs.root(), &config);
    /// let (num_choices, sequence_length) = (3, 128);
    /// let input_tensor = Tensor::rand(&[num_choices, sequence_length], (Int64, device));
    /// let mask = Tensor::zeros(&[num_choices, sequence_length], (Int64, device));
    /// let token_type_ids = Tensor::zeros(&[num_choices, sequence_length], (Int64, device));
    /// let position_ids = Tensor::arange(sequence_length, (Int64, device))
    ///     .expand(&[num_choices, sequence_length], true);
    ///
    /// let model_output = no_grad(|| {
    ///     roberta_model.forward_t(
    ///         &input_tensor,
    ///         Some(&mask),
    ///         Some(&token_type_ids),
    ///         Some(&position_ids),
    ///         false,
    ///     )
    /// });
    /// ```
    pub fn forward_t(
        &self,
        input_ids: &Tensor,
        mask: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        train: bool,
    ) -> RobertaSequenceClassificationOutput {
        let num_choices = input_ids.size()[1];

        let input_ids = Some(input_ids.view((-1, *input_ids.size().last().unwrap())));
        let mask = mask.map(|tensor| tensor.view((-1, *tensor.size().last().unwrap())));
        let token_type_ids =
            token_type_ids.map(|tensor| tensor.view((-1, *tensor.size().last().unwrap())));
        let position_ids =
            position_ids.map(|tensor| tensor.view((-1, *tensor.size().last().unwrap())));

        let base_model_output = self
            .roberta
            .forward_t(
                input_ids.as_ref(),
                mask.as_ref(),
                token_type_ids.as_ref(),
                position_ids.as_ref(),
                None,
                None,
                None,
                train,
            )
            .unwrap();

        let logits = base_model_output
            .pooled_output
            .unwrap()
            .apply_t(&self.dropout, train)
            .apply(&self.classifier)
            .view((-1, num_choices));
        RobertaSequenceClassificationOutput {
            logits,
            all_hidden_states: base_model_output.all_hidden_states,
            all_attentions: base_model_output.all_attentions,
        }
    }
}

/// # RoBERTa for token classification (e.g. NER, POS)
/// Token-level classifier predicting a label for each token provided. Note that because of bpe tokenization, the labels predicted are
/// not necessarily aligned with words in the sentence.
/// It is made of the following blocks:
/// - `roberta`: Base RoBERTa model
/// - `classifier`: Linear layer for token classification
pub struct RobertaForTokenClassification {
    roberta: BertModel<RobertaEmbeddings>,
    dropout: Dropout,
    classifier: nn::Linear,
}

impl RobertaForTokenClassification {
    /// Build a new `RobertaForTokenClassification`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the RobertaForMaskedLM model
    /// * `config` - `RobertaConfig` object defining the model architecture and vocab size
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::roberta::{RobertaConfig, RobertaForMultipleChoice};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = RobertaConfig::from_file(config_path);
    /// let roberta = RobertaForMultipleChoice::new(&p.root() / "roberta", &config);
    /// ```
    pub fn new<'p, P>(
        p: P,
        config: &BertConfig,
    ) -> Result<RobertaForTokenClassification, RustBertError>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let roberta =
            BertModel::<RobertaEmbeddings>::new_with_optional_pooler(p / "roberta", config, false);
        let dropout = Dropout::new(config.hidden_dropout_prob);
        let num_labels = config
            .id2label
            .as_ref()
            .ok_or_else(|| {
                RustBertError::InvalidConfigurationError(
                    "num_labels not provided in configuration".to_string(),
                )
            })?
            .len() as i64;
        let classifier = nn::linear(
            p / "classifier",
            config.hidden_size,
            num_labels,
            Default::default(),
        );

        Ok(RobertaForTokenClassification {
            roberta,
            dropout,
            classifier,
        })
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see `input_embeds`)
    /// * `mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `token_type_ids` -Optional segment id of shape (*batch size*, *sequence_length*). Convention is value of 0 for the first sentence (incl. *</s>*) and 1 for the second sentence. If None set to 0.
    /// * `position_ids` - Optional position ids of shape (*batch size*, *sequence_length*). If None, will be incremented from 0.
    /// * `input_embeds` - Optional pre-computed input embeddings of shape (*batch size*, *sequence_length*, *hidden_size*). If None, input ids must be provided (see `input_ids`)
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `RobertaTokenClassificationOutput` containing:
    ///   - `logits` - `Tensor` of shape (*batch size*, *sequence_length*, *num_labels*) containing the logits for each of the input tokens and classes
    ///   - `all_hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `all_attentions` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rust_bert::bert::BertConfig;
    /// # use tch::{nn, Device, Tensor, no_grad};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::Int64;
    /// use rust_bert::roberta::RobertaForTokenClassification;
    /// # let config_path = Path::new("path/to/config.json");
    /// # let vocab_path = Path::new("path/to/vocab.txt");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = BertConfig::from_file(config_path);
    /// # let roberta_model = RobertaForTokenClassification::new(&vs.root(), &config).unwrap();
    /// let (batch_size, sequence_length) = (64, 128);
    /// let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    /// let mask = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    /// let token_type_ids = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    /// let position_ids = Tensor::arange(sequence_length, (Int64, device))
    ///     .expand(&[batch_size, sequence_length], true);
    ///
    /// let model_output = no_grad(|| {
    ///     roberta_model.forward_t(
    ///         Some(&input_tensor),
    ///         Some(&mask),
    ///         Some(&token_type_ids),
    ///         Some(&position_ids),
    ///         None,
    ///         false,
    ///     )
    /// });
    /// ```
    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        mask: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        train: bool,
    ) -> RobertaTokenClassificationOutput {
        let base_model_output = self
            .roberta
            .forward_t(
                input_ids,
                mask,
                token_type_ids,
                position_ids,
                input_embeds,
                None,
                None,
                train,
            )
            .unwrap();

        let logits = base_model_output
            .hidden_state
            .apply_t(&self.dropout, train)
            .apply(&self.classifier);

        RobertaTokenClassificationOutput {
            logits,
            all_hidden_states: base_model_output.all_hidden_states,
            all_attentions: base_model_output.all_attentions,
        }
    }
}

/// # RoBERTa for question answering
/// Extractive question-answering model based on a RoBERTa language model. Identifies the segment of a context that answers a provided question.
/// Please note that a significant amount of pre- and post-processing is required to perform end-to-end question answering.
/// See the question answering pipeline (also provided in this crate) for more details.
/// It is made of the following blocks:
/// - `roberta`: Base RoBERTa model
/// - `qa_outputs`: Linear layer for question answering
pub struct RobertaForQuestionAnswering {
    roberta: BertModel<RobertaEmbeddings>,
    qa_outputs: nn::Linear,
}

impl RobertaForQuestionAnswering {
    /// Build a new `RobertaForQuestionAnswering`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the RobertaForMaskedLM model
    /// * `config` - `RobertaConfig` object defining the model architecture and vocab size
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::roberta::{RobertaConfig, RobertaForQuestionAnswering};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = RobertaConfig::from_file(config_path);
    /// let roberta = RobertaForQuestionAnswering::new(&p.root() / "roberta", &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &BertConfig) -> RobertaForQuestionAnswering
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let roberta =
            BertModel::<RobertaEmbeddings>::new_with_optional_pooler(p / "roberta", config, false);
        let num_labels = 2;
        let qa_outputs = nn::linear(
            p / "qa_outputs",
            config.hidden_size,
            num_labels,
            Default::default(),
        );

        RobertaForQuestionAnswering {
            roberta,
            qa_outputs,
        }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see `input_embeds`)
    /// * `mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `token_type_ids` -Optional segment id of shape (*batch size*, *sequence_length*). Convention is value of 0 for the first sentence (incl. *</s>*) and 1 for the second sentence. If None set to 0.
    /// * `position_ids` - Optional position ids of shape (*batch size*, *sequence_length*). If None, will be incremented from 0.
    /// * `input_embeds` - Optional pre-computed input embeddings of shape (*batch size*, *sequence_length*, *hidden_size*). If None, input ids must be provided (see `input_ids`)
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `RobertaQuestionAnsweringOutput` containing:
    ///   - `start_logits` - `Tensor` of shape (*batch size*, *sequence_length*) containing the logits for start of the answer
    ///   - `end_logits` - `Tensor` of shape (*batch size*, *sequence_length*) containing the logits for end of the answer
    ///   - `all_hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `all_attentions` - `Option<Vec<Vec<Tensor>>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rust_bert::bert::BertConfig;
    /// # use tch::{nn, Device, Tensor, no_grad};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::Int64;
    /// use rust_bert::roberta::RobertaForQuestionAnswering;
    /// # let config_path = Path::new("path/to/config.json");
    /// # let vocab_path = Path::new("path/to/vocab.txt");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = BertConfig::from_file(config_path);
    /// # let roberta_model = RobertaForQuestionAnswering::new(&vs.root(), &config);
    /// let (batch_size, sequence_length) = (64, 128);
    /// let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    /// let mask = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    /// let token_type_ids = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    /// let position_ids = Tensor::arange(sequence_length, (Int64, device))
    ///     .expand(&[batch_size, sequence_length], true);
    ///
    /// let model_output = no_grad(|| {
    ///     roberta_model.forward_t(
    ///         Some(&input_tensor),
    ///         Some(&mask),
    ///         Some(&token_type_ids),
    ///         Some(&position_ids),
    ///         None,
    ///         false,
    ///     )
    /// });
    /// ```
    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        mask: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        train: bool,
    ) -> RobertaQuestionAnsweringOutput {
        let base_model_output = self
            .roberta
            .forward_t(
                input_ids,
                mask,
                token_type_ids,
                position_ids,
                input_embeds,
                None,
                None,
                train,
            )
            .unwrap();

        let sequence_output = base_model_output.hidden_state.apply(&self.qa_outputs);
        let logits = sequence_output.split(1, -1);
        let (start_logits, end_logits) = (&logits[0], &logits[1]);
        let start_logits = start_logits.squeeze_dim(-1);
        let end_logits = end_logits.squeeze_dim(-1);

        RobertaQuestionAnsweringOutput {
            start_logits,
            end_logits,
            all_hidden_states: base_model_output.all_hidden_states,
            all_attentions: base_model_output.all_attentions,
        }
    }
}

/// # RoBERTa for sentence embeddings
/// Transformer usable in [`SentenceEmbeddingsModel`](crate::pipelines::sentence_embeddings::SentenceEmbeddingsModel).
pub type RobertaForSentenceEmbeddings = BertModel<RobertaEmbeddings>;

/// Container for the RoBERTa masked LM model output.
pub struct RobertaMaskedLMOutput {
    /// Logits for the vocabulary items at each sequence position
    pub prediction_scores: Tensor,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all intermediate layers
    pub all_attentions: Option<Vec<Tensor>>,
}

/// Container for the RoBERTa sequence classification model output.
pub struct RobertaSequenceClassificationOutput {
    /// Logits for each input (sequence) for each target class
    pub logits: Tensor,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all intermediate layers
    pub all_attentions: Option<Vec<Tensor>>,
}

/// Container for the RoBERTa token classification model output.
pub struct RobertaTokenClassificationOutput {
    /// Logits for each sequence item (token) for each target class
    pub logits: Tensor,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all intermediate layers
    pub all_attentions: Option<Vec<Tensor>>,
}

/// Container for the RoBERTa question answering model output.
pub struct RobertaQuestionAnsweringOutput {
    /// Logits for the start position for token of each input sequence
    pub start_logits: Tensor,
    /// Logits for the end position for token of each input sequence
    pub end_logits: Tensor,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all intermediate layers
    pub all_attentions: Option<Vec<Tensor>>,
}
