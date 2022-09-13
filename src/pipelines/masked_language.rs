// Copyright 2019-present, the HuggingFace Inc. team, The Google AI Language Team and Facebook, Inc.
// Copyright 2019-2020 Guillaume Becquin
// Copyright 2020 Maarten van Gompel
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//! # Masked language pipeline (e.g. Fill Mask)
//! More generic masked language pipeline, works with multiple models (Bert, Roberta)
//!
//!  ```no_run
//! use rust_bert::pipelines::common::ModelType;
//! use rust_bert::pipelines::masked_language::{MaskedLanguageConfig, MaskedLanguageModel};
//! use rust_bert::resources::RemoteResource;
//! use rust_bert::roberta::{
//!     RobertaConfigResources, RobertaMergesResources, RobertaModelResources, RobertaVocabResources,
//! };
//! # fn main() -> anyhow::Result<()> {
//! //Load a configuration
//! let config = MaskedLanguageConfig::new(
//!     ModelType::Roberta,
//!     RemoteResource::from_pretrained(RobertaModelResources::DISTILROBERTA_BASE),
//!     None,
//!     RemoteResource::from_pretrained(RobertaConfigResources::DISTILROBERTA_BASE),
//!     RemoteResource::from_pretrained(RobertaVocabResources::DISTILROBERTA_BASE),
//!     RemoteResource::from_pretrained(RobertaMergesResources::DISTILROBERTA_BASE),
//!     true,
//!     None,
//!     None,
//! );
//! //Create the model
//! let mask_language_model = MaskedLanguageModel::new(config)?;
//! let input = [
//!     "Looks like one <mask> is missing!",
//!     "The goal of life is <mask>.",
//! ];
//!
//! //Run model
//! let output = mask_language_model.predict(&input, vec![5, 6]);
//! for word in output {
//!     println!("{:?}", word);
//! }
//!
//! #Ok(())
//! #}
//! ```
//!
//!
use crate::bert::BertForMaskedLM;
use crate::codebert::CodeBertForMaskedLM;
use crate::common::error::RustBertError;
use crate::deberta::DebertaForMaskedLM;
use crate::deberta_v2::DebertaV2ForMaskedLM;
use crate::fnet::FNetForMaskedLM;
use crate::pipelines::common::{ConfigOption, ModelType, TokenizerOption};
use crate::resources::{LocalResource, ResourceProvider};
use crate::roberta::RobertaForMaskedLM;
#[cfg(feature = "remote")]
use crate::{
    bert::{BertConfigResources, BertModelResources, BertVocabResources},
    resources::RemoteResource,
};
use rust_tokenizers::tokenizer::MultiThreadedTokenizer;
use rust_tokenizers::tokenizer::TruncationStrategy;
use rust_tokenizers::vocab::Vocab;
use rust_tokenizers::TokenizedInput;
use std::borrow::Borrow;
use tch::nn::VarStore;
use tch::{nn, no_grad, Device, Tensor};

/// # Configuration for MaskedLanguageModel
/// Contains information regarding the model to load and device to place the model on.
pub struct MaskedLanguageConfig {
    /// Model type
    pub model_type: ModelType,
    /// Model weights resource (default: pretrained BERT model on CoNLL)
    pub model_resource: Option<Box<dyn ResourceProvider + Send>>,
    /// Local model weights resource instead of RemoteResource
    pub model_local_resource: Option<LocalResource>,
    /// Config resource (default: pretrained BERT model on CoNLL)
    pub config_resource: Box<dyn ResourceProvider + Send>,
    /// Vocab resource (default: pretrained BERT model on CoNLL)
    pub vocab_resource: Box<dyn ResourceProvider + Send>,
    /// Merges resource (default: None)
    pub merges_resource: Option<Box<dyn ResourceProvider + Send>>,
    /// Automatically lower case all input upon tokenization (assumes a lower-cased model)
    pub lower_case: bool,
    /// Flag indicating if the tokenizer should strip accents (normalization). Only used for BERT / ALBERT models
    pub strip_accents: Option<bool>,
    /// Flag indicating if the tokenizer should add a white space before each tokenized input (needed for some Roberta models)
    pub add_prefix_space: Option<bool>,
    /// Device to place the model on (default: CUDA/GPU when available)
    pub device: Device,
}

impl MaskedLanguageConfig {
    /// Instantiate a new masked language configuration of the supplied type.
    ///
    /// # Arguments
    ///
    /// * `model_type` - `ModelType` indicating the model type to load (must match with the actual data to be loaded!)
    /// * model_resource - The `ResourceProvider` pointing to the model for RemoteResource to load (e.g.  model.ot)
    /// * model_local_resource - The `ResourceProvider` pointing to the model for LocalResource to load (e.g.  model.ot)
    /// * config - The `ResourceProvider` pointing to the model configuration to load (e.g. config.json)
    /// * vocab - The `ResourceProvider` pointing to the tokenizer's vocabulary to load (e.g.  vocab.txt/vocab.json)
    /// * vocab - An optional `ResourceProvider` pointing to the tokenizer's merge file to load (e.g.  merges.txt), needed only for Roberta.
    /// * lower_case - A `bool` indicating whether the tokenizer should lower case all input (in case of a lower-cased model)
    pub fn new<R>(
        model_type: ModelType,
        model_resource: impl Into<Option<R>>,
        model_local_resource: impl Into<Option<LocalResource>>,
        config_resource: R,
        vocab_resource: R,
        merges_resource: impl Into<Option<R>>,
        lower_case: bool,
        strip_accents: impl Into<Option<bool>>,
        add_prefix_space: impl Into<Option<bool>>,
    ) -> MaskedLanguageConfig
    where
        R: ResourceProvider + Send + 'static,
    {
        MaskedLanguageConfig {
            model_type,
            model_resource: model_resource.into().map(|r| Box::new(r) as Box<_>),
            model_local_resource: model_local_resource.into(),
            config_resource: Box::new(config_resource),
            vocab_resource: Box::new(vocab_resource),
            merges_resource: merges_resource.into().map(|r| Box::new(r) as Box<_>),
            lower_case,
            strip_accents: strip_accents.into(),
            add_prefix_space: add_prefix_space.into(),
            device: Device::cuda_if_available(),
        }
    }
}
#[cfg(feature = "remote")]
impl Default for MaskedLanguageConfig {
    /// Provides a BERT language model
    fn default() -> MaskedLanguageConfig {
        MaskedLanguageConfig::new(
            ModelType::Bert,
            RemoteResource::from_pretrained(BertModelResources::BERT),
            None,
            RemoteResource::from_pretrained(BertConfigResources::BERT),
            RemoteResource::from_pretrained(BertVocabResources::BERT),
            None,
            true,
            None,
            None,
        )
    }
}
#[allow(clippy::large_enum_variant)]
/// # Abstraction that holds one particular masked language model, for any of the supported models
pub enum MaskedLanguageOption {
    /// Bert for Masked Language
    Bert(BertForMaskedLM),
    /// DeBERTa for Masked Language
    Deberta(DebertaForMaskedLM),
    /// DeBERTa V2 for Masked Language
    DebertaV2(DebertaV2ForMaskedLM),
    /// Roberta for Masked Language
    Roberta(RobertaForMaskedLM),
    /// CodeBert for Masked Language
    CodeBert(CodeBertForMaskedLM),
    /// XLMRoberta for Masked Language
    XLMRoberta(RobertaForMaskedLM),
    /// FNet for Masked Language
    FNet(FNetForMaskedLM),
}

impl MaskedLanguageOption {
    /// Instantiate a new masked language model of the supplied type.
    ///
    /// # Arguments
    ///
    /// * `model_type` - `ModelType` indicating the model type to load (must match with the actual data to be loaded)
    /// * `p` - `tch::nn::Path` path to the model file to load (e.g. model.ot)
    /// * `config` - A configuration (the model type of the configuration must be compatible with the value for
    /// `model_type`)
    pub fn new<'p, P>(
        model_type: ModelType,
        p: P,
        config: &ConfigOption,
    ) -> Result<Self, RustBertError>
    where
        P: Borrow<nn::Path<'p>>,
    {
        match model_type {
            ModelType::Bert => {
                if let ConfigOption::Bert(config) = config {
                    Ok(MaskedLanguageOption::Bert(BertForMaskedLM::new(p, config)))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a BertConfig for Bert!".to_string(),
                    ))
                }
            }
            ModelType::Deberta => {
                if let ConfigOption::Deberta(config) = config {
                    Ok(MaskedLanguageOption::Deberta(DebertaForMaskedLM::new(
                        p, config,
                    )))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a DebertaConfig for DeBERTa!".to_string(),
                    ))
                }
            }
            ModelType::DebertaV2 => {
                if let ConfigOption::DebertaV2(config) = config {
                    Ok(MaskedLanguageOption::DebertaV2(DebertaV2ForMaskedLM::new(
                        p, config,
                    )))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a DebertaV2Config for DeBERTa V2!".to_string(),
                    ))
                }
            }
            ModelType::Roberta => {
                if let ConfigOption::Roberta(config) = config {
                    Ok(MaskedLanguageOption::Roberta(RobertaForMaskedLM::new(
                        p, config,
                    )))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a BertConfig for Roberta!".to_string(),
                    ))
                }
            }
            ModelType::CodeBert => {
                if let ConfigOption::CodeBert(config) = config {
                    Ok(MaskedLanguageOption::CodeBert(CodeBertForMaskedLM::new(
                        p, config,
                    )))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a CodeBertConfig for CodeBert!".to_string(),
                    ))
                }
            }
            ModelType::XLMRoberta => {
                if let ConfigOption::Bert(config) = config {
                    Ok(MaskedLanguageOption::XLMRoberta(RobertaForMaskedLM::new(
                        p, config,
                    )))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a BertConfig for Roberta!".to_string(),
                    ))
                }
            }
            ModelType::FNet => {
                if let ConfigOption::FNet(config) = config {
                    Ok(MaskedLanguageOption::FNet(FNetForMaskedLM::new(p, config)))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a FNetConfig for FNet!".to_string(),
                    ))
                }
            }
            _ => Err(RustBertError::InvalidConfigurationError(format!(
                "Masked Language not implemented for {:?}!",
                model_type
            ))),
        }
    }

    /// Returns the `ModelType` for this MaskedLanguageOption
    pub fn model_type(&self) -> ModelType {
        match *self {
            Self::Bert(_) => ModelType::Bert,
            Self::Deberta(_) => ModelType::Deberta,
            Self::DebertaV2(_) => ModelType::DebertaV2,
            Self::Roberta(_) => ModelType::Roberta,
            Self::CodeBert(_) => ModelType::CodeBert,
            Self::XLMRoberta(_) => ModelType::Roberta,
            Self::FNet(_) => ModelType::FNet,
        }
    }

    /// Interface method to forward_t() of the particular models.
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
    ) -> Tensor {
        match *self {
            Self::Bert(ref model) => {
                model
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
                    .prediction_scores
            }

            Self::Deberta(ref model) => {
                model
                    .forward_t(
                        input_ids,
                        mask,
                        token_type_ids,
                        position_ids,
                        input_embeds,
                        train,
                    )
                    .expect("Error in Deberta forward_t")
                    .logits
            }
            Self::DebertaV2(ref model) => {
                model
                    .forward_t(
                        input_ids,
                        mask,
                        token_type_ids,
                        position_ids,
                        input_embeds,
                        train,
                    )
                    .expect("Error in Deberta V2 forward_t")
                    .logits
            }

            Self::Roberta(ref model) | Self::XLMRoberta(ref model) => {
                model
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
                    .prediction_scores
            }
            Self::CodeBert(ref model) => {
                model
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
                    .prediction_scores
            }
            Self::FNet(ref model) => {
                model
                    .forward_t(input_ids, token_type_ids, position_ids, input_embeds, train)
                    .expect("Error in FNet forward pass.")
                    .prediction_scores
            }
        }
    }
}

/// # MaskedLanguageModel for Masked Language (e.g. Fill Mask)
pub struct MaskedLanguageModel {
    tokenizer: TokenizerOption,
    language_encode: MaskedLanguageOption,
    var_store: VarStore,
    max_length: usize,
}

impl MaskedLanguageModel {
    /// Build a new `MaskedLanguageModel`
    ///
    /// # Arguments
    ///
    /// * `config` - `MaskedLanguageConfig` object containing the resource references (model, vocabulary, configuration) and device placement (CPU/GPU)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> anyhow::Result<()> {
    /// use rust_bert::pipelines::masked_language::MaskedLanguageModel;
    ///
    /// let model = MaskedLanguageModel::new(Default::default())?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(config: MaskedLanguageConfig) -> Result<MaskedLanguageModel, RustBertError> {
        let config_path = config.config_resource.get_local_path()?;
        let vocab_path = config.vocab_resource.get_local_path()?;
        let weights_path = if config.model_local_resource.is_none() {
            config.model_resource.unwrap().get_local_path()?
        } else {
            config.model_local_resource.unwrap().get_local_path()?
        };
        let merges_path = if let Some(merges_resource) = &config.merges_resource {
            Some(merges_resource.get_local_path()?)
        } else {
            None
        };
        let device = config.device;

        let tokenizer = TokenizerOption::from_file(
            config.model_type,
            vocab_path.to_str().unwrap(),
            merges_path.as_deref().map(|path| path.to_str().unwrap()),
            config.lower_case,
            config.strip_accents,
            config.add_prefix_space,
        )?;
        let mut var_store = VarStore::new(device);
        let model_config = ConfigOption::from_file(config.model_type, config_path);
        let max_length = model_config
            .get_max_len()
            .map(|v| v as usize)
            .unwrap_or(usize::MAX);

        let language_encode =
            MaskedLanguageOption::new(config.model_type, &var_store.root(), &model_config)?;
        var_store.load(weights_path)?;
        Ok(MaskedLanguageModel {
            tokenizer,
            language_encode,
            var_store,
            max_length,
        })
    }

    fn prepare_for_model<'a, S>(&self, input: S) -> Tensor
    where
        S: AsRef<[&'a str]>,
    {
        let tokenized_input: Vec<TokenizedInput> = self.tokenizer.encode_list(
            input.as_ref(),
            self.max_length,
            &TruncationStrategy::LongestFirst,
            0,
        );
        let max_len = tokenized_input
            .iter()
            .map(|input| input.token_ids.len())
            .max()
            .unwrap();
        let tokenized_input_tensors: Vec<tch::Tensor> = tokenized_input
            .iter()
            .map(|input| input.token_ids.clone())
            .map(|mut input| {
                input.extend(vec![0; max_len - input.len()]);
                input
            })
            .map(|input| Tensor::of_slice(&(input)))
            .collect::<Vec<_>>();
        Tensor::stack(tokenized_input_tensors.as_slice(), 0).to(self.var_store.device())
    }

    fn get_vocab(&self, input: Tensor) -> Result<String, RustBertError> {
        let word = match self.tokenizer {
            TokenizerOption::Bert(ref tokenizer) => {
                MultiThreadedTokenizer::vocab(tokenizer).id_to_token(&input.int64_value(&[]))
            }
            TokenizerOption::Deberta(ref tokenizer) => {
                MultiThreadedTokenizer::vocab(tokenizer).id_to_token(&input.int64_value(&[]))
            }
            TokenizerOption::DebertaV2(ref tokenizer) => {
                MultiThreadedTokenizer::vocab(tokenizer).id_to_token(&input.int64_value(&[]))
            }
            TokenizerOption::Roberta(ref tokenizer) => {
                MultiThreadedTokenizer::vocab(tokenizer).id_to_token(&input.int64_value(&[]))
            }
            TokenizerOption::CodeBert(ref tokenizer) => {
                MultiThreadedTokenizer::vocab(tokenizer).id_to_token(&input.int64_value(&[]))
            }
            TokenizerOption::XLMRoberta(ref tokenizer) => {
                MultiThreadedTokenizer::vocab(tokenizer).id_to_token(&input.int64_value(&[]))
            }
            TokenizerOption::FNet(ref tokenizer) => {
                MultiThreadedTokenizer::vocab(tokenizer).id_to_token(&input.int64_value(&[]))
            }
            _ => return Err(RustBertError::InvalidConfigurationError(
                "Masked Language currently supports Bert|Deberta|DebertaV2|Roberta|CodeBert|XLMRoberta|FNet!".to_string(),
            )),
        };
        Ok(word)
    }

    /// Mask texts
    ///
    /// # Arguments
    ///
    /// * `input` - `&[&str]` Array of texts to mask.
    ///
    /// * `index` - `Vec<i64>` Index array for indicating the positions of masked words.
    ///
    /// # Returns
    ///
    /// * `Vec<String>` containing masked words for input texts
    ///
    /// # Example
    ///
    /// ```no_run
    ///
    /// #fn main() -> anyhow::Result<()> {
    /// #use rust_bert::pipelines::masked_language::MaskedLanguageModel;
    /// //    Set-up model
    /// let mask_language_model = MaskedLanguageModel::new(Default::default())?;
    ///
    /// //    Define input
    /// let input = [
    ///     "Looks like one [MASK] is missing",
    ///     "It was a very nice and [MASK] day",
    /// ];
    ///
    /// //    Run model
    /// let output = mask_language_model.predict(&input, vec![4, 7]);
    /// for word in output {
    ///     println!("{:?}", word);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn predict<'a, S>(&self, input: S, index: Vec<i64>) -> Vec<String>
    where
        S: AsRef<[&'a str]>,
    {
        let input_tensor = self.prepare_for_model(input.as_ref());
        let output = no_grad(|| {
            let output = self.language_encode.forward_t(
                Some(&input_tensor),
                None,
                None,
                None,
                None,
                None,
                None,
                false,
            );
            output.to(Device::Cpu)
        });
        let token_indices = output.size()[0];

        let mut words: Vec<String> = vec![];
        for token_idx in 0..token_indices {
            let word_idx = output
                .get(token_idx)
                .get(index[token_idx as usize])
                .argmax(0, false);
            let word = self.get_vocab(word_idx).unwrap();
            words.push(word)
        }
        words
    }
}
#[cfg(test)]
mod test {
    use super::*;

    #[test]
    #[ignore] // no need to run, compilation is enough to verify it is Send
    fn test() {
        let config = MaskedLanguageConfig::default();
        let _: Box<dyn Send> = Box::new(MaskedLanguageModel::new(config));
    }
}
