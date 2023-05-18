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
//! Fill in the missing / masked words in input sequences. The pattern to use to specify
//! a masked word can be specified in the `MaskedLanguageConfig` (`mask_token`). and allows
//! multiple masked tokens per input sequence.
//!
//!  ```no_run
//!use rust_bert::bert::{BertConfigResources, BertModelResources, BertVocabResources};
//!use rust_bert::pipelines::common::ModelType;
//!use rust_bert::pipelines::masked_language::{MaskedLanguageConfig, MaskedLanguageModel};
//!use rust_bert::resources::RemoteResource;
//! fn main() -> anyhow::Result<()> {
//!
//!     let config = MaskedLanguageConfig::new(
//!         ModelType::Bert,
//!         RemoteResource::from_pretrained(BertModelResources::BERT),
//!         RemoteResource::from_pretrained(BertConfigResources::BERT),
//!         RemoteResource::from_pretrained(BertVocabResources::BERT),
//!         None,
//!         true,
//!         None,
//!         None,
//!         Some(String::from("<mask>")),
//!     );
//!
//!     let mask_language_model = MaskedLanguageModel::new(config)?;
//!     let input = [
//!         "Hello I am a <mask> student",
//!         "Paris is the <mask> of France. It is <mask> in Europe.",
//!     ];
//!
//!     let output = mask_language_model.predict(input)?;
//!     Ok(())
//! }
//! ```
//!
use crate::bert::BertForMaskedLM;
use crate::common::error::RustBertError;
use crate::deberta::DebertaForMaskedLM;
use crate::deberta_v2::DebertaV2ForMaskedLM;
use crate::fnet::FNetForMaskedLM;
use crate::pipelines::common::{ConfigOption, ModelType, TokenizerOption};
use crate::resources::ResourceProvider;
use crate::roberta::RobertaForMaskedLM;
#[cfg(feature = "remote")]
use crate::{
    bert::{BertConfigResources, BertModelResources, BertVocabResources},
    resources::RemoteResource,
};
use rust_tokenizers::tokenizer::TruncationStrategy;
use rust_tokenizers::TokenizedInput;
use std::borrow::Borrow;
use std::convert::TryFrom;
use tch::nn::VarStore;
use tch::{nn, no_grad, Device, Tensor};

#[derive(Debug, Clone)]
/// Output container for masked language model pipeline.
pub struct MaskedToken {
    /// String representation of the masked word
    pub text: String,
    /// Vocabulary index for the masked word
    pub id: i64,
    /// Score for the masked word
    pub score: f64,
}

/// # Configuration for MaskedLanguageModel
/// Contains information regarding the model to load and device to place the model on.
pub struct MaskedLanguageConfig {
    /// Model type
    pub model_type: ModelType,
    /// Model weights resource (default: pretrained BERT model on CoNLL)
    pub model_resource: Box<dyn ResourceProvider + Send>,
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
    /// Token used for masking words. This is the token which the model will try to predict.
    pub mask_token: Option<String>,
    /// Device to place the model on (default: CUDA/GPU when available)
    pub device: Device,
}

impl MaskedLanguageConfig {
    /// Instantiate a new masked language configuration of the supplied type.
    ///
    /// # Arguments
    ///
    /// * `model_type` - `ModelType` indicating the model type to load (must match with the actual data to be loaded!)
    /// * model_resource - The `ResourceProvider` pointing to the model to load (e.g.  model.ot)
    /// * config - The `ResourceProvider` pointing to the model configuration to load (e.g. config.json)
    /// * vocab - The `ResourceProvider` pointing to the tokenizer's vocabulary to load (e.g.  vocab.txt/vocab.json)
    /// * vocab - An optional `ResourceProvider` pointing to the tokenizer's merge file to load (e.g.  merges.txt), needed only for Roberta.
    /// * lower_case - A `bool` indicating whether the tokenizer should lower case all input (in case of a lower-cased model)
    /// * mask_token - A token used for model to predict masking words..
    pub fn new<RM, RC, RV>(
        model_type: ModelType,
        model_resource: RM,
        config_resource: RC,
        vocab_resource: RV,
        merges_resource: Option<RV>,
        lower_case: bool,
        strip_accents: impl Into<Option<bool>>,
        add_prefix_space: impl Into<Option<bool>>,
        mask_token: impl Into<Option<String>>,
    ) -> MaskedLanguageConfig
    where
        RM: ResourceProvider + Send + 'static,
        RC: ResourceProvider + Send + 'static,
        RV: ResourceProvider + Send + 'static,
    {
        MaskedLanguageConfig {
            model_type,
            model_resource: Box::new(model_resource),
            config_resource: Box::new(config_resource),
            vocab_resource: Box::new(vocab_resource),
            merges_resource: merges_resource.map(|r| Box::new(r) as Box<_>),
            lower_case,
            strip_accents: strip_accents.into(),
            add_prefix_space: add_prefix_space.into(),
            mask_token: mask_token.into(),
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
            RemoteResource::from_pretrained(BertConfigResources::BERT),
            RemoteResource::from_pretrained(BertVocabResources::BERT),
            None,
            true,
            None,
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
                "Masked Language is not implemented for {model_type:?}!",
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
    mask_token: Option<String>,
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
        let vocab_path = config.vocab_resource.get_local_path()?;
        let merges_path = config
            .merges_resource
            .as_ref()
            .map(|resource| resource.get_local_path())
            .transpose()?;

        let tokenizer = TokenizerOption::from_file(
            config.model_type,
            vocab_path.to_str().unwrap(),
            merges_path.as_deref().map(|path| path.to_str().unwrap()),
            config.lower_case,
            config.strip_accents,
            config.add_prefix_space,
        )?;
        Self::new_with_tokenizer(config, tokenizer)
    }

    /// Build a new `MaskedLanguageModel` with a provided tokenizer.
    ///
    /// # Arguments
    ///
    /// * `config` - `MaskedLanguageConfig` object containing the resource references (model, vocabulary, configuration) and device placement (CPU/GPU)
    /// * `tokenizer` - `TokenizerOption` tokenizer to use for masked language modeling
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> anyhow::Result<()> {
    /// use rust_bert::pipelines::common::{ModelType, TokenizerOption};
    /// use rust_bert::pipelines::masked_language::MaskedLanguageModel;
    /// let tokenizer = TokenizerOption::from_file(
    ///     ModelType::Bert,
    ///     "path/to/vocab.txt",
    ///     None,
    ///     false,
    ///     None,
    ///     None,
    /// )?;
    /// let model = MaskedLanguageModel::new_with_tokenizer(Default::default(), tokenizer)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new_with_tokenizer(
        config: MaskedLanguageConfig,
        tokenizer: TokenizerOption,
    ) -> Result<MaskedLanguageModel, RustBertError> {
        let config_path = config.config_resource.get_local_path()?;
        let weights_path = config.model_resource.get_local_path()?;
        let device = config.device;

        let mut var_store = VarStore::new(device);
        let model_config = ConfigOption::from_file(config.model_type, config_path);
        let max_length = model_config
            .get_max_len()
            .map(|v| v as usize)
            .unwrap_or(usize::MAX);

        let language_encode =
            MaskedLanguageOption::new(config.model_type, var_store.root(), &model_config)?;
        var_store.load(weights_path)?;
        let mask_token = config.mask_token;
        Ok(MaskedLanguageModel {
            tokenizer,
            language_encode,
            mask_token,
            var_store,
            max_length,
        })
    }

    /// Get a reference to the model tokenizer.
    pub fn get_tokenizer(&self) -> &TokenizerOption {
        &self.tokenizer
    }

    /// Get a mutable reference to the model tokenizer.
    pub fn get_tokenizer_mut(&mut self) -> &mut TokenizerOption {
        &mut self.tokenizer
    }

    /// Replace custom user-provided mask token by language model mask token.
    fn replace_mask_token<'a, S>(
        &self,
        input: S,
        mask_token: &str,
    ) -> Result<Vec<String>, RustBertError>
    where
        S: AsRef<[&'a str]>,
    {
        let model_mask_token = self.tokenizer.get_mask_value().ok_or_else(||
            RustBertError::InvalidConfigurationError("Tokenizer does ot have a default mask token and no mask token provided in configuration. \
            Please provide a `mask_token` in the configuration.".into()))?;
        let output = input
            .as_ref()
            .iter()
            .map(|&x| x.replace(mask_token, model_mask_token))
            .collect::<Vec<_>>();
        Ok(output)
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
        let tokenized_input_tensors = tokenized_input
            .iter()
            .map(|input| input.token_ids.clone())
            .map(|mut input| {
                input.extend(vec![0; max_len - input.len()]);
                input
            })
            .map(|input| Tensor::from_slice(&(input)))
            .collect::<Vec<_>>();
        Tensor::stack(tokenized_input_tensors.as_slice(), 0).to(self.var_store.device())
    }

    /// Mask texts
    ///
    /// # Arguments
    ///
    /// * `input` - `&[&str]` Array of texts to mask.
    ///
    /// # Returns
    ///
    /// * `Vec<String>` containing masked words for input texts
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> anyhow::Result<()> {
    /// use rust_bert::pipelines::masked_language::MaskedLanguageModel;
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
    /// let output = mask_language_model.predict(&input)?;
    /// for word in output {
    ///     println!("{:?}", word);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn predict<'a, S>(&self, input: S) -> Result<Vec<Vec<MaskedToken>>, RustBertError>
    where
        S: AsRef<[&'a str]>,
    {
        let input_tensor = if let Some(mask_token) = &self.mask_token {
            let input_with_replaced_mask = self.replace_mask_token(input.as_ref(), mask_token)?;
            self.prepare_for_model(
                input_with_replaced_mask
                    .iter()
                    .map(|w| w.as_str())
                    .collect::<Vec<&str>>(),
            )
        } else {
            self.prepare_for_model(input.as_ref())
        };

        let output = no_grad(|| {
            self.language_encode.forward_t(
                Some(&input_tensor),
                None,
                None,
                None,
                None,
                None,
                None,
                false,
            )
        });
        // get the position of mask_token in input texts
        let mask_token_id =
            self.tokenizer
                .get_mask_id()
                .ok_or_else(|| RustBertError::InvalidConfigurationError(
                    "Tokenizer does not have a mask token id, Please use a tokenizer/model with a mask token.".into(),
                ))?;
        let mask_token_mask = input_tensor.eq(mask_token_id);
        let mut output_tokens = Vec::with_capacity(input.as_ref().len());
        for input_id in 0..input.as_ref().len() as i64 {
            let mut sequence_tokens = vec![];
            let sequence_mask = mask_token_mask.get(input_id);
            if bool::try_from(sequence_mask.any())? {
                let mask_scores = output
                    .get(input_id)
                    .index_select(0, &sequence_mask.argwhere().squeeze_dim(1));
                let (token_scores, token_ids) = mask_scores.max_dim(1, false);
                for (id, score) in token_ids.iter::<i64>()?.zip(token_scores.iter::<f64>()?) {
                    let text = self.tokenizer.decode(&[id], false, true);
                    sequence_tokens.push(MaskedToken { text, id, score });
                }
            }
            output_tokens.push(sequence_tokens);
        }
        Ok(output_tokens)
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
