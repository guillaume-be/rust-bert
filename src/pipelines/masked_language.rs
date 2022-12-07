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
//!use rust_bert::bert::{BertConfigResources, BertModelResources, BertVocabResources};
//!use rust_bert::pipelines::common::ModelType;
//!use rust_bert::pipelines::masked_language::{MaskedLanguageConfig, MaskedLanguageModel};
//!use rust_bert::resources::RemoteResource;
//!fn main() -> anyhow::Result<()> {
//!    //    Set-up model
//!    let config = MaskedLanguageConfig::new(
//!        ModelType::Bert,
//!        RemoteResource::from_pretrained(BertModelResources::BERT),
//!        RemoteResource::from_pretrained(BertConfigResources::BERT),
//!        RemoteResource::from_pretrained(BertVocabResources::BERT),
//!        None,
//!        true,
//!        None,
//!        None,
//!        None,
//!    );
//!
//!    let mask_language_model = MaskedLanguageModel::new(config)?;
//!    //    Define input
//!    let input = [
//!        "Looks like one [mask] is missing",
//!        "Paris is the [MASK] of France",
//!    ];
//!
//!    //    Run model
//!    let output = mask_language_model.predict(&input);
//!    for word in output {
//!        println!("{:?}", word);
//!    }
//!
//! # Ok(())
//! # }
//! ```
//!
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
use lazy_regex::regex_replace_all;
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
                "Masked Language is not implemented for {:?}!",
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

/// #Masked token for masked language model to predict
pub enum MaskTokenOption {
    /// Bert for Masked Token
    Bert(String),
    /// DeBERTa for Masked Token
    Deberta(String),
    /// DeBERTa V2 for Masked Token
    DebertaV2(String),
    /// Roberta for Masked Token
    Roberta(String),
    /// XLMRoberta for Masked Token
    XLMRoberta(String),
    /// FNet for Masked Token
    FNet(String),
}
impl MaskTokenOption {
    /// Instantiate a new masked token object of the supplied type.
    ///
    /// # Arguments
    ///
    /// * `model_type` - `ModelType` indicating the model type to load (must match with the actual data to be loaded)
    /// * `p` - `tch::nn::Path` path to the model file to load (e.g. model.ot)
    /// * `mask_token` - A configuration (the configuration must in accordance with the `model_type`)
    pub fn new(
        model_type: ModelType,
        mask_token: Option<String>,
    ) -> Result<MaskTokenOption, RustBertError> {
        match model_type {
            ModelType::Bert => {
                if mask_token.is_none() {
                    Ok(MaskTokenOption::Bert(String::from("[MASK]")))
                } else {
                    Ok(MaskTokenOption::Bert(mask_token.unwrap()))
                }
            }
            ModelType::Deberta => {
                if mask_token.is_none() {
                    Ok(MaskTokenOption::Deberta(String::from("[MASK]")))
                } else {
                    Ok(MaskTokenOption::Deberta(mask_token.unwrap()))
                }
            }
            ModelType::DebertaV2 => {
                if mask_token.is_none() {
                    Ok(MaskTokenOption::DebertaV2(String::from("[MASK]")))
                } else {
                    Ok(MaskTokenOption::DebertaV2(mask_token.unwrap()))
                }
            }
            ModelType::Roberta => {
                if mask_token.is_none() {
                    Ok(MaskTokenOption::Roberta(String::from("<MASK]>")))
                } else {
                    Ok(MaskTokenOption::Roberta(mask_token.unwrap()))
                }
            }
            ModelType::XLMRoberta => {
                if mask_token.is_none() {
                    Ok(MaskTokenOption::XLMRoberta(String::from("<MASK>")))
                } else {
                    Ok(MaskTokenOption::XLMRoberta(mask_token.unwrap()))
                }
            }
            ModelType::FNet => {
                if mask_token.is_none() {
                    Ok(MaskTokenOption::FNet(String::from("[MASK]")))
                } else {
                    Ok(MaskTokenOption::FNet(mask_token.unwrap()))
                }
            }
            _ => Err(RustBertError::InvalidConfigurationError(format!(
                "MaskTokenOption is not implemented for {:?}!",
                model_type
            ))),
        }
    }

    /// Return the `mask_token` for this MaskTokenOption
    pub fn get_mask_token(&self) -> &String {
        match *self {
            Self::Bert(ref mask_token) => mask_token,
            Self::Deberta(ref mask_token) => mask_token,
            Self::DebertaV2(ref mask_token) => mask_token,
            Self::Roberta(ref mask_token) => mask_token,
            Self::XLMRoberta(ref mask_token) => mask_token,
            Self::FNet(ref mask_token) => mask_token,
        }
    }
}
/// # MaskedLanguageModel for Masked Language (e.g. Fill Mask)
pub struct MaskedLanguageModel {
    tokenizer: TokenizerOption,
    language_encode: MaskedLanguageOption,
    mask_token: MaskTokenOption,
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
        let weights_path = config.model_resource.get_local_path()?;
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
        let mask_token = MaskTokenOption::new(config.model_type, config.mask_token)?;
        Ok(MaskedLanguageModel {
            tokenizer,
            language_encode,
            mask_token,
            var_store,
            max_length,
        })
    }

    // replace wrong mask_token in input text to correct mask_token
    fn replace_mask_token<'a, S>(&self, input: S) -> Vec<String>
    where
        S: AsRef<[&'a str]>,
    {
        let mask_token = self.mask_token.get_mask_token();
        let output = input
            .as_ref()
            .iter()
            .map(|&x| {
                regex_replace_all!(
                    r#"(<\w+?>|\[\w+?\])"#i,
                    x,
                    |_,_| format!("{}",mask_token),
                )
                .to_string()
            })
            .collect::<Vec<_>>();
        output
    }

    fn prepare_for_model(&self, input: &Vec<String>) -> Tensor {
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

    /// Return the word predicted by model
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
            TokenizerOption::XLMRoberta(ref tokenizer) => {
                MultiThreadedTokenizer::vocab(tokenizer).id_to_token(&input.int64_value(&[]))
            }
            TokenizerOption::FNet(ref tokenizer) => {
                MultiThreadedTokenizer::vocab(tokenizer).id_to_token(&input.int64_value(&[]))
            }
            _ => return Err(RustBertError::InvalidConfigurationError(
                "Masked Language currently supports Bert|Deberta|DebertaV2|Roberta|XLMRoberta|FNet!".to_string(),
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
    /// # Returns
    ///
    /// * `Vec<String>` containing masked words for input texts
    ///
    /// # Example
    ///
    /// ```no_run
    ///
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
    /// let output = mask_language_model.predict(&input);
    /// for word in output {
    ///     println!("{:?}", word);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn predict<'a, S>(&self, input: S) -> Vec<String>
    where
        S: AsRef<[&'a str]>,
    {
        let input_with_token = self.replace_mask_token(input);
        let input_tensor = self.prepare_for_model(&input_with_token);
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
        // get the position of mask_token in input texts
        let mask_token = self.mask_token.get_mask_token();
        let index = input_with_token
            .iter()
            .map(|x| {
                let pos = match x
                    .split(' ')
                    .collect::<Vec<_>>()
                    .iter()
                    .position(|&r| r == mask_token)
                {
                    Some(val) => val,
                    None => panic!("You should provide a mask in sentences! Such as  \"Looks like one [MASK] is missing.\""),
                };
                pos + 1
            } as i64)
            .collect::<Vec<_>>();
        // extract the predicted words
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
