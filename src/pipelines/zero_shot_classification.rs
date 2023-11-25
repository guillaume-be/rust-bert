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

//! # Zero-shot classification pipeline
//! Performs zero-shot classification on input sentences with provided labels using a model fine-tuned for Natural Language Inference.
//! The default model is a BART model fine-tuned on a MNLI. From a list of input sequences to classify and a list of target labels,
//! single-class or multi-label classification is performed, translating the classification task to an inference task.
//! The default template for translation to inference task is `This example is about {}.`. This template can be updated to a more specific
//! value that may match better the use case, for example `This review is about a {product_class}`.
//!
//! - `predict` performs single-class classification (one and exactly one label must be true for each provided input)
//! - `predict_multilabel` performs multi-label classification (zero, one or more labels may be true for each provided input)
//!
//! ```no_run
//! # use rust_bert::pipelines::zero_shot_classification::ZeroShotClassificationModel;
//! # fn main() -> anyhow::Result<()> {
//! let sequence_classification_model = ZeroShotClassificationModel::new(Default::default())?;
//!  let input_sentence = "Who are you voting for in 2020?";
//!  let input_sequence_2 = "The prime minister has announced a stimulus package which was widely criticized by the opposition.";
//!  let candidate_labels = &["politics", "public health", "economics", "sports"];
//!  let output = sequence_classification_model.predict_multilabel(
//!      &[input_sentence, input_sequence_2],
//!      candidate_labels,
//!      None,
//!      128,
//!  );
//! # Ok(())
//! # }
//! ```
//!
//! outputs:
//! ```no_run
//! # use rust_bert::pipelines::sequence_classification::Label;
//! let output = [
//!     [
//!         Label {
//!             text: "politics".to_string(),
//!             score: 0.972,
//!             id: 0,
//!             sentence: 0,
//!         },
//!         Label {
//!             text: "public health".to_string(),
//!             score: 0.032,
//!             id: 1,
//!             sentence: 0,
//!         },
//!         Label {
//!             text: "economy".to_string(),
//!             score: 0.006,
//!             id: 2,
//!             sentence: 0,
//!         },
//!         Label {
//!             text: "sports".to_string(),
//!             score: 0.004,
//!             id: 3,
//!             sentence: 0,
//!         },
//!     ],
//!     [
//!         Label {
//!             text: "politics".to_string(),
//!             score: 0.943,
//!             id: 0,
//!             sentence: 1,
//!         },
//!         Label {
//!             text: "economy".to_string(),
//!             score: 0.985,
//!             id: 2,
//!             sentence: 1,
//!         },
//!         Label {
//!             text: "public health".to_string(),
//!             score: 0.0818,
//!             id: 1,
//!             sentence: 1,
//!         },
//!         Label {
//!             text: "sports".to_string(),
//!             score: 0.001,
//!             id: 3,
//!             sentence: 1,
//!         },
//!     ],
//! ]
//! .to_vec();
//! ```

use crate::albert::AlbertForSequenceClassification;
use crate::bart::BartForSequenceClassification;
use crate::bert::BertForSequenceClassification;
use crate::deberta::DebertaForSequenceClassification;
use crate::deberta_v2::DebertaV2ForSequenceClassification;
use crate::distilbert::DistilBertModelClassifier;
use crate::longformer::LongformerForSequenceClassification;
use crate::mobilebert::MobileBertForSequenceClassification;
use crate::pipelines::common::{
    cast_var_store, ConfigOption, ModelResource, ModelType, TokenizerOption,
};
use crate::pipelines::sequence_classification::Label;
use crate::resources::ResourceProvider;
use crate::roberta::RobertaForSequenceClassification;
use crate::xlnet::XLNetForSequenceClassification;
use crate::RustBertError;
use rust_tokenizers::tokenizer::TruncationStrategy;
use rust_tokenizers::TokenizedInput;

#[cfg(feature = "onnx")]
use crate::pipelines::onnx::{config::ONNXEnvironmentConfig, ONNXEncoder};
#[cfg(feature = "remote")]
use crate::{
    bart::{BartConfigResources, BartMergesResources, BartModelResources, BartVocabResources},
    resources::RemoteResource,
};
use tch::kind::Kind::{Bool, Float};
use tch::nn::VarStore;
use tch::{no_grad, Device, Kind, Tensor};

/// # Configuration for ZeroShotClassificationModel
/// Contains information regarding the model to load and device to place the model on.
pub struct ZeroShotClassificationConfig {
    /// Model type
    pub model_type: ModelType,
    /// Model weights resource (default: pretrained BERT model on CoNLL)
    pub model_resource: ModelResource,
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
    /// Model weights precision. If not provided, will default to full precision on CPU, or the loaded weights precision otherwise
    pub kind: Option<Kind>,
}

impl ZeroShotClassificationConfig {
    /// Instantiate a new zero shot classification configuration of the supplied type.
    ///
    /// # Arguments
    ///
    /// * `model_type` - `ModelType` indicating the model type to load (must match with the actual data to be loaded!)
    /// * model - The `ResourceProvider` pointing to the model to load (e.g.  model.ot)
    /// * config - The `ResourceProvider` pointing to the model configuration to load (e.g. config.json)
    /// * vocab - The `ResourceProvider` pointing to the tokenizer's vocabulary to load (e.g.  vocab.txt/vocab.json)
    /// * merges - An optional `ResourceProvider` pointing to the tokenizer's merge file to load (e.g.  merges.txt), needed only for Roberta.
    /// * lower_case - A `bool` indicating whether the tokenizer should lower case all input (in case of a lower-cased model)
    pub fn new<RC, RV>(
        model_type: ModelType,
        model_resource: ModelResource,
        config_resource: RC,
        vocab_resource: RV,
        merges_resource: Option<RV>,
        lower_case: bool,
        strip_accents: impl Into<Option<bool>>,
        add_prefix_space: impl Into<Option<bool>>,
    ) -> ZeroShotClassificationConfig
    where
        RC: ResourceProvider + Send + 'static,
        RV: ResourceProvider + Send + 'static,
    {
        ZeroShotClassificationConfig {
            model_type,
            model_resource,
            config_resource: Box::new(config_resource),
            vocab_resource: Box::new(vocab_resource),
            merges_resource: merges_resource.map(|r| Box::new(r) as Box<_>),
            lower_case,
            strip_accents: strip_accents.into(),
            add_prefix_space: add_prefix_space.into(),
            device: Device::cuda_if_available(),
            kind: None,
        }
    }
}

#[cfg(feature = "remote")]
impl Default for ZeroShotClassificationConfig {
    /// Provides a default zero-shot classification model (English)
    fn default() -> ZeroShotClassificationConfig {
        ZeroShotClassificationConfig {
            model_type: ModelType::Bart,
            model_resource: ModelResource::Torch(Box::new(RemoteResource::from_pretrained(
                BartModelResources::BART_MNLI,
            ))),
            config_resource: Box::new(RemoteResource::from_pretrained(
                BartConfigResources::BART_MNLI,
            )),
            vocab_resource: Box::new(RemoteResource::from_pretrained(
                BartVocabResources::BART_MNLI,
            )),
            merges_resource: Some(Box::new(RemoteResource::from_pretrained(
                BartMergesResources::BART_MNLI,
            ))),
            lower_case: false,
            strip_accents: None,
            add_prefix_space: None,
            device: Device::cuda_if_available(),
            kind: None,
        }
    }
}

/// # Abstraction that holds one particular zero shot classification model, for any of the supported models
/// The models are using a classification architecture that should be trained on Natural Language Inference.
/// The models should output a Tensor of size > 2 in the label dimension, with the first logit corresponding
/// to contradiction and the last logit corresponding to entailment.
#[allow(clippy::large_enum_variant)]
pub enum ZeroShotClassificationOption {
    /// Bart for Sequence Classification
    Bart(BartForSequenceClassification),
    /// DeBERTa for Sequence Classification
    Deberta(DebertaForSequenceClassification),
    /// DeBERTaV2 for Sequence Classification
    DebertaV2(DebertaV2ForSequenceClassification),
    /// Bert for Sequence Classification
    Bert(BertForSequenceClassification),
    /// DistilBert for Sequence Classification
    DistilBert(DistilBertModelClassifier),
    /// MobileBert for Sequence Classification
    MobileBert(MobileBertForSequenceClassification),
    /// Roberta for Sequence Classification
    Roberta(RobertaForSequenceClassification),
    /// XLMRoberta for Sequence Classification
    XLMRoberta(RobertaForSequenceClassification),
    /// Albert for Sequence Classification
    Albert(AlbertForSequenceClassification),
    /// XLNet for Sequence Classification
    XLNet(XLNetForSequenceClassification),
    /// Longformer for Sequence Classification
    Longformer(LongformerForSequenceClassification),
    /// ONNX model for Sequence Classification
    #[cfg(feature = "onnx")]
    ONNX(ONNXEncoder),
}

impl ZeroShotClassificationOption {
    /// Instantiate a new zer-shot classification model of the supplied type.
    ///
    /// # Arguments
    ///
    /// * `ZeroShotClassificationConfig` - Zero-shot classification pipeline configuration. The type of model created will be inferred from the
    ///     `ModelResources` (Torch or ONNX) and `ModelType` (Architecture for Torch models) variants provided and
    pub fn new(config: &ZeroShotClassificationConfig) -> Result<Self, RustBertError> {
        match config.model_resource {
            ModelResource::Torch(_) => Self::new_torch(config),
            #[cfg(feature = "onnx")]
            ModelResource::ONNX(_) => Self::new_onnx(config),
        }
    }

    fn new_torch(config: &ZeroShotClassificationConfig) -> Result<Self, RustBertError> {
        let device = config.device;
        let weights_path = config.model_resource.get_torch_local_path()?;
        let mut var_store = VarStore::new(device);
        let model_config =
            &ConfigOption::from_file(config.model_type, config.config_resource.get_local_path()?);
        let model_type = config.model_type;
        let model = match model_type {
            ModelType::Bart => {
                if let ConfigOption::Bart(config) = model_config {
                    Ok(Self::Bart(
                        BartForSequenceClassification::new(var_store.root(), config)?,
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a BartConfig for Bart!".to_string(),
                    ))
                }
            }
            ModelType::Deberta => {
                if let ConfigOption::Deberta(config) = model_config {
                    Ok(Self::Deberta(
                        DebertaForSequenceClassification::new(var_store.root(), config)?,
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a DebertaConfig for DeBERTa!".to_string(),
                    ))
                }
            }
            ModelType::DebertaV2 => {
                if let ConfigOption::DebertaV2(config) = model_config {
                    Ok(Self::DebertaV2(
                        DebertaV2ForSequenceClassification::new(var_store.root(), config)?,
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a DebertaConfig for DeBERTaV2!".to_string(),
                    ))
                }
            }
            ModelType::Bert => {
                if let ConfigOption::Bert(config) = model_config {
                    Ok(Self::Bert(
                        BertForSequenceClassification::new(var_store.root(), config)?,
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a BertConfig for Bert!".to_string(),
                    ))
                }
            }
            ModelType::DistilBert => {
                if let ConfigOption::DistilBert(config) = model_config {
                    Ok(Self::DistilBert(
                        DistilBertModelClassifier::new(var_store.root(), config)?,
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a DistilBertConfig for DistilBert!".to_string(),
                    ))
                }
            }
            ModelType::MobileBert => {
                if let ConfigOption::MobileBert(config) = model_config {
                    Ok(Self::MobileBert(
                        MobileBertForSequenceClassification::new(var_store.root(), config)?,
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a MobileBertConfig for MobileBert!".to_string(),
                    ))
                }
            }
            ModelType::Roberta => {
                if let ConfigOption::Roberta(config) = model_config {
                    Ok(Self::Roberta(
                        RobertaForSequenceClassification::new(var_store.root(), config)?,
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a RobertaConfig for Roberta!".to_string(),
                    ))
                }
            }
            ModelType::XLMRoberta => {
                if let ConfigOption::Bert(config) = model_config {
                    Ok(Self::XLMRoberta(
                        RobertaForSequenceClassification::new(var_store.root(), config)?,
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a BertConfig for Roberta!".to_string(),
                    ))
                }
            }
            ModelType::Albert => {
                if let ConfigOption::Albert(config) = model_config {
                    Ok(Self::Albert(
                        AlbertForSequenceClassification::new(var_store.root(), config)?,
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply an AlbertConfig for Albert!".to_string(),
                    ))
                }
            }
            ModelType::XLNet => {
                if let ConfigOption::XLNet(config) = model_config {
                    Ok(Self::XLNet(
                        XLNetForSequenceClassification::new(var_store.root(), config)?,
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply an AlbertConfig for Albert!".to_string(),
                    ))
                }
            }
            ModelType::Longformer => {
                if let ConfigOption::Longformer(config) = model_config {
                    Ok(Self::Longformer(
                        LongformerForSequenceClassification::new(var_store.root(), config)?,
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a LongformerConfig for Longformer!".to_string(),
                    ))
                }
            }
            #[cfg(feature = "onnx")]
            ModelType::ONNX => Err(RustBertError::InvalidConfigurationError(
                "A `ModelType::ONNX` ModelType was provided in the configuration with `ModelResources::TORCH`, these are incompatible".to_string(),
            )),
            _ => Err(RustBertError::InvalidConfigurationError(format!(
                "Zero shot classification not implemented for {model_type:?}!",
            ))),
        }?;
        var_store.load(weights_path)?;
        cast_var_store(&mut var_store, config.kind, device);
        Ok(model)
    }

    #[cfg(feature = "onnx")]
    pub fn new_onnx(config: &ZeroShotClassificationConfig) -> Result<Self, RustBertError> {
        let onnx_config = ONNXEnvironmentConfig::from_device(config.device);
        let environment = onnx_config.get_environment()?;
        let encoder_file = config
            .model_resource
            .get_onnx_local_paths()?
            .encoder_path
            .ok_or(RustBertError::InvalidConfigurationError(
                "An encoder file must be provided for zero-shot classification ONNX models."
                    .to_string(),
            ))?;

        Ok(Self::ONNX(ONNXEncoder::new(
            encoder_file,
            &environment,
            &onnx_config,
        )?))
    }

    /// Returns the `ModelType` for this SequenceClassificationOption
    pub fn model_type(&self) -> ModelType {
        match *self {
            Self::Bart(_) => ModelType::Bart,
            Self::Deberta(_) => ModelType::Deberta,
            Self::DebertaV2(_) => ModelType::DebertaV2,
            Self::Bert(_) => ModelType::Bert,
            Self::Roberta(_) => ModelType::Roberta,
            Self::XLMRoberta(_) => ModelType::Roberta,
            Self::DistilBert(_) => ModelType::DistilBert,
            Self::MobileBert(_) => ModelType::MobileBert,
            Self::Albert(_) => ModelType::Albert,
            Self::XLNet(_) => ModelType::XLNet,
            Self::Longformer(_) => ModelType::Longformer,
            #[cfg(feature = "onnx")]
            Self::ONNX(_) => ModelType::ONNX,
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
        train: bool,
    ) -> Tensor {
        match *self {
            Self::Bart(ref model) => {
                model
                    .forward_t(
                        input_ids.expect("`input_ids` must be provided for BART models"),
                        mask,
                        None,
                        None,
                        None,
                        train,
                    )
                    .decoder_output
            }
            Self::Bert(ref model) => {
                model
                    .forward_t(
                        input_ids,
                        mask,
                        token_type_ids,
                        position_ids,
                        input_embeds,
                        train,
                    )
                    .logits
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
                    .expect("Error in DeBERTa forward_t")
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
                    .expect("Error in DeBERTaV2 forward_t")
                    .logits
            }
            Self::DistilBert(ref model) => {
                model
                    .forward_t(input_ids, mask, input_embeds, train)
                    .expect("Error in distilbert forward_t")
                    .logits
            }
            Self::MobileBert(ref model) => {
                model
                    .forward_t(input_ids, None, None, input_embeds, mask, train)
                    .expect("Error in mobilebert forward_t")
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
                        train,
                    )
                    .logits
            }
            Self::Albert(ref model) => {
                model
                    .forward_t(
                        input_ids,
                        mask,
                        token_type_ids,
                        position_ids,
                        input_embeds,
                        train,
                    )
                    .logits
            }
            Self::XLNet(ref model) => {
                model
                    .forward_t(
                        input_ids,
                        mask,
                        None,
                        None,
                        None,
                        token_type_ids,
                        input_embeds,
                        train,
                    )
                    .logits
            }
            Self::Longformer(ref model) => {
                model
                    .forward_t(
                        input_ids,
                        mask,
                        None,
                        token_type_ids,
                        position_ids,
                        input_embeds,
                        train,
                    )
                    .expect("Error in Longformer forward pass.")
                    .logits
            }
            #[cfg(feature = "onnx")]
            Self::ONNX(ref model) => model
                .forward(
                    input_ids,
                    mask.map(|tensor| tensor.to_kind(Kind::Int64)).as_ref(),
                    token_type_ids,
                    position_ids,
                    input_embeds,
                )
                .expect("Error in ONNX forward pass.")
                .logits
                .unwrap(),
        }
    }
}

pub type ZeroShotTemplate = Box<dyn Fn(&str) -> String>;
/// Template used to transform the zero-shot classification labels into a set of
/// natural language hypotheses for natural language inference.
///
/// For example, transform `[positive, negative]` into
/// `[This is a positive review, This is a negative review]`
///
/// The function should take a `&str` as an input and return the formatted String.
///
/// This transformation has a strong impact on the resulting classification accuracy.
/// If no function is provided for zero-shot classification, the default templating
/// function will be used:
///
/// ```rust
/// fn default_template(label: &str) -> String {
///     format!("This example is about {}.", label)
/// }
/// ```

/// # ZeroShotClassificationModel for Zero Shot Classification
pub struct ZeroShotClassificationModel {
    tokenizer: TokenizerOption,
    zero_shot_classifier: ZeroShotClassificationOption,
    device: Device,
}

impl ZeroShotClassificationModel {
    /// Build a new `ZeroShotClassificationModel`
    ///
    /// # Arguments
    ///
    /// * `config` - `SequenceClassificationConfig` object containing the resource references (model, vocabulary, configuration) and device placement (CPU/GPU)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> anyhow::Result<()> {
    /// use rust_bert::pipelines::sequence_classification::SequenceClassificationModel;
    ///
    /// let model = SequenceClassificationModel::new(Default::default())?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(
        config: ZeroShotClassificationConfig,
    ) -> Result<ZeroShotClassificationModel, RustBertError> {
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

    /// Build a new `ZeroShotClassificationModel` with a provided tokenizer.
    ///
    /// # Arguments
    ///
    /// * `config` - `SequenceClassificationConfig` object containing the resource references (model, vocabulary, configuration) and device placement (CPU/GPU)
    /// * `tokenizer` - `TokenizerOption` tokenizer to use for zero-shot classification.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> anyhow::Result<()> {
    /// use rust_bert::pipelines::common::{ModelType, TokenizerOption};
    /// use rust_bert::pipelines::sequence_classification::SequenceClassificationModel;
    /// let tokenizer = TokenizerOption::from_file(
    ///     ModelType::Bert,
    ///     "path/to/vocab.txt",
    ///     None,
    ///     false,
    ///     None,
    ///     None,
    /// )?;
    /// let model = SequenceClassificationModel::new_with_tokenizer(Default::default(), tokenizer)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new_with_tokenizer(
        config: ZeroShotClassificationConfig,
        tokenizer: TokenizerOption,
    ) -> Result<ZeroShotClassificationModel, RustBertError> {
        let device = config.device;
        let zero_shot_classifier = ZeroShotClassificationOption::new(&config)?;

        Ok(ZeroShotClassificationModel {
            tokenizer,
            zero_shot_classifier,
            device,
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

    fn prepare_for_model<'a, S, T>(
        &self,
        inputs: S,
        labels: T,
        template: Option<ZeroShotTemplate>,
        max_len: usize,
    ) -> Result<(Tensor, Tensor, Tensor), RustBertError>
    where
        S: AsRef<[&'a str]>,
        T: AsRef<[&'a str]>,
    {
        let label_sentences: Vec<String> = match template {
            Some(function) => labels
                .as_ref()
                .iter()
                .map(|label| function(label))
                .collect(),
            None => labels
                .as_ref()
                .iter()
                .map(|label| format!("This example is about {label}."))
                .collect(),
        };

        let text_pair_list = inputs
            .as_ref()
            .iter()
            .flat_map(|input| {
                label_sentences
                    .iter()
                    .map(move |label_sentence| (*input, label_sentence.as_str()))
            })
            .collect::<Vec<(&str, &str)>>();

        let mut tokenized_input: Vec<TokenizedInput> = self.tokenizer.encode_pair_list(
            text_pair_list.as_ref(),
            max_len,
            &TruncationStrategy::LongestFirst,
            0,
        );
        let max_len = tokenized_input
            .iter()
            .map(|input| input.token_ids.len())
            .max()
            .ok_or_else(|| RustBertError::ValueError("Got empty iterator as input".to_string()))?;

        let pad_id = self
            .tokenizer
            .get_pad_id()
            .expect("The Tokenizer used for sequence classification should contain a PAD id");
        let input_ids = tokenized_input
            .iter_mut()
            .map(|input| {
                input.token_ids.resize(max_len, pad_id);
                Tensor::from_slice(&(input.token_ids))
            })
            .collect::<Vec<_>>();
        let token_type_ids = tokenized_input
            .iter_mut()
            .map(|input| {
                input
                    .segment_ids
                    .resize(max_len, *input.segment_ids.last().unwrap_or(&0));
                Tensor::from_slice(&(input.segment_ids))
            })
            .collect::<Vec<_>>();

        let input_ids = Tensor::stack(input_ids.as_slice(), 0).to(self.device);
        let token_type_ids = Tensor::stack(token_type_ids.as_slice(), 0)
            .to(self.device)
            .to_kind(Kind::Int64);
        let mask = input_ids
            .ne(self
                .tokenizer
                .get_pad_id()
                .expect("The Tokenizer used for zero shot classification should contain a PAD id"))
            .to_kind(Bool);

        Ok((input_ids, mask, token_type_ids))
    }

    /// Zero shot classification with 1 (and exactly 1) true label.
    ///
    /// # Arguments
    ///
    /// * `input` - `&[&str]` Array of texts to classify.
    /// * `labels` - `&[&str]` Possible labels for the inputs.
    /// * `template` - `Option<Box<dyn Fn(&str) -> String>>` closure to build label propositions. If None, will default to `"This example is {}."`.
    /// * `max_length` -`usize` Maximum sequence length for the inputs. If needed, the input sequence will be truncated before the label template.
    ///
    /// # Returns
    ///
    /// * `Result<Vec<Label>, RustBertError>` containing the most likely label for each input sentence or error, if any.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> anyhow::Result<()> {
    /// use rust_bert::pipelines::zero_shot_classification::ZeroShotClassificationModel;
    ///
    /// let sequence_classification_model = ZeroShotClassificationModel::new(Default::default())?;
    ///
    /// let input_sentence = "Who are you voting for in 2020?";
    /// let input_sequence_2 = "The prime minister has announced a stimulus package which was widely criticized by the opposition.";
    /// let candidate_labels = &["politics", "public health", "economics", "sports"];
    ///
    /// let output = sequence_classification_model.predict(
    ///     &[input_sentence, input_sequence_2],
    ///     candidate_labels,
    ///     None,
    ///     128,
    /// );
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// outputs:
    /// ```no_run
    /// # use rust_bert::pipelines::sequence_classification::Label;
    /// let output = [
    ///     Label {
    ///         text: "politics".to_string(),
    ///         score: 0.959,
    ///         id: 0,
    ///         sentence: 0,
    ///     },
    ///     Label {
    ///         text: "economy".to_string(),
    ///         score: 0.642,
    ///         id: 2,
    ///         sentence: 1,
    ///     },
    /// ]
    /// .to_vec();
    /// ```
    pub fn predict<'a, S, T>(
        &self,
        inputs: S,
        labels: T,
        template: Option<ZeroShotTemplate>,
        max_length: usize,
    ) -> Result<Vec<Label>, RustBertError>
    where
        S: AsRef<[&'a str]>,
        T: AsRef<[&'a str]>,
    {
        let num_inputs = inputs.as_ref().len();
        let (input_tensor, mask, token_type_ids) =
            self.prepare_for_model(inputs.as_ref(), labels.as_ref(), template, max_length)?;

        let output = no_grad(|| {
            let output = self.zero_shot_classifier.forward_t(
                Some(&input_tensor),
                Some(&mask),
                Some(&token_type_ids),
                None,
                None,
                false,
            );
            output.view((num_inputs as i64, labels.as_ref().len() as i64, -1i64))
        });

        let scores = output.softmax(1, Float).select(-1, -1);
        let label_indices = scores.as_ref().argmax(-1, true).squeeze_dim(1);
        let scores = scores
            .gather(1, &label_indices.unsqueeze(-1), false)
            .squeeze_dim(1);
        let label_indices = label_indices.iter::<i64>()?.collect::<Vec<i64>>();
        let scores = scores.iter::<f64>()?.collect::<Vec<f64>>();

        let mut output_labels: Vec<Label> = vec![];
        for sentence_idx in 0..label_indices.len() {
            let label_string = labels.as_ref()[label_indices[sentence_idx] as usize].to_string();
            let label = Label {
                text: label_string,
                score: scores[sentence_idx],
                id: label_indices[sentence_idx],
                sentence: sentence_idx,
            };
            output_labels.push(label)
        }
        Ok(output_labels)
    }

    /// Zero shot multi-label classification with 0, 1 or no true label.
    ///
    /// # Arguments
    ///
    /// * `input` - `&[&str]` Array of texts to classify.
    /// * `labels` - `&[&str]` Possible labels for the inputs.
    /// * `template` - `Option<Box<dyn Fn(&str) -> String>>` closure to build label propositions. If None, will default to `"This example is about {}."`.
    /// * `max_length` -`usize` Maximum sequence length for the inputs. If needed, the input sequence will be truncated before the label template.
    ///
    /// # Returns
    ///
    /// * `Result<Vec<Vec<Label>>, RustBertError>` containing a vector of labels and their probability for each input text, or error, if any.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> anyhow::Result<()> {
    /// use rust_bert::pipelines::zero_shot_classification::ZeroShotClassificationModel;
    ///
    /// let sequence_classification_model = ZeroShotClassificationModel::new(Default::default())?;
    ///
    /// let input_sentence = "Who are you voting for in 2020?";
    /// let input_sequence_2 = "The central bank is meeting today to discuss monetary policy.";
    /// let candidate_labels = &["politics", "public health", "economics", "sports"];
    ///
    /// let output = sequence_classification_model.predict_multilabel(
    ///     &[input_sentence, input_sequence_2],
    ///     candidate_labels,
    ///     None,
    ///     128,
    /// );
    /// # Ok(())
    /// # }
    /// ```
    /// outputs:
    /// ```no_run
    /// # use rust_bert::pipelines::sequence_classification::Label;
    /// let output = [
    ///     [
    ///         Label {
    ///             text: "politics".to_string(),
    ///             score: 0.972,
    ///             id: 0,
    ///             sentence: 0,
    ///         },
    ///         Label {
    ///             text: "public health".to_string(),
    ///             score: 0.032,
    ///             id: 1,
    ///             sentence: 0,
    ///         },
    ///         Label {
    ///             text: "economy".to_string(),
    ///             score: 0.006,
    ///             id: 2,
    ///             sentence: 0,
    ///         },
    ///         Label {
    ///             text: "sports".to_string(),
    ///             score: 0.004,
    ///             id: 3,
    ///             sentence: 0,
    ///         },
    ///     ],
    ///     [
    ///         Label {
    ///             text: "politics".to_string(),
    ///             score: 0.975,
    ///             id: 0,
    ///             sentence: 1,
    ///         },
    ///         Label {
    ///             text: "economy".to_string(),
    ///             score: 0.852,
    ///             id: 2,
    ///             sentence: 1,
    ///         },
    ///         Label {
    ///             text: "public health".to_string(),
    ///             score: 0.0818,
    ///             id: 1,
    ///             sentence: 1,
    ///         },
    ///         Label {
    ///             text: "sports".to_string(),
    ///             score: 0.001,
    ///             id: 3,
    ///             sentence: 1,
    ///         },
    ///     ],
    /// ]
    /// .to_vec();
    /// ```
    pub fn predict_multilabel<'a, S, T>(
        &self,
        inputs: S,
        labels: T,
        template: Option<ZeroShotTemplate>,
        max_length: usize,
    ) -> Result<Vec<Vec<Label>>, RustBertError>
    where
        S: AsRef<[&'a str]>,
        T: AsRef<[&'a str]>,
    {
        let num_inputs = inputs.as_ref().len();
        let (input_tensor, mask, token_type_ids) =
            self.prepare_for_model(inputs.as_ref(), labels.as_ref(), template, max_length)?;

        let output = no_grad(|| {
            let output = self.zero_shot_classifier.forward_t(
                Some(&input_tensor),
                Some(&mask),
                Some(&token_type_ids),
                None,
                None,
                false,
            );
            output.view((num_inputs as i64, labels.as_ref().len() as i64, -1i64))
        });
        let scores = output.slice(-1, 0, 3, 2).softmax(-1, Float).select(-1, -1);

        let mut output_labels = vec![];
        for sentence_idx in 0..num_inputs {
            let mut sentence_labels = vec![];

            for (label_index, score) in scores
                .select(0, sentence_idx as i64)
                .iter::<f64>()?
                .enumerate()
            {
                let label_string = labels.as_ref()[label_index].to_string();
                let label = Label {
                    text: label_string,
                    score,
                    id: label_index as i64,
                    sentence: sentence_idx,
                };
                sentence_labels.push(label);
            }
            output_labels.push(sentence_labels);
        }
        Ok(output_labels)
    }
}
#[cfg(test)]
mod test {
    use super::*;

    #[test]
    #[ignore] // no need to run, compilation is enough to verify it is Send
    fn test() {
        let config = ZeroShotClassificationConfig::default();
        let _: Box<dyn Send> = Box::new(ZeroShotClassificationModel::new(config));
    }
}
