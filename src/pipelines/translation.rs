// Copyright 2018-2020 The HuggingFace Inc. team.
// Copyright 2020 Marian Team Authors
// Copyright 2019-2020 Guillaume Becquin
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! # Translation pipeline
//! Translation based on the Marian encoder-decoder architecture
//! Include techniques such as beam search, top-k and nucleus sampling, temperature setting and repetition penalty.
//! Pre-trained and ready-to-use models are available by creating a configuration from the `Language` enum.
//! These models have been trained by the [Opus-MT team from Language Technology at the University of Helsinki](https://github.com/Helsinki-NLP/Opus-MT).
//! The Rust model files are hosted by [Hugging Face Inc](https://huggingface.co).
//! Currently supported languages are :
//! - English <-> French
//! - English <-> Spanish
//! - English <-> Portuguese
//! - English <-> Italian
//! - English <-> Catalan
//! - English <-> German
//! - English <-> Russian
//! - French <-> German
//!
//! Customized Translation models can be loaded by creating a configuration from local files.
//! The dependencies will be downloaded to the user's home directory, under ~/.cache/.rustbert/{translation-model-name}
//!
//!
//! ```no_run
//! # fn main() -> anyhow::Result<()> {
//! # use rust_bert::pipelines::generation_utils::LanguageGenerator;
//! use rust_bert::pipelines::translation::{Language, TranslationConfig, TranslationModel};
//! use tch::Device;
//! let translation_config =
//!     TranslationConfig::new(Language::EnglishToFrench, Device::cuda_if_available());
//! let mut model = TranslationModel::new(translation_config)?;
//!
//! let input = ["This is a sentence to be translated"];
//!
//! let output = model.translate(&input);
//! # Ok(())
//! # }
//! ```
//!
//! Output: \
//! ```no_run
//! # let output =
//! "Il s'agit d'une phrase à traduire"
//! # ;
//! ```

use tch::{Device, Tensor};

use crate::common::error::RustBertError;
use crate::common::resources::{RemoteResource, Resource};
use crate::marian::{
    MarianConfigResources, MarianGenerator, MarianModelResources, MarianPrefix, MarianSpmResources,
    MarianVocabResources,
};
use crate::pipelines::common::ModelType;
use crate::pipelines::generation_utils::{GenerateConfig, LanguageGenerator};
use crate::t5::{T5ConfigResources, T5Generator, T5ModelResources, T5Prefix, T5VocabResources};

/// Pretrained languages available for direct use
pub enum Language {
    FrenchToEnglish,
    CatalanToEnglish,
    SpanishToEnglish,
    PortugueseToEnglish,
    ItalianToEnglish,
    RomanianToEnglish,
    GermanToEnglish,
    RussianToEnglish,
    DutchToEnglish,
    EnglishToFrench,
    EnglishToCatalan,
    EnglishToSpanish,
    EnglishToPortuguese,
    EnglishToItalian,
    EnglishToRomanian,
    EnglishToGerman,
    EnglishToRussian,
    EnglishToDutch,
    EnglishToFrenchV2,
    EnglishToGermanV2,
    FrenchToGerman,
    GermanToFrench,
}

struct RemoteTranslationResources {
    model_resource: (&'static str, &'static str),
    config_resource: (&'static str, &'static str),
    vocab_resource: (&'static str, &'static str),
    merges_resource: (&'static str, &'static str),
    prefix: Option<&'static str>,
    model_type: ModelType,
}

impl RemoteTranslationResources {
    pub const ENGLISH2FRENCH: RemoteTranslationResources = Self {
        model_resource: MarianModelResources::ENGLISH2ROMANCE,
        config_resource: MarianConfigResources::ENGLISH2ROMANCE,
        vocab_resource: MarianVocabResources::ENGLISH2ROMANCE,
        merges_resource: MarianSpmResources::ENGLISH2ROMANCE,
        prefix: MarianPrefix::ENGLISH2FRENCH,
        model_type: ModelType::Marian,
    };
    pub const ENGLISH2FRENCH_V2: RemoteTranslationResources = Self {
        model_resource: T5ModelResources::T5_BASE,
        config_resource: T5ConfigResources::T5_BASE,
        vocab_resource: T5VocabResources::T5_BASE,
        merges_resource: T5VocabResources::T5_BASE,
        prefix: T5Prefix::ENGLISH2FRENCH,
        model_type: ModelType::T5,
    };
    pub const ENGLISH2GERMAN_V2: RemoteTranslationResources = Self {
        model_resource: T5ModelResources::T5_BASE,
        config_resource: T5ConfigResources::T5_BASE,
        vocab_resource: T5VocabResources::T5_BASE,
        merges_resource: T5VocabResources::T5_BASE,
        prefix: T5Prefix::ENGLISH2GERMAN,
        model_type: ModelType::T5,
    };
    pub const ENGLISH2CATALAN: RemoteTranslationResources = Self {
        model_resource: MarianModelResources::ENGLISH2ROMANCE,
        config_resource: MarianConfigResources::ENGLISH2ROMANCE,
        vocab_resource: MarianVocabResources::ENGLISH2ROMANCE,
        merges_resource: MarianSpmResources::ENGLISH2ROMANCE,
        prefix: MarianPrefix::ENGLISH2CATALAN,
        model_type: ModelType::Marian,
    };
    pub const ENGLISH2SPANISH: RemoteTranslationResources = Self {
        model_resource: MarianModelResources::ENGLISH2ROMANCE,
        config_resource: MarianConfigResources::ENGLISH2ROMANCE,
        vocab_resource: MarianVocabResources::ENGLISH2ROMANCE,
        merges_resource: MarianSpmResources::ENGLISH2ROMANCE,
        prefix: MarianPrefix::ENGLISH2SPANISH,
        model_type: ModelType::Marian,
    };
    pub const ENGLISH2PORTUGUESE: RemoteTranslationResources = Self {
        model_resource: MarianModelResources::ENGLISH2ROMANCE,
        config_resource: MarianConfigResources::ENGLISH2ROMANCE,
        vocab_resource: MarianVocabResources::ENGLISH2ROMANCE,
        merges_resource: MarianSpmResources::ENGLISH2ROMANCE,
        prefix: MarianPrefix::ENGLISH2PORTUGUESE,
        model_type: ModelType::Marian,
    };
    pub const ENGLISH2ITALIAN: RemoteTranslationResources = Self {
        model_resource: MarianModelResources::ENGLISH2ROMANCE,
        config_resource: MarianConfigResources::ENGLISH2ROMANCE,
        vocab_resource: MarianVocabResources::ENGLISH2ROMANCE,
        merges_resource: MarianSpmResources::ENGLISH2ROMANCE,
        prefix: MarianPrefix::ENGLISH2ITALIAN,
        model_type: ModelType::Marian,
    };
    pub const ENGLISH2ROMANIAN: RemoteTranslationResources = Self {
        model_resource: MarianModelResources::ENGLISH2ROMANCE,
        config_resource: MarianConfigResources::ENGLISH2ROMANCE,
        vocab_resource: MarianVocabResources::ENGLISH2ROMANCE,
        merges_resource: MarianSpmResources::ENGLISH2ROMANCE,
        prefix: MarianPrefix::ENGLISH2ROMANIAN,
        model_type: ModelType::Marian,
    };
    pub const ENGLISH2GERMAN: RemoteTranslationResources = Self {
        model_resource: MarianModelResources::ENGLISH2GERMAN,
        config_resource: MarianConfigResources::ENGLISH2GERMAN,
        vocab_resource: MarianVocabResources::ENGLISH2GERMAN,
        merges_resource: MarianSpmResources::ENGLISH2GERMAN,
        prefix: MarianPrefix::ENGLISH2GERMAN,
        model_type: ModelType::Marian,
    };
    pub const ENGLISH2RUSSIAN: RemoteTranslationResources = Self {
        model_resource: MarianModelResources::ENGLISH2RUSSIAN,
        config_resource: MarianConfigResources::ENGLISH2RUSSIAN,
        vocab_resource: MarianVocabResources::ENGLISH2RUSSIAN,
        merges_resource: MarianSpmResources::ENGLISH2RUSSIAN,
        prefix: MarianPrefix::ENGLISH2RUSSIAN,
        model_type: ModelType::Marian,
    };
    pub const FRENCH2ENGLISH: RemoteTranslationResources = Self {
        model_resource: MarianModelResources::ROMANCE2ENGLISH,
        config_resource: MarianConfigResources::ROMANCE2ENGLISH,
        vocab_resource: MarianVocabResources::ROMANCE2ENGLISH,
        merges_resource: MarianSpmResources::ROMANCE2ENGLISH,
        prefix: MarianPrefix::FRENCH2ENGLISH,
        model_type: ModelType::Marian,
    };
    pub const CATALAN2ENGLISH: RemoteTranslationResources = Self {
        model_resource: MarianModelResources::ROMANCE2ENGLISH,
        config_resource: MarianConfigResources::ROMANCE2ENGLISH,
        vocab_resource: MarianVocabResources::ROMANCE2ENGLISH,
        merges_resource: MarianSpmResources::ROMANCE2ENGLISH,
        prefix: MarianPrefix::CATALAN2ENGLISH,
        model_type: ModelType::Marian,
    };
    pub const SPANISH2ENGLISH: RemoteTranslationResources = Self {
        model_resource: MarianModelResources::ROMANCE2ENGLISH,
        config_resource: MarianConfigResources::ROMANCE2ENGLISH,
        vocab_resource: MarianVocabResources::ROMANCE2ENGLISH,
        merges_resource: MarianSpmResources::ROMANCE2ENGLISH,
        prefix: MarianPrefix::SPANISH2ENGLISH,
        model_type: ModelType::Marian,
    };
    pub const PORTUGUESE2ENGLISH: RemoteTranslationResources = Self {
        model_resource: MarianModelResources::ROMANCE2ENGLISH,
        config_resource: MarianConfigResources::ROMANCE2ENGLISH,
        vocab_resource: MarianVocabResources::ROMANCE2ENGLISH,
        merges_resource: MarianSpmResources::ROMANCE2ENGLISH,
        prefix: MarianPrefix::PORTUGUESE2ENGLISH,
        model_type: ModelType::Marian,
    };
    pub const ITALIAN2ENGLISH: RemoteTranslationResources = Self {
        model_resource: MarianModelResources::ROMANCE2ENGLISH,
        config_resource: MarianConfigResources::ROMANCE2ENGLISH,
        vocab_resource: MarianVocabResources::ROMANCE2ENGLISH,
        merges_resource: MarianSpmResources::ROMANCE2ENGLISH,
        prefix: MarianPrefix::ITALIAN2ENGLISH,
        model_type: ModelType::Marian,
    };
    pub const ROMANIAN2ENGLISH: RemoteTranslationResources = Self {
        model_resource: MarianModelResources::ROMANCE2ENGLISH,
        config_resource: MarianConfigResources::ROMANCE2ENGLISH,
        vocab_resource: MarianVocabResources::ROMANCE2ENGLISH,
        merges_resource: MarianSpmResources::ROMANCE2ENGLISH,
        prefix: MarianPrefix::ROMANIAN2ENGLISH,
        model_type: ModelType::Marian,
    };
    pub const GERMAN2ENGLISH: RemoteTranslationResources = Self {
        model_resource: MarianModelResources::GERMAN2ENGLISH,
        config_resource: MarianConfigResources::GERMAN2ENGLISH,
        vocab_resource: MarianVocabResources::GERMAN2ENGLISH,
        merges_resource: MarianSpmResources::GERMAN2ENGLISH,
        prefix: MarianPrefix::GERMAN2ENGLISH,
        model_type: ModelType::Marian,
    };
    pub const RUSSIAN2ENGLISH: RemoteTranslationResources = Self {
        model_resource: MarianModelResources::RUSSIAN2ENGLISH,
        config_resource: MarianConfigResources::RUSSIAN2ENGLISH,
        vocab_resource: MarianVocabResources::RUSSIAN2ENGLISH,
        merges_resource: MarianSpmResources::RUSSIAN2ENGLISH,
        prefix: MarianPrefix::RUSSIAN2ENGLISH,
        model_type: ModelType::Marian,
    };
    pub const FRENCH2GERMAN: RemoteTranslationResources = Self {
        model_resource: MarianModelResources::FRENCH2GERMAN,
        config_resource: MarianConfigResources::FRENCH2GERMAN,
        vocab_resource: MarianVocabResources::FRENCH2GERMAN,
        merges_resource: MarianSpmResources::FRENCH2GERMAN,
        prefix: MarianPrefix::FRENCH2GERMAN,
        model_type: ModelType::Marian,
    };
    pub const GERMAN2FRENCH: RemoteTranslationResources = Self {
        model_resource: MarianModelResources::GERMAN2FRENCH,
        config_resource: MarianConfigResources::GERMAN2FRENCH,
        vocab_resource: MarianVocabResources::GERMAN2FRENCH,
        merges_resource: MarianSpmResources::GERMAN2FRENCH,
        prefix: MarianPrefix::GERMAN2FRENCH,
        model_type: ModelType::Marian,
    };
    pub const ENGLISH2DUTCH: RemoteTranslationResources = Self {
        model_resource: MarianModelResources::ENGLISH2DUTCH,
        config_resource: MarianConfigResources::ENGLISH2DUTCH,
        vocab_resource: MarianVocabResources::ENGLISH2DUTCH,
        merges_resource: MarianSpmResources::ENGLISH2DUTCH,
        prefix: MarianPrefix::ENGLISH2DUTCH,
        model_type: ModelType::Marian,
    };
    pub const DUTCH2ENGLISH: RemoteTranslationResources = Self {
        model_resource: MarianModelResources::DUTCH2ENGLISH,
        config_resource: MarianConfigResources::DUTCH2ENGLISH,
        vocab_resource: MarianVocabResources::DUTCH2ENGLISH,
        merges_resource: MarianSpmResources::DUTCH2ENGLISH,
        prefix: MarianPrefix::DUTCH2ENGLISH,
        model_type: ModelType::Marian,
    };
}

/// # Configuration for text translation
/// Contains information regarding the model to load, mirrors the GenerationConfig, with a
/// different set of default parameters and sets the device to place the model on.
pub struct TranslationConfig {
    /// Model weights resource (default: pretrained BART model on CNN-DM)
    pub model_resource: Resource,
    /// Config resource (default: pretrained BART model on CNN-DM)
    pub config_resource: Resource,
    /// Vocab resource (default: pretrained BART model on CNN-DM)
    pub vocab_resource: Resource,
    /// Merges resource (default: pretrained BART model on CNN-DM)
    pub merges_resource: Resource,
    /// Minimum sequence length (default: 0)
    pub min_length: i64,
    /// Maximum sequence length (default: 20)
    pub max_length: i64,
    /// Sampling flag. If true, will perform top-k and/or nucleus sampling on generated tokens, otherwise greedy (deterministic) decoding (default: true)
    pub do_sample: bool,
    /// Early stopping flag indicating if the beam search should stop as soon as `num_beam` hypotheses have been generated (default: false)
    pub early_stopping: bool,
    /// Number of beams for beam search (default: 5)
    pub num_beams: i64,
    /// Temperature setting. Values higher than 1 will improve originality at the risk of reducing relevance (default: 1.0)
    pub temperature: f64,
    /// Top_k values for sampling tokens. Value higher than 0 will enable the feature (default: 0)
    pub top_k: i64,
    /// Top_p value for [Nucleus sampling, Holtzman et al.](http://arxiv.org/abs/1904.09751). Keep top tokens until cumulative probability reaches top_p (default: 0.9)
    pub top_p: f64,
    /// Repetition penalty (mostly useful for CTRL decoders). Values higher than 1 will penalize tokens that have been already generated. (default: 1.0)
    pub repetition_penalty: f64,
    /// Exponential penalty based on the length of the hypotheses generated (default: 1.0)
    pub length_penalty: f64,
    /// Number of allowed repetitions of n-grams. Values higher than 0 turn on this feature (default: 3)
    pub no_repeat_ngram_size: i64,
    /// Number of sequences to return for each prompt text (default: 1)
    pub num_return_sequences: i64,
    /// Device to place the model on (default: CUDA/GPU when available)
    pub device: Device,
    /// Prefix to append translation inputs with
    pub prefix: Option<String>,
    /// Number of beam groups for diverse beam generation. If provided and higher than 1, will split the beams into beam subgroups leading to more diverse generation.
    pub num_beam_groups: Option<i64>,
    /// Diversity penalty for diverse beam search. High values will enforce more difference between beam groups (default: 5.5)
    pub diversity_penalty: Option<f64>,
    /// Model type used for translation
    pub model_type: ModelType,
}

impl TranslationConfig {
    /// Create a new `TranslationCondiguration` from an available language.
    ///
    /// # Arguments
    ///
    /// * `language` - `Language` enum value (e.g. `Language::EnglishToFrench`)
    /// * `device` - `Device` to place the model on (CPU/GPU)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> anyhow::Result<()> {
    /// use rust_bert::pipelines::translation::{Language, TranslationConfig};
    /// use tch::Device;
    ///
    /// let translation_config =
    ///     TranslationConfig::new(Language::FrenchToEnglish, Device::cuda_if_available());
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(language: Language, device: Device) -> TranslationConfig {
        let translation_resource = match language {
            Language::EnglishToFrench => RemoteTranslationResources::ENGLISH2FRENCH,
            Language::EnglishToCatalan => RemoteTranslationResources::ENGLISH2CATALAN,
            Language::EnglishToSpanish => RemoteTranslationResources::ENGLISH2SPANISH,
            Language::EnglishToPortuguese => RemoteTranslationResources::ENGLISH2PORTUGUESE,
            Language::EnglishToItalian => RemoteTranslationResources::ENGLISH2ITALIAN,
            Language::EnglishToRomanian => RemoteTranslationResources::ENGLISH2ROMANIAN,
            Language::EnglishToGerman => RemoteTranslationResources::ENGLISH2GERMAN,
            Language::EnglishToRussian => RemoteTranslationResources::ENGLISH2RUSSIAN,
            Language::EnglishToDutch => RemoteTranslationResources::ENGLISH2DUTCH,

            Language::FrenchToEnglish => RemoteTranslationResources::FRENCH2ENGLISH,
            Language::CatalanToEnglish => RemoteTranslationResources::CATALAN2ENGLISH,
            Language::SpanishToEnglish => RemoteTranslationResources::SPANISH2ENGLISH,
            Language::PortugueseToEnglish => RemoteTranslationResources::PORTUGUESE2ENGLISH,
            Language::ItalianToEnglish => RemoteTranslationResources::ITALIAN2ENGLISH,
            Language::RomanianToEnglish => RemoteTranslationResources::ROMANIAN2ENGLISH,
            Language::GermanToEnglish => RemoteTranslationResources::GERMAN2ENGLISH,
            Language::RussianToEnglish => RemoteTranslationResources::RUSSIAN2ENGLISH,
            Language::DutchToEnglish => RemoteTranslationResources::DUTCH2ENGLISH,

            Language::EnglishToFrenchV2 => RemoteTranslationResources::ENGLISH2FRENCH_V2,
            Language::EnglishToGermanV2 => RemoteTranslationResources::ENGLISH2GERMAN_V2,

            Language::FrenchToGerman => RemoteTranslationResources::FRENCH2GERMAN,
            Language::GermanToFrench => RemoteTranslationResources::GERMAN2FRENCH,
        };
        let model_resource = Resource::Remote(RemoteResource::from_pretrained(
            translation_resource.model_resource,
        ));
        let config_resource = Resource::Remote(RemoteResource::from_pretrained(
            translation_resource.config_resource,
        ));
        let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(
            translation_resource.vocab_resource,
        ));
        let merges_resource = Resource::Remote(RemoteResource::from_pretrained(
            translation_resource.merges_resource,
        ));
        let prefix = match translation_resource.prefix {
            Some(value) => Some(value.to_string()),
            None => None,
        };
        TranslationConfig {
            model_resource,
            config_resource,
            vocab_resource,
            merges_resource,
            min_length: 0,
            max_length: 512,
            do_sample: false,
            early_stopping: true,
            num_beams: 6,
            temperature: 1.0,
            top_k: 50,
            top_p: 1.0,
            repetition_penalty: 1.0,
            length_penalty: 1.0,
            no_repeat_ngram_size: 0,
            num_return_sequences: 1,
            device,
            prefix,
            num_beam_groups: None,
            diversity_penalty: None,
            model_type: translation_resource.model_type,
        }
    }

    /// Create a new `TranslationConfiguration` from custom (e.g. local) resources.
    ///
    /// # Arguments
    ///
    /// * `model_resource` - `Resource` pointing to the model
    /// * `config_resource` - `Resource` pointing to the configuration
    /// * `vocab_resource` - `Resource` pointing to the vocabulary
    /// * `sentence_piece_resource` - `Resource` pointing to the sentence piece model of the source language
    /// * `device` - `Device` to place the model on (CPU/GPU)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> anyhow::Result<()> {
    /// use rust_bert::pipelines::common::ModelType;
    /// use rust_bert::pipelines::translation::TranslationConfig;
    /// use rust_bert::resources::{LocalResource, Resource};
    /// use std::path::PathBuf;
    /// use tch::Device;
    ///
    /// let config_resource = Resource::Local(LocalResource {
    ///     local_path: PathBuf::from("path/to/config.json"),
    /// });
    /// let model_resource = Resource::Local(LocalResource {
    ///     local_path: PathBuf::from("path/to/model.ot"),
    /// });
    /// let vocab_resource = Resource::Local(LocalResource {
    ///     local_path: PathBuf::from("path/to/vocab.json"),
    /// });
    /// let sentence_piece_resource = Resource::Local(LocalResource {
    ///     local_path: PathBuf::from("path/to/spiece.model"),
    /// });
    ///
    /// let translation_config = TranslationConfig::new_from_resources(
    ///     model_resource,
    ///     config_resource,
    ///     vocab_resource,
    ///     sentence_piece_resource,
    ///     Some(">>fr<<".to_string()),
    ///     Device::cuda_if_available(),
    ///     ModelType::Marian,
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub fn new_from_resources(
        model_resource: Resource,
        config_resource: Resource,
        vocab_resource: Resource,
        sentence_piece_resource: Resource,
        prefix: Option<String>,
        device: Device,
        model_type: ModelType,
    ) -> TranslationConfig {
        TranslationConfig {
            model_resource,
            config_resource,
            vocab_resource,
            merges_resource: sentence_piece_resource,
            min_length: 0,
            max_length: 512,
            do_sample: false,
            early_stopping: true,
            num_beams: 6,
            temperature: 1.0,
            top_k: 50,
            top_p: 1.0,
            repetition_penalty: 1.0,
            length_penalty: 1.0,
            no_repeat_ngram_size: 0,
            num_return_sequences: 1,
            device,
            prefix,
            num_beam_groups: None,
            diversity_penalty: None,
            model_type,
        }
    }
}

impl From<TranslationConfig> for GenerateConfig {
    fn from(config: TranslationConfig) -> GenerateConfig {
        GenerateConfig {
            model_resource: config.model_resource,
            config_resource: config.config_resource,
            merges_resource: config.merges_resource,
            vocab_resource: config.vocab_resource,
            min_length: config.min_length,
            max_length: config.max_length,
            do_sample: config.do_sample,
            early_stopping: config.early_stopping,
            num_beams: config.num_beams,
            temperature: config.temperature,
            top_k: config.top_k,
            top_p: config.top_p,
            repetition_penalty: config.repetition_penalty,
            length_penalty: config.length_penalty,
            no_repeat_ngram_size: config.no_repeat_ngram_size,
            num_return_sequences: config.num_return_sequences,
            num_beam_groups: config.num_beam_groups,
            diversity_penalty: config.diversity_penalty,
            device: config.device,
        }
    }
}

/// # Abstraction that holds one particular translation model, for any of the supported models
pub enum TranslationOption {
    /// Translator based on Marian model
    Marian(MarianGenerator),
    /// Translator based on T5 model
    T5(T5Generator),
}

impl TranslationOption {
    pub fn new(config: TranslationConfig) -> Result<Self, RustBertError> {
        match config.model_type {
            ModelType::Marian => Ok(TranslationOption::Marian(MarianGenerator::new(
                config.into(),
            )?)),
            ModelType::T5 => Ok(TranslationOption::T5(T5Generator::new(config.into())?)),
            _ => Err(RustBertError::InvalidConfigurationError(format!(
                "Translation not implemented for {:?}!",
                config.model_type
            ))),
        }
    }

    /// Returns the `ModelType` for this TranslationOption
    pub fn model_type(&self) -> ModelType {
        match *self {
            Self::Marian(_) => ModelType::Marian,
            Self::T5(_) => ModelType::T5,
        }
    }

    /// Interface method to generate() of the particular models.
    pub fn generate<'a, S>(
        &self,
        prompt_texts: Option<S>,
        attention_mask: Option<Tensor>,
    ) -> Vec<String>
    where
        S: AsRef<[&'a str]>,
    {
        match *self {
            Self::Marian(ref model) => {
                model.generate(prompt_texts, attention_mask, None, None, None)
            }
            Self::T5(ref model) => model.generate(prompt_texts, attention_mask, None, None, None),
        }
    }
}

/// # TranslationModel to perform translation
pub struct TranslationModel {
    model: TranslationOption,
    prefix: Option<String>,
}

impl TranslationModel {
    /// Build a new `TranslationModel`
    ///
    /// # Arguments
    ///
    /// * `translation_config` - `TranslationConfig` object containing the resource references (model, vocabulary, configuration), translation options and device placement (CPU/GPU)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> anyhow::Result<()> {
    /// use rust_bert::pipelines::translation::{Language, TranslationConfig, TranslationModel};
    /// use tch::Device;
    ///
    /// let translation_config =
    ///     TranslationConfig::new(Language::FrenchToEnglish, Device::cuda_if_available());
    /// let mut summarization_model = TranslationModel::new(translation_config)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(translation_config: TranslationConfig) -> Result<TranslationModel, RustBertError> {
        let prefix = translation_config.prefix.clone();
        let model = TranslationOption::new(translation_config)?;

        Ok(TranslationModel { model, prefix })
    }

    /// Translates texts provided
    ///
    /// # Arguments
    ///
    /// * `input` - `&[&str]` Array of texts to summarize.
    ///
    /// # Returns
    /// * `Vec<String>` Translated texts
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> anyhow::Result<()> {
    /// use rust_bert::pipelines::generation_utils::LanguageGenerator;
    /// use rust_bert::pipelines::translation::{Language, TranslationConfig, TranslationModel};
    /// use tch::Device;
    ///
    /// let translation_config =
    ///     TranslationConfig::new(Language::EnglishToFrench, Device::cuda_if_available());
    /// let model = TranslationModel::new(translation_config)?;
    ///
    /// let input = ["This is a sentence to be translated"];
    ///
    /// let output = model.translate(&input);
    /// # Ok(())
    /// # }
    /// ```
    pub fn translate<'a, S>(&self, texts: S) -> Vec<String>
    where
        S: AsRef<[&'a str]>,
    {
        match &self.prefix {
            Some(value) => {
                let texts = texts
                    .as_ref()
                    .iter()
                    .map(|&v| format!("{}{}", value, v))
                    .collect::<Vec<String>>();
                self.model.generate(
                    Some(texts.iter().map(AsRef::as_ref).collect::<Vec<&str>>()),
                    None,
                )
            }
            None => self.model.generate(Some(texts), None),
        }
    }
}
#[cfg(test)]
mod test {
    use super::*;

    #[test]
    #[ignore] // no need to run, compilation is enough to verify it is Send
    fn test() {
        let config = TranslationConfig::new(Language::FrenchToEnglish, Device::cuda_if_available());
        let _: Box<dyn Send> = Box::new(TranslationModel::new(config));
    }
}
