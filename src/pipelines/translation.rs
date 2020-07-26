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
//! # fn main() -> failure::Fallible<()> {
//! # use rust_bert::pipelines::generation::LanguageGenerator;
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

use crate::common::resources::{RemoteResource, Resource};
use crate::marian::{
    MarianConfigResources, MarianModelResources, MarianPrefix, MarianSpmResources,
    MarianVocabResources,
};
use crate::pipelines::common::ModelType;
use crate::pipelines::generation::{
    GenerateConfig, LanguageGenerator, MarianGenerator, T5Generator,
};
use crate::t5::{T5ConfigResources, T5ModelResources, T5Prefix, T5VocabResources};
use tch::{Device, Tensor};

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
    EnglishToFrench,
    EnglishToCatalan,
    EnglishToSpanish,
    EnglishToPortuguese,
    EnglishToItalian,
    EnglishToRomanian,
    EnglishToGerman,
    EnglishToRussian,
    EnglishToFrenchV2,
    EnglishToGermanV2,
    FrenchToGerman,
    GermanToFrench,
}

struct RemoteTranslationResources;

impl RemoteTranslationResources {
    pub const ENGLISH2FRENCH: (
        (&'static str, &'static str),
        (&'static str, &'static str),
        (&'static str, &'static str),
        (&'static str, &'static str),
        Option<&'static str>,
        ModelType,
    ) = (
        MarianModelResources::ENGLISH2ROMANCE,
        MarianConfigResources::ENGLISH2ROMANCE,
        MarianVocabResources::ENGLISH2ROMANCE,
        MarianSpmResources::ENGLISH2ROMANCE,
        MarianPrefix::ENGLISH2FRENCH,
        ModelType::Marian,
    );
    pub const ENGLISH2FRENCH_V2: (
        (&'static str, &'static str),
        (&'static str, &'static str),
        (&'static str, &'static str),
        (&'static str, &'static str),
        Option<&'static str>,
        ModelType,
    ) = (
        T5ModelResources::T5_BASE,
        T5ConfigResources::T5_BASE,
        T5VocabResources::T5_BASE,
        T5VocabResources::T5_BASE,
        T5Prefix::ENGLISH2FRENCH,
        ModelType::T5,
    );
    pub const ENGLISH2GERMAN_V2: (
        (&'static str, &'static str),
        (&'static str, &'static str),
        (&'static str, &'static str),
        (&'static str, &'static str),
        Option<&'static str>,
        ModelType,
    ) = (
        T5ModelResources::T5_BASE,
        T5ConfigResources::T5_BASE,
        T5VocabResources::T5_BASE,
        T5VocabResources::T5_BASE,
        T5Prefix::ENGLISH2GERMAN,
        ModelType::T5,
    );
    pub const ENGLISH2CATALAN: (
        (&'static str, &'static str),
        (&'static str, &'static str),
        (&'static str, &'static str),
        (&'static str, &'static str),
        Option<&'static str>,
        ModelType,
    ) = (
        MarianModelResources::ENGLISH2ROMANCE,
        MarianConfigResources::ENGLISH2ROMANCE,
        MarianVocabResources::ENGLISH2ROMANCE,
        MarianSpmResources::ENGLISH2ROMANCE,
        MarianPrefix::ENGLISH2CATALAN,
        ModelType::Marian,
    );
    pub const ENGLISH2SPANISH: (
        (&'static str, &'static str),
        (&'static str, &'static str),
        (&'static str, &'static str),
        (&'static str, &'static str),
        Option<&'static str>,
        ModelType,
    ) = (
        MarianModelResources::ENGLISH2ROMANCE,
        MarianConfigResources::ENGLISH2ROMANCE,
        MarianVocabResources::ENGLISH2ROMANCE,
        MarianSpmResources::ENGLISH2ROMANCE,
        MarianPrefix::ENGLISH2SPANISH,
        ModelType::Marian,
    );
    pub const ENGLISH2PORTUGUESE: (
        (&'static str, &'static str),
        (&'static str, &'static str),
        (&'static str, &'static str),
        (&'static str, &'static str),
        Option<&'static str>,
        ModelType,
    ) = (
        MarianModelResources::ENGLISH2ROMANCE,
        MarianConfigResources::ENGLISH2ROMANCE,
        MarianVocabResources::ENGLISH2ROMANCE,
        MarianSpmResources::ENGLISH2ROMANCE,
        MarianPrefix::ENGLISH2PORTUGUESE,
        ModelType::Marian,
    );
    pub const ENGLISH2ITALIAN: (
        (&'static str, &'static str),
        (&'static str, &'static str),
        (&'static str, &'static str),
        (&'static str, &'static str),
        Option<&'static str>,
        ModelType,
    ) = (
        MarianModelResources::ENGLISH2ROMANCE,
        MarianConfigResources::ENGLISH2ROMANCE,
        MarianVocabResources::ENGLISH2ROMANCE,
        MarianSpmResources::ENGLISH2ROMANCE,
        MarianPrefix::ENGLISH2ITALIAN,
        ModelType::Marian,
    );
    pub const ENGLISH2ROMANIAN: (
        (&'static str, &'static str),
        (&'static str, &'static str),
        (&'static str, &'static str),
        (&'static str, &'static str),
        Option<&'static str>,
        ModelType,
    ) = (
        MarianModelResources::ENGLISH2ROMANCE,
        MarianConfigResources::ENGLISH2ROMANCE,
        MarianVocabResources::ENGLISH2ROMANCE,
        MarianSpmResources::ENGLISH2ROMANCE,
        MarianPrefix::ENGLISH2ROMANIAN,
        ModelType::Marian,
    );
    pub const ENGLISH2GERMAN: (
        (&'static str, &'static str),
        (&'static str, &'static str),
        (&'static str, &'static str),
        (&'static str, &'static str),
        Option<&'static str>,
        ModelType,
    ) = (
        MarianModelResources::ENGLISH2GERMAN,
        MarianConfigResources::ENGLISH2GERMAN,
        MarianVocabResources::ENGLISH2GERMAN,
        MarianSpmResources::ENGLISH2GERMAN,
        MarianPrefix::ENGLISH2GERMAN,
        ModelType::Marian,
    );
    pub const ENGLISH2RUSSIAN: (
        (&'static str, &'static str),
        (&'static str, &'static str),
        (&'static str, &'static str),
        (&'static str, &'static str),
        Option<&'static str>,
        ModelType,
    ) = (
        MarianModelResources::ENGLISH2RUSSIAN,
        MarianConfigResources::ENGLISH2RUSSIAN,
        MarianVocabResources::ENGLISH2RUSSIAN,
        MarianSpmResources::ENGLISH2RUSSIAN,
        MarianPrefix::ENGLISH2RUSSIAN,
        ModelType::Marian,
    );

    pub const FRENCH2ENGLISH: (
        (&'static str, &'static str),
        (&'static str, &'static str),
        (&'static str, &'static str),
        (&'static str, &'static str),
        Option<&'static str>,
        ModelType,
    ) = (
        MarianModelResources::ROMANCE2ENGLISH,
        MarianConfigResources::ROMANCE2ENGLISH,
        MarianVocabResources::ROMANCE2ENGLISH,
        MarianSpmResources::ROMANCE2ENGLISH,
        MarianPrefix::FRENCH2ENGLISH,
        ModelType::Marian,
    );
    pub const CATALAN2ENGLISH: (
        (&'static str, &'static str),
        (&'static str, &'static str),
        (&'static str, &'static str),
        (&'static str, &'static str),
        Option<&'static str>,
        ModelType,
    ) = (
        MarianModelResources::ROMANCE2ENGLISH,
        MarianConfigResources::ROMANCE2ENGLISH,
        MarianVocabResources::ROMANCE2ENGLISH,
        MarianSpmResources::ROMANCE2ENGLISH,
        MarianPrefix::CATALAN2ENGLISH,
        ModelType::Marian,
    );
    pub const SPANISH2ENGLISH: (
        (&'static str, &'static str),
        (&'static str, &'static str),
        (&'static str, &'static str),
        (&'static str, &'static str),
        Option<&'static str>,
        ModelType,
    ) = (
        MarianModelResources::ROMANCE2ENGLISH,
        MarianConfigResources::ROMANCE2ENGLISH,
        MarianVocabResources::ROMANCE2ENGLISH,
        MarianSpmResources::ROMANCE2ENGLISH,
        MarianPrefix::SPANISH2ENGLISH,
        ModelType::Marian,
    );
    pub const PORTUGUESE2ENGLISH: (
        (&'static str, &'static str),
        (&'static str, &'static str),
        (&'static str, &'static str),
        (&'static str, &'static str),
        Option<&'static str>,
        ModelType,
    ) = (
        MarianModelResources::ROMANCE2ENGLISH,
        MarianConfigResources::ROMANCE2ENGLISH,
        MarianVocabResources::ROMANCE2ENGLISH,
        MarianSpmResources::ROMANCE2ENGLISH,
        MarianPrefix::PORTUGUESE2ENGLISH,
        ModelType::Marian,
    );
    pub const ITALIAN2ENGLISH: (
        (&'static str, &'static str),
        (&'static str, &'static str),
        (&'static str, &'static str),
        (&'static str, &'static str),
        Option<&'static str>,
        ModelType,
    ) = (
        MarianModelResources::ROMANCE2ENGLISH,
        MarianConfigResources::ROMANCE2ENGLISH,
        MarianVocabResources::ROMANCE2ENGLISH,
        MarianSpmResources::ROMANCE2ENGLISH,
        MarianPrefix::ITALIAN2ENGLISH,
        ModelType::Marian,
    );
    pub const ROMANIAN2ENGLISH: (
        (&'static str, &'static str),
        (&'static str, &'static str),
        (&'static str, &'static str),
        (&'static str, &'static str),
        Option<&'static str>,
        ModelType,
    ) = (
        MarianModelResources::ROMANCE2ENGLISH,
        MarianConfigResources::ROMANCE2ENGLISH,
        MarianVocabResources::ROMANCE2ENGLISH,
        MarianSpmResources::ROMANCE2ENGLISH,
        MarianPrefix::ROMANIAN2ENGLISH,
        ModelType::Marian,
    );
    pub const GERMAN2ENGLISH: (
        (&'static str, &'static str),
        (&'static str, &'static str),
        (&'static str, &'static str),
        (&'static str, &'static str),
        Option<&'static str>,
        ModelType,
    ) = (
        MarianModelResources::GERMAN2ENGLISH,
        MarianConfigResources::GERMAN2ENGLISH,
        MarianVocabResources::GERMAN2ENGLISH,
        MarianSpmResources::GERMAN2ENGLISH,
        MarianPrefix::GERMAN2ENGLISH,
        ModelType::Marian,
    );
    pub const RUSSIAN2ENGLISH: (
        (&'static str, &'static str),
        (&'static str, &'static str),
        (&'static str, &'static str),
        (&'static str, &'static str),
        Option<&'static str>,
        ModelType,
    ) = (
        MarianModelResources::RUSSIAN2ENGLISH,
        MarianConfigResources::RUSSIAN2ENGLISH,
        MarianVocabResources::RUSSIAN2ENGLISH,
        MarianSpmResources::RUSSIAN2ENGLISH,
        MarianPrefix::RUSSIAN2ENGLISH,
        ModelType::Marian,
    );

    pub const FRENCH2GERMAN: (
        (&'static str, &'static str),
        (&'static str, &'static str),
        (&'static str, &'static str),
        (&'static str, &'static str),
        Option<&'static str>,
        ModelType,
    ) = (
        MarianModelResources::FRENCH2GERMAN,
        MarianConfigResources::FRENCH2GERMAN,
        MarianVocabResources::FRENCH2GERMAN,
        MarianSpmResources::FRENCH2GERMAN,
        MarianPrefix::FRENCH2GERMAN,
        ModelType::Marian,
    );
    pub const GERMAN2FRENCH: (
        (&'static str, &'static str),
        (&'static str, &'static str),
        (&'static str, &'static str),
        (&'static str, &'static str),
        Option<&'static str>,
        ModelType,
    ) = (
        MarianModelResources::GERMAN2FRENCH,
        MarianConfigResources::GERMAN2FRENCH,
        MarianVocabResources::GERMAN2FRENCH,
        MarianSpmResources::GERMAN2FRENCH,
        MarianPrefix::GERMAN2FRENCH,
        ModelType::Marian,
    );
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
    pub min_length: u64,
    /// Maximum sequence length (default: 20)
    pub max_length: u64,
    /// Sampling flag. If true, will perform top-k and/or nucleus sampling on generated tokens, otherwise greedy (deterministic) decoding (default: true)
    pub do_sample: bool,
    /// Early stopping flag indicating if the beam search should stop as soon as `num_beam` hypotheses have been generated (default: false)
    pub early_stopping: bool,
    /// Number of beams for beam search (default: 5)
    pub num_beams: u64,
    /// Temperature setting. Values higher than 1 will improve originality at the risk of reducing relevance (default: 1.0)
    pub temperature: f64,
    /// Top_k values for sampling tokens. Value higher than 0 will enable the feature (default: 0)
    pub top_k: u64,
    /// Top_p value for [Nucleus sampling, Holtzman et al.](http://arxiv.org/abs/1904.09751). Keep top tokens until cumulative probability reaches top_p (default: 0.9)
    pub top_p: f64,
    /// Repetition penalty (mostly useful for CTRL decoders). Values higher than 1 will penalize tokens that have been already generated. (default: 1.0)
    pub repetition_penalty: f64,
    /// Exponential penalty based on the length of the hypotheses generated (default: 1.0)
    pub length_penalty: f64,
    /// Number of allowed repetitions of n-grams. Values higher than 0 turn on this feature (default: 3)
    pub no_repeat_ngram_size: u64,
    /// Number of sequences to return for each prompt text (default: 1)
    pub num_return_sequences: u64,
    /// Device to place the model on (default: CUDA/GPU when available)
    pub device: Device,
    /// Prefix to append translation inputs with
    pub prefix: Option<String>,
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
    /// # fn main() -> failure::Fallible<()> {
    /// use rust_bert::pipelines::translation::{Language, TranslationConfig};
    /// use tch::Device;
    ///
    /// let translation_config =
    ///     TranslationConfig::new(Language::FrenchToEnglish, Device::cuda_if_available());
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(language: Language, device: Device) -> TranslationConfig {
        let (model_resource, config_resource, vocab_resource, merges_resource, prefix, model_type) =
            match language {
                Language::EnglishToFrench => RemoteTranslationResources::ENGLISH2FRENCH,
                Language::EnglishToCatalan => RemoteTranslationResources::ENGLISH2CATALAN,
                Language::EnglishToSpanish => RemoteTranslationResources::ENGLISH2SPANISH,
                Language::EnglishToPortuguese => RemoteTranslationResources::ENGLISH2PORTUGUESE,
                Language::EnglishToItalian => RemoteTranslationResources::ENGLISH2ITALIAN,
                Language::EnglishToRomanian => RemoteTranslationResources::ENGLISH2ROMANIAN,
                Language::EnglishToGerman => RemoteTranslationResources::ENGLISH2GERMAN,
                Language::EnglishToRussian => RemoteTranslationResources::ENGLISH2RUSSIAN,

                Language::FrenchToEnglish => RemoteTranslationResources::FRENCH2ENGLISH,
                Language::CatalanToEnglish => RemoteTranslationResources::CATALAN2ENGLISH,
                Language::SpanishToEnglish => RemoteTranslationResources::SPANISH2ENGLISH,
                Language::PortugueseToEnglish => RemoteTranslationResources::PORTUGUESE2ENGLISH,
                Language::ItalianToEnglish => RemoteTranslationResources::ITALIAN2ENGLISH,
                Language::RomanianToEnglish => RemoteTranslationResources::ROMANIAN2ENGLISH,
                Language::GermanToEnglish => RemoteTranslationResources::GERMAN2ENGLISH,
                Language::RussianToEnglish => RemoteTranslationResources::RUSSIAN2ENGLISH,

                Language::EnglishToFrenchV2 => RemoteTranslationResources::ENGLISH2FRENCH_V2,
                Language::EnglishToGermanV2 => RemoteTranslationResources::ENGLISH2GERMAN_V2,

                Language::FrenchToGerman => RemoteTranslationResources::FRENCH2GERMAN,
                Language::GermanToFrench => RemoteTranslationResources::GERMAN2FRENCH,
            };
        let model_resource = Resource::Remote(RemoteResource::from_pretrained(model_resource));
        let config_resource = Resource::Remote(RemoteResource::from_pretrained(config_resource));
        let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(vocab_resource));
        let merges_resource = Resource::Remote(RemoteResource::from_pretrained(merges_resource));
        let prefix = match prefix {
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
            early_stopping: false,
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
            model_type,
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
    /// # fn main() -> failure::Fallible<()> {
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
            early_stopping: false,
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
            model_type,
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
    pub fn new(config: TranslationConfig) -> Self {
        let generate_config = GenerateConfig {
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
            device: config.device,
        };
        match config.model_type {
            ModelType::Marian => {
                TranslationOption::Marian(MarianGenerator::new(generate_config).unwrap())
            }
            ModelType::T5 => TranslationOption::T5(T5Generator::new(generate_config).unwrap()),
            ModelType::Bert => {
                panic!("Translation not implemented for Electra!");
            }
            ModelType::DistilBert => {
                panic!("Translation not implemented for DistilBert!");
            }
            ModelType::Roberta => {
                panic!("Translation not implemented for Roberta!");
            }
            ModelType::XLMRoberta => {
                panic!("Translation not implemented for XLMRoberta!");
            }
            ModelType::Electra => {
                panic!("Translation not implemented for Electra!");
            }
            ModelType::Albert => {
                panic!("Translation not implemented for Albert!");
            }
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
    pub fn generate(
        &self,
        prompt_texts: Option<Vec<&str>>,
        attention_mask: Option<Tensor>,
    ) -> Vec<String> {
        match *self {
            Self::Marian(ref model) => model.generate(prompt_texts, attention_mask),
            Self::T5(ref model) => model.generate(prompt_texts, attention_mask),
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
    /// # fn main() -> failure::Fallible<()> {
    /// use rust_bert::pipelines::translation::{Language, TranslationConfig, TranslationModel};
    /// use tch::Device;
    ///
    /// let translation_config =
    ///     TranslationConfig::new(Language::FrenchToEnglish, Device::cuda_if_available());
    /// let mut summarization_model = TranslationModel::new(translation_config)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(translation_config: TranslationConfig) -> failure::Fallible<TranslationModel> {
        let prefix = translation_config.prefix.clone();
        let model = TranslationOption::new(translation_config);

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
    /// # fn main() -> failure::Fallible<()> {
    /// use rust_bert::pipelines::generation::LanguageGenerator;
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
    pub fn translate(&self, texts: &[&str]) -> Vec<String> {
        match &self.prefix {
            Some(value) => {
                let texts: Vec<String> = texts
                    .into_iter()
                    .map(|&v| format!("{} {}", value, v))
                    .collect();
                self.model
                    .generate(Some(texts.iter().map(AsRef::as_ref).collect()), None)
            }
            None => self.model.generate(Some(texts.to_vec()), None),
        }
    }
}
