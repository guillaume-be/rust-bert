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

use tch::Device;

use crate::common::error::RustBertError;
use crate::m2m_100::M2M100Generator;
use crate::marian::MarianGenerator;
use crate::mbart::MBartGenerator;
use crate::pipelines::common::ModelType;
use crate::pipelines::generation_utils::private_generation_utils::PrivateLanguageGenerator;
use crate::pipelines::generation_utils::{GenerateConfig, GenerateOptions, LanguageGenerator};
use crate::resources::ResourceProvider;
use crate::t5::T5Generator;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt;
use std::fmt::{Debug, Display};

/// Language
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum Language {
    Afrikaans,
    Danish,
    Dutch,
    German,
    English,
    Icelandic,
    Luxembourgish,
    Norwegian,
    Swedish,
    WesternFrisian,
    Yiddish,
    Asturian,
    Catalan,
    French,
    Galician,
    Italian,
    Occitan,
    Portuguese,
    Romanian,
    Spanish,
    Belarusian,
    Bosnian,
    Bulgarian,
    Croatian,
    Czech,
    Macedonian,
    Polish,
    Russian,
    Serbian,
    Slovak,
    Slovenian,
    Ukrainian,
    Estonian,
    Finnish,
    Hungarian,
    Latvian,
    Lithuanian,
    Albanian,
    Armenian,
    Georgian,
    Greek,
    Breton,
    Irish,
    ScottishGaelic,
    Welsh,
    Azerbaijani,
    Bashkir,
    Kazakh,
    Turkish,
    Uzbek,
    Japanese,
    Korean,
    Vietnamese,
    ChineseMandarin,
    Bengali,
    Gujarati,
    Hindi,
    Kannada,
    Marathi,
    Nepali,
    Oriya,
    Panjabi,
    Sindhi,
    Sinhala,
    Urdu,
    Tamil,
    Cebuano,
    Iloko,
    Indonesian,
    Javanese,
    Malagasy,
    Malay,
    Malayalam,
    Sundanese,
    Tagalog,
    Burmese,
    CentralKhmer,
    Lao,
    Thai,
    Mongolian,
    Arabic,
    Hebrew,
    Pashto,
    Farsi,
    Amharic,
    Fulah,
    Hausa,
    Igbo,
    Lingala,
    Luganda,
    NorthernSotho,
    Somali,
    Swahili,
    Swati,
    Tswana,
    Wolof,
    Xhosa,
    Yoruba,
    Zulu,
    HaitianCreole,
}

impl Display for Language {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", {
            let input_string = format!("{:?}", self);
            let mut output: Vec<&str> = Vec::new();
            let mut start: usize = 0;

            for (c_pos, c) in input_string.char_indices() {
                if c.is_uppercase() {
                    if start < c_pos {
                        output.push(&input_string[start..c_pos]);
                    }
                    start = c_pos;
                }
            }
            if start < input_string.len() {
                output.push(&input_string[start..]);
            }
            output.join(" ")
        })
    }
}

impl Language {
    pub fn get_iso_639_1_code(&self) -> &'static str {
        match self {
            Language::Afrikaans => "af",
            Language::Danish => "da",
            Language::Dutch => "nl",
            Language::German => "de",
            Language::English => "en",
            Language::Icelandic => "is",
            Language::Luxembourgish => "lb",
            Language::Norwegian => "no",
            Language::Swedish => "sv",
            Language::WesternFrisian => "fy",
            Language::Yiddish => "yi",
            Language::Asturian => "ast",
            Language::Catalan => "ca",
            Language::French => "fr",
            Language::Galician => "gl",
            Language::Italian => "it",
            Language::Occitan => "oc",
            Language::Portuguese => "pt",
            Language::Romanian => "ro",
            Language::Spanish => "es",
            Language::Belarusian => "be",
            Language::Bosnian => "bs",
            Language::Bulgarian => "bg",
            Language::Croatian => "hr",
            Language::Czech => "cs",
            Language::Macedonian => "mk",
            Language::Polish => "pl",
            Language::Russian => "ru",
            Language::Serbian => "sr",
            Language::Slovak => "sk",
            Language::Slovenian => "sl",
            Language::Ukrainian => "uk",
            Language::Estonian => "et",
            Language::Finnish => "fi",
            Language::Hungarian => "hu",
            Language::Latvian => "lv",
            Language::Lithuanian => "lt",
            Language::Albanian => "sq",
            Language::Armenian => "hy",
            Language::Georgian => "ka",
            Language::Greek => "el",
            Language::Breton => "br",
            Language::Irish => "ga",
            Language::ScottishGaelic => "gd",
            Language::Welsh => "cy",
            Language::Azerbaijani => "az",
            Language::Bashkir => "ba",
            Language::Kazakh => "kk",
            Language::Turkish => "tr",
            Language::Uzbek => "uz",
            Language::Japanese => "ja",
            Language::Korean => "ko",
            Language::Vietnamese => "vi",
            Language::ChineseMandarin => "zh",
            Language::Bengali => "bn",
            Language::Gujarati => "gu",
            Language::Hindi => "hi",
            Language::Kannada => "kn",
            Language::Marathi => "mr",
            Language::Nepali => "ne",
            Language::Oriya => "or",
            Language::Panjabi => "pa",
            Language::Sindhi => "sd",
            Language::Sinhala => "si",
            Language::Urdu => "ur",
            Language::Tamil => "ta",
            Language::Cebuano => "ceb",
            Language::Iloko => "ilo",
            Language::Indonesian => "id",
            Language::Javanese => "jv",
            Language::Malagasy => "mg",
            Language::Malay => "ms",
            Language::Malayalam => "ml",
            Language::Sundanese => "su",
            Language::Tagalog => "tl",
            Language::Burmese => "my",
            Language::CentralKhmer => "km",
            Language::Lao => "lo",
            Language::Thai => "th",
            Language::Mongolian => "mn",
            Language::Arabic => "ar",
            Language::Hebrew => "he",
            Language::Pashto => "ps",
            Language::Farsi => "fa",
            Language::Amharic => "am",
            Language::Fulah => "ff",
            Language::Hausa => "ha",
            Language::Igbo => "ig",
            Language::Lingala => "ln",
            Language::Luganda => "lg",
            Language::NorthernSotho => "nso",
            Language::Somali => "so",
            Language::Swahili => "sw",
            Language::Swati => "ss",
            Language::Tswana => "tn",
            Language::Wolof => "wo",
            Language::Xhosa => "xh",
            Language::Yoruba => "yo",
            Language::Zulu => "zu",
            Language::HaitianCreole => "ht",
        }
    }

    pub fn get_iso_639_3_code(&self) -> &'static str {
        match self {
            Language::Afrikaans => "afr",
            Language::Danish => "dan",
            Language::Dutch => "nld",
            Language::German => "deu",
            Language::English => "eng",
            Language::Icelandic => "isl",
            Language::Luxembourgish => "ltz",
            Language::Norwegian => "nor",
            Language::Swedish => "swe",
            Language::WesternFrisian => "fry",
            Language::Yiddish => "yid",
            Language::Asturian => "ast",
            Language::Catalan => "cat",
            Language::French => "fra",
            Language::Galician => "glg",
            Language::Italian => "ita",
            Language::Occitan => "oci",
            Language::Portuguese => "por",
            Language::Romanian => "ron",
            Language::Spanish => "spa",
            Language::Belarusian => "bel",
            Language::Bosnian => "bos",
            Language::Bulgarian => "bul",
            Language::Croatian => "hrv",
            Language::Czech => "ces",
            Language::Macedonian => "mkd",
            Language::Polish => "pol",
            Language::Russian => "rus",
            Language::Serbian => "srp",
            Language::Slovak => "slk",
            Language::Slovenian => "slv",
            Language::Ukrainian => "ukr",
            Language::Estonian => "est",
            Language::Finnish => "fin",
            Language::Hungarian => "hun",
            Language::Latvian => "lav",
            Language::Lithuanian => "lit",
            Language::Albanian => "sqi",
            Language::Armenian => "hye",
            Language::Georgian => "kat",
            Language::Greek => "ell",
            Language::Breton => "bre",
            Language::Irish => "gle",
            Language::ScottishGaelic => "gla",
            Language::Welsh => "cym",
            Language::Azerbaijani => "aze",
            Language::Bashkir => "bak",
            Language::Kazakh => "kaz",
            Language::Turkish => "tur",
            Language::Uzbek => "uzb",
            Language::Japanese => "jpn",
            Language::Korean => "kor",
            Language::Vietnamese => "vie",
            Language::ChineseMandarin => "cmn",
            Language::Bengali => "ben",
            Language::Gujarati => "guj",
            Language::Hindi => "hin",
            Language::Kannada => "kan",
            Language::Marathi => "mar",
            Language::Nepali => "nep",
            Language::Oriya => "ori",
            Language::Panjabi => "pan",
            Language::Sindhi => "snd",
            Language::Sinhala => "sin",
            Language::Urdu => "urd",
            Language::Tamil => "tam",
            Language::Cebuano => "ceb",
            Language::Iloko => "ilo",
            Language::Indonesian => "ind",
            Language::Javanese => "jav",
            Language::Malagasy => "mlg",
            Language::Malay => "msa",
            Language::Malayalam => "mal",
            Language::Sundanese => "sun",
            Language::Tagalog => "tgl",
            Language::Burmese => "mya",
            Language::CentralKhmer => "khm",
            Language::Lao => "lao",
            Language::Thai => "tha",
            Language::Mongolian => "mon",
            Language::Arabic => "ara",
            Language::Hebrew => "heb",
            Language::Pashto => "pus",
            Language::Farsi => "fas",
            Language::Amharic => "amh",
            Language::Fulah => "ful",
            Language::Hausa => "hau",
            Language::Igbo => "ibo",
            Language::Lingala => "lin",
            Language::Luganda => "lug",
            Language::NorthernSotho => "nso",
            Language::Somali => "som",
            Language::Swahili => "swa",
            Language::Swati => "ssw",
            Language::Tswana => "tsn",
            Language::Wolof => "wol",
            Language::Xhosa => "xho",
            Language::Yoruba => "yor",
            Language::Zulu => "zul",
            Language::HaitianCreole => "hat",
        }
    }
}

/// # Configuration for text translation
/// Contains information regarding the model to load, mirrors the GenerationConfig, with a
/// different set of default parameters and sets the device to place the model on.
pub struct TranslationConfig {
    /// Model type used for translation
    pub model_type: ModelType,
    /// Model weights resource
    pub model_resource: Box<dyn ResourceProvider + Send>,
    /// Config resource
    pub config_resource: Box<dyn ResourceProvider + Send>,
    /// Vocab resource
    pub vocab_resource: Box<dyn ResourceProvider + Send>,
    /// Merges resource
    pub merges_resource: Option<Box<dyn ResourceProvider + Send>>,
    /// Supported source languages
    pub source_languages: HashSet<Language>,
    /// Supported target languages
    pub target_languages: HashSet<Language>,
    /// Minimum sequence length (default: 0)
    pub min_length: i64,
    /// Maximum sequence length (default: 512)
    pub max_length: Option<i64>,
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
    /// Number of beam groups for diverse beam generation. If provided and higher than 1, will split the beams into beam subgroups leading to more diverse generation.
    pub num_beam_groups: Option<i64>,
    /// Diversity penalty for diverse beam search. High values will enforce more difference between beam groups (default: 5.5)
    pub diversity_penalty: Option<f64>,
}

impl TranslationConfig {
    /// Create a new `TranslationConfiguration` from an available language.
    ///
    /// # Arguments
    ///
    /// * `language` - `Language` enum value (e.g. `Language::EnglishToFrench`)
    /// * `device` - `Device` to place the model on (CPU/GPU)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> anyhow::Result<()> {     ///
    /// use rust_bert::marian::{
    ///     MarianConfigResources, MarianModelResources, MarianSourceLanguages, MarianSpmResources,
    ///     MarianTargetLanguages, MarianVocabResources,
    /// };
    /// use rust_bert::pipelines::common::ModelType;
    /// use rust_bert::pipelines::translation::TranslationConfig;
    /// use rust_bert::resources::RemoteResource;
    /// use tch::Device;
    ///
    /// let model_resource = RemoteResource::from_pretrained(MarianModelResources::ROMANCE2ENGLISH);
    /// let config_resource = RemoteResource::from_pretrained(MarianConfigResources::ROMANCE2ENGLISH);
    /// let vocab_resource = RemoteResource::from_pretrained(MarianVocabResources::ROMANCE2ENGLISH);
    /// let spm_resource = RemoteResource::from_pretrained(MarianSpmResources::ROMANCE2ENGLISH);
    ///
    /// let source_languages = MarianSourceLanguages::ROMANCE2ENGLISH;
    /// let target_languages = MarianTargetLanguages::ROMANCE2ENGLISH;
    ///
    /// let translation_config = TranslationConfig::new(
    ///     ModelType::Marian,
    ///     model_resource,
    ///     config_resource,
    ///     vocab_resource,
    ///     Some(spm_resource),
    ///     source_languages,
    ///     target_languages,
    ///     Device::cuda_if_available(),
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub fn new<RM, RC, RV, S, T>(
        model_type: ModelType,
        model_resource: RM,
        config_resource: RC,
        vocab_resource: RV,
        merges_resource: Option<RV>,
        source_languages: S,
        target_languages: T,
        device: impl Into<Option<Device>>,
    ) -> TranslationConfig
    where
        RM: ResourceProvider + Send + 'static,
        RC: ResourceProvider + Send + 'static,
        RV: ResourceProvider + Send + 'static,
        S: AsRef<[Language]>,
        T: AsRef<[Language]>,
    {
        let device = device.into().unwrap_or_else(Device::cuda_if_available);

        TranslationConfig {
            model_type,
            model_resource: Box::new(model_resource),
            config_resource: Box::new(config_resource),
            vocab_resource: Box::new(vocab_resource),
            merges_resource: merges_resource.map(|r| Box::new(r) as Box<_>),
            source_languages: source_languages.as_ref().iter().cloned().collect(),
            target_languages: target_languages.as_ref().iter().cloned().collect(),
            device,
            min_length: 0,
            max_length: Some(512),
            do_sample: false,
            early_stopping: true,
            num_beams: 3,
            temperature: 1.0,
            top_k: 50,
            top_p: 1.0,
            repetition_penalty: 1.0,
            length_penalty: 1.0,
            no_repeat_ngram_size: 0,
            num_return_sequences: 1,
            num_beam_groups: None,
            diversity_penalty: None,
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

#[allow(clippy::large_enum_variant)]
/// # Abstraction that holds one particular translation model, for any of the supported models
pub enum TranslationOption {
    /// Translator based on Marian model
    Marian(MarianGenerator),
    /// Translator based on T5 model
    T5(T5Generator),
    /// Translator based on MBart50 model
    MBart(MBartGenerator),
    /// Translator based on M2M100 model
    M2M100(M2M100Generator),
}

impl TranslationOption {
    pub fn new(config: TranslationConfig) -> Result<Self, RustBertError> {
        match config.model_type {
            ModelType::Marian => Ok(TranslationOption::Marian(MarianGenerator::new(
                config.into(),
            )?)),
            ModelType::T5 => Ok(TranslationOption::T5(T5Generator::new(config.into())?)),
            ModelType::MBart => Ok(TranslationOption::MBart(MBartGenerator::new(
                config.into(),
            )?)),
            ModelType::M2M100 => Ok(TranslationOption::M2M100(M2M100Generator::new(
                config.into(),
            )?)),
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
            Self::MBart(_) => ModelType::MBart,
            Self::M2M100(_) => ModelType::M2M100,
        }
    }

    fn validate_and_get_prefix_and_forced_bos_id(
        &self,
        source_language: Option<&Language>,
        target_language: Option<&Language>,
        supported_source_languages: &HashSet<Language>,
        supported_target_languages: &HashSet<Language>,
    ) -> Result<(Option<String>, Option<i64>), RustBertError> {
        if let Some(source_language) = source_language {
            if !supported_source_languages.contains(source_language) {
                return Err(RustBertError::ValueError(format!(
                    "{} not in list of supported languages: {:?}",
                    source_language, supported_source_languages
                )));
            }
        }

        if let Some(target_language) = target_language {
            if !supported_target_languages.contains(target_language) {
                return Err(RustBertError::ValueError(format!(
                    "{} not in list of supported languages: {:?}",
                    target_language, supported_target_languages
                )));
            }
        }

        Ok(match *self {
            Self::Marian(_) => {
                if supported_target_languages.len() > 1 {
                    (
                        Some(format!(
                            ">>{}<< ",
                            match target_language {
                                Some(value) => value.get_iso_639_1_code(),
                                None => {
                                    return Err(RustBertError::ValueError(format!(
                                        "Missing target language for Marian \
                                        (multiple languages supported by model: {:?}, \
                                        need to specify target language)",
                                        supported_target_languages
                                    )));
                                }
                            }
                        )),
                        None,
                    )
                } else {
                    (None, None)
                }
            }
            Self::T5(_) => (
                Some(format!(
                    "translate {} to {}:",
                    match source_language {
                        Some(value) => value,
                        None => {
                            return Err(RustBertError::ValueError(
                                "Missing source language for T5".to_string(),
                            ));
                        }
                    },
                    match target_language {
                        Some(value) => value,
                        None => {
                            return Err(RustBertError::ValueError(
                                "Missing target language for T5".to_string(),
                            ));
                        }
                    }
                )),
                None,
            ),
            Self::MBart(ref model) => (
                Some(format!(
                    ">>{}<< ",
                    match source_language {
                        Some(value) => value.get_iso_639_1_code(),
                        None => {
                            return Err(RustBertError::ValueError(format!(
                                "Missing source language for MBart\
                                (multiple languages supported by model: {:?}, \
                                need to specify target language)",
                                supported_source_languages
                            )));
                        }
                    }
                )),
                if let Some(target_language) = target_language {
                    Some(
                        model._get_tokenizer().convert_tokens_to_ids(&[format!(
                            ">>{}<<",
                            target_language.get_iso_639_1_code()
                        )])[0],
                    )
                } else {
                    return Err(RustBertError::ValueError(format!(
                        "Missing target language for MBart\
                        (multiple languages supported by model: {:?}, \
                        need to specify target language)",
                        supported_target_languages
                    )));
                },
            ),
            Self::M2M100(ref model) => (
                Some(match source_language {
                    Some(value) => {
                        let language_code = value.get_iso_639_1_code();
                        match language_code.len() {
                            2 => format!(">>{}.<< ", language_code),
                            3 => format!(">>{}<< ", language_code),
                            _ => {
                                return Err(RustBertError::ValueError(
                                    "Invalid ISO 639-3 code".to_string(),
                                ));
                            }
                        }
                    }
                    None => {
                        return Err(RustBertError::ValueError(format!(
                            "Missing source language for M2M100 \
                            (multiple languages supported by model: {:?}, \
                            need to specify target language)",
                            supported_source_languages
                        )));
                    }
                }),
                if let Some(target_language) = target_language {
                    let language_code = target_language.get_iso_639_1_code();
                    Some(
                        model._get_tokenizer().convert_tokens_to_ids(&[
                            match language_code.len() {
                                2 => format!(">>{}.<<", language_code),
                                3 => format!(">>{}<<", language_code),
                                _ => {
                                    return Err(RustBertError::ValueError(
                                        "Invalid ISO 639-3 code".to_string(),
                                    ));
                                }
                            },
                        ])[0],
                    )
                } else {
                    return Err(RustBertError::ValueError(format!(
                        "Missing target language for M2M100 \
                        (multiple languages supported by model: {:?}, \
                        need to specify target language)",
                        supported_target_languages
                    )));
                },
            ),
        })
    }

    /// Interface method to generate() of the particular models.
    pub fn generate<S>(
        &self,
        prompt_texts: Option<&[S]>,
        forced_bos_token_id: Option<i64>,
    ) -> Vec<String>
    where
        S: AsRef<str> + Sync,
    {
        match *self {
            Self::Marian(ref model) => model
                .generate(prompt_texts, None)
                .into_iter()
                .map(|output| output.text)
                .collect(),
            Self::T5(ref model) => model
                .generate(prompt_texts, None)
                .into_iter()
                .map(|output| output.text)
                .collect(),
            Self::MBart(ref model) => {
                let generate_options = GenerateOptions {
                    forced_bos_token_id,
                    ..Default::default()
                };
                model
                    .generate(prompt_texts, Some(generate_options))
                    .into_iter()
                    .map(|output| output.text)
                    .collect()
            }
            Self::M2M100(ref model) => {
                let generate_options = GenerateOptions {
                    forced_bos_token_id,
                    ..Default::default()
                };
                model
                    .generate(prompt_texts, Some(generate_options))
                    .into_iter()
                    .map(|output| output.text)
                    .collect()
            }
        }
    }
}

/// # TranslationModel to perform translation
pub struct TranslationModel {
    model: TranslationOption,
    supported_source_languages: HashSet<Language>,
    supported_target_languages: HashSet<Language>,
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
    /// # fn main() -> anyhow::Result<()> {     ///
    /// use rust_bert::marian::{
    ///     MarianConfigResources, MarianModelResources, MarianSourceLanguages, MarianSpmResources,
    ///     MarianTargetLanguages, MarianVocabResources,
    /// };
    /// use rust_bert::pipelines::common::ModelType;
    /// use rust_bert::pipelines::translation::{TranslationConfig, TranslationModel};
    /// use rust_bert::resources::RemoteResource;
    /// use tch::Device;
    ///
    /// let model_resource = RemoteResource::from_pretrained(MarianModelResources::ROMANCE2ENGLISH);
    /// let config_resource = RemoteResource::from_pretrained(MarianConfigResources::ROMANCE2ENGLISH);
    /// let vocab_resource = RemoteResource::from_pretrained(MarianVocabResources::ROMANCE2ENGLISH);
    /// let spm_resource = RemoteResource::from_pretrained(MarianSpmResources::ROMANCE2ENGLISH);
    ///
    /// let source_languages = MarianSourceLanguages::ROMANCE2ENGLISH;
    /// let target_languages = MarianTargetLanguages::ROMANCE2ENGLISH;
    ///
    /// let translation_config = TranslationConfig::new(
    ///     ModelType::Marian,
    ///     model_resource,
    ///     config_resource,
    ///     vocab_resource,
    ///     Some(spm_resource),
    ///     source_languages,
    ///     target_languages,
    ///     Device::cuda_if_available(),
    /// );
    /// let mut summarization_model = TranslationModel::new(translation_config)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(translation_config: TranslationConfig) -> Result<TranslationModel, RustBertError> {
        let supported_source_languages = translation_config.source_languages.clone();
        let supported_target_languages = translation_config.target_languages.clone();

        let model = TranslationOption::new(translation_config)?;

        Ok(TranslationModel {
            model,
            supported_source_languages,
            supported_target_languages,
        })
    }

    /// Translates texts provided
    ///
    /// # Arguments
    /// * `input` - `&[&str]` Array of texts to summarize.
    ///
    /// # Returns
    /// * `Vec<String>` Translated texts
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> anyhow::Result<()> {
    /// use rust_bert::marian::{
    ///     MarianConfigResources, MarianModelResources, MarianSourceLanguages, MarianSpmResources,
    ///     MarianTargetLanguages, MarianVocabResources,
    /// };
    /// use rust_bert::pipelines::common::ModelType;
    /// use rust_bert::pipelines::translation::{Language, TranslationConfig, TranslationModel};
    /// use rust_bert::resources::RemoteResource;
    /// use tch::Device;
    ///
    /// let model_resource = RemoteResource::from_pretrained(MarianModelResources::ENGLISH2ROMANCE);
    /// let config_resource = RemoteResource::from_pretrained(MarianConfigResources::ENGLISH2ROMANCE);
    /// let vocab_resource = RemoteResource::from_pretrained(MarianVocabResources::ENGLISH2ROMANCE);
    /// let merges_resource = RemoteResource::from_pretrained(MarianSpmResources::ENGLISH2ROMANCE);
    /// let source_languages = MarianSourceLanguages::ENGLISH2ROMANCE;
    /// let target_languages = MarianTargetLanguages::ENGLISH2ROMANCE;
    ///
    /// let translation_config = TranslationConfig::new(
    ///     ModelType::Marian,
    ///     model_resource,
    ///     config_resource,
    ///     vocab_resource,
    ///     Some(merges_resource),
    ///     source_languages,
    ///     target_languages,
    ///     Device::cuda_if_available(),
    /// );
    /// let model = TranslationModel::new(translation_config)?;
    ///
    /// let input = ["This is a sentence to be translated"];
    /// let source_language = None;
    /// let target_language = Language::French;
    ///
    /// let output = model.translate(&input, source_language, target_language);
    /// # Ok(())
    /// # }
    /// ```
    pub fn translate<S>(
        &self,
        texts: &[S],
        source_language: impl Into<Option<Language>>,
        target_language: impl Into<Option<Language>>,
    ) -> Result<Vec<String>, RustBertError>
    where
        S: AsRef<str> + Sync,
    {
        let (prefix, forced_bos_token_id) = self.model.validate_and_get_prefix_and_forced_bos_id(
            source_language.into().as_ref(),
            target_language.into().as_ref(),
            &self.supported_source_languages,
            &self.supported_target_languages,
        )?;

        Ok(match prefix {
            Some(value) => {
                let texts = texts
                    .iter()
                    .map(|v| format!("{}{}", value, v.as_ref()))
                    .collect::<Vec<String>>();
                self.model.generate(Some(&texts), forced_bos_token_id)
            }
            None => self.model.generate(Some(texts), forced_bos_token_id),
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::marian::{
        MarianConfigResources, MarianModelResources, MarianSourceLanguages, MarianSpmResources,
        MarianTargetLanguages, MarianVocabResources,
    };
    use crate::resources::RemoteResource;

    #[test]
    #[ignore] // no need to run, compilation is enough to verify it is Send
    fn test() {
        let model_resource = RemoteResource::from_pretrained(MarianModelResources::ROMANCE2ENGLISH);
        let config_resource =
            RemoteResource::from_pretrained(MarianConfigResources::ROMANCE2ENGLISH);
        let vocab_resource = RemoteResource::from_pretrained(MarianVocabResources::ROMANCE2ENGLISH);
        let merges_resource = RemoteResource::from_pretrained(MarianSpmResources::ROMANCE2ENGLISH);

        let source_languages = MarianSourceLanguages::ROMANCE2ENGLISH;
        let target_languages = MarianTargetLanguages::ROMANCE2ENGLISH;

        let translation_config = TranslationConfig::new(
            ModelType::Marian,
            model_resource,
            config_resource,
            vocab_resource,
            Some(merges_resource),
            source_languages,
            target_languages,
            Device::cuda_if_available(),
        );
        let _: Box<dyn Send> = Box::new(TranslationModel::new(translation_config));
    }
}
