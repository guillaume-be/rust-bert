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

use tch::{Device, Tensor};

use crate::common::error::RustBertError;
use crate::common::resources::Resource;
use crate::m2m_100::{
    M2M100ConfigResources, M2M100Generator, M2M100MergesResources, M2M100ModelResources,
    M2M100SourceLanguages, M2M100TargetLanguages, M2M100VocabResources,
};
use crate::marian::{
    MarianConfigResources, MarianGenerator, MarianModelResources, MarianSourceLanguages,
    MarianSpmResources, MarianTargetLanguages, MarianVocabResources,
};
use crate::mbart::{
    MBartConfigResources, MBartGenerator, MBartModelResources, MBartSourceLanguages,
    MBartTargetLanguages, MBartVocabResources,
};
use crate::pipelines::common::ModelType;
use crate::pipelines::generation_utils::private_generation_utils::PrivateLanguageGenerator;
use crate::pipelines::generation_utils::{GenerateConfig, LanguageGenerator};
use crate::resources::RemoteResource;
use crate::t5::T5Generator;
use std::collections::HashSet;
use std::fmt;
use std::fmt::{Debug, Display};

/// Language
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
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
    pub model_resource: Resource,
    /// Config resource
    pub config_resource: Resource,
    /// Vocab resource
    pub vocab_resource: Resource,
    /// Merges resource
    pub merges_resource: Resource,
    /// Supported source languages
    pub source_languages: HashSet<Language>,
    /// Supported target languages
    pub target_languages: HashSet<Language>,
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
    /// # fn main() -> anyhow::Result<()> {
    /// use rust_bert::marian::{
    ///     MarianConfigResources, MarianModelResources, MarianSourceLanguages, MarianTargetLanguages,
    ///     MarianVocabResources,
    /// };
    /// use rust_bert::pipelines::common::ModelType;
    /// use rust_bert::pipelines::translation::{OldLanguage, TranslationConfig};
    /// use rust_bert::resources::{RemoteResource, Resource};
    /// use tch::Device;
    ///
    /// let model_resource = Resource::Remote(RemoteResource::from_pretrained(
    ///     MarianModelResources::ROMANCE2ENGLISH,
    /// ));
    /// let config_resource = Resource::Remote(RemoteResource::from_pretrained(
    ///     MarianConfigResources::ROMANCE2ENGLISH,
    /// ));
    /// let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(
    ///     MarianVocabResources::ROMANCE2ENGLISH,
    /// ));
    ///
    /// let source_languages = MarianSourceLanguages::ROMANCE2ENGLISH.iter().collect();
    /// let target_languages = MarianTargetLanguages::ROMANCE2ENGLISH.iter().collect();
    ///
    /// let translation_config = TranslationConfig::new(
    ///     ModelType::Marian,
    ///     model_resource,
    ///     config_resource,
    ///     vocab_resource.clone(),
    ///     vocab_resource,
    ///     source_languages,
    ///     target_languages,
    ///     device: Device::cuda_if_available(),
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub fn new<S, T>(
        model_type: ModelType,
        model_resource: Resource,
        config_resource: Resource,
        vocab_resource: Resource,
        merges_resource: Resource,
        source_languages: S,
        target_languages: T,
        device: impl Into<Option<Device>>,
    ) -> TranslationConfig
    where
        S: AsRef<[Language]>,
        T: AsRef<[Language]>,
    {
        let device = device.into().unwrap_or_else(|| Device::cuda_if_available());

        TranslationConfig {
            model_type,
            model_resource,
            config_resource,
            vocab_resource,
            merges_resource,
            source_languages: source_languages.as_ref().iter().cloned().collect(),
            target_languages: target_languages.as_ref().iter().cloned().collect(),
            device,
            min_length: 0,
            max_length: 512,
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
                    source_language.to_string(),
                    supported_source_languages
                )));
            }
        }

        if let Some(target_language) = target_language {
            if !supported_target_languages.contains(target_language) {
                return Err(RustBertError::ValueError(format!(
                    "{} not in list of supported languages: {:?}",
                    target_language.to_string(),
                    supported_target_languages
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
                                    return Err(RustBertError::ValueError(
                                        "Missing target language for Marian".to_string(),
                                    ));
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
                        Some(value) => value.to_string(),
                        None => {
                            return Err(RustBertError::ValueError(
                                "Missing source language for T5".to_string(),
                            ));
                        }
                    },
                    match target_language {
                        Some(value) => value.to_string(),
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
                            return Err(RustBertError::ValueError(
                                "Missing source language for MBart".to_string(),
                            ));
                        }
                    }
                )),
                if let Some(target_language) = target_language {
                    Some(
                        model._get_tokenizer().convert_tokens_to_ids([format!(
                            ">>{}<<",
                            target_language.get_iso_639_1_code()
                        )])[0],
                    )
                } else {
                    return Err(RustBertError::ValueError(
                        "Missing target language for MBart".to_string(),
                    ));
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
                        return Err(RustBertError::ValueError(
                            "Missing source language for M2M100".to_string(),
                        ));
                    }
                }),
                if let Some(target_language) = target_language {
                    let language_code = target_language.get_iso_639_1_code();
                    Some(
                        model
                            ._get_tokenizer()
                            .convert_tokens_to_ids([match language_code.len() {
                                2 => format!(">>{}.<<", language_code),
                                3 => format!(">>{}<<", language_code),
                                _ => {
                                    return Err(RustBertError::ValueError(
                                        "Invalid ISO 639-3 code".to_string(),
                                    ));
                                }
                            }])[0],
                    )
                } else {
                    return Err(RustBertError::ValueError(
                        "Missing target language for MBart".to_string(),
                    ));
                },
            ),
        })
    }

    /// Interface method to generate() of the particular models.
    pub fn generate<'a, S>(
        &self,
        prompt_texts: Option<S>,
        attention_mask: Option<Tensor>,
        forced_bos_token_id: Option<i64>,
    ) -> Vec<String>
    where
        S: AsRef<[&'a str]>,
    {
        match *self {
            Self::Marian(ref model) => model
                .generate(
                    prompt_texts,
                    attention_mask,
                    None,
                    None,
                    None,
                    None,
                    None,
                    false,
                )
                .into_iter()
                .map(|output| output.text)
                .collect(),
            Self::T5(ref model) => model
                .generate(
                    prompt_texts,
                    attention_mask,
                    None,
                    None,
                    None,
                    None,
                    None,
                    false,
                )
                .into_iter()
                .map(|output| output.text)
                .collect(),
            Self::MBart(ref model) => model
                .generate(
                    prompt_texts,
                    attention_mask,
                    None,
                    None,
                    None,
                    forced_bos_token_id,
                    None,
                    false,
                )
                .into_iter()
                .map(|output| output.text)
                .collect(),
            Self::M2M100(ref model) => model
                .generate(
                    prompt_texts,
                    attention_mask,
                    None,
                    None,
                    None,
                    forced_bos_token_id,
                    None,
                    false,
                )
                .into_iter()
                .map(|output| output.text)
                .collect(),
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
    /// # fn main() -> anyhow::Result<()> {
    /// use rust_bert::pipelines::translation::{OldLanguage, TranslationConfig, TranslationModel};
    /// use tch::Device;
    /// use rust_bert::resources::{Resource, RemoteResource};
    /// use rust_bert::marian::{MarianConfigResources, MarianModelResources, MarianVocabResources, MarianSourceLanguages, MarianTargetLanguages};
    /// use rust_bert::pipelines::common::ModelType;
    ///
    /// let model_resource = Resource::Remote(RemoteResource::from_pretrained(
    ///     MarianModelResources::ROMANCE2ENGLISH,
    /// ));
    /// let config_resource = Resource::Remote(RemoteResource::from_pretrained(
    ///     MarianConfigResources::ROMANCE2ENGLISH,
    /// ));
    /// let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(
    ///     MarianVocabResources::ROMANCE2ENGLISH,
    /// ));
    ///
    /// let source_languages = MarianSourceLanguages::ROMANCE2ENGLISH.iter().collect();
    /// let target_languages = MarianTargetLanguages::ROMANCE2ENGLISH.iter().collect();
    ///
    /// let translation_config = TranslationConfig::new(
    ///     ModelType::Marian,
    ///     model_resource,
    ///     config_resource,
    ///     vocab_resource.clone(),
    ///     vocab_resource,
    ///     source_languages,
    ///     target_languages,
    ///     device: Device::cuda_if_available(),
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
    /// use rust_bert::pipelines::translation::{OldLanguage, TranslationConfig, TranslationModel, Language};
    /// use tch::Device;
    /// use rust_bert::resources::{Resource, RemoteResource};
    /// use rust_bert::marian::{MarianConfigResources, MarianModelResources, MarianVocabResources, MarianSourceLanguages, MarianTargetLanguages, MarianSpmResources};
    /// use rust_bert::pipelines::common::ModelType;
    ///
    /// let model_resource = Resource::Remote(RemoteResource::from_pretrained(
    ///     MarianModelResources::ENGLISH2ROMANCE,
    /// ));
    /// let config_resource = Resource::Remote(RemoteResource::from_pretrained(
    ///     MarianConfigResources::ENGLISH2ROMANCE,
    /// ));
    /// let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(
    ///     MarianVocabResources::ENGLISH2ROMANCE,
    /// ));
    /// let merges_resource = Resource::Remote(RemoteResource::from_pretrained(
    ///      MarianSpmResources::ENGLISH2ROMANCE,
    /// ));
    /// let source_languages = MarianSourceLanguages::ENGLISH2ROMANCE.iter().collect();
    /// let target_languages = MarianTargetLanguages::ENGLISH2ROMANCE.iter().collect();
    ///
    /// let translation_config = TranslationConfig::new(
    ///     ModelType::Marian,
    ///     model_resource,
    ///     config_resource,
    ///     vocab_resource,
    ///     merges_resource,
    ///     source_languages,
    ///     target_languages,
    ///     device: Device::cuda_if_available(),
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
    pub fn translate<'a, S>(
        &self,
        texts: S,
        source_language: impl Into<Option<Language>>,
        target_language: impl Into<Option<Language>>,
    ) -> Result<Vec<String>, RustBertError>
    where
        S: AsRef<[&'a str]>,
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
                    .as_ref()
                    .iter()
                    .map(|&v| format!("{}{}", value, v))
                    .collect::<Vec<String>>();
                self.model.generate(
                    Some(texts.iter().map(AsRef::as_ref).collect::<Vec<&str>>()),
                    None,
                    forced_bos_token_id,
                )
            }
            None => self.model.generate(Some(texts), None, forced_bos_token_id),
        })
    }
}

struct TranslationResources {
    model_type: ModelType,
    model_resource: Resource,
    config_resource: Resource,
    vocab_resource: Resource,
    merges_resource: Resource,
    source_languages: Vec<Language>,
    target_languages: Vec<Language>,
}

#[derive(Clone, Copy, PartialEq)]
enum ModelSize {
    Medium,
    Large,
    XLarge,
}

pub struct TranslationModelBuilder<S, T>
where
    S: AsRef<[Language]> + Debug,
    T: AsRef<[Language]> + Debug,
{
    model_type: Option<ModelType>,
    source_languages: Option<S>,
    target_languages: Option<T>,
    device: Option<Device>,
    model_size: Option<ModelSize>,
}

impl<S, T> TranslationModelBuilder<S, T>
where
    S: AsRef<[Language]> + Debug,
    T: AsRef<[Language]> + Debug,
{
    pub fn new() -> TranslationModelBuilder<S, T> {
        TranslationModelBuilder {
            model_type: None,
            source_languages: None,
            target_languages: None,
            device: None,
            model_size: None,
        }
    }

    pub fn with_device(&mut self, device: Device) -> &mut Self {
        self.device = Some(device);
        self
    }

    pub fn with_model_type(&mut self, model_type: ModelType) -> &mut Self {
        self.model_type = Some(model_type);
        self
    }

    pub fn with_medium_model(&mut self) -> &mut Self {
        if let Some(model_type) = self.model_type {
            if model_type != ModelType::Marian {
                eprintln!(
                    "Model selection overwritten: was {:?}, replaced by {:?}",
                    self.model_type.unwrap(),
                    ModelType::Marian
                );
            }
        }
        self.model_type = Some(ModelType::Marian);
        self.model_size = Some(ModelSize::Medium);
        self
    }

    pub fn with_large_model(&mut self) -> &mut Self {
        if let Some(model_type) = self.model_type {
            if model_type != ModelType::M2M100 {
                eprintln!(
                    "Model selection overwritten: was {:?}, replaced by {:?}",
                    self.model_type.unwrap(),
                    ModelType::M2M100
                );
            }
        }
        self.model_type = Some(ModelType::M2M100);
        self.model_size = Some(ModelSize::Large);
        self
    }

    pub fn with_xlarge_model(&mut self) -> &mut Self {
        if let Some(model_type) = self.model_type {
            if model_type != ModelType::M2M100 {
                eprintln!(
                    "Model selection overwritten: was {:?}, replaced by {:?}",
                    self.model_type.unwrap(),
                    ModelType::M2M100
                );
            }
        }
        self.model_type = Some(ModelType::M2M100);
        self.model_size = Some(ModelSize::XLarge);
        self
    }

    pub fn with_source_languages(&mut self, source_languages: S) -> &mut Self {
        self.source_languages = Some(source_languages);
        self
    }

    pub fn with_target_languages(&mut self, target_languages: T) -> &mut Self {
        self.target_languages = Some(target_languages);
        self
    }

    fn get_default_model(
        &self,
        source_languages: Option<&S>,
        target_languages: Option<&T>,
    ) -> Result<TranslationResources, RustBertError> {
        Ok(
            match self.get_marian_model(source_languages, target_languages) {
                Ok(marian_resources) => marian_resources,
                Err(_) => match self.model_size {
                    Some(value) if value == ModelSize::XLarge => {
                        self.get_m2m100_xlarge_resources(source_languages, target_languages)?
                    }
                    _ => self.get_m2m100_large_resources(source_languages, target_languages)?,
                },
            },
        )
    }

    fn get_marian_model(
        &self,
        source_languages: Option<&S>,
        target_languages: Option<&T>,
    ) -> Result<TranslationResources, RustBertError> {
        let (resources, source_languages, target_languages) =
            if let (Some(source_languages), Some(target_languages)) =
                (source_languages, target_languages)
            {
                match (source_languages.as_ref(), target_languages.as_ref()) {
                    ([Language::English], [Language::German]) => (
                        (
                            MarianModelResources::ENGLISH2GERMAN,
                            MarianConfigResources::ENGLISH2GERMAN,
                            MarianVocabResources::ENGLISH2GERMAN,
                            MarianSpmResources::ENGLISH2GERMAN,
                        ),
                        MarianSourceLanguages::ENGLISH2GERMAN
                            .iter()
                            .cloned()
                            .collect(),
                        MarianTargetLanguages::ENGLISH2GERMAN
                            .iter()
                            .cloned()
                            .collect(),
                    ),
                    ([Language::English], [Language::Russian]) => (
                        (
                            MarianModelResources::ENGLISH2RUSSIAN,
                            MarianConfigResources::ENGLISH2RUSSIAN,
                            MarianVocabResources::ENGLISH2RUSSIAN,
                            MarianSpmResources::ENGLISH2RUSSIAN,
                        ),
                        MarianSourceLanguages::ENGLISH2RUSSIAN
                            .iter()
                            .cloned()
                            .collect(),
                        MarianTargetLanguages::ENGLISH2RUSSIAN
                            .iter()
                            .cloned()
                            .collect(),
                    ),
                    ([Language::English], [Language::Dutch]) => (
                        (
                            MarianModelResources::ENGLISH2DUTCH,
                            MarianConfigResources::ENGLISH2DUTCH,
                            MarianVocabResources::ENGLISH2DUTCH,
                            MarianSpmResources::ENGLISH2DUTCH,
                        ),
                        MarianSourceLanguages::ENGLISH2DUTCH
                            .iter()
                            .cloned()
                            .collect(),
                        MarianTargetLanguages::ENGLISH2DUTCH
                            .iter()
                            .cloned()
                            .collect(),
                    ),
                    ([Language::English], [Language::ChineseMandarin]) => (
                        (
                            MarianModelResources::ENGLISH2CHINESE,
                            MarianConfigResources::ENGLISH2CHINESE,
                            MarianVocabResources::ENGLISH2CHINESE,
                            MarianSpmResources::ENGLISH2CHINESE,
                        ),
                        MarianSourceLanguages::ENGLISH2CHINESE
                            .iter()
                            .cloned()
                            .collect(),
                        MarianTargetLanguages::ENGLISH2CHINESE
                            .iter()
                            .cloned()
                            .collect(),
                    ),
                    ([Language::English], [Language::Swedish]) => (
                        (
                            MarianModelResources::ENGLISH2SWEDISH,
                            MarianConfigResources::ENGLISH2SWEDISH,
                            MarianVocabResources::ENGLISH2SWEDISH,
                            MarianSpmResources::ENGLISH2SWEDISH,
                        ),
                        MarianSourceLanguages::ENGLISH2SWEDISH
                            .iter()
                            .cloned()
                            .collect(),
                        MarianTargetLanguages::ENGLISH2SWEDISH
                            .iter()
                            .cloned()
                            .collect(),
                    ),
                    ([Language::English], [Language::Arabic]) => (
                        (
                            MarianModelResources::ENGLISH2ARABIC,
                            MarianConfigResources::ENGLISH2ARABIC,
                            MarianVocabResources::ENGLISH2ARABIC,
                            MarianSpmResources::ENGLISH2ARABIC,
                        ),
                        MarianSourceLanguages::ENGLISH2ARABIC
                            .iter()
                            .cloned()
                            .collect(),
                        MarianTargetLanguages::ENGLISH2ARABIC
                            .iter()
                            .cloned()
                            .collect(),
                    ),
                    ([Language::English], [Language::Hindi]) => (
                        (
                            MarianModelResources::ENGLISH2HINDI,
                            MarianConfigResources::ENGLISH2HINDI,
                            MarianVocabResources::ENGLISH2HINDI,
                            MarianSpmResources::ENGLISH2HINDI,
                        ),
                        MarianSourceLanguages::ENGLISH2HINDI
                            .iter()
                            .cloned()
                            .collect(),
                        MarianTargetLanguages::ENGLISH2HINDI
                            .iter()
                            .cloned()
                            .collect(),
                    ),
                    ([Language::English], [Language::Hebrew]) => (
                        (
                            MarianModelResources::ENGLISH2HEBREW,
                            MarianConfigResources::ENGLISH2HEBREW,
                            MarianVocabResources::ENGLISH2HEBREW,
                            MarianSpmResources::ENGLISH2HEBREW,
                        ),
                        MarianSourceLanguages::ENGLISH2HEBREW
                            .iter()
                            .cloned()
                            .collect(),
                        MarianTargetLanguages::ENGLISH2HEBREW
                            .iter()
                            .cloned()
                            .collect(),
                    ),
                    ([Language::German], [Language::English]) => (
                        (
                            MarianModelResources::GERMAN2ENGLISH,
                            MarianConfigResources::GERMAN2ENGLISH,
                            MarianVocabResources::GERMAN2ENGLISH,
                            MarianSpmResources::GERMAN2ENGLISH,
                        ),
                        MarianSourceLanguages::GERMAN2ENGLISH
                            .iter()
                            .cloned()
                            .collect(),
                        MarianTargetLanguages::GERMAN2ENGLISH
                            .iter()
                            .cloned()
                            .collect(),
                    ),
                    ([Language::Russian], [Language::English]) => (
                        (
                            MarianModelResources::RUSSIAN2ENGLISH,
                            MarianConfigResources::RUSSIAN2ENGLISH,
                            MarianVocabResources::RUSSIAN2ENGLISH,
                            MarianSpmResources::RUSSIAN2ENGLISH,
                        ),
                        MarianSourceLanguages::RUSSIAN2ENGLISH
                            .iter()
                            .cloned()
                            .collect(),
                        MarianTargetLanguages::RUSSIAN2ENGLISH
                            .iter()
                            .cloned()
                            .collect(),
                    ),
                    ([Language::Dutch], [Language::English]) => (
                        (
                            MarianModelResources::DUTCH2ENGLISH,
                            MarianConfigResources::DUTCH2ENGLISH,
                            MarianVocabResources::DUTCH2ENGLISH,
                            MarianSpmResources::DUTCH2ENGLISH,
                        ),
                        MarianSourceLanguages::DUTCH2ENGLISH
                            .iter()
                            .cloned()
                            .collect(),
                        MarianTargetLanguages::DUTCH2ENGLISH
                            .iter()
                            .cloned()
                            .collect(),
                    ),
                    ([Language::ChineseMandarin], [Language::English]) => (
                        (
                            MarianModelResources::CHINESE2ENGLISH,
                            MarianConfigResources::CHINESE2ENGLISH,
                            MarianVocabResources::CHINESE2ENGLISH,
                            MarianSpmResources::CHINESE2ENGLISH,
                        ),
                        MarianSourceLanguages::CHINESE2ENGLISH
                            .iter()
                            .cloned()
                            .collect(),
                        MarianTargetLanguages::CHINESE2ENGLISH
                            .iter()
                            .cloned()
                            .collect(),
                    ),
                    ([Language::Swedish], [Language::English]) => (
                        (
                            MarianModelResources::SWEDISH2ENGLISH,
                            MarianConfigResources::SWEDISH2ENGLISH,
                            MarianVocabResources::SWEDISH2ENGLISH,
                            MarianSpmResources::SWEDISH2ENGLISH,
                        ),
                        MarianSourceLanguages::SWEDISH2ENGLISH
                            .iter()
                            .cloned()
                            .collect(),
                        MarianTargetLanguages::SWEDISH2ENGLISH
                            .iter()
                            .cloned()
                            .collect(),
                    ),
                    ([Language::Arabic], [Language::English]) => (
                        (
                            MarianModelResources::ARABIC2ENGLISH,
                            MarianConfigResources::ARABIC2ENGLISH,
                            MarianVocabResources::ARABIC2ENGLISH,
                            MarianSpmResources::ARABIC2ENGLISH,
                        ),
                        MarianSourceLanguages::ARABIC2ENGLISH
                            .iter()
                            .cloned()
                            .collect(),
                        MarianTargetLanguages::ARABIC2ENGLISH
                            .iter()
                            .cloned()
                            .collect(),
                    ),
                    ([Language::Hindi], [Language::English]) => (
                        (
                            MarianModelResources::HINDI2ENGLISH,
                            MarianConfigResources::HINDI2ENGLISH,
                            MarianVocabResources::HINDI2ENGLISH,
                            MarianSpmResources::HINDI2ENGLISH,
                        ),
                        MarianSourceLanguages::HINDI2ENGLISH
                            .iter()
                            .cloned()
                            .collect(),
                        MarianTargetLanguages::HINDI2ENGLISH
                            .iter()
                            .cloned()
                            .collect(),
                    ),
                    ([Language::Hebrew], [Language::English]) => (
                        (
                            MarianModelResources::HEBREW2ENGLISH,
                            MarianConfigResources::HEBREW2ENGLISH,
                            MarianVocabResources::HEBREW2ENGLISH,
                            MarianSpmResources::HEBREW2ENGLISH,
                        ),
                        MarianSourceLanguages::HEBREW2ENGLISH
                            .iter()
                            .cloned()
                            .collect(),
                        MarianTargetLanguages::HEBREW2ENGLISH
                            .iter()
                            .cloned()
                            .collect(),
                    ),
                    ([Language::English], languages)
                        if languages
                            .iter()
                            .all(|lang| MarianTargetLanguages::ENGLISH2ROMANCE.contains(lang)) =>
                    {
                        (
                            (
                                MarianModelResources::ENGLISH2ROMANCE,
                                MarianConfigResources::ENGLISH2ROMANCE,
                                MarianVocabResources::ENGLISH2ROMANCE,
                                MarianSpmResources::ENGLISH2ROMANCE,
                            ),
                            MarianSourceLanguages::ENGLISH2ROMANCE
                                .iter()
                                .cloned()
                                .collect(),
                            MarianTargetLanguages::ENGLISH2ROMANCE
                                .iter()
                                .cloned()
                                .collect(),
                        )
                    }
                    (languages, [Language::English])
                        if languages
                            .iter()
                            .all(|lang| MarianSourceLanguages::ROMANCE2ENGLISH.contains(lang)) =>
                    {
                        (
                            (
                                MarianModelResources::ENGLISH2ROMANCE,
                                MarianConfigResources::ENGLISH2ROMANCE,
                                MarianVocabResources::ENGLISH2ROMANCE,
                                MarianSpmResources::ENGLISH2ROMANCE,
                            ),
                            MarianSourceLanguages::ENGLISH2ROMANCE
                                .iter()
                                .cloned()
                                .collect(),
                            MarianTargetLanguages::ENGLISH2ROMANCE
                                .iter()
                                .cloned()
                                .collect(),
                        )
                    }
                    (_, _) => {
                        return Err(RustBertError::InvalidConfigurationError(format!(
                            "No Pretrained Marian configuration found for {:?} to {:?} translation",
                            source_languages, target_languages
                        )));
                    }
                }
            } else {
                return Err(RustBertError::InvalidConfigurationError(format!(
                    "Source and target languages must be provided for Marian models"
                )));
            };

        Ok(TranslationResources {
            model_type: ModelType::Marian,
            model_resource: Resource::Remote(RemoteResource::from_pretrained(resources.0)),
            config_resource: Resource::Remote(RemoteResource::from_pretrained(resources.1)),
            vocab_resource: Resource::Remote(RemoteResource::from_pretrained(resources.2)),
            merges_resource: Resource::Remote(RemoteResource::from_pretrained(resources.3)),
            source_languages,
            target_languages,
        })
    }

    fn get_mbart50_resources(
        &self,
        source_languages: Option<&S>,
        target_languages: Option<&T>,
    ) -> Result<TranslationResources, RustBertError> {
        if let Some(source_languages) = source_languages {
            if !source_languages
                .as_ref()
                .iter()
                .all(|lang| MBartSourceLanguages::MBART50_MANY_TO_MANY.contains(lang))
            {
                return Err(RustBertError::ValueError(format!(
                    "{:?} not in list of supported languages: {:?}",
                    source_languages.as_ref(),
                    MBartSourceLanguages::MBART50_MANY_TO_MANY
                )));
            }
        }

        if let Some(target_languages) = target_languages {
            if !target_languages
                .as_ref()
                .iter()
                .all(|lang| MBartTargetLanguages::MBART50_MANY_TO_MANY.contains(lang))
            {
                return Err(RustBertError::ValueError(format!(
                    "{:?} not in list of supported languages: {:?}",
                    target_languages,
                    MBartTargetLanguages::MBART50_MANY_TO_MANY
                )));
            }
        }

        Ok(TranslationResources {
            model_type: ModelType::MBart,
            model_resource: Resource::Remote(RemoteResource::from_pretrained(
                MBartModelResources::MBART50_MANY_TO_MANY,
            )),
            config_resource: Resource::Remote(RemoteResource::from_pretrained(
                MBartConfigResources::MBART50_MANY_TO_MANY,
            )),
            vocab_resource: Resource::Remote(RemoteResource::from_pretrained(
                MBartVocabResources::MBART50_MANY_TO_MANY,
            )),
            merges_resource: Resource::Remote(RemoteResource::from_pretrained(
                MBartVocabResources::MBART50_MANY_TO_MANY,
            )),
            source_languages: MBartSourceLanguages::MBART50_MANY_TO_MANY
                .iter()
                .cloned()
                .collect(),
            target_languages: MBartTargetLanguages::MBART50_MANY_TO_MANY
                .iter()
                .cloned()
                .collect(),
        })
    }

    fn get_m2m100_large_resources(
        &self,
        source_languages: Option<&S>,
        target_languages: Option<&T>,
    ) -> Result<TranslationResources, RustBertError> {
        if let Some(source_languages) = source_languages {
            if !source_languages
                .as_ref()
                .iter()
                .all(|lang| M2M100SourceLanguages::M2M100_418M.contains(lang))
            {
                return Err(RustBertError::ValueError(format!(
                    "{:?} not in list of supported languages: {:?}",
                    source_languages.as_ref(),
                    M2M100SourceLanguages::M2M100_418M
                )));
            }
        }

        if let Some(target_languages) = target_languages {
            if !target_languages
                .as_ref()
                .iter()
                .all(|lang| M2M100TargetLanguages::M2M100_418M.contains(lang))
            {
                return Err(RustBertError::ValueError(format!(
                    "{:?} not in list of supported languages: {:?}",
                    target_languages,
                    M2M100TargetLanguages::M2M100_418M
                )));
            }
        }

        Ok(TranslationResources {
            model_type: ModelType::M2M100,
            model_resource: Resource::Remote(RemoteResource::from_pretrained(
                M2M100ModelResources::M2M100_418M,
            )),
            config_resource: Resource::Remote(RemoteResource::from_pretrained(
                M2M100ConfigResources::M2M100_418M,
            )),
            vocab_resource: Resource::Remote(RemoteResource::from_pretrained(
                M2M100VocabResources::M2M100_418M,
            )),
            merges_resource: Resource::Remote(RemoteResource::from_pretrained(
                M2M100MergesResources::M2M100_418M,
            )),
            source_languages: M2M100SourceLanguages::M2M100_418M.iter().cloned().collect(),
            target_languages: M2M100TargetLanguages::M2M100_418M.iter().cloned().collect(),
        })
    }

    fn get_m2m100_xlarge_resources(
        &self,
        source_languages: Option<&S>,
        target_languages: Option<&T>,
    ) -> Result<TranslationResources, RustBertError> {
        if let Some(source_languages) = source_languages {
            if !source_languages
                .as_ref()
                .iter()
                .all(|lang| M2M100SourceLanguages::M2M100_1_2B.contains(lang))
            {
                return Err(RustBertError::ValueError(format!(
                    "{:?} not in list of supported languages: {:?}",
                    source_languages.as_ref(),
                    M2M100SourceLanguages::M2M100_1_2B
                )));
            }
        }

        if let Some(target_languages) = target_languages {
            if !target_languages
                .as_ref()
                .iter()
                .all(|lang| M2M100TargetLanguages::M2M100_1_2B.contains(lang))
            {
                return Err(RustBertError::ValueError(format!(
                    "{:?} not in list of supported languages: {:?}",
                    target_languages,
                    M2M100TargetLanguages::M2M100_1_2B
                )));
            }
        }

        Ok(TranslationResources {
            model_type: ModelType::M2M100,
            model_resource: Resource::Remote(RemoteResource::from_pretrained(
                M2M100ModelResources::M2M100_1_2B,
            )),
            config_resource: Resource::Remote(RemoteResource::from_pretrained(
                M2M100ConfigResources::M2M100_1_2B,
            )),
            vocab_resource: Resource::Remote(RemoteResource::from_pretrained(
                M2M100VocabResources::M2M100_1_2B,
            )),
            merges_resource: Resource::Remote(RemoteResource::from_pretrained(
                M2M100MergesResources::M2M100_1_2B,
            )),
            source_languages: M2M100SourceLanguages::M2M100_1_2B.iter().cloned().collect(),
            target_languages: M2M100TargetLanguages::M2M100_1_2B.iter().cloned().collect(),
        })
    }

    pub fn create_model(&self) -> Result<TranslationModel, RustBertError> {
        let device = self.device.unwrap_or_else(|| Device::cuda_if_available());

        let translation_resources = match (
            &self.model_type,
            &self.source_languages,
            &self.target_languages,
        ) {
            (Some(ModelType::M2M100), source_languages, target_languages) => {
                match self.model_size {
                    Some(value) if value == ModelSize::XLarge => self.get_m2m100_xlarge_resources(
                        source_languages.as_ref(),
                        target_languages.as_ref(),
                    )?,
                    _ => self.get_m2m100_large_resources(
                        source_languages.as_ref(),
                        target_languages.as_ref(),
                    )?,
                }
            }
            (Some(ModelType::MBart), source_languages, target_languages) => {
                self.get_mbart50_resources(source_languages.as_ref(), target_languages.as_ref())?
            }
            (Some(ModelType::Marian), source_languages, target_languages) => {
                self.get_marian_model(source_languages.as_ref(), target_languages.as_ref())?
            }
            (None, source_languages, target_languages) => {
                self.get_default_model(source_languages.as_ref(), target_languages.as_ref())?
            }
            (_, None, None) | (_, _, None) | (_, None, _) => {
                return Err(RustBertError::InvalidConfigurationError(format!(
                    "Source and target languages must be specified for {:?}",
                    self.model_type.unwrap()
                )));
            }
            (Some(model_type), _, _) => {
                return Err(RustBertError::InvalidConfigurationError(format!(
                    "Automated translation model builder not implemented for {:?}",
                    model_type
                )));
            }
        };

        let translation_config = TranslationConfig::new(
            translation_resources.model_type,
            translation_resources.model_resource,
            translation_resources.config_resource,
            translation_resources.vocab_resource,
            translation_resources.merges_resource,
            translation_resources.source_languages,
            translation_resources.target_languages,
            device,
        );
        TranslationModel::new(translation_config)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::marian::{
        MarianConfigResources, MarianModelResources, MarianSourceLanguages, MarianTargetLanguages,
        MarianVocabResources,
    };
    use crate::resources::RemoteResource;

    #[test]
    #[ignore] // no need to run, compilation is enough to verify it is Send
    fn test() {
        use rust_bert::marian::{
            MarianConfigResources, MarianModelResources, MarianSourceLanguages,
            MarianTargetLanguages, MarianVocabResources,
        };
        use rust_bert::pipelines::common::ModelType;
        use rust_bert::pipelines::translation::{OldLanguage, TranslationConfig};
        use rust_bert::resources::{RemoteResource, Resource};
        use tch::Device;

        let model_resource = Resource::Remote(RemoteResource::from_pretrained(
            MarianModelResources::ROMANCE2ENGLISH,
        ));
        let config_resource = Resource::Remote(RemoteResource::from_pretrained(
            MarianConfigResources::ROMANCE2ENGLISH,
        ));
        let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(
            MarianVocabResources::ROMANCE2ENGLISH,
        ));

        let source_languages = MarianSourceLanguages::ROMANCE2ENGLISH.iter().collect();
        let target_languages = MarianTargetLanguages::ROMANCE2ENGLISH.iter().collect();

        let translation_config = TranslationConfig::new(
            ModelType::Marian,
            model_resource,
            config_resource,
            vocab_resource.clone(),
            vocab_resource,
            source_languages,
            target_languages,
            device: Device::cuda_if_available(),
        );
        let _: Box<dyn Send> = Box::new(TranslationModel::new(config));
    }
}
