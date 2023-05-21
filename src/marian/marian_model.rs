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

use crate::bart::{BartConfig, BartModel, BartModelOutput, LayerState};
use crate::pipelines::common::{ModelType, TokenizerOption};
use crate::pipelines::generation_utils::private_generation_utils::{
    PreparedInput, PrivateLanguageGenerator,
};
use crate::pipelines::generation_utils::{Cache, GenerateConfig, LMModelOutput, LanguageGenerator};
use crate::pipelines::translation::Language;
use crate::{Config, RustBertError};
use rust_tokenizers::tokenizer::TruncationStrategy;
use std::borrow::Borrow;
use tch::nn::Init;
use tch::{nn, Kind, Tensor};

/// # Marian Pretrained model weight files
pub struct MarianModelResources;

/// # Marian Pretrained model config files
pub struct MarianConfigResources;

/// # Marian Pretrained model vocab files
pub struct MarianVocabResources;

/// # Marian Pretrained sentence piece model files
pub struct MarianSpmResources;

/// # Marian source languages pre-sets
pub struct MarianSourceLanguages;

/// # Marian target languages pre-sets
pub struct MarianTargetLanguages;

/// # Marian translation model pre-sets
pub struct MarianModelPreset;

impl MarianModelResources {
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>. Modified with conversion to C-array format.
    pub const ENGLISH2ROMANCE: (&'static str, &'static str) = (
        "marian-mt-en-ROMANCE/model",
        "https://huggingface.co/Helsinki-NLP/opus-mt-en-ROMANCE/resolve/main/rust_model.ot",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>. Modified with conversion to C-array format.
    pub const ROMANCE2ENGLISH: (&'static str, &'static str) = (
        "marian-mt-ROMANCE-en/model",
        "https://huggingface.co/Helsinki-NLP/opus-mt-ROMANCE-en/resolve/main/rust_model.ot",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>. Modified with conversion to C-array format.
    pub const ENGLISH2GERMAN: (&'static str, &'static str) = (
        "marian-mt-en-de/model",
        "https://huggingface.co/Helsinki-NLP/opus-mt-en-de/resolve/main/rust_model.ot",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>. Modified with conversion to C-array format.
    pub const GERMAN2ENGLISH: (&'static str, &'static str) = (
        "marian-mt-de-en/model",
        "https://huggingface.co/Helsinki-NLP/opus-mt-de-en/resolve/main/rust_model.ot",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>. Modified with conversion to C-array format.
    pub const ENGLISH2RUSSIAN: (&'static str, &'static str) = (
        "marian-mt-en-ru/model",
        "https://huggingface.co/Helsinki-NLP/opus-mt-en-ru/resolve/main/rust_model.ot",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>. Modified with conversion to C-array format.
    pub const RUSSIAN2ENGLISH: (&'static str, &'static str) = (
        "marian-mt-ru-en/model",
        "https://huggingface.co/Helsinki-NLP/opus-mt-ru-en/resolve/main/rust_model.ot",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>. Modified with conversion to C-array format.
    pub const FRENCH2GERMAN: (&'static str, &'static str) = (
        "marian-mt-fr-de/model",
        "https://huggingface.co/Helsinki-NLP/opus-mt-fr-de/resolve/main/rust_model.ot",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>. Modified with conversion to C-array format.
    pub const GERMAN2FRENCH: (&'static str, &'static str) = (
        "marian-mt-de-fr/model",
        "https://huggingface.co/Helsinki-NLP/opus-mt-de-fr/resolve/main/rust_model.ot",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>. Modified with conversion to C-array format.
    pub const ENGLISH2DUTCH: (&'static str, &'static str) = (
        "marian-mt-en-nl/model",
        "https://huggingface.co/Helsinki-NLP/opus-mt-en-nl/resolve/main/rust_model.ot",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>. Modified with conversion to C-array format.
    pub const DUTCH2ENGLISH: (&'static str, &'static str) = (
        "marian-mt-nl-en/model",
        "https://huggingface.co/Helsinki-NLP/opus-mt-nl-en/resolve/main/rust_model.ot",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>. Modified with conversion to C-array format.
    pub const ENGLISH2CHINESE: (&'static str, &'static str) = (
        "marian-mt-en-zh/model",
        "https://huggingface.co/Helsinki-NLP/opus-mt-en-zh/resolve/main/rust_model.ot",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>. Modified with conversion to C-array format.
    pub const CHINESE2ENGLISH: (&'static str, &'static str) = (
        "marian-mt-zh-en/model",
        "https://huggingface.co/Helsinki-NLP/opus-mt-zh-en/resolve/main/rust_model.ot",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>. Modified with conversion to C-array format.
    pub const ENGLISH2SWEDISH: (&'static str, &'static str) = (
        "marian-mt-en-sv/model",
        "https://huggingface.co/Helsinki-NLP/opus-mt-en-sv/resolve/main/rust_model.ot",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>. Modified with conversion to C-array format.
    pub const SWEDISH2ENGLISH: (&'static str, &'static str) = (
        "marian-mt-sv-en/model",
        "https://huggingface.co/Helsinki-NLP/opus-mt-sv-en/resolve/main/rust_model.ot",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>. Modified with conversion to C-array format.
    pub const ARABIC2ENGLISH: (&'static str, &'static str) = (
        "marian-mt-ar-en/model",
        "https://huggingface.co/Helsinki-NLP/opus-mt-ar-en/resolve/main/rust_model.ot",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>. Modified with conversion to C-array format.
    pub const ENGLISH2ARABIC: (&'static str, &'static str) = (
        "marian-mt-en-ar/model",
        "https://huggingface.co/Helsinki-NLP/opus-mt-en-ar/resolve/main/rust_model.ot",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>. Modified with conversion to C-array format.
    pub const HINDI2ENGLISH: (&'static str, &'static str) = (
        "marian-mt-hi-en/model",
        "https://huggingface.co/Helsinki-NLP/opus-mt-hi-en/resolve/main/rust_model.ot",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>. Modified with conversion to C-array format.
    pub const ENGLISH2HINDI: (&'static str, &'static str) = (
        "marian-mt-en-hi/model",
        "https://huggingface.co/Helsinki-NLP/opus-mt-en-hi/resolve/main/rust_model.ot",
    );
    /// Shared under Apache 2.0 License license at <https://huggingface.co/tiedeman/opus-mt-he-en>. Modified with conversion to C-array format.
    pub const HEBREW2ENGLISH: (&'static str, &'static str) = (
        "marian-mt-he-en/model",
        "https://huggingface.co/Helsinki-NLP/opus-mt-he-en/resolve/main/rust_model.ot",
    );
    /// Shared under Apache 2.0 License license at <https://huggingface.co/tiedeman/opus-mt-en-he>. Modified with conversion to C-array format.
    pub const ENGLISH2HEBREW: (&'static str, &'static str) = (
        "marian-mt-en-he/model",
        "https://huggingface.co/Helsinki-NLP/opus-mt-en-he/resolve/main/rust_model.ot",
    );
}

impl MarianConfigResources {
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>.
    pub const ENGLISH2ROMANCE: (&'static str, &'static str) = (
        "marian-mt-en-ROMANCE/config",
        "https://huggingface.co/Helsinki-NLP/opus-mt-en-ROMANCE/resolve/main/config.json",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>.
    pub const ROMANCE2ENGLISH: (&'static str, &'static str) = (
        "marian-mt-ROMANCE-en/config",
        "https://huggingface.co/Helsinki-NLP/opus-mt-ROMANCE-en/resolve/main/config.json",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>.
    pub const ENGLISH2GERMAN: (&'static str, &'static str) = (
        "marian-mt-en-de/config",
        "https://huggingface.co/Helsinki-NLP/opus-mt-en-de/resolve/main/config.json",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>.
    pub const GERMAN2ENGLISH: (&'static str, &'static str) = (
        "marian-mt-de-en/config",
        "https://huggingface.co/Helsinki-NLP/opus-mt-de-en/resolve/main/config.json",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>.
    pub const ENGLISH2RUSSIAN: (&'static str, &'static str) = (
        "marian-mt-en-ru/config",
        "https://huggingface.co/Helsinki-NLP/opus-mt-en-ru/resolve/main/config.json",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>.
    pub const RUSSIAN2ENGLISH: (&'static str, &'static str) = (
        "marian-mt-ru-en/config",
        "https://huggingface.co/Helsinki-NLP/opus-mt-ru-en/resolve/main/config.json",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>.
    pub const FRENCH2GERMAN: (&'static str, &'static str) = (
        "marian-mt-fr-de/config",
        "https://huggingface.co/Helsinki-NLP/opus-mt-fr-de/resolve/main/config.json",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>.
    pub const GERMAN2FRENCH: (&'static str, &'static str) = (
        "marian-mt-de-fr/config",
        "https://huggingface.co/Helsinki-NLP/opus-mt-de-fr/resolve/main/config.json",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>.
    pub const ENGLISH2DUTCH: (&'static str, &'static str) = (
        "marian-mt-en-nl/config",
        "https://huggingface.co/Helsinki-NLP/opus-mt-en-nl/resolve/main/config.json",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>.
    pub const DUTCH2ENGLISH: (&'static str, &'static str) = (
        "marian-mt-nl-en/config",
        "https://huggingface.co/Helsinki-NLP/opus-mt-nl-en/resolve/main/config.json",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>.
    pub const CHINESE2ENGLISH: (&'static str, &'static str) = (
        "marian-mt-zh-en/config",
        "https://huggingface.co/Helsinki-NLP/opus-mt-zh-en/resolve/main/config.json",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>.
    pub const ENGLISH2CHINESE: (&'static str, &'static str) = (
        "marian-mt-en-zh/config",
        "https://huggingface.co/Helsinki-NLP/opus-mt-en-zh/resolve/main/config.json",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>.
    pub const ENGLISH2SWEDISH: (&'static str, &'static str) = (
        "marian-mt-en-sv/config",
        "https://huggingface.co/Helsinki-NLP/opus-mt-en-sv/resolve/main/config.json",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>.
    pub const SWEDISH2ENGLISH: (&'static str, &'static str) = (
        "marian-mt-sv-en/config",
        "https://huggingface.co/Helsinki-NLP/opus-mt-sv-en/resolve/main/config.json",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>.
    pub const ARABIC2ENGLISH: (&'static str, &'static str) = (
        "marian-mt-ar-en/config",
        "https://huggingface.co/Helsinki-NLP/opus-mt-ar-en/resolve/main/config.json",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>.
    pub const ENGLISH2ARABIC: (&'static str, &'static str) = (
        "marian-mt-en-ar/config",
        "https://huggingface.co/Helsinki-NLP/opus-mt-en-ar/resolve/main/config.json",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>.
    pub const HINDI2ENGLISH: (&'static str, &'static str) = (
        "marian-mt-hi-en/config",
        "https://huggingface.co/Helsinki-NLP/opus-mt-hi-en/resolve/main/config.json",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>.
    pub const ENGLISH2HINDI: (&'static str, &'static str) = (
        "marian-mt-en-hi/config",
        "https://huggingface.co/Helsinki-NLP/opus-mt-en-hi/resolve/main/config.json",
    );
    /// Shared under Apache 2.0 License license at <https://huggingface.co/tiedeman/opus-mt-he-en>. Modified with conversion to C-array format.
    pub const HEBREW2ENGLISH: (&'static str, &'static str) = (
        "marian-mt-he-en/config",
        "https://huggingface.co/Helsinki-NLP/opus-mt-he-en/resolve/main/config.json",
    );
    /// Shared under Apache 2.0 License license at <https://huggingface.co/tiedeman/opus-mt-en-he>. Modified with conversion to C-array format.
    pub const ENGLISH2HEBREW: (&'static str, &'static str) = (
        "marian-mt-en-he/config",
        "https://huggingface.co/Helsinki-NLP/opus-mt-en-he/resolve/main/config.json",
    );
}

impl MarianVocabResources {
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>.
    pub const ENGLISH2ROMANCE: (&'static str, &'static str) = (
        "marian-mt-en-ROMANCE/vocab",
        "https://huggingface.co/Helsinki-NLP/opus-mt-en-ROMANCE/resolve/main/vocab.json",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>.
    pub const ROMANCE2ENGLISH: (&'static str, &'static str) = (
        "marian-mt-ROMANCE-en/vocab",
        "https://huggingface.co/Helsinki-NLP/opus-mt-ROMANCE-en/resolve/main/vocab.json",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>.
    pub const ENGLISH2GERMAN: (&'static str, &'static str) = (
        "marian-mt-en-de/vocab",
        "https://huggingface.co/Helsinki-NLP/opus-mt-en-de/resolve/main/vocab.json",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>.
    pub const GERMAN2ENGLISH: (&'static str, &'static str) = (
        "marian-mt-de-en/vocab",
        "https://huggingface.co/Helsinki-NLP/opus-mt-de-en/resolve/main/vocab.json",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>.
    pub const ENGLISH2RUSSIAN: (&'static str, &'static str) = (
        "marian-mt-en-ru/vocab",
        "https://huggingface.co/Helsinki-NLP/opus-mt-en-ru/resolve/main/vocab.json",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>.
    pub const RUSSIAN2ENGLISH: (&'static str, &'static str) = (
        "marian-mt-ru-en/vocab",
        "https://huggingface.co/Helsinki-NLP/opus-mt-ru-en/resolve/main/vocab.json",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>.
    pub const FRENCH2GERMAN: (&'static str, &'static str) = (
        "marian-mt-fr-de/vocab",
        "https://huggingface.co/Helsinki-NLP/opus-mt-fr-de/resolve/main/vocab.json",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>.
    pub const GERMAN2FRENCH: (&'static str, &'static str) = (
        "marian-mt-de-fr/vocab",
        "https://huggingface.co/Helsinki-NLP/opus-mt-de-fr/resolve/main/vocab.json",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>.
    pub const ENGLISH2DUTCH: (&'static str, &'static str) = (
        "marian-mt-en-nl/vocab",
        "https://huggingface.co/Helsinki-NLP/opus-mt-en-nl/resolve/main/vocab.json",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>.
    pub const DUTCH2ENGLISH: (&'static str, &'static str) = (
        "marian-mt-nl-en/vocab",
        "https://huggingface.co/Helsinki-NLP/opus-mt-nl-en/resolve/main/vocab.json",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>.
    pub const CHINESE2ENGLISH: (&'static str, &'static str) = (
        "marian-mt-zh-en/vocab",
        "https://huggingface.co/Helsinki-NLP/opus-mt-zh-en/resolve/main/vocab.json",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>.
    pub const ENGLISH2CHINESE: (&'static str, &'static str) = (
        "marian-mt-en-zh/vocab",
        "https://huggingface.co/Helsinki-NLP/opus-mt-en-zh/resolve/main/vocab.json",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>.
    pub const ENGLISH2SWEDISH: (&'static str, &'static str) = (
        "marian-mt-en-sv/vocab",
        "https://huggingface.co/Helsinki-NLP/opus-mt-en-sv/resolve/main/vocab.json",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>.
    pub const SWEDISH2ENGLISH: (&'static str, &'static str) = (
        "marian-mt-sv-en/vocab",
        "https://huggingface.co/Helsinki-NLP/opus-mt-sv-en/resolve/main/vocab.json",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>.
    pub const ARABIC2ENGLISH: (&'static str, &'static str) = (
        "marian-mt-ar-en/vocab",
        "https://huggingface.co/Helsinki-NLP/opus-mt-ar-en/resolve/main/vocab.json",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>.
    pub const ENGLISH2ARABIC: (&'static str, &'static str) = (
        "marian-mt-en-ar/vocab",
        "https://huggingface.co/Helsinki-NLP/opus-mt-en-ar/resolve/main/vocab.json",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>.
    pub const HINDI2ENGLISH: (&'static str, &'static str) = (
        "marian-mt-hi-en/vocab",
        "https://huggingface.co/Helsinki-NLP/opus-mt-hi-en/resolve/main/vocab.json",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>.
    pub const ENGLISH2HINDI: (&'static str, &'static str) = (
        "marian-mt-en-hi/vocab",
        "https://huggingface.co/Helsinki-NLP/opus-mt-en-hi/resolve/main/vocab.json",
    );
    /// Shared under Apache 2.0 License license at <https://huggingface.co/tiedeman/opus-mt-he-en>. Modified with conversion to C-array format.
    pub const HEBREW2ENGLISH: (&'static str, &'static str) = (
        "marian-mt-he-en/vocab",
        "https://huggingface.co/Helsinki-NLP/opus-mt-he-en/resolve/main/vocab.json",
    );
    /// Shared under Apache 2.0 License license at <https://huggingface.co/tiedeman/opus-mt-en-he>. Modified with conversion to C-array format.
    pub const ENGLISH2HEBREW: (&'static str, &'static str) = (
        "marian-mt-en-he/vocab",
        "https://huggingface.co/Helsinki-NLP/opus-mt-en-he/resolve/main/vocab.json",
    );
}

impl MarianSpmResources {
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>.
    pub const ENGLISH2ROMANCE: (&'static str, &'static str) = (
        "marian-mt-en-ROMANCE/spiece",
        "https://huggingface.co/Helsinki-NLP/opus-mt-en-ROMANCE/resolve/main/source.spm",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>.
    pub const ROMANCE2ENGLISH: (&'static str, &'static str) = (
        "marian-mt-ROMANCE-en/spiece",
        "https://huggingface.co/Helsinki-NLP/opus-mt-ROMANCE-en/resolve/main/source.spm",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>.
    pub const ENGLISH2GERMAN: (&'static str, &'static str) = (
        "marian-mt-en-de/spiece",
        "https://huggingface.co/Helsinki-NLP/opus-mt-en-de/resolve/main/source.spm",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>.
    pub const GERMAN2ENGLISH: (&'static str, &'static str) = (
        "marian-mt-de-en/spiece",
        "https://huggingface.co/Helsinki-NLP/opus-mt-de-en/resolve/main/source.spm",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>.
    pub const ENGLISH2RUSSIAN: (&'static str, &'static str) = (
        "marian-mt-en-ru/spiece",
        "https://huggingface.co/Helsinki-NLP/opus-mt-en-ru/resolve/main/source.spm",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>.
    pub const RUSSIAN2ENGLISH: (&'static str, &'static str) = (
        "marian-mt-ru-en/spiece",
        "https://huggingface.co/Helsinki-NLP/opus-mt-ru-en/resolve/main/source.spm",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>.
    pub const FRENCH2GERMAN: (&'static str, &'static str) = (
        "marian-mt-fr-de/spiece",
        "https://huggingface.co/Helsinki-NLP/opus-mt-fr-de/resolve/main/source.spm",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>.
    pub const GERMAN2FRENCH: (&'static str, &'static str) = (
        "marian-mt-de-fr/spiece",
        "https://huggingface.co/Helsinki-NLP/opus-mt-de-fr/resolve/main/source.spm",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>.
    pub const ENGLISH2DUTCH: (&'static str, &'static str) = (
        "marian-mt-en-nl/spiece",
        "https://huggingface.co/Helsinki-NLP/opus-mt-en-nl/resolve/main/source.spm",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>.
    pub const DUTCH2ENGLISH: (&'static str, &'static str) = (
        "marian-mt-nl-en/spiece",
        "https://huggingface.co/Helsinki-NLP/opus-mt-nl-en/resolve/main/source.spm",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>.
    pub const CHINESE2ENGLISH: (&'static str, &'static str) = (
        "marian-mt-zh-en/spiece",
        "https://huggingface.co/Helsinki-NLP/opus-mt-zh-en/resolve/main/source.spm",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>.
    pub const ENGLISH2CHINESE: (&'static str, &'static str) = (
        "marian-mt-en-zh/spiece",
        "https://huggingface.co/Helsinki-NLP/opus-mt-en-zh/resolve/main/source.spm",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>.
    pub const ENGLISH2SWEDISH: (&'static str, &'static str) = (
        "marian-mt-en-sv/spiece",
        "https://huggingface.co/Helsinki-NLP/opus-mt-en-sv/resolve/main/source.spm",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>.
    pub const SWEDISH2ENGLISH: (&'static str, &'static str) = (
        "marian-mt-sv-en/spiece",
        "https://huggingface.co/Helsinki-NLP/opus-mt-sv-en/resolve/main/source.spm",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>.
    pub const ARABIC2ENGLISH: (&'static str, &'static str) = (
        "marian-mt-ar-en/spiece",
        "https://huggingface.co/Helsinki-NLP/opus-mt-ar-en/resolve/main/source.spm",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>.
    pub const ENGLISH2ARABIC: (&'static str, &'static str) = (
        "marian-mt-en-ar/spiece",
        "https://huggingface.co/Helsinki-NLP/opus-mt-en-ar/resolve/main/source.spm",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>.
    pub const HINDI2ENGLISH: (&'static str, &'static str) = (
        "marian-mt-hi-en/spiece",
        "https://huggingface.co/Helsinki-NLP/opus-mt-hi-en/resolve/main/source.spm",
    );
    /// Shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>.
    pub const ENGLISH2HINDI: (&'static str, &'static str) = (
        "marian-mt-en-hi/spiece",
        "https://huggingface.co/Helsinki-NLP/opus-mt-en-hi/resolve/main/source.spm",
    );
    /// Shared under Apache 2.0 License license at <https://huggingface.co/tiedeman/opus-mt-he-en>. Modified with conversion to C-array format.
    pub const HEBREW2ENGLISH: (&'static str, &'static str) = (
        "marian-mt-he-en/spiece",
        "https://huggingface.co/Helsinki-NLP/opus-mt-he-en/resolve/main/source.spm",
    );
    /// Shared under Apache 2.0 License license at <https://huggingface.co/tiedeman/opus-mt-en-he>. Modified with conversion to C-array format.
    pub const ENGLISH2HEBREW: (&'static str, &'static str) = (
        "marian-mt-en-he/spiece",
        "https://huggingface.co/Helsinki-NLP/opus-mt-en-he/resolve/main/source.spm",
    );
}

impl MarianSourceLanguages {
    pub const ENGLISH2ROMANCE: [Language; 1] = [Language::English];
    pub const ENGLISH2GERMAN: [Language; 1] = [Language::English];
    pub const ENGLISH2RUSSIAN: [Language; 1] = [Language::English];
    pub const ENGLISH2DUTCH: [Language; 1] = [Language::English];
    pub const ENGLISH2CHINESE: [Language; 1] = [Language::English];
    pub const ENGLISH2SWEDISH: [Language; 1] = [Language::English];
    pub const ENGLISH2ARABIC: [Language; 1] = [Language::English];
    pub const ENGLISH2HINDI: [Language; 1] = [Language::English];
    pub const ENGLISH2HEBREW: [Language; 1] = [Language::English];
    pub const ROMANCE2ENGLISH: [Language; 7] = [
        Language::French,
        Language::Spanish,
        Language::Italian,
        Language::Catalan,
        Language::Romanian,
        Language::Portuguese,
        Language::Occitan,
    ];
    pub const GERMAN2ENGLISH: [Language; 1] = [Language::German];
    pub const RUSSIAN2ENGLISH: [Language; 1] = [Language::Russian];
    pub const DUTCH2ENGLISH: [Language; 1] = [Language::Dutch];
    pub const CHINESE2ENGLISH: [Language; 1] = [Language::ChineseMandarin];
    pub const SWEDISH2ENGLISH: [Language; 1] = [Language::Swedish];
    pub const ARABIC2ENGLISH: [Language; 1] = [Language::Arabic];
    pub const HINDI2ENGLISH: [Language; 1] = [Language::Hindi];
    pub const HEBREW2ENGLISH: [Language; 1] = [Language::Hebrew];
    pub const FRENCH2GERMAN: [Language; 1] = [Language::French];
    pub const GERMAN2FRENCH: [Language; 1] = [Language::German];
}

impl MarianTargetLanguages {
    pub const ENGLISH2ROMANCE: [Language; 7] = [
        Language::French,
        Language::Spanish,
        Language::Italian,
        Language::Catalan,
        Language::Romanian,
        Language::Portuguese,
        Language::Occitan,
    ];
    pub const ENGLISH2GERMAN: [Language; 1] = [Language::German];
    pub const ENGLISH2RUSSIAN: [Language; 1] = [Language::Russian];
    pub const ENGLISH2DUTCH: [Language; 1] = [Language::Dutch];
    pub const ENGLISH2CHINESE: [Language; 1] = [Language::ChineseMandarin];
    pub const ENGLISH2SWEDISH: [Language; 1] = [Language::Swedish];
    pub const ENGLISH2ARABIC: [Language; 1] = [Language::Arabic];
    pub const ENGLISH2HINDI: [Language; 1] = [Language::Hindi];
    pub const ENGLISH2HEBREW: [Language; 1] = [Language::Hebrew];
    pub const ROMANCE2ENGLISH: [Language; 1] = [Language::English];
    pub const GERMAN2ENGLISH: [Language; 1] = [Language::English];
    pub const RUSSIAN2ENGLISH: [Language; 1] = [Language::English];
    pub const DUTCH2ENGLISH: [Language; 1] = [Language::English];
    pub const CHINESE2ENGLISH: [Language; 1] = [Language::English];
    pub const SWEDISH2ENGLISH: [Language; 1] = [Language::English];
    pub const ARABIC2ENGLISH: [Language; 1] = [Language::English];
    pub const HINDI2ENGLISH: [Language; 1] = [Language::English];
    pub const HEBREW2ENGLISH: [Language; 1] = [Language::English];
    pub const FRENCH2GERMAN: [Language; 1] = [Language::German];
    pub const GERMAN2FRENCH: [Language; 1] = [Language::French];
}

/// # Marian model configuration
/// Defines the Marian model architecture (e.g. number of layers, hidden layer size, label mapping...)
pub type MarianConfig = BartConfig;

/// # Marian Model for conditional generation
/// Marian model with a vocabulary decoding head
/// It is made of the following blocks:
/// - `base_model`: `BartModel` Base BART model
/// - `linear`: Linear layer with bias tied to the weights of the token id embeddings
pub struct MarianForConditionalGeneration {
    base_model: BartModel,
    final_logits_bias: Tensor,
}

impl MarianForConditionalGeneration {
    /// Build a new `MarianForConditionalGeneration`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the BART model
    /// * `config` - `MarianConfig` object defining the model architecture
    /// * `generation_mode` - flag indicating if the model should run in generation mode (a decoder start token must then be provided)
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::marian::{MarianConfig, MarianForConditionalGeneration};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = MarianConfig::from_file(config_path);
    /// let model = MarianForConditionalGeneration::new(&p.root(), &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &MarianConfig) -> MarianForConditionalGeneration
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let base_model = BartModel::new(p / "model", config);
        let final_logits_bias = p.var(
            "final_logits_bias",
            &[1, config.vocab_size],
            Init::Const(0.),
        );
        MarianForConditionalGeneration {
            base_model,
            final_logits_bias,
        }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *source_sequence_length*). Must be provided when not running in generation mode
    /// * `attention_mask` - Optional attention mask of shape (*batch size*, *source_sequence_length*) for the encoder positions. Positions with a mask with value 0 will be masked.
    /// * `encoder_outputs` - Optional tuple made of a tensor of shape (*batch size*, *source_sequence_length*, *encoder_hidden_dim*) and optional vectors of tensors of length *num_encoder_layers* with shape (*batch size*, *source_sequence_length*, *hidden_size*).
    /// These correspond to the encoder last hidden state and optional hidden states/attention weights for encoder layers. When provided, the encoder hidden state will not be recalculated. Useful for generation tasks.
    /// * `decoder_input_ids` - Optional input tensor of shape (*batch size*, *target_sequence_length*). Must be provided when running in generation mode (e.g. initialized with a BOS token)
    /// * `decoder_attention_mask` - Optional attention mask of shape (*batch size*, *target_sequence_length*) for the decoder positions. Positions with a mask with value 0 will be masked.
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `BartModelOutput` containing:
    ///   - `decoder_output` - `Tensor` of shape (*batch size*, *target_sequence_length*, *vocab_size*) representing the logits for each vocabulary item and position
    ///   - `cache` - `(Option<Tensor>, Option<Vec<&LayerState, &LayerState>>)` of length *n_layer* containing the encoder padding mask and past keys and values for both the self attention and the encoder cross attention of each layer of the decoder.
    ///   - `all_decoder_hidden_states` - `Option<Vec<Tensor>>` of length *num_decoder_layers* with shape (*batch size*, *target_sequence_length*, *hidden_size*)
    ///   - `all_decoder_attentions` - `Option<Vec<Tensor>>` of length *num_decoder_layers* with shape (*batch size*, *target_sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use tch::{nn, Device, Tensor, no_grad};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::{Int64, Double};
    /// use rust_bert::bart::BartConfig;
    /// use rust_bert::marian::MarianForConditionalGeneration;
    /// # let config_path = Path::new("path/to/config.json");
    /// # let vocab_path = Path::new("path/to/vocab.txt");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = BartConfig::from_file(config_path);
    /// # let mut marian_model = MarianForConditionalGeneration::new(&vs.root(), &config);
    /// let (batch_size, source_sequence_length, target_sequence_length) = (64, 128, 56);
    /// let input_tensor = Tensor::rand(&[batch_size, source_sequence_length], (Int64, device));
    /// let target_tensor = Tensor::rand(&[batch_size, target_sequence_length], (Int64, device));
    /// let encoder_attention_mask =
    ///     Tensor::ones(&[batch_size, source_sequence_length], (Int64, device));
    /// let decoder_attention_mask =
    ///     Tensor::ones(&[batch_size, source_sequence_length], (Int64, device));
    ///
    /// let model_output = no_grad(|| {
    ///     marian_model.forward_t(
    ///         Some(&input_tensor),
    ///         Some(&encoder_attention_mask),
    ///         None,
    ///         Some(&target_tensor),
    ///         Some(&decoder_attention_mask),
    ///         None,
    ///         false,
    ///     )
    /// });
    /// ```
    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        encoder_outputs: Option<&Tensor>,
        decoder_input_ids: Option<&Tensor>,
        decoder_attention_mask: Option<&Tensor>,
        old_layer_states: Option<Vec<(Option<LayerState>, Option<LayerState>)>>,
        train: bool,
    ) -> BartModelOutput {
        let base_model_output = self.base_model.forward_t(
            input_ids,
            attention_mask,
            decoder_input_ids,
            encoder_outputs,
            decoder_attention_mask,
            old_layer_states,
            train,
        );

        let lm_logits = base_model_output
            .decoder_output
            .linear::<Tensor>(&self.base_model.embeddings.ws, None)
            + &self.final_logits_bias;
        BartModelOutput {
            decoder_output: lm_logits,
            ..base_model_output
        }
    }

    pub fn encode(&self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> Tensor {
        self.base_model
            .encoder
            .forward_t(
                input_ids,
                attention_mask,
                &self.base_model.embeddings,
                false,
            )
            .hidden_state
    }
}

/// # Language generation model based on the Marian architecture for machine translation
pub struct MarianGenerator {
    model: MarianForConditionalGeneration,
    tokenizer: TokenizerOption,
    var_store: nn::VarStore,
    generate_config: GenerateConfig,
    bos_token_id: Option<i64>,
    eos_token_ids: Option<Vec<i64>>,
    pad_token_id: Option<i64>,
    is_encoder_decoder: bool,
    vocab_size: i64,
    decoder_start_id: Option<i64>,
    max_position_embeddings: i64,
}

impl MarianGenerator {
    /// Build a new `marianGenerator`
    ///
    /// # Arguments
    ///
    /// * `vocab_path` - Path to the model vocabulary, expected to have a structure following the [Transformers library](https://github.com/huggingface/transformers) convention
    /// * `sentencepiece_model_path` - Path to the sentencepiece model (native protobuf expected)
    /// * `config_path` - Path to the model configuration, expected to have a structure following the [Transformers library](https://github.com/huggingface/transformers) convention
    /// * `weights_path` - Path to the model weight files. These need to be converted form the `.bin` to `.ot` format using the utility script provided.
    /// * `device` - Device to run the model on, e.g. `Device::Cpu` or `Device::Cuda(0)`
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use std::path::PathBuf;
    /// # use tch::Device;
    /// # fn main() -> anyhow::Result<()> {
    /// use rust_bert::marian::MarianGenerator;
    /// use rust_bert::pipelines::generation_utils::GenerateConfig;
    /// # let mut home: PathBuf = dirs::home_dir().unwrap();
    /// # home.push("rustbert");
    /// # home.push("marian-mt-en-fr");
    /// # let config_path = &home.as_path().join("config.json");
    /// # let vocab_path = &home.as_path().join("vocab.json");
    /// # let merges_path = &home.as_path().join("spiece.model");
    /// # let weights_path = &home.as_path().join("model.ot");
    /// let device = Device::cuda_if_available();
    /// let generate_config = GenerateConfig {
    ///     max_length: Some(512),
    ///     do_sample: true,
    ///     num_beams: 6,
    ///     temperature: 1.0,
    ///     num_return_sequences: 1,
    ///     ..Default::default()
    /// };
    /// let marian_generator = MarianGenerator::new(generate_config)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(generate_config: GenerateConfig) -> Result<MarianGenerator, RustBertError> {
        let vocab_path = generate_config.vocab_resource.get_local_path()?;
        let sentence_piece_path = generate_config
            .merges_resource
            .as_ref()
            .ok_or_else(|| {
                RustBertError::InvalidConfigurationError(
                    "Marian expects a merges (SentencePiece model) resources to be provided"
                        .to_string(),
                )
            })?
            .get_local_path()?;

        let tokenizer = TokenizerOption::from_file(
            ModelType::Marian,
            vocab_path.to_str().unwrap(),
            Some(sentence_piece_path.to_str().unwrap()),
            false,
            None,
            None,
        )?;

        Self::new_with_tokenizer(generate_config, tokenizer)
    }

    pub fn new_with_tokenizer(
        generate_config: GenerateConfig,
        tokenizer: TokenizerOption,
    ) -> Result<MarianGenerator, RustBertError> {
        let config_path = generate_config.config_resource.get_local_path()?;
        let weights_path = generate_config.model_resource.get_local_path()?;
        let device = generate_config.device;

        generate_config.validate();
        let mut var_store = nn::VarStore::new(device);

        let config = BartConfig::from_file(config_path);
        let model = MarianForConditionalGeneration::new(var_store.root(), &config);
        var_store.load(weights_path)?;

        let bos_token_id = Some(config.bos_token_id.unwrap_or(0));
        let eos_token_ids = Some(match config.eos_token_id {
            Some(value) => vec![value],
            None => vec![0],
        });
        let pad_token_id = Some(config.pad_token_id.unwrap_or(58100));

        let vocab_size = config.vocab_size;
        let is_encoder_decoder = true;
        let decoder_start_id =
            Some(tokenizer.get_pad_id().ok_or(RustBertError::TokenizerError(
                "The tokenizer must contain a pad token ID to be used as BOS".to_string(),
            ))?);
        let max_position_embeddings = config.max_position_embeddings;

        Ok(MarianGenerator {
            model,
            tokenizer,
            var_store,
            generate_config,
            bos_token_id,
            eos_token_ids,
            pad_token_id,
            is_encoder_decoder,
            vocab_size,
            decoder_start_id,
            max_position_embeddings,
        })
    }

    fn force_token_id_generation(&self, scores: &mut Tensor, token_ids: &[i64]) {
        let impossible_tokens: Vec<i64> = (0..self.get_vocab_size())
            .filter(|pos| !token_ids.contains(pos))
            .collect();
        let impossible_tokens = Tensor::from_slice(&impossible_tokens).to_device(scores.device());
        let _ = scores.index_fill_(1, &impossible_tokens, f64::NEG_INFINITY);
    }
}

impl PrivateLanguageGenerator for MarianGenerator {
    fn _get_tokenizer(&self) -> &TokenizerOption {
        &self.tokenizer
    }
    fn _get_tokenizer_mut(&mut self) -> &mut TokenizerOption {
        &mut self.tokenizer
    }
    fn get_var_store(&self) -> &nn::VarStore {
        &self.var_store
    }
    fn get_var_store_mut(&mut self) -> &mut nn::VarStore {
        &mut self.var_store
    }
    fn get_config(&self) -> &GenerateConfig {
        &self.generate_config
    }
    fn get_bos_id(&self) -> Option<i64> {
        self.bos_token_id
    }
    fn get_eos_ids(&self) -> Option<&Vec<i64>> {
        self.eos_token_ids.as_ref()
    }
    fn get_pad_id(&self) -> Option<i64> {
        self.pad_token_id
    }
    fn is_encoder_decoder(&self) -> bool {
        self.is_encoder_decoder
    }
    fn get_vocab_size(&self) -> i64 {
        self.vocab_size
    }
    fn get_decoder_start_id(&self) -> Option<i64> {
        self.decoder_start_id
    }
    fn get_max_positions_embeddings(&self) -> i64 {
        self.max_position_embeddings
    }

    fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        cache: Cache,
        attention_mask: Option<&Tensor>,
        _token_type_ids: Option<&Tensor>,
        _position_ids: Option<&Tensor>,
        _input_embeds: Option<&Tensor>,
        encoder_outputs: Option<&Tensor>,
        decoder_input_ids: Option<&Tensor>,
        train: bool,
    ) -> Result<LMModelOutput, RustBertError> {
        let base_model_output = match cache {
            Cache::BARTCache(cached_layer_states) => self.model.forward_t(
                input_ids,
                attention_mask,
                encoder_outputs,
                decoder_input_ids,
                None,
                cached_layer_states,
                train,
            ),
            Cache::None => self.model.forward_t(
                input_ids,
                attention_mask,
                encoder_outputs,
                decoder_input_ids,
                None,
                None,
                train,
            ),
            _ => {
                return Err(RustBertError::ValueError(
                    "Cache not compatible with Marian Model".into(),
                ));
            }
        };

        Ok(LMModelOutput {
            lm_logits: base_model_output.decoder_output,
            cache: Cache::BARTCache(base_model_output.cache),
        })
    }

    fn prepare_scores_for_generation(
        &self,
        scores: &mut Tensor,
        current_length: i64,
        max_length: Option<i64>,
        _forced_bos_token_id: Option<i64>,
    ) {
        let _ = scores.index_fill_(
            1,
            &Tensor::from_slice(&[self.get_pad_id().unwrap()])
                .to_kind(Kind::Int64)
                .to_device(scores.device()),
            f64::NEG_INFINITY,
        );
        if let Some(max_length) = max_length {
            if current_length == max_length - 1 {
                self.force_token_id_generation(scores, self.get_eos_ids().as_ref().unwrap());
            }
        }
    }

    fn encode(&self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> Option<Tensor> {
        Some(self.model.encode(input_ids, attention_mask))
    }

    fn prepare_inputs_for_generation<'a>(
        &self,
        input_ids: Tensor,
        encoder_outputs: Option<&'a Tensor>,
        past: Cache,
        attention_mask: Tensor,
    ) -> PreparedInput<'a> {
        match past {
            Cache::BARTCache(past) => PreparedInput {
                prepared_input: None,
                prepared_attention_mask: Some(attention_mask),
                prepared_encoder_output: encoder_outputs,
                prepared_decoder_input: Some(input_ids.narrow(1, -1, 1)),
                prepared_position_ids: None,
                prepared_past: Cache::BARTCache(past),
            },
            Cache::None => PreparedInput {
                prepared_input: None,
                prepared_attention_mask: Some(attention_mask),
                prepared_encoder_output: encoder_outputs,
                prepared_decoder_input: Some(input_ids),
                prepared_position_ids: None,
                prepared_past: Cache::BARTCache(None),
            },
            _ => panic!("Cache type incompatible with Marian"),
        }
    }

    fn encode_prompt_text<S>(
        &self,
        prompt_text: &[S],
        max_len: Option<i64>,
        pad_token_id: Option<i64>,
    ) -> Tensor
    where
        S: AsRef<str> + Sync,
    {
        let tokens = self._get_tokenizer().encode_list(
            prompt_text,
            max_len
                .map(|max_len| max_len as usize)
                .unwrap_or(usize::MAX),
            &TruncationStrategy::LongestFirst,
            0,
        );
        let token_ids = tokens
            .into_iter()
            .map(|tokenized_input| tokenized_input.token_ids)
            .collect::<Vec<Vec<i64>>>();

        let max_len = token_ids.iter().map(|input| input.len()).max().unwrap();

        let pad_token = match pad_token_id {
            Some(value) => value,
            None => self._get_tokenizer().get_unk_id(),
        };

        let token_ids = token_ids
            .into_iter()
            .map(|mut input| {
                let temp = vec![pad_token; max_len - input.len()];
                input.extend(temp);
                input
            })
            .map(|tokens| Tensor::from_slice(&tokens).to(self.get_var_store().device()))
            .collect::<Vec<Tensor>>();

        Tensor::stack(&token_ids, 0)
    }

    fn reorder_cache(
        &self,
        past: &mut Cache,
        encoder_outputs: Option<Tensor>,
        beam_indices: &Tensor,
    ) -> Option<Tensor> {
        let encoder_outputs = encoder_outputs.map(|value| value.index_select(0, beam_indices));

        match past {
            Cache::BARTCache(old_cache_option) => match old_cache_option {
                Some(old_cache) => {
                    for (self_layer_state, encoder_layer_state) in old_cache.iter_mut() {
                        if self_layer_state.is_some() {
                            self_layer_state
                                .as_mut()
                                .unwrap()
                                .reorder_cache(beam_indices)
                        };
                        if encoder_layer_state.is_some() {
                            encoder_layer_state
                                .as_mut()
                                .unwrap()
                                .reorder_cache(beam_indices)
                        };
                    }
                }
                None => {}
            },
            Cache::None => {}
            _ => {
                panic!("Invalid cache for BART model");
            }
        };
        encoder_outputs
    }
}

impl LanguageGenerator for MarianGenerator {}
