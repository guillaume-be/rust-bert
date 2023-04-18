// Copyright 2019-present, the HuggingFace Inc. team, The Google AI Language Team and Facebook, Inc.
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

//! # Part Of Speech pipeline
//! Extracts Part of Speech tags (Noun, Verb, Adjective...) from text.
//! A lightweight pretrained model using MobileBERT is available for English.
//!
//! The example below illustrate how to run the model:
//! ```no_run
//! # fn main() -> anyhow::Result<()> {
//! use rust_bert::pipelines::pos_tagging::POSModel;
//! let pos_model = POSModel::new(Default::default())?;
//!
//! let input = ["My name is Amélie. How are you?"];
//! let output = pos_model.predict(&input);
//! # Ok(())
//! # }
//! ```
//! Output: \
//! ```no_run
//! # use rust_bert::pipelines::pos_tagging::POSTag;
//! # let output =
//! [[
//!     POSTag {
//!         word: String::from("My"),
//!         score: 0.2465,
//!         label: String::from("PRP"),
//!     },
//!     POSTag {
//!         word: String::from("name"),
//!         score: 0.8551,
//!         label: String::from("NN"),
//!     },
//!     POSTag {
//!         word: String::from("is"),
//!         score: 0.8072,
//!         label: String::from("VBZ"),
//!     },
//!     POSTag {
//!         word: String::from("Amélie"),
//!         score: 0.8102,
//!         label: String::from("NNP"),
//!     },
//!     POSTag {
//!         word: String::from("."),
//!         score: 1.0,
//!         label: String::from("."),
//!     },
//!     POSTag {
//!         word: String::from("How"),
//!         score: 0.4994,
//!         label: String::from("WRB"),
//!     },
//!     POSTag {
//!         word: String::from("are"),
//!         score: 0.928,
//!         label: String::from("VBP"),
//!     },
//!     POSTag {
//!         word: String::from("you"),
//!         score: 0.3690,
//!         label: String::from("NN"),
//!     },
//!     POSTag {
//!         word: String::from("?"),
//!         score: 1.0,
//!         label: String::from("."),
//!     },
//! ]]
//! # ;
//! ```
//!
//! To run the pipeline for another language, change the POSModel configuration from its default (see the NER pipeline for an illustration).

use crate::common::error::RustBertError;
use crate::pipelines::token_classification::{TokenClassificationConfig, TokenClassificationModel};
use serde::{Deserialize, Serialize};

use crate::pipelines::common::TokenizerOption;
#[cfg(feature = "remote")]
use {
    crate::{
        mobilebert::{
            MobileBertConfigResources, MobileBertModelResources, MobileBertVocabResources,
        },
        pipelines::{common::ModelType, token_classification::LabelAggregationOption},
        resources::RemoteResource,
    },
    tch::Device,
};

#[derive(Debug, Serialize, Deserialize)]
/// # Part of Speech tag
pub struct POSTag {
    /// String representation of the word
    pub word: String,
    /// Confidence score
    pub score: f64,
    /// Part-of-speech label (e.g. NN, VB...)
    pub label: String,
}

//type alias for some backward compatibility
pub struct POSConfig {
    token_classification_config: TokenClassificationConfig,
}

#[cfg(feature = "remote")]
impl Default for POSConfig {
    /// Provides a Part of speech tagging model (English)
    fn default() -> POSConfig {
        POSConfig {
            token_classification_config: TokenClassificationConfig {
                model_type: ModelType::MobileBert,
                model_resource: Box::new(RemoteResource::from_pretrained(
                    MobileBertModelResources::MOBILEBERT_ENGLISH_POS,
                )),
                config_resource: Box::new(RemoteResource::from_pretrained(
                    MobileBertConfigResources::MOBILEBERT_ENGLISH_POS,
                )),
                vocab_resource: Box::new(RemoteResource::from_pretrained(
                    MobileBertVocabResources::MOBILEBERT_ENGLISH_POS,
                )),
                merges_resource: None,
                lower_case: true,
                strip_accents: Some(true),
                add_prefix_space: None,
                device: Device::cuda_if_available(),
                label_aggregation_function: LabelAggregationOption::First,
                batch_size: 64,
            },
        }
    }
}

impl From<POSConfig> for TokenClassificationConfig {
    fn from(pos_config: POSConfig) -> Self {
        pos_config.token_classification_config
    }
}

/// # POSModel to extract Part of Speech tags
pub struct POSModel {
    token_classification_model: TokenClassificationModel,
}

impl POSModel {
    /// Build a new `POSModel`
    ///
    /// # Arguments
    ///
    /// * `pos_config` - `POSConfig` object containing the resource references (model, vocabulary, configuration) and device placement (CPU/GPU)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> anyhow::Result<()> {
    /// use rust_bert::pipelines::pos_tagging::POSModel;
    ///
    /// let pos_model = POSModel::new(Default::default())?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(pos_config: POSConfig) -> Result<POSModel, RustBertError> {
        let model = TokenClassificationModel::new(pos_config.into())?;
        Ok(POSModel {
            token_classification_model: model,
        })
    }

    /// Build a new `POSModel` with a provided tokenizer.
    ///
    /// # Arguments
    ///
    /// * `pos_config` - `POSConfig` object containing the resource references (model, vocabulary, configuration) and device placement (CPU/GPU)
    /// * `tokenizer` - `TokenizerOption` tokenizer to use for POS tagging.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> anyhow::Result<()> {
    /// use rust_bert::pipelines::common::{ModelType, TokenizerOption};
    /// use rust_bert::pipelines::pos_tagging::POSModel;
    /// let tokenizer = TokenizerOption::from_file(
    ///     ModelType::Bert,
    ///     "path/to/vocab.txt",
    ///     None,
    ///     false,
    ///     None,
    ///     None,
    /// )?;
    /// let pos_model = POSModel::new_with_tokenizer(Default::default(), tokenizer)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new_with_tokenizer(
        pos_config: POSConfig,
        tokenizer: TokenizerOption,
    ) -> Result<POSModel, RustBertError> {
        let model = TokenClassificationModel::new_with_tokenizer(pos_config.into(), tokenizer)?;
        Ok(POSModel {
            token_classification_model: model,
        })
    }

    /// Extract entities from a text
    ///
    /// # Arguments
    ///
    /// * `input` - `&[&str]` Array of texts to extract entities from.
    ///
    /// # Returns
    ///
    /// * `Vec<Vec<POSTag>>` containing Part of Speech tags for the inputs provided
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> anyhow::Result<()> {
    /// # use rust_bert::pipelines::pos_tagging::POSModel;
    ///
    /// let pos_model = POSModel::new(Default::default())?;
    /// let input = [
    ///     "My name is Amy. I live in Paris.",
    ///     "Paris is a city in France.",
    /// ];
    /// let output = pos_model.predict(&input);
    /// # Ok(())
    /// # }
    /// ```
    pub fn predict<S>(&self, input: &[S]) -> Vec<Vec<POSTag>>
    where
        S: AsRef<str>,
    {
        self.token_classification_model
            .predict(input, true, false)
            .into_iter()
            .map(|sequence_tokens| {
                sequence_tokens
                    .into_iter()
                    .map(|mut token| {
                        if (Self::is_punctuation(token.text.as_str()))
                            & ((token.score < 0.5) | token.score.is_nan())
                        {
                            token.label = String::from(".");
                            token.score = 1f64;
                        };
                        token
                    })
                    .map(|token| POSTag {
                        word: token.text,
                        score: token.score,
                        label: token.label,
                    })
                    .collect::<Vec<POSTag>>()
            })
            .collect::<Vec<Vec<POSTag>>>()
    }

    fn is_punctuation(string: &str) -> bool {
        string.chars().all(|c| c.is_ascii_punctuation())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    #[ignore] // no need to run, compilation is enough to verify it is Send
    fn test() {
        let config = POSConfig::default();
        let _: Box<dyn Send> = Box::new(POSModel::new(config));
    }
}
