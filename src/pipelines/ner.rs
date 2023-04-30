// Copyright 2019-present, the HuggingFace Inc. team, The Google AI Language Team and Facebook, Inc.
// Copyright (c) 2018 chakki (https://github.com/chakki-works/seqeval/blob/master/seqeval/metrics/sequence_labeling.py)
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

//! # Named Entity Recognition pipeline
//! Extracts entities (Person, Location, Organization, Miscellaneous) from text.
//! Pretrained models are available for the following languages:
//! - English
//! - German
//! - Spanish
//! - Dutch
//!
//! The default NER mode is an English BERT cased large model finetuned on CoNNL03, contributed by the [MDZ Digital Library team at the Bavarian State Library](https://github.com/dbmdz)
//! All resources for this model can be downloaded using the Python utility script included in this repository.
//! 1. Set-up a Python virtual environment and install dependencies (in ./requirements.txt)
//! 2. Run the conversion script python /utils/download-dependencies_bert_ner.py.
//! The dependencies will be downloaded to the user's home directory, under ~/rustbert/bert-ner
//!
//! The example below illustrate how to run the model for the default English NER model
//! ```no_run
//! use rust_bert::pipelines::ner::NERModel;
//! # fn main() -> anyhow::Result<()> {
//! let ner_model = NERModel::new(Default::default())?;
//!
//! let input = [
//!     "My name is Amy. I live in Paris.",
//!     "Paris is a city in France.",
//! ];
//! let output = ner_model.predict(&input);
//! # Ok(())
//! # }
//! ```
//! Output: \
//! ```no_run
//! # use rust_bert::pipelines::ner::Entity;
//! # use rust_tokenizers::Offset;
//! # let output =
//! [
//!     [
//!         Entity {
//!             word: String::from("Amy"),
//!             score: 0.9986,
//!             label: String::from("I-PER"),
//!             offset: Offset { begin: 11, end: 14 },
//!         },
//!         Entity {
//!             word: String::from("Paris"),
//!             score: 0.9985,
//!             label: String::from("I-LOC"),
//!             offset: Offset { begin: 26, end: 31 },
//!         },
//!     ],
//!     [
//!         Entity {
//!             word: String::from("Paris"),
//!             score: 0.9988,
//!             label: String::from("I-LOC"),
//!             offset: Offset { begin: 0, end: 5 },
//!         },
//!         Entity {
//!             word: String::from("France"),
//!             score: 0.9993,
//!             label: String::from("I-LOC"),
//!             offset: Offset { begin: 19, end: 25 },
//!         },
//!     ],
//! ]
//! # ;
//! ```
//!
//! To run the pipeline for another language, change the NERModel configuration from its default:
//!
//! ```no_run
//! use rust_bert::pipelines::common::ModelType;
//! use rust_bert::pipelines::ner::NERModel;
//! use rust_bert::pipelines::token_classification::TokenClassificationConfig;
//! use rust_bert::resources::RemoteResource;
//! use rust_bert::roberta::{
//!     RobertaConfigResources, RobertaModelResources, RobertaVocabResources,
//! };
//! use tch::Device;
//!
//! # fn main() -> anyhow::Result<()> {
//! let ner_config = TokenClassificationConfig {
//!     model_type: ModelType::XLMRoberta,
//!     model_resource: Box::new(RemoteResource::from_pretrained(
//!         RobertaModelResources::XLM_ROBERTA_NER_DE,
//!     )),
//!     config_resource: Box::new(RemoteResource::from_pretrained(
//!         RobertaConfigResources::XLM_ROBERTA_NER_DE,
//!     )),
//!     vocab_resource: Box::new(RemoteResource::from_pretrained(
//!         RobertaVocabResources::XLM_ROBERTA_NER_DE,
//!     )),
//!     lower_case: false,
//!     device: Device::cuda_if_available(),
//!     ..Default::default()
//! };
//!
//! let ner_model = NERModel::new(ner_config)?;
//!
//! //    Define input
//! let input = [
//!     "Mein Name ist Amélie. Ich lebe in Paris.",
//!     "Paris ist eine Stadt in Frankreich.",
//! ];
//! let output = ner_model.predict(&input);
//! # Ok(())
//! # }
//! ```
//! The XLMRoberta models for the languages are defined as follows:
//!
//! | **Language** |**Model name**|
//! :-----:|:----:
//! English| XLM_ROBERTA_NER_EN |
//! German| XLM_ROBERTA_NER_DE |
//! Spanish| XLM_ROBERTA_NER_ES |
//! Dutch| XLM_ROBERTA_NER_NL |

use crate::common::error::RustBertError;
use crate::pipelines::common::TokenizerOption;
use crate::pipelines::token_classification::{
    Token, TokenClassificationConfig, TokenClassificationModel,
};
use rust_tokenizers::Offset;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
/// # Entity generated by a `NERModel`
pub struct Entity {
    /// String representation of the Entity
    pub word: String,
    /// Confidence score
    pub score: f64,
    /// Entity label (e.g. ORG, LOC...)
    pub label: String,
    /// Token offsets
    pub offset: Offset,
}

//type alias for some backward compatibility
type NERConfig = TokenClassificationConfig;

/// # NERModel to extract named entities
pub struct NERModel {
    token_classification_model: TokenClassificationModel,
}

impl NERModel {
    /// Build a new `NERModel`
    ///
    /// # Arguments
    ///
    /// * `ner_config` - `NERConfig` object containing the resource references (model, vocabulary, configuration) and device placement (CPU/GPU)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> anyhow::Result<()> {
    /// use rust_bert::pipelines::ner::NERModel;
    ///
    /// let ner_model = NERModel::new(Default::default())?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(ner_config: NERConfig) -> Result<NERModel, RustBertError> {
        let model = TokenClassificationModel::new(ner_config)?;
        Ok(NERModel {
            token_classification_model: model,
        })
    }

    /// Build a new `NERModel` with a provided tokenizer.
    ///
    /// # Arguments
    ///
    /// * `ner_config` - `NERConfig` object containing the resource references (model, vocabulary, configuration) and device placement (CPU/GPU)
    /// * `tokenizer` - `TokenizerOption` tokenizer to use for token classification
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> anyhow::Result<()> {
    /// use rust_bert::pipelines::common::{ModelType, TokenizerOption};
    /// use rust_bert::pipelines::ner::NERModel;
    /// let tokenizer = TokenizerOption::from_file(
    ///     ModelType::Bert,
    ///     "path/to/vocab.txt",
    ///     None,
    ///     false,
    ///     None,
    ///     None,
    /// )?;
    /// let ner_model = NERModel::new_with_tokenizer(Default::default(), tokenizer)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new_with_tokenizer(
        ner_config: NERConfig,
        tokenizer: TokenizerOption,
    ) -> Result<NERModel, RustBertError> {
        let model = TokenClassificationModel::new_with_tokenizer(ner_config, tokenizer)?;
        Ok(NERModel {
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
    /// * `Vec<Vec<Entity>>` containing extracted entities
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> anyhow::Result<()> {
    /// # use rust_bert::pipelines::ner::NERModel;
    ///
    /// let ner_model = NERModel::new(Default::default())?;
    /// let input = [
    ///     "My name is Amy. I live in Paris.",
    ///     "Paris is a city in France.",
    /// ];
    /// let output = ner_model.predict(&input);
    /// # Ok(())
    /// # }
    /// ```
    pub fn predict<S>(&self, input: &[S]) -> Vec<Vec<Entity>>
    where
        S: AsRef<str>,
    {
        self.token_classification_model
            .predict(input, true, false)
            .into_iter()
            .map(|sequence_tokens| {
                sequence_tokens
                    .into_iter()
                    .filter(|token| token.label != "O")
                    .map(|token| Entity {
                        offset: token.offset.unwrap(),
                        word: token.text,
                        score: token.score,
                        label: token.label,
                    })
                    .collect::<Vec<Entity>>()
            })
            .collect::<Vec<Vec<Entity>>>()
    }

    /// Extract full entities from a text performing entity chunking. Follows the algorithm for entities
    /// chunking described in [Erik F. Tjong Kim Sang, Jorn Veenstra, Representing Text Chunks](https://www.aclweb.org/anthology/E99-1023/)
    /// The proposed implementation is inspired by the [Python seqeval library](https://github.com/chakki-works/seqeval) (shared under MIT license).
    ///
    /// # Arguments
    ///
    /// * `input` - `&[&str]` Array of texts to extract entities from.
    ///
    /// # Returns
    ///
    /// * `Vec<Entity>` containing consolidated extracted entities
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> anyhow::Result<()> {
    /// # use rust_bert::pipelines::ner::NERModel;
    ///
    /// let ner_model = NERModel::new(Default::default())?;
    /// let input = ["Asked John Smith about Acme Corp"];
    /// let output = ner_model.predict_full_entities(&input);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Outputs:
    ///
    /// Output: \
    /// ```no_run
    /// # use rust_bert::pipelines::question_answering::Answer;
    /// # use rust_bert::pipelines::ner::Entity;
    /// # use rust_tokenizers::Offset;
    /// # let output =
    /// [[
    ///     Entity {
    ///         word: String::from("John Smith"),
    ///         score: 0.9747,
    ///         label: String::from("PER"),
    ///         offset: Offset { begin: 6, end: 16 },
    ///     },
    ///     Entity {
    ///         word: String::from("Acme Corp"),
    ///         score: 0.8847,
    ///         label: String::from("I-LOC"),
    ///         offset: Offset { begin: 23, end: 32 },
    ///     },
    /// ]]
    /// # ;
    /// ```
    pub fn predict_full_entities<S>(&self, input: &[S]) -> Vec<Vec<Entity>>
    where
        S: AsRef<str>,
    {
        let tokens = self.token_classification_model.predict(input, true, false);
        let mut entities: Vec<Vec<Entity>> = Vec::new();

        for sequence_tokens in tokens {
            entities.push(Self::consolidate_entities(&sequence_tokens));
        }
        entities
    }

    fn consolidate_entities(tokens: &[Token]) -> Vec<Entity> {
        let mut entities: Vec<Entity> = Vec::new();

        let mut entity_builder = EntityBuilder::new();
        for (position, token) in tokens.iter().enumerate() {
            let tag = token.get_tag();
            let label = token.get_label();
            if let Some(entity) = entity_builder.handle_current_tag(tag, label, position, tokens) {
                entities.push(entity)
            }
        }
        if let Some(entity) = entity_builder.flush_and_reset(tokens.len(), tokens) {
            entities.push(entity);
        }
        entities
    }
}

struct EntityBuilder<'a> {
    previous_node: Option<(usize, Tag, &'a str)>,
}

impl<'a> EntityBuilder<'a> {
    fn new() -> Self {
        EntityBuilder {
            previous_node: None,
        }
    }

    fn handle_current_tag(
        &mut self,
        tag: Tag,
        label: &'a str,
        position: usize,
        tokens: &[Token],
    ) -> Option<Entity> {
        match tag {
            Tag::Outside => self.flush_and_reset(position, tokens),
            Tag::Begin | Tag::Single => {
                let entity = self.flush_and_reset(position, tokens);
                self.start_new(position, tag, label);
                entity
            }
            Tag::Inside | Tag::End => {
                if let Some((_, previous_tag, previous_label)) = self.previous_node {
                    if (previous_tag == Tag::End)
                        | (previous_tag == Tag::Single)
                        | (previous_label != label)
                    {
                        let entity = self.flush_and_reset(position, tokens);
                        self.start_new(position, tag, label);
                        entity
                    } else {
                        None
                    }
                } else {
                    self.start_new(position, tag, label);
                    None
                }
            }
        }
    }

    fn flush_and_reset(&mut self, position: usize, tokens: &[Token]) -> Option<Entity> {
        let entity = if let Some((start, _, label)) = self.previous_node {
            let entity_tokens = &tokens[start..position];
            Some(Entity {
                word: entity_tokens
                    .iter()
                    .map(|token| token.text.as_str())
                    .collect::<Vec<&str>>()
                    .join(" "),
                score: entity_tokens.iter().map(|token| token.score).product(),
                label: label.to_string(),
                offset: Offset {
                    begin: entity_tokens.first()?.offset?.begin,
                    end: entity_tokens.last()?.offset?.end,
                },
            })
        } else {
            None
        };
        self.previous_node = None;
        entity
    }

    fn start_new(&mut self, position: usize, tag: Tag, label: &'a str) {
        self.previous_node = Some((position, tag, label))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Tag {
    Begin,
    Inside,
    Outside,
    End,
    Single,
}

impl Token {
    fn get_tag(&self) -> Tag {
        match self.label.split('-').collect::<Vec<&str>>()[0] {
            "B" => Tag::Begin,
            "I" => Tag::Inside,
            "O" => Tag::Outside,
            "E" => Tag::End,
            "S" => Tag::Single,
            _ => panic!("Invalid tag encountered for token {:?}", self),
        }
    }

    fn get_label(&self) -> &str {
        let split_label = self.label.split('-').collect::<Vec<&str>>();
        if split_label.len() > 1 {
            split_label[1]
        } else {
            ""
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    #[ignore] // no need to run, compilation is enough to verify it is Send
    fn test() {
        let config = NERConfig::default();
        let _: Box<dyn Send> = Box::new(NERModel::new(config));
    }
}
