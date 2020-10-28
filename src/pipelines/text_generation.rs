// Copyright 2020 The Facebook AI Research Team Authors
// Copyright 2020-present, the HuggingFace Inc. team.
// Copyright 2020 Guillaume Becquin
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! # Text generation pipeline
//! Text generation pipeline from a prompt text.
//! Include techniques such as beam search, top-k and nucleus sampling, temperature setting and repetition penalty.
//! By default, the dependencies for this model will be downloaded for a GPT2-medium model.
//! Customized text generation models models can be loaded by overwriting the resources in the configuration.
//! The dependencies will be downloaded to the user's home directory, under ~/.cache/.rustbert/gpt2
use crate::common::error::RustBertError;
use crate::pipelines::common::{ModelType, TokenizerOption};
use crate::pipelines::generation_utils::private_generation_utils::PrivateLanguageGenerator;
use crate::pipelines::generation_utils::{
    GPT2Generator, GenerateConfig, LanguageGenerator, OpenAIGenerator, XLNetGenerator,
};
use itertools::Itertools;
use tch::Tensor;

/// # Abstraction that holds one particular textgeneration model, for any of the supported models
pub enum TextGenerationOption {
    /// Text Generator based on GPT2 model
    GPT2(GPT2Generator),
    /// Text Generator based on GPT model
    GPT(OpenAIGenerator),
    /// Text Generator based on XLNet model
    XLNet(XLNetGenerator),
}

impl TextGenerationOption {
    pub fn new(config: GenerateConfig, model_type: ModelType) -> Result<Self, RustBertError> {
        Ok(match model_type {
            ModelType::GPT2 => TextGenerationOption::GPT2(GPT2Generator::new(config)?),
            ModelType::OpenAiGpt => TextGenerationOption::GPT(OpenAIGenerator::new(config)?),
            ModelType::XLNet => TextGenerationOption::XLNet(XLNetGenerator::new(config)?),
            ModelType::Bert => {
                panic!("Text generation not implemented for Electra!");
            }
            ModelType::Bart => {
                panic!("Text generation not implemented for BART!");
            }
            ModelType::T5 => {
                panic!("Text generation not implemented for T5!");
            }
            ModelType::DistilBert => {
                panic!("Text generation not implemented for DistilBert!");
            }
            ModelType::Roberta => {
                panic!("Text generation not implemented for Roberta!");
            }
            ModelType::XLMRoberta => {
                panic!("Text generation not implemented for XLMRoberta!");
            }
            ModelType::Electra => {
                panic!("Text generation not implemented for Electra!");
            }
            ModelType::Albert => {
                panic!("Text generation not implemented for Albert!");
            }
            ModelType::Marian => {
                panic!("Text generation not implemented for Marian!");
            }
        })
    }

    /// Returns the `ModelType` for this TextGenerationOption
    pub fn model_type(&self) -> ModelType {
        match *self {
            Self::GPT2(_) => ModelType::GPT2,
            Self::GPT(_) => ModelType::OpenAiGpt,
            Self::XLNet(_) => ModelType::XLNet,
        }
    }

    /// Interface method to access tokenizer
    pub fn get_tokenizer(&self) -> &TokenizerOption {
        match self {
            Self::GPT2(model_ref) => model_ref.get_tokenizer(),
            Self::GPT(model_ref) => model_ref.get_tokenizer(),
            Self::XLNet(model_ref) => model_ref.get_tokenizer(),
        }
    }

    /// Interface method to generate() of the particular models.
    pub fn generate_indices<'a, S>(
        &self,
        prompt_texts: Option<S>,
        attention_mask: Option<Tensor>,
        min_length: Option<i64>,
        max_length: Option<i64>,
    ) -> Vec<Vec<i64>>
    where
        S: AsRef<[&'a str]>,
    {
        match *self {
            Self::GPT2(ref model) => {
                model.generate_indices(prompt_texts, attention_mask, min_length, max_length, None)
            }
            Self::GPT(ref model) => {
                model.generate_indices(prompt_texts, attention_mask, min_length, max_length, None)
            }
            Self::XLNet(ref model) => {
                model.generate_indices(prompt_texts, attention_mask, min_length, max_length, None)
            }
        }
    }
}

/// # TextGenerationModel to generate texts from a prompt
pub struct TextGenerationModel {
    model: TextGenerationOption,
    prefix: Option<String>,
    prefix_length: Option<i64>,
    min_length: i64,
    max_length: i64,
}

impl TextGenerationModel {
    /// Build a new `TextGenerationModel`
    ///
    /// # Arguments
    ///
    /// * `generation_config` - `GenerateConfig` object containing the resource references (model, vocabulary, configuration), generation options and device placement (CPU/GPU)
    /// * `model_type` - `ModelType` enum variant indicating the type of model to use for generation
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> anyhow::Result<()> {
    /// use rust_bert::pipelines::common::ModelType;
    /// use rust_bert::pipelines::text_generation::TextGenerationModel;
    ///
    /// let generation_model = TextGenerationModel::new(Default::default(), ModelType::GPT2)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(
        generation_config: GenerateConfig,
        model_type: ModelType,
    ) -> Result<TextGenerationModel, RustBertError> {
        let prefix = match model_type {
            ModelType::XLNet => Some(
                "In 1991, the remains of Russian Tsar Nicholas II and his family \
(except for Alexei and Maria) are discovered. \
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the \
remainder of the story. 1883 Western Siberia, \
a young Grigori Rasputin is asked by his father and a group of men to perform magic. \
Rasputin has a vision and denounces one of the men as a horse thief. Although his \
father initially slaps him for making such an accusation, Rasputin watches as the \
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of \
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous, \
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"
                    .to_string(),
            ),
            _ => None,
        };

        let min_length = generation_config.min_length;
        let max_length = generation_config.max_length;
        let model = TextGenerationOption::new(generation_config, model_type)?;
        let prefix_length = if let Some(prefix) = &prefix {
            Some(model.get_tokenizer().tokenize(prefix).len() as i64)
        } else {
            None
        };

        Ok(TextGenerationModel {
            model,
            prefix,
            prefix_length,
            min_length,
            max_length,
        })
    }

    /// Generate texts from provided prompts
    ///
    /// # Arguments
    ///
    /// * `input` - `&[&str]` Array of texts to summarize.
    /// * `prefix` - `impl Into<Option<&'a str>>`: Optional string to pass as a prefix for generation. Will be excluded from generated sequences.
    ///
    /// # Returns
    /// * `Vec<String>` Generated texts
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> anyhow::Result<()> {
    /// use rust_bert::pipelines::common::ModelType;
    /// use rust_bert::pipelines::text_generation::TextGenerationModel;
    ///
    /// let model = TextGenerationModel::new(Default::default(), ModelType::XLNet)?;
    ///
    /// let input = ["The dog", "The cat was"];
    /// let prefix = None;
    ///
    /// let output = model.generate(&input, prefix);
    /// # Ok(())
    /// # }
    /// ```
    pub fn generate<'a, S>(&self, texts: S, prefix: impl Into<Option<&'a str>>) -> Vec<String>
    where
        S: AsRef<[&'a str]>,
    {
        let (prefix, prefix_length) = match (prefix.into(), &self.prefix) {
            (Some(query_prefix), _) => (
                Some(query_prefix),
                Some(self.model.get_tokenizer().tokenize(query_prefix).len() as i64),
            ),
            (None, Some(pipeline_prefix)) => (Some(pipeline_prefix.as_str()), self.prefix_length),
            (None, None) => (None, None),
        };
        let generated_indices = match (prefix, prefix_length) {
            (None, _) => self.model.generate_indices(Some(texts), None, None, None),
            (Some(prefix), Some(prefix_length)) => {
                let texts = texts
                    .as_ref()
                    .iter()
                    .map(|text| format!("{} {}", prefix, text))
                    .collect_vec();
                self.model.generate_indices(
                    Some(texts.iter().map(|x| &**x).collect::<Vec<&str>>()),
                    None,
                    Some(self.min_length + prefix_length),
                    Some(self.max_length + prefix_length),
                )
            }
            _ => panic!("Prefix length not defined but prefix provided!"),
        };

        let mut output = Vec::with_capacity(generated_indices.len());
        for generated_sequence in generated_indices {
            output.push(self.model.get_tokenizer().decode(
                if prefix_length.is_some() {
                    generated_sequence
                        .into_iter()
                        .skip(prefix_length.unwrap_or(0) as usize)
                        .collect_vec()
                } else {
                    generated_sequence
                },
                true,
                true,
            ));
        }
        output
    }
}
