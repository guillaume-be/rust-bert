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
use tch::{Device, Tensor};

use crate::common::error::RustBertError;
use crate::common::resources::RemoteResource;
use crate::gpt2::{
    GPT2Generator, Gpt2ConfigResources, Gpt2MergesResources, Gpt2ModelResources, Gpt2VocabResources,
};
use crate::openai_gpt::OpenAIGenerator;
use crate::pipelines::common::{ModelType, TokenizerOption};
use crate::pipelines::generation_utils::private_generation_utils::PrivateLanguageGenerator;
use crate::pipelines::generation_utils::{GenerateConfig, LanguageGenerator};
use crate::reformer::ReformerGenerator;
use crate::resources::Resource;
use crate::xlnet::XLNetGenerator;

/// # Configuration for text generation
/// Contains information regarding the model to load, mirrors the GenerateConfig, with a
/// different set of default parameters and sets the device to place the model on.
pub struct TextGenerationConfig {
    /// Model type
    pub model_type: ModelType,
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
    /// Number of beam groups for diverse beam generation. If provided and higher than 1, will split the beams into beam subgroups leading to more diverse generation.
    pub num_beam_groups: Option<i64>,
    /// Diversity penalty for diverse beam search. High values will enforce more difference between beam groups (default: 5.5)
    pub diversity_penalty: Option<f64>,
    /// Device to place the model on (default: CUDA/GPU when available)
    pub device: Device,
}

impl TextGenerationConfig {
    /// Instantiate a new text generation configuration of the supplied type.
    ///
    /// # Arguments
    ///
    /// * `model_type` - `ModelType` indicating the model type to load (must match with the actual data to be loaded!)
    /// * model_resource - The `Resource` pointing to the model to load (e.g.  model.ot)
    /// * config_resource - The `Resource' pointing to the model configuration to load (e.g. config.json)
    /// * vocab_resource - The `Resource' pointing to the tokenizer's vocabulary to load (e.g.  vocab.txt/vocab.json)
    /// * merges_resource - The `Resource`  pointing to the tokenizer's merge file or SentencePiece model to load (e.g.  merges.txt).
    pub fn new(
        model_type: ModelType,
        model_resource: Resource,
        config_resource: Resource,
        vocab_resource: Resource,
        merges_resource: Resource,
    ) -> TextGenerationConfig {
        TextGenerationConfig {
            model_type,
            model_resource,
            config_resource,
            vocab_resource,
            merges_resource,
            device: Device::cuda_if_available(),
            ..Default::default()
        }
    }
}

impl Default for TextGenerationConfig {
    fn default() -> TextGenerationConfig {
        TextGenerationConfig {
            model_type: ModelType::GPT2,
            model_resource: Resource::Remote(RemoteResource::from_pretrained(
                Gpt2ModelResources::GPT2_MEDIUM,
            )),
            config_resource: Resource::Remote(RemoteResource::from_pretrained(
                Gpt2ConfigResources::GPT2_MEDIUM,
            )),
            vocab_resource: Resource::Remote(RemoteResource::from_pretrained(
                Gpt2VocabResources::GPT2_MEDIUM,
            )),
            merges_resource: Resource::Remote(RemoteResource::from_pretrained(
                Gpt2MergesResources::GPT2_MEDIUM,
            )),
            min_length: 0,
            max_length: 20,
            do_sample: true,
            early_stopping: true,
            num_beams: 5,
            temperature: 1.0,
            top_k: 0,
            top_p: 0.9,
            repetition_penalty: 1.0,
            length_penalty: 1.0,
            no_repeat_ngram_size: 3,
            num_return_sequences: 1,
            num_beam_groups: None,
            diversity_penalty: None,
            device: Device::cuda_if_available(),
        }
    }
}

impl From<TextGenerationConfig> for GenerateConfig {
    fn from(config: TextGenerationConfig) -> GenerateConfig {
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

/// # Abstraction that holds one particular text generation model, for any of the supported models
pub enum TextGenerationOption {
    /// Text Generator based on GPT2 model
    GPT2(GPT2Generator),
    /// Text Generator based on GPT model
    GPT(OpenAIGenerator),
    /// Text Generator based on XLNet model
    XLNet(XLNetGenerator),
    /// Text Generator based on Reformer model
    Reformer(ReformerGenerator),
}

impl TextGenerationOption {
    pub fn new(config: TextGenerationConfig) -> Result<Self, RustBertError> {
        match config.model_type {
            ModelType::GPT2 => Ok(TextGenerationOption::GPT2(GPT2Generator::new(
                config.into(),
            )?)),
            ModelType::OpenAiGpt => Ok(TextGenerationOption::GPT(OpenAIGenerator::new(
                config.into(),
            )?)),
            ModelType::XLNet => Ok(TextGenerationOption::XLNet(XLNetGenerator::new(
                config.into(),
            )?)),
            ModelType::Reformer => Ok(TextGenerationOption::Reformer(ReformerGenerator::new(
                config.into(),
            )?)),
            _ => Err(RustBertError::InvalidConfigurationError(format!(
                "Text generation not implemented for {:?}!",
                config.model_type
            ))),
        }
    }

    /// Returns the `ModelType` for this TextGenerationOption
    pub fn model_type(&self) -> ModelType {
        match *self {
            Self::GPT2(_) => ModelType::GPT2,
            Self::GPT(_) => ModelType::OpenAiGpt,
            Self::XLNet(_) => ModelType::XLNet,
            Self::Reformer(_) => ModelType::Reformer,
        }
    }

    /// Interface method to access tokenizer
    pub fn get_tokenizer(&self) -> &TokenizerOption {
        match self {
            Self::GPT2(model_ref) => model_ref.get_tokenizer(),
            Self::GPT(model_ref) => model_ref.get_tokenizer(),
            Self::XLNet(model_ref) => model_ref.get_tokenizer(),
            Self::Reformer(model_ref) => model_ref.get_tokenizer(),
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
            Self::Reformer(ref model) => {
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
    /// let generation_model = TextGenerationModel::new(Default::default())?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(
        generation_config: TextGenerationConfig,
    ) -> Result<TextGenerationModel, RustBertError> {
        let prefix = match generation_config.model_type {
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
        let model = TextGenerationOption::new(generation_config)?;
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
    /// let model = TextGenerationModel::new(Default::default())?;
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
                    .collect::<Vec<String>>();
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
                        .collect::<Vec<i64>>()
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

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    #[ignore] // no need to run, compilation is enough to verify it is Send
    fn test() {
        let config = TextGenerationConfig::default();
        let _: Box<dyn Send> = Box::new(TextGenerationModel::new(config));
    }
}
