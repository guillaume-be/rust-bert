// Copyright 2019-present Microsoft
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

/// # Disclaimer
/// This repository aims to facilitate research in large-scale pre-training for conversational data.
/// This toolkit contains only part of the modeling machinery needed to actually produce a model
/// weight file in a running dialog. On its own, this model provides only information about the
/// weights of various text spans; in order for a researcher to actually use it, they will need
/// to bring conversational data of their own and decode the response generation from the pretrained
/// system. Neither the author of this repository or Microsoft are responsible for any generation
/// from the 3rd party utilization of the pretrained system.
///
///
///
///
use crate::common::resources::{RemoteResource, Resource};
use crate::gpt2::{
    Gpt2ConfigResources, Gpt2MergesResources, Gpt2ModelResources, Gpt2VocabResources,
};
use crate::pipelines::generation::private_generation_utils::PrivateLanguageGenerator;
use crate::pipelines::generation::{GPT2Generator, GenerateConfig, LanguageGenerator};
use rust_tokenizers::preprocessing::tokenizer::tokenization_utils::truncate_sequences;
use rust_tokenizers::{Tokenizer, TruncationStrategy};
use tch::{Device, Tensor};

/// # Configuration for multi-turn classification
/// Contains information regarding the model to load, mirrors the GenerationConfig, with a
/// different set of default parameters and sets the device to place the model on.
pub struct ConversationConfig {
    /// Model weights resource (default: DialoGPT-medium)
    pub model_resource: Resource,
    /// Config resource (default: DialoGPT-medium)
    pub config_resource: Resource,
    /// Vocab resource (default: DialoGPT-medium)
    pub vocab_resource: Resource,
    /// Merges resource (default: DialoGPT-medium)
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
}

impl Default for ConversationConfig {
    fn default() -> ConversationConfig {
        ConversationConfig {
            model_resource: Resource::Remote(RemoteResource::from_pretrained(
                Gpt2ModelResources::DIALOGPT_MEDIUM,
            )),
            config_resource: Resource::Remote(RemoteResource::from_pretrained(
                Gpt2ConfigResources::DIALOGPT_MEDIUM,
            )),
            vocab_resource: Resource::Remote(RemoteResource::from_pretrained(
                Gpt2VocabResources::DIALOGPT_MEDIUM,
            )),
            merges_resource: Resource::Remote(RemoteResource::from_pretrained(
                Gpt2MergesResources::DIALOGPT_MEDIUM,
            )),
            min_length: 0,
            max_length: 1000,
            do_sample: true,
            early_stopping: false,
            num_beams: 1,
            temperature: 1.0,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.0,
            length_penalty: 1.0,
            no_repeat_ngram_size: 3,
            num_return_sequences: 1,
            device: Device::cuda_if_available(),
        }
    }
}

/// # Conversation model
pub struct ConversationModel {
    model: GPT2Generator,
    eos_token_id: i64,
}

impl ConversationModel {
    /// Build a new `ConversationModel`
    ///
    /// # Arguments
    ///
    /// * `conversation_config` - `ConversationConfig` object containing the resource references (model, vocabulary, configuration), conversation options and device placement (CPU/GPU)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> failure::Fallible<()> {
    /// use rust_bert::pipelines::conversation::ConversationModel;
    ///
    /// let conversation_model = ConversationModel::new(Default::default())?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(conversation_config: ConversationConfig) -> failure::Fallible<ConversationModel> {
        let generate_config = GenerateConfig {
            model_resource: conversation_config.model_resource,
            config_resource: conversation_config.config_resource,
            merges_resource: conversation_config.merges_resource,
            vocab_resource: conversation_config.vocab_resource,
            min_length: conversation_config.min_length,
            max_length: conversation_config.max_length,
            do_sample: conversation_config.do_sample,
            early_stopping: conversation_config.early_stopping,
            num_beams: conversation_config.num_beams,
            temperature: conversation_config.temperature,
            top_k: conversation_config.top_k,
            top_p: conversation_config.top_p,
            repetition_penalty: conversation_config.repetition_penalty,
            length_penalty: conversation_config.length_penalty,
            no_repeat_ngram_size: conversation_config.no_repeat_ngram_size,
            num_return_sequences: conversation_config.num_return_sequences,
            device: conversation_config.device,
        };

        let model = GPT2Generator::new(generate_config)?;
        let eos_token_id = *model.get_eos_ids().as_ref().unwrap().first().unwrap();
        Ok(ConversationModel {
            model,
            eos_token_id,
        })
    }

    /// Perform a multi-turn conversation based on user input
    ///
    /// # Arguments
    ///
    /// * `input` - `&[&str]` Array of user input texts.
    ///
    /// # Returns
    /// * `Vec<String>` Responses from the model for each input
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> failure::Fallible<()> {
    /// use rust_bert::pipelines::generation::LanguageGenerator;
    /// use rust_bert::pipelines::conversation::ConversationModel;
    /// let model = ConversationModel::new(Default::default())?;
    ///
    /// let input = ["Hello, how are you?"];
    ///
    /// let output = model.reply(&input);
    /// # Ok(())
    /// # }
    /// ```
    pub fn reply(&self, texts: &[&str]) -> Vec<String> {
        // ToDo: add possibility to pass a History object as an input (or create a History) containing a Cache object
        // ToDo: move encoding step to this method to handle teh <eos> token addition
        // ToDo: create a `generate` sub-function that takes input ids & a Option<Cache> as an input
        // ToDo: update base `generate` function to perform some preparation steps and then delegate to the lower level `generate` taking input ids & cache as input
        // ToDo: update return of function to return a Vec<String> and a History

        let prompt_ids = self.encode_input(texts);
        self.model.generate_from_ids_and_past(prompt_ids, None)
    }

    fn encode_input(&self, texts: &[&str]) -> Tensor {
        let tokens = self.model.get_tokenizer().tokenize_list(texts.to_vec());
        let max_len = self.model.get_config().max_length;
        let pad_token = match self.model.get_pad_id() {
            Some(value) => *value,
            None => self.eos_token_id,
        };
        let token_ids = tokens
            .into_iter()
            .map(|prompt_tokens| {
                self.model
                    .get_tokenizer()
                    .convert_tokens_to_ids(&prompt_tokens)
            })
            .map(|mut tokens| {
                tokens.push(self.eos_token_id);
                tokens
            })
            .collect::<Vec<Vec<i64>>>();

        let num_truncated_tokens = token_ids
            .iter()
            .map(|token_ids| {
                if token_ids.len() > max_len as usize {
                    token_ids.len() - max_len as usize
                } else {
                    0
                }
            })
            .collect::<Vec<usize>>();

        let token_ids = token_ids
            .into_iter()
            .zip(num_truncated_tokens)
            .map(|(tokens, num_truncated_tokens)| {
                truncate_sequences(
                    tokens,
                    None,
                    vec![],
                    None,
                    vec![],
                    None,
                    vec![],
                    None,
                    num_truncated_tokens,
                    &TruncationStrategy::LongestFirst,
                    0,
                )
                .unwrap()
                .0
            })
            .collect::<Vec<Vec<i64>>>();

        let max_len = token_ids.iter().map(|input| input.len()).max().unwrap();

        let token_ids = token_ids
            .into_iter()
            .map(|input| {
                let mut temp = vec![pad_token; max_len - input.len()];
                temp.extend(input);
                temp
            })
            .map(|tokens| Tensor::of_slice(&tokens).to(self.model.get_var_store().device()))
            .collect::<Vec<Tensor>>();

        Tensor::stack(&token_ids, 0)
    }
}
