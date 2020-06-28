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
use itertools::Itertools;
use rust_tokenizers::preprocessing::tokenizer::tokenization_utils::truncate_sequences;
use rust_tokenizers::{Tokenizer, TruncationStrategy};
use std::collections::HashMap;
use tch::{Device, Tensor};
use uuid::Uuid;

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
            no_repeat_ngram_size: 0,
            num_return_sequences: 1,
            device: Device::cuda_if_available(),
        }
    }
}

#[derive(Debug)]
pub struct Conversation {
    pub past_user_inputs: Vec<String>,
    pub generated_responses: Vec<String>,
    pub new_user_input: Option<String>,
    pub history: Vec<i64>,
}

impl Conversation {
    pub fn new(text: String) -> Conversation {
        Conversation {
            past_user_inputs: vec![],
            generated_responses: vec![],
            new_user_input: Some(text),
            history: vec![],
        }
    }

    pub fn add_user_input(&mut self, text: String) -> Result<(), &'static str> {
        if self.new_user_input.is_some() {
            Err("User input already provided for this conversation")
        } else {
            self.new_user_input = Some(text);
            Ok(())
        }
    }

    pub fn contains_new_input(&self) -> bool {
        self.new_user_input.is_some()
    }

    pub fn mark_processed(&mut self) {
        if self.new_user_input.is_some() {
            self.past_user_inputs
                .push(self.new_user_input.clone().unwrap());
            self.new_user_input = None;
        }
    }

    pub fn get_last_input(&self) -> &str {
        if self.new_user_input.is_some() {
            self.new_user_input.as_ref().unwrap().as_str()
        } else {
            self.past_user_inputs.last().unwrap().as_str()
        }
    }

    pub fn get_last_response(&self) -> Option<&str> {
        if !self.generated_responses.is_empty() {
            Some(self.generated_responses.last().unwrap().as_str())
        } else {
            None
        }
    }
}

#[derive(Debug)]
pub struct ConversationManager {
    conversations: HashMap<Uuid, Conversation>,
}

impl ConversationManager {
    pub fn new() -> ConversationManager {
        ConversationManager {
            conversations: HashMap::new(),
        }
    }

    pub fn get_active_conversations(&mut self) -> (Vec<&Uuid>, Vec<&mut Conversation>) {
        let mut active_uuid = vec![];
        let mut active_conversations = vec![];
        for (uuid, conversation) in self.conversations.iter_mut() {
            if conversation.new_user_input.is_some() {
                active_uuid.push(uuid);
                active_conversations.push(conversation)
            }
        }
        (active_uuid, active_conversations)
    }

    pub fn get(&mut self, uuid: &Uuid) -> Option<&mut Conversation> {
        self.conversations.get_mut(uuid)
    }

    pub fn get_all(&mut self) -> HashMap<&Uuid, &Conversation> {
        let mut output = HashMap::with_capacity(self.conversations.len());
        for (uuid, conversation) in self.conversations.iter() {
            output.insert(uuid, conversation);
        }
        output
    }

    pub fn create(&mut self, text: &str) -> Uuid {
        let conversation = Conversation::new(text.to_string());
        self.add(conversation)
    }

    pub fn add(&mut self, conversation: Conversation) -> Uuid {
        let mut uuid = Uuid::new_v4();
        while self.conversations.contains_key(&uuid) {
            uuid = Uuid::new_v4();
        }
        self.conversations.insert(uuid, conversation);
        uuid
    }

    pub fn remove(&mut self, uuid: &Uuid) -> Option<Conversation> {
        self.conversations.remove(uuid)
    }

    pub fn clear(&mut self) {
        self.conversations = HashMap::new();
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
    /// * `conversation_manager` - `&mut ConversationManager` Conversation manager keeping track of active conversations
    ///
    /// # Returns
    /// * `HashMap<&Uuid, &str>` Responses from the model for each acttive conversation, referenced by Uuid
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> failure::Fallible<()> {
    /// use rust_bert::pipelines::generation::LanguageGenerator;
    /// use rust_bert::pipelines::conversation::{ConversationModel, ConversationManager};
    /// let model = ConversationModel::new(Default::default())?;
    ///
    /// let mut conversation_manager = ConversationManager::new();
    /// conversation_manager.create("Hello, how are you?");
    ///
    /// let output = model.generate_responses(&mut conversation_manager);
    /// # Ok(())
    /// # }
    /// ```
    pub fn generate_responses<'a>(
        &self,
        conversation_manager: &'a mut ConversationManager,
    ) -> HashMap<&'a Uuid, &'a str> {
        let (active_uuid, active_conversations) = conversation_manager.get_active_conversations();
        if !active_uuid.is_empty() {
            let texts = active_conversations
                .iter()
                .map(|c| c.new_user_input.as_ref().unwrap().as_str())
                .collect_vec();

            let history = active_conversations
                .iter()
                .map(|c| &c.history)
                .collect_vec();

            let prompt_ids = self.encode_prompts(texts.as_slice());
            let input_tensor = self.concat_input_history(prompt_ids, history);
            let input_length = *input_tensor.size().last().unwrap() as usize;
            let mut generated = self.model.generate_from_ids_and_past(input_tensor, None);
            self.clean_padding_indices(&mut generated);

            let mut output = HashMap::with_capacity(active_uuid.len());

            for ((conversation, generated_sequence), uuid) in active_conversations
                .into_iter()
                .zip(generated.into_iter())
                .zip(active_uuid.into_iter())
            {
                conversation
                    .generated_responses
                    .push(self.model.get_tokenizer().decode(
                        generated_sequence[input_length..].to_vec(),
                        true,
                        true,
                    ));
                conversation.history = generated_sequence;
                conversation.mark_processed();
                output.insert(uuid, conversation.get_last_response().unwrap());
            }
            output
        } else {
            HashMap::new()
        }
    }

    fn clean_padding_indices(&self, model_output: &mut Vec<Vec<i64>>) {
        // In case inputs are sent as batch, this cleans the padding indices in the history for shorter outputs
        let pad_token = match self.model.get_pad_id() {
            Some(value) => *value,
            None => self.eos_token_id,
        };
        for sequence_history in model_output {
            let index = sequence_history
                .iter()
                .rev()
                .position(|&r| r != pad_token)
                .unwrap();
            sequence_history.drain(sequence_history.len() - index + 1..);
        }
    }

    fn concat_input_history(&self, inputs: Vec<Vec<i64>>, history: Vec<&Vec<i64>>) -> Tensor {
        let max_len = self.model.get_config().max_length;
        let pad_token = match self.model.get_pad_id() {
            Some(value) => *value,
            None => self.eos_token_id,
        };

        assert_eq!(
            inputs.len(),
            history.len(),
            "Length of inputs shoudl equal length of history"
        );

        let mut concatenated_inputs = Vec::with_capacity(inputs.len());
        for (input, history) in inputs.iter().zip(history.iter()) {
            let mut concatenated_element = Vec::with_capacity(input.len() + history.len());
            concatenated_element.extend_from_slice(history);
            concatenated_element.extend_from_slice(input);
            concatenated_inputs.push(concatenated_element);
        }

        let num_truncated_tokens = concatenated_inputs
            .iter()
            .map(|token_ids| {
                if token_ids.len() > max_len as usize {
                    token_ids.len() - max_len as usize
                } else {
                    0
                }
            })
            .collect::<Vec<usize>>();

        let concatenated_inputs = concatenated_inputs
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

        let max_len = concatenated_inputs
            .iter()
            .map(|input| input.len())
            .max()
            .unwrap();

        let concatenated_inputs = concatenated_inputs
            .into_iter()
            .map(|input| {
                let mut temp = vec![pad_token; max_len - input.len()];
                temp.extend(input);
                temp
            })
            .map(|tokens| Tensor::of_slice(&tokens).to(self.model.get_var_store().device()))
            .collect::<Vec<Tensor>>();

        Tensor::stack(&concatenated_inputs, 0)
    }

    fn encode_prompts(&self, texts: &[&str]) -> Vec<Vec<i64>> {
        let tokens = self.model.get_tokenizer().tokenize_list(texts.to_vec());

        tokens
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
            .collect::<Vec<Vec<i64>>>()
    }
}