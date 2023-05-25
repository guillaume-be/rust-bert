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

//! # Multi-turn dialogue
//! Conversation model based on Microsoft's [DialoGPT](https://github.com/microsoft/DialoGPT).
//! This pipeline allows the generation of single or multi-turn conversations between a human and a model.
//! The DialoGPT's page states that
//! > The human evaluation results indicate that the response generated from DialoGPT is comparable to human response quality
//! > under a single-turn conversation Turing test. ([DialoGPT repository](https://github.com/microsoft/DialoGPT))
//!
//!
//! The dependencies will be downloaded to the user's home directory, under ~/.cache/.rustbert/dialgpt-medium
//! The following illustrates how to run a 2-turns conversation using a conversation manager:
//! ```no_run
//! # fn main() -> anyhow::Result<()> {
//! use rust_bert::pipelines::conversation::{ConversationManager, ConversationModel};
//! let conversation_model = ConversationModel::new(Default::default())?;
//! let mut conversation_manager = ConversationManager::new();
//!
//! let conversation_id =
//!     conversation_manager.create("Going to the movies tonight - any suggestions?");
//! let output = conversation_model.generate_responses(&mut conversation_manager);
//!
//! let _ = conversation_manager
//!     .get(&conversation_id)
//!     .unwrap()
//!     .add_user_input("Is it an action movie?")?;
//!
//! let output = conversation_model.generate_responses(&mut conversation_manager);
//!
//! # Ok(())
//! # }
//! ```
//!
//! Example output: \
//! ```no_run
//! # let output = [
//! "{a0cb3c15-9a5a-4a34-958d-95eddac0215a: \"The Big Lebowski\"}",
//! "{a0cb3c15-9a5a-4a34-958d-95eddac0215a: \"It's a comedy.\"}"
//! # ];
//! ```
//!
//! # Disclaimer
//! The authors of this repository are not responsible for any generation
//! from the 3rd party utilization of the pretrained system.
use crate::common::error::RustBertError;
use crate::gpt2::GPT2Generator;
use crate::pipelines::common::{ModelType, TokenizerOption};
use crate::pipelines::generation_utils::private_generation_utils::PrivateLanguageGenerator;
use crate::pipelines::generation_utils::{GenerateConfig, LanguageGenerator};
use crate::resources::ResourceProvider;
use std::collections::HashMap;
use tch::{Device, Kind, Tensor};
use uuid::Uuid;

#[cfg(feature = "remote")]
use crate::{
    gpt2::{Gpt2ConfigResources, Gpt2MergesResources, Gpt2ModelResources, Gpt2VocabResources},
    resources::RemoteResource,
};

/// # Configuration for multi-turn classification
/// Contains information regarding the model to load, mirrors the GenerationConfig, with a
/// different set of default parameters and sets the device to place the model on.
pub struct ConversationConfig {
    /// Model type
    pub model_type: ModelType,
    /// Model weights resource (default: DialoGPT-medium)
    pub model_resource: Box<dyn ResourceProvider + Send>,
    /// Config resource (default: DialoGPT-medium)
    pub config_resource: Box<dyn ResourceProvider + Send>,
    /// Vocab resource (default: DialoGPT-medium)
    pub vocab_resource: Box<dyn ResourceProvider + Send>,
    /// Merges resource (default: DialoGPT-medium)
    pub merges_resource: Option<Box<dyn ResourceProvider + Send>>,
    /// Minimum sequence length (default: 0)
    pub min_length: i64,
    /// Maximum sequence length (default: 20)
    pub max_length: Option<i64>,
    /// Minimum free length available for generated responses (default: 32)
    pub min_length_for_response: i64,
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

#[cfg(feature = "remote")]
impl Default for ConversationConfig {
    fn default() -> ConversationConfig {
        ConversationConfig {
            model_type: ModelType::GPT2,
            model_resource: Box::new(RemoteResource::from_pretrained(
                Gpt2ModelResources::DIALOGPT_MEDIUM,
            )),
            config_resource: Box::new(RemoteResource::from_pretrained(
                Gpt2ConfigResources::DIALOGPT_MEDIUM,
            )),
            vocab_resource: Box::new(RemoteResource::from_pretrained(
                Gpt2VocabResources::DIALOGPT_MEDIUM,
            )),
            merges_resource: Some(Box::new(RemoteResource::from_pretrained(
                Gpt2MergesResources::DIALOGPT_MEDIUM,
            ))),
            min_length: 0,
            max_length: Some(1000),
            min_length_for_response: 64,
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
            num_beam_groups: None,
            diversity_penalty: None,
            device: Device::cuda_if_available(),
        }
    }
}

impl From<ConversationConfig> for GenerateConfig {
    fn from(config: ConversationConfig) -> GenerateConfig {
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

#[derive(Debug, Clone)]
/// Data structure keeping track of a conversation in the system. It contains past user inputs and
/// generated answers, a history of the tokens generated and a placeholder for new user inputs to be
/// processed by the system if submitted for prediction
pub struct Conversation {
    /// Past user inputs that have already been processed
    pub past_user_inputs: Vec<String>,
    /// Past system generated responses
    pub generated_responses: Vec<String>,
    /// New user input that needs to be processed
    pub new_user_input: Option<String>,
    ///  History of the tokens passed as an input and generated so far used as context for next turn generation
    pub history: Vec<Vec<i64>>,
}

impl Conversation {
    /// Build a new `Conversation` with an initial user input
    ///
    /// # Arguments
    ///
    /// * `text` - `String` with the initial user input to start a conversation
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::pipelines::conversation::Conversation;
    ///
    /// let conversation = Conversation::new("Hi there!");
    /// ```
    pub fn new(text: &str) -> Conversation {
        Conversation {
            past_user_inputs: vec![],
            generated_responses: vec![],
            new_user_input: Some(text.to_string()),
            history: vec![],
        }
    }

    /// Build a new `Conversation` placeholder without user input
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::pipelines::conversation::Conversation;
    ///
    /// let conversation = Conversation::new_empty();
    /// ```
    pub fn new_empty() -> Conversation {
        Conversation {
            past_user_inputs: vec![],
            generated_responses: vec![],
            new_user_input: None,
            history: vec![],
        }
    }

    /// Adds a new user input to the conversation. This method returns an error if an unprocessed
    /// user input already exists
    ///
    /// # Arguments
    ///
    /// * `text` - `&str` with the additional user input to continue a conversation
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::pipelines::conversation::Conversation;
    ///
    /// let mut conversation = Conversation::new_empty();
    /// conversation.add_user_input("Hi there!").unwrap();
    /// ```
    pub fn add_user_input(&mut self, text: &str) -> Result<(), RustBertError> {
        if self.new_user_input.is_some() {
            Err(RustBertError::ValueError(
                "User input already provided for this conversation".into(),
            ))
        } else {
            self.new_user_input = Some(text.to_string());
            Ok(())
        }
    }

    /// Adds a new user input to the conversation. If an unprocessed user input already exists,
    /// its contents are overwritten by the new value provided.
    ///
    /// # Arguments
    ///
    /// * `text` - `&str` with the additional user input to continue a conversation
    ///
    /// # Returns
    ///
    /// * `Option<String>` containing overwritten string if applicable
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::pipelines::conversation::Conversation;
    ///
    /// let mut conversation = Conversation::new_empty();
    /// conversation
    ///     .add_user_input("This input will not be used")
    ///     .unwrap();
    /// let unused_string = conversation.add_user_input_with_overwrite("Hi there!");
    /// ```
    pub fn add_user_input_with_overwrite(&mut self, text: &str) -> Option<String> {
        let old_user_input = if self.new_user_input.is_some() {
            self.new_user_input.clone()
        } else {
            None
        };
        self.new_user_input = Some(text.to_string());
        old_user_input
    }

    /// Returns `true` if the conversation contains new user inputs to process
    ///
    /// # Returns
    ///
    /// * `bool` flag indicating if the conversation contains new inputs to process
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::pipelines::conversation::Conversation;
    ///
    /// let mut conversation = Conversation::new_empty();
    /// let false_value = conversation.contains_new_input();
    /// conversation
    ///     .add_user_input("This input will not be used")
    ///     .unwrap();
    /// let true_value = conversation.contains_new_input();
    /// ```
    pub fn contains_new_input(&self) -> bool {
        self.new_user_input.is_some()
    }

    /// Marks the conversation as processed and moves the user input that was up for
    /// processing to the past user inputs.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::pipelines::conversation::Conversation;
    ///
    /// let mut conversation = Conversation::new_empty();
    /// let false_value = conversation.contains_new_input();
    /// conversation
    ///     .add_user_input("This input will not be used")
    ///     .unwrap();
    /// let true_value = conversation.contains_new_input();
    /// conversation.mark_processed();
    /// let false_value = conversation.contains_new_input();
    /// assert_eq!(conversation.past_user_inputs.len(), 1usize);
    /// ```
    pub fn mark_processed(&mut self) {
        if self.new_user_input.is_some() {
            self.past_user_inputs
                .push(self.new_user_input.clone().unwrap());
            self.new_user_input = None;
        }
    }

    /// Returns the last user input provided (including non-processed inputs).
    ///
    /// # Returns
    ///
    /// * `Option<&str>` representation of the last user input provided
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::pipelines::conversation::Conversation;
    ///
    /// let mut conversation = Conversation::new_empty();
    /// let none_value = conversation.get_last_input();
    /// conversation
    ///     .add_user_input("This input will not be used")
    ///     .unwrap();
    /// let last_provided_input = conversation.get_last_input();
    /// assert_eq!(last_provided_input, Some("This input will not be used"));
    /// ```
    pub fn get_last_input(&self) -> Option<&str> {
        if self.new_user_input.is_some() {
            Some(self.new_user_input.as_ref().unwrap().as_str())
        } else if !self.past_user_inputs.is_empty() {
            Some(self.past_user_inputs.last().unwrap().as_str())
        } else {
            None
        }
    }

    /// Returns the last response generated by the system.
    ///
    /// # Returns
    ///
    /// * `Option<&str>` representation of the last response generated by the system.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::pipelines::conversation::Conversation;
    ///
    /// let mut conversation = Conversation::new("Hi There");
    /// let non_value = conversation.get_last_response();
    /// ```
    pub fn get_last_response(&self) -> Option<&str> {
        if !self.generated_responses.is_empty() {
            Some(self.generated_responses.last().unwrap().as_str())
        } else {
            None
        }
    }

    fn append(&mut self, text: &str, ids: &[i64]) {
        match &self.new_user_input {
            Some(_) => {
                self.mark_processed();
                if self.past_user_inputs.len() >= self.generated_responses.len() {
                    self.generated_responses.push(text.to_string());
                } else {
                    let _ = self.add_user_input(text);
                }
            }
            None => {
                let _ = self.add_user_input(text);
            }
        }
        self.history.push(ids.to_vec());
    }

    /// Initializes a conversation form a prior state. It is assumed that a conversation always
    /// start from a user interaction.
    ///
    /// # Arguments
    /// - texts: sequence of strings, alternating between past user inputs and past generated responses.
    /// - ids: sequence of sequence of ids, alternating between past user inputs and past generated responses.
    /// These can be generated via a `ConversationModel`'s `encode_prompts`.
    ///
    /// # Example:
    ///
    /// ```no_run
    /// # fn main() -> anyhow::Result<()> {
    /// use rust_bert::pipelines::conversation::{ConversationManager, ConversationModel};
    /// use rust_bert::pipelines::generation_utils::LanguageGenerator;
    /// let model = ConversationModel::new(Default::default())?;
    ///
    /// let mut conversation_manager = ConversationManager::new();
    /// let history = [
    ///     "Going to the movies tonight - any suggestions?",
    ///     "The Big Lebowski",
    ///     "Is it an action movie?",
    /// ];
    /// let encoded_history = model.encode_prompts(&history);
    ///
    /// let conversation_1_id = conversation_manager.create_empty();
    /// let _ = conversation_manager
    ///     .get(&conversation_1_id)
    ///     .unwrap()
    ///     .load_from_history(&history, &encoded_history);
    /// # Ok(())
    /// # }
    /// ```
    pub fn load_from_history<S, SI>(&mut self, texts: &[S], ids: &[SI])
    where
        S: AsRef<str>,
        SI: AsRef<[i64]>,
    {
        for (round_text, round_ids) in texts.iter().zip(ids.iter()) {
            self.append(round_text.as_ref(), round_ids.as_ref());
        }

        if texts.len() / 2 == 1 {
            self.history.pop();
        }
    }
}

/// Data structure allowing the management of conversations and main input to the dialogue model.
/// It contains a `HashMap` of conversations with `UUID` keys
#[derive(Debug)]
pub struct ConversationManager {
    conversations: HashMap<Uuid, Conversation>,
}

impl ConversationManager {
    /// Build a new `ConversationManager`
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::pipelines::conversation::ConversationManager;
    ///
    /// let conversation_manager = ConversationManager::new();
    /// ```
    pub fn new() -> ConversationManager {
        ConversationManager {
            conversations: HashMap::new(),
        }
    }

    /// Returns a list of the active conversations (containing new inputs to be processed by the model)
    ///
    /// # Returns
    ///
    /// * `(Vec<&Uuid>, Vec<&mut Conversation>)` Tuple of vectors with the active `UUID` and `Conversations`
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::pipelines::conversation::{Conversation, ConversationManager};
    ///
    /// let mut conversation_manager = ConversationManager::new();
    ///
    /// let conversation = Conversation::new("Hi there!");
    /// let empty_conversation = Conversation::new_empty();
    /// let conversation_id = conversation_manager.add(conversation);
    /// let empty_conversation_id = conversation_manager.add(empty_conversation);
    ///
    /// let active_conversations = conversation_manager.get_active_conversations();
    /// assert_eq!(active_conversations.0.len(), 1usize);
    /// ```
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

    /// Returns a mutable reference to the conversation wih the provided UUID
    ///
    /// # Arguments
    ///
    /// * `uuid` - `&Uuid` of the conversation to retrieve
    ///
    /// # Returns
    ///
    /// * `Option<&mut Conversation>` Optional mutable reference to the conversation matching the UUID provided
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::pipelines::conversation::{Conversation, ConversationManager};
    ///
    /// let mut conversation_manager = ConversationManager::new();
    ///
    /// let conversation = Conversation::new("Hi there!");
    /// let conversation_id = conversation_manager.add(conversation);
    ///
    /// let conversation_ref = conversation_manager.get(&conversation_id);
    /// ```
    pub fn get(&mut self, uuid: &Uuid) -> Option<&mut Conversation> {
        self.conversations.get_mut(uuid)
    }

    /// Returns a HashMap containing references to all conversations stored in the manager
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::pipelines::conversation::{Conversation, ConversationManager};
    ///
    /// let mut conversation_manager = ConversationManager::new();
    ///
    /// let conversation = Conversation::new("Hi there!");
    /// let conversation_id = conversation_manager.add(conversation);
    ///
    /// let all_conversations = conversation_manager.get_all();
    /// ```
    pub fn get_all(&mut self) -> HashMap<&Uuid, &Conversation> {
        let mut output = HashMap::with_capacity(self.conversations.len());
        for (uuid, conversation) in self.conversations.iter() {
            output.insert(uuid, conversation);
        }
        output
    }

    /// Creates a conversation and add it to the conversation manager
    ///
    /// # Arguments
    ///
    /// * `text` - `&str` string slice with an original user input
    ///
    /// # Returns
    ///
    /// * `Uuid` for the conversation created
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::pipelines::conversation::{Conversation, ConversationManager};
    ///
    /// let mut conversation_manager = ConversationManager::new();
    ///
    /// let conversation_id = conversation_manager.create("Hi there!");
    /// ```
    pub fn create(&mut self, text: &str) -> Uuid {
        let conversation = Conversation::new(text);
        self.add(conversation)
    }

    /// Creates an empty conversation and add it to the conversation manager
    ///
    /// # Returns
    ///
    /// * `Uuid` for the conversation created
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::pipelines::conversation::{Conversation, ConversationManager};
    ///
    /// let mut conversation_manager = ConversationManager::new();
    ///
    /// let conversation_id = conversation_manager.create_empty();
    /// ```
    pub fn create_empty(&mut self) -> Uuid {
        let conversation = Conversation::new_empty();
        self.add(conversation)
    }

    /// Adds an existing conversation to the conversation manager
    ///
    /// # Arguments
    ///
    /// * `conversation` - `Conversation` to be added to the conversation manager
    ///
    /// # Returns
    ///
    /// * `Uuid` for the conversation created
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::pipelines::conversation::{Conversation, ConversationManager};
    ///
    /// let mut conversation_manager = ConversationManager::new();
    ///
    /// let conversation = Conversation::new("Hi there!");
    /// let conversation_id = conversation_manager.add(conversation);
    /// ```
    pub fn add(&mut self, conversation: Conversation) -> Uuid {
        let mut uuid = Uuid::new_v4();
        while self.conversations.contains_key(&uuid) {
            uuid = Uuid::new_v4();
        }
        self.conversations.insert(uuid, conversation);
        uuid
    }

    /// Deregister a conversation from the conversation manager
    ///
    /// # Arguments
    ///
    /// * `uuid` - `&Uuid` of the conversation to deregister from the conversation manager
    ///
    /// # Returns
    ///
    /// * `Option<Conversation>` de-registered conversation
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::pipelines::conversation::{Conversation, ConversationManager};
    ///
    /// let mut conversation_manager = ConversationManager::new();
    ///
    /// let conversation_id = conversation_manager.create("Hi there!");
    /// conversation_manager.remove(&conversation_id);
    /// ```
    pub fn remove(&mut self, uuid: &Uuid) -> Option<Conversation> {
        self.conversations.remove(uuid)
    }

    /// Clear all conversations from the conversation manager, and returns the conversations and their
    /// former UUID.
    ///
    /// # Returns
    ///
    /// * `HashMap<Uuid, Conversation>` de-registered conversations
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::pipelines::conversation::{Conversation, ConversationManager};
    ///
    /// let mut conversation_manager = ConversationManager::new();
    ///
    /// let conversation_id = conversation_manager.create("Hi there!");
    /// let conversations = conversation_manager.clear();
    /// ```
    pub fn clear(&mut self) -> HashMap<Uuid, Conversation> {
        let mut output = HashMap::with_capacity(self.conversations.len());
        for (uuid, conversation) in self.conversations.iter() {
            output.insert(*uuid, conversation.clone());
        }
        self.conversations = HashMap::new();
        output
    }
}

impl Default for ConversationManager {
    fn default() -> Self {
        Self::new()
    }
}

/// # Abstraction that holds one particular conversation model, for any of the supported models
pub enum ConversationOption {
    /// Conversation based on GPT2 model
    GPT2(GPT2Generator),
}

impl ConversationOption {
    pub fn new(config: ConversationConfig) -> Result<Self, RustBertError> {
        match config.model_type {
            ModelType::GPT2 => Ok(ConversationOption::GPT2(GPT2Generator::new(config.into())?)),
            _ => Err(RustBertError::InvalidConfigurationError(
                "GPT2 is currently the only supported model for conversation generation"
                    .to_string(),
            )),
        }
    }

    pub fn new_with_tokenizer(
        config: ConversationConfig,
        tokenizer: TokenizerOption,
    ) -> Result<Self, RustBertError> {
        match config.model_type {
            ModelType::GPT2 => Ok(ConversationOption::GPT2(GPT2Generator::new_with_tokenizer(
                config.into(),
                tokenizer,
            )?)),
            _ => Err(RustBertError::InvalidConfigurationError(
                "GPT2 is currently the only supported model for conversation generation"
                    .to_string(),
            )),
        }
    }

    pub fn get_eos_id(&self) -> Result<i64, RustBertError> {
        match self {
            Self::GPT2(model_ref) => {
                Ok(*model_ref.get_eos_ids().as_ref().unwrap().first().unwrap())
            }
        }
    }

    /// Get a reference to the model tokenizer.
    pub fn get_tokenizer(&self) -> &TokenizerOption {
        match self {
            Self::GPT2(model_ref) => model_ref._get_tokenizer(),
        }
    }

    /// Get a mutable reference to the model tokenizer.
    pub fn get_tokenizer_mut(&mut self) -> &TokenizerOption {
        match self {
            Self::GPT2(model_ref) => model_ref._get_tokenizer_mut(),
        }
    }

    /// Returns the `ModelType` for this ConversationOption
    pub fn model_type(&self) -> ModelType {
        match *self {
            Self::GPT2(_) => ModelType::GPT2,
        }
    }

    /// Interface method to generate_from_ids_and_past() of the particular models.
    pub fn generate_from_ids_and_past(
        &self,
        input_ids: Tensor,
        attention_mask: Option<Tensor>,
    ) -> Vec<Vec<i64>> {
        match *self {
            Self::GPT2(ref model) => model
                .generate_from_ids_and_past(input_ids, attention_mask, None)
                .into_iter()
                .map(|output| output.indices)
                .collect(),
        }
    }
}

/// # Conversation model
/// Processes a ConversationManager and generate system responses for active conversations.
pub struct ConversationModel {
    model: ConversationOption,
    eos_token_id: i64,
    max_allowed_context_length: Option<i64>,
    device: Device,
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
    /// # fn main() -> anyhow::Result<()> {
    /// use rust_bert::pipelines::conversation::ConversationModel;
    ///
    /// let conversation_model = ConversationModel::new(Default::default())?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(
        conversation_config: ConversationConfig,
    ) -> Result<ConversationModel, RustBertError> {
        let max_allowed_length = conversation_config
            .max_length
            .map(|max_length| max_length - conversation_config.min_length_for_response);
        let device = conversation_config.device;
        let model = ConversationOption::new(conversation_config)?;
        let eos_token_id = model.get_eos_id()?;
        Ok(ConversationModel {
            model,
            eos_token_id,
            max_allowed_context_length: max_allowed_length,
            device,
        })
    }

    /// Build a new `ConversationModel` with a provided tokenizer.
    ///
    /// # Arguments
    ///
    /// * `conversation_config` - `ConversationConfig` object containing the resource references (model, vocabulary, configuration), conversation options and device placement (CPU/GPU)
    /// * `tokenizer` - `TokenizerOption` tokenizer to use for conversation
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> anyhow::Result<()> {
    /// use rust_bert::pipelines::common::{ModelType, TokenizerOption};
    /// use rust_bert::pipelines::conversation::ConversationModel;
    /// let tokenizer = TokenizerOption::from_file(
    ///     ModelType::GPT2,
    ///     "path/to/vocab.json",
    ///     Some("path/to/merges.txt"),
    ///     false,
    ///     None,
    ///     None,
    /// )?;
    /// let conversation_model = ConversationModel::new_with_tokenizer(Default::default(), tokenizer)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new_with_tokenizer(
        conversation_config: ConversationConfig,
        tokenizer: TokenizerOption,
    ) -> Result<ConversationModel, RustBertError> {
        let max_allowed_length = conversation_config
            .max_length
            .map(|max_length| max_length - conversation_config.min_length_for_response);
        let device = conversation_config.device;
        let model = ConversationOption::new_with_tokenizer(conversation_config, tokenizer)?;
        let eos_token_id = model.get_eos_id()?;
        Ok(ConversationModel {
            model,
            eos_token_id,
            max_allowed_context_length: max_allowed_length,
            device,
        })
    }

    /// Perform a multi-turn conversation based on user input
    ///
    /// # Arguments
    ///
    /// * `conversation_manager` - `&mut ConversationManager` Conversation manager keeping track of active conversations
    ///
    /// # Returns
    /// * `HashMap<&Uuid, &str>` Responses from the model for each active conversation, referenced by Uuid
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> anyhow::Result<()> {
    /// use rust_bert::pipelines::conversation::{ConversationManager, ConversationModel};
    /// use rust_bert::pipelines::generation_utils::LanguageGenerator;
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
                .collect::<Vec<&str>>();

            let history = active_conversations
                .iter()
                .map(|c| c.history.iter().flatten().copied().collect())
                .collect::<Vec<Vec<i64>>>();

            let prompt_ids = self.encode_prompts(texts.as_ref());
            let (input_tensor, attention_mask) =
                self.concat_input_history(prompt_ids.as_ref(), history);
            let input_length = *input_tensor.size().last().unwrap() as usize;
            let mut generated = self
                .model
                .generate_from_ids_and_past(input_tensor, Some(attention_mask));
            let removed_padding_quantities = self.clean_padding_indices(&mut generated);

            let mut output = HashMap::with_capacity(active_uuid.len());

            for (
                ((conversation, (generated_sequence, conversation_promp_ids)), uuid),
                removed_padding,
            ) in active_conversations
                .into_iter()
                .zip(generated.into_iter().zip(prompt_ids.into_iter()))
                .zip(active_uuid.into_iter())
                .zip(removed_padding_quantities.into_iter())
            {
                let generated_response = &generated_sequence[input_length - removed_padding.0..];
                conversation
                    .generated_responses
                    .push(
                        self.model
                            .get_tokenizer()
                            .decode(generated_response, true, true),
                    );
                conversation.history.push(conversation_promp_ids);
                conversation.history.push(generated_response.to_vec());
                conversation.mark_processed();
                output.insert(uuid, conversation.get_last_response().unwrap());
            }
            output
        } else {
            HashMap::new()
        }
    }

    fn clean_padding_indices(&self, model_output: &mut Vec<Vec<i64>>) -> Vec<(usize, usize)> {
        // In case inputs are sent as batch, this cleans the padding indices in the history for shorter outputs
        let pad_token = self
            .model
            .get_tokenizer()
            .get_pad_id()
            .unwrap_or(self.eos_token_id);
        let mut removed_tokens = Vec::with_capacity(model_output.len());
        for sequence_history in model_output {
            let index_end = sequence_history
                .iter()
                .rev()
                .position(|&r| r != pad_token)
                .unwrap();
            let index_start = sequence_history
                .iter()
                .position(|&r| r != pad_token)
                .unwrap();
            if index_end > 0 {
                sequence_history.drain(sequence_history.len() - index_end + 1..);
            }
            sequence_history.drain(..index_start);
            removed_tokens.push((index_start, index_end));
        }
        removed_tokens
    }

    fn concat_input_history(
        &self,
        inputs: &[Vec<i64>],
        history: Vec<Vec<i64>>,
    ) -> (Tensor, Tensor) {
        // Concatenates the history token indices with new user input
        let pad_token = self
            .model
            .get_tokenizer()
            .get_pad_id()
            .unwrap_or(self.eos_token_id);

        assert_eq!(
            inputs.len(),
            history.len(),
            "Length of inputs should equal length of history"
        );

        let mut concatenated_inputs = Vec::with_capacity(inputs.len());
        for (input, history) in inputs.iter().zip(history.iter()) {
            let mut concatenated_element = Vec::with_capacity(input.len() + history.len());
            concatenated_element.extend_from_slice(history);
            concatenated_element.extend_from_slice(input);
            concatenated_inputs.push(concatenated_element);
        }

        let truncated_concatenated_inputs = concatenated_inputs
            .iter()
            .map(|input| match self.max_allowed_context_length {
                Some(max_allowed_context_length)
                    if input.len() > max_allowed_context_length as usize =>
                {
                    let start = self.get_truncated_input_index(
                        input,
                        max_allowed_context_length as usize,
                        pad_token,
                    );
                    &input[start..]
                }
                _ => input.as_slice(),
            })
            .collect::<Vec<&[i64]>>();

        let max_len = truncated_concatenated_inputs
            .iter()
            .map(|input| input.len())
            .max()
            .unwrap();

        let attention_mask = Tensor::ones(
            [inputs.len() as i64, max_len as i64],
            (Kind::Int8, self.device),
        );

        let concatenated_inputs = truncated_concatenated_inputs
            .into_iter()
            .enumerate()
            .map(|(input_idx, input)| {
                let _ = attention_mask
                    .get(input_idx as i64)
                    .slice(0, 0, (max_len - input.len()) as i64, 1)
                    .fill_(0);
                let mut padded_input = vec![pad_token; max_len - input.len()];
                padded_input.extend(input);
                padded_input
            })
            .map(|tokens| Tensor::from_slice(&tokens).to(self.device))
            .collect::<Vec<Tensor>>();

        (Tensor::stack(&concatenated_inputs, 0), attention_mask)
    }

    fn get_truncated_input_index(
        &self,
        history: &[i64],
        max_length: usize,
        pad_token: i64,
    ) -> usize {
        let start_length = history.len();
        let eos_indices: Vec<usize> = history
            .iter()
            .enumerate()
            .filter(|(i, &e)| {
                (e == pad_token)
                    & (*i != start_length - 1)
                    & ((start_length as isize - max_length as isize - *i as isize) < 0)
            })
            .map(|(i, _)| i + 1)
            .collect();

        // Return the position of the first EOS index that fits the max length requirement.
        // If it does not exist, no solution exists and truncate text at a non-EOS position
        *eos_indices.first().unwrap_or(&(start_length - max_length))
    }

    /// Encodes prompts into Vectors of indices to be processed by the model. This method may be used to
    /// initialize the history of a conversation with a prior state.
    ///
    /// # Example:
    ///
    /// ```no_run
    /// # fn main() -> anyhow::Result<()> {
    /// use rust_bert::pipelines::conversation::{ConversationManager, ConversationModel};
    /// use rust_bert::pipelines::generation_utils::LanguageGenerator;
    /// let model = ConversationModel::new(Default::default())?;
    /// let history = [
    ///     "Going to the movies tonight - any suggestions?",
    ///     "The Big Lebowski",
    ///     "Is it an action movie?",
    /// ];
    /// let encoded_history = model.encode_prompts(&history);
    /// # Ok(())
    /// # }
    /// ```
    pub fn encode_prompts(&self, texts: &[&str]) -> Vec<Vec<i64>> {
        // Encode the user prompt into token ids
        let tokens = self.model.get_tokenizer().tokenize_list(texts);

        tokens
            .into_iter()
            .map(|prompt_tokens| {
                self.model
                    .get_tokenizer()
                    .convert_tokens_to_ids(&prompt_tokens)
            })
            .map(|mut tokens| {
                if let Some(max_allowed_context_length) = self.max_allowed_context_length {
                    tokens.truncate(max_allowed_context_length as usize - 1);
                }
                tokens.push(self.eos_token_id);
                tokens
            })
            .collect::<Vec<Vec<i64>>>()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    #[ignore] // no need to run, compilation is enough to verify it is Send
    fn test() {
        let config = ConversationConfig::default();
        let _: Box<dyn Send> = Box::new(ConversationModel::new(config));
    }
}
