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

use crate::gpt2::gpt2::LMHeadModel;
use tch::{Tensor, Device, nn};
use rust_tokenizers::{Tokenizer, OpenAiGptTokenizer, OpenAiGptVocab, Vocab, TruncationStrategy, Gpt2Tokenizer, Gpt2Vocab};
use crate::openai_gpt::openai_gpt::OpenAIGPTLMHeadModel;
use std::path::Path;
use crate::{Gpt2Config, GPT2LMHeadModel};
use crate::common::config::Config;
use rust_tokenizers::tokenization_utils::truncate_sequences;
use failure::_core::cmp::max;

pub struct OpenAIGenerator {
    model: OpenAIGPTLMHeadModel,
    tokenizer: OpenAiGptTokenizer,
    var_store: nn::VarStore,
}

impl OpenAIGenerator {
    pub fn new(vocab_path: &Path, merges_path: &Path, model_config_path: &Path, model_weight_path: &Path, device: Device)
               -> failure::Fallible<OpenAIGenerator> {
        let mut var_store = nn::VarStore::new(device);
        let tokenizer = OpenAiGptTokenizer::from_file(vocab_path.to_str().unwrap(), merges_path.to_str().unwrap(), true);
        let config = Gpt2Config::from_file(model_config_path);
        let model = OpenAIGPTLMHeadModel::new(&var_store.root(), &config);
        var_store.load(model_weight_path)?;
        Ok(OpenAIGenerator { model, tokenizer, var_store })
    }
}

impl LanguageGenerator<OpenAIGPTLMHeadModel, OpenAiGptVocab, OpenAiGptTokenizer> for OpenAIGenerator {
    fn get_model(&self) -> &OpenAIGPTLMHeadModel { &self.model }
    fn get_tokenizer(&self) -> &OpenAiGptTokenizer { &self.tokenizer }
    fn get_var_store(&self) -> &nn::VarStore { &self.var_store }
}

pub struct GPT2Generator {
    model: GPT2LMHeadModel,
    tokenizer: Gpt2Tokenizer,
    var_store: nn::VarStore,
}

impl GPT2Generator {
    pub fn new(vocab_path: &Path, merges_path: &Path, model_config_path: &Path, model_weight_path: &Path, device: Device)
               -> failure::Fallible<GPT2Generator> {
        let mut var_store = nn::VarStore::new(device);
        let tokenizer = Gpt2Tokenizer::from_file(vocab_path.to_str().unwrap(), merges_path.to_str().unwrap(), false);
        let config = Gpt2Config::from_file(model_config_path);
        let model = GPT2LMHeadModel::new(&var_store.root(), &config);
        var_store.load(model_weight_path)?;
        Ok(GPT2Generator { model, tokenizer, var_store })
    }
}

impl LanguageGenerator<GPT2LMHeadModel, Gpt2Vocab, Gpt2Tokenizer> for GPT2Generator {
    fn get_model(&self) -> &GPT2LMHeadModel { &self.model }
    fn get_tokenizer(&self) -> &Gpt2Tokenizer { &self.tokenizer }
    fn get_var_store(&self) -> &nn::VarStore { &self.var_store }
}


pub trait LanguageGenerator<T: LMHeadModel, V: Vocab, U: Tokenizer<V>> {
    fn get_model(&self) -> &T;
    fn get_tokenizer(&self) -> &U;
    fn get_var_store(&self) -> &nn::VarStore;

    fn encode_prompt_text(&self, prompt_text: &str, max_len: usize) -> Tensor {
        let token_ids = self.get_tokenizer().convert_tokens_to_ids(&self.get_tokenizer().tokenize(prompt_text));
        let num_truncated_tokens = if token_ids.len() > max_len { token_ids.len() - max_len } else { 0 };
        let (token_ids, _, _) = truncate_sequences(token_ids,
                                                   None,
                                                   num_truncated_tokens,
                                                   &TruncationStrategy::LongestFirst,
                                                   0).unwrap();
        Tensor::of_slice(&token_ids).unsqueeze(0).to(self.get_var_store().device())
    }


    fn generate(&self, prompt_text: &str, max_length: usize, do_sample: bool, num_beams: i64, temperature: f64,
                top_k: i64, top_p: f64, repetition_penalty: f64, length_penalty: f64, num_return_sequences: i64) -> Tensor {
        self.encode_prompt_text(prompt_text, max_length)
//        Tensor::new()
    }
}