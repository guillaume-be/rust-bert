// Copyright 2018 The Google AI Language Team Authors, Facebook AI Research authors.
// Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
use tch::kind::Kind::Int64;

pub struct OpenAIGenerator {
    model: OpenAIGPTLMHeadModel,
    tokenizer: OpenAiGptTokenizer,
    var_store: nn::VarStore,
    bos_token_id: i64,
    eos_token_ids: Vec<i64>,
    pad_token_id: i64,
}

impl OpenAIGenerator {
    pub fn new(vocab_path: &Path, merges_path: &Path, model_config_path: &Path, model_weight_path: &Path, device: Device)
               -> failure::Fallible<OpenAIGenerator> {
        let mut var_store = nn::VarStore::new(device);
        let tokenizer = OpenAiGptTokenizer::from_file(vocab_path.to_str().unwrap(), merges_path.to_str().unwrap(), true);
        let config = Gpt2Config::from_file(model_config_path);
        let model = OpenAIGPTLMHeadModel::new(&var_store.root(), &config);
        var_store.load(model_weight_path)?;

        let bos_token_id = 0i64;
        let eos_token_ids = vec!(0i64);
        let pad_token_id = 0i64;

        Ok(OpenAIGenerator { model, tokenizer, var_store, bos_token_id, eos_token_ids, pad_token_id })
    }
}

impl LanguageGenerator<OpenAIGPTLMHeadModel, OpenAiGptVocab, OpenAiGptTokenizer> for OpenAIGenerator {
    fn get_model(&self) -> &OpenAIGPTLMHeadModel { &self.model }
    fn get_tokenizer(&self) -> &OpenAiGptTokenizer { &self.tokenizer }
    fn get_var_store(&self) -> &nn::VarStore { &self.var_store }
    fn get_bos_id(&self) -> &i64 { &self.bos_token_id }
    fn get_eos_ids(&self) -> &Vec<i64> { &self.eos_token_ids }
    fn get_pad_id(&self) -> &i64 { &self.pad_token_id }
}

pub struct GPT2Generator {
    model: GPT2LMHeadModel,
    tokenizer: Gpt2Tokenizer,
    var_store: nn::VarStore,
    bos_token_id: i64,
    eos_token_ids: Vec<i64>,
    pad_token_id: i64,
}

impl GPT2Generator {
    pub fn new(vocab_path: &Path, merges_path: &Path, model_config_path: &Path, model_weight_path: &Path, device: Device)
               -> failure::Fallible<GPT2Generator> {
        let mut var_store = nn::VarStore::new(device);
        let tokenizer = Gpt2Tokenizer::from_file(vocab_path.to_str().unwrap(), merges_path.to_str().unwrap(), false);
        let config = Gpt2Config::from_file(model_config_path);
        let model = GPT2LMHeadModel::new(&var_store.root(), &config);
        var_store.load(model_weight_path)?;

        let bos_token_id = tokenizer.vocab().token_to_id(Gpt2Vocab::bos_value());
        let eos_token_ids = vec!(tokenizer.vocab().token_to_id(Gpt2Vocab::eos_value()));
        let pad_token_id = tokenizer.vocab().token_to_id(Gpt2Vocab::eos_value());

        Ok(GPT2Generator { model, tokenizer, var_store, bos_token_id, eos_token_ids, pad_token_id })
    }
}

impl LanguageGenerator<GPT2LMHeadModel, Gpt2Vocab, Gpt2Tokenizer> for GPT2Generator {
    fn get_model(&self) -> &GPT2LMHeadModel { &self.model }
    fn get_tokenizer(&self) -> &Gpt2Tokenizer { &self.tokenizer }
    fn get_var_store(&self) -> &nn::VarStore { &self.var_store }
    fn get_bos_id(&self) -> &i64 { &self.bos_token_id }
    fn get_eos_ids(&self) -> &Vec<i64> { &self.eos_token_ids }
    fn get_pad_id(&self) -> &i64 { &self.pad_token_id }
}

pub trait LanguageGenerator<T: LMHeadModel, V: Vocab, U: Tokenizer<V>> {
    fn get_model(&self) -> &T;
    fn get_tokenizer(&self) -> &U;
    fn get_var_store(&self) -> &nn::VarStore;
    fn get_bos_id(&self) -> &i64;
    fn get_eos_ids(&self) -> &Vec<i64>;
    fn get_pad_id(&self) -> &i64;

    fn encode_prompt_text(&self, prompt_text: &str, max_len: u64) -> Tensor {
        let token_ids = self.get_tokenizer().convert_tokens_to_ids(&self.get_tokenizer().tokenize(prompt_text));
        let num_truncated_tokens = if token_ids.len() > max_len as usize { token_ids.len() - max_len as usize } else { 0 };
        let (token_ids, _, _) = truncate_sequences(token_ids,
                                                   None,
                                                   num_truncated_tokens,
                                                   &TruncationStrategy::LongestFirst,
                                                   0).unwrap();
        Tensor::of_slice(&token_ids).unsqueeze(0).to(self.get_var_store().device())
    }

    fn generate(&self, prompt_text: Option<&str>, max_length: u64, do_sample: bool, num_beams: u64, temperature: f64,
                top_k: u64, top_p: f64, repetition_penalty: f64, length_penalty: f64, num_return_sequences: u64) -> Tensor {
        let input_ids = match prompt_text {
            Some(text) => self.encode_prompt_text(text, max_length),
            None => {
                Tensor::ones(&[1, 1], (Int64, self.get_var_store().device())) * *self.get_bos_id()
            }
        };

        assert!(temperature > 0f64, "temperature must positive");
        assert!((top_p >= 0f64) & (top_p <= 1f64), "top_p must be 0 and 1");
        assert!(repetition_penalty >= 1f64, "repetition_penalty must be greater than 1");
        assert!(length_penalty > 0f64, "length_penalty must be strictly greater than 1");
        assert!(num_return_sequences > 0u64, "num_return_sequences must be strictly greater than 0");
        assert!(num_beams > 0u64, "num_beams must be strictly greater than 0");

        if !do_sample {
            if num_beams == 1 {
                assert_eq!(num_return_sequences, 1, "num_return_sequences must be set to 1 for greedy decoding")
            } else {
                assert!(num_beams >= num_return_sequences, "num_return_sequences must be lower than the number of beams")
            }
        }

        let cur_len = *input_ids.size().last().unwrap();
        let vocab_size = self.get_tokenizer().vocab().values().len();

        let (effective_batch_size, effective_batch_mult) = match do_sample {
            true => (input_ids.size()[0] * num_return_sequences as i64, num_return_sequences as i64),
            false => (input_ids.size()[0], 1)
        };

        let input_ids = if (num_return_sequences > 1) | (num_beams > 1) {
            input_ids
                .unsqueeze(1)
                .expand(&[input_ids.size()[0], effective_batch_mult * num_beams as i64, cur_len], true)
                .contiguous()
                .view((effective_batch_size * num_beams as i64, cur_len))
        } else {
            input_ids
        };

        input_ids
    }
}