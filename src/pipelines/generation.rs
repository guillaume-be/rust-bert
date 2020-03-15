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
use tch::kind::Kind::{Int64, Float, Bool};
use std::collections::HashMap;
use std::cmp::{min, max};

pub struct OpenAIGenerator {
    model: OpenAIGPTLMHeadModel,
    tokenizer: OpenAiGptTokenizer,
    var_store: nn::VarStore,
    bos_token_id: Option<i64>,
    eos_token_ids: Option<Vec<i64>>,
    pad_token_id: Option<i64>,
}

impl OpenAIGenerator {
    pub fn new(vocab_path: &Path, merges_path: &Path, model_config_path: &Path, model_weight_path: &Path, device: Device)
               -> failure::Fallible<OpenAIGenerator> {
        let mut var_store = nn::VarStore::new(device);
        let tokenizer = OpenAiGptTokenizer::from_file(vocab_path.to_str().unwrap(), merges_path.to_str().unwrap(), true);
        let config = Gpt2Config::from_file(model_config_path);
        let model = OpenAIGPTLMHeadModel::new(&var_store.root(), &config);
        var_store.load(model_weight_path)?;

        let bos_token_id = None;
        let eos_token_ids = None;
        let pad_token_id = None;

        Ok(OpenAIGenerator { model, tokenizer, var_store, bos_token_id, eos_token_ids, pad_token_id })
    }
}

impl LanguageGenerator<OpenAIGPTLMHeadModel, OpenAiGptVocab, OpenAiGptTokenizer> for OpenAIGenerator {
    fn get_model(&self) -> &OpenAIGPTLMHeadModel { &self.model }
    fn get_tokenizer(&self) -> &OpenAiGptTokenizer { &self.tokenizer }
    fn get_var_store(&self) -> &nn::VarStore { &self.var_store }
    fn get_bos_id(&self) -> &Option<i64> { &self.bos_token_id }
    fn get_eos_ids(&self) -> &Option<Vec<i64>> { &self.eos_token_ids }
    fn get_pad_id(&self) -> &Option<i64> { &self.pad_token_id }
}

pub struct GPT2Generator {
    model: GPT2LMHeadModel,
    tokenizer: Gpt2Tokenizer,
    var_store: nn::VarStore,
    bos_token_id: Option<i64>,
    eos_token_ids: Option<Vec<i64>>,
    pad_token_id: Option<i64>,
}

impl GPT2Generator {
    pub fn new(vocab_path: &Path, merges_path: &Path, model_config_path: &Path, model_weight_path: &Path, device: Device)
               -> failure::Fallible<GPT2Generator> {
        let mut var_store = nn::VarStore::new(device);
        let tokenizer = Gpt2Tokenizer::from_file(vocab_path.to_str().unwrap(), merges_path.to_str().unwrap(), false);
        let config = Gpt2Config::from_file(model_config_path);
        let model = GPT2LMHeadModel::new(&var_store.root(), &config);
        var_store.load(model_weight_path)?;

        let bos_token_id = Some(tokenizer.vocab().token_to_id(Gpt2Vocab::bos_value()));
        let eos_token_ids = Some(vec!(tokenizer.vocab().token_to_id(Gpt2Vocab::eos_value())));
        let pad_token_id = None;

        Ok(GPT2Generator { model, tokenizer, var_store, bos_token_id, eos_token_ids, pad_token_id })
    }
}

impl LanguageGenerator<GPT2LMHeadModel, Gpt2Vocab, Gpt2Tokenizer> for GPT2Generator {
    fn get_model(&self) -> &GPT2LMHeadModel { &self.model }
    fn get_tokenizer(&self) -> &Gpt2Tokenizer { &self.tokenizer }
    fn get_var_store(&self) -> &nn::VarStore { &self.var_store }
    fn get_bos_id(&self) -> &Option<i64> { &self.bos_token_id }
    fn get_eos_ids(&self) -> &Option<Vec<i64>> { &self.eos_token_ids }
    fn get_pad_id(&self) -> &Option<i64> { &self.pad_token_id }

    fn prepare_inputs_for_generation(&self, input_ids: Tensor, past: Option<Vec<Tensor>>, _attention_mask: Tensor) -> (Tensor, Option<Vec<Tensor>>) {
        if past.is_some() {
            (input_ids.select(1, -1).unsqueeze(-1), past)
        } else {
            (input_ids, past)
        }
    }
}

pub trait LanguageGenerator<T: LMHeadModel, V: Vocab, U: Tokenizer<V>> {
    fn get_model(&self) -> &T;
    fn get_tokenizer(&self) -> &U;
    fn get_var_store(&self) -> &nn::VarStore;
    fn get_bos_id(&self) -> &Option<i64>;
    fn get_eos_ids(&self) -> &Option<Vec<i64>>;
    fn get_pad_id(&self) -> &Option<i64>;

    fn prepare_inputs_for_generation(&self, input_ids: Tensor, past: Option<Vec<Tensor>>, _attention_mask: Tensor) -> (Tensor, Option<Vec<Tensor>>) {
        (input_ids, past)
    }

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

    fn enforce_repetition_penalty(&self, next_token_logits: &mut Tensor, batch_size: i64, num_beams: u64, prev_output_tokens: &Tensor, repetition_penalty: f64) {
        for i in 0..(batch_size * num_beams as i64) {
            for token_position in 0..prev_output_tokens.get(i).size()[0] {
                let token = prev_output_tokens.get(i).int64_value(&[token_position]);
                let updated_value = &next_token_logits.double_value(&[i, token]);
                if updated_value < &0f64 {
                    &next_token_logits.get(i).index_fill_(0, &Tensor::of_slice(&[token]).to_kind(Int64).to_device(next_token_logits.device()), updated_value * repetition_penalty);
                } else {
                    &next_token_logits.get(i).index_fill_(0, &Tensor::of_slice(&[token]).to_kind(Int64).to_device(next_token_logits.device()), updated_value / repetition_penalty);
                }
            }
        }
    }

    fn get_banned_tokens(&self, input_ids: &Tensor, no_repeat_ngram_size: i64, cur_len: i64) -> Vec<Vec<i64>> {
//        Ported from hugging face's transformers and fairseq (https://github.com/pytorch/fairseq/blob/master/fairseq/sequence_generator.py)
        if cur_len + 1 < no_repeat_ngram_size {
            vec!(vec!())
        } else {
            let num_hypothesis = *input_ids.size().first().unwrap();
            let mut banned_tokens: Vec<Vec<i64>> = Vec::with_capacity(num_hypothesis as usize);
            for hypothesis_index in 0..num_hypothesis {
                let hypothesis_input_ids = input_ids.get(hypothesis_index);
                let mut generated_ngram: HashMap<Vec<i64>, Vec<i64>> = HashMap::new();
                let input: Vec<i64> = (0..hypothesis_input_ids.size1().unwrap()).collect();
                let query = hypothesis_input_ids
                    .slice(0,
                           cur_len + 1 - no_repeat_ngram_size,
                           *hypothesis_input_ids.size().last().unwrap(),
                           1).iter::<i64>()
                    .unwrap()
                    .collect::<Vec<i64>>();
                let ngram_indices: Vec<(i64, i64)> = input
                    .windows(3)
                    .map(|win| (*win.first().unwrap(), *win.last().unwrap()))
                    .collect();
                for ngram in ngram_indices.into_iter() {
                    let ngram = hypothesis_input_ids
                        .slice(0, ngram.0, ngram.1 + 1, 1)
                        .iter::<i64>()
                        .unwrap()
                        .collect::<Vec<i64>>();
                    let key = ngram[..ngram.len() - 1].to_vec();
                    let value = *ngram.last().unwrap();
                    if generated_ngram.contains_key(&key) {
                        generated_ngram.get_mut(&key).unwrap().push(value)
                    } else {
                        generated_ngram.insert(key, vec!(value));
                    }
                }
                let hypothesis_banned_tokens = match generated_ngram.get(&query) {
                    Some(banned_tokens) => banned_tokens.clone(),
                    None => vec!()
                };
                banned_tokens.push(hypothesis_banned_tokens);
            }
            banned_tokens
        }
    }

    fn top_k_top_p_filtering(&self, logits: &mut Tensor, top_k: i64, top_p: f64, min_tokens_to_keep: i64) {
//        Nucleus and top-k filtering introduced by Holtzman et al. (http://arxiv.org/abs/1904.09751)
//        Ported from https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        let vocab_size = *logits.size().last().unwrap();
        if top_k > 0 {
            let top_k = vocab_size - min(max(top_k, min_tokens_to_keep), vocab_size);
            let (_, indices_to_remove) = logits.topk(top_k, -1, false, false);
            for index in 0..*logits.size().first().unwrap() {
                &logits.get(index).index_fill_(0, &indices_to_remove.get(index), std::f64::NEG_INFINITY);
            }
        }

        if top_p < 1f64 {
            let (sorted_logits, sorted_indices) = logits.sort(-1, true);
            let cumulative_probabilities = sorted_logits.softmax(-1, Float).cumsum(-1, Float);
            let mut sorted_indices_to_remove = cumulative_probabilities.ge(top_p).to_kind(Int64);
            if min_tokens_to_keep > 1 {
                &sorted_indices_to_remove.index_fill_(1, &Tensor::arange1(0, min_tokens_to_keep + 1, (Int64, logits.device())), 0);
            }
            let _ = sorted_indices_to_remove.index_copy_(1,
                                                         &Tensor::arange1(1, vocab_size, (Int64, logits.device())),
                                                         &sorted_indices_to_remove.slice(1, 0, vocab_size - 1, 1).copy());
            let _ = sorted_indices_to_remove.index_fill_(1, &Tensor::of_slice(&[0]).to_kind(Int64).to_device(sorted_indices_to_remove.device()), 0);
            let indices_to_remove = sorted_indices_to_remove.scatter(1, &sorted_indices, &sorted_indices_to_remove).to_kind(Bool);
            let _ = logits.masked_fill_(&indices_to_remove, std::f64::NEG_INFINITY);
        }
    }

    fn generate(&self, prompt_text: Option<&str>, min_length: u64, max_length: u64, do_sample: bool, _early_stopping: bool, num_beams: u64, temperature: f64, top_k: u64,
                top_p: f64, repetition_penalty: f64, length_penalty: f64, no_repeat_ngram_size: u64, num_return_sequences: u64, attention_mask: Option<Tensor>)
                -> Vec<String> {
        let input_ids = match prompt_text {
            Some(text) => self.encode_prompt_text(text, max_length),
            None => match self.get_bos_id() {
                Some(bos_id) => Tensor::ones(&[1, 1], (Int64, self.get_var_store().device())) * *bos_id,
                None => panic!("A model with a BOS token must be used to start generation with an empty input")
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
        let batch_size = *input_ids.size().first().unwrap();
        let bos_token_id = *self.get_bos_id();
        let eos_token_ids = self.get_eos_ids().clone();

        let (effective_batch_size, effective_batch_mult) = match do_sample {
            true => (batch_size * num_return_sequences as i64, num_return_sequences as i64),
            false => (batch_size, 1)
        };

        let attention_mask = match attention_mask {
            Some(value) => value,
            None => {
                match self.get_pad_id() {
                    Some(pad_id) => input_ids.ne(*pad_id),
                    None => input_ids.ones_like()
                }
            }
        };

        let pad_token_id = match self.get_pad_id() {
            Some(value) => Some(*value),
            None => match &eos_token_ids {
                Some(eos_ids) => Some(eos_ids[0]),
                None => None
            }
        };

        let (input_ids, attention_mask) = if (num_return_sequences > 1) | (num_beams > 1) {
            (input_ids
                 .unsqueeze(1)
                 .expand(&[batch_size, effective_batch_mult * num_beams as i64, cur_len], true)
                 .contiguous()
                 .view((effective_batch_size * num_beams as i64, cur_len)),
             attention_mask
                 .unsqueeze(1)
                 .expand(&[batch_size, effective_batch_mult * num_beams as i64, cur_len], true)
                 .contiguous()
                 .view((effective_batch_size * num_beams as i64, cur_len))
            )
        } else {
            (input_ids, attention_mask)
        };

        let decoded = self.generate_no_beam_search(input_ids, cur_len, min_length, max_length, do_sample, temperature, top_k, top_p, repetition_penalty,
                                                   no_repeat_ngram_size, bos_token_id, pad_token_id, eos_token_ids, effective_batch_size, attention_mask);


        let num_sequences = *decoded.size().first().unwrap();
        let mut output = Vec::with_capacity(num_sequences as usize);
        for sequence_index in 0..num_sequences {
            output.push(self.get_tokenizer().decode(decoded
                                                        .as_ref()
                                                        .get(sequence_index)
                                                        .iter::<i64>()
                                                        .unwrap()
                                                        .collect::<Vec<i64>>(), true, true));
        }
        output
    }

    fn generate_no_beam_search(&self, input_ids: Tensor, cur_len: i64, min_length: u64, max_length: u64, do_sample: bool,
                               temperature: f64, top_k: u64, top_p: f64, repetition_penalty: f64, no_repeat_ngram_size: u64,
                               _bos_token_id: Option<i64>, pad_token_id: Option<i64>, eos_token_ids: Option<Vec<i64>>,
                               batch_size: i64, attention_mask: Tensor) -> Tensor {
        let mut unfinished_sentences = Tensor::ones(&[batch_size], (Int64, self.get_var_store().device()));
        let mut sentence_lengths: Tensor = Tensor::ones(&[batch_size], (Int64, self.get_var_store().device())) * max_length as i64;
        let mut attention_mask = attention_mask.copy();
        let mut input_ids = input_ids.copy();
        let mut past: Option<Vec<Tensor>> = None;
        let mut outputs: Tensor;
        let mut cur_len = cur_len as u64;

        while cur_len < max_length {
            let (prepared_input, prepared_past) = self.prepare_inputs_for_generation(input_ids.copy(), past, attention_mask.copy());
            let temp = self.get_model().forward_t(&Some(prepared_input), &prepared_past, &None, &None, &None, &None, false).unwrap();
            outputs = temp.0;
            past = temp.1;
            let mut next_token_logits = outputs.select(1, -1);

//            Reduce probability for repeated inputs
            if repetition_penalty > 1f64 {
                self.enforce_repetition_penalty(&mut next_token_logits, batch_size, 1, &input_ids, repetition_penalty)
            }

//            Get banned tokens and set their probability to 0
            let banned_tokens = self.get_banned_tokens(&input_ids, no_repeat_ngram_size as i64, cur_len as i64);
            for (batch_index, index_banned_token) in (0..banned_tokens.len() as i64).zip(banned_tokens) {
                &next_token_logits.get(batch_index).index_fill_(0, &Tensor::of_slice(&index_banned_token).to_device(next_token_logits.device()), std::f64::NEG_INFINITY);
            }

//            Do not allow eos token if min length is not reached
            if (&eos_token_ids.is_some()) & (cur_len < min_length) {
                &next_token_logits.index_fill_(1, &Tensor::of_slice(eos_token_ids.as_ref().unwrap()), std::f64::NEG_INFINITY);
            }

//            Top-k and top-p sampling
            let next_token = if do_sample {
                if temperature > 1f64 {
                    next_token_logits = next_token_logits / temperature;
                }
                self.top_k_top_p_filtering(&mut next_token_logits, top_k as i64, top_p, 1);
                let probabilities = next_token_logits.softmax(-1, Float);
                probabilities.multinomial(1, false).squeeze1(1)
            } else {
                next_token_logits.argmax(-1, false)
            };

//            Add tokens to unfinished sentences
            let tokens_to_add = match &eos_token_ids {
                Some(_) => next_token * &unfinished_sentences - pad_token_id.unwrap() * (&unfinished_sentences - 1),
                None => next_token
            };

            input_ids = Tensor::cat(&[input_ids, tokens_to_add.unsqueeze(-1)], -1);

            if eos_token_ids.is_some() {
                for eos_token_id in eos_token_ids.as_ref().unwrap() {
                    let sentence_with_eos = tokens_to_add.eq(*eos_token_id).to_kind(Int64);
                    let sentence_with_eos: Tensor = sentence_with_eos * &unfinished_sentences;
                    let _ = sentence_lengths.masked_fill_(&sentence_with_eos.to_kind(Bool).to_device(sentence_lengths.device()), cur_len as i64 + 1);
                    unfinished_sentences = -unfinished_sentences * (sentence_with_eos - 1);
                }
                if i64::from(unfinished_sentences.max()) == 0 {
                    break;
                }
            }

            attention_mask = Tensor::cat(&[attention_mask.as_ref(), Tensor::ones(&[*attention_mask.size().first().unwrap(), 1],
                                                                                 (Int64, attention_mask.device())).as_ref()], -1);
            cur_len += 1;
        }

        let decoded = if i64::from(&sentence_lengths.min().ne1(&sentence_lengths.max())) > 0 {
            match pad_token_id {
                Some(pad_value) => {
                    let decoded: Tensor = Tensor::ones(&[batch_size, i64::from(sentence_lengths.max())], (Int64, input_ids.device())) * pad_value;
                    for hypothesis_index in 0..*input_ids.size().first().unwrap() {
                        let _ = decoded.get(hypothesis_index).index_copy_(0,
                                                                          &Tensor::arange1(0,
                                                                                           i64::from(sentence_lengths.get(hypothesis_index)),
                                                                                           (Int64, input_ids.device())),
                                                                          &input_ids.get(hypothesis_index));
                    }
                    decoded
                }
                None => input_ids
            }
        } else {
            input_ids
        };
        decoded
    }
}