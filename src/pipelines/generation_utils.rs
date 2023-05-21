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

//! # Natural Language Generation utilities
//! Set of text generation utilities, serving as a basis for TextGenerationModel, SummarizationModels and TranslationModels.
//! Include techniques such as beam search, top-k and nucleus sampling, temperature setting and repetition penalty.
//! Supports batch generation of sentences from several prompts. Sequences will be left-padded with the model's padding token if present, the unknown token otherwise.
//! This may impact the results and it is recommended to submit prompts of similar length for best results.
//!
//! ```no_run
//! # fn main() -> anyhow::Result<()> {
//! use rust_bert::gpt2::GPT2Generator;
//! use rust_bert::pipelines::generation_utils::{
//!     GenerateConfig, GenerateOptions, LanguageGenerator,
//! };
//!
//! let generate_config = GenerateConfig {
//!     do_sample: true,
//!     num_beams: 5,
//!     temperature: 1.1,
//!     num_return_sequences: 3,
//!     ..Default::default()
//! };
//! let mut gpt2_generator = GPT2Generator::new(generate_config)?;
//!
//! let input_context = "The dog";
//! let second_input_context = "The cat was";
//!
//! let generate_options = GenerateOptions {
//!     min_length: Some(32),
//!     max_length: Some(128),
//!     output_scores: true,
//!     ..Default::default()
//! };
//!
//! let output = gpt2_generator.generate(
//!     Some(&[input_context, second_input_context]),
//!     Some(generate_options),
//! );
//! # Ok(())
//! # }
//! ```
//!
//! Example output: \
//! ```no_run
//! # let output =
//! [
//!     "The dog's owners, however, did not want to be named. According to the lawsuit, the animal's owner, a 29-year",
//!     "The dog has always been part of the family. \"He was always going to be my dog and he was always looking out for me",
//!     "The dog has been able to stay in the home for more than three months now. \"It's a very good dog. She's",
//!     "The cat was discovered earlier this month in the home of a relative of the deceased. The cat\'s owner, who wished to remain anonymous,",
//!     "The cat was pulled from the street by two-year-old Jazmine.\"I didn't know what to do,\" she said",
//!     "The cat was attacked by two stray dogs and was taken to a hospital. Two other cats were also injured in the attack and are being treated."
//! ]
//! # ;
//! ```

use tch::kind::Kind::Int64;
use tch::{no_grad, Device, Tensor};

use crate::bart::LayerState as BartLayerState;
use crate::common::resources::ResourceProvider;
use crate::gpt_j::LayerState as GPTJLayerState;
use crate::gpt_neo::LayerState as GPTNeoLayerState;
use crate::pipelines::generation_utils::private_generation_utils::{
    InternalGenerateOptions, PrivateLanguageGenerator,
};
use crate::prophetnet::LayerState as ProphetNetLayerState;
use crate::reformer::LayerState as ReformerLayerState;
use crate::t5::LayerState as T5LayerState;
use crate::xlnet::LayerState as XLNetLayerState;

use self::ordered_float::OrderedFloat;
use crate::pipelines::common::TokenizerOption;

#[cfg(feature = "remote")]
use crate::{
    gpt2::{Gpt2ConfigResources, Gpt2MergesResources, Gpt2ModelResources, Gpt2VocabResources},
    resources::RemoteResource,
};

extern crate ordered_float;

/// # Configuration for text generation
pub struct GenerateConfig {
    /// Model weights resource (default: pretrained GPT2 model)
    pub model_resource: Box<dyn ResourceProvider + Send>,
    /// Config resource (default: pretrained GPT2 model)
    pub config_resource: Box<dyn ResourceProvider + Send>,
    /// Vocab resource (default: pretrained GPT2 model)
    pub vocab_resource: Box<dyn ResourceProvider + Send>,
    /// Merges resource (default: pretrained GPT2 model)
    pub merges_resource: Option<Box<dyn ResourceProvider + Send>>,
    /// Minimum sequence length (default: 0)
    pub min_length: i64,
    /// Maximum sequence length (default: 20)
    pub max_length: Option<i64>,
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
impl Default for GenerateConfig {
    fn default() -> GenerateConfig {
        GenerateConfig {
            model_resource: Box::new(RemoteResource::from_pretrained(Gpt2ModelResources::GPT2)),
            config_resource: Box::new(RemoteResource::from_pretrained(Gpt2ConfigResources::GPT2)),
            vocab_resource: Box::new(RemoteResource::from_pretrained(Gpt2VocabResources::GPT2)),
            merges_resource: Some(Box::new(RemoteResource::from_pretrained(
                Gpt2MergesResources::GPT2,
            ))),
            min_length: 0,
            max_length: Some(56),
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

impl GenerateConfig {
    pub(crate) fn validate(&self) {
        assert!(self.temperature > 0f64, "temperature must positive");
        assert!(
            (self.top_p >= 0f64) & (self.top_p <= 1f64),
            "top_p must be 0 and 1"
        );
        assert!(
            self.repetition_penalty >= 1f64,
            "repetition_penalty must be greater than 1"
        );
        assert!(
            self.length_penalty > 0f64,
            "length_penalty must be strictly greater than 0"
        );
        assert!(
            self.num_return_sequences > 0i64,
            "num_return_sequences must be strictly greater than 0"
        );
        assert!(
            self.num_beams > 0i64,
            "num_beams must be strictly greater than 0"
        );

        if !self.do_sample {
            if self.num_beams == 1 {
                assert_eq!(
                    self.num_return_sequences, 1,
                    "num_return_sequences must be set to 1 for greedy decoding"
                )
            } else {
                assert!(
                    self.num_beams >= self.num_return_sequences,
                    "num_return_sequences must be lower than the number of beams"
                )
            }
        }
        if let Some(num_beam_groups_value) = self.num_beam_groups {
            if num_beam_groups_value > 1 {
                assert_eq!(
                    self.num_beams % num_beam_groups_value,
                    0,
                    "num_beam_groups must be a multiple of num_beam_groups"
                )
            }
        }
    }
}

#[derive(Debug)]
pub enum Cache {
    GPT2Cache(Option<Vec<Tensor>>),
    BARTCache(Option<Vec<(Option<BartLayerState>, Option<BartLayerState>)>>),
    T5Cache(Option<Vec<(Option<T5LayerState>, Option<T5LayerState>)>>),
    LongT5Cache(Option<Vec<(Option<T5LayerState>, Option<T5LayerState>)>>),
    XLNetCache(Option<Vec<Option<XLNetLayerState>>>),
    ReformerCache(Option<Vec<Option<ReformerLayerState>>>),
    ProphetNetCache(Option<Vec<(Option<ProphetNetLayerState>, Option<ProphetNetLayerState>)>>),
    GPTNeoCache(Option<Vec<Option<GPTNeoLayerState>>>),
    GPTJCache(Option<Vec<Option<GPTJLayerState>>>),
    None,
}

pub(crate) mod private_generation_utils {
    use std::cmp::{max, min};
    use std::collections::HashMap;
    use std::convert::TryFrom;
    use std::mem;

    use rust_tokenizers::tokenizer::{truncate_sequences, TruncationStrategy};
    use rust_tokenizers::TokenIdsWithOffsets;
    use tch::{nn, Device, Kind, Tensor};

    use crate::pipelines::common::TokenizerOption;
    use crate::pipelines::generation_utils::{
        BeamHypotheses, Cache, GenerateConfig, LMModelOutput, PrefixAllowedFunction,
    };

    use super::ordered_float::OrderedFloat;
    use crate::common::kind::get_positive_infinity;
    use crate::RustBertError;

    pub struct InternalGenerateOptions<'a> {
        pub min_length: i64,
        pub max_length: Option<i64>,
        pub do_sample: bool,
        pub temperature: f64,
        pub top_k: i64,
        pub top_p: f64,
        pub repetition_penalty: f64,
        pub no_repeat_ngram_size: i64,
        pub pad_token_id: Option<i64>,
        pub eos_token_ids: Option<Vec<i64>>,
        pub num_return_sequences: i64,
        pub early_stopping: bool,
        pub num_beams: i64,
        pub length_penalty: f64,
        pub num_beam_groups: Option<i64>,
        pub diversity_penalty: Option<f64>,
        pub forced_bos_token_id: Option<i64>,
        pub bad_word_ids: Option<&'a Vec<Vec<i64>>>,
    }

    pub struct PreparedInput<'a> {
        pub prepared_input: Option<Tensor>,
        pub prepared_attention_mask: Option<Tensor>,
        pub prepared_encoder_output: Option<&'a Tensor>,
        pub prepared_decoder_input: Option<Tensor>,
        pub prepared_position_ids: Option<Tensor>,
        pub prepared_past: Cache,
    }

    pub struct GeneratedOutputWithScores {
        pub indices: Tensor,
        pub scores: Option<Vec<f64>>,
        pub token_scores: Option<Vec<Vec<f64>>>,
    }

    pub trait PrivateLanguageGenerator {
        fn _get_tokenizer(&self) -> &TokenizerOption;
        fn _get_tokenizer_mut(&mut self) -> &mut TokenizerOption;
        fn get_var_store(&self) -> &nn::VarStore;
        fn get_var_store_mut(&mut self) -> &mut nn::VarStore;
        fn get_config(&self) -> &GenerateConfig;
        fn get_bos_id(&self) -> Option<i64>;
        fn get_eos_ids(&self) -> Option<&Vec<i64>>;
        fn get_pad_id(&self) -> Option<i64>;
        fn is_encoder_decoder(&self) -> bool;
        fn get_vocab_size(&self) -> i64;
        fn get_decoder_start_id(&self) -> Option<i64>;
        fn get_max_positions_embeddings(&self) -> i64;

        fn forward_t(
            &self,
            input_ids: Option<&Tensor>,
            layer_past: Cache,
            attention_mask: Option<&Tensor>,
            token_type_ids: Option<&Tensor>,
            position_ids: Option<&Tensor>,
            input_embeds: Option<&Tensor>,
            encoder_outputs: Option<&Tensor>,
            decoder_input_ids: Option<&Tensor>,
            train: bool,
        ) -> Result<LMModelOutput, RustBertError>;

        fn prepare_scores_for_generation(
            &self,
            _scores: &mut Tensor,
            _current_length: i64,
            _max_length: Option<i64>,
            _forced_bos_token_id: Option<i64>,
        ) {
        }

        fn encode(&self, _input_ids: &Tensor, _attention_mask: Option<&Tensor>) -> Option<Tensor> {
            None
        }

        fn prepare_inputs_for_generation<'a>(
            &self,
            input_ids: Tensor,
            _encoder_outputs: Option<&'a Tensor>,
            past: Cache,
            attention_mask: Tensor,
        ) -> PreparedInput<'a> {
            PreparedInput {
                prepared_input: Some(input_ids),
                prepared_attention_mask: Some(attention_mask),
                prepared_encoder_output: None,
                prepared_decoder_input: None,
                prepared_position_ids: None,
                prepared_past: past,
            }
        }

        fn encode_prompt_text<S>(
            &self,
            prompt_text: &[S],
            max_len: Option<i64>,
            pad_token_id: Option<i64>,
        ) -> Tensor
        where
            S: AsRef<str> + Sync,
        {
            let tokens = self._get_tokenizer().tokenize_list(prompt_text);
            let token_ids = tokens
                .into_iter()
                .map(|prompt_tokens| self._get_tokenizer().convert_tokens_to_ids(&prompt_tokens))
                .collect::<Vec<Vec<i64>>>();

            let num_truncated_tokens = token_ids
                .iter()
                .map(|token_ids| {
                    max_len
                        .map(|max_len| {
                            if token_ids.len() > max_len as usize {
                                token_ids.len() - max_len as usize
                            } else {
                                0
                            }
                        })
                        .unwrap_or(0)
                })
                .collect::<Vec<usize>>();

            let token_ids = token_ids
                .into_iter()
                .zip(num_truncated_tokens)
                .map(|(tokens, num_truncated_tokens)| {
                    truncate_sequences(
                        TokenIdsWithOffsets {
                            ids: tokens,
                            offsets: vec![],
                            reference_offsets: vec![],
                            masks: vec![],
                        },
                        None,
                        num_truncated_tokens,
                        &TruncationStrategy::LongestFirst,
                        0,
                    )
                    .unwrap()
                    .0
                    .ids
                })
                .collect::<Vec<Vec<i64>>>();

            let max_len = token_ids.iter().map(|input| input.len()).max().unwrap();

            let pad_token = match pad_token_id {
                Some(value) => value,
                None => self._get_tokenizer().get_unk_id(),
            };

            let token_ids = token_ids
                .into_iter()
                .map(|input| {
                    let mut temp = vec![pad_token; max_len - input.len()];
                    temp.extend(input);
                    temp
                })
                .map(|tokens| Tensor::from_slice(&tokens).to(self.get_var_store().device()))
                .collect::<Vec<Tensor>>();
            Tensor::stack(&token_ids, 0)
        }

        fn enforce_repetition_penalty(
            &self,
            next_token_logits: &mut Tensor,
            batch_size: i64,
            num_beams: i64,
            prev_output_tokens: &Tensor,
            repetition_penalty: f64,
        ) {
            for i in 0..(batch_size * num_beams) {
                for token_position in 0..prev_output_tokens.get(i).size()[0] {
                    let token = prev_output_tokens.get(i).int64_value(&[token_position]);
                    let updated_value = &next_token_logits.double_value(&[i, token]);
                    if updated_value < &0f64 {
                        let _ = next_token_logits.get(i).index_fill_(
                            0,
                            &Tensor::from_slice(&[token])
                                .to_kind(Kind::Int64)
                                .to_device(next_token_logits.device()),
                            updated_value * repetition_penalty,
                        );
                    } else {
                        let _ = next_token_logits.get(i).index_fill_(
                            0,
                            &Tensor::from_slice(&[token])
                                .to_kind(Kind::Int64)
                                .to_device(next_token_logits.device()),
                            updated_value / repetition_penalty,
                        );
                    }
                }
            }
        }

        fn get_banned_tokens(
            &self,
            input_ids: &Tensor,
            no_repeat_ngram_size: i64,
            cur_len: i64,
        ) -> Vec<Vec<i64>> {
            //        Ported from hugging face's transformers and fairseq (https://github.com/pytorch/fairseq/blob/master/fairseq/sequence_generator.py)
            if cur_len + 1 < no_repeat_ngram_size {
                vec![vec![]]
            } else {
                let input_ids = input_ids.to(Device::Cpu);
                let num_hypothesis = *input_ids.size().first().unwrap();
                let mut banned_tokens: Vec<Vec<i64>> = Vec::with_capacity(num_hypothesis as usize);
                for hypothesis_index in 0..num_hypothesis {
                    let hypothesis_input_ids = input_ids.get(hypothesis_index);
                    let mut generated_ngram: HashMap<Vec<i64>, Vec<i64>> = HashMap::new();
                    let input: Vec<i64> = (0..hypothesis_input_ids.size1().unwrap()).collect();
                    let hypothesis_input_ids = hypothesis_input_ids
                        .iter::<i64>()
                        .unwrap()
                        .collect::<Vec<i64>>();
                    let query = &hypothesis_input_ids
                        [cur_len as usize + 1 - no_repeat_ngram_size as usize..]
                        .to_vec();
                    for ngram in input
                        .windows(no_repeat_ngram_size as usize)
                        .map(|win| (*win.first().unwrap(), *win.last().unwrap()))
                    {
                        let ngram = &hypothesis_input_ids[ngram.0 as usize..ngram.1 as usize + 1];
                        let key = ngram[..no_repeat_ngram_size as usize - 1].to_vec();
                        let value = *ngram.last().unwrap();
                        generated_ngram
                            .entry(key)
                            .or_insert_with(|| vec![value])
                            .push(value);
                    }
                    let hypothesis_banned_tokens = match generated_ngram.get(query) {
                        Some(banned_tokens) => banned_tokens.clone(),
                        None => vec![],
                    };
                    banned_tokens.push(hypothesis_banned_tokens);
                }
                banned_tokens
            }
        }

        fn top_k_top_p_filtering(
            &self,
            logits: &mut Tensor,
            top_k: i64,
            top_p: f64,
            min_tokens_to_keep: i64,
        ) {
            //        Nucleus and top-k filtering introduced by Holtzman et al. (http://arxiv.org/abs/1904.09751)
            //        Ported from https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
            let vocab_size = *logits.size().last().unwrap();
            if top_k > 0 {
                let top_k = vocab_size - min(max(top_k, min_tokens_to_keep), vocab_size);
                let (_, indices_to_remove) = logits.topk(top_k, -1, false, false);
                for index in 0..*logits.size().first().unwrap() {
                    let _ = logits.get(index).index_fill_(
                        0,
                        &indices_to_remove.get(index),
                        f64::NEG_INFINITY,
                    );
                }
            }
            if top_p < 1f64 {
                let (sorted_logits, sorted_indices) = logits.sort(-1, true);
                let cumulative_probabilities = sorted_logits
                    .softmax(-1, sorted_logits.kind())
                    .cumsum(-1, sorted_logits.kind());
                let mut sorted_indices_to_remove =
                    cumulative_probabilities.ge(top_p).to_kind(Kind::Int64);
                if min_tokens_to_keep > 1 {
                    let _ = sorted_indices_to_remove.index_fill_(
                        1,
                        &Tensor::arange_start(
                            0,
                            min_tokens_to_keep + 1,
                            (Kind::Int64, logits.device()),
                        ),
                        0,
                    );
                }
                let _ = sorted_indices_to_remove.index_copy_(
                    1,
                    &Tensor::arange_start(1, vocab_size, (Kind::Int64, logits.device())),
                    &sorted_indices_to_remove
                        .slice(1, 0, vocab_size - 1, 1)
                        .copy(),
                );
                let _ = sorted_indices_to_remove.index_fill_(
                    1,
                    &Tensor::from_slice(&[0])
                        .to_kind(Kind::Int64)
                        .to_device(sorted_indices_to_remove.device()),
                    0,
                );
                let indices_to_remove = sorted_indices_to_remove
                    .scatter(1, &sorted_indices, &sorted_indices_to_remove)
                    .to_kind(Kind::Bool);
                let _ = logits.masked_fill_(&indices_to_remove, f64::NEG_INFINITY);
            }
        }

        fn run_hamming_diversity_penalty(
            &self,
            scores: &mut Tensor,
            current_tokens: &Tensor,
            diversity_penalty: f64,
            num_beams: i64,
            batch_size: i64,
            group_size: i64,
            group_start_index: i64,
        ) {
            if group_start_index > 0 {
                let vocab_size = *scores.size().last().unwrap();
                for batch_index in 0..batch_size {
                    let previous_group_tokens = current_tokens.slice(
                        0,
                        batch_index * num_beams,
                        batch_index * num_beams + group_start_index,
                        1,
                    );
                    let diversity_penalty = previous_group_tokens
                        .bincount::<Tensor>(None, vocab_size)
                        * diversity_penalty;
                    let _ = scores
                        .slice(
                            0,
                            batch_index * group_size,
                            (batch_index + 1) * group_size,
                            1,
                        )
                        .subtract_(&diversity_penalty);
                }
            }
        }

        fn apply_prefix_allowed_tokens_function(
            &self,
            prefix_allowed_tokens_fn: &dyn Fn(i64, &Tensor) -> Vec<i64>,
            num_beams: i64,
            input_ids: &Tensor,
            scores: &mut Tensor,
        ) {
            let mask = scores.new_full(
                scores.size().as_slice(),
                get_positive_infinity(scores.kind()).unwrap(),
                (scores.kind(), scores.device()),
            );
            for idx in 0..scores.size()[0] {
                let batch_id = idx / num_beams;
                let allowed_tokens: Vec<i64> =
                    prefix_allowed_tokens_fn(batch_id, &input_ids.get(idx));
                let _ = mask.get(idx).index_fill_(
                    0,
                    &Tensor::from_slice(allowed_tokens.as_slice()).to(scores.device()),
                    0,
                );
            }
            let _ = scores.subtract_(&mask);
        }

        fn split_bad_word_ids<'a>(
            &self,
            bad_word_ids: Option<&'a Vec<Vec<i64>>>,
        ) -> (Option<Vec<i64>>, Option<Vec<&'a Vec<i64>>>) {
            if let Some(bad_word_ids) = bad_word_ids {
                let mut bad_word_ids_length_1 = vec![];
                let mut bad_word_ids_length_greater_than_1 = vec![];
                for bad_word in bad_word_ids {
                    if bad_word.len() == 1 {
                        bad_word_ids_length_1.push(bad_word[0]);
                    } else {
                        bad_word_ids_length_greater_than_1.push(bad_word);
                    }
                }
                let bad_word_ids_length_1 = if !bad_word_ids_length_1.is_empty() {
                    Some(bad_word_ids_length_1)
                } else {
                    None
                };
                let bad_word_ids_length_greater_than_1 =
                    if !bad_word_ids_length_greater_than_1.is_empty() {
                        Some(bad_word_ids_length_greater_than_1)
                    } else {
                        None
                    };
                (bad_word_ids_length_1, bad_word_ids_length_greater_than_1)
            } else {
                (None, None)
            }
        }

        fn tokens_match(&self, prev_tokens: &[i64], tokens: &[i64]) -> bool {
            if tokens.is_empty() {
                true
            } else if tokens.len() > prev_tokens.len() {
                false
            } else {
                &prev_tokens[prev_tokens.len() - tokens.len()..] == tokens
            }
        }

        fn calc_static_bad_word_mask(
            &self,
            scores: &Tensor,
            bad_words_id_length_1: &[i64],
        ) -> Tensor {
            let mut static_bad_words_mask =
                Tensor::zeros([scores.size()[1]], (Kind::Int8, scores.device()));
            let _ = static_bad_words_mask.index_fill_(
                0,
                &Tensor::from_slice(bad_words_id_length_1).to_device(scores.device()),
                1,
            );
            static_bad_words_mask.unsqueeze(0).totype(Kind::Bool)
        }

        fn get_dynamic_bad_word_ids(
            &self,
            prev_tokens: &[Vec<i64>],
            bad_word_ids_length_greater_than_1: &[&Vec<i64>],
        ) -> Vec<Vec<i64>> {
            let mut banned_tokens = Vec::new();
            for prev_token_sequence in prev_tokens {
                let mut sequence_banned_tokens = Vec::new();
                for bad_word_ids in bad_word_ids_length_greater_than_1 {
                    if self
                        .tokens_match(prev_token_sequence, &bad_word_ids[..bad_word_ids.len() - 1])
                    {
                        sequence_banned_tokens.push(*bad_word_ids.last().unwrap());
                    }
                }
                banned_tokens.push(sequence_banned_tokens);
            }

            banned_tokens
        }

        fn ban_bad_words(
            &self,
            dynamic_bad_words: Option<&Vec<&Vec<i64>>>,
            static_bad_words_mask: Option<&Tensor>,
            token_ids: &Tensor,
            scores: &mut Tensor,
        ) {
            let longest_bad_word = dynamic_bad_words
                .iter()
                .map(|bad_word| bad_word.len())
                .max()
                .unwrap() as i64;

            let last_token_ids = token_ids.slice(1, -longest_bad_word, None, 1);
            let mut prev_tokens = Vec::new();
            for sequence_idx in 0..token_ids.size()[0] {
                prev_tokens.push(
                    last_token_ids
                        .get(sequence_idx)
                        .iter::<i64>()
                        .unwrap()
                        .collect::<Vec<i64>>(),
                )
            }

            let dynamic_bad_words_mask = if let Some(dynamic_bad_words) = dynamic_bad_words {
                let dynamic_banned_tokens =
                    self.get_dynamic_bad_word_ids(&prev_tokens, dynamic_bad_words);
                let dynamic_banned_mask =
                    Tensor::zeros(scores.size().as_slice(), (Kind::Int, scores.device()));
                for (sequence_index, sequence_ban_tokens) in
                    dynamic_banned_tokens.iter().enumerate()
                {
                    if !sequence_ban_tokens.is_empty() {
                        let _ = dynamic_banned_mask.get(sequence_index as i64).index_fill_(
                            0,
                            &Tensor::from_slice(sequence_ban_tokens).to_device(scores.device()),
                            1,
                        );
                    }
                }
                Some(dynamic_banned_mask.to_kind(Kind::Bool))
            } else {
                None
            };

            let combined_bad_word_mask = {
                if let (Some(static_mask), Some(dynamic_mask)) =
                    (static_bad_words_mask, &dynamic_bad_words_mask)
                {
                    Some(static_mask.bitwise_or_tensor(dynamic_mask))
                } else {
                    None
                }
            };

            let bad_word_mask = if combined_bad_word_mask.is_some() {
                combined_bad_word_mask.as_ref()
            } else if static_bad_words_mask.is_some() {
                static_bad_words_mask
            } else if dynamic_bad_words_mask.is_some() {
                dynamic_bad_words_mask.as_ref()
            } else {
                None
            };

            if let Some(bad_word_mask) = bad_word_mask {
                let _ = scores.masked_fill_(bad_word_mask, f64::NEG_INFINITY);
            }
        }

        fn generate_no_beam_search(
            &self,
            input_ids: Tensor,
            encoder_outputs: Option<Tensor>,
            cur_len: i64,
            batch_size: i64,
            attention_mask: Tensor,
            gen_opt: InternalGenerateOptions,
            prefix_allowed_tokens_fn: Option<PrefixAllowedFunction>,
            output_scores: bool,
        ) -> GeneratedOutputWithScores {
            let mut unfinished_sentences =
                Tensor::ones([batch_size], (Kind::Int64, self.get_var_store().device()));
            let mut sentence_lengths: Tensor =
                Tensor::ones([batch_size], (Kind::Int64, self.get_var_store().device()));
            let (bad_word_ids_length_1, bad_word_ids_length_greater_than_1) =
                self.split_bad_word_ids(gen_opt.bad_word_ids);
            let mut static_bad_words_mask: Option<Tensor> = None;
            let mut attention_mask = attention_mask.copy();
            let mut input_ids = input_ids.copy();
            let mut past: Cache = Cache::None;
            let mut outputs: Tensor;
            let mut current_length = cur_len;
            let mut token_scores_output: Option<Vec<Tensor>> =
                if output_scores { Some(vec![]) } else { None };

            loop {
                let prepared_input = self.prepare_inputs_for_generation(
                    input_ids.copy(),
                    encoder_outputs.as_ref(),
                    past,
                    attention_mask.copy(),
                );
                let temp = self
                    .forward_t(
                        prepared_input.prepared_input.as_ref(),
                        prepared_input.prepared_past,
                        prepared_input.prepared_attention_mask.as_ref(),
                        None,
                        prepared_input.prepared_position_ids.as_ref(),
                        None,
                        prepared_input.prepared_encoder_output,
                        prepared_input.prepared_decoder_input.as_ref(),
                        false,
                    )
                    .unwrap();
                outputs = temp.lm_logits;
                past = temp.cache;

                let mut next_token_logits = outputs.select(1, -1);
                // Reduce probability for repeated inputs
                if gen_opt.repetition_penalty > 1f64 {
                    self.enforce_repetition_penalty(
                        &mut next_token_logits,
                        batch_size,
                        1,
                        &input_ids,
                        gen_opt.repetition_penalty,
                    )
                }

                // Get bad word_ids and set their probability to 0
                if gen_opt.bad_word_ids.is_some() {
                    // Calculate static bad words masks if not set yet
                    if let Some(bad_word_ids_length_1) = &bad_word_ids_length_1 {
                        if static_bad_words_mask.is_none() {
                            static_bad_words_mask = Some(self.calc_static_bad_word_mask(
                                &next_token_logits,
                                bad_word_ids_length_1,
                            ));
                        }
                    }
                    self.ban_bad_words(
                        bad_word_ids_length_greater_than_1.as_ref(),
                        static_bad_words_mask.as_ref(),
                        &input_ids,
                        &mut next_token_logits,
                    );
                }

                // Get banned tokens and set their probability to 0
                if gen_opt.no_repeat_ngram_size > 0 {
                    let banned_tokens = self.get_banned_tokens(
                        &input_ids,
                        gen_opt.no_repeat_ngram_size,
                        current_length,
                    );
                    for (batch_index, index_banned_token) in
                        (0..banned_tokens.len() as i64).zip(banned_tokens)
                    {
                        let _ = next_token_logits.get(batch_index).index_fill_(
                            0,
                            &Tensor::from_slice(&index_banned_token)
                                .to_device(next_token_logits.device()),
                            f64::NEG_INFINITY,
                        );
                    }
                }

                // Apply custom prefix constraint function
                if let Some(prefix_allowed_tokens_function) = prefix_allowed_tokens_fn {
                    self.apply_prefix_allowed_tokens_function(
                        prefix_allowed_tokens_function,
                        1,
                        &input_ids,
                        &mut next_token_logits,
                    )
                }

                // Do not allow eos token if min length is not reached
                if (gen_opt.eos_token_ids.is_some()) & (current_length < gen_opt.min_length) {
                    let _ = next_token_logits.index_fill_(
                        1,
                        &Tensor::from_slice(gen_opt.eos_token_ids.as_ref().unwrap())
                            .to(next_token_logits.device()),
                        f64::NEG_INFINITY,
                    );
                }

                self.prepare_scores_for_generation(
                    &mut next_token_logits,
                    current_length,
                    gen_opt.max_length,
                    gen_opt.forced_bos_token_id,
                );

                // Top-k and top-p sampling
                let next_token = if gen_opt.do_sample {
                    if gen_opt.temperature > 1f64 {
                        next_token_logits /= gen_opt.temperature;
                    }
                    self.top_k_top_p_filtering(
                        &mut next_token_logits,
                        gen_opt.top_k,
                        gen_opt.top_p,
                        1,
                    );
                    let probabilities = next_token_logits.softmax(-1, next_token_logits.kind());
                    probabilities.multinomial(1, false).squeeze_dim(1)
                } else {
                    next_token_logits.argmax(-1, false)
                };

                if let Some(prev_scores) = token_scores_output.as_mut() {
                    let finished_mask = unfinished_sentences.eq(0);
                    prev_scores.push(
                        next_token_logits
                            .log_softmax(-1, next_token_logits.kind())
                            .gather(1, &next_token.reshape([-1, 1]), true)
                            .squeeze()
                            .masked_fill(&finished_mask, 0),
                    );
                };

                // Add tokens to unfinished sentences
                let tokens_to_add = match &gen_opt.eos_token_ids {
                    Some(_) => {
                        next_token * &unfinished_sentences
                            - gen_opt.pad_token_id.unwrap() * (&unfinished_sentences - 1)
                    }
                    None => next_token,
                };

                input_ids = Tensor::cat(&[input_ids, tokens_to_add.unsqueeze(-1)], -1);
                if gen_opt.eos_token_ids.is_some() {
                    for eos_token_id in gen_opt.eos_token_ids.as_ref().unwrap() {
                        let sentence_with_eos =
                            tokens_to_add.eq(*eos_token_id).to_kind(Kind::Int64);
                        let sentence_with_eos: Tensor = sentence_with_eos * &unfinished_sentences;
                        let _ = sentence_lengths.masked_fill_(
                            &sentence_with_eos
                                .to_kind(Kind::Bool)
                                .to_device(sentence_lengths.device()),
                            current_length + 1,
                        );
                        unfinished_sentences = -unfinished_sentences * (sentence_with_eos - 1);
                    }
                    if i64::try_from(unfinished_sentences.max()).unwrap() == 0 {
                        break;
                    }
                }
                if !self.is_encoder_decoder() {
                    attention_mask = Tensor::cat(
                        &[
                            attention_mask.as_ref(),
                            Tensor::ones(
                                [*attention_mask.size().first().unwrap(), 1],
                                (Kind::Int64, attention_mask.device()),
                            )
                            .as_ref(),
                        ],
                        -1,
                    );
                }
                current_length += 1;
                if let Some(max_length) = gen_opt.max_length {
                    if current_length >= max_length {
                        let _ = sentence_lengths.masked_fill_(
                            &unfinished_sentences
                                .to_kind(Kind::Bool)
                                .to_device(sentence_lengths.device()),
                            current_length,
                        );
                        break;
                    }
                }
            }
            let scores_output = token_scores_output.as_ref().map(|scores_tensor| {
                (Tensor::stack(scores_tensor, 1).sum_dim_intlist(
                    [1].as_slice(),
                    false,
                    Kind::Float,
                ) / sentence_lengths.pow_tensor_scalar(gen_opt.length_penalty))
                .iter::<f64>()
                .unwrap()
                .collect::<Vec<f64>>()
            });
            let token_scores_output = token_scores_output.map(|score_tensors| {
                Tensor::stack(&score_tensors, 1)
                    .split(1, 0)
                    .iter()
                    .map(|sequence_scores| {
                        sequence_scores
                            .squeeze_dim(0)
                            .iter::<f64>()
                            .unwrap()
                            .collect::<Vec<f64>>()
                    })
                    .collect()
            });
            GeneratedOutputWithScores {
                indices: input_ids,
                scores: scores_output,
                token_scores: token_scores_output,
            }
        }

        fn generate_beam_search(
            &self,
            mut input_ids: Tensor,
            encoder_outputs: Option<Tensor>,
            cur_len: i64,
            batch_size: i64,
            mut attention_mask: Tensor,
            gen_opt: InternalGenerateOptions,
            prefix_allowed_tokens_fn: Option<PrefixAllowedFunction>,
            output_scores: bool,
        ) -> GeneratedOutputWithScores {
            let num_beam_groups = gen_opt.num_beam_groups.unwrap_or(1);
            let num_sub_beams = gen_opt.num_beams / num_beam_groups;
            let diversity_penalty = gen_opt.diversity_penalty.unwrap_or(5.5);
            let (bad_word_ids_length_1, bad_word_ids_length_greater_than_1) =
                self.split_bad_word_ids(gen_opt.bad_word_ids);
            let mut static_bad_words_mask: Option<Tensor> = None;

            let mut hypotheses = (0..batch_size)
                .map(|_| {
                    BeamHypotheses::new(
                        gen_opt.num_beams,
                        gen_opt.max_length,
                        gen_opt.length_penalty,
                        gen_opt.early_stopping,
                    )
                })
                .collect::<Vec<BeamHypotheses>>();

            let vocab_size = self.get_vocab_size();
            let beam_scores = Tensor::ones(
                [batch_size, gen_opt.num_beams],
                (Kind::Float, self.get_var_store().device()),
            ) * -1e9;
            let _ = beam_scores
                .slice(1, 0, *beam_scores.size().last().unwrap(), num_sub_beams)
                .fill_(0);

            let mut beam_scores = beam_scores.view_([-1]);
            let mut beam_tokens = Tensor::zeros(
                [batch_size * gen_opt.num_beams],
                (Kind::Int64, self.get_var_store().device()),
            );
            let mut beam_indices = Tensor::zeros(
                [batch_size * gen_opt.num_beams],
                (Kind::Int64, self.get_var_store().device()),
            );
            let mut saved_beam_scores: Option<Vec<Tensor>> =
                if output_scores { Some(vec![]) } else { None };
            let mut current_tokens = Tensor::new();

            let mut past: Cache = Cache::None;
            let mut done = vec![false; batch_size as usize];

            let mut outputs: Tensor;
            let mut encoder_outputs = encoder_outputs;
            let mut current_length = cur_len;

            loop {
                if num_beam_groups > 1 {
                    current_tokens = Tensor::zeros(
                        [batch_size * gen_opt.num_beams],
                        (input_ids.kind(), input_ids.device()),
                    );
                }
                let prepared_input = self.prepare_inputs_for_generation(
                    input_ids.copy(),
                    encoder_outputs.as_ref(),
                    past,
                    attention_mask.copy(),
                );
                let temp = self
                    .forward_t(
                        prepared_input.prepared_input.as_ref(),
                        prepared_input.prepared_past,
                        prepared_input.prepared_attention_mask.as_ref(),
                        None,
                        prepared_input.prepared_position_ids.as_ref(),
                        None,
                        prepared_input.prepared_encoder_output,
                        prepared_input.prepared_decoder_input.as_ref(),
                        false,
                    )
                    .unwrap();
                outputs = temp.lm_logits;
                past = temp.cache;

                for beam_group_index in 0..num_beam_groups {
                    let group_start_index = beam_group_index * num_sub_beams;
                    let group_end_index = min(group_start_index + num_sub_beams, gen_opt.num_beams);
                    let group_size = group_end_index - group_start_index;

                    let (group_input_ids, batch_group_indices) = if num_beam_groups > 1 {
                        let mut batch_group_indices: Vec<i64> =
                            Vec::with_capacity((batch_size * group_size) as usize);
                        for batch_index in 0..batch_size {
                            batch_group_indices.extend(
                                (group_start_index..group_end_index)
                                    .map(|value| value + batch_index * gen_opt.num_beams),
                            )
                        }
                        let batch_group_indices =
                            Tensor::from_slice(batch_group_indices.as_slice())
                                .to(input_ids.device());
                        (
                            Some(input_ids.index_select(0, &batch_group_indices)),
                            Some(batch_group_indices),
                        )
                    } else {
                        (None, None)
                    };

                    let mut next_token_logits = if num_beam_groups <= 1 {
                        outputs.select(1, -1)
                    } else {
                        outputs
                            .select(1, -1)
                            .index_select(0, batch_group_indices.as_ref().unwrap())
                    };
                    // Reduce probability for repeated inputs
                    if gen_opt.repetition_penalty > 1f64 {
                        self.enforce_repetition_penalty(
                            &mut next_token_logits,
                            batch_size,
                            1,
                            group_input_ids.as_ref().unwrap_or(&input_ids),
                            gen_opt.repetition_penalty,
                        )
                    }

                    if gen_opt.temperature > 1f64 {
                        next_token_logits /= gen_opt.temperature;
                    }
                    self.prepare_scores_for_generation(
                        &mut next_token_logits,
                        current_length,
                        gen_opt.max_length,
                        gen_opt.forced_bos_token_id,
                    );

                    let mut scores = next_token_logits.log_softmax(-1, next_token_logits.kind());

                    // Do not allow eos token if min length is not reached
                    if (gen_opt.eos_token_ids.is_some()) & (current_length < gen_opt.min_length) {
                        let _ = scores.index_fill_(
                            1,
                            &Tensor::from_slice(gen_opt.eos_token_ids.as_ref().unwrap())
                                .to(scores.device()),
                            f64::NEG_INFINITY,
                        );
                    }

                    // Get bad word_ids and set their probability to 0
                    if gen_opt.bad_word_ids.is_some() {
                        // Calculate static bad words masks if not set yet
                        if let Some(bad_word_ids_length_1) = &bad_word_ids_length_1 {
                            if static_bad_words_mask.is_none() {
                                static_bad_words_mask = Some(
                                    self.calc_static_bad_word_mask(&scores, bad_word_ids_length_1),
                                );
                            }
                        }
                        self.ban_bad_words(
                            bad_word_ids_length_greater_than_1.as_ref(),
                            static_bad_words_mask.as_ref(),
                            group_input_ids.as_ref().unwrap_or(&input_ids),
                            &mut scores,
                        );
                    }

                    // Get repeated tokens and set their probability to 0
                    if gen_opt.no_repeat_ngram_size > 0 {
                        let banned_tokens = self.get_banned_tokens(
                            group_input_ids.as_ref().unwrap_or(&input_ids),
                            gen_opt.no_repeat_ngram_size,
                            current_length,
                        );
                        for (batch_index, index_banned_token) in
                            (0..banned_tokens.len() as i64).zip(banned_tokens)
                        {
                            let _ = scores.get(batch_index).index_fill_(
                                0,
                                &Tensor::from_slice(&index_banned_token)
                                    .to_device(next_token_logits.device()),
                                f64::NEG_INFINITY,
                            );
                        }
                    }

                    // Update scores with diversity penalty
                    if num_beam_groups > 1 {
                        self.run_hamming_diversity_penalty(
                            &mut scores,
                            &current_tokens,
                            diversity_penalty,
                            gen_opt.num_beams,
                            batch_size,
                            group_size,
                            group_start_index,
                        );
                    }

                    // Apply custom prefix constraint function
                    if let Some(prefix_allowed_tokens_function) = prefix_allowed_tokens_fn {
                        self.apply_prefix_allowed_tokens_function(
                            prefix_allowed_tokens_function,
                            num_sub_beams,
                            &input_ids,
                            &mut scores,
                        )
                    }

                    let mut next_scores: Tensor = &scores
                        + (if num_beam_groups > 1 {
                            beam_scores
                                .index_select(0, batch_group_indices.as_ref().unwrap())
                                .unsqueeze(-1)
                                .expand_as(&scores)
                        } else {
                            beam_scores.unsqueeze(-1).expand_as(&scores)
                        });

                    let (next_scores, next_tokens) = if gen_opt.do_sample {
                        self.top_k_top_p_filtering(
                            &mut next_scores,
                            gen_opt.top_k,
                            gen_opt.top_p,
                            2,
                        );
                        let _scores = next_scores
                            .contiguous()
                            .view((batch_size, group_size * vocab_size));

                        let probabilities = _scores.softmax(-1, _scores.kind());
                        let next_tokens = probabilities.multinomial(2 * group_size, false);
                        let _scores = _scores.gather(-1, &next_tokens, false);
                        let (_scores, next_scores_indices) = _scores.sort(1, true);
                        let next_tokens = next_tokens.gather(-1, &next_scores_indices, false);
                        (_scores, next_tokens)
                    } else {
                        let _scores = next_scores
                            .contiguous()
                            .view((batch_size, group_size * vocab_size));
                        _scores.topk(2 * group_size, 1, true, true)
                    };

                    let eos_token_ids = gen_opt.eos_token_ids.as_ref();
                    let beam_ids_tensor = &next_tokens.divide_scalar_mode(vocab_size, "floor");
                    let effective_beam_ids_tensor =
                        (&next_tokens.ones_like().cumsum(0, Kind::Int64) - 1) * group_size
                            + beam_ids_tensor;
                    let token_id_tensor = &next_tokens - beam_ids_tensor * vocab_size;
                    let (max_scores, _) = next_scores.max_dim(1, false);
                    let mut eos_mask = token_id_tensor.ones_like();
                    if let Some(eos_token_id) = eos_token_ids {
                        eos_mask -= token_id_tensor.eq(eos_token_id[0]).to_kind(Kind::Int64);
                    }
                    let eos_mask2 = eos_mask
                        .cumsum(1, Kind::Int64)
                        .le(group_size)
                        .to_kind(Kind::Bool)
                        .logical_and(&eos_mask);

                    let group_beam_scores = next_scores.masked_select(&eos_mask2);
                    let group_beam_tokens = token_id_tensor.masked_select(&eos_mask2);
                    let group_beam_indices = effective_beam_ids_tensor.masked_select(&eos_mask2);
                    let eos_pos = (eos_mask.ones_like() - eos_mask).nonzero();

                    for eos_idx in 0..eos_pos.size()[0] {
                        let eos_data = eos_pos.get(eos_idx);
                        let batch_index = eos_data.int64_value(&[0]);
                        if !done[batch_index as usize] {
                            let beam_index_pos = eos_data.int64_value(&[1]);
                            let is_beam_token_worse_than_top_num_beams =
                                beam_index_pos >= gen_opt.num_beams;
                            if is_beam_token_worse_than_top_num_beams {
                                continue;
                            }
                            let effective_beam_id = effective_beam_ids_tensor
                                .int64_value(&[batch_index, beam_index_pos]);
                            let beam_token_score =
                                next_scores.double_value(&[batch_index, beam_index_pos]);
                            let saved_beam_scores =
                                saved_beam_scores.as_ref().map(|step_wise_scores| {
                                    Tensor::stack(step_wise_scores, 1)
                                        .get(effective_beam_id)
                                        .copy()
                                });
                            hypotheses[batch_index as usize].add(
                                input_ids.get(effective_beam_id).copy(),
                                beam_token_score,
                                saved_beam_scores,
                            );
                        }
                    }

                    for batch_index in 0..batch_size {
                        if done[batch_index as usize] {
                            let _ = group_beam_scores
                                .narrow(0, batch_index * gen_opt.num_beams, gen_opt.num_beams)
                                .fill_(0f64);
                            let _ = group_beam_tokens
                                .narrow(0, batch_index * gen_opt.num_beams, gen_opt.num_beams)
                                .fill_(gen_opt.pad_token_id.unwrap());
                            let _ = group_beam_indices
                                .narrow(0, batch_index * gen_opt.num_beams, gen_opt.num_beams)
                                .fill_(0);
                            continue;
                        } else {
                            done[batch_index as usize] |= hypotheses[batch_index as usize]
                                .is_done(max_scores.double_value(&[batch_index]), current_length);
                        }
                    }

                    if num_beam_groups <= 1 {
                        beam_scores = group_beam_scores.view(-1);
                        beam_tokens = group_beam_tokens.view(-1);
                        beam_indices = group_beam_indices.view(-1);
                    } else {
                        let _ = beam_scores.index_copy_(
                            0,
                            batch_group_indices.as_ref().unwrap(),
                            &group_beam_scores,
                        );
                        let _ = beam_tokens.index_copy_(
                            0,
                            batch_group_indices.as_ref().unwrap(),
                            &group_beam_tokens,
                        );
                        let new_indices = gen_opt.num_beams
                            * group_beam_indices.divide_scalar_mode(group_size, "floor")
                            + group_start_index
                            + group_beam_indices.remainder(group_size);
                        let _ = beam_indices.index_copy_(
                            0,
                            batch_group_indices.as_ref().unwrap(),
                            &new_indices,
                        );
                        let _ = current_tokens.index_copy_(
                            0,
                            batch_group_indices.as_ref().unwrap(),
                            &group_beam_tokens,
                        );
                    }
                }

                if let Some(scores_output) = saved_beam_scores.as_mut() {
                    scores_output.push(beam_scores.copy());
                }
                if done.iter().all(|&x| x) {
                    break;
                }

                input_ids = Tensor::cat(
                    &[
                        input_ids.index_select(0, &beam_indices),
                        beam_tokens.unsqueeze(1),
                    ],
                    -1,
                );

                current_length += 1;
                if let Some(max_length) = gen_opt.max_length {
                    if current_length >= max_length {
                        break;
                    }
                }
                encoder_outputs = self.reorder_cache(&mut past, encoder_outputs, &beam_indices);

                if !self.is_encoder_decoder() {
                    attention_mask = Tensor::cat(
                        &[
                            attention_mask.as_ref(),
                            Tensor::ones(
                                [*attention_mask.size().first().unwrap(), 1],
                                (Kind::Int64, attention_mask.device()),
                            )
                            .as_ref(),
                        ],
                        -1,
                    );
                }
            }

            let mut batch_index = 0i64;

            let mut saved_beam_scores = saved_beam_scores
                .map(|step_wise_scores| Tensor::stack(&step_wise_scores, 1).split(1, 0));
            loop {
                if batch_index == batch_size {
                    break;
                }
                if done[batch_index as usize] {
                    batch_index += 1;
                    continue;
                }
                for beam_index in 0..gen_opt.num_beams {
                    let effective_beam_id = batch_index * gen_opt.num_beams + beam_index;
                    let beam_saved_token_scores = saved_beam_scores.as_mut().map(|saved_tokens| {
                        mem::replace(&mut saved_tokens[effective_beam_id as usize], Tensor::new())
                    });
                    let final_score = f64::try_from(beam_scores.get(effective_beam_id)).unwrap();
                    let final_tokens = input_ids.get(effective_beam_id);
                    hypotheses[batch_index as usize].add(
                        final_tokens,
                        final_score,
                        beam_saved_token_scores,
                    );
                }
                batch_index += 1;
            }
            let (output_batch_size, output_num_return_sequences_per_batch) = if gen_opt.do_sample {
                (batch_size, 1)
            } else {
                (
                    batch_size * gen_opt.num_return_sequences,
                    gen_opt.num_return_sequences,
                )
            };

            let mut sentence_lengths =
                Tensor::zeros([output_batch_size], (Kind::Int64, input_ids.device()));
            let mut best_ids = vec![];

            let mut scores_output = if output_scores {
                Some(Vec::with_capacity(best_ids.len()))
            } else {
                None
            };
            let mut token_scores_output = if output_scores {
                Some(Vec::with_capacity(best_ids.len()))
            } else {
                None
            };
            for (hypothesis_index, hypothesis) in hypotheses.iter().enumerate() {
                let mut sorted_hypotheses = hypothesis.clone();
                sorted_hypotheses
                    .beams
                    .sort_by_key(|(score, _, _)| OrderedFloat(*score));
                for j in 0..output_num_return_sequences_per_batch {
                    let effective_batch_index =
                        output_num_return_sequences_per_batch * hypothesis_index as i64 + j;

                    let (best_score, best_hyp, best_token_scores) =
                        sorted_hypotheses.beams.pop().unwrap();
                    let _ = sentence_lengths.index_fill_(
                        0,
                        &Tensor::from_slice(&[effective_batch_index]).to(sentence_lengths.device()),
                        *best_hyp.size().first().unwrap(),
                    );
                    best_ids.push(best_hyp);
                    if let Some(current_best_scores) = &mut scores_output {
                        current_best_scores.push(best_score);
                    }
                    if let Some(current_best_token_scores) = &mut token_scores_output {
                        current_best_token_scores.push(
                            best_token_scores
                                .unwrap()
                                .iter::<f64>()
                                .unwrap()
                                .collect::<Vec<f64>>(),
                        );
                    }
                }
            }
            let sentence_max_length = gen_opt
                .max_length
                .map(|max_length| {
                    min(
                        i64::try_from(sentence_lengths.max()).unwrap() + 1,
                        max_length,
                    )
                })
                .unwrap_or(i64::try_from(sentence_lengths.max()).unwrap() + 1);

            let mut decoded = input_ids.new_empty(
                [output_batch_size, sentence_max_length],
                (Kind::Int64, input_ids.device()),
            );
            if i64::try_from(sentence_lengths.max()).unwrap()
                != i64::try_from(sentence_lengths.min()).unwrap()
            {
                let _ = decoded.fill_(
                    gen_opt
                        .pad_token_id
                        .unwrap_or_else(|| gen_opt.eos_token_ids.as_ref().unwrap()[0]),
                );
            }
            for (hypothesis_index, best_id) in best_ids.iter().enumerate() {
                let _ = decoded.get(hypothesis_index as i64).index_copy_(
                    0,
                    &Tensor::arange_start(
                        0,
                        i64::try_from(sentence_lengths.get(hypothesis_index as i64)).unwrap(),
                        (Kind::Int64, input_ids.device()),
                    ),
                    best_id,
                );
                let sentence_length =
                    i64::try_from(sentence_lengths.get(hypothesis_index as i64)).unwrap();
                let sentence_length_max = gen_opt
                    .max_length
                    .unwrap_or_else(|| i64::try_from(sentence_lengths.max()).unwrap());
                if sentence_length < sentence_length_max {
                    let _ = decoded.get(hypothesis_index as i64).index_fill_(
                        0,
                        &Tensor::from_slice(&[sentence_length]).to_device(input_ids.device()),
                        gen_opt.eos_token_ids.as_ref().unwrap()[0],
                    );
                }
            }
            GeneratedOutputWithScores {
                indices: decoded,
                scores: scores_output,
                token_scores: token_scores_output,
            }
        }

        fn reorder_cache(
            &self,
            past: &mut Cache,
            _encoder_outputs: Option<Tensor>,
            _beam_indices: &Tensor,
        ) -> Option<Tensor> {
            match past {
                Cache::None => None,
                _ => {
                    panic!("Not implemented");
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
/// # Generated text output
/// Contains generated text and an optional log-likelihood score for the generated sequence
pub struct GeneratedTextOutput {
    pub text: String,
    pub score: Option<f64>,
}

#[derive(Debug, Clone)]
/// # Generated indices output
/// Contains generated indices and an optional log-likelihood score for the generated sequence and individual tokens
pub struct GeneratedIndicesOutput {
    pub indices: Vec<i64>,
    pub score: Option<f64>,
    pub token_scores: Option<Vec<f64>>,
}

pub type PrefixAllowedFunction<'a> = &'a dyn Fn(i64, &Tensor) -> Vec<i64>;
/// Type alias for a function defining allowed tokens based on current tokens generated.
/// This function should take a `batch_id` and associated tensor of already generated tokens and
/// should return a vector of allowed tokens. This is useful for controlled generation, i.e.
/// deterministic generation of a token continuation if a sequence of token occurs.

#[derive(Clone, Copy, Default)]
/// # Generation options for text generation.
/// When provided to a `generate` method, these options will take priority over the `GenerateConfig` used to create the
/// `LanguageGenerator`. Some of these options may be left as `None`, options without a value will individually default
/// to the `GenerateConfig`.
pub struct GenerateOptions<'a> {
    /// Minimum sequence length
    pub min_length: Option<i64>,
    /// Maximum sequence length
    pub max_length: Option<i64>,
    /// Maximum number of new tokens to generate (useful for causal generation models).
    /// Only one of `max_length` and `max_new_tokens` should be provided.
    /// When both are given, `max_new_tokens` is ignored and the `max_length` setting is used.
    pub max_new_tokens: Option<i64>,
    /// Early stopping flag indicating if the beam search should stop as soon as `num_beam` hypotheses have been generated
    pub early_stopping: Option<bool>,
    /// Number of sequences to return for each prompt text
    pub num_return_sequences: Option<i64>,
    /// Number of beams for beam search
    pub num_beams: Option<i64>,
    pub num_beam_groups: Option<i64>,
    /// Sampling flag. If true, will perform top-k and/or nucleus sampling on generated tokens, otherwise greedy (deterministic) decoding
    pub do_sample: Option<bool>,
    /// Temperature setting. Values higher than 1 will improve originality at the risk of reducing relevance
    pub temperature: Option<f64>,
    /// Top_k values for sampling tokens. Value higher than 0 will enable the feature
    pub top_k: Option<i64>,
    /// Top_p value for [Nucleus sampling, Holtzman et al.](http://arxiv.org/abs/1904.09751). Keep top tokens until cumulative probability reaches top_p
    pub top_p: Option<f64>,
    /// Repetition penalty (mostly useful for CTRL decoders). Values higher than 1 will penalize tokens that have been already generated.
    pub repetition_penalty: Option<f64>,
    /// Exponential penalty based on the length of the hypotheses generated
    pub length_penalty: Option<f64>,
    /// Number of allowed repetitions of n-grams. Values higher than 0 turn on this feature
    pub no_repeat_ngram_size: Option<i64>,
    /// Diversity penalty for diverse beam search. High values will enforce more difference between beam groups
    pub diversity_penalty: Option<f64>,
    /// Decoder start token id
    pub decoder_start_token_id: Option<i64>,
    /// Forced first token generated
    pub forced_bos_token_id: Option<i64>,
    /// Function to control the generation process. The function should take a `batch_id` (i64) and a tensor of token_ids already generated and returns a `Vec<i64>` of allowed tokens.
    pub prefix_allowed_tokens_fn: Option<PrefixAllowedFunction<'a>>,
    /// List of bad word ids (may be a sequence of word ids) that will be banned during the generation
    pub bad_word_ids: Option<&'a Vec<Vec<i64>>>,
    /// Flag indicating if text generation scores should be returned
    pub output_scores: bool,
}

macro_rules! unpack_config {
    ($field_name:ident, $generate_options: ident, $generate_config: ident) => {
        $generate_options.map_or($generate_config.$field_name, |opts| {
            opts.$field_name.unwrap_or($generate_config.$field_name)
        })
    };
}

/// # Common trait for text generation models.
/// Main API for text generation
pub trait LanguageGenerator: PrivateLanguageGenerator {
    /// Generate text based on a vector of promp texts.
    ///
    /// # Arguments
    ///
    /// * `prompt_texts` - `Option<Vec<&str>>` Optional vector of text prompts. An empty prompt to the model may be passed if the model implement a `bos_id`.
    /// * `generate_options` - `Option<GenerateOptions>` Optional set of generate options. If not (or partially) provided, will use the settings provided when creating the generator
    ///
    /// # Returns
    /// * `Vec<TextOutput>` Vector of length *number_of_prompts* x *num_return_sequences* containing TextOutput with the generated texts and the generation score if `output_scores` is true.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use std::path::PathBuf;
    /// # use tch::Device;
    /// # fn main() -> anyhow::Result<()> {
    /// use rust_bert::gpt2::GPT2Generator;
    /// use rust_bert::pipelines::generation_utils::{
    ///     GenerateConfig, GenerateOptions, LanguageGenerator,
    /// };
    /// use tch::Tensor;
    /// # let mut home: PathBuf = dirs::home_dir().unwrap();
    /// # home.push("rustbert");
    /// # home.push("gpt2");
    /// # let config_path = &home.as_path().join("config.json");
    /// # let vocab_path = &home.as_path().join("vocab.txt");
    /// # let merges_path = &home.as_path().join("merges.txt");
    /// # let weights_path = &home.as_path().join("model.ot");
    /// let device = Device::cuda_if_available();
    /// let generate_config = GenerateConfig {
    ///     max_length: Some(30),
    ///     do_sample: true,
    ///     num_beams: 5,
    ///     temperature: 1.1,
    ///     num_return_sequences: 3,
    ///     ..Default::default()
    /// };
    /// let gpt2_generator = GPT2Generator::new(generate_config)?;
    /// let input_context = "The dog";
    /// let second_input_context = "The cat was";
    ///
    /// //Example custom function for fine-grained generation control
    /// fn force_one_paragraph(_batch_id: i64, previous_token_ids: &Tensor) -> Vec<i64> {
    ///     let paragraph_tokens = [198, 628];
    ///
    ///     for paragraph_token in paragraph_tokens.iter() {
    ///         if previous_token_ids
    ///             .iter::<i64>()
    ///             .unwrap()
    ///             .collect::<Vec<i64>>()
    ///             .contains(paragraph_token)
    ///         {
    ///             return vec![50256];
    ///         }
    ///     }
    ///     (0..50255).collect()
    /// }
    ///
    /// let generate_options = GenerateOptions {
    ///     min_length: Some(32),
    ///     max_length: Some(128),
    ///     output_scores: true,
    ///     prefix_allowed_tokens_fn: Some(&force_one_paragraph),
    ///     ..Default::default()
    /// };
    ///
    /// let output = gpt2_generator.generate(
    ///     Some(&[input_context, second_input_context]),
    ///     Some(generate_options),
    /// );
    /// # Ok(())
    /// # }
    /// ```
    /// Example output: \
    /// ```no_run
    /// # let output =
    /// [
    ///     "The dog's owners, however, did not want to be named. According to the lawsuit, the animal's owner, a 29-year",
    ///     "The dog has always been part of the family. \"He was always going to be my dog and he was always looking out for me",
    ///     "The dog has been able to stay in the home for more than three months now. \"It's a very good dog. She's",
    ///     "The cat was discovered earlier this month in the home of a relative of the deceased. The cat\'s owner, who wished to remain anonymous,",
    ///     "The cat was pulled from the street by two-year-old Jazmine.\"I didn't know what to do,\" she said",
    ///     "The cat was attacked by two stray dogs and was taken to a hospital. Two other cats were also injured in the attack and are being treated."
    /// ]
    /// # ;
    /// ```
    fn generate<S>(
        &self,
        prompt_texts: Option<&[S]>,
        generate_options: Option<GenerateOptions>,
    ) -> Vec<GeneratedTextOutput>
    where
        S: AsRef<str> + Sync,
    {
        let indices_outputs = self.generate_indices(prompt_texts, generate_options);
        let mut output = Vec::with_capacity(indices_outputs.len());
        for generated_sequence in indices_outputs {
            output.push(GeneratedTextOutput {
                text: self
                    ._get_tokenizer()
                    .decode(&generated_sequence.indices, true, true),
                score: generated_sequence.score,
            });
        }
        output
    }

    /// Generate token indices without decoding (useful for token-level operations before returning final text or as validation step during training).
    ///
    /// # Arguments
    ///
    /// * `prompt_texts` - `Option<Vec<&str>>` Optional vector of text prompts. An empty prompt to the model may be passed if the model implement a `bos_id`.
    /// * `generate_options` - `Option<GenerateOptions>` Optional set of generate options. If not (or partially) provided, will use the settings provided when creating the generator
    ///
    /// # Returns
    /// * `Vec<IndicesOutput>` Vector of length *number_of_prompts* x *num_return_sequences* containing IndicesOutput with the generated indices and the generation score if `output_scores` is true.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use std::path::PathBuf;
    /// # use tch::Device;
    /// # fn main() -> anyhow::Result<()> {
    /// use rust_bert::gpt2::GPT2Generator;
    /// use rust_bert::pipelines::generation_utils::{
    ///     GenerateConfig, GenerateOptions, LanguageGenerator,
    /// };
    /// use tch::Tensor;
    /// # let mut home: PathBuf = dirs::home_dir().unwrap();
    /// # home.push("rustbert");
    /// # home.push("gpt2");
    /// # let config_path = &home.as_path().join("config.json");
    /// # let vocab_path = &home.as_path().join("vocab.txt");
    /// # let merges_path = &home.as_path().join("merges.txt");
    /// # let weights_path = &home.as_path().join("model.ot");
    /// let device = Device::cuda_if_available();
    /// let generate_config = GenerateConfig {
    ///     max_length: Some(30),
    ///     do_sample: true,
    ///     num_beams: 5,
    ///     temperature: 1.1,
    ///     num_return_sequences: 3,
    ///     ..Default::default()
    /// };
    /// let gpt2_generator = GPT2Generator::new(generate_config)?;
    /// let input_context = "The dog";
    /// let second_input_context = "The cat was";
    ///
    /// //Example custom function for fine-grained generation control
    /// fn force_one_paragraph(_batch_id: i64, previous_token_ids: &Tensor) -> Vec<i64> {
    ///     let paragraph_tokens = [198, 628];
    ///
    ///     for paragraph_token in paragraph_tokens.iter() {
    ///         if previous_token_ids
    ///             .iter::<i64>()
    ///             .unwrap()
    ///             .collect::<Vec<i64>>()
    ///             .contains(paragraph_token)
    ///         {
    ///             return vec![50256];
    ///         }
    ///     }
    ///     (0..50255).collect()
    /// }
    ///
    /// let generate_options = GenerateOptions {
    ///     min_length: Some(32),
    ///     max_length: Some(128),
    ///     output_scores: true,
    ///     prefix_allowed_tokens_fn: Some(&force_one_paragraph),
    ///     ..Default::default()
    /// };
    ///
    /// let output = gpt2_generator.generate_indices(
    ///     Some(&[input_context, second_input_context]),
    ///     Some(generate_options),
    /// );
    /// # Ok(())
    /// # }
    /// ```
    fn generate_indices<S>(
        &self,
        prompt_texts: Option<&[S]>,
        generate_options: Option<GenerateOptions>,
    ) -> Vec<GeneratedIndicesOutput>
    where
        S: AsRef<str> + Sync,
    {
        let eos_token_ids = self.get_eos_ids();

        let config = self.get_config();

        let max_length = generate_options.map_or(config.max_length, |generate_options| {
            generate_options.max_length
        });
        let encoding_max_len = if self.is_encoder_decoder() {
            Some(self.get_max_positions_embeddings())
        } else {
            max_length
        };
        let pad_token_id = match self.get_pad_id() {
            Some(value) => Some(value),
            None => eos_token_ids.as_ref().map(|eos_ids| eos_ids[0]),
        };

        let input_ids = match prompt_texts {
            Some(prompts) if !prompts.is_empty() => {
                self.encode_prompt_text(prompts, encoding_max_len, pad_token_id)
            }
            None => match self.get_bos_id() {
                Some(bos_id) => {
                    Tensor::ones([1, 1], (Int64, self.get_var_store().device())) * bos_id
                }
                None => panic!(
                    "A model with a BOS token must be used to start generation with an empty input"
                ),
            },
            _ => return Vec::new(),
        };
        self.generate_from_ids_and_past(input_ids, None, generate_options)
    }

    /// Generate token indices given a list of indices (useful when the input has been pre-tokenized).
    /// Returns a list of output tokens that need to be decoded using a tokenizer.
    ///
    /// # Arguments
    ///
    /// * `input_ids` - `Tensor` pre-tokenized and encoded input for generation.
    /// * `generate_options` - `Option<GenerateOptions>` Optional set of generate options. If not (or partially) provided, will use the settings provided when creating the generator
    ///
    /// # Returns
    /// * `Vec<IndicesOutput>` Vector of length *number_of_prompts* x *num_return_sequences* containing IndicesOutput with the generated indices and the generation score if `output_scores` is true.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use std::path::PathBuf;
    /// # use tch::Device;
    /// # fn main() -> anyhow::Result<()> {
    /// use rust_bert::gpt2::GPT2Generator;
    /// use rust_bert::pipelines::generation_utils::{
    ///     GenerateConfig, GenerateOptions, LanguageGenerator,
    /// };
    /// use tch::{Kind, Tensor};
    /// # let mut home: PathBuf = dirs::home_dir().unwrap();
    /// # home.push("rustbert");
    /// # home.push("gpt2");
    /// # let config_path = &home.as_path().join("config.json");
    /// # let vocab_path = &home.as_path().join("vocab.txt");
    /// # let merges_path = &home.as_path().join("merges.txt");
    /// # let weights_path = &home.as_path().join("model.ot");
    /// let device = Device::cuda_if_available();
    ///
    /// let gpt2_generator = GPT2Generator::new(Default::default())?;
    /// let input_tensor = Tensor::randn(&[32, 128], (Kind::Int64, Device::Cpu));
    /// let input_mask = Tensor::ones(&[32, 128], (Kind::Int64, Device::Cpu));
    ///
    /// let generate_options = GenerateOptions {
    ///     min_length: Some(32),
    ///     max_length: Some(128),
    ///     output_scores: true,
    ///     ..Default::default()
    /// };
    ///
    /// let output = gpt2_generator.generate_from_ids_and_past(
    ///     input_tensor,
    ///     Some(input_mask),
    ///     Some(generate_options),
    /// );
    /// # Ok(())
    /// # }
    /// ```
    fn generate_from_ids_and_past(
        &self,
        mut input_ids: Tensor,
        mut attention_mask: Option<Tensor>,
        generate_options: Option<GenerateOptions>,
    ) -> Vec<GeneratedIndicesOutput> {
        let eos_token_ids = PrivateLanguageGenerator::get_eos_ids(self).cloned();

        let config = PrivateLanguageGenerator::get_config(self);

        // Set generation options. Priority goes to options provided to the `generate` method, then
        // model configuration, then default values.
        let do_sample = unpack_config!(do_sample, generate_options, config);
        let num_return_sequences = unpack_config!(num_return_sequences, generate_options, config);
        let num_beams = unpack_config!(num_beams, generate_options, config);
        let min_length = unpack_config!(min_length, generate_options, config);
        let early_stopping = unpack_config!(early_stopping, generate_options, config);
        let temperature = unpack_config!(temperature, generate_options, config);
        let top_k = unpack_config!(top_k, generate_options, config);
        let top_p = unpack_config!(top_p, generate_options, config);
        let repetition_penalty = unpack_config!(repetition_penalty, generate_options, config);
        let length_penalty = unpack_config!(length_penalty, generate_options, config);
        let no_repeat_ngram_size = unpack_config!(no_repeat_ngram_size, generate_options, config);
        let num_beam_groups = generate_options.map_or(config.num_beam_groups, |opts| {
            opts.num_beam_groups.or(config.num_beam_groups)
        });
        let diversity_penalty = generate_options.map_or(config.diversity_penalty, |opts| {
            opts.diversity_penalty.or(config.diversity_penalty)
        });
        let decoder_start_token_id = generate_options.and_then(|opts| opts.decoder_start_token_id);
        let forced_bos_token_id = generate_options.and_then(|opts| opts.forced_bos_token_id);
        let bad_word_ids = generate_options.and_then(|opts| opts.bad_word_ids);
        let prefix_allowed_tokens_fn =
            generate_options.and_then(|opts| opts.prefix_allowed_tokens_fn);
        let output_scores = generate_options.map_or(false, |opts| opts.output_scores);

        let pad_token_id = match self.get_pad_id() {
            Some(value) => Some(value),
            None => eos_token_ids.as_ref().map(|eos_ids| eos_ids[0]),
        };

        let input_id_size = input_ids.size();
        let mut input_ids_len = *input_id_size.last().unwrap();
        if input_ids_len == 0 {
            input_ids = Tensor::ones(
                [*input_id_size.first().unwrap(), 1],
                (Int64, input_ids.device()),
            ) * self
                .get_bos_id()
                .expect("`bos_token_id` has to be defined when no `input_ids` are provided.");
            attention_mask = Some(Tensor::ones(
                [*input_id_size.first().unwrap(), 1],
                (Int64, input_ids.device()),
            ));
            input_ids_len += 1;
        }

        let cur_len = if !self.is_encoder_decoder() {
            *input_ids.size().last().unwrap()
        } else {
            1
        };
        let batch_size = *input_ids.size().first().unwrap();

        let (effective_batch_size, effective_batch_mult) = match do_sample {
            true => (batch_size * num_return_sequences, num_return_sequences),
            false => (batch_size, 1),
        };

        let attention_mask = match attention_mask {
            Some(value) => value,
            None => match pad_token_id {
                Some(pad_id) => input_ids.ne(pad_id).to_kind(Int64),
                None => input_ids.ones_like().to_kind(Int64),
            },
        };

        let encoder_outputs = if self.is_encoder_decoder() {
            let encoder_outputs = self.encode(&input_ids, Some(&attention_mask)).unwrap();
            let expanded_batch_indices = Tensor::arange(batch_size, (Int64, input_ids.device()))
                .view((-1, 1))
                .repeat([1, num_beams * effective_batch_mult])
                .view(-1);
            Some(encoder_outputs.index_select(0, &expanded_batch_indices))
        } else {
            None
        };

        let (input_ids, attention_mask) = if !self.is_encoder_decoder() {
            if (num_return_sequences > 1) | (num_beams > 1) {
                (
                    input_ids
                        .unsqueeze(1)
                        .expand(
                            [batch_size, effective_batch_mult * num_beams, cur_len],
                            true,
                        )
                        .contiguous()
                        .view((effective_batch_size * num_beams, cur_len)),
                    attention_mask
                        .unsqueeze(1)
                        .expand(
                            [batch_size, effective_batch_mult * num_beams, cur_len],
                            true,
                        )
                        .contiguous()
                        .view((effective_batch_size * num_beams, cur_len)),
                )
            } else {
                (input_ids, attention_mask)
            }
        } else {
            let decoder_start_token_id = decoder_start_token_id.unwrap_or_else(|| {
                self.get_decoder_start_id()
                    .expect("decoder start id must be specified for encoder decoders")
            });
            let input_ids = Tensor::full(
                [effective_batch_size * num_beams, 1],
                decoder_start_token_id,
                (Int64, input_ids.device()),
            );
            let attention_mask = if (num_return_sequences > 1) | (num_beams > 1) {
                attention_mask
                    .unsqueeze(1)
                    .expand(
                        [batch_size, effective_batch_mult * num_beams, input_ids_len],
                        true,
                    )
                    .contiguous()
                    .view((effective_batch_size * num_beams, input_ids_len))
            } else {
                attention_mask
            };
            (input_ids, attention_mask)
        };

        let max_length = if let Some(generate_options) = generate_options {
            match (generate_options.max_length, generate_options.max_new_tokens) {
                (Some(max_length), _) => Some(max_length),
                (None, Some(max_new_tokens)) => {
                    Some(max_new_tokens + input_ids.size().last().unwrap())
                }
                (None, None) => config.max_length,
            }
        } else {
            config.max_length
        };

        if max_length.is_none() & eos_token_ids.is_none() {
            panic!("No maximum length given for a model without an EOS token. \
            This would lead to an infinite generation loop. Please provide a `max_length` or `max_new_tokens`")
        }

        let gen_opt = InternalGenerateOptions {
            min_length,
            max_length,
            do_sample,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            no_repeat_ngram_size,
            pad_token_id,
            eos_token_ids,
            num_return_sequences,
            early_stopping,
            num_beams,
            length_penalty,
            num_beam_groups,
            diversity_penalty,
            forced_bos_token_id,
            bad_word_ids,
        };

        let generated_output_with_scores = no_grad(|| {
            if num_beams > 1 {
                self.generate_beam_search(
                    input_ids,
                    encoder_outputs,
                    cur_len,
                    effective_batch_size,
                    attention_mask,
                    gen_opt,
                    prefix_allowed_tokens_fn,
                    output_scores,
                )
            } else {
                self.generate_no_beam_search(
                    input_ids,
                    encoder_outputs,
                    cur_len,
                    effective_batch_size,
                    attention_mask,
                    gen_opt,
                    prefix_allowed_tokens_fn,
                    output_scores,
                )
            }
        });
        let (decoded, scores, mut token_scores) = (
            generated_output_with_scores.indices,
            generated_output_with_scores.scores,
            generated_output_with_scores.token_scores,
        );
        let num_sequences = *decoded.size().first().unwrap();
        let mut output = Vec::with_capacity(num_sequences as usize);
        for sequence_index in 0..num_sequences {
            let indices = decoded
                .as_ref()
                .get(sequence_index)
                .iter::<i64>()
                .unwrap()
                .collect::<Vec<i64>>();
            let score = scores
                .as_ref()
                .map(|scores_value| scores_value[sequence_index as usize]);

            let token_scores = token_scores
                .as_mut()
                .map(|token_scores| std::mem::take(&mut token_scores[sequence_index as usize]));

            output.push(GeneratedIndicesOutput {
                indices,
                score,
                token_scores,
            });
        }
        output
    }

    /// Returns a reference to the text generator's tokenizer
    ///
    /// # Returns
    /// * `&TokenizerOption` Reference to the generator's tokenizer.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use std::path::PathBuf;
    /// # use tch::Device;
    /// # fn main() -> anyhow::Result<()> {
    /// use rust_bert::gpt2::GPT2Generator;
    /// use rust_bert::pipelines::generation_utils::{GenerateConfig, LanguageGenerator};
    /// use tch::Tensor;
    /// # let mut home: PathBuf = dirs::home_dir().unwrap();
    /// # home.push("rustbert");
    /// # home.push("gpt2");
    /// # let config_path = &home.as_path().join("config.json");
    /// # let vocab_path = &home.as_path().join("vocab.txt");
    /// # let merges_path = &home.as_path().join("merges.txt");
    /// # let weights_path = &home.as_path().join("model.ot");
    /// let device = Device::cuda_if_available();
    /// let generate_config = GenerateConfig {
    ///     max_length: Some(30),
    ///     do_sample: true,
    ///     num_beams: 5,
    ///     temperature: 1.1,
    ///     num_return_sequences: 3,
    ///     ..Default::default()
    /// };
    /// let gpt2_generator = GPT2Generator::new(generate_config)?;
    /// let tokenizer = gpt2_generator.get_tokenizer();
    /// tokenizer.tokenize("Hello, world!");
    /// # Ok(())
    /// # }
    /// ```
    fn get_tokenizer(&self) -> &TokenizerOption {
        self._get_tokenizer()
    }

    fn get_tokenizer_mut(&mut self) -> &mut TokenizerOption {
        self._get_tokenizer_mut()
    }

    fn half(&mut self) {
        self.get_var_store_mut().half();
    }

    fn float(&mut self) {
        self.get_var_store_mut().float();
    }

    fn set_device(&mut self, device: Device) {
        self.get_var_store_mut().set_device(device);
    }
}

#[derive(Debug)]
struct BeamHypotheses {
    max_length: Option<i64>,
    length_penalty: f64,
    early_stopping: bool,
    num_beams: i64,
    beams: Vec<(f64, Tensor, Option<Tensor>)>,
    worst_score: f64,
}

impl Clone for BeamHypotheses {
    fn clone(&self) -> Self {
        BeamHypotheses {
            max_length: self.max_length,
            length_penalty: self.length_penalty,
            early_stopping: self.early_stopping,
            num_beams: self.num_beams,
            beams: self
                .beams
                .iter()
                .map(|(score, tensor, scores_tensor)| {
                    (
                        *score,
                        tensor.copy(),
                        scores_tensor
                            .as_ref()
                            .map(|scores_tensor| scores_tensor.copy()),
                    )
                })
                .collect::<Vec<(f64, Tensor, Option<Tensor>)>>(),
            worst_score: self.worst_score,
        }
    }
}

impl BeamHypotheses {
    fn new(
        num_beams: i64,
        max_length: Option<i64>,
        length_penalty: f64,
        early_stopping: bool,
    ) -> BeamHypotheses {
        BeamHypotheses {
            max_length: max_length.map(|max_length| max_length - 1),
            length_penalty,
            early_stopping,
            num_beams,
            beams: Vec::with_capacity(num_beams as usize + 1),
            worst_score: 1e9f64,
        }
    }

    fn len(&self) -> i64 {
        self.beams.len() as i64
    }

    fn add(
        &mut self,
        hypothesis: Tensor,
        sum_log_probabilities: f64,
        token_scores: Option<Tensor>,
    ) {
        let score =
            sum_log_probabilities / ((hypothesis.size()[0] as f64).powf(self.length_penalty));
        if (self.len() < self.num_beams) | (score > self.worst_score) {
            let token_scores = token_scores.map(|scores_tensor| {
                scores_tensor.squeeze_dim(0).diff::<Tensor>(
                    1,
                    0,
                    Some(Tensor::zeros(
                        [1],
                        (scores_tensor.kind(), scores_tensor.device()),
                    )),
                    None,
                )
            });
            self.beams.push((score, hypothesis, token_scores));
            if self.len() > self.num_beams {
                let (worst_score_position, _) = self
                    .beams
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, (score, _, _))| OrderedFloat(*score))
                    .unwrap();
                let _ = self.beams.remove(worst_score_position);
            }
            self.worst_score = self
                .beams
                .iter()
                .min_by_key(|(score, _, _)| OrderedFloat(*score))
                .unwrap()
                .0;
        }
    }

    fn is_done(&self, best_sum_log_probabilities: f64, current_length: i64) -> bool {
        if self.len() < self.num_beams {
            false
        } else if self.early_stopping {
            true
        } else {
            self.worst_score
                >= best_sum_log_probabilities / (current_length as f64).powf(self.length_penalty)
        }
    }
}

/// Container holding a language model output for generation tasks
pub struct LMModelOutput {
    /// Logits for each vocab item and position
    pub lm_logits: Tensor,
    /// cached state for improved efficiency during decoding
    pub cache: Cache,
}
