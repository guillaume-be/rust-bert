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

use crate::pipelines::generation::{BartGenerator, GenerateConfig, LanguageGenerator};
use std::path::Path;
use tch::Device;

/// # Configuration for text summarization
/// Mirrors the GenerationConfig, with a different set of default parameters
pub struct SummarizationConfig {
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
}

impl Default for SummarizationConfig {
    fn default() -> SummarizationConfig {
        SummarizationConfig {
            min_length: 56,
            max_length: 142,
            do_sample: false,
            early_stopping: false,
            num_beams: 3,
            temperature: 1.0,
            top_k: 50,
            top_p: 1.0,
            repetition_penalty: 1.0,
            length_penalty: 1.0,
            no_repeat_ngram_size: 3,
            num_return_sequences: 1,
        }
    }
}


pub struct SummarizationModel {
    model: BartGenerator
}

impl SummarizationModel {
    pub fn new(vocab_path: &Path, merges_path: &Path, config_path: &Path, weights_path: &Path,
               summarization_config: SummarizationConfig, device: Device)
               -> failure::Fallible<SummarizationModel> {
        let generate_config = GenerateConfig {
            min_length: summarization_config.min_length,
            max_length: summarization_config.max_length,
            do_sample: summarization_config.do_sample,
            early_stopping: summarization_config.early_stopping,
            num_beams: summarization_config.num_beams,
            temperature: summarization_config.temperature,
            top_k: summarization_config.top_k,
            top_p: summarization_config.top_p,
            repetition_penalty: summarization_config.repetition_penalty,
            length_penalty: summarization_config.length_penalty,
            no_repeat_ngram_size: summarization_config.no_repeat_ngram_size,
            num_return_sequences: summarization_config.num_return_sequences,

        };
        let model = BartGenerator::new(vocab_path, merges_path, config_path, weights_path,
                                           generate_config, device)?;

        Ok(SummarizationModel { model })
    }

    pub fn summarize(&mut self, texts: &[&str]) -> Vec<String> {
        self.model.generate(Some(texts.to_vec()), None)
    }
}