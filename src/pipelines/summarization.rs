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


//! # Summarization pipeline
//! Abstractive summarization of texts based on the BART encoder-decoder architecture
//! Include techniques such as beam search, top-k and nucleus sampling, temperature setting and repetition penalty.
//! By default, the dependencies for this model will be downloaded for a BART model finetuned on CNN/DM.
//! Customized BART models can be loaded by overwriting the resources in the configuration.
//! The dependencies will be downloaded to the user's home directory, under ~/.cache/.rustbert/bart-cnn
//!
//!
//! ```no_run
//!# fn main() -> failure::Fallible<()> {
//!# use rust_bert::pipelines::generation::LanguageGenerator;
//! use rust_bert::pipelines::summarization::SummarizationModel;
//! let mut model = SummarizationModel::new(Default::default())?;
//!
//! let input = ["In findings published Tuesday in Cornell University's arXiv by a team of scientists
//!from the University of Montreal and a separate report published Wednesday in Nature Astronomy by a team
//!from University College London (UCL), the presence of water vapour was confirmed in the atmosphere of K2-18b,
//!a planet circling a star in the constellation Leo. This is the first such discovery in a planet in its star's
//!habitable zone — not too hot and not too cold for liquid water to exist. The Montreal team, led by Björn Benneke,
//!used data from the NASA's Hubble telescope to assess changes in the light coming from K2-18b's star as the planet
//!passed between it and Earth. They found that certain wavelengths of light, which are usually absorbed by water,
//!weakened when the planet was in the way, indicating not only does K2-18b have an atmosphere, but the atmosphere
//!contains water in vapour form. The team from UCL then analyzed the Montreal team's data using their own software
//!and confirmed their conclusion. This was not the first time scientists have found signs of water on an exoplanet,
//!but previous discoveries were made on planets with high temperatures or other pronounced differences from Earth.
//!\"This is the first potentially habitable planet where the temperature is right and where we now know there is water,\"
//!said UCL astronomer Angelos Tsiaras. \"It's the best candidate for habitability right now.\" \"It's a good sign\",
//!said Ryan Cloutier of the Harvard–Smithsonian Center for Astrophysics, who was not one of either study's authors.
//!\"Overall,\" he continued, \"the presence of water in its atmosphere certainly improves the prospect of K2-18b being
//!a potentially habitable planet, but further observations will be required to say for sure. \"
//!K2-18b was first identified in 2015 by the Kepler space telescope. It is about 110 light-years from Earth and larger
//!but less dense. Its star, a red dwarf, is cooler than the Sun, but the planet's orbit is much closer, such that a year
//!on K2-18b lasts 33 Earth days. According to The Guardian, astronomers were optimistic that NASA's James Webb space
//!telescope — scheduled for launch in 2021 — and the European Space Agency's 2028 ARIEL program, could reveal more
//!about exoplanets like K2-18b."];
//!
//! let output = model.summarize(&input);
//!# Ok(())
//!# }
//! ```
//! (New sample credits: [WikiNews](https://en.wikinews.org/wiki/Astronomers_find_water_vapour_in_atmosphere_of_exoplanet_K2-18b))
//!
//! Example output: \
//! ```no_run
//!# let output =
//! "Scientists have found water vapour on K2-18b, a planet 110 light-years from Earth.
//!  This is the first such discovery in a planet in its star's habitable zone.
//!  The planet is not too hot and not too cold for liquid water to exist."
//!# ;
//!```

use crate::pipelines::generation::{BartGenerator, GenerateConfig, LanguageGenerator};
use tch::Device;
use crate::common::resources::{Resource, RemoteResource};
use crate::bart::{BartModelResources, BartConfigResources, BartVocabResources, BartMergesResources};

/// # Configuration for text summarization
/// Contains information regarding the model to load, mirrors the GenerationConfig, with a
/// different set of default parameters and sets the device to place the model on.
pub struct SummarizationConfig {
    /// Model weights resource (default: pretrained BART model on CNN-DM)
    pub model_resource: Resource,
    /// Config resource (default: pretrained BART model on CNN-DM)
    pub config_resource: Resource,
    /// Vocab resource (default: pretrained BART model on CNN-DM)
    pub vocab_resource: Resource,
    /// Merges resource (default: pretrained BART model on CNN-DM)
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

impl Default for SummarizationConfig {
    fn default() -> SummarizationConfig {
        SummarizationConfig {
            model_resource: Resource::Remote(RemoteResource::from_pretrained(BartModelResources::BART_CNN)),
            config_resource: Resource::Remote(RemoteResource::from_pretrained(BartConfigResources::BART_CNN)),
            vocab_resource: Resource::Remote(RemoteResource::from_pretrained(BartVocabResources::BART_CNN)),
            merges_resource: Resource::Remote(RemoteResource::from_pretrained(BartMergesResources::BART_CNN)),
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
            device: Device::cuda_if_available(),
        }
    }
}

/// # SummarizationModel to perform summarization
pub struct SummarizationModel {
    model: BartGenerator
}

impl SummarizationModel {
    /// Build a new `SummarizationModel`
    ///
    /// # Arguments
    ///
    /// * `summarization_config` - `SummarizationConfig` object containing the resource references (model, vocabulary, configuration), summarization options and device placement (CPU/GPU)
    ///
    /// # Example
    ///
    /// ```no_run
    ///# fn main() -> failure::Fallible<()> {
    /// use rust_bert::pipelines::summarization::SummarizationModel;
    ///
    /// let mut summarization_model =  SummarizationModel::new(Default::default())?;
    ///# Ok(())
    ///# }
    /// ```
    ///
    pub fn new(summarization_config: SummarizationConfig)
               -> failure::Fallible<SummarizationModel> {
        let generate_config = GenerateConfig {
            model_resource: summarization_config.model_resource,
            config_resource: summarization_config.config_resource,
            merges_resource: summarization_config.merges_resource,
            vocab_resource: summarization_config.vocab_resource,
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
            device: summarization_config.device,
        };

        let model = BartGenerator::new(generate_config)?;

        Ok(SummarizationModel { model })
    }

    /// Summarize texts provided
    ///
    /// # Arguments
    ///
    /// * `input` - `&[&str]` Array of texts to summarize.
    ///
    /// # Returns
    /// * `Vec<String>` Summarized texts
    ///
    /// # Example
    ///
    /// ```no_run
    ///# fn main() -> failure::Fallible<()> {
    /// use rust_bert::pipelines::generation::LanguageGenerator;
    /// use rust_bert::pipelines::summarization::SummarizationModel;
    /// let mut model = SummarizationModel::new(Default::default())?;
    ///
    /// let input = ["In findings published Tuesday in Cornell University's arXiv by a team of scientists
    ///from the University of Montreal and a separate report published Wednesday in Nature Astronomy by a team
    ///from University College London (UCL), the presence of water vapour was confirmed in the atmosphere of K2-18b,
    ///a planet circling a star in the constellation Leo. This is the first such discovery in a planet in its star's
    ///habitable zone — not too hot and not too cold for liquid water to exist. The Montreal team, led by Björn Benneke,
    ///used data from the NASA's Hubble telescope to assess changes in the light coming from K2-18b's star as the planet
    ///passed between it and Earth. They found that certain wavelengths of light, which are usually absorbed by water,
    ///weakened when the planet was in the way, indicating not only does K2-18b have an atmosphere, but the atmosphere
    ///contains water in vapour form. The team from UCL then analyzed the Montreal team's data using their own software
    ///and confirmed their conclusion. This was not the first time scientists have found signs of water on an exoplanet,
    ///but previous discoveries were made on planets with high temperatures or other pronounced differences from Earth.
    ///\"This is the first potentially habitable planet where the temperature is right and where we now know there is water,\"
    ///said UCL astronomer Angelos Tsiaras. \"It's the best candidate for habitability right now.\" \"It's a good sign\",
    ///said Ryan Cloutier of the Harvard–Smithsonian Center for Astrophysics, who was not one of either study's authors.
    ///\"Overall,\" he continued, \"the presence of water in its atmosphere certainly improves the prospect of K2-18b being
    ///a potentially habitable planet, but further observations will be required to say for sure. \"
    ///K2-18b was first identified in 2015 by the Kepler space telescope. It is about 110 light-years from Earth and larger
    ///but less dense. Its star, a red dwarf, is cooler than the Sun, but the planet's orbit is much closer, such that a year
    ///on K2-18b lasts 33 Earth days. According to The Guardian, astronomers were optimistic that NASA's James Webb space
    ///telescope — scheduled for launch in 2021 — and the European Space Agency's 2028 ARIEL program, could reveal more
    ///about exoplanets like K2-18b."];
    ///
    /// let output = model.summarize(&input);
    ///# Ok(())
    ///# }
    /// ```
    /// (New sample credits: [WikiNews](https://en.wikinews.org/wiki/Astronomers_find_water_vapour_in_atmosphere_of_exoplanet_K2-18b))
    ///
    pub fn summarize(&mut self, texts: &[&str]) -> Vec<String> {
        self.model.generate(Some(texts.to_vec()), None)
    }
}