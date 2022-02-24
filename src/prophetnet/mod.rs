//! # ProphetNet (ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training)
//!
//! Implementation of the ProphetNet language model ([ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training](https://arxiv.org/abs/2001.04063) Qi, Yan, Gong, Liu, Duan, Chen, Zhang, Zhou, 2020).
//! The base model is implemented in the `prophetnet_model::ProphetNetModel` struct. Two language model heads have also been implemented:
//! - Conditional language generation (encoder-decoder architecture): `prophetnet_model::ProphetNetForConditionalGeneration` implementing the common `generation_utils::LMHeadModel` trait shared between the models used for generation (see `pipelines` for more information)
//! - Causal language generation (decoder architecture): `prophetnet_model::ProphetNetForCausalGeneration`
//!
//! # Model set-up and pre-trained weights loading
//!
//! A full working example (summarization) is provided in `examples/summarization_prophetnet`, run with `cargo run --example summarization_prophetnet`.
//! All models expect the following resources:
//! - Configuration file expected to have a structure following the [Transformers library](https://github.com/huggingface/transformers)
//! - Model weights are expected to have a structure and parameter names following the [Transformers library](https://github.com/huggingface/transformers). A conversion using the Python utility scripts is required to convert the `.bin` weights to the `.ot` format.
//! - `ProphetNetTokenizer` using a `vocab.txt` vocabulary
//!
//!
//! ```no_run
//! use rust_bert::pipelines::common::ModelType;
//! use rust_bert::pipelines::summarization::{SummarizationConfig, SummarizationModel};
//! use rust_bert::prophetnet::{
//!     ProphetNetConfigResources, ProphetNetModelResources, ProphetNetVocabResources,
//! };
//! use rust_bert::resources::RemoteResource;
//! use tch::Device;
//!
//! fn main() -> anyhow::Result<()> {
//!     let config_resource = Box::new(RemoteResource::from_pretrained(
//!         ProphetNetConfigResources::PROPHETNET_LARGE_CNN_DM,
//!     ));
//!     let vocab_resource = Box::new(RemoteResource::from_pretrained(
//!         ProphetNetVocabResources::PROPHETNET_LARGE_CNN_DM,
//!     ));
//!     let weights_resource = Box::new(RemoteResource::from_pretrained(
//!         ProphetNetModelResources::PROPHETNET_LARGE_CNN_DM,
//!     ));
//!
//!     let summarization_config = SummarizationConfig {
//!         model_type: ModelType::ProphetNet,
//!         model_resource: weights_resource,
//!         config_resource,
//!         vocab_resource: vocab_resource.clone(),
//!         merges_resource: vocab_resource,
//!         length_penalty: 1.2,
//!         num_beams: 4,
//!         no_repeat_ngram_size: 3,
//!         device: Device::cuda_if_available(),
//!         ..Default::default()
//!     };
//!     let summarization_model = SummarizationModel::new(summarization_config)?;
//!
//!     let input = ["In findings published Tuesday in Cornell University's arXiv by a team of scientists \
//! from the University of Montreal and a separate report published Wednesday in Nature Astronomy by a team \
//! from University College London (UCL), the presence of water vapour was confirmed in the atmosphere of K2-18b, \
//! a planet circling a star in the constellation Leo. This is the first such discovery in a planet in its star's \
//! habitable zone — not too hot and not too cold for liquid water to exist. The Montreal team, led by Björn Benneke, \
//! used data from the NASA's Hubble telescope to assess changes in the light coming from K2-18b's star as the planet \
//! passed between it and Earth. They found that certain wavelengths of light, which are usually absorbed by water, \
//! weakened when the planet was in the way, indicating not only does K2-18b have an atmosphere, but the atmosphere \
//! contains water in vapour form. The team from UCL then analyzed the Montreal team's data using their own software \
//! and confirmed their conclusion. This was not the first time scientists have found signs of water on an exoplanet, \
//! but previous discoveries were made on planets with high temperatures or other pronounced differences from Earth. \
//! \"This is the first potentially habitable planet where the temperature is right and where we now know there is water,\" \
//! said UCL astronomer Angelos Tsiaras. \"It's the best candidate for habitability right now.\" \"It's a good sign\", \
//! said Ryan Cloutier of the Harvard–Smithsonian Center for Astrophysics, who was not one of either study's authors. \
//! \"Overall,\" he continued, \"the presence of water in its atmosphere certainly improves the prospect of K2-18b being \
//! a potentially habitable planet, but further observations will be required to say for sure. \" \
//! K2-18b was first identified in 2015 by the Kepler space telescope. It is about 110 light-years from Earth and larger \
//! but less dense. Its star, a red dwarf, is cooler than the Sun, but the planet's orbit is much closer, such that a year \
//! on K2-18b lasts 33 Earth days. According to The Guardian, astronomers were optimistic that NASA's James Webb space \
//! telescope — scheduled for launch in 2021 — and the European Space Agency's 2028 ARIEL program, could reveal more \
//! about exoplanets like K2-18b."];
//!
//!     //    Credits: WikiNews, CC BY 2.5 license (https://en.wikinews.org/wiki/Astronomers_find_water_vapour_in_atmosphere_of_exoplanet_K2-18b)
//!     let _output = summarization_model.summarize(&input);
//!     for sentence in _output {
//!         println!("{}", sentence);
//!     }
//!
//!     Ok(())
//! }
//! ```

mod attention;
mod decoder;
mod embeddings;
mod encoder;
mod prophetnet_model;

pub use attention::LayerState;
pub use prophetnet_model::{
    ProphetNetConditionalGenerator, ProphetNetConfig, ProphetNetConfigResources,
    ProphetNetForCausalGeneration, ProphetNetForConditionalGeneration, ProphetNetGenerationOutput,
    ProphetNetModel, ProphetNetModelResources, ProphetNetOutput, ProphetNetVocabResources,
};
