//! # T5 (Text-To-Text Transfer Transformer)
//!
//! Implementation of the T5 language model ([Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) Raffel, Shazeer, Roberts, Lee, Narang, Matena, Zhou, Li, Liu, 2019).
//! The base model is implemented in the `t5_model::T5Model` struct. This model includes a language model head: `t5_model::T5ForConditionalGeneration`
//! implementing the common `generation_utils::LMHeadModel` trait shared between the models used for generation (see `pipelines` for more information).
//!
//! # Model set-up and pre-trained weights loading
//!
//! A full working example (summarization) is provided in `examples/summarization_t5`, run with `cargo run --example summarization_t5`.
//! All models expect the following resources:
//! - Configuration file expected to have a structure following the [Transformers library](https://github.com/huggingface/transformers)
//! - Model weights are expected to have a structure and parameter names following the [Transformers library](https://github.com/huggingface/transformers). A conversion using the Python utility scripts is required to convert the `.bin` weights to the `.ot` format.
//! - `T5Tokenizer` using a `spiece.model` sentence piece model
//!
//! Pretrained models for a number of language pairs are available and can be downloaded using RemoteResources.
//!
//! ```no_run
//! # fn main() -> anyhow::Result<()> {
//! #
//! use tch::{nn, Device};
//! # use std::path::PathBuf;
//! use rust_bert::resources::{LocalResource, ResourceProvider};
//! use rust_bert::t5::{T5Config, T5ForConditionalGeneration};
//! use rust_bert::Config;
//! use rust_tokenizers::tokenizer::T5Tokenizer;
//!
//! let config_resource = LocalResource {
//!     local_path: PathBuf::from("path/to/config.json"),
//! };
//! let sentence_piece_resource = LocalResource {
//!     local_path: PathBuf::from("path/to/spiece.model"),
//! };
//! let weights_resource = LocalResource {
//!     local_path: PathBuf::from("path/to/model.ot"),
//! };
//! let config_path = config_resource.get_local_path()?;
//! let spiece_path = sentence_piece_resource.get_local_path()?;
//! let weights_path = weights_resource.get_local_path()?;
//!
//! let device = Device::cuda_if_available();
//! let mut vs = nn::VarStore::new(device);
//! let tokenizer = T5Tokenizer::from_file(spiece_path.to_str().unwrap(), true);
//! let config = T5Config::from_file(config_path);
//! let t5_model = T5ForConditionalGeneration::new(&vs.root(), &config, false, false);
//! vs.load(weights_path)?;
//!
//! # Ok(())
//! # }
//! ```

mod attention;
mod encoder;
mod layer_norm;
mod t5_model;

pub use attention::LayerState;
pub use t5_model::{
    T5Config, T5ConfigResources, T5ForConditionalGeneration, T5Generator, T5Model, T5ModelOutput,
    T5ModelResources, T5Prefix, T5SourceLanguages, T5TargetLanguages, T5VocabResources,
};
