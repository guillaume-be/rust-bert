//! # LongT5 (Efficient Text-To-Text Transformer for Long Sequences)
//!
//! Implementation of the LongT5 language model ([LongT5: Efficient Text-To-Text Transformer for Long Sequences](https://arxiv.org/abs/2112.07916) Guo, Ainslie, Uthus, Ontanon, Ni, Sung, Yang, 2021).
//! The base model is implemented in the `longt5_model::LongT5Model` struct. This model includes a language model head: `longt5_model::LongT5ForConditionalGeneration`
//! implementing the common `generation_utils::LanguageGenerator` trait shared between the models used for generation (see `pipelines` for more information).
//!
//! # Model set-up and pre-trained weights loading
//!
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
//! use rust_bert::longt5::{LongT5Config, LongT5ForConditionalGeneration};
//! use rust_bert::resources::{LocalResource, ResourceProvider};
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
//! let config = LongT5Config::from_file(config_path);
//! let longt5_model = LongT5ForConditionalGeneration::new(&vs.root(), &config);
//! vs.load(weights_path)?;
//!
//! # Ok(())
//! # }
//! ```

mod attention;
mod encoder;
mod layer_norm;
mod longt5_model;

pub use attention::LayerState;
pub use longt5_model::{
    LongT5Config, LongT5ConfigResources, LongT5ForConditionalGeneration, LongT5Generator,
    LongT5Model, LongT5ModelResources, LongT5VocabResources,
};
