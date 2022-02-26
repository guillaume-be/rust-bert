//! # GPT2 (Radford et al.)
//!
//! Implementation of the GPT2 language model ([Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) Radford, Wu, Child, Luan, Amodei, Sutskever 2019).
//! The base model is implemented in the `gpt2_model::Gpt2Model` struct. The model also includes a language model head: `gpt2_model::GPT2LMHeadModel`
//! implementing the common `generation_utils::LMHeadModel` trait shared between the models used for generation (see `pipelines` for more information).
//!
//! # Model set-up and pre-trained weights loading
//!
//! A full working example is provided in `examples/generation_gpt2`, run with `cargo run --example generation_gpt2`.
//! All models expect the following resources:
//! - Configuration file expected to have a structure following the [Transformers library](https://github.com/huggingface/transformers)
//! - Model weights are expected to have a structure and parameter names following the [Transformers library](https://github.com/huggingface/transformers). A conversion using the Python utility scripts is required to convert the `.bin` weights to the `.ot` format.
//! - `Gpt2Tokenizer` using a `vocab.txt` vocabulary and `merges.txt` 2-gram merges
//! Pretrained models are available and can be downloaded using RemoteResources.
//!
//! ```no_run
//! # fn main() -> anyhow::Result<()> {
//! #
//! use tch::{nn, Device};
//! # use std::path::PathBuf;
//! use rust_bert::gpt2::{GPT2LMHeadModel, Gpt2Config};
//! use rust_bert::resources::{LocalResource, ResourceProvider};
//! use rust_bert::Config;
//! use rust_tokenizers::tokenizer::Gpt2Tokenizer;
//!
//! let config_resource = LocalResource {
//!     local_path: PathBuf::from("path/to/config.json"),
//! };
//! let vocab_resource = LocalResource {
//!     local_path: PathBuf::from("path/to/vocab.txt"),
//! };
//! let merges_resource = LocalResource {
//!     local_path: PathBuf::from("path/to/vocab.txt"),
//! };
//! let weights_resource = LocalResource {
//!     local_path: PathBuf::from("path/to/model.ot"),
//! };
//! let config_path = config_resource.get_local_path()?;
//! let vocab_path = vocab_resource.get_local_path()?;
//! let merges_path = merges_resource.get_local_path()?;
//! let weights_path = weights_resource.get_local_path()?;
//!
//! let device = Device::cuda_if_available();
//! let mut vs = nn::VarStore::new(device);
//! let tokenizer: Gpt2Tokenizer = Gpt2Tokenizer::from_file(
//!     vocab_path.to_str().unwrap(),
//!     merges_path.to_str().unwrap(),
//!     true,
//! )?;
//! let config = Gpt2Config::from_file(config_path);
//! let gpt2_model = GPT2LMHeadModel::new(&vs.root(), &config);
//! vs.load(weights_path)?;
//!
//! # Ok(())
//! # }
//! ```

pub(crate) mod attention;
mod gpt2_model;
pub(crate) mod transformer;

pub use gpt2_model::{
    GPT2Generator, GPT2LMHeadModel, Gpt2Config, Gpt2ConfigResources, Gpt2MergesResources,
    Gpt2Model, Gpt2ModelOutput, Gpt2ModelResources, Gpt2VocabResources,
};
