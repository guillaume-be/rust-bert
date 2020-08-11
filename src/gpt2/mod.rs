//! # GPT2 (Radford et al.)
//!
//! Implementation of the GPT2 language model ([Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) Radford, Wu, Child, Luan, Amodei, Sutskever 2019).
//! The base model is implemented in the `gpt2::Gpt2Model` struct. The model also includes a language model head: `gpt2::GPT2LMHeadModel`
//! implementing the common `generation::LMHeadModel` trait shared between the models used for generation (see `pipelines` for more information).
//!
//! # Model set-up and pre-trained weights loading
//!
//! A full working example is provided in `examples/summarization.rs`, run with `cargo run --example gpt2`.
//! All models expect the following resources:
//! - Configuration file expected to have a structure following the [Transformers library](https://github.com/huggingface/transformers)
//! - Model weights are expected to have a structure and parameter names following the [Transformers library](https://github.com/huggingface/transformers). A conversion using the Python utility scripts is required to convert the `.bin` weights to the `.ot` format.
//! - `Gpt2Tokenizer` using a `vocab.txt` vocabulary and `merges.txt` 2-gram merges
//! Pretrained models are available and can be downloaded using RemoteResources.
//!
//! ```no_run
//! # fn main() -> anyhow::Result<()> {
//! #
//! use rust_tokenizers::Gpt2Tokenizer;
//! use tch::{nn, Device};
//! # use std::path::PathBuf;
//! use rust_bert::gpt2::{GPT2LMHeadModel, Gpt2Config};
//! use rust_bert::resources::{download_resource, LocalResource, Resource};
//! use rust_bert::Config;
//!
//! let config_resource = Resource::Local(LocalResource {
//!     local_path: PathBuf::from("path/to/config.json"),
//! });
//! let vocab_resource = Resource::Local(LocalResource {
//!     local_path: PathBuf::from("path/to/vocab.txt"),
//! });
//! let merges_resource = Resource::Local(LocalResource {
//!     local_path: PathBuf::from("path/to/vocab.txt"),
//! });
//! let weights_resource = Resource::Local(LocalResource {
//!     local_path: PathBuf::from("path/to/model.ot"),
//! });
//! let config_path = download_resource(&config_resource)?;
//! let vocab_path = download_resource(&vocab_resource)?;
//! let merges_path = download_resource(&merges_resource)?;
//! let weights_path = download_resource(&weights_resource)?;
//!
//! let device = Device::cuda_if_available();
//! let mut vs = nn::VarStore::new(device);
//! let tokenizer: Gpt2Tokenizer = Gpt2Tokenizer::from_file(
//!     vocab_path.to_str().unwrap(),
//!     merges_path.to_str().unwrap(),
//!     true,
//! );
//! let config = Gpt2Config::from_file(config_path);
//! let gpt2_model = GPT2LMHeadModel::new(&vs.root(), &config);
//! vs.load(weights_path)?;
//!
//! # Ok(())
//! # }
//! ```

pub(crate) mod attention;
mod gpt2;
pub(crate) mod transformer;

pub use gpt2::{
    GPT2LMHeadModel, Gpt2Config, Gpt2ConfigResources, Gpt2MergesResources, Gpt2Model,
    Gpt2ModelResources, Gpt2VocabResources, GptActivation,
};
