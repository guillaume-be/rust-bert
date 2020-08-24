//! # GPT (Radford et al.)
//!
//! Implementation of the GPT2 language model ([Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) Radford, Narasimhan, Salimans, Sutskever 2018).
//! The base model is implemented in the `openai_gpt::OpenAiGptModel` struct. The model also includes a language model head: `openai_gpt::OpenAIGPTLMHeadModel`
//! implementing the common `generation::LMHeadModel` trait shared between the models used for generation (see `pipelines` for more information).
//!
//! # Model set-up and pre-trained weights loading
//!
//! A full working example is provided in `examples/openai_gpt.rs`, run with `cargo run --example openai_gpt`.
//! All models expect the following resources:
//! - Configuration file expected to have a structure following the [Transformers library](https://github.com/huggingface/transformers)
//! - Model weights are expected to have a structure and parameter names following the [Transformers library](https://github.com/huggingface/transformers). A conversion using the Python utility scripts is required to convert the `.bin` weights to the `.ot` format.
//! - `GptTokenizer` using a `vocab.txt` vocabulary and `merges.txt` 2-gram merges
//! Pretrained models are available and can be downloaded using RemoteResources.
//!
//! ```no_run
//! # fn main() -> anyhow::Result<()> {
//! use rust_tokenizers::OpenAiGptTokenizer;
//! use tch::{nn, Device};
//! # use std::path::PathBuf;
//! use rust_bert::gpt2::Gpt2Config;
//! use rust_bert::openai_gpt::OpenAiGptModel;
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
//! let tokenizer: OpenAiGptTokenizer = OpenAiGptTokenizer::from_file(
//!     vocab_path.to_str().unwrap(),
//!     merges_path.to_str().unwrap(),
//!     true,
//! )?;
//! let config = Gpt2Config::from_file(config_path);
//! let gpt_model = OpenAiGptModel::new(&vs.root(), &config);
//! vs.load(weights_path)?;
//!
//! # Ok(())
//! # }
//! ```

mod openai_gpt;
mod transformer;

pub use openai_gpt::{
    OpenAIGPTLMHeadModel, OpenAiGptConfigResources, OpenAiGptMergesResources, OpenAiGptModel,
    OpenAiGptModelResources, OpenAiGptVocabResources,
};
