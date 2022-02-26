//! # GPT (Radford et al.)
//!
//! Implementation of the GPT2 language model ([Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) Radford, Narasimhan, Salimans, Sutskever 2018).
//! The base model is implemented in the `openai_gpt_model::OpenAiGptModel` struct. The model also includes a language model head: `openai_gpt_model::OpenAIGPTLMHeadModel`
//! implementing the common `generation_utils::LMHeadModel` trait shared between the models used for generation (see `pipelines` for more information).
//!
//! # Model set-up and pre-trained weights loading
//!
//! All models expect the following resources:
//! - Configuration file expected to have a structure following the [Transformers library](https://github.com/huggingface/transformers)
//! - Model weights are expected to have a structure and parameter names following the [Transformers library](https://github.com/huggingface/transformers). A conversion using the Python utility scripts is required to convert the `.bin` weights to the `.ot` format.
//! - `GptTokenizer` using a `vocab.txt` vocabulary and `merges.txt` 2-gram merges
//! Pretrained models are available and can be downloaded using RemoteResources.
//!
//! ```no_run
//! # fn main() -> anyhow::Result<()> {
//! use tch::{nn, Device};
//! # use std::path::PathBuf;
//! use rust_bert::gpt2::Gpt2Config;
//! use rust_bert::openai_gpt::OpenAiGptModel;
//! use rust_bert::resources::{LocalResource, ResourceProvider};
//! use rust_bert::Config;
//! use rust_tokenizers::tokenizer::OpenAiGptTokenizer;
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

mod openai_gpt_model;
mod transformer;

pub use openai_gpt_model::{
    OpenAIGPTLMHeadModel, OpenAIGenerator, OpenAiGptConfigResources, OpenAiGptMergesResources,
    OpenAiGptModel, OpenAiGptModelOutput, OpenAiGptModelResources, OpenAiGptVocabResources,
};
