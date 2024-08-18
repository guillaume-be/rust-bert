//! # GPT-J
//!
//! Implementation of the GPT-J language model
//!
//! # Model set-up and pre-trained weights loading
//!
//! ```no_run
//! # fn main() -> anyhow::Result<()> {
//! #
//! use tch::{nn, Device};
//! # use std::path::PathBuf;
//! use rust_bert::gpt_j::{GptJConfig, GptJLMHeadModel};
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
//! let config = GptJConfig::from_file(config_path);
//! let gpt_j_model = GptJLMHeadModel::new(&vs.root(), &config);
//! vs.load(weights_path)?;
//!
//! # Ok(())
//! # }
//! ```

mod attention;
mod gpt_j_model;
mod transformer;

pub use gpt_j_model::{
    GptJConfig, GptJConfigResources, GptJGenerator, GptJLMHeadModel, GptJMergesResources,
    GptJModel, GptJModelOutput, GptJModelResources, GptJVocabResources,
};

pub use attention::LayerState;
