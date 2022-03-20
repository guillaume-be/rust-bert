//! # Pegasus (Zhang et al.)
//!
//! Implementation of the Pegasus language model ([PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization](https://arxiv.org/abs/1912.08777) Zhang, Zhao, Saleh, Liu, 2019).
//! The base model is implemented in the `pegasus_model::PegasusModel` struct and leverages an implementation that is broadly similar to BART. The model also includes a language model head: `pegasus_model::PegasusForConditionalGeneration`
//! implementing the common `generation_utils::LMHeadModel` trait shared between the models used for generation (see `pipelines` for more information).
//!
//! # Model set-up and pre-trained weights loading
//!
//! A full working example is provided in `examples/summarization_pegasus`, run with `cargo run --example summarization_pegasus`.
//! All models expect the following resources:
//! - Configuration file expected to have a structure following the [Transformers library](https://github.com/huggingface/transformers)
//! - Model weights are expected to have a structure and parameter names following the [Transformers library](https://github.com/huggingface/transformers). A conversion using the Python utility scripts is required to convert the `.bin` weights to the `.ot` format.
//! - `PegasusTokenizer` using a `spiece.model` vocabulary and unigram model.
//! Pretrained models are available and can be downloaded using RemoteResources.
//!
//! ```no_run
//! # fn main() -> anyhow::Result<()> {
//! #
//! use tch::{nn, Device};
//! # use std::path::PathBuf;
//! use rust_bert::pegasus::{PegasusConfig, PegasusModel};
//! use rust_bert::resources::{LocalResource, ResourceProvider};
//! use rust_bert::Config;
//! use rust_tokenizers::tokenizer::PegasusTokenizer;
//!
//! let config_resource = LocalResource {
//!     local_path: PathBuf::from("path/to/config.json"),
//! };
//! let vocab_resource = LocalResource {
//!     local_path: PathBuf::from("path/to/spiece.model"),
//! };
//! let weights_resource = LocalResource {
//!     local_path: PathBuf::from("path/to/model.ot"),
//! };
//! let config_path = config_resource.get_local_path()?;
//! let vocab_path = vocab_resource.get_local_path()?;
//! let weights_path = weights_resource.get_local_path()?;
//!
//! let device = Device::cuda_if_available();
//! let mut vs = nn::VarStore::new(device);
//! let tokenizer: PegasusTokenizer =
//!     PegasusTokenizer::from_file(vocab_path.to_str().unwrap(), false)?;
//! let config = PegasusConfig::from_file(config_path);
//! let pegasus_model = PegasusModel::new(&vs.root(), &config);
//! vs.load(weights_path)?;
//!
//! # Ok(())
//! # }
//! ```

mod attention;
mod decoder;
mod embeddings;
mod encoder;
mod pegasus_model;

pub use attention::LayerState;
pub use pegasus_model::{
    PegasusConditionalGenerator, PegasusConfig, PegasusConfigResources,
    PegasusForConditionalGeneration, PegasusModel, PegasusModelResources, PegasusVocabResources,
};
