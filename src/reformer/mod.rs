//! # Reformer: The Efficient Transformer (Kitaev et al.)
//!
//! Implementation of the Reformer language model ([Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451) Kitaev, kaiser, Levskaya, 2020).
//! The base model is implemented in the `reformer_model::ReformerModel` struct. The model also includes a language model head: `reformer_model::ReformerModelWithLMHead`
//! implementing the common `generation_utils::LMHeadModel` trait shared between the models used for generation (see `pipelines` for more information).
//!
//! # Model set-up and pre-trained weights loading
//!
//! A full working example is provided in `examples/generation_reformer`, run with `cargo run --example generation_reformer`.
//! All models expect the following resources:
//! - Configuration file expected to have a structure following the [Transformers library](https://github.com/huggingface/transformers)
//! - Model weights are expected to have a structure and parameter names following the [Transformers library](https://github.com/huggingface/transformers). A conversion using the Python utility scripts is required to convert the `.bin` weights to the `.ot` format.
//! - `ReformerTokenizer` using a `spiece.model` BPE model
//! Pretrained models on "Crime and Punishment" (Dostoevsky) are available and can be downloaded using RemoteResources.
//!
//! ```no_run
//! # fn main() -> anyhow::Result<()> {
//! #
//! use tch::{nn, Device};
//! # use std::path::PathBuf;
//! use rust_bert::reformer::{ReformerConfig, ReformerModel};
//! use rust_bert::resources::{LocalResource, ResourceProvider};
//! use rust_bert::Config;
//! use rust_tokenizers::tokenizer::ReformerTokenizer;
//!
//! let config_resource = LocalResource {
//!     local_path: PathBuf::from("path/to/config.json"),
//! };
//! let weights_resource = LocalResource {
//!     local_path: PathBuf::from("path/to/weights.ot"),
//! };
//! let vocab_resource = LocalResource {
//!     local_path: PathBuf::from("path/to/spiece.model"),
//! };
//! let config_path = config_resource.get_local_path()?;
//! let weights_path = weights_resource.get_local_path()?;
//! let vocab_path = vocab_resource.get_local_path()?;
//!
//! let device = Device::cuda_if_available();
//! let mut vs = nn::VarStore::new(device);
//! let tokenizer: ReformerTokenizer =
//!     ReformerTokenizer::from_file(vocab_path.to_str().unwrap(), true)?;
//! let config = ReformerConfig::from_file(config_path);
//! let bart_model = ReformerModel::new(&vs.root(), &config);
//! vs.load(weights_path)?;
//!
//! # Ok(())
//! # }
//! ```

mod attention;
mod attention_utils;
mod embeddings;
mod encoder;
mod reformer_model;

pub use attention::LayerState;
pub use reformer_model::{
    ReformerClassificationOutput, ReformerConfig, ReformerConfigResources,
    ReformerForQuestionAnswering, ReformerForSequenceClassification, ReformerGenerator,
    ReformerModel, ReformerModelResources, ReformerModelWithLMHead,
    ReformerQuestionAnsweringModelOutput, ReformerVocabResources,
};
