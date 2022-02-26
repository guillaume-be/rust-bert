//! # M2M-100 (Fan et al.)
//!
//! Implementation of the M2M-100 language model ([Beyond English-Centric Multilingual Machine Translation](https://arxiv.org/abs/2010.11125) Fan, Bhosale, Schwenk, Ma, El-Kishky, Goyal, Baines, Celebi, Wenzel, Chaudhary, Goyal, Birch, Liptchinsky, Edunov, Grave, Auli, Joulin, 2020).
//! The base model is implemented in the `m2m_100::M2M100Model` struct. The model also includes a language model head: `m2m_100::M2M100ForConditionalGeneration`
//! implementing the common `generation_utils::LMHeadModel` trait shared between the models used for generation (see `pipelines` for more information).
//! This model allows for direct translation between 100 languages.
//! The translation capabilities are illustrated in `examples/translation_m2m100`, run with `cargo run --example translation_m2m100`.
//!
//! # Model set-up and pre-trained weights loading
//!
//! All models expect the following resources:
//! - Configuration file expected to have a structure following the [Transformers library](https://github.com/huggingface/transformers)
//! - Model weights are expected to have a structure and parameter names following the [Transformers library](https://github.com/huggingface/transformers). A conversion using the Python utility scripts is required to convert the `.bin` weights to the `.ot` format.
//! - `M2M100Tokenizer` using a `config.json` vocabulary and a `spiece.model` SentencePiece BPE model
//! Pretrained models are available and can be downloaded using RemoteResources.
//!
//! ```no_run
//! # fn main() -> anyhow::Result<()> {
//! #
//! use tch::{nn, Device};
//! # use std::path::PathBuf;
//! use rust_bert::m2m_100::{M2M100Config, M2M100Model};
//! use rust_bert::resources::{LocalResource, ResourceProvider};
//! use rust_bert::Config;
//! use rust_tokenizers::tokenizer::M2M100Tokenizer;
//!
//! let config_resource = LocalResource {
//!     local_path: PathBuf::from("path/to/config.json"),
//! };
//! let vocab_resource = LocalResource {
//!     local_path: PathBuf::from("path/to/vocab.txt"),
//! };
//! let merges_resource = LocalResource {
//!     local_path: PathBuf::from("path/to/spiece.model"),
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
//! let tokenizer: M2M100Tokenizer = M2M100Tokenizer::from_files(
//!     vocab_path.to_str().unwrap(),
//!     merges_path.to_str().unwrap(),
//!     false,
//! )?;
//! let config = M2M100Config::from_file(config_path);
//! let m2m100_model = M2M100Model::new(&vs.root(), &config);
//! vs.load(weights_path)?;
//!
//! # Ok(())
//! # }
//! ```

mod attention;
mod decoder;
mod embeddings;
mod encoder;
mod m2m_100_model;

pub use m2m_100_model::{
    M2M100Config, M2M100ConfigResources, M2M100ForConditionalGeneration, M2M100Generator,
    M2M100MergesResources, M2M100Model, M2M100ModelResources, M2M100SourceLanguages,
    M2M100TargetLanguages, M2M100VocabResources,
};

pub use attention::LayerState;
