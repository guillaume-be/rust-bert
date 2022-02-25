//! # MBart (Liu et al.)
//!
//! Implementation of the MBart language model ([Multilingual Denoising Pre-training for Neural Machine Translation](https://arxiv.org/abs/2001.08210) Liu, Gu, Goyal, Li, Edunov, Ghazvininejad, Lewis, Zettlemoyer, 2020).
//! The base model is implemented in the `mbart_model::MBartModel` struct. The model also includes a language model head: `mbart_model::MBartForConditionalGeneration`
//! implementing the common `generation_utils::LMHeadModel` trait shared between the models used for generation (see `pipelines` for more information).
//!
//! # Model set-up and pre-trained weights loading
//!
//! The translation capabilities are illustrated in `examples/translation_mbart`, run with `cargo run --example translation_mbart`.
//! All models expect the following resources:
//! - Configuration file expected to have a structure following the [Transformers library](https://github.com/huggingface/transformers)
//! - Model weights are expected to have a structure and parameter names following the [Transformers library](https://github.com/huggingface/transformers). A conversion using the Python utility scripts is required to convert the `.bin` weights to the `.ot` format.
//! - `MBart50Tokenizer` using a `spiece.model` SentencePiece model
//! Pretrained models are available and can be downloaded using RemoteResources.
//!
//! ```no_run
//! # fn main() -> anyhow::Result<()> {
//! #
//! use tch::{nn, Device};
//! # use std::path::PathBuf;
//! use rust_bert::mbart::{MBartConfig, MBartModel};
//! use rust_bert::resources::{LocalResource, ResourceProvider};
//! use rust_bert::Config;
//! use rust_tokenizers::tokenizer::MBart50Tokenizer;
//!
//! let config_resource = LocalResource {
//!     local_path: PathBuf::from("path/to/config.json"),
//! };
//! let vocab_resource = LocalResource {
//!     local_path: PathBuf::from("path/to/vocab.txt"),
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
//! let tokenizer: MBart50Tokenizer =
//!     MBart50Tokenizer::from_file(vocab_path.to_str().unwrap(), false)?;
//! let config = MBartConfig::from_file(config_path);
//! let mbart_model = MBartModel::new(&vs.root(), &config);
//! vs.load(weights_path)?;
//!
//! # Ok(())
//! # }
//! ```

mod attention;
mod decoder;
mod embeddings;
mod encoder;
mod mbart_model;

pub use mbart_model::{
    MBartConfig, MBartConfigResources, MBartForConditionalGeneration,
    MBartForSequenceClassification, MBartGenerator, MBartModel, MBartModelOutput,
    MBartModelResources, MBartSourceLanguages, MBartTargetLanguages, MBartVocabResources,
};

pub use attention::LayerState;
pub(crate) use decoder::MBartDecoderLayer;
pub(crate) use encoder::MBartEncoderLayer;
