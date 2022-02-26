//! # Marian
//!
//! Implementation of the Marian language model ([Marian: Fast Neural Machine Translation in {C++}](http://www.aclweb.org/anthology/P18-4020) Junczys-Dowmunt, Grundkiewicz, Dwojak, Hoang, Heafield, Neckermann, Seide, Germann, Fikri Aji, Bogoychev, Martins, Birch, 2018).
//! The base model is implemented in the `bart_model::BartModel` struct. This model includes a language model head: `marian_model::MarianForConditionalGeneration`
//! implementing the common `generation_utils::LMHeadModel` trait shared between the models used for generation (see `pipelines` for more information).
//!
//! # Model set-up and pre-trained weights loading
//!
//! A full working example is provided in `examples/translation_marian`, run with `cargo run --example translation_marian`.
//! All models expect the following resources:
//! - Configuration file expected to have a structure following the [Transformers library](https://github.com/huggingface/transformers)
//! - Model weights are expected to have a structure and parameter names following the [Transformers library](https://github.com/huggingface/transformers). A conversion using the Python utility scripts is required to convert the `.bin` weights to the `.ot` format.
//! - `MarianTokenizer` using a `vocab.json` vocabulary and `spiece.model` sentence piece model
//!
//! Pretrained models for a number of language pairs are available and can be downloaded using RemoteResources. These are shared under Creative Commons Attribution 4.0 International License license by the Opus-MT team from Language Technology at the University of Helsinki at <https://github.com/Helsinki-NLP/Opus-MT>.
//!
//! ```no_run
//! # fn main() -> anyhow::Result<()> {
//! #
//! use tch::{nn, Device};
//! # use std::path::PathBuf;
//! use rust_bert::bart::{BartConfig, BartModel};
//! use rust_bert::marian::MarianForConditionalGeneration;
//! use rust_bert::resources::{LocalResource, ResourceProvider};
//! use rust_bert::Config;
//! use rust_tokenizers::tokenizer::MarianTokenizer;
//!
//! let config_resource = LocalResource {
//!     local_path: PathBuf::from("path/to/config.json"),
//! };
//! let vocab_resource = LocalResource {
//!     local_path: PathBuf::from("path/to/vocab.json"),
//! };
//! let sentence_piece_resource = LocalResource {
//!     local_path: PathBuf::from("path/to/spiece.model"),
//! };
//! let weights_resource = LocalResource {
//!     local_path: PathBuf::from("path/to/model.ot"),
//! };
//! let config_path = config_resource.get_local_path()?;
//! let vocab_path = vocab_resource.get_local_path()?;
//! let spiece_path = sentence_piece_resource.get_local_path()?;
//! let weights_path = weights_resource.get_local_path()?;
//!
//! let device = Device::cuda_if_available();
//! let mut vs = nn::VarStore::new(device);
//! let tokenizer = MarianTokenizer::from_files(
//!     vocab_path.to_str().unwrap(),
//!     spiece_path.to_str().unwrap(),
//!     true,
//! );
//! let config = BartConfig::from_file(config_path);
//! let marian_model = MarianForConditionalGeneration::new(&vs.root(), &config);
//! vs.load(weights_path)?;
//!
//! # Ok(())
//! # }
//! ```

mod marian_model;

pub use marian_model::{
    MarianConfigResources, MarianForConditionalGeneration, MarianGenerator, MarianModelPreset,
    MarianModelResources, MarianSourceLanguages, MarianSpmResources, MarianTargetLanguages,
    MarianVocabResources,
};
