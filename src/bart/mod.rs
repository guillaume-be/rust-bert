//! # BART (Lewis et al.)
//!
//! Implementation of the BART language model ([BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461) Lewis, Liu, Goyal, Ghazvininejad, Mohamed, Levy, Stoyanov, Zettlemoyer, 2019).
//! The base model is implemented in the `bart::BartModel` struct. The model also includes a language model head: `bart::BartForConditionalGeneration`
//! implementing the common `generation::LMHeadModel` trait shared between the models used for generation (see `pipelines` for more information).
//!
//! # Model set-up and pre-trained weights loading
//!
//! A full working example is provided in `examples/bart.rs`, run with `cargo run --example bart`.
//! Alternatively, the summarization capabilities are illustrated in `examples/summarization.rs`, run with `cargo run --example summarization`.
//! All models expect the following resources:
//! - Configuration file expected to have a structure following the [Transformers library](https://github.com/huggingface/transformers)
//! - Model weights are expected to have a structure and parameter names following the [Transformers library](https://github.com/huggingface/transformers). A conversion using the Python utility scripts is required to convert the `.bin` weights to the `.ot` format.
//! - `RobertaTokenizer` using a `vocab.txt` vocabulary and `merges.txt` 2-gram merges
//!
//! ```no_run
//!# fn main() -> failure::Fallible<()> {
//!#
//!# let mut home: PathBuf = dirs::home_dir().unwrap();
//!# home.push("rustbert");
//!# home.push("bart-large-cnn");
//!# let config_path = &home.as_path().join("config.json");
//!# let vocab_path = &home.as_path().join("vocab.txt");
//!# let merges_path = &home.as_path().join("merges.txt");
//!# let weights_path = &home.as_path().join("model.ot");
//! use rust_tokenizers::RobertaTokenizer;
//! use tch::{nn, Device};
//!# use std::path::PathBuf;
//! use rust_bert::Config;
//! use rust_bert::bart::{BartConfig, BartModel};
//!
//! let device = Device::cuda_if_available();
//! let mut vs = nn::VarStore::new(device);
//! let tokenizer: RobertaTokenizer = RobertaTokenizer::from_file(vocab_path.to_str().unwrap(), merges_path.to_str().unwrap(), true);
//! let config = BartConfig::from_file(config_path);
//! let gpt2_model = BartModel::new(&vs.root(), &config, false);
//! vs.load(weights_path)?;
//!
//!# Ok(())
//!# }
//! ```

mod bart;
mod attention;
mod encoder;
mod decoder;
mod embeddings;

pub use bart::{BartModelResources, BartConfigResources, BartVocabResources, BartMergesResources,
               BartConfig, Activation, BartModel, BartForSequenceClassification, BartForConditionalGeneration};
pub use attention::LayerState;