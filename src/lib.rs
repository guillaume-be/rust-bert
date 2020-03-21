//! Ready-to-use NLP pipelines and Transformer-based models
//!
//! Rust native Transformer-based models implementation. Port of the [Transformers](https://github.com/huggingface/transformers) library, using the tch-rs crate and pre-processing from rust-tokenizers.
//! Supports multithreaded tokenization and GPU inference. This repository exposes the model base architecture, task-specific heads (see below) and ready-to-use pipelines.
//!
//! # Quick Start
//!
//! This crate can be used in two different ways:
//! - Ready-to-use NLP pipelines for Sentiment Analysis, Named Entity Recognition, Question-Answering or Language Generation. More information on these can be found in the `pipelines` module.
//! ```no_run
//! use tch::Device;
//! use rust_bert::pipelines::question_answering::{QuestionAnsweringModel, QaInput};
//!# use std::path::PathBuf;
//!
//!# fn main() -> failure::Fallible<()> {
//!# let mut home: PathBuf = dirs::home_dir().unwrap();
//!# home.push("rustbert");
//!# home.push("distilbert-qa");
//!# let config_path = &home.as_path().join("config.json");
//!# let vocab_path = &home.as_path().join("vocab.txt");
//!# let weights_path = &home.as_path().join("model.ot");
//!
//! let device = Device::cuda_if_available();
//! let qa_model = QuestionAnsweringModel::new(vocab_path,
//!                                            config_path,
//!                                            weights_path, device)?;
//!
//! let question = String::from("Where does Amy live ?");
//! let context = String::from("Amy lives in Amsterdam");
//! let answers = qa_model.predict(&vec!(QaInput { question, context }), 1, 32);
//! # Ok(())
//! # }
//! ```
//! - Transformer models base architectures with customized heads. These allow to load pre-trained models for customized inference in Rust
//!
//!  | |**DistilBERT**|**BERT**|**RoBERTa**|**GPT**|**GPT2**
//! :-----:|:-----:|:-----:|:-----:|:-----:|:-----:
//! Masked LM|✅ |✅ |✅ | | |
//! Sequence classification|✅ |✅ |✅| | |
//! Token classification|✅ |✅ | ✅| | |
//! Question answering|✅ |✅ |✅| | |
//! Multiple choices| |✅ |✅| | |
//! Next token prediction| | | |✅|✅|
//! Natural Language Generation| | | |✅|✅|
//!
//! # Loading pre-trained models
//!
//! The architectures defined in this crate are compatible with model trained in the [Transformers](https://github.com/huggingface/transformers) library.
//! The model configuration and vocabulary are downloaded directly from Huggingface's repository.
//! The model weights need to be converter to a binary format that can be read by Libtorch (the original .bin files are pickles and cannot be used directly).
//! A Python script for downloading the required files & running the necessary steps is provided for all models classes in this library.
//! Further models can be loaded by extending the python scripts to point to the desired model.
//!
//!
//! 1. Compile the package: cargo build --release
//! 2. Download the model files & perform necessary conversions
//!     - Set-up a virtual environment and install dependencies
//!     - run the conversion script python /utils/download-dependencies_{MODEL_TO_DOWNLOAD}.py. The dependencies will be downloaded to the user's home directory, under ~/rustbert/{}
//! 3. Run the example cargo run --release
//!


pub mod distilbert;
pub mod bert;
pub mod roberta;
pub mod openai_gpt;
pub mod gpt2;
mod common;
pub mod pipelines;

pub use common::config::Config;
