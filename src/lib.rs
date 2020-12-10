//! # Ready-to-use NLP pipelines and Transformer-based models
//!
//! Rust native Transformer-based models implementation. Port of the [Transformers](https://github.com/huggingface/transformers) library, using the tch-rs crate and pre-processing from rust-tokenizers.
//! Supports multithreaded tokenization and GPU inference. This repository exposes the model base architecture, task-specific heads (see below) and ready-to-use pipelines.
//!
//! # Quick Start
//!
//! This crate can be used in two different ways:
//! - Ready-to-use NLP pipelines for:
//!     - Translation
//!     - Summarization
//!     - Multi-turn dialogue
//!     - Zero-shot classification
//!     - Sentiment Analysis
//!     - Named Entity Recognition
//!     - Question-Answering
//!     - Language Generation.
//!
//! More information on these can be found in the [`pipelines` module](./pipelines/index.html)
//! ```no_run
//! use rust_bert::pipelines::question_answering::{QaInput, QuestionAnsweringModel};
//!
//! # fn main() -> anyhow::Result<()> {
//! let qa_model = QuestionAnsweringModel::new(Default::default())?;
//!
//! let question = String::from("Where does Amy live ?");
//! let context = String::from("Amy lives in Amsterdam");
//! let answers = qa_model.predict(&vec![QaInput { question, context }], 1, 32);
//! # Ok(())
//! # }
//! ```
//! - Transformer models base architectures with customized heads. These allow to load pre-trained models for customized inference in Rust
//!
//! | |**Sequence classification**|**Token classification**|**Question answering**|**Text Generation**|**Summarization**|**Translation**|**Masked LM**|
//! :-----:|:----:|:----:|:-----:|:----:|:-----:|:----:|:----:
//! DistilBERT|✅|✅|✅| | | |✅|
//! BERT|✅|✅|✅| | | |✅|
//! RoBERTa|✅|✅|✅| | | |✅|
//! GPT| | | |✅ | | | |
//! GPT2| | | |✅ | | | |
//! BART|✅| | |✅ |✅| | |
//! Marian| | | |  | |✅| |
//! Electra | |✅| | | | |✅|
//! ALBERT |✅|✅|✅| | | |✅|
//! T5 | | | |✅ |✅|✅| |
//! XLNet|✅|✅|✅|✅ | | |✅|
//! Reformer|✅| |✅|✅ | | |✅|
//!
//! # Loading pre-trained models
//!
//! A number of pretrained model configuration, weights and vocabulary are downloaded directly from [Huggingface's model repository](https://huggingface.co/models).
//! The list of models available with Rust-compatible weights is available in the example ./examples/download_all_dependencies.rs. Additional models can be added if of interest, please raise an issue.
//!
//! In order to load custom weights to the library, these need to be converter to a binary format that can be read by Libtorch (the original `.bin` files are pickles and cannot be used directly).
//! Several Python scripts to load Pytorch weights and convert them to the appropriate format are provided and can be adapted based on the model needs.
//!
//! The procedure for building custom weights or re-building pretrained weights is as follows:
//! 1. Compile the package: cargo build --release
//! 2. Download the model files & perform necessary conversions
//!     - Set-up a virtual environment and install dependencies
//!     - run the conversion script python /utils/download-dependencies_{MODEL_TO_DOWNLOAD}.py. The dependencies will be downloaded to the user's home directory, under ~/rustbert/{}
//! 3. Run the example cargo run --release

pub mod albert;
pub mod bart;
pub mod bert;
mod common;
pub mod distilbert;
pub mod electra;
pub mod gpt2;
pub mod marian;
pub mod openai_gpt;
pub mod pipelines;
pub mod reformer;
pub mod roberta;
pub mod t5;
pub mod xlnet;

pub use common::error::RustBertError;
pub use common::resources;
pub use common::{Activation, Config};
