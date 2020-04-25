//! # BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (Devlin et al.)
//!
//! Implementation of the BERT language model ([https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805) Devlin, Chang, Lee, Toutanova, 2018).
//! The base model is implemented in the `bert::BertModel` struct. Several language model heads have also been implemented, including:
//! - Masked language model: `bert::BertForMaskedLM`
//! - Multiple choices: `bert:BertForMultipleChoice`
//! - Question answering: `bert::BertForQuestionAnswering`
//! - Sequence classification: `bert::BertForSequenceClassification`
//! - Token classification (e.g. NER, POS tagging): `bert::BertForTokenClassification`
//!
//! # Model set-up and pre-trained weights loading
//!
//! A full working example is provided in `examples/bert.rs`, run with `cargo run --example bert`.
//! The example below illustrate a Masked language model example, the structure is similar for other models.
//! All models expect the following resources:
//! - Configuration file expected to have a structure following the [Transformers library](https://github.com/huggingface/transformers)
//! - Model weights are expected to have a structure and parameter names following the [Transformers library](https://github.com/huggingface/transformers). A conversion using the Python utility scripts is required to convert the `.bin` weights to the `.ot` format.
//! - `BertTokenizer` using a `vocab.txt` vocabulary
//!
//! ```no_run
//!# fn main() -> failure::Fallible<()> {
//!#
//!# let mut home: PathBuf = dirs::home_dir().unwrap();
//!# home.push("rustbert");
//!# home.push("bert");
//!# let config_path = &home.as_path().join("config.json");
//!# let vocab_path = &home.as_path().join("vocab.txt");
//!# let weights_path = &home.as_path().join("model.ot");
//! use rust_tokenizers::BertTokenizer;
//! use tch::{nn, Device};
//!# use std::path::PathBuf;
//! use rust_bert::bert::{BertForMaskedLM, BertConfig};
//! use rust_bert::Config;
//!
//! let device = Device::cuda_if_available();
//! let mut vs = nn::VarStore::new(device);
//! let tokenizer: BertTokenizer = BertTokenizer::from_file(vocab_path.to_str().unwrap(), true);
//! let config = BertConfig::from_file(config_path);
//! let bert_model = BertForMaskedLM::new(&vs.root(), &config);
//! vs.load(weights_path)?;
//!
//!# Ok(())
//!# }
//! ```


mod bert;
mod embeddings;
mod attention;
mod encoder;

pub use bert::{BertModelDependencies, BertConfigDependencies, BertTokenizerDependencies,
               BertConfig, Activation, BertModel, BertForTokenClassification, BertForMultipleChoice,
               BertForMaskedLM, BertForSequenceClassification, BertForQuestionAnswering};
pub use embeddings::{BertEmbedding, BertEmbeddings};