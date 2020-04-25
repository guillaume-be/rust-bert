//! # DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter (Sanh et al.)
//!
//! Implementation of the DistilBERT language model ([https://arxiv.org/abs/1910.01108](https://arxiv.org/abs/1910.01108) Sanh, Debut, Chaumond, Wolf, 2019).
//! The base model is implemented in the `distilbert::DistilBertModel` struct. Several language model heads have also been implemented, including:
//! - Masked language model: `distilbert::DistilBertForMaskedLM`
//! - Question answering: `distilbert::DistilBertForQuestionAnswering`
//! - Sequence classification: `distilbert::DistilBertForSequenceClassification`
//! - Token classification (e.g. NER, POS tagging): `distilbert::DistilBertForTokenClassification`
//!
//! # Model set-up and pre-trained weights loading
//!
//! A full working example is provided in `examples/distilbert_masked_lm.rs`, run with `cargo run --example distilbert_masked_lm`.
//! The example below illustrate a DistilBERT Masked language model example, the structure is similar for other models.
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
//!# home.push("distilbert");
//!# let config_path = &home.as_path().join("config.json");
//!# let vocab_path = &home.as_path().join("vocab.txt");
//!# let weights_path = &home.as_path().join("model.ot");
//! use rust_tokenizers::BertTokenizer;
//! use tch::{nn, Device};
//!# use std::path::PathBuf;
//! use rust_bert::Config;
//! use rust_bert::distilbert::{DistilBertModelMaskedLM, DistilBertConfig};
//!
//! let device = Device::cuda_if_available();
//! let mut vs = nn::VarStore::new(device);
//! let tokenizer: BertTokenizer = BertTokenizer::from_file(vocab_path.to_str().unwrap(), true);
//! let config = DistilBertConfig::from_file(config_path);
//! let bert_model = DistilBertModelMaskedLM::new(&vs.root(), &config);
//! vs.load(weights_path)?;
//!
//!# Ok(())
//!# }
//! ```



mod distilbert;
mod embeddings;
mod attention;
mod transformer;

pub use distilbert::{DistilBertModelResources, DistilBertConfigResources, DistilBertVocabResources,
                     DistilBertConfig, Activation, DistilBertModel, DistilBertForQuestionAnswering, DistilBertForTokenClassification,
                     DistilBertModelMaskedLM, DistilBertModelClassifier};
