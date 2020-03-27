//! # RoBERTa: A Robustly Optimized BERT Pretraining Approach (Liu et al.)
//!
//! Implementation of the RoBERTa language model ([https://arxiv.org/abs/1907.11692](https://arxiv.org/abs/1907.11692) Liu, Ott, Goyal, Du, Joshi, Chen, Levy, Lewis, Zettlemoyer, Stoyanov, 2019).
//! The base model is implemented in the `bert::BertModel` struct. Several language model heads have also been implemented, including:
//! - Masked language model: `roberta::RobertaForMaskedLM`
//! - Multiple choices: `roberta:RobertaForMultipleChoice`
//! - Question answering: `roberta::RobertaForQuestionAnswering`
//! - Sequence classification: `roberta::RobertaForSequenceClassification`
//! - Token classification (e.g. NER, POS tagging): `roberta::RobertaForTokenClassification`
//!
//! # Model set-up and pre-trained weights loading
//!
//! A full working example is provided in `examples/robert.rs`, run with `cargo run --example roberta`.
//! The example below illustrate a Masked language model example, the structure is similar for other models.
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
//!# home.push("bert");
//!# let config_path = &home.as_path().join("config.json");
//!# let vocab_path = &home.as_path().join("vocab.txt");
//!# let merges_path = &home.as_path().join("merges.txt");
//!# let weights_path = &home.as_path().join("model.ot");
//! use rust_tokenizers::RobertaTokenizer;
//! use tch::{nn, Device};
//!# use std::path::PathBuf;
//! use rust_bert::bert::BertConfig;
//! use rust_bert::Config;
//! use rust_bert::roberta::RobertaForMaskedLM;
//!
//! let device = Device::cuda_if_available();
//! let mut vs = nn::VarStore::new(device);
//! let tokenizer: RobertaTokenizer = RobertaTokenizer::from_file(vocab_path.to_str().unwrap(), merges_path.to_str().unwrap(), true);
//! let config = BertConfig::from_file(config_path);
//! let bert_model = RobertaForMaskedLM::new(&vs.root(), &config);
//! vs.load(weights_path)?;
//!
//!# Ok(())
//!# }
//! ```


mod embeddings;
mod roberta;

pub use roberta::{RobertaForMaskedLM, RobertaForMultipleChoice, RobertaForTokenClassification, RobertaForQuestionAnswering, RobertaForSequenceClassification};
pub use embeddings::RobertaEmbeddings;