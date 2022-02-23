//! # ALBERT: A Lite BERT for Self-supervised Learning of Language Representations (Lan et al.)
//!
//! Implementation of the ALBERT language model ([https://arxiv.org/abs/1909.11942](https://arxiv.org/abs/1909.11942) Lan, Chen, Goodman, Gimpel, Sharma, Soricut, 2019).
//! This model offers a greatly reduced memory footprint for similar effective size (number and size of layers). The computational cost remains however similar to the original BERT model.
//! The base model is implemented in the `albert_model::AlbertModel` struct. Several language model heads have also been implemented, including:
//! - Masked language model: `albert_model::AlbertForMaskedLM`
//! - Multiple choices: `albert_model:AlbertForMultipleChoice`
//! - Question answering: `albert_model::AlbertForQuestionAnswering`
//! - Sequence classification: `albert_model::AlbertForSequenceClassification`
//! - Token classification (e.g. NER, POS tagging): `albert_model::AlbertForTokenClassification`
//!
//! # Model set-up and pre-trained weights loading
//!
//! The example below illustrate a Masked language model example, the structure is similar for other models.
//! All models expect the following resources:
//! - Configuration file expected to have a structure following the [Transformers library](https://github.com/huggingface/transformers)
//! - Model weights are expected to have a structure and parameter names following the [Transformers library](https://github.com/huggingface/transformers). A conversion using the Python utility scripts is required to convert the `.bin` weights to the `.ot` format.
//! - `BertTokenizer` using a `vocab.txt` vocabulary
//! Pretrained models are available and can be downloaded using RemoteResources.
//!
//! ```no_run
//! # fn main() -> anyhow::Result<()> {
//! #
//! use tch::{nn, Device};
//! # use std::path::PathBuf;
//! use rust_bert::albert::{AlbertConfig, AlbertForMaskedLM};
//! use rust_bert::resources::{LocalResource, ResourceProvider};
//! use rust_bert::Config;
//! use rust_tokenizers::tokenizer::AlbertTokenizer;
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
//! let device = Device::cuda_if_available();
//! let mut vs = nn::VarStore::new(device);
//! let tokenizer: AlbertTokenizer =
//!     AlbertTokenizer::from_file(vocab_path.to_str().unwrap(), true, true)?;
//! let config = AlbertConfig::from_file(config_path);
//! let bert_model = AlbertForMaskedLM::new(&vs.root(), &config);
//! vs.load(weights_path)?;
//!
//! # Ok(())
//! # }
//! ```

mod albert_model;
mod attention;
mod embeddings;
mod encoder;

pub use albert_model::{
    AlbertConfig, AlbertConfigResources, AlbertForMaskedLM, AlbertForMultipleChoice,
    AlbertForQuestionAnswering, AlbertForSequenceClassification, AlbertForTokenClassification,
    AlbertMaskedLMOutput, AlbertModel, AlbertModelResources, AlbertOutput,
    AlbertQuestionAnsweringOutput, AlbertSequenceClassificationOutput,
    AlbertTokenClassificationOutput, AlbertVocabResources,
};
