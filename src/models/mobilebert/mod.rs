//! # MobileBERT (A Compact Task-agnostic BERT for Resource-Limited Devices)
//!
//! Implementation of the MobileBERT language model ([MobileBERT: A Compact Task-agnostic BERT for Resource-Limited Devices](https://arxiv.org/abs/2004.02984) Sun, Yu, Song, Liu, Yang, Zhou, 2020).
//! The base model is implemented in the `mobilebert_model::MobileBertModel` struct. Several language model heads have also been implemented, including:
//! - Multiple choices: `mobilebert_model:MobileBertForMultipleChoice`
//! - Question answering: `mobilebert_model::MobileBertForQuestionAnswering`
//! - Sequence classification: `mobilebert_model::MobileBertForSequenceClassification`
//! - Token classification (e.g. NER, POS tagging): `mobilebert_model::MobileBertForTokenClassification`.
//!
//! # Model set-up and pre-trained weights loading
//!
//! All models expect the following resources:
//! - Configuration file expected to have a structure following the [Transformers library](https://github.com/huggingface/transformers)
//! - Model weights are expected to have a structure and parameter names following the [Transformers library](https://github.com/huggingface/transformers). A conversion using the Python utility scripts is required to convert the `.bin` weights to the `.ot` format.
//! - `BertTokenizer` using a `vocab.txt` vocabulary
//! Pretrained models for a number of language pairs are available and can be downloaded using RemoteResources.
//!
//! ```no_run
//! # fn main() -> anyhow::Result<()> {
//! #
//! use tch::{nn, Device};
//! # use std::path::PathBuf;
//! use rust_bert::mobilebert::{
//!     MobileBertConfig, MobileBertConfigResources, MobileBertForMaskedLM,
//!     MobileBertModelResources, MobileBertVocabResources,
//! };
//! use rust_bert::resources::{RemoteResource, ResourceProvider};
//! use rust_bert::Config;
//! use rust_tokenizers::tokenizer::BertTokenizer;
//!
//! let config_resource =
//!     RemoteResource::from_pretrained(MobileBertConfigResources::MOBILEBERT_UNCASED);
//! let vocab_resource =
//!     RemoteResource::from_pretrained(MobileBertVocabResources::MOBILEBERT_UNCASED);
//! let weights_resource =
//!     RemoteResource::from_pretrained(MobileBertModelResources::MOBILEBERT_UNCASED);
//! let config_path = config_resource.get_local_path()?;
//! let vocab_path = vocab_resource.get_local_path()?;
//! let weights_path = weights_resource.get_local_path()?;
//! let device = Device::cuda_if_available();
//! let mut vs = nn::VarStore::new(device);
//! let tokenizer: BertTokenizer =
//!     BertTokenizer::from_file(vocab_path.to_str().unwrap(), true, true)?;
//! let config = MobileBertConfig::from_file(config_path);
//! let bert_model = MobileBertForMaskedLM::new(&vs.root(), &config);
//! vs.load(weights_path)?;
//!
//! # Ok(())
//! # }
//! ```

mod attention;
mod embeddings;
mod encoder;
mod mobilebert_model;

pub use mobilebert_model::{
    MobileBertConfig, MobileBertConfigResources, MobileBertForMaskedLM,
    MobileBertForMultipleChoice, MobileBertForQuestionAnswering,
    MobileBertForSequenceClassification, MobileBertForTokenClassification, MobileBertModel,
    MobileBertModelResources, MobileBertVocabResources, NoNorm, NormalizationType,
};
