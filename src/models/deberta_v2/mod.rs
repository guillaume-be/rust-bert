//! # DeBERTa V2 (He et al.)
//!
//! Implementation of the DeBERTa V2/V3 language model ([DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing](https://arxiv.org/abs/2111.09543) He, Gao, Chen, 2021).
//! The base model is implemented in the `deberta_v2_model::DebertaV2Model` struct. Several language model heads have also been implemented, including:
//! - Question answering: `deberta_v2_model::DebertaV2ForQuestionAnswering`
//! - Sequence classification: `deberta_v2_model::DebertaV2ForSequenceClassification`
//! - Token classification (e.g. NER, POS tagging): `deberta_v2_model::DebertaV2ForTokenClassification`.
//!
//! # Model set-up and pre-trained weights loading
//!
//! All models expect the following resources:
//! - Configuration file expected to have a structure following the [Transformers library](https://github.com/huggingface/transformers)
//! - Model weights are expected to have a structure and parameter names following the [Transformers library](https://github.com/huggingface/transformers). A conversion using the Python utility scripts is required to convert the `.bin` weights to the `.ot` format.
//! - `DebertaV2Tokenizer` using a `spiece.model` SentencePiece model file
//! Pretrained models for a number of language pairs are available and can be downloaded using RemoteResources.
//!
//! ```no_run
//! # fn main() -> anyhow::Result<()> {
//! #
//! use tch::{nn, Device};
//! # use std::path::PathBuf;
//! use rust_bert::deberta_v2::{
//!     DebertaV2Config, DebertaV2ConfigResources, DebertaV2ForSequenceClassification,
//!     DebertaV2ModelResources, DebertaV2VocabResources,
//! };
//! use rust_bert::resources::{RemoteResource, ResourceProvider};
//! use rust_bert::Config;
//! use rust_tokenizers::tokenizer::DeBERTaV2Tokenizer;
//!
//! let config_resource =
//!     RemoteResource::from_pretrained(DebertaV2ConfigResources::DEBERTA_V3_BASE);
//! let vocab_resource = RemoteResource::from_pretrained(DebertaV2VocabResources::DEBERTA_V3_BASE);
//! let weights_resource =
//!     RemoteResource::from_pretrained(DebertaV2ModelResources::DEBERTA_V3_BASE);
//! let config_path = config_resource.get_local_path()?;
//! let vocab_path = vocab_resource.get_local_path()?;
//! let weights_path = weights_resource.get_local_path()?;
//! let device = Device::cuda_if_available();
//! let mut vs = nn::VarStore::new(device);
//! let tokenizer =
//!     DeBERTaV2Tokenizer::from_file(vocab_path.to_str().unwrap(), false, false, false)?;
//! let config = DebertaV2Config::from_file(config_path);
//! let deberta_model = DebertaV2ForSequenceClassification::new(&vs.root(), &config);
//! vs.load(weights_path)?;
//!
//! # Ok(())
//! # }
//! ```

mod attention;
mod deberta_v2_model;
mod embeddings;
mod encoder;

pub use deberta_v2_model::{
    DebertaV2Config, DebertaV2ConfigResources, DebertaV2ForMaskedLM, DebertaV2ForQuestionAnswering,
    DebertaV2ForSequenceClassification, DebertaV2ForTokenClassification, DebertaV2Model,
    DebertaV2ModelResources, DebertaV2QuestionAnsweringOutput,
    DebertaV2SequenceClassificationOutput, DebertaV2TokenClassificationOutput,
    DebertaV2VocabResources,
};
