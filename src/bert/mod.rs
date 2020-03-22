//! # BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
//!
//! Implementation of the BERT language model (https://arxiv.org/abs/1810.04805 Devlin, Chang, Lee, Toutanova, 2018).
//! The base model is implemented in the `bert::BertModel` struct. Several language model heads have also been implemented, including:
//! - Masked language model: `bert::BertForMaskedLM`
//! - Multiple choices: `bert:BertForMultipleChoice`
//! - Question answering: ``bert::BertForQuestionAnswering`
//! -
//!
//!
//!
//! # Quick Start
//!


mod bert;
mod embeddings;
mod attention;
mod encoder;

pub use bert::{BertConfig, BertModel, BertForTokenClassification, BertForMultipleChoice, BertForMaskedLM, BertForSequenceClassification, BertForQuestionAnswering};
pub(crate) use embeddings::BertEmbedding;