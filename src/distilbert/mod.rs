//! # DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter (Sanh et al.)
//!
//! Implementation of the DistilBERT language model ([https://arxiv.org/abs/1910.01108](https://arxiv.org/abs/1910.01108) Sanh, Debut, Chaumond, Wolf, 2019).
//! The base model is implemented in the `distilbert_model::DistilBertModel` struct. Several language model heads have also been implemented, including:
//! - Masked language model: `distilbert_model::DistilBertForMaskedLM`
//! - Question answering: `distilbert_model::DistilBertForQuestionAnswering`
//! - Sequence classification: `distilbert_model::DistilBertForSequenceClassification`
//! - Token classification (e.g. NER, POS tagging): `distilbert_model::DistilBertForTokenClassification`
//!
//! # Model set-up and pre-trained weights loading
//!
//! The example below illustrate a DistilBERT Masked language model example, the structure is similar for other models.
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
//! use rust_bert::distilbert::{
//!     DistilBertConfig, DistilBertConfigResources, DistilBertModelMaskedLM,
//!     DistilBertModelResources, DistilBertVocabResources,
//! };
//! use rust_bert::resources::{LocalResource, ResourceProvider};
//! use rust_bert::Config;
//! use rust_tokenizers::tokenizer::BertTokenizer;
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
//! let tokenizer: BertTokenizer =
//!     BertTokenizer::from_file(vocab_path.to_str().unwrap(), true, true)?;
//! let config = DistilBertConfig::from_file(config_path);
//! let bert_model = DistilBertModelMaskedLM::new(&vs.root(), &config);
//! vs.load(weights_path)?;
//!
//! # Ok(())
//! # }
//! ```

mod attention;
mod distilbert_model;
mod embeddings;
mod transformer;

pub use distilbert_model::{
    DistilBertConfig, DistilBertConfigResources, DistilBertForQuestionAnswering,
    DistilBertForTokenClassification, DistilBertMaskedLMOutput, DistilBertModel,
    DistilBertModelClassifier, DistilBertModelMaskedLM, DistilBertModelResources,
    DistilBertQuestionAnsweringOutput, DistilBertSequenceClassificationOutput,
    DistilBertTokenClassificationOutput, DistilBertVocabResources,
};
