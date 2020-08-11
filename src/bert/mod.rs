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
//! Pretrained models are available and can be downloaded using RemoteResources.
//!
//! ```no_run
//! # fn main() -> anyhow::Result<()> {
//! #
//! use rust_tokenizers::BertTokenizer;
//! use tch::{nn, Device};
//! # use std::path::PathBuf;
//! use rust_bert::bert::{BertConfig, BertForMaskedLM};
//! use rust_bert::resources::{download_resource, LocalResource, Resource};
//! use rust_bert::Config;
//!
//! let config_resource = Resource::Local(LocalResource {
//!     local_path: PathBuf::from("path/to/config.json"),
//! });
//! let vocab_resource = Resource::Local(LocalResource {
//!     local_path: PathBuf::from("path/to/vocab.txt"),
//! });
//! let weights_resource = Resource::Local(LocalResource {
//!     local_path: PathBuf::from("path/to/model.ot"),
//! });
//! let config_path = download_resource(&config_resource)?;
//! let vocab_path = download_resource(&vocab_resource)?;
//! let weights_path = download_resource(&weights_resource)?;
//! let device = Device::cuda_if_available();
//! let mut vs = nn::VarStore::new(device);
//! let tokenizer: BertTokenizer = BertTokenizer::from_file(vocab_path.to_str().unwrap(), true);
//! let config = BertConfig::from_file(config_path);
//! let bert_model = BertForMaskedLM::new(&vs.root(), &config);
//! vs.load(weights_path)?;
//!
//! # Ok(())
//! # }
//! ```

mod attention;
mod bert;
mod embeddings;
pub(crate) mod encoder;

pub use bert::{
    Activation, BertConfig, BertConfigResources, BertForMaskedLM, BertForMultipleChoice,
    BertForQuestionAnswering, BertForSequenceClassification, BertForTokenClassification, BertModel,
    BertModelResources, BertVocabResources,
};
pub use embeddings::{BertEmbedding, BertEmbeddings};
