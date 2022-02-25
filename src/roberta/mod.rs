//! # RoBERTa: A Robustly Optimized BERT Pretraining Approach (Liu et al.)
//!
//! Implementation of the RoBERTa language model ([https://arxiv.org/abs/1907.11692](https://arxiv.org/abs/1907.11692) Liu, Ott, Goyal, Du, Joshi, Chen, Levy, Lewis, Zettlemoyer, Stoyanov, 2019).
//! The base model is implemented in the `bert_model::BertModel` struct. Several language model heads have also been implemented, including:
//! - Masked language model: `roberta_model::RobertaForMaskedLM`
//! - Multiple choices: `roberta_model:RobertaForMultipleChoice`
//! - Question answering: `roberta_model::RobertaForQuestionAnswering`
//! - Sequence classification: `roberta_model::RobertaForSequenceClassification`
//! - Token classification (e.g. NER, POS tagging): `roberta_model::RobertaForTokenClassification`
//!
//! # Model set-up and pre-trained weights loading
//!
//! The example below illustrate a Masked language model example, the structure is similar for other models.
//! All models expect the following resources:
//! - Configuration file expected to have a structure following the [Transformers library](https://github.com/huggingface/transformers)
//! - Model weights are expected to have a structure and parameter names following the [Transformers library](https://github.com/huggingface/transformers). A conversion using the Python utility scripts is required to convert the `.bin` weights to the `.ot` format.
//! - `RobertaTokenizer` using a `vocab.txt` vocabulary and `merges.txt` 2-gram merges
//! Pretrained models are available and can be downloaded using RemoteResources.
//!
//! ```no_run
//! # fn main() -> anyhow::Result<()> {
//! #
//! use tch::{nn, Device};
//! # use std::path::PathBuf;
//! use rust_bert::bert::BertConfig;
//! use rust_bert::resources::{LocalResource, ResourceProvider};
//! use rust_bert::roberta::RobertaForMaskedLM;
//! use rust_bert::Config;
//! use rust_tokenizers::tokenizer::RobertaTokenizer;
//!
//! let config_resource = LocalResource {
//!     local_path: PathBuf::from("path/to/config.json"),
//! };
//! let vocab_resource = LocalResource {
//!     local_path: PathBuf::from("path/to/vocab.txt"),
//! };
//! let merges_resource = LocalResource {
//!     local_path: PathBuf::from("path/to/merges.txt"),
//! };
//! let weights_resource = LocalResource {
//!     local_path: PathBuf::from("path/to/model.ot"),
//! };
//! let config_path = config_resource.get_local_path()?;
//! let vocab_path = vocab_resource.get_local_path()?;
//! let merges_path = merges_resource.get_local_path()?;
//! let weights_path = weights_resource.get_local_path()?;
//!
//! let device = Device::cuda_if_available();
//! let mut vs = nn::VarStore::new(device);
//! let tokenizer: RobertaTokenizer = RobertaTokenizer::from_file(
//!     vocab_path.to_str().unwrap(),
//!     merges_path.to_str().unwrap(),
//!     true,
//!     true,
//! )?;
//! let config = BertConfig::from_file(config_path);
//! let bert_model = RobertaForMaskedLM::new(&vs.root(), &config);
//! vs.load(weights_path)?;
//!
//! # Ok(())
//! # }
//! ```

mod embeddings;
mod roberta_model;

pub use embeddings::RobertaEmbeddings;
pub use roberta_model::{
    RobertaConfigResources, RobertaForMaskedLM, RobertaForMultipleChoice,
    RobertaForQuestionAnswering, RobertaForSequenceClassification, RobertaForTokenClassification,
    RobertaMergesResources, RobertaModelResources, RobertaVocabResources,
};
