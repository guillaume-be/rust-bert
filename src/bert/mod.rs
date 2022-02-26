//! # BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (Devlin et al.)
//!
//! Implementation of the BERT language model ([https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805) Devlin, Chang, Lee, Toutanova, 2018).
//! The base model is implemented in the `bert_model::BertModel` struct. Several language model heads have also been implemented, including:
//! - Masked language model: `bert_model::BertForMaskedLM`
//! - Multiple choices: `bert_model:BertForMultipleChoice`
//! - Question answering: `bert_model::BertForQuestionAnswering`
//! - Sequence classification: `bert_model::BertForSequenceClassification`
//! - Token classification (e.g. NER, POS tagging): `bert_model::BertForTokenClassification`
//!
//! # Model set-up and pre-trained weights loading
//!
//! A full working example is provided in `examples/masked_language_model_bert`, run with `cargo run --example masked_language_model_bert`.
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
//! use rust_bert::bert::{BertConfig, BertForMaskedLM};
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
//! let config = BertConfig::from_file(config_path);
//! let bert_model = BertForMaskedLM::new(&vs.root(), &config);
//! vs.load(weights_path)?;
//!
//! # Ok(())
//! # }
//! ```

mod attention;
mod bert_model;
mod embeddings;
pub(crate) mod encoder;

pub use bert_model::{
    BertConfig, BertConfigResources, BertForMaskedLM, BertForMultipleChoice,
    BertForQuestionAnswering, BertForSequenceClassification, BertForTokenClassification,
    BertMaskedLMOutput, BertModel, BertModelOutput, BertModelResources,
    BertQuestionAnsweringOutput, BertSequenceClassificationOutput, BertTokenClassificationOutput,
    BertVocabResources,
};
pub use embeddings::{BertEmbedding, BertEmbeddings};
pub use encoder::{BertEncoder, BertEncoderOutput, BertLayer, BertLayerOutput, BertPooler};
