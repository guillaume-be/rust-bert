//! # FNet, Mixing Tokens with Fourier Transforms (Lee-Thorp et al.)
//!
//! Implementation of the FNet language model ([https://arxiv.org/abs/2105.03824](https://arxiv.org/abs/2105.03824) Lee-Thorp, Ainslie, Eckstein, Ontanon, 2021).
//! The base model is implemented in the `fnet_model::FNetModel` struct. Several language model heads have also been implemented, including:
//! - Masked language model: `fnet_model::FNetForMaskedLM`
//! - Question answering: `fnet_model::FNetForQuestionAnswering`
//! - Sequence classification: `fnet_model::FNetForSequenceClassification`
//! - Token classification (e.g. NER, POS tagging): `fnet_model::FNetForTokenClassification`
//!
//! # Model set-up and pre-trained weights loading
//!
//! The example below illustrate a FNet Masked language model example, the structure is similar for other models.
//! All models expect the following resources:
//! - Configuration file expected to have a structure following the [Transformers library](https://github.com/huggingface/transformers)
//! - Model weights are expected to have a structure and parameter names following the [Transformers library](https://github.com/huggingface/transformers). A conversion using the Python utility scripts is required to convert the `.bin` weights to the `.ot` format.
//! - `FNetTokenizer` using a `spiece.model` SentencePiece (BPE) model file
//! Pretrained models are available and can be downloaded using RemoteResources.
//!
//! ```no_run
//! # fn main() -> anyhow::Result<()> {
//! #
//! use tch::{nn, Device};
//! # use std::path::PathBuf;
//! use rust_bert::fnet::{FNetConfig, FNetForMaskedLM};
//! use rust_bert::resources::{LocalResource, ResourceProvider};
//! use rust_bert::Config;
//! use rust_tokenizers::tokenizer::{BertTokenizer, FNetTokenizer};
//!
//! let config_resource = LocalResource {
//!     local_path: PathBuf::from("path/to/config.json"),
//! };
//! let vocab_resource = LocalResource {
//!     local_path: PathBuf::from("path/to/spiece.model"),
//! };
//! let weights_resource = LocalResource {
//!     local_path: PathBuf::from("path/to/model.ot"),
//! };
//! let config_path = config_resource.get_local_path()?;
//! let vocab_path = vocab_resource.get_local_path()?;
//! let weights_path = weights_resource.get_local_path()?;
//! let device = Device::cuda_if_available();
//! let mut vs = nn::VarStore::new(device);
//! let tokenizer: FNetTokenizer =
//!     FNetTokenizer::from_file(vocab_path.to_str().unwrap(), true, true)?;
//! let config = FNetConfig::from_file(config_path);
//! let bert_model = FNetForMaskedLM::new(&vs.root(), &config);
//! vs.load(weights_path)?;
//!
//! # Ok(())
//! # }
//! ```

mod attention;
mod embeddings;
mod encoder;
mod fnet_model;

pub use fnet_model::{
    FNetConfig, FNetConfigResources, FNetForMaskedLM, FNetForMultipleChoice,
    FNetForQuestionAnswering, FNetForSequenceClassification, FNetForTokenClassification,
    FNetMaskedLMOutput, FNetModel, FNetModelOutput, FNetModelResources,
    FNetQuestionAnsweringOutput, FNetSequenceClassificationOutput, FNetTokenClassificationOutput,
    FNetVocabResources,
};
